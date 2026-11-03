"""Panoptes, the default durable audit sink.

Panoptes stores events in a hash-chained SQLite log: each row's hash covers the row and the
hash before it, so editing, deleting or reordering a stored event breaks verification. Each
AuditEvent becomes one Panoptes schema-v5 event of type `agentgate.<kind>`, under agent type
`agentgate`. The agentgate fields travel in `attributes`, Panoptes' open object for a
producer's own fields; the event's run, user and correlation id are Panoptes' own fields.

An event with no run has no run id: its session is `customer:<customer id>`. An event with
no user has no user, and a user carries no auth method, since agentgate does not authenticate
users (the host does).

The sink writes with strict ingest. Panoptes redacts secret-shaped strings, and redaction
would silently rewrite an id, so an identifier field that redaction would change makes
Panoptes refuse the event (AuditRejected) instead of storing it altered. Free-form details
are still redacted as Panoptes sees fit. Panoptes keeps timestamps to the millisecond.

What is stored is the event's `recorded()` form: a field the gate referenced (a caller's
value that names nothing real, or a real id Panoptes would refuse) holds its `ref:`
reference, and attributes.references says which fields those are and why. The gate checks
values with `storable_identifier`, Panoptes' own rule, before it emits them.

`read_events` reads a log back through Panoptes' read API, read-only, as the AuditEvents it
stores, in the order they were written: each referenced field holds its `ref:` reference,
as stored. That is what `agentgate audit show` prints and what `agentgate audit check-store`
compares the store with. It reads logs written under schema v4 as well, which carry the same
fields in `context["agentgate"]` under event type `decision_point`. Only rows stored under
agent type `agentgate` are read.

An agentgate log is chained from its first event, so verification also fails for events
outside the chain. `verify_log` opens the file read-only, so it neither migrates it nor
starts a chain, and verification does not change what it checks.
"""

from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from panoptes.db import AuditDB, NotAnAuditDatabaseError
from panoptes.schema import ValidationError, validate_identifier

from agentgate.audit.base import AuditEvent, AuditRejected, EventKind, Reference

# The Panoptes agent type every agentgate event is stored under, and the namespace of its
# event types.
AGENT_TYPE = "agentgate"
EVENT_NAMESPACE = "agentgate"

# Panoptes event schema version these events are written against.
SCHEMA_VERSION = 5

# Panoptes' event type for events written under schema v4, which agentgate still reads.
LEGACY_EVENT_TYPE = "decision_point"

_STATUS: dict[EventKind, str] = {
    EventKind.RUN_STARTED: "success",
    EventKind.ALLOW: "success",
    EventKind.APPROVED: "success",
    EventKind.EXECUTED: "success",
    EventKind.RETRIEVAL: "success",
    EventKind.DENY: "blocked",
    EventKind.REJECTED: "blocked",
    EventKind.EXPIRED: "blocked",
    EventKind.CAP_HIT: "blocked",
    EventKind.INVARIANT_VIOLATION: "blocked",
    EventKind.APPROVAL_REQUIRED: "pending",
    EventKind.POLICY_LOADED: "success",
    EventKind.OPERATOR_ACTION: "success",
}

# The attribute paths holding identifiers, which strict ingest refuses to rewrite.
_STRICT_PATHS = (
    "attributes.customer_id",
    "attributes.user_id",
    "attributes.run_id",
    "attributes.tool",
    "attributes.reason",
    "attributes.policy_version",
    "attributes.exchange_id",
    "attributes.rule_ids[]",
)


@lru_cache(maxsize=4096)
def storable_identifier(value: str) -> bool:
    """Whether Panoptes stores this value as an identifier exactly as written.

    False for empty, over-long, control-character and secret-shaped values, which Panoptes
    refuses as identifiers (and would rewrite as "[REDACTED ...]" anywhere else).
    """
    try:
        validate_identifier(value, "value")
    except ValidationError:
        return False
    return True


def event_type(kind: EventKind) -> str:
    return f"{EVENT_NAMESPACE}.{kind.value}"


def to_panoptes_event(event: AuditEvent) -> dict[str, Any]:
    """Map one AuditEvent, in its recorded form, to a Panoptes v5 event."""
    references = {name: ref.why for name, ref in sorted(event.references.items())}
    event = event.recorded()
    agent: dict[str, Any] = {
        "type": AGENT_TYPE,
        "session_id": event.run_id or f"customer:{event.customer_id}",
    }
    if event.run_id is not None:
        agent["run_id"] = event.run_id
    attributes: dict[str, Any] = {
        "kind": event.kind.value,
        "customer_id": event.customer_id,
        "user_id": event.user_id,
        "role": _value(event.role),
        "run_id": event.run_id,
        "tool": event.tool,
        "reason": event.reason,
        "rule_ids": list(event.rule_ids),
        "policy_version": event.policy_version,
        "latency_us": event.latency_us,
        "details": event.details,
        # Fields holding a reference instead of a value, and why.
        "references": references,
    }
    if event.exchange_id is not None:
        attributes["exchange_id"] = event.exchange_id
    mapped: dict[str, Any] = {
        "event_id": event.event_id,
        "timestamp": event.timestamp.astimezone(UTC).isoformat(),
        "event_type": event_type(event.kind),
        "agent": agent,
        "action": {
            "tool_name": event.tool or event.kind.value,
            "parameters": {"action": _value(event.action), "subject": event.subject},
            "target": event.subject,
        },
        "outcome": {"status": _STATUS[event.kind], "summary": f"{event.kind}: {event.reason}"},
        "context": {"schema_version": SCHEMA_VERSION},
        "attributes": attributes,
    }
    if event.correlation_id is not None:
        mapped["correlation_id"] = event.correlation_id
    if event.user_id is not None:
        # The host application authenticated the user before the run started. agentgate
        # records who, and no auth method: it never saw a credential.
        mapped["user"] = {"id": event.user_id}
    return mapped


def _value(member: StrEnum | None) -> str | None:
    return None if member is None else member.value


class PanoptesSink:
    """Records each event in a Panoptes hash-chained SQLite log, or raises."""

    def __init__(
        self, db_path: str | Path | None = None, *, create: bool = True, db: AuditDB | None = None
    ) -> None:
        """Open the log at db_path, creating it and its parent directories unless create is
        False, or write through `db`, an AuditDB the caller owns (the model proxy's, so the
        sink and the proxy share one object and one chain). A sink given a `db` never closes
        it.

        With create=False a missing file raises FileNotFoundError and a file without a
        Panoptes events table raises ValueError, so nothing is created or changed by mistake.
        """
        if (db_path is None) == (db is None):
            raise ValueError("give a path or an AuditDB, not both and not neither")
        self._owns_db = db is None
        if db is not None:
            self._db = db
        else:
            assert db_path is not None
            path = Path(db_path)
            if not create:
                _open_readonly(path).close()
            # Panoptes creates missing parent directories itself, owner-only. An agentgate
            # log is always chained.
            self._db = AuditDB(str(path), chain=True)
        self._closed = False

    @property
    def db(self) -> AuditDB:
        return self._db

    def record(self, event: AuditEvent) -> None:
        if self._closed:
            raise RuntimeError("audit sink is closed")
        # Panoptes raises on a rejected event, a conflicting event_id or a storage error.
        # Only a rejection becomes AuditRejected; it is raised before anything is written.
        try:
            self._db.insert_event(to_panoptes_event(event), strict=True, strict_paths=_STRICT_PATHS)
        except ValidationError as exc:
            raise AuditRejected("Panoptes refused the event") from exc

    def verify(self) -> dict[str, Any]:
        """Panoptes' chain verification result; result["valid"] is True when intact.

        Stricter than Panoptes alone: every stored event must be on the chain, since
        agentgate starts its logs empty. Events outside it mean the chain was stripped.
        """
        return _verified(self._db)

    def count(self) -> int:
        """Number of events stored."""
        total: int = self._db.get_stats()["total_events"]
        return total

    def close(self) -> None:
        self._closed = True
        if self._owns_db:
            self._db.close()


def _verified(db: AuditDB) -> dict[str, Any]:
    result: dict[str, Any] = dict(db.verify_chain())
    if not result["valid"]:
        return result
    if result.get("chain") != "intact" or not result.get("initialized"):
        return _broken(result, "events exist but the hash chain is not initialized")
    outside = result.get("legacy_event_count") or 0
    if outside:
        return _broken(result, f"{outside} events are outside the hash chain")
    return result


def _broken(result: dict[str, Any], error: str) -> dict[str, Any]:
    return {**result, "valid": False, "error": error, "broken_event_id": None}


def verify_log(db_path: str | Path) -> tuple[dict[str, Any], int]:
    """Verify the log at db_path without writing to it, and count its events.

    Raises FileNotFoundError for a missing file and ValueError for a file that is not a
    Panoptes log. A log whose chain Panoptes finds broken is reported as broken. The file is
    opened read-only: nothing is migrated, started or re-anchored.
    """
    db = _open_readonly(Path(db_path))
    try:
        return _verified(db), int(db.get_stats()["total_events"])
    finally:
        db.close()


def read_events(db_path: str | Path) -> list[AuditEvent]:
    """Every agentgate event in the Panoptes log at db_path, oldest first, as stored.

    Opens the file read-only and writes nothing. Raises FileNotFoundError for a missing file,
    ValueError for a file that is not a Panoptes log or a row that is not a well-formed
    agentgate event. Rows Panoptes holds for other agents are skipped. This does not verify
    the chain; verify_log does.
    """
    db = _open_readonly(Path(db_path))
    try:
        # Filtered on Panoptes' agent type: any writer can put agentgate-looking fields in
        # its events, but only agentgate's events carry its agent type.
        return [_stored_event(row) for row in db.iter_events(agent_type=AGENT_TYPE, order="rowid")]
    finally:
        db.close()


def _open_readonly(path: Path) -> AuditDB:
    if not path.is_file():
        raise FileNotFoundError(f"audit database not found: {path}")
    try:
        return AuditDB.open_readonly(str(path))
    except NotAnAuditDatabaseError as exc:
        raise ValueError(f"not a Panoptes audit database: {path}") from exc


def _stored_event(row: dict[str, Any]) -> AuditEvent:
    event_id = str(row.get("event_id"))
    try:
        attributes = row.get("attributes") or {}
        # Under schema v4 the same fields were in the context.
        fields = attributes if "kind" in attributes else row["context"]["agentgate"]
        action = row.get("action") or {}
        parameters = action.get("parameters") or {}
        values: dict[str, Any] = {
            "event_id": event_id,
            "timestamp": _aware(datetime.fromisoformat(str(row["timestamp"]))),
            "kind": fields["kind"],
            "customer_id": fields["customer_id"],
            "user_id": fields.get("user_id"),
            "role": fields.get("role"),
            "run_id": fields.get("run_id"),
            "tool": fields.get("tool"),
            "action": parameters.get("action"),
            "subject": action.get("target"),
            "reason": fields["reason"],
            "rule_ids": tuple(fields.get("rule_ids") or ()),
            "policy_version": fields.get("policy_version"),
            "latency_us": fields.get("latency_us"),
            "details": fields.get("details") or {},
            "exchange_id": fields.get("exchange_id"),
            "correlation_id": row.get("correlation_id"),
        }
        # Each referenced field already holds its reference; the event says why.
        values["references"] = {
            name: Reference(ref=values[name], why=why)
            for name, why in (fields.get("references") or {}).items()
        }
        return AuditEvent.model_validate(values)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"event {event_id}: not a well-formed agentgate event") from exc


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
