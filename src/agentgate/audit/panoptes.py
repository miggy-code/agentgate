"""Panoptes, the default durable audit sink.

Panoptes stores events in a hash-chained SQLite log: each row's hash covers the row and the
hash before it, so editing, deleting or reordering a stored event breaks verification. Each
AuditEvent becomes one Panoptes "decision_point" event. The agentgate fields travel in
context["agentgate"], which Panoptes keeps as written apart from its secret redaction.

Panoptes redacts secret-shaped strings and derives data_accessed tags from the action before
storing. Tags only add information, but redaction would silently rewrite an id, so every
identifier is checked with Panoptes' own identifier rule first and an id that Panoptes would
rewrite is refused instead of being stored altered. Free-form details are redacted as
Panoptes sees fit. Panoptes keeps timestamps to the millisecond.

What is stored is the event's `recorded()` form: a field the gate referenced (a caller's
value that names nothing real, or a real id Panoptes would refuse) holds its `ref:`
reference, and context["agentgate"]["references"] says which fields those are and why. The
gate checks values with `storable_identifier`, Panoptes' own rule, before it emits them. A
refused event raises AuditRejected, so the gate can tell it from a storage failure.

`read_events` reads a log back, read-only, as the AuditEvents it stores, in the order they
were written: each referenced field holds its `ref:` reference, as stored. That is what
`agentgate audit show` prints and what `agentgate audit check-store` compares the store with.
"""

import json
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from panoptes.db import AuditDB
from panoptes.schema import ValidationError, validate_identifier

from agentgate.audit.base import AuditEvent, AuditRejected, EventKind, Reference

# Panoptes needs a session and run id on every event; this marks events outside any run.
NO_RUN = "no-run"

# Panoptes event schema version these events are written against. Version 4 is the first
# that accepts agent type "agentgate" and outcome status "pending".
SCHEMA_VERSION = 4

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


def to_panoptes_event(event: AuditEvent) -> dict[str, Any]:
    """Map one AuditEvent, in its recorded form, to a Panoptes event. Raises Panoptes'
    ValidationError if an id would not survive storage."""
    references = {name: ref.why for name, ref in sorted(event.references.items())}
    event = event.recorded()
    _check_ids(event)
    run_id = event.run_id or NO_RUN
    mapped: dict[str, Any] = {
        "event_id": event.event_id,
        "timestamp": event.timestamp.astimezone(UTC).isoformat(),
        "event_type": "decision_point",
        "agent": {"type": "agentgate", "session_id": run_id, "run_id": run_id},
        "action": {
            "tool_name": event.tool or event.kind.value,
            "parameters": {"action": _value(event.action), "subject": event.subject},
            "target": event.subject,
        },
        "outcome": {"status": _STATUS[event.kind], "summary": f"{event.kind}: {event.reason}"},
        "context": {
            "schema_version": SCHEMA_VERSION,
            # Panoptes keeps unknown context keys, so this is the free-form metadata slot.
            # user_id, run_id and tool are repeated here because Panoptes fills in
            # "local_user" for a missing user, and the agent and tool_name fields above
            # carry placeholders when the event has no run or tool.
            "agentgate": {
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
            },
        },
    }
    if event.user_id is not None:
        # The host application authenticated the user before the run started, not Panoptes.
        # "cli" is the closest of Panoptes' fixed auth methods (api_key, oauth, cli): a
        # local caller asserting the identity, with no credential presented to Panoptes.
        mapped["user"] = {"id": event.user_id, "auth_method": "cli"}
    return mapped


def _value(member: StrEnum | None) -> str | None:
    return None if member is None else member.value


def _check_ids(event: AuditEvent) -> None:
    fields: list[tuple[str, str | None]] = [
        ("event_id", event.event_id),
        ("customer_id", event.customer_id),
        ("user_id", event.user_id),
        ("run_id", event.run_id),
        ("tool", event.tool),
        ("subject", event.subject),
        ("reason", event.reason),
        ("policy_version", event.policy_version),
        *(("rule_ids", rule_id) for rule_id in event.rule_ids),
    ]
    for name, value in fields:
        if value is not None:
            # Rejects empty, over-long, control-character and secret-shaped values.
            validate_identifier(value, name)


class PanoptesSink:
    """Records each event in a Panoptes hash-chained SQLite log, or raises."""

    def __init__(self, db_path: str | Path, *, create: bool = True) -> None:
        """Open the log, creating it and its parent directories unless create is False.

        With create=False a missing file raises FileNotFoundError and a file without a
        Panoptes events table raises ValueError, so nothing is created or changed by mistake.
        """
        path = Path(db_path)
        if not create:
            _check_existing(path)
        # Panoptes creates missing parent directories itself, owner-only.
        self._db = AuditDB(str(path))
        self._closed = False

    def record(self, event: AuditEvent) -> None:
        if self._closed:
            raise RuntimeError("audit sink is closed")
        # Panoptes raises on a rejected event, a conflicting event_id or a storage error.
        # Only a rejection becomes AuditRejected; it is raised before anything is written.
        try:
            self._db.insert_event(to_panoptes_event(event))
        except ValidationError as exc:
            raise AuditRejected("Panoptes refused the event") from exc

    def verify(self) -> dict[str, Any]:
        """Panoptes' chain verification result; result["valid"] is True when intact."""
        result: dict[str, Any] = self._db.verify_chain()
        return result

    def count(self) -> int:
        """Number of events stored."""
        total: int = self._db.get_stats()["total_events"]
        return total

    def close(self) -> None:
        self._closed = True
        self._db.close()


def read_events(db_path: str | Path) -> list[AuditEvent]:
    """Every agentgate event in the Panoptes log at db_path, oldest first, as stored.

    Opens the file read-only and writes nothing. Raises FileNotFoundError for a missing file,
    ValueError for a file that is not a Panoptes log or a row that is not a well-formed
    agentgate event. Rows Panoptes holds for other agents are skipped. This does not verify
    the chain; PanoptesSink.verify does.
    """
    path = Path(db_path)
    _check_existing(path)
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT event_id, timestamp, action, context FROM events ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    events: list[AuditEvent] = []
    for event_id, timestamp, action, context in rows:
        event = _stored_event(str(event_id), str(timestamp), action, context)
        if event is not None:
            events.append(event)
    return events


def _stored_event(
    event_id: str, timestamp: str, action_json: object, context_json: object
) -> AuditEvent | None:
    try:
        context = json.loads(str(context_json))
        fields = context.get("agentgate") if isinstance(context, dict) else None
        if fields is None:
            return None
        action = json.loads(str(action_json))
        parameters = action.get("parameters") or {}
        values: dict[str, Any] = {
            "event_id": event_id,
            "timestamp": _aware(datetime.fromisoformat(timestamp)),
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
        }
        # Each referenced field already holds its reference; the context says why.
        values["references"] = {
            name: Reference(ref=values[name], why=why)
            for name, why in (fields.get("references") or {}).items()
        }
        return AuditEvent.model_validate(values)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"event {event_id}: not a well-formed agentgate event") from exc


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _check_existing(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"audit database not found: {path}")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'events'"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"not a Panoptes audit database: {path}") from exc
    if row is None:
        raise ValueError(f"not a Panoptes audit database: {path}")
