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
"""

import sqlite3
from datetime import UTC
from enum import StrEnum
from pathlib import Path
from typing import Any

from panoptes.db import AuditDB
from panoptes.schema import validate_identifier

from agentgate.audit.base import AuditEvent, EventKind

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
}


def to_panoptes_event(event: AuditEvent) -> dict[str, Any]:
    """Map one AuditEvent to a Panoptes event. Raises if an id would not survive storage."""
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
        self._db.insert_event(to_panoptes_event(event))

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
