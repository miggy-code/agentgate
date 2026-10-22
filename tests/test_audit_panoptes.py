import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from panoptes.db import AuditDB, EventConflictError
from panoptes.schema import (
    ALLOWED_OUTCOME_STATUSES,
    ValidationError,
    normalize_event,
    validate_event,
)

from agentgate.audit.base import AuditEvent, EventKind
from agentgate.audit.panoptes import PanoptesSink, to_panoptes_event
from agentgate.models import Action, Role

T0 = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


def _event(kind: EventKind = EventKind.ALLOW, **overrides: Any) -> AuditEvent:
    data: dict[str, Any] = {
        "timestamp": T0,
        "kind": kind,
        "customer_id": "acme",
        "user_id": "ana",
        "role": Role.STAFF,
        "run_id": "run-1",
        "tool": "docs.read",
        "action": Action.READ,
        "subject": "doc-1",
        "reason": "matched_allow",
        "rule_ids": ("read-internal", "staff-docs"),
        "policy_version": "sha256:0123456789abcdef",
        "latency_us": 42,
        "details": {"label": "internal"},
    }
    return AuditEvent.model_validate(data | overrides)


@pytest.fixture
def sink(tmp_path: Path) -> Any:
    opened = PanoptesSink(tmp_path / "audit.db")
    yield opened
    opened.close()


# Mapping


def test_every_kind_maps_to_an_allowed_status() -> None:
    for kind in EventKind:
        status = to_panoptes_event(_event(kind))["outcome"]["status"]
        assert status in ALLOWED_OUTCOME_STATUSES


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        (EventKind.ALLOW, "success"),
        (EventKind.RETRIEVAL, "success"),
        (EventKind.DENY, "blocked"),
        (EventKind.CAP_HIT, "blocked"),
        (EventKind.INVARIANT_VIOLATION, "blocked"),
        (EventKind.APPROVAL_REQUIRED, "pending"),
    ],
)
def test_status_by_kind(kind: EventKind, status: str) -> None:
    assert to_panoptes_event(_event(kind))["outcome"]["status"] == status


def test_full_mapping() -> None:
    event = _event(EventKind.DENY, reason="label_too_high", timestamp=T0.astimezone())
    assert to_panoptes_event(event) == {
        "event_id": event.event_id,
        "timestamp": "2026-10-22T12:00:00+00:00",
        "event_type": "decision_point",
        "agent": {"type": "agentgate", "session_id": "run-1", "run_id": "run-1"},
        "user": {"id": "ana", "auth_method": "cli"},
        "action": {
            "tool_name": "docs.read",
            "parameters": {"action": "read", "subject": "doc-1"},
            "target": "doc-1",
        },
        "outcome": {"status": "blocked", "summary": "deny: label_too_high"},
        "context": {
            "schema_version": 4,
            "agentgate": {
                "kind": "deny",
                "customer_id": "acme",
                "user_id": "ana",
                "role": "staff",
                "run_id": "run-1",
                "tool": "docs.read",
                "reason": "label_too_high",
                "rule_ids": ["read-internal", "staff-docs"],
                "policy_version": "sha256:0123456789abcdef",
                "latency_us": 42,
                "details": {"label": "internal"},
            },
        },
    }


def test_event_without_user_run_or_tool() -> None:
    event = _event(EventKind.CAP_HIT, user_id=None, role=None, run_id=None, tool=None)
    mapped = to_panoptes_event(event)
    assert "user" not in mapped
    assert mapped["agent"] == {"type": "agentgate", "session_id": "no-run", "run_id": "no-run"}
    assert mapped["action"]["tool_name"] == "cap_hit"
    assert mapped["context"]["agentgate"]["user_id"] is None
    assert mapped["context"]["agentgate"]["run_id"] is None


def test_panoptes_normalization_keeps_ids_intact() -> None:
    # A subject that trips Panoptes' "prod" data classifier gains a tag but keeps its value.
    for kind in EventKind:
        mapped = to_panoptes_event(_event(kind, subject="prod-runbook"))
        validate_event(mapped)
        normalized = normalize_event(mapped)
        validate_event(normalized)
        assert normalized["action"]["target"] == "prod-runbook"
        assert normalized["action"]["parameters"] == mapped["action"]["parameters"]
        assert normalized["action"]["tool_name"] == mapped["action"]["tool_name"]
        assert normalized["context"]["agentgate"] == mapped["context"]["agentgate"]
        assert normalized["user"] == mapped["user"]
        assert normalized["agent"]["run_id"] == "run-1"


def test_id_panoptes_would_redact_is_refused() -> None:
    # Panoptes would store this subject as "[REDACTED API KEY]"; an altered id is no record.
    with pytest.raises(ValidationError, match="subject"):
        to_panoptes_event(_event(subject="sk-proj-abcdefghijklmnopqrstuvwx"))


# Recording


def test_round_trip_of_every_kind(sink: PanoptesSink, tmp_path: Path) -> None:
    events = [
        _event(kind, reason=f"reason_{kind.value}", timestamp=T0 + timedelta(seconds=index))
        for index, kind in enumerate(EventKind)
    ]
    events.append(_event(EventKind.CAP_HIT, user_id=None, role=None, run_id=None, tool=None))
    for event in events:
        sink.record(event)

    db = AuditDB(str(tmp_path / "audit.db"))
    try:
        stored = {row["event_id"]: row for row in db.query(limit=1000)}
    finally:
        db.close()
    assert len(stored) == len(events)
    for event in events:
        row = stored[event.event_id]
        fields = row["context"]["agentgate"]
        assert row["event_type"] == "decision_point"
        assert row["agent"]["type"] == "agentgate"
        assert fields["kind"] == event.kind.value
        assert fields["customer_id"] == event.customer_id
        assert fields["user_id"] == event.user_id
        assert fields["role"] == (event.role and event.role.value)
        assert fields["run_id"] == event.run_id
        assert fields["tool"] == event.tool
        assert fields["reason"] == event.reason
        assert fields["rule_ids"] == list(event.rule_ids)
        assert fields["policy_version"] == event.policy_version
        assert fields["latency_us"] == event.latency_us
        assert fields["details"] == event.details
        assert row["agent"]["run_id"] == (event.run_id or "no-run")
        assert row["action"]["target"] == event.subject
        assert row["outcome"]["summary"] == f"{event.kind.value}: {event.reason}"
        if event.user_id is not None:
            assert row["user"]["id"] == event.user_id


def test_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "audit.db"
    created = PanoptesSink(path)
    try:
        created.record(_event())
        assert created.count() == 1
    finally:
        created.close()
    assert path.is_file()


def test_rejected_event_raises_and_stores_nothing(sink: PanoptesSink) -> None:
    with pytest.raises(ValidationError):
        sink.record(_event(tool="sk-proj-abcdefghijklmnopqrstuvwx"))
    assert sink.count() == 0


def test_storage_failure_propagates(sink: PanoptesSink, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(event: dict[str, Any]) -> str:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sink._db, "insert_event", broken)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        sink.record(_event())


def test_conflicting_event_id_raises(sink: PanoptesSink) -> None:
    first = _event()
    sink.record(first)
    sink.record(first)  # an identical retry is idempotent
    rewritten = _event(EventKind.DENY, event_id=first.event_id)
    with pytest.raises(EventConflictError):
        sink.record(rewritten)
    assert sink.count() == 1


def test_record_after_close_raises(tmp_path: Path) -> None:
    closed = PanoptesSink(tmp_path / "audit.db")
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.record(_event())


# Verification


def test_chain_verifies_after_many_events(sink: PanoptesSink) -> None:
    for index in range(250):
        kind = list(EventKind)[index % len(EventKind)]
        sink.record(_event(kind, timestamp=T0 + timedelta(milliseconds=index)))
    result = sink.verify()
    assert result["valid"] is True
    assert result["checked_events"] == 250
    assert sink.count() == 250


def _tamper(path: Path, sql: str, *params: object) -> None:
    # Edit the stored row directly, the way someone with file access would.
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(sql, params).rowcount == 1
        conn.commit()
    finally:
        conn.close()


def test_editing_a_stored_event_breaks_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    sink = PanoptesSink(path)
    try:
        events = [_event(timestamp=T0 + timedelta(seconds=i)) for i in range(5)]
        events[2] = _event(EventKind.DENY, reason="label_too_high", timestamp=events[2].timestamp)
        for event in events:
            sink.record(event)
        assert sink.verify()["valid"] is True

        # Rewrite the deny as an allow, leaving its hash alone.
        _tamper(
            path,
            "UPDATE events SET outcome = json_set(outcome, '$.status', 'success') "
            "WHERE event_id = ?",
            events[2].event_id,
        )

        result = sink.verify()
        assert result["valid"] is False
        assert result["broken_event_id"] == events[2].event_id
        assert result["error"] == "Canonical event hash mismatch"
    finally:
        sink.close()


def test_deleting_a_stored_event_breaks_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    sink = PanoptesSink(path)
    try:
        events = [_event(timestamp=T0 + timedelta(seconds=i)) for i in range(3)]
        for event in events:
            sink.record(event)
        _tamper(path, "DELETE FROM events WHERE event_id = ?", events[1].event_id)
        assert sink.verify()["valid"] is False
    finally:
        sink.close()


def test_open_existing_refuses_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        PanoptesSink(path, create=False)
    assert not path.exists()


def test_open_existing_refuses_a_non_panoptes_file(tmp_path: Path) -> None:
    path = tmp_path / "other.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE notes (body TEXT)")
    conn.close()
    with pytest.raises(ValueError, match="not a Panoptes audit database"):
        PanoptesSink(path, create=False)

    text = tmp_path / "notes.txt"
    text.write_text("not a database, just some text that is long enough to have a header\n")
    with pytest.raises(ValueError, match="not a Panoptes audit database"):
        PanoptesSink(text, create=False)
