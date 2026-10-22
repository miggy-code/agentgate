from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agentgate.audit import AuditEvent, EventKind, MemorySink
from agentgate.clock import FixedClock, SystemClock

T0 = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)


def test_fixed_clock_moves_only_when_advanced() -> None:
    clock = FixedClock(T0)
    assert clock.now() == T0
    clock.advance(timedelta(minutes=61))
    assert clock.now() == T0 + timedelta(minutes=61)


def test_fixed_clock_needs_aware_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FixedClock(datetime(2026, 10, 22))


def test_system_clock_is_utc() -> None:
    assert SystemClock().now().utcoffset() == timedelta(0)


def test_audit_event_defaults_and_memory_sink() -> None:
    sink = MemorySink()
    event = AuditEvent(timestamp=T0, kind=EventKind.DENY, customer_id="acme", reason="x")
    sink.record(event)
    assert sink.events == [event]
    assert len(event.event_id) == 36
    assert event.rule_ids == ()


def test_audit_event_rejects_naive_time_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(timestamp=datetime(2026, 1, 1), kind=EventKind.DENY, customer_id="a", reason="x")
    with pytest.raises(ValidationError):
        AuditEvent.model_validate(
            {"timestamp": T0, "kind": "deny", "customer_id": "a", "reason": "x", "body": "..."}
        )
