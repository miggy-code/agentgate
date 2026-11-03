"""A log written under Panoptes schema v4 still reads, verifies and takes new events.

tests/fixtures/audit-v4/audit.db was written by agentgate at commit c718ca1 (before the v5
sink), with Panoptes 0.1.1: four setup events, a run, an allowed and executed read, and two
denials. Its events carry their fields in context["agentgate"] under event type
decision_point.
"""

import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentgate.audit.base import AuditEvent, EventKind
from agentgate.audit.panoptes import PanoptesSink, read_events, verify_log
from agentgate.cli import app

FIXTURE = Path(__file__).parent / "fixtures" / "audit-v4" / "audit.db"
KINDS = [
    EventKind.POLICY_LOADED,
    EventKind.POLICY_LOADED,
    EventKind.OPERATOR_ACTION,
    EventKind.OPERATOR_ACTION,
    EventKind.RUN_STARTED,
    EventKind.ALLOW,
    EventKind.EXECUTED,
    EventKind.DENY,
    EventKind.DENY,
]
runner = CliRunner()


@pytest.fixture
def log(tmp_path: Path) -> Path:
    # A copy: even a read-only open may leave -wal and -shm files beside the database.
    path = tmp_path / "audit.db"
    shutil.copy(FIXTURE, path)
    return path


def test_the_fixture_is_a_v4_log() -> None:
    conn = sqlite3.connect(f"{FIXTURE.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT event_type, agent_type, json_extract(context, '$.schema_version')"
            " FROM events"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("decision_point", "agentgate", 4)]


def test_a_v4_log_reads_as_the_events_it_holds(log: Path) -> None:
    events = read_events(log)
    assert [e.kind for e in events] == KINDS
    run = events[4:]
    assert len({e.run_id for e in run}) == 1 and run[0].run_id is not None
    assert {e.user_id for e in run} == {"kc-staff-nadine"}
    assert [e.reason for e in events[5:]] == [
        "matched_allow",
        "executed",
        "document_not_found",
        "default_deny",
    ]
    # Setup events have no run, and the v4 placeholder run id is not read as one.
    assert all(e.run_id is None and e.user_id is None for e in events[:4])
    assert all(e.exchange_id is None and e.correlation_id is None for e in events)


def test_a_v4_log_verifies_and_reading_it_changes_nothing(log: Path) -> None:
    before = log.read_bytes()
    result, count = verify_log(log)
    assert result["valid"] is True and result["chain"] == "intact" and count == 9
    read_events(log)
    assert log.read_bytes() == before


def test_the_verify_command_accepts_a_v4_log(log: Path) -> None:
    result = runner.invoke(app, ["audit", "verify", "--db", str(log)])
    assert result.exit_code == 0
    assert result.stdout == f"{log}  9 events  chain intact\n"


def test_a_v4_log_takes_v5_events_on_the_same_chain(log: Path) -> None:
    sink = PanoptesSink(log)
    try:
        new = AuditEvent(
            timestamp=datetime(2026, 11, 3, 12, 0, tzinfo=UTC),
            kind=EventKind.ALLOW,
            customer_id="kettle-creek-hvac",
            reason="model_allowed",
            exchange_id="exch-1",
            correlation_id="trial-1",
        )
        sink.record(new)
        assert sink.verify()["valid"] is True
    finally:
        sink.close()
    events = read_events(log)
    assert [e.kind for e in events] == [*KINDS, EventKind.ALLOW]
    assert events[-1].exchange_id == "exch-1" and events[-1].correlation_id == "trial-1"
    result, count = verify_log(log)
    assert result["valid"] is True and count == 10


def test_a_v4_row_that_was_edited_is_still_caught(log: Path) -> None:
    conn = sqlite3.connect(log)
    try:
        conn.execute(
            "UPDATE events SET context = json_set(context, '$.agentgate.reason', 'forged')"
            " WHERE rowid = 8"
        )
        conn.commit()
    finally:
        conn.close()
    result, _ = verify_log(log)
    assert result["valid"] is False
