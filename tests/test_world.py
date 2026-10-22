"""open_world and open_memory_world: seeding once, reopening without duplicates, and refusing
policies whose customer the database does not hold.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentgate.approvals import ApprovalStatus
from agentgate.audit.base import AuditSink, EventKind, MemorySink
from agentgate.audit.panoptes import PanoptesSink
from agentgate.clock import Clock, FixedClock, SystemClock
from agentgate.fixtures import FixtureError, load_fixtures
from agentgate.gate import Executed, PendingApproval
from agentgate.models import Action
from agentgate.store import Store
from agentgate.world import World, WorldError, open_memory_world, open_world
from conftest import FIXTURES, JR, KC, POLICIES, START

DIRECTORY_CUSTOMER = "beatrice.l@mail-sample.example"
EMAIL = {"to": [DIRECTORY_CUSTOMER], "subject": "Your visit", "body": "See you Monday."}


def _open(tmp_path: Path, clock: Clock | None = None, sink: AuditSink | None = None) -> World:
    return open_world(
        db_path=tmp_path / "data" / "agentgate.db",
        audit_path=tmp_path / "data" / "audit.db",
        policies=POLICIES,
        fixtures=FIXTURES,
        clock=clock,
        sink=sink,
    )


def _count(store: Store, table: str) -> int:
    with store.locked():
        row = store.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"])


def _fixture_counts() -> tuple[int, int]:
    with Store() as store:
        summary = load_fixtures(store, FIXTURES)
    return summary.total_users, summary.total_documents


def test_first_open_loads_fixtures_and_indexes(tmp_path: Path) -> None:
    users, documents = _fixture_counts()
    with _open(tmp_path) as world:
        assert [c.id for c in world.store.list_customers()] == [JR, KC]
        assert _count(world.store, "users") == users
        assert _count(world.store, "documents") == documents
        assert _count(world.store, "retrieval_lengths") == documents
        assert sorted(world.engines) == [JR, KC]
        assert isinstance(world.sink, PanoptesSink)
        assert isinstance(world.clock, SystemClock)
    assert (tmp_path / "data" / "agentgate.db").is_file()
    assert (tmp_path / "data" / "audit.db").is_file()


def test_reopen_keeps_rows_and_approvals_without_duplicates(tmp_path: Path) -> None:
    users, documents = _fixture_counts()
    with _open(tmp_path, clock=FixedClock(START)) as world:
        ctx = world.gate.start_run(KC, "kc-staff-nadine")
        outcome = world.gate.call_tool(ctx, "email.send", Action.SEND, EMAIL)
        assert isinstance(outcome, PendingApproval)

    with _open(tmp_path, clock=FixedClock(START)) as world:
        assert _count(world.store, "users") == users
        assert _count(world.store, "documents") == documents
        assert _count(world.store, "customers") == 2
        assert _count(world.store, "retrieval_lengths") == documents
        approval = world.gate.approvals.get(outcome.approval_id)
        assert approval is not None and approval.status is ApprovalStatus.PENDING
        # The run from the first session is still known, so it can run the approved call.
        world.gate.approve(outcome.approval_id, "kc-mgr-marisol")
        executed = world.gate.execute_approved(
            ctx, outcome.approval_id, "email.send", Action.SEND, EMAIL
        )
        assert isinstance(executed, Executed)
        assert isinstance(world.sink, PanoptesSink)
        assert world.sink.verify()["valid"]


def test_reopen_rebuilds_an_empty_index(tmp_path: Path) -> None:
    _open(tmp_path).close()
    with Store(tmp_path / "data" / "agentgate.db") as store, store.transaction():
        store.connection.execute("DELETE FROM retrieval_lengths")
        store.connection.execute("DELETE FROM retrieval_postings")
    with _open(tmp_path) as world:
        assert _count(world.store, "retrieval_lengths") == _count(world.store, "documents")


def test_given_sink_is_used_and_left_open(tmp_path: Path) -> None:
    sink = MemorySink()
    with _open(tmp_path, sink=sink) as world:
        world.gate.start_run(KC, "kc-staff-nadine")
        assert world.sink is sink
        assert not world.owns_sink
    assert [e.kind for e in sink.events] == [EventKind.RUN_STARTED]
    assert not (tmp_path / "data" / "audit.db").exists()


def test_open_memory_world() -> None:
    clock = FixedClock(datetime(2026, 10, 22, 15, 0, tzinfo=UTC))
    with open_memory_world(policies=POLICIES, fixtures=FIXTURES, clock=clock) as world:
        assert isinstance(world.sink, MemorySink)
        assert world.clock is clock
        ctx = world.gate.start_run(KC, "kc-staff-nadine")
        outcome = world.gate.call_tool(
            ctx, "docs.search", Action.READ, {"query": "furnace igniter", "limit": 5}
        )
        assert isinstance(outcome, Executed)
        assert outcome.content["count"]
        assert world.sink.events[0].kind is EventKind.RUN_STARTED


def test_policy_customer_missing_from_database(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(FIXTURES / KC, fixtures / KC)
    with pytest.raises(WorldError, match=f"policy customer '{JR}' is not in the database"):
        open_world(
            db_path=tmp_path / "data" / "agentgate.db",
            audit_path=tmp_path / "data" / "audit.db",
            policies=POLICIES,
            fixtures=fixtures,
        )
    # Nothing is opened for audit when the world cannot be built.
    assert not (tmp_path / "data" / "audit.db").exists()


def test_missing_fixtures_directory(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="fixtures directory not found"):
        open_memory_world(policies=POLICIES, fixtures=tmp_path / "nope")
