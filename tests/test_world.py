"""open_world and open_memory_world: seeding once, reopening without duplicates, and refusing
policies whose customer the database does not hold.
"""

import os
import shutil
import stat
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from agentgate.approvals import ApprovalStatus
from agentgate.audit.base import AuditSink, EventKind, MemorySink
from agentgate.audit.panoptes import PanoptesSink
from agentgate.clock import Clock, FixedClock, SystemClock
from agentgate.fixtures import FixtureError, load_fixtures
from agentgate.gate import Denied, Executed, PendingApproval
from agentgate.models import Action, UserStatus
from agentgate.spend import Reservation
from agentgate.store import Store, StoreError, table_names
from agentgate.world import (
    UserAdminError,
    World,
    WorldError,
    open_memory_world,
    open_world,
    reinstate_user,
    suspend_user,
)
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


# Fixture sync on reopen


def _copy_fixtures(tmp_path: Path) -> Path:
    target = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, target)
    return target


def _open_with(
    tmp_path: Path,
    fixtures: Path,
    *,
    policies: Path = POLICIES,
    clock: Clock | None = None,
) -> World:
    return open_world(
        db_path=tmp_path / "data" / "agentgate.db",
        audit_path=tmp_path / "data" / "audit.db",
        policies=policies,
        fixtures=fixtures,
        sink=MemorySink(),
        clock=FixedClock(START) if clock is None else clock,
    )


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))


def test_first_open_reports_no_sync(tmp_path: Path) -> None:
    with _open_with(tmp_path, FIXTURES) as world:
        assert world.fixture_sync is None
        assert world.fixtures == FIXTURES and world.policies == POLICIES
    with _open_with(tmp_path, FIXTURES) as world:
        assert world.fixture_sync is not None and not world.fixture_sync.changed


def test_reopen_applies_suspension_role_change_and_relabel(tmp_path: Path) -> None:
    fixtures = _copy_fixtures(tmp_path)
    with _open_with(tmp_path, fixtures) as world:
        live = world.gate.start_run(KC, "kc-staff-nadine")
        manager = world.gate.start_run(KC, "kc-mgr-marisol")
        assert isinstance(
            world.gate.call_tool(
                live, "docs.read", Action.READ, {"document_id": "kc-service-menu"}
            ),
            Executed,
        )

    users = fixtures / KC / "users.yaml"
    _edit(
        users,
        "nadine@kettlecreek.example\n  role: staff\n  status: active",
        ("nadine@kettlecreek.example\n  role: staff\n  status: suspended"),
    )
    _edit(
        users,
        "marisol@kettlecreek.example\n  role: manager",
        "marisol@kettlecreek.example\n  role: staff",
    )
    menu = fixtures / KC / "docs" / "kc-service-menu.md"
    _edit(menu, "label: public", "label: internal")
    _edit(menu, "# ", "# Zephyrine ")

    with _open_with(tmp_path, fixtures) as world:
        sync = world.fixture_sync
        assert sync is not None
        assert sync.users_updated == (
            ("kc-mgr-marisol", ("role",)),
            ("kc-staff-nadine", ("status",)),
        )
        assert sync.documents_updated == ("kc-service-menu",)
        read = {"document_id": "kc-service-menu"}
        assert world.gate.call_tool(live, "docs.read", Action.READ, read) == Denied(
            reason="suspended_user"
        )
        assert world.gate.call_tool(manager, "docs.read", Action.READ, read) == Denied(
            reason="role_changed"
        )
        doc = world.store.get_document("kc-service-menu")
        assert doc is not None and doc.label == "internal"
        # The index was rebuilt, so the new word is searchable.
        owner = world.gate.start_run(KC, "kc-owner-dale")
        found = world.gate.call_tool(owner, "docs.search", Action.READ, {"query": "zephyrine"})
        assert isinstance(found, Executed)
        documents = found.content["documents"]
        assert isinstance(documents, list)
        assert [d["id"] for d in documents if isinstance(d, dict)] == ["kc-service-menu"]


def test_a_customer_added_to_fixtures_and_policies_opens(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    policies = tmp_path / "policies"
    shutil.copytree(FIXTURES / KC, fixtures / KC)
    shutil.copytree(POLICIES / KC, policies / KC)
    with _open_with(tmp_path, fixtures, policies=policies) as world:
        assert sorted(world.engines) == [KC]
    shutil.copytree(FIXTURES / JR, fixtures / JR)
    shutil.copytree(POLICIES / JR, policies / JR)
    with _open_with(tmp_path, fixtures, policies=policies) as world:
        assert world.fixture_sync is not None
        assert world.fixture_sync.customers_added == (JR,)
        assert sorted(world.engines) == [JR, KC]
        ctx = world.gate.start_run(JR, "jr-owner-marisol")
        found = world.gate.call_tool(ctx, "docs.search", Action.READ, {"query": "shelf"})
        assert isinstance(found, Executed)


def test_runtime_notes_survive_a_sync_that_rebuilds_the_index(tmp_path: Path) -> None:
    fixtures = _copy_fixtures(tmp_path)
    note = {"title": "Quokka note", "label": "internal", "body": "Remember the quokka."}
    with _open_with(tmp_path, fixtures) as world:
        ctx = world.gate.start_run(KC, "kc-owner-dale")
        written = world.gate.call_tool(ctx, "notes.write", Action.WRITE, note)
        assert isinstance(written, Executed)
    _edit(fixtures / KC / "docs" / "kc-service-menu.md", "label: public", "label: internal")
    with _open_with(tmp_path, fixtures) as world:
        assert world.fixture_sync is not None
        assert world.fixture_sync.documents_not_in_fixtures == ()
        ctx = world.gate.start_run(KC, "kc-owner-dale")
        found = world.gate.call_tool(ctx, "docs.search", Action.READ, {"query": "quokka"})
        assert isinstance(found, Executed) and found.content["count"] == 1


def test_reopen_needs_the_fixtures_directory(tmp_path: Path) -> None:
    _open_with(tmp_path, FIXTURES).close()
    with pytest.raises(FixtureError, match="fixtures directory not found"):
        _open_with(tmp_path, tmp_path / "nope")


# Operator suspension


def test_suspend_and_reinstate_through_the_world(tmp_path: Path) -> None:
    fixtures = _copy_fixtures(tmp_path)
    with _open_with(tmp_path, fixtures) as world:
        live = world.gate.start_run(KC, "kc-staff-nadine")
        with pytest.raises(UserAdminError, match="owners only"):
            suspend_user(world, "kc-staff-colby", acting_user_id="kc-mgr-marisol")
        with pytest.raises(UserAdminError, match="unknown user 'kc-staff-colby'"):
            suspend_user(world, "kc-staff-colby", acting_user_id="jr-owner-marisol")
        with pytest.raises(UserAdminError, match="cannot suspend themselves"):
            suspend_user(world, "kc-owner-dale", acting_user_id="kc-owner-dale")
        with pytest.raises(UserAdminError, match="is not active"):
            suspend_user(world, "kc-staff-colby", acting_user_id="kc-staff-gavin")
        suspended = suspend_user(world, "kc-staff-nadine", acting_user_id="kc-owner-dale")
        assert suspended.status is UserStatus.SUSPENDED
        read = {"document_id": "kc-service-menu"}
        assert world.gate.call_tool(live, "docs.read", Action.READ, read) == Denied(
            reason="suspended_user"
        )
    with _open_with(tmp_path, fixtures) as world:
        user = world.store.get_user("kc-staff-nadine")
        assert user is not None and user.status is UserStatus.SUSPENDED
        assert world.fixture_sync is not None
        assert world.fixture_sync.users_kept_suspended == ("kc-staff-nadine",)
        with pytest.raises(UserAdminError, match="was not suspended by an operator"):
            reinstate_user(world, "kc-staff-gavin", acting_user_id="kc-owner-dale")
        back = reinstate_user(world, "kc-staff-nadine", acting_user_id="kc-owner-dale")
        assert back.status is UserStatus.ACTIVE
        # A user the fixtures suspend stays suspended once the operator flag is gone.
        _edit(
            fixtures / KC / "users.yaml",
            "colby@kettlecreek.example\n  role: staff\n  status: active",
            "colby@kettlecreek.example\n  role: staff\n  status: suspended",
        )
        suspend_user(world, "kc-staff-colby", acting_user_id="kc-owner-dale")
        colby = reinstate_user(world, "kc-staff-colby", acting_user_id="kc-owner-dale")
        assert colby.status is UserStatus.SUSPENDED


_SUSPEND_IN_ANOTHER_PROCESS = """
import sys
from pathlib import Path
from agentgate.audit.base import MemorySink
from agentgate.world import open_world, suspend_user
tmp, policies, fixtures = (Path(arg) for arg in sys.argv[1:])
with open_world(
    db_path=tmp / "data" / "agentgate.db",
    audit_path=tmp / "data" / "audit.db",
    policies=policies,
    fixtures=fixtures,
    sink=MemorySink(),
) as world:
    suspend_user(world, "kc-staff-nadine", acting_user_id="kc-owner-dale")
"""


def test_a_suspension_in_another_process_stops_a_live_run(tmp_path: Path) -> None:
    with _open_with(tmp_path, FIXTURES, clock=SystemClock()) as world:
        live = world.gate.start_run(KC, "kc-staff-nadine")
        read = {"document_id": "kc-service-menu"}
        assert isinstance(world.gate.call_tool(live, "docs.read", Action.READ, read), Executed)
        subprocess.run(
            [
                sys.executable,
                "-c",
                _SUSPEND_IN_ANOTHER_PROCESS,
                str(tmp_path),
                str(POLICIES),
                str(FIXTURES),
            ],
            check=True,
            timeout=60,
        )
        assert world.gate.call_tool(live, "docs.read", Action.READ, read) == Denied(
            reason="suspended_user"
        )
        assert world.store.operator_suspended("kc-staff-nadine")


# Paths and permissions


def test_store_and_audit_log_must_be_different_files(tmp_path: Path) -> None:
    same = tmp_path / "data" / "same.db"
    with pytest.raises(WorldError, match="must be different files"):
        open_world(db_path=same, audit_path=same, policies=POLICIES, fixtures=FIXTURES)
    assert not same.exists() and not same.parent.exists()
    # The same file by another spelling.
    _open(tmp_path).close()
    link = tmp_path / "link.db"
    link.symlink_to(tmp_path / "data" / "agentgate.db")
    with pytest.raises(WorldError, match="must be different files"):
        open_world(
            db_path=tmp_path / "data" / "agentgate.db",
            audit_path=link,
            policies=POLICIES,
            fixtures=FIXTURES,
        )


def test_a_store_path_holding_the_audit_log_is_refused(tmp_path: Path) -> None:
    _open(tmp_path).close()
    audit = tmp_path / "data" / "audit.db"
    before = audit.read_bytes()
    with pytest.raises(StoreError, match="is a Panoptes audit database"):
        open_world(
            db_path=audit, audit_path=tmp_path / "other.db", policies=POLICIES, fixtures=FIXTURES
        )
    assert audit.read_bytes() == before
    assert not (tmp_path / "other.db").exists()


def test_an_audit_path_holding_the_store_is_refused(tmp_path: Path) -> None:
    _open(tmp_path).close()
    store = tmp_path / "data" / "agentgate.db"
    tables = table_names(store)
    with pytest.raises(WorldError, match="is an agentgate store, not a Panoptes audit database"):
        open_world(
            db_path=tmp_path / "new.db", audit_path=store, policies=POLICIES, fixtures=FIXTURES
        )
    assert table_names(store) == tables
    assert not (tmp_path / "new.db").exists()


def test_open_world_makes_owner_only_directories_and_files(tmp_path: Path) -> None:
    previous = os.umask(0o022)
    try:
        shared = tmp_path / "shared"
        shared.mkdir(mode=0o755)
        open_world(
            db_path=shared / "data" / "agentgate.db",
            audit_path=shared / "data" / "audit.db",
            policies=POLICIES,
            fixtures=FIXTURES,
        ).close()
    finally:
        os.umask(previous)
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755
    assert stat.S_IMODE((shared / "data").stat().st_mode) == 0o700
    assert stat.S_IMODE((shared / "data" / "agentgate.db").stat().st_mode) == 0o600
    assert stat.S_IMODE((shared / "data" / "audit.db").stat().st_mode) == 0o600


# Stale spend reservations


def _leak(world: World, count: int) -> None:
    for _ in range(count):
        ctx = world.gate.start_run(KC, "kc-staff-nadine")
        assert isinstance(world.gate.charge_model(ctx, Decimal("0.50")), Reservation)


def _day_spent(world: World) -> Decimal:
    customer = world.store.get_customer(KC)
    assert customer is not None
    return world.gate.ledger.customer_day_spent(KC, world.gate.ledger.customer_day(customer))


def test_open_reclaims_reservations_past_the_lease(tmp_path: Path) -> None:
    clock = FixedClock(START)
    with _open_with(tmp_path, FIXTURES, clock=clock) as world:
        _leak(world, 40)
        assert _day_spent(world) == Decimal("20.00")
    clock.advance(timedelta(minutes=10))
    with _open_with(tmp_path, FIXTURES, clock=clock) as world:
        # Inside the lease: they may still be calls in flight.
        assert world.stale_released == 0
        assert _day_spent(world) == Decimal("20.00")
    clock.advance(timedelta(minutes=6))
    with _open_with(tmp_path, FIXTURES, clock=clock) as world:
        assert world.stale_released == 40
        assert _day_spent(world) == 0
        ctx = world.gate.start_run(KC, "kc-owner-dale")
        read = {"document_id": "kc-service-menu"}
        assert isinstance(world.gate.call_tool(ctx, "docs.read", Action.READ, read), Executed)


def test_the_lease_can_be_turned_off_or_shortened(tmp_path: Path) -> None:
    clock = FixedClock(START)
    with _open_with(tmp_path, FIXTURES, clock=clock) as world:
        _leak(world, 2)
    clock.advance(timedelta(minutes=30))
    db = tmp_path / "data" / "agentgate.db"
    kwargs = {"db_path": db, "audit_path": tmp_path / "a.db", "policies": POLICIES}
    with open_world(
        **kwargs, fixtures=FIXTURES, clock=clock, sink=MemorySink(), reservation_lease=None
    ) as world:
        assert world.stale_released == 0
        assert _day_spent(world) == Decimal("1.00")
    with open_world(
        **kwargs,
        fixtures=FIXTURES,
        clock=clock,
        sink=MemorySink(),
        reservation_lease=timedelta(minutes=45),
    ) as world:
        assert world.stale_released == 0
    with open_world(
        **kwargs,
        fixtures=FIXTURES,
        clock=clock,
        sink=MemorySink(),
        reservation_lease=timedelta(minutes=20),
    ) as world:
        assert world.stale_released == 2
