"""Open everything a gate needs in one call: store, fixtures, index, policies, audit, gate.

The CLI, the demo and tests all build their gate here, so they agree on how a world is put
together. open_world keeps its state on disk and can be reopened: fixtures are loaded into a
database that has no customers yet, and synced into one that has (fixtures.sync_fixtures:
new customers, users and documents are added, changed ones updated, nothing deleted, a user
removed from the fixtures suspended), so a suspension, role change or relabelling in the
fixtures takes effect on the next open. Approvals, runs and spend from earlier sessions are
still there. The search index is rebuilt whenever the sync added or changed a document.
`World.fixture_sync` says what the sync did.

On every open, open_world also reclaims spend reservations left open longer than the lease
(15 minutes by default; `reservation_lease=None` turns it off), so a host that died between
reserve and settle does not hold the customer's daily budget (spend.SpendLedger.
release_stale). `World.stale_released` is the count.

open_world refuses a store and an audit log that are the same file, and an audit path that
holds an agentgate store, before it writes anything; Store itself refuses a store path that
holds a Panoptes audit log. Directories it creates for the store are owner-only (0700), and
the store file is 0600 (see Store).

Every policy must name a customer the database holds; a policy for a missing customer is
refused rather than silently ignored. A customer without a policy gets no engine, and the
gate denies its runs with `no_policy`.
"""

import contextlib
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Self

from agentgate.audit.base import AuditSink, MemorySink
from agentgate.audit.panoptes import PanoptesSink
from agentgate.clock import Clock, SystemClock
from agentgate.fixtures import FixtureError, FixtureSync, load_fixtures, sync_fixtures
from agentgate.gate import Gate
from agentgate.models import Role, User, UserStatus
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.testing import load_policy_bundles
from agentgate.retrieval import DocumentIndex
from agentgate.store import Store, make_private_dirs, table_names

DEFAULT_DB = Path("data/agentgate.db")
DEFAULT_AUDIT_DB = Path("data/audit.db")
DEFAULT_POLICIES = Path("policies")
DEFAULT_FIXTURES = Path("fixtures")
# How long a spend reservation may stay open before open_world reclaims it.
DEFAULT_RESERVATION_LEASE = timedelta(minutes=15)

_log = logging.getLogger(__name__)


class WorldError(ValueError):
    """The database, policies and fixtures do not fit together."""


@dataclass
class World:
    store: Store
    gate: Gate
    index: DocumentIndex
    sink: AuditSink
    engines: dict[str, PolicyEngine]
    clock: Clock
    # Whether close() also closes the sink: true only for a sink open_world opened itself.
    owns_sink: bool = False
    # What syncing the fixtures into an existing database changed; None when the fixtures
    # were loaded into an empty one.
    fixture_sync: FixtureSync | None = None
    # Stale spend reservations reclaimed when the world was opened.
    stale_released: int = 0
    # The policy and fixtures directories the world was built from.
    policies: Path | None = None
    fixtures: Path | None = None

    def close(self) -> None:
        """Close the store, and the audit sink if this world opened it."""
        try:
            if self.owns_sink and isinstance(self.sink, PanoptesSink):
                self.sink.close()
        finally:
            self.store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_world(
    *,
    db_path: Path,
    audit_path: Path,
    policies: Path,
    fixtures: Path,
    clock: Clock | None = None,
    sink: AuditSink | None = None,
    reservation_lease: timedelta | None = DEFAULT_RESERVATION_LEASE,
) -> World:
    """Open (or create) the database at db_path and build a gate over it.

    The audit log is a Panoptes database at audit_path unless a sink is given, in which case
    audit_path is not touched and the caller keeps ownership of the sink.
    """
    if sink is None:
        check_paths(db_path, audit_path)
    make_private_dirs(db_path.parent)
    store = Store(db_path)
    try:
        return _build(store, policies, fixtures, clock, sink, audit_path, reservation_lease)
    except BaseException:
        store.close()
        raise


def check_paths(db_path: Path, audit_path: Path) -> None:
    """Refuse a store and audit log that are one file, or an audit path holding a store."""
    if _same_file(db_path, audit_path):
        raise WorldError(f"{db_path}: the store and the audit log must be different files")
    tables = table_names(audit_path)
    if tables is not None and "customers" in tables and "events" not in tables:
        raise WorldError(f"{audit_path}: is an agentgate store, not a Panoptes audit database")


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        # One of them does not exist yet: compare where they would be.
        return a.resolve() == b.resolve()


def open_memory_world(
    *,
    policies: Path,
    fixtures: Path,
    clock: Clock | None = None,
    sink: AuditSink | None = None,
) -> World:
    """A world in an in-memory database, with a MemorySink unless a sink is given."""
    store = Store()
    try:
        return _build(
            store, policies, fixtures, clock, MemorySink() if sink is None else sink, None, None
        )
    except BaseException:
        store.close()
        raise


def _build(
    store: Store,
    policies: Path,
    fixtures: Path,
    clock: Clock | None,
    sink: AuditSink | None,
    audit_path: Path | None,
    reservation_lease: timedelta | None,
) -> World:
    # Needed on every open, not only the first: the sync is how fixture changes, a
    # suspension among them, reach an existing database.
    if not fixtures.is_dir():
        raise FixtureError(f"{fixtures}: fixtures directory not found")
    synced: FixtureSync | None = None
    if not store.list_customers():
        load_fixtures(store, fixtures)
    else:
        synced = sync_fixtures(store, fixtures)
        for line in synced.lines():
            _log.info("fixture sync: %s", line)
    index = DocumentIndex(store)
    # Also rebuilt when the index is empty, for a database populated some other way or one
    # whose earlier session stopped between loading and indexing.
    if synced is None or synced.documents_changed or _index_is_empty(store):
        index.rebuild()

    bundles = load_policy_bundles(policies)
    engines: dict[str, PolicyEngine] = {}
    for customer_id in sorted(bundles):
        customer = store.get_customer(customer_id)
        if customer is None:
            raise WorldError(
                f"{policies / customer_id}: policy customer {customer_id!r} is not in the database"
            )
        engines[customer_id] = PolicyEngine(bundles[customer_id], customer)

    # Opened last, so a world that fails to build creates no audit log.
    owns_sink = sink is None
    if sink is None:
        assert audit_path is not None
        sink = PanoptesSink(audit_path)
    try:
        world_clock = SystemClock() if clock is None else clock
        gate = Gate(store=store, engines=engines, sink=sink, clock=world_clock, index=index)
        released = 0 if reservation_lease is None else gate.ledger.release_stale(reservation_lease)
    except BaseException:
        if owns_sink and isinstance(sink, PanoptesSink):
            with contextlib.suppress(Exception):
                sink.close()
        raise
    return World(
        store=store,
        gate=gate,
        index=index,
        sink=sink,
        engines=engines,
        clock=world_clock,
        owns_sink=owns_sink,
        fixture_sync=synced,
        stale_released=released,
        policies=policies,
        fixtures=fixtures,
    )


def _index_is_empty(store: Store) -> bool:
    with store.locked():
        row = store.connection.execute("SELECT 1 FROM retrieval_lengths LIMIT 1").fetchone()
    return row is None


class UserAdminError(ValueError):
    """An operator change to a user was refused."""


def _owner_of(world: World, acting_user_id: str) -> User:
    acting = world.store.get_user(acting_user_id)
    if acting is None:
        raise UserAdminError(f"unknown user {acting_user_id!r}")
    if acting.status is not UserStatus.ACTIVE:
        raise UserAdminError(f"user {acting_user_id!r} is not active")
    if acting.role is not Role.OWNER:
        raise UserAdminError(f"role {acting.role.value!r} may not change users; owners only")
    return acting


def _target(world: World, acting: User, user_id: str) -> User:
    target = world.store.get_user(user_id)
    # Another customer's user reads exactly like a missing one.
    if target is None or target.customer_id != acting.customer_id:
        raise UserAdminError(f"unknown user {user_id!r}")
    return target


def suspend_user(world: World, user_id: str, *, acting_user_id: str) -> User:
    """Suspend a user of the acting owner's customer, as `agentgate users suspend` does.

    The suspension is sticky: fixture sync keeps the user suspended even if the fixtures
    say active, until reinstate_user. It applies to the user's next gate call in any process
    over the same database, live runs included. Owners cannot suspend themselves, so a
    customer cannot lose its last owner this way."""
    acting = _owner_of(world, acting_user_id)
    target = _target(world, acting, user_id)
    if target.id == acting.id:
        raise UserAdminError("an owner cannot suspend themselves")
    suspended = world.store.suspend_user(target.id)
    assert suspended is not None
    return suspended


def reinstate_user(world: World, user_id: str, *, acting_user_id: str) -> User:
    """Lift an operator suspension, as `agentgate users reinstate` does, and apply the
    fixtures again, so the user's status is what the fixtures say (a user the fixtures
    suspend, or no longer list, stays suspended)."""
    acting = _owner_of(world, acting_user_id)
    target = _target(world, acting, user_id)
    if not world.store.operator_suspended(target.id):
        raise UserAdminError(f"user {user_id!r} was not suspended by an operator")
    world.store.reinstate_user(target.id)
    if world.fixtures is not None:
        sync_fixtures(world.store, world.fixtures)
    reinstated = world.store.get_user(target.id)
    assert reinstated is not None
    return reinstated
