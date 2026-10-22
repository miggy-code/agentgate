"""Open everything a gate needs in one call: store, fixtures, index, policies, audit, gate.

The CLI, the demo and tests all build their gate here, so they agree on how a world is put
together. open_world keeps its state on disk and can be reopened: fixtures are loaded only
into a database that has no customers yet, so reopening never reloads or duplicates rows,
and approvals, runs and spend from earlier sessions are still there.

Every policy must name a customer the database holds; a policy for a missing customer is
refused rather than silently ignored. A customer without a policy gets no engine, and the
gate denies its runs with `no_policy`.
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from agentgate.audit.base import AuditSink, MemorySink
from agentgate.audit.panoptes import PanoptesSink
from agentgate.clock import Clock, SystemClock
from agentgate.fixtures import FixtureError, load_fixtures
from agentgate.gate import Gate
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.testing import load_policy_bundles
from agentgate.retrieval import DocumentIndex
from agentgate.store import Store

DEFAULT_DB = Path("data/agentgate.db")
DEFAULT_AUDIT_DB = Path("data/audit.db")
DEFAULT_POLICIES = Path("policies")
DEFAULT_FIXTURES = Path("fixtures")


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
) -> World:
    """Open (or create) the database at db_path and build a gate over it.

    The audit log is a Panoptes database at audit_path unless a sink is given, in which case
    audit_path is not touched and the caller keeps ownership of the sink.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(db_path)
    try:
        return _build(store, policies, fixtures, clock, sink, audit_path)
    except BaseException:
        store.close()
        raise


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
        return _build(store, policies, fixtures, clock, MemorySink() if sink is None else sink)
    except BaseException:
        store.close()
        raise


def _build(
    store: Store,
    policies: Path,
    fixtures: Path,
    clock: Clock | None,
    sink: AuditSink | None,
    audit_path: Path | None = None,
) -> World:
    loaded = False
    if not store.list_customers():
        if not fixtures.is_dir():
            raise FixtureError(f"{fixtures}: fixtures directory not found")
        load_fixtures(store, fixtures)
        loaded = True
    index = DocumentIndex(store)
    # Also rebuilt when the index is empty, for a database populated some other way or one
    # whose earlier session stopped between loading and indexing.
    if loaded or _index_is_empty(store):
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
    )


def _index_is_empty(store: Store) -> bool:
    with store.locked():
        row = store.connection.execute("SELECT 1 FROM retrieval_lengths LIMIT 1").fetchone()
    return row is None
