"""Shared fixtures and helpers.

- `world`: a gate over the real fixtures and policies, with a memory sink and a fixed clock.
  Every test that uses it also checks, on teardown, that no audit event it produced carries
  a fixture canary.
- `Mutation` and `assert_breaks`: the one harness for red-team mutation tests.
"""

import inspect
import os
import re
import textwrap
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agentgate.audit.base import AuditEvent, AuditSink, EventKind, MemorySink
from agentgate.clock import Clock, FixedClock
from agentgate.context import RunContext
from agentgate.fixtures import find_canaries, load_fixtures
from agentgate.gate import Gate
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import load_bundles
from agentgate.redteam import core
from agentgate.redteam.core import Arena, Category, SuiteResult, all_scenarios, run_suite
from agentgate.redteam.transport import Transport
from agentgate.retrieval import DocumentIndex
from agentgate.store import Store
from agentgate.tools.base import ToolAdapter
from agentgate.world import World as RealWorld
from agentgate.world import open_world

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
POLICIES = ROOT / "policies"
KC = "kettle-creek-hvac"
JR = "juniper-row-apothecary"
# 10:00 in Chicago and 09:00 in Denver, well inside one local day for both customers.
START = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX = re.compile("[\u2500-\u257f]")


def plain_text(text: str) -> str:
    """Click usage errors with colour codes, box drawing and wrapping removed.

    Typer draws usage errors with rich, and some environments (GitHub Actions among them)
    force colour, which splits option names across escape codes.
    """
    return " ".join(_BOX.sub(" ", _ANSI.sub("", text)).split())


@dataclass
class World:
    store: Store
    index: DocumentIndex
    engines: dict[str, PolicyEngine]
    sink: MemorySink
    clock: FixedClock
    gate: Gate

    def run(self, user_id: str) -> RunContext:
        user = self.store.get_user(user_id)
        assert user is not None
        return self.gate.start_run(user.customer_id, user_id)

    def events(self, kind: EventKind | None = None) -> list[AuditEvent]:
        return [e for e in self.sink.events if kind is None or e.kind is kind]

    def last(self, kind: EventKind) -> AuditEvent:
        matching = self.events(kind)
        assert matching, f"no {kind} event"
        return matching[-1]

    def set_user(self, user_id: str, **fields: str) -> None:
        """Change a user row directly, as an admin tool would between two calls."""
        with self.store.transaction():
            for column, value in fields.items():
                assert column in {"status", "role", "customer_id"}
                self.store.connection.execute(
                    f"UPDATE users SET {column} = ? WHERE id = ?", (value, user_id)
                )

    def gate_with(
        self,
        *,
        sink: AuditSink | None = None,
        engines: Mapping[str, PolicyEngine] | None = None,
        tools: Mapping[str, ToolAdapter] | None = None,
    ) -> Gate:
        """Another gate over the same store, for a different sink, policy set or tools."""
        return Gate(
            store=self.store,
            engines=self.engines if engines is None else engines,
            sink=self.sink if sink is None else sink,
            clock=self.clock,
            index=self.index,
            tools=tools,
        )


def build_world(sink: MemorySink | None = None) -> World:
    store = Store()
    load_fixtures(store, FIXTURES)
    index = DocumentIndex(store)
    index.rebuild()
    engines: dict[str, PolicyEngine] = {}
    for customer_id, bundle in load_bundles(POLICIES).items():
        customer = store.get_customer(customer_id)
        assert customer is not None
        engines[customer_id] = PolicyEngine(bundle, customer)
    memory = MemorySink() if sink is None else sink
    clock = FixedClock(START)
    gate = Gate(store=store, engines=engines, sink=memory, clock=clock, index=index)
    return World(store=store, index=index, engines=engines, sink=memory, clock=clock, gate=gate)


def open_file_world(
    tmp_path: Path, clock: Clock | None = None, sink: AuditSink | None = None
) -> RealWorld:
    """A world on files under tmp_path/data, as the CLI opens it."""
    return open_world(
        db_path=tmp_path / "data" / "agentgate.db",
        audit_path=tmp_path / "data" / "audit.db",
        policies=POLICIES,
        fixtures=FIXTURES,
        clock=clock,
        sink=sink,
    )


def email_args(*to: str, subject: str = "Your visit", body: str = "See you Monday.") -> Any:
    return {"to": list(to), "subject": subject, "body": body}


def index_rows(store: Store) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """Every retrieval index row, in a fixed order."""
    conn = store.connection
    postings = conn.execute("SELECT * FROM retrieval_postings ORDER BY term, doc_id").fetchall()
    lengths = conn.execute("SELECT * FROM retrieval_lengths ORDER BY doc_id").fetchall()
    return [tuple(r) for r in postings], [tuple(r) for r in lengths]


def assert_no_canaries(events: list[AuditEvent]) -> None:
    dumped = "\n".join(event.model_dump_json() for event in events)
    assert find_canaries(dumped) == []


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def world() -> Iterator[World]:
    built = build_world()
    yield built
    assert_no_canaries(built.sink.events)
    built.store.close()


@pytest.fixture
def loose_umask() -> Iterator[None]:
    # The usual default, so a file SQLite created itself would come out 0644.
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


# Red-team mutation tests. A mutation removes one protection; every scenario that relies on it
# must then fail on a failed check or a leak, not a crash. Without this, a scenario that
# passes could be testing nothing. That the scenarios pass unmutated is tested once, for the
# whole suite, in test_redteam_core.py.

Edit = tuple[str, str]


@dataclass(frozen=True)
class Mutation:
    name: str
    breaks: tuple[str, ...]
    # (owner, attribute, edits): the attribute recompiled from its source with each edit.
    edits: tuple[tuple[Any, str, tuple[Edit, ...]], ...] = ()
    # (owner, attribute, replacement): the attribute replaced outright.
    replace: tuple[tuple[Any, str, Any], ...] = ()
    # Extra globals the edited source needs.
    names: Mapping[str, Any] = field(default_factory=dict)
    # Applied to each arena's world as it opens, before the arena copies its engines.
    world: Callable[[Any, pytest.MonkeyPatch], None] | None = None
    # Applied to each arena once it is built.
    arena: Callable[[Arena, pytest.MonkeyPatch], None] | None = None
    # A failed check or leak containing this must appear in every broken scenario.
    must_fail: str | None = None
    # The unprotected code raises inside some of these scenarios, which the suite also
    # counts as a failure. Otherwise a broken scenario must fail on a check or a leak.
    crashes: bool = False


def mutant(owner: Any, attribute: str, edits: tuple[Edit, ...], names: Mapping[str, Any]) -> Any:
    """The attribute recompiled from its own source with each edit applied exactly once."""
    original = inspect.getattr_static(owner, attribute)
    source = textwrap.dedent(inspect.getsource(original))
    for old, new in edits:
        assert source.count(old) == 1, f"{attribute}: edit target found {source.count(old)} times"
        source = source.replace(old, new)
    namespace = {**original.__globals__, **names}
    exec(compile(source, f"<mutant {attribute}>", "exec"), namespace)
    return namespace[attribute]


def apply_mutation(mutation: Mutation, monkeypatch: pytest.MonkeyPatch) -> None:
    for owner, attribute, edits in mutation.edits:
        monkeypatch.setattr(owner, attribute, mutant(owner, attribute, edits, mutation.names))
    for owner, attribute, replacement in mutation.replace:
        monkeypatch.setattr(owner, attribute, replacement)
    if (on_world := mutation.world) is not None:
        real_open = core.open_memory_world

        def open_mutated(**kwargs: Any) -> Any:
            opened = real_open(**kwargs)
            on_world(opened, monkeypatch)
            return opened

        monkeypatch.setattr(core, "open_memory_world", open_mutated)
    if (on_arena := mutation.arena) is not None:
        real_init = Arena.__init__

        def init(self: Arena, **kwargs: Any) -> None:
            real_init(self, **kwargs)
            on_arena(self, monkeypatch)

        monkeypatch.setattr(Arena, "__init__", init)


def run_mutated(
    mutation: Mutation,
    monkeypatch: pytest.MonkeyPatch,
    transport: Transport = Transport.GATE,
    ids: Iterable[str] | None = None,
) -> SuiteResult:
    apply_mutation(mutation, monkeypatch)
    chosen = mutation.breaks if ids is None else ids
    return run_suite(ids=chosen, policies=POLICIES, fixtures=FIXTURES, transport=transport)


def assert_breaks(
    mutation: Mutation, monkeypatch: pytest.MonkeyPatch, transport: Transport = Transport.GATE
) -> None:
    for result in run_mutated(mutation, monkeypatch, transport).results:
        found = [c.name for c in result.checks if not c.passed] + list(result.leaks)
        assert not result.passed, f"{mutation.name}: {result.id} passed without its protection"
        if not (mutation.crashes and result.error is not None):
            assert result.error is None and found, f"{result.id}: error {result.error}"
        if mutation.must_fail is not None:
            assert any(mutation.must_fail in item for item in found), (result.id, found)


def assert_every_attack_has_a_mutation(
    mutations: Iterable[Mutation], *categories: Category
) -> None:
    covered = {scenario_id for m in mutations for scenario_id in m.breaks}
    # The examples module's pe-000 has its own mutation test in test_redteam_core.py.
    attacks = {
        s.id
        for s in all_scenarios()
        if s.category in categories and not s.module.endswith(".examples")
    }
    assert attacks - covered == set()
