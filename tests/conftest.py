"""Shared fixtures: a gate over the real fixtures and policies, with a memory sink and a fixed
clock. Every test that uses `world` also checks, on teardown, that no audit event it
produced carries a fixture canary.
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentgate.audit.base import AuditEvent, AuditSink, EventKind, MemorySink
from agentgate.clock import FixedClock
from agentgate.context import RunContext
from agentgate.fixtures import find_canaries, load_fixtures
from agentgate.gate import Gate
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import load_bundles
from agentgate.retrieval import DocumentIndex
from agentgate.store import Store
from agentgate.tools.base import ToolAdapter

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


def assert_no_canaries(events: list[AuditEvent]) -> None:
    dumped = "\n".join(event.model_dump_json() for event in events)
    assert find_canaries(dumped) == []


@pytest.fixture
def world() -> Iterator[World]:
    built = build_world()
    yield built
    assert_no_canaries(built.sink.events)
    built.store.close()
