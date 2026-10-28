"""File worlds for scenarios about state that outlives a process.

Most scenarios run in the arena's in-memory world. A crashed host, an operator's suspension,
a fixture change, a mistyped store path or a row edited on disk need a database that
outlives a process, so FileWorlds opens worlds on files in a temporary directory, on the
arena's clock, as separate hosts would open the same files. Agent-side calls on a file world
go over the arena's transport like the arena's own: under `mcp` through an MCP session to a
gateway built for that world and run, under `gate` to its gate directly, and both are
counted in the arena's call counts.

The leak detector covers file worlds too. What each call's outcome shows is kept per world,
and before a world closes the detector runs over its store, its events and those outcomes,
against reference engines built when the world opened, so a scenario that tampers with a
world's engines cannot move entitlement. What it finds fails the scenario like an arena leak.
"""

import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import JsonValue

from agentgate.audit.base import AuditEvent, MemorySink
from agentgate.audit.panoptes import read_events
from agentgate.context import RunContext
from agentgate.gate import Outcome
from agentgate.gateway.server import build_gateway
from agentgate.policy.engine import PolicyEngine
from agentgate.redteam.core import Arena, visible_text
from agentgate.redteam.leaks import Exposure, find_leaks
from agentgate.redteam.transport import Direct, GatewaySession, GatewaySessions, Transport
from agentgate.world import DEFAULT_FIXTURES, DEFAULT_POLICIES, OpenRecord, World, open_world


@dataclass
class _Watched:
    """What the leak detector needs about one open file world."""

    reference: dict[str, PolicyEngine]
    # None for a durable world, whose events are read back from the audit log.
    sink: MemorySink | None
    exposures: list[Exposure] = field(default_factory=list)


class FileWorlds:
    """A temporary directory of file worlds, opened and called as separate hosts would.

    With `copy`, the policies and fixtures are copied in first, so a scenario may edit them.
    """

    def __init__(self, arena: Arena, *, copy: bool = False) -> None:
        self._arena = arena
        self._stack = ExitStack()
        self._tmp = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
        self._sessions = GatewaySessions()
        self._by_run: dict[tuple[int, str], GatewaySession] = {}
        self._worlds: list[World] = []
        self._watched: dict[int, _Watched] = {}
        self.db = self._tmp / "data" / "agentgate.db"
        self.audit = self._tmp / "data" / "audit.db"
        self.policies = arena.world.policies or DEFAULT_POLICIES
        self.fixtures = arena.world.fixtures or DEFAULT_FIXTURES
        if copy:
            self.policies = Path(shutil.copytree(self.policies, self._tmp / "policies"))
            self.fixtures = self.copy_fixtures()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self._sessions.close()
            if exc_type is None:
                for world in self._worlds:
                    self._check(world)
        finally:
            try:
                for world in self._worlds:
                    world.close()
            finally:
                self._stack.close()

    def path(self, name: str) -> Path:
        return self._tmp / name

    def copy_fixtures(self) -> Path:
        """A copy of the arena's fixtures this scenario may edit."""
        return Path(shutil.copytree(self.fixtures, self._tmp / "fixtures"))

    def open(self, fixtures: Path | None = None, *, durable: bool = False) -> World:
        """Open the store file as a new host would: fixtures synced, stale spend reclaimed.

        A durable world records to the Panoptes log at `self.audit`, opens included, as the
        CLI does; otherwise events go to a memory sink.
        """
        sink = None if durable else MemorySink()
        world = open_world(
            db_path=self.db,
            audit_path=self.audit,
            policies=self.policies,
            fixtures=self.fixtures if fixtures is None else fixtures,
            clock=self._arena.clock,
            **({"record_open": OpenRecord.ALL} if sink is None else {"sink": sink}),
        )
        reference: dict[str, PolicyEngine] = {}
        for customer_id, engine in world.engines.items():
            customer = world.store.get_customer(customer_id)
            if customer is not None:
                reference[customer_id] = PolicyEngine(engine.bundle, customer)
        self._watched[id(world)] = _Watched(reference=reference, sink=sink)
        self._worlds.append(world)
        return world

    def close(self, world: World) -> None:
        """Close one world, as its host exiting would, ending its MCP sessions first."""
        for key in [key for key in self._by_run if key[0] == id(world)]:
            self._by_run.pop(key).close()
        try:
            self._check(world)
        finally:
            self._worlds.remove(world)
            world.close()

    def _check(self, world: World) -> None:
        """Run the leak detector over one world, before it closes."""
        watched = self._watched.pop(id(world), None)
        if watched is None:
            return
        events = read_events(self.audit) if watched.sink is None else watched.sink.events
        self._arena.add_leaks(
            find_leaks(
                store=world.store,
                engines=watched.reference,
                events=events,
                exposures=watched.exposures,
            )
        )

    def events(self) -> list[AuditEvent]:
        return read_events(self.audit)

    def call(self, world: World, ctx: RunContext, tool: str, args: dict[str, JsonValue]) -> Outcome:
        """One agent-side call on a file world, over the arena's transport."""
        outcome = self._call(world, ctx, tool, args)
        watched = self._watched.get(id(world))
        if watched is not None:
            watched.exposures.append(Exposure(ctx.run_id, visible_text(outcome)))
        return outcome

    def _call(
        self, world: World, ctx: RunContext, tool: str, args: dict[str, JsonValue]
    ) -> Outcome:
        if self._arena.transport is Transport.GATE:
            self._arena.direct_calls[Direct.GATE_TRANSPORT] += 1
            config = world.engines[ctx.customer_id].bundle.tools[tool]
            return world.gate.call_tool(ctx, tool, sorted(config.actions)[0], args)
        key = (id(world), ctx.run_id)
        session = self._by_run.get(key)
        if session is None:
            session = self._sessions.open(f"{id(world)}-{ctx.run_id}", build_gateway(world, ctx))
            self._by_run[key] = session
        self._arena.mcp_calls += 1
        return session.call(tool, args).outcome


def edit_file(path: Path, old: str, new: str) -> bool:
    """Replace the one occurrence of old in the file; False, changing nothing, otherwise."""
    text = path.read_text()
    if text.count(old) != 1:
        return False
    path.write_text(text.replace(old, new))
    return True
