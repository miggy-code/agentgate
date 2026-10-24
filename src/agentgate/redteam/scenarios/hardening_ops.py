"""Operational hardening: what a crashed host, an operator's suspension, a fixture change and
a mistyped path do to the gate.

These are the failures of running agentgate rather than of attacking it: a host killed
between reserving and settling a call, a user suspended from the command line while their
run is live, fixtures edited under an existing database, and a store pointed at the audit
log. Most of them need a database that outlives a process, so beside the arena's in-memory
world a scenario opens file worlds in a temporary directory (`_Disk`), on the arena's clock
and with a memory sink, as separate processes would open the same files. Agent-side calls
on a file world go over the arena's transport like the arena's own: under `mcp` through an
MCP session to a gateway built for that world and run, under `gate` to its gate directly,
and both are counted in the arena's call counts.

Check names and details carry no paths, ids or timestamps, so results stay deterministic.
"""

import hashlib
import shutil
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import JsonValue

from agentgate.audit.base import EventKind, MemorySink
from agentgate.audit.panoptes import PanoptesSink
from agentgate.context import RunContext
from agentgate.fixtures import sync_fixtures
from agentgate.gate import IdentityError, Outcome
from agentgate.gateway.server import build_gateway
from agentgate.models import Role, UserStatus
from agentgate.redteam.core import Arena, Category, describe, scenario
from agentgate.redteam.transport import Direct, GatewaySession, GatewaySessions, Transport
from agentgate.spend import Reservation
from agentgate.store import Store, table_names
from agentgate.world import (
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    DEFAULT_RESERVATION_LEASE,
    UserAdminError,
    World,
    open_world,
    suspend_user,
)

KC = "kettle-creek-hvac"
FC = Category.FAIL_CLOSED
ID = Category.IDENTITY_SPOOFING

OWNER = "kc-owner-dale"
MANAGER = "kc-mgr-marisol"
STAFF = "kc-staff-nadine"
OTHER_STAFF = "kc-staff-priya"
JR_OWNER = "jr-owner-marisol"

SERVICE_MENU = "kc-service-menu"
DISPATCH = "kc-dispatch-procedure"
CUSTOMER_LIST = "kc-customer-list"

# The per-run cap in the shipped policy, and enough crashed runs to fill the 20.00 day cap.
RUN_CAP = Decimal("0.50")
CRASHES = 40
LEASE = DEFAULT_RESERVATION_LEASE
LEASE_MINUTES = int(LEASE.total_seconds() // 60)


def _read(document_id: str) -> dict[str, JsonValue]:
    return {"document_id": document_id}


class _Disk:
    """A temporary directory of file worlds, opened and called as separate hosts would."""

    def __init__(self, arena: Arena) -> None:
        self._arena = arena
        self._stack = ExitStack()
        self._tmp = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
        self._sessions = GatewaySessions()
        self._by_run: dict[tuple[int, str], GatewaySession] = {}
        self._worlds: list[World] = []
        world = arena.world
        self.policies = world.policies or DEFAULT_POLICIES
        self.fixtures = world.fixtures or DEFAULT_FIXTURES
        self.db = self._tmp / "data" / "agentgate.db"

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
        target = self._tmp / "fixtures"
        shutil.copytree(self.fixtures, target)
        return target

    def open(self, fixtures: Path | None = None) -> World:
        """Open the store file as a new host would: fixtures synced, stale spend reclaimed."""
        world = open_world(
            db_path=self.db,
            audit_path=self._tmp / "data" / "audit.db",
            policies=self.policies,
            fixtures=self.fixtures if fixtures is None else fixtures,
            clock=self._arena.clock,
            sink=MemorySink(),
        )
        self._worlds.append(world)
        return world

    def close(self, world: World) -> None:
        """Close one world, as its host exiting would, ending its MCP sessions first."""
        for key in [key for key in self._by_run if key[0] == id(world)]:
            self._by_run.pop(key).close()
        self._worlds.remove(world)
        world.close()

    def call(self, world: World, ctx: RunContext, tool: str, args: dict[str, JsonValue]) -> Outcome:
        """One agent-side call on a file world, over the arena's transport."""
        if self._arena.transport is Transport.GATE:
            self._arena.direct_calls[Direct.GATE_TRANSPORT] += 1
            engine = world.engines[ctx.customer_id]
            config = engine.bundle.tools[tool]
            return world.gate.call_tool(ctx, tool, sorted(config.actions)[0], args)
        key = (id(world), ctx.run_id)
        session = self._by_run.get(key)
        if session is None:
            session = self._sessions.open(f"{id(world)}-{ctx.run_id}", build_gateway(world, ctx))
            self._by_run[key] = session
        self._arena.mcp_calls += 1
        return session.call(tool, args).outcome


def _recount(store: Store, customer_id: str) -> Decimal:
    """The customer's spend summed straight from the reservation rows, every day together."""
    with store.locked():
        rows = store.connection.execute(
            "SELECT status, reserved_usd, actual_usd FROM spend_reservations"
            " WHERE customer_id = ? AND status != 'released'",
            (customer_id,),
        ).fetchall()
    return sum(
        (Decimal(r["actual_usd"] if r["status"] == "settled" else r["reserved_usd"]) for r in rows),
        Decimal(0),
    )


def _day_spent(world: World, customer_id: str = KC) -> Decimal:
    customer = world.store.get_customer(customer_id)
    assert customer is not None
    return world.gate.ledger.customer_day_spent(
        customer_id, world.gate.ledger.customer_day(customer)
    )


def _crash(gate_charge: Callable[[RunContext, Decimal], object], ctxs: list[RunContext]) -> int:
    """Each run reserves a model call at the run cap; its host then dies before settling."""
    reserved = 0
    for ctx in ctxs:
        if isinstance(gate_charge(ctx, RUN_CAP), Reservation):
            reserved += 1
    return reserved


@scenario(
    "fc-015",
    FC,
    "Leaked spend reservations of crashed runs are reclaimed after the lease",
)
def leaked_reservations_are_reclaimed(arena: Arena) -> None:
    # Forty hosts each reserve a model call at the per-run cap and die before settling.
    runs = [arena.start(STAFF) for _ in range(CRASHES)]
    reservations = [arena.gate.charge_model(ctx, RUN_CAP) for ctx in runs]
    leaked = [r for r in reservations if isinstance(r, Reservation)]
    arena.expect("every crashed run reserved", len(leaked) == CRASHES, f"{len(leaked)} reserved")
    owner = arena.start(OWNER)
    arena.expect_cap(
        arena.call(owner, "docs.read", _read(SERVICE_MENU)),
        "customer_day_cap",
        name="leaked reservations fill the customer day",
    )

    # Inside the lease a reservation may be a slow call still in flight: it is kept.
    arena.advance(LEASE_MINUTES - 1)
    early = arena.gate.ledger.release_stale(LEASE)
    arena.expect("nothing reclaimed inside the lease", early == 0, f"{early} reclaimed")
    arena.advance(2)
    reclaimed = arena.gate.ledger.release_stale(LEASE)
    arena.expect(
        "the sweep reclaims every leaked reservation", reclaimed == CRASHES, f"{reclaimed}"
    )
    spent = _day_spent(arena.world)
    arena.expect("the customer day is free again", spent == 0, f"day spent {spent}")
    arena.expect_executed(
        arena.call(owner, "docs.read", _read(SERVICE_MENU)), name="a fresh run's call executes"
    )
    fresh = arena.gate.charge_model(arena.start(OWNER), Decimal("0.10"))
    arena.expect("a fresh model call is reserved", isinstance(fresh, Reservation), describe(fresh))

    # A host that was only slow settles after the sweep: the money it spent still counts.
    if leaked:
        arena.gate.settle_model(runs[0], leaked[0], Decimal("0.40"))
        arena.gate.ledger.release(leaked[1])
    spent = _day_spent(arena.world)
    recount = _recount(arena.store, KC)
    arena.expect("a late settle is charged", spent >= Decimal("0.40"), f"day spent {spent}")
    arena.expect("the day total matches a recount", spent == recount, f"{spent} vs {recount}")
    calls = arena.gate.ledger.run_calls(runs[2].run_id)
    arena.expect("a reclaimed call still counts toward its run", calls == 1, f"{calls} calls")

    # The same crash on disk: the next host to open the store reclaims the leaks itself.
    with _Disk(arena) as disk:
        host = disk.open()
        crashed = _crash(
            host.gate.charge_model, [host.gate.start_run(KC, STAFF) for _ in range(CRASHES)]
        )
        arena.expect("every crashed host reserved", crashed == CRASHES, f"{crashed} reserved")
        disk.close(host)
        arena.advance(LEASE_MINUTES + 1)
        reopened = disk.open()
        arena.expect(
            "reopening reclaims the leaks",
            reopened.stale_released == CRASHES,
            f"{reopened.stale_released} reclaimed",
        )
        spent = _day_spent(reopened)
        arena.expect("the reopened day is free", spent == 0, f"day spent {spent}")
        ctx = reopened.gate.start_run(KC, OWNER)
        arena.expect_executed(
            disk.call(reopened, ctx, "docs.read", _read(SERVICE_MENU)),
            name="a run on the reopened store executes",
        )


def _refused(arena: Arena, attempt: Callable[[], object], name: str, expected: str) -> None:
    try:
        attempt()
        got = "allowed"
    except UserAdminError as exc:
        got = str(exc)
    arena.expect(name, got == expected, f"got {got}")


@scenario("id-015", ID, "A user suspended through the CLI is refused on the live run's next call")
def operator_suspension_reaches_live_runs(arena: Arena) -> None:
    live = arena.start(STAFF)
    arena.expect_executed(arena.call(live, "docs.read", _read(DISPATCH)), name="read before")

    # Only an owner of the user's own customer may suspend them.
    _refused(
        arena,
        lambda: suspend_user(arena.world, OTHER_STAFF, acting_user_id=STAFF),
        "staff cannot suspend",
        "role 'staff' may not change users; owners only",
    )
    _refused(
        arena,
        lambda: suspend_user(arena.world, STAFF, acting_user_id=JR_OWNER),
        "another customer's owner sees no such user",
        f"unknown user {STAFF!r}",
    )
    _refused(
        arena,
        lambda: suspend_user(arena.world, OWNER, acting_user_id=OWNER),
        "an owner cannot suspend themselves",
        "an owner cannot suspend themselves",
    )
    # What `agentgate users suspend kc-staff-nadine --as kc-owner-dale` does.
    suspend_user(arena.world, STAFF, acting_user_id=OWNER)
    calls: tuple[tuple[str, dict[str, JsonValue]], ...] = (
        ("docs.read", _read(DISPATCH)),
        ("docs.search", {"query": "dispatch procedure"}),
        ("email.send", {"to": ["dale@kettlecreek.example"], "subject": "Hi", "body": "Hello."}),
    )
    for tool, args in calls:
        arena.expect_denied(
            arena.call(live, tool, args), "suspended_user", name=f"{tool} after suspension"
        )
    arena.expect_audit(EventKind.DENY, reason="suspended_user", run=live)
    started = arena.try_start(STAFF)
    got = started.reason if isinstance(started, IdentityError) else describe(started)
    arena.expect("no new run", got == "suspended_user", f"got {got}")

    # Sticky: the fixtures still say active, and syncing them must not reactivate her.
    fixtures = arena.world.fixtures or DEFAULT_FIXTURES
    sync = sync_fixtures(arena.store, fixtures)
    user = arena.store.get_user(STAFF)
    status = "missing" if user is None else user.status.value
    arena.expect("fixture sync keeps the suspension", status == "suspended", status)
    arena.expect("the sync says so", STAFF in sync.users_kept_suspended, "")
    arena.expect_denied(
        arena.call(live, "docs.read", _read(DISPATCH)), "suspended_user", name="still refused"
    )

    # Two hosts on one store file: the gateway's live run, and a CLI process suspending.
    with _Disk(arena) as disk:
        gateway = disk.open()
        ctx = gateway.gate.start_run(KC, OTHER_STAFF)
        arena.expect_executed(
            disk.call(gateway, ctx, "docs.read", _read(DISPATCH)), name="live run reads"
        )
        cli = disk.open()
        suspend_user(cli, OTHER_STAFF, acting_user_id=OWNER)
        disk.close(cli)
        arena.expect_denied(
            disk.call(gateway, ctx, "docs.read", _read(DISPATCH)),
            "suspended_user",
            name="the other connection's live run is refused",
        )
        reopened = disk.open()
        again = reopened.store.get_user(OTHER_STAFF)
        status = "missing" if again is None else again.status.value
        arena.expect("still suspended after a reopen", status == "suspended", status)


def _edit(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if text.count(old) != 1:
        return False
    path.write_text(text.replace(old, new))
    return True


@scenario("id-016", ID, "A role change and a suspension in the fixtures take effect on reopen")
def fixture_changes_apply_on_reopen(arena: Arena) -> None:
    with _Disk(arena) as disk:
        fixtures = disk.copy_fixtures()
        first = disk.open(fixtures)
        manager = first.gate.start_run(KC, MANAGER)
        staff = first.gate.start_run(KC, OTHER_STAFF)
        arena.expect_executed(
            disk.call(first, manager, "docs.read", _read(CUSTOMER_LIST)), name="manager reads"
        )
        arena.expect_executed(
            disk.call(first, staff, "docs.read", _read(DISPATCH)), name="staff reads"
        )
        disk.close(first)

        users = fixtures / KC / "users.yaml"
        demoted = _edit(
            users,
            "email: marisol@kettlecreek.example\n  role: manager",
            "email: marisol@kettlecreek.example\n  role: staff",
        )
        suspended = _edit(
            users,
            "email: priya@kettlecreek.example\n  role: staff\n  status: active",
            "email: priya@kettlecreek.example\n  role: staff\n  status: suspended",
        )
        arena.expect("fixtures edited", demoted and suspended, "")

        second = disk.open(fixtures)
        sync = second.fixture_sync
        updated = dict(sync.users_updated) if sync is not None else {}
        arena.expect(
            "the sync reports both changes",
            updated == {MANAGER: ("role",), OTHER_STAFF: ("status",)},
            ", ".join(sorted(updated)),
        )
        arena.expect_denied(
            disk.call(second, manager, "docs.read", _read(CUSTOMER_LIST)),
            "role_changed",
            name="the manager's old run is refused",
        )
        arena.expect_denied(
            disk.call(second, staff, "docs.read", _read(DISPATCH)),
            "suspended_user",
            name="the suspended user's old run is refused",
        )
        fresh = second.gate.start_run(KC, MANAGER)
        arena.expect("a new run has the new role", fresh.role is Role.STAFF, fresh.role.value)
        arena.expect_denied(
            disk.call(second, fresh, "docs.read", _read(CUSTOMER_LIST)),
            name="the demoted user cannot read the customer list",
        )
        try:
            second.gate.start_run(KC, OTHER_STAFF)
            got = "started"
        except IdentityError as err:
            got = err.reason
        arena.expect("the suspended user cannot start a run", got == "suspended_user", got)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


def _refused_open(arena: Arena, name: str, attempt: Callable[[], World]) -> None:
    try:
        world = attempt()
    except ValueError as exc:
        got = f"refused {type(exc).__name__}"
    else:
        world.close()
        got = "opened"
    arena.expect(name, got.startswith("refused"), got)


@scenario("fc-016", FC, "A store pointed at the audit log, or sharing its file, is refused")
def store_pointed_at_the_audit_log(arena: Arena) -> None:
    with _Disk(arena) as disk:
        store_path = disk.path("agentgate.db")
        audit_path = disk.path("audit.db")

        def world_at(db: Path, audit: Path) -> World:
            return open_world(
                db_path=db,
                audit_path=audit,
                policies=disk.policies,
                fixtures=disk.fixtures,
                clock=arena.clock,
            )

        honest = world_at(store_path, audit_path)
        honest.gate.start_run(KC, STAFF)
        honest.close()
        audit_before = _digest(audit_path)
        store_before = _digest(store_path)

        _refused_open(
            arena, "--db at the audit log", lambda: world_at(audit_path, disk.path("x.db"))
        )
        arena.expect("the audit log is untouched", _digest(audit_path) == audit_before, "")
        tables = table_names(audit_path) or set()
        arena.expect("no store tables in the audit log", "customers" not in tables, "")

        same = disk.path("same.db")
        _refused_open(arena, "--db and --audit-db the same file", lambda: world_at(same, same))
        arena.expect("nothing created for the shared path", not same.exists(), "")

        _refused_open(
            arena, "--audit-db at the store", lambda: world_at(disk.path("y.db"), store_path)
        )
        arena.expect("the store is untouched", _digest(store_path) == store_before, "")

        sink = PanoptesSink(audit_path, create=False)
        try:
            valid = bool(sink.verify()["valid"])
            count = sink.count()
        finally:
            sink.close()
        arena.expect("the audit log still verifies", valid and count == 1, f"{count} events")

        # A world on the same paths still opens, so the refusals were about the paths.
        reopened = world_at(store_path, audit_path)
        user = reopened.store.get_user(STAFF)
        reopened.close()
        arena.expect(
            "the honest paths still open",
            user is not None and user.status is UserStatus.ACTIVE,
            "",
        )


@dataclass(frozen=True)
class _Modes:
    directory: int
    store: int


def _modes(directory: Path, store: Path) -> _Modes:
    return _Modes(directory.stat().st_mode & 0o777, store.stat().st_mode & 0o777)


@scenario("fc-017", FC, "The store file is owner-only, and a world-readable one is tightened")
def store_file_is_owner_only(arena: Arena) -> None:
    with _Disk(arena) as disk:
        world = disk.open()
        disk.close(world)
        modes = _modes(disk.db.parent, disk.db)
        arena.expect("the store file is 0600", modes.store & 0o077 == 0, f"{modes.store:04o}")
        arena.expect(
            "the directory made for it is 0700",
            modes.directory & 0o077 == 0,
            f"{modes.directory:04o}",
        )
        disk.db.chmod(0o644)
        reopened = disk.open()
        before = reopened.store.tightened_from
        disk.close(reopened)
        mode = disk.db.stat().st_mode & 0o777
        arena.expect("a world-readable store is tightened on open", mode == 0o600, f"{mode:04o}")
        arena.expect("the old mode is reported", before == 0o644, f"{before}")
