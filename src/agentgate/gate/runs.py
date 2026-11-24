"""Runs: who each run is, what it has seen, and what it may still spend.

The model only proposes calls. Who is asking comes from a RunContext the gate created in
`start` from an identity the host application already authenticated, and each call checks
that context against the gate's own `runs` row and re-reads the user, so a hand-built or
altered context, or a user suspended mid-run, is refused before anything else happens.

The run's max label is the most sensitive label of anything returned to the model so far.
It only rises, and rules such as no-external-after-confidential read it, so reading a
confidential document closes doors for the rest of the run, including for approvals granted
earlier.
"""

import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from pydantic import JsonValue

from agentgate.audit.base import EventKind
from agentgate.clock import Clock
from agentgate.context import RunContext
from agentgate.domain import Action, Customer, Role, User, UserStatus
from agentgate.gate._values import (
    TOOL_ID,
    UNKNOWN_CUSTOMER,
    as_text,
    get_customer,
    get_user,
    is_lookupable,
    safe_action,
    safe_role,
)
from agentgate.gate.outcomes import CapReached, Denied, GateReason
from agentgate.gate.recorder import AuditRecorder, Call
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import Policy, SpendLimits
from agentgate.spend import CapHit, Reservation, SpendLedger
from agentgate.store import Store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    started_at TEXT NOT NULL,
    max_label TEXT
) STRICT
"""


@dataclass(frozen=True)
class CheckedRun:
    """A run checked against its `runs` row. ctx is rebuilt from that row, not the caller."""

    ctx: RunContext
    user: User
    customer: Customer
    engine: PolicyEngine

    @property
    def policy(self) -> Policy:
        return self.engine.bundle.policy


class RunRegistry:
    """The `runs` table: starts runs, checks each call's run against it, and reserves spend."""

    def __init__(
        self,
        *,
        store: Store,
        engines: Mapping[str, PolicyEngine],
        ledger: SpendLedger,
        recorder: AuditRecorder,
        clock: Clock,
    ) -> None:
        self._store = store
        self._engines = engines
        self._ledger = ledger
        self._recorder = recorder
        self._clock = clock
        with store.locked():
            store.connection.execute(_SCHEMA)

    def start(
        self,
        customer_id: str,
        user_id: str,
        *,
        record: Callable[[RunContext], None] | None = None,
    ) -> RunContext:
        engine = self._engines.get(customer_id)
        version = None if engine is None else engine.version
        call = Call(
            time.perf_counter_ns(),
            customer_id,
            user_id=user_id,
            policy_version=version,
            claimed=frozenset({"customer_id", "user_id"}),
        )
        user = get_user(self._store, user_id)
        if user is None:
            self._recorder.identity_denied(call, GateReason.UNKNOWN_USER, "unknown user")
        call = replace(call, claimed=frozenset({"customer_id"}))
        # Checked before status, so nothing about another customer's user is revealed.
        if user.customer_id != customer_id:
            self._recorder.identity_denied(
                call, GateReason.WRONG_CUSTOMER, "user of another customer"
            )
        # The user's own customer: from here every field names something the store knows.
        call = replace(call, role=user.role, claimed=frozenset())
        if user.status is not UserStatus.ACTIVE:
            self._recorder.identity_denied(call, GateReason.SUSPENDED_USER, "user is suspended")
        customer = get_customer(self._store, customer_id)
        if engine is None or customer is None:
            self._recorder.identity_denied(call, GateReason.NO_POLICY, "customer has no policy")

        ctx = RunContext(
            run_id=str(uuid.uuid4()),
            customer_id=customer_id,
            user_id=user.id,
            role=user.role,
            started_at=self._clock.now(),
        )
        # Recorded inside the transaction, so a run the audit log never saw does not exist.
        with self._store.transaction():
            self._store.connection.execute(
                "INSERT INTO runs (run_id, customer_id, user_id, role, started_at, max_label)"
                " VALUES (?, ?, ?, ?, ?, NULL)",
                (
                    ctx.run_id,
                    ctx.customer_id,
                    ctx.user_id,
                    ctx.role.value,
                    ctx.started_at.isoformat(),
                ),
            )
            if record is not None:
                record(ctx)
            self._recorder.emit(
                replace(call, run_id=ctx.run_id), EventKind.RUN_STARTED, GateReason.RUN_STARTED
            )
        return ctx

    def max_label(self, run_id: str) -> str | None:
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT max_label FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else row["max_label"]

    def begin(self, ctx: RunContext, tool: str | None, action: Action | str | None) -> Call:
        """A call described by what the caller claims, until the run row confirms it."""
        claimed_customer = getattr(ctx, "customer_id", None)
        claimed_user = getattr(ctx, "user_id", None)
        claimed_run = getattr(ctx, "run_id", None)
        return Call(
            time.perf_counter_ns(),
            claimed_customer if isinstance(claimed_customer, str) else UNKNOWN_CUSTOMER,
            user_id=claimed_user if isinstance(claimed_user, str) else None,
            role=safe_role(getattr(ctx, "role", None)),
            run_id=claimed_run if isinstance(claimed_run, str) else None,
            tool=tool if tool is not None and TOOL_ID.fullmatch(tool) else None,
            action=safe_action(action),
            claimed=frozenset({"customer_id", "user_id", "run_id"}),
            asked_tool=as_text(tool),
        )

    def check(self, ctx: RunContext, call: Call) -> CheckedRun | Denied:
        if not isinstance(ctx, RunContext) or not isinstance(ctx.run_id, str):
            return self._recorder.deny(call, GateReason.INVALID_RUN)
        if not is_lookupable(ctx.run_id):
            return self._recorder.deny(call, GateReason.INVALID_RUN, details={"run_known": False})
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (ctx.run_id,)
            ).fetchone()
        if row is None:
            return self._recorder.deny(call, GateReason.INVALID_RUN, details={"run_known": False})
        engine = self._engines.get(row["customer_id"])
        # From here the event names the real run, not whatever the caller claimed.
        call = replace(
            call,
            customer_id=row["customer_id"],
            user_id=row["user_id"],
            role=Role(row["role"]),
            policy_version=None if engine is None else engine.version,
            claimed=frozenset(),
        )
        mismatched: list[JsonValue] = [
            name
            for name, claimed in (
                ("customer_id", ctx.customer_id),
                ("user_id", ctx.user_id),
                ("role", ctx.role),
            )
            if row[name] != claimed
        ]
        if mismatched:
            return self._recorder.deny(
                call, GateReason.INVALID_RUN, details={"mismatched": mismatched}
            )

        user = self._store.get_user(row["user_id"])
        if user is None:
            return self._recorder.deny(call, GateReason.UNKNOWN_USER)
        if user.customer_id != row["customer_id"]:
            return self._recorder.deny(call, GateReason.WRONG_CUSTOMER)
        if user.status is not UserStatus.ACTIVE:
            return self._recorder.deny(call, GateReason.SUSPENDED_USER)
        if user.role.value != row["role"]:
            return self._recorder.deny(call, GateReason.ROLE_CHANGED)
        customer = self._store.get_customer(row["customer_id"])
        if engine is None or customer is None:
            return self._recorder.deny(call, GateReason.NO_POLICY)
        trusted = RunContext(
            run_id=row["run_id"],
            customer_id=row["customer_id"],
            user_id=row["user_id"],
            role=Role(row["role"]),
            started_at=datetime.fromisoformat(row["started_at"]),
        )
        return CheckedRun(ctx=trusted, user=user, customer=customer, engine=engine)

    def reserve(
        self,
        run: CheckedRun,
        call: Call,
        amount: Decimal,
        details: dict[str, JsonValue] | None = None,
        limits: SpendLimits | None = None,
    ) -> Reservation | CapReached:
        reserved = self._ledger.reserve(
            customer=run.customer,
            run_id=run.ctx.run_id,
            amount_usd=amount,
            limits=run.policy.spend if limits is None else limits,
        )
        if isinstance(reserved, Reservation):
            return reserved
        self._recorder.emit(
            call,
            EventKind.CAP_HIT,
            reserved.cap.value,
            subject=reserved.cap.value,
            decision=True,
            details=_cap_details(reserved) | (details or {}),
        )
        return CapReached(cap=reserved.cap)

    def raise_label(self, run: CheckedRun, labels: Iterable[str]) -> str | None:
        with self._store.transaction():
            row = self._store.connection.execute(
                "SELECT max_label FROM runs WHERE run_id = ?", (run.ctx.run_id,)
            ).fetchone()
            current: str | None = row["max_label"]
            updated = current
            for label in labels:
                updated = _higher(run.policy, updated, label)
            if updated != current:
                self._store.connection.execute(
                    "UPDATE runs SET max_label = ? WHERE run_id = ?", (updated, run.ctx.run_id)
                )
        return updated

    def role(self, run_id: str) -> Role | None:
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT role FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else Role(row["role"])


def bind(call: Call, run: CheckedRun) -> Call:
    return replace(
        call,
        customer_id=run.ctx.customer_id,
        user_id=run.ctx.user_id,
        role=run.ctx.role,
        run_id=run.ctx.run_id,
        policy_version=run.engine.version,
        claimed=frozenset(),
    )


def _cap_details(hit: CapHit) -> dict[str, JsonValue]:
    def number(value: Decimal | int) -> JsonValue:
        return value if isinstance(value, int) else str(value)

    return {
        "used": number(hit.used),
        "limit": number(hit.limit),
        "requested": str(hit.requested),
    }


def _higher(policy: Policy, current: str | None, label: str) -> str | None:
    if current is None:
        return label
    current_rank = policy.label_rank(current)
    label_rank = policy.label_rank(label)
    # An unknown label is kept: the engine denies every later request of a run whose max
    # label it does not know, which is the safe way to fail.
    if label_rank is None:
        return label
    if current_rank is None:
        return current
    return label if label_rank > current_rank else current
