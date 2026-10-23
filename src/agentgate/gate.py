"""The gate: every tool call an agent run proposes is checked, decided, recorded, then run.

The model only proposes calls. Who is asking comes from a RunContext the gate created in
start_run from an identity the host application already authenticated, and each call checks
that context against the gate's own `runs` row and re-reads the user, so a hand-built or
altered context, or a user suspended mid-run, is refused before anything else happens.

A call then goes through the same steps every time: strict argument validation, the facts
of the call (a document is loaded and checked for tenant first), a spend reservation, the
policy decision with the run's max label, escalation to approval of allows that are risky
or that reach someone outside the company through a customer-facing tool, and only for an
allow, an audit record and then the tool. If the allow cannot be
recorded the tool does not run. Every outcome becomes at least one audit event carrying ids,
labels, counts and reason codes, never document bodies, message text or argument values.

Approval state changes follow the same rule as run starts: each one is recorded inside the
transaction that makes it. An approval is created, approved, rejected, expired or claimed
for execution only if its event (APPROVAL_REQUIRED, APPROVED, REJECTED, EXPIRED, or the
ALLOW of the approved call) is recorded; if the sink refuses the event, the change is rolled
back. A claim whose ALLOW cannot be recorded leaves the approval approved and unused.

The run's max label is the most sensitive label of anything returned to the model so far.
It only rises, and rules such as no-external-after-confidential read it, so reading a
confidential document closes doors for the rest of the run, including for approvals granted
earlier: execute_approved decides again with the current label and policy.

Model-facing reasons are the engine's reason codes, with one exception: every denial of a
call that names a document (a missing id, another customer's document, or one of the run's
own customer that the run may not read) reads as `not_found`, so document ids, which
reveal titles, cannot be probed within or across tenants. Such calls are decided before
spend, so a cap cannot tell them apart either. The audit event keeps the real reason.
"""

import json
import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import NoReturn

from pydantic import BaseModel, JsonValue, ValidationError

from agentgate.approvals import (
    Approval,
    ApprovalError,
    ApprovalErrorReason,
    ApprovalQueue,
    args_hash,
)
from agentgate.audit.base import AuditEvent, AuditSink, EventKind
from agentgate.clock import Clock
from agentgate.context import RunContext
from agentgate.models import Action, Customer, Role, Tool, User, UserStatus
from agentgate.policy.engine import DocumentFacts, PolicyEngine, Reason, Request
from agentgate.policy.schema import Effect, Policy
from agentgate.retrieval import DocumentIndex, RetrievalInvariantError, RetrievalResult
from agentgate.spend import CapHit, CapKind, Reservation, SpendLedger
from agentgate.store import Store
from agentgate.tools.base import ToolAdapter, ToolEnv, ToolFacts, ToolResult
from agentgate.tools.builtin import default_tools

SEARCH_TOOL = "docs.search"
# The tags of a note being written, as the engine sees it when deciding the write.
NOTE_TAGS = frozenset({"note"})

# Risk added to a tool's base risk: any recipient outside the customer's domains, and 5 per
# full 100 USD of spend up to 40.
EXTERNAL_RECIPIENT_RISK = 25
SPEND_RISK_PER_100_USD = 5
MAX_SPEND_RISK = 40

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

# Values the model chose (a tool name) go into events only in this shape, which cannot
# carry free text or a canary.
_TOOL_ID = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_APPROVAL_ID = re.compile(r"^apr_[A-Za-z0-9_-]{1,64}$")
_ADDRESS = re.compile(r"^[^@\s,;:<>()\[\]\"\\]+@([a-z0-9-]+(?:\.[a-z0-9-]+)+)$")
# Customer id for audit events about a caller that matches no customer at all.
UNKNOWN_CUSTOMER = "unknown"


class GateReason(StrEnum):
    """Reason codes the gate adds to the engine's Reason values."""

    RUN_STARTED = "run_started"
    INVALID_RUN = "invalid_run"
    UNKNOWN_USER = "unknown_user"
    SUSPENDED_USER = "suspended_user"
    WRONG_CUSTOMER = "wrong_customer"
    # The user's role changed since the run started; the run's grants no longer apply.
    ROLE_CHANGED = "role_changed"
    NO_POLICY = "no_policy"
    UNKNOWN_TOOL = "unknown_tool"
    ACTION_NOT_SUPPORTED = "action_not_supported"
    INVALID_ARGS = "invalid_args"
    NOT_FOUND = "not_found"
    DOCUMENT_NOT_FOUND = "document_not_found"
    APPROVAL_NOT_FOUND = "approval_not_found"
    APPROVAL_RUN_MISMATCH = "approval_run_mismatch"
    WRITE_DOWN = "write_down"
    ESCALATED_CUSTOMER_FACING = "escalated_customer_facing"
    ESCALATED_RISK = "escalated_risk"
    AUDIT_UNAVAILABLE = "audit_unavailable"
    TOOL_ERROR = "tool_error"
    RETRIEVAL_FILTER = "retrieval_filter"
    RETRIEVAL_INVARIANT = "retrieval_invariant"
    EXECUTED = "executed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class Executed:
    tool: str
    content: dict[str, JsonValue]
    cost_usd: Decimal


@dataclass(frozen=True)
class Denied:
    # Model-facing reason code.
    reason: str


@dataclass(frozen=True)
class PendingApproval:
    approval_id: str
    expires_at: datetime


@dataclass(frozen=True)
class CapReached:
    cap: CapKind


Outcome = Executed | Denied | PendingApproval | CapReached


class IdentityError(Exception):
    """A run cannot start, or a model charge cannot be made, for this identity."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class _Run:
    """A run checked against its `runs` row. ctx is rebuilt from that row, not the caller."""

    ctx: RunContext
    user: User
    customer: Customer
    engine: PolicyEngine

    @property
    def policy(self) -> Policy:
        return self.engine.bundle.policy


@dataclass(frozen=True)
class _Call:
    """Who and what one gate call is about, for its audit events."""

    started_ns: int
    customer_id: str
    user_id: str | None = None
    role: Role | None = None
    run_id: str | None = None
    tool: str | None = None
    action: Action | None = None
    policy_version: str | None = None

    def latency_us(self) -> int:
        return (time.perf_counter_ns() - self.started_ns) // 1000


@dataclass(frozen=True)
class _Prepared:
    adapter: ToolAdapter
    config: Tool
    action: Action
    args: BaseModel
    facts: ToolFacts
    document: DocumentFacts | None
    # The id of a document the call reads, for event subjects.
    document_id: str | None


@dataclass(frozen=True)
class _Verdict:
    effect: Effect
    reason: str
    rule_ids: tuple[str, ...]
    policy_version: str


class Gate:
    """Decides, records and runs the tool calls of agent runs for every customer."""

    def __init__(
        self,
        *,
        store: Store,
        engines: Mapping[str, PolicyEngine],
        sink: AuditSink,
        clock: Clock,
        index: DocumentIndex,
        tools: Mapping[str, ToolAdapter] | None = None,
    ) -> None:
        for customer_id, engine in engines.items():
            if customer_id != engine.bundle.customer_id:
                raise ValueError(
                    f"engine for {engine.bundle.customer_id!r} keyed as {customer_id!r}"
                )
        adapters = default_tools() if tools is None else dict(tools)
        for tool_id, adapter in adapters.items():
            if tool_id != adapter.tool_id:
                raise ValueError(f"tool {adapter.tool_id!r} keyed as {tool_id!r}")
        self._store = store
        self._engines = dict(engines)
        self._sink = sink
        self._clock = clock
        self._index = index
        self._tools = adapters
        self._ledger = SpendLedger(store, clock)
        self._approvals = ApprovalQueue(store, clock)
        with store.locked():
            store.connection.execute(_SCHEMA)

    @property
    def approvals(self) -> ApprovalQueue:
        return self._approvals

    @property
    def ledger(self) -> SpendLedger:
        return self._ledger

    # Runs

    def start_run(self, customer_id: str, user_id: str) -> RunContext:
        engine = self._engines.get(customer_id)
        version = None if engine is None else engine.version
        call = _Call(time.perf_counter_ns(), customer_id, user_id=user_id, policy_version=version)
        user = self._store.get_user(user_id)
        if user is None:
            self._identity_denied(call, GateReason.UNKNOWN_USER, "unknown user")
        # Checked before status, so nothing about another customer's user is revealed.
        if user.customer_id != customer_id:
            self._identity_denied(call, GateReason.WRONG_CUSTOMER, "user of another customer")
        call = replace(call, role=user.role)
        if user.status is not UserStatus.ACTIVE:
            self._identity_denied(call, GateReason.SUSPENDED_USER, "user is suspended")
        customer = self._store.get_customer(customer_id)
        if engine is None or customer is None:
            self._identity_denied(call, GateReason.NO_POLICY, "customer has no policy")

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
            self._emit(
                replace(call, run_id=ctx.run_id), EventKind.RUN_STARTED, GateReason.RUN_STARTED
            )
        return ctx

    def run_max_label(self, run_id: str) -> str | None:
        """The most sensitive label returned to the model in this run, or None."""
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT max_label FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else row["max_label"]

    # Tool calls

    def call_tool(
        self, ctx: RunContext, tool: str, action: Action, args: Mapping[str, JsonValue]
    ) -> Outcome:
        call = self._begin(ctx, tool, action)
        run = self._check_run(ctx, call)
        if isinstance(run, Denied):
            return run
        call = _bind(call, run)
        prepared = self._prepare(run, call, tool, action, args)
        if isinstance(prepared, Denied):
            return prepared
        if prepared.config.id == SEARCH_TOOL:
            return self._search(run, call, prepared)

        subject = prepared.document_id
        verdict: _Verdict | None = None
        if subject is not None:
            # A call naming a document is decided before spend, so a denied id and a
            # missing one take the same path: neither reaches the ledger, so a cap cannot
            # tell them apart either.
            verdict = self._verdict(run, prepared)
            if verdict.effect is Effect.DENY:
                return self._deny_verdict(call, prepared, verdict, subject)
        reservation = self._reserve(run, call, prepared.config.cost_usd)
        if isinstance(reservation, CapReached):
            return reservation
        if verdict is None:
            verdict = self._verdict(run, prepared)

        if verdict.effect is Effect.ALLOW:
            escalation = self._escalation(run, prepared)
            if escalation is not None:
                verdict = replace(verdict, effect=Effect.REQUIRE_APPROVAL, reason=escalation)

        if verdict.effect is Effect.DENY:
            self._ledger.release(reservation)
            return self._deny_verdict(call, prepared, verdict, subject)

        if verdict.effect is Effect.REQUIRE_APPROVAL:
            self._ledger.release(reservation)
            risk = self._risk(run, prepared)

            def record_request(created: Approval) -> None:
                self._emit(
                    call,
                    EventKind.APPROVAL_REQUIRED,
                    verdict.reason,
                    subject=created.id,
                    rule_ids=verdict.rule_ids,
                    policy_version=verdict.policy_version,
                    decision=True,
                    details={"risk_score": risk, "document_id": subject},
                )

            # A request the audit log cannot record raises here and leaves no approval.
            approval = self._approvals.create(
                customer_id=run.ctx.customer_id,
                run_id=run.ctx.run_id,
                requested_by=run.ctx.user_id,
                tool=prepared.config.id,
                action=prepared.action,
                args=dict(args),
                risk_score=risk,
                rule_ids=verdict.rule_ids,
                policy_version=verdict.policy_version,
                ttl=timedelta(minutes=run.policy.approvals.ttl_minutes),
                record=record_request,
            )
            return PendingApproval(approval_id=approval.id, expires_at=approval.expires_at)

        return self._execute(run, call, prepared, reservation, verdict, subject=subject)

    def execute_approved(
        self,
        ctx: RunContext,
        approval_id: str,
        tool: str,
        action: Action,
        args: Mapping[str, JsonValue],
    ) -> Outcome:
        """Run an approved call, deciding again with the run's current label and policy."""
        call = self._begin(ctx, tool, action)
        run = self._check_run(ctx, call)
        if isinstance(run, Denied):
            return run
        call = _bind(call, run)
        safe_id = approval_id if _APPROVAL_ID.fullmatch(approval_id) else None
        approval = self._approvals.get(approval_id)
        if approval is None:
            return self._deny(
                call,
                GateReason.APPROVAL_NOT_FOUND,
                model_reason=GateReason.NOT_FOUND,
                subject=safe_id,
            )
        # Another run's approval, even the same user's, looks exactly like a missing one.
        if approval.customer_id != run.ctx.customer_id or approval.run_id != run.ctx.run_id:
            return self._deny(
                call,
                GateReason.APPROVAL_RUN_MISMATCH,
                model_reason=GateReason.NOT_FOUND,
                subject=safe_id,
            )

        prepared = self._prepare(run, call, tool, action, args)
        if isinstance(prepared, Denied):
            return prepared
        # The policy may have changed, or the run may have read confidential data, since
        # the approval was granted; a deny now wins over the earlier approval.
        verdict = self._verdict(run, prepared)
        if verdict.effect is Effect.DENY:
            return self._deny_verdict(call, prepared, verdict, approval.id)
        reservation = self._reserve(run, call, prepared.config.cost_usd)
        if isinstance(reservation, CapReached):
            return reservation

        def record_allow(claimed: Approval) -> None:
            details: dict[str, JsonValue] = {"approved_by": claimed.decided_by}
            self._record_allow(call, prepared, verdict, claimed.id, details)

        try:
            # The args the caller is about to run, not the stored ones. The ALLOW is
            # recorded inside the claim, so an approval is consumed only once it is on record.
            self._approvals.claim_for_execution(
                approval.id,
                customer_id=run.ctx.customer_id,
                tool=prepared.config.id,
                action=prepared.action,
                args=args,
                record=_unrecorded_on_failure(record_allow),
                expired=_unrecorded_on_failure(self._record_expired),
            )
        except _Unrecorded:
            self._ledger.release(reservation)
            return Denied(GateReason.AUDIT_UNAVAILABLE.value)
        except ApprovalError as err:
            self._ledger.release(reservation)
            return self._deny(
                call,
                err.reason.value,
                subject=approval.id,
                rule_ids=approval.rule_ids,
                policy_version=verdict.policy_version,
            )
        return self._execute(
            run,
            call,
            prepared,
            reservation,
            verdict,
            subject=approval.id,
            allow_recorded=True,
        )

    # Model spend

    def charge_model(self, ctx: RunContext, estimated_usd: Decimal) -> Reservation | CapReached:
        """Reserve a model call against the run and customer-day caps.

        Raises IdentityError, after recording a DENY, for an invalid run or user.
        """
        call = self._begin(ctx, None, None)
        run = self._check_run(ctx, call)
        if isinstance(run, Denied):
            raise IdentityError(run.reason, "run or user is not valid for model calls")
        call = _bind(call, run)
        reserved = self._reserve(run, call, estimated_usd, details={"source": "model"})
        return reserved

    def settle_model(self, ctx: RunContext, reservation: Reservation, actual_usd: Decimal) -> None:
        if reservation.run_id != ctx.run_id:
            raise ValueError("reservation belongs to another run")
        self._ledger.settle(reservation, actual_usd)

    # Approvals

    def approve(self, approval_id: str, approver_user_id: str, note: str | None = None) -> Approval:
        return self._decide_approval(approval_id, approver_user_id, note, approve=True)

    def reject(self, approval_id: str, approver_user_id: str, note: str | None = None) -> Approval:
        return self._decide_approval(approval_id, approver_user_id, note, approve=False)

    def expire_due(self) -> list[Approval]:
        """Expire every approval past its expiry. If any EXPIRED event cannot be recorded,
        none is expired and the sink's error propagates."""
        return self._approvals.expire_due(record=self._record_expired)

    def _record_expired(self, approval: Approval) -> None:
        call = _Call(
            time.perf_counter_ns(),
            approval.customer_id,
            user_id=approval.requested_by,
            role=self._run_role(approval.run_id),
            run_id=approval.run_id,
            tool=approval.tool,
            action=approval.action,
            policy_version=approval.policy_version,
        )
        self._emit(
            call,
            EventKind.EXPIRED,
            GateReason.EXPIRED,
            subject=approval.id,
            rule_ids=approval.rule_ids,
        )

    def _decide_approval(
        self, approval_id: str, approver_user_id: str, note: str | None, *, approve: bool
    ) -> Approval:
        started = time.perf_counter_ns()
        approver = self._store.get_user(approver_user_id)
        approval = self._approvals.get(approval_id)
        safe_id = approval_id if _APPROVAL_ID.fullmatch(approval_id) else None
        if approver is None:
            customer_id = UNKNOWN_CUSTOMER if approval is None else approval.customer_id
            call = _Call(started, customer_id, user_id=approver_user_id)
            err = ApprovalError(ApprovalErrorReason.APPROVER_NOT_ALLOWED, "unknown approver")
            self._emit(call, EventKind.DENY, err.reason.value, subject=safe_id, decision=True)
            raise err

        call = _Call(started, approver.customer_id, user_id=approver.id, role=approver.role)
        # Only an approval of the approver's own customer lends its run, tool and version to
        # the event, so one customer's audit trail never names another customer's runs.
        if approval is not None and approval.customer_id == approver.customer_id:
            call = replace(
                call,
                run_id=approval.run_id,
                tool=approval.tool,
                action=approval.action,
                policy_version=approval.policy_version,
            )
        # The approver's own customer policy: for an approval of that customer this is the
        # approval's policy, and for any other the queue answers not_found.
        engine = self._engines.get(approver.customer_id)
        kind = EventKind.APPROVED if approve else EventKind.REJECTED
        reason = GateReason.APPROVED if approve else GateReason.REJECTED
        decision_call = call

        def record_decision(decided: Approval) -> None:
            self._emit(
                decision_call,
                kind,
                reason,
                subject=decided.id,
                rule_ids=decided.rule_ids,
                decision=True,
                details={
                    "requested_by": decided.requested_by,
                    "risk_score": decided.risk_score,
                    "has_note": note is not None,
                },
            )

        try:
            if engine is None:
                raise ApprovalError(
                    ApprovalErrorReason.APPROVER_NOT_ALLOWED, "approver's customer has no policy"
                )
            settings = engine.bundle.policy.approvals
            # The decision commits only if its event is recorded; a sink error propagates
            # and leaves the approval as it was.
            decide = self._approvals.approve if approve else self._approvals.reject
            return decide(
                approval_id, approver, settings, note, record_decision, self._record_expired
            )
        except ApprovalError as err:
            self._emit(call, EventKind.DENY, err.reason.value, subject=safe_id, decision=True)
            raise

    # Steps

    def _begin(self, ctx: RunContext, tool: str | None, action: Action | str | None) -> _Call:
        """A call described by what the caller claims, until the run row confirms it."""
        claimed_customer = getattr(ctx, "customer_id", None)
        claimed_user = getattr(ctx, "user_id", None)
        claimed_run = getattr(ctx, "run_id", None)
        return _Call(
            time.perf_counter_ns(),
            claimed_customer if isinstance(claimed_customer, str) else UNKNOWN_CUSTOMER,
            user_id=claimed_user if isinstance(claimed_user, str) else None,
            role=_safe_role(getattr(ctx, "role", None)),
            run_id=claimed_run if isinstance(claimed_run, str) else None,
            tool=tool if tool is not None and _TOOL_ID.fullmatch(tool) else None,
            action=_safe_action(action),
        )

    def _check_run(self, ctx: RunContext, call: _Call) -> _Run | Denied:
        if not isinstance(ctx, RunContext) or not isinstance(ctx.run_id, str):
            return self._deny(call, GateReason.INVALID_RUN)
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (ctx.run_id,)
            ).fetchone()
        if row is None:
            return self._deny(call, GateReason.INVALID_RUN, details={"run_known": False})
        engine = self._engines.get(row["customer_id"])
        # From here the event names the real run, not whatever the caller claimed.
        call = replace(
            call,
            customer_id=row["customer_id"],
            user_id=row["user_id"],
            role=Role(row["role"]),
            policy_version=None if engine is None else engine.version,
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
            return self._deny(call, GateReason.INVALID_RUN, details={"mismatched": mismatched})

        user = self._store.get_user(row["user_id"])
        if user is None:
            return self._deny(call, GateReason.UNKNOWN_USER)
        if user.customer_id != row["customer_id"]:
            return self._deny(call, GateReason.WRONG_CUSTOMER)
        if user.status is not UserStatus.ACTIVE:
            return self._deny(call, GateReason.SUSPENDED_USER)
        if user.role.value != row["role"]:
            return self._deny(call, GateReason.ROLE_CHANGED)
        customer = self._store.get_customer(row["customer_id"])
        if engine is None or customer is None:
            return self._deny(call, GateReason.NO_POLICY)
        trusted = RunContext(
            run_id=row["run_id"],
            customer_id=row["customer_id"],
            user_id=row["user_id"],
            role=Role(row["role"]),
            started_at=datetime.fromisoformat(row["started_at"]),
        )
        return _Run(ctx=trusted, user=user, customer=customer, engine=engine)

    def _prepare(
        self,
        run: _Run,
        call: _Call,
        tool: str,
        action: Action | str,
        args: Mapping[str, JsonValue],
    ) -> _Prepared | Denied:
        adapter = self._tools.get(tool)
        config = run.engine.bundle.tools.get(tool)
        if adapter is None or config is None:
            return self._deny(call, GateReason.UNKNOWN_TOOL)
        checked_action = _safe_action(action)
        if checked_action is None:
            return self._deny(call, GateReason.ACTION_NOT_SUPPORTED)
        validated = _validate_args(adapter, config.id, checked_action, args)
        if isinstance(validated, dict):
            return self._deny(call, GateReason.INVALID_ARGS, details=validated)
        if tool == SEARCH_TOOL and checked_action is not Action.READ:
            # Search never consults the engine per call, so its built-in check lives here.
            return self._deny(call, Reason.ACTION_NOT_SUPPORTED)

        facts = adapter.facts(validated)
        document: DocumentFacts | None = None
        if facts.document_id is not None:
            stored = self._store.get_document(facts.document_id)
            if stored is None:
                return self._deny(
                    call,
                    GateReason.DOCUMENT_NOT_FOUND,
                    model_reason=GateReason.NOT_FOUND,
                    subject=facts.document_id,
                )
            document = DocumentFacts.from_document(stored)
            if stored.customer_id != run.ctx.customer_id:
                # Decided here, before spend, so a missing id and a foreign one take the
                # same path and cannot be told apart by a cap or by timing of the steps.
                decision = run.engine.decide(
                    _request(run, config.id, checked_action, document, facts, None)
                )
                return self._deny(
                    call,
                    decision.reason.value,
                    model_reason=GateReason.NOT_FOUND,
                    subject=facts.document_id,
                    rule_ids=decision.rule_ids,
                    policy_version=decision.policy_version,
                )
        elif facts.write_label is not None:
            document = DocumentFacts(
                customer_id=run.ctx.customer_id, label=facts.write_label, tags=NOTE_TAGS
            )
        return _Prepared(
            adapter=adapter,
            config=config,
            action=checked_action,
            args=validated,
            facts=facts,
            document=document,
            document_id=facts.document_id,
        )

    def _reserve(
        self,
        run: _Run,
        call: _Call,
        amount: Decimal,
        details: dict[str, JsonValue] | None = None,
    ) -> Reservation | CapReached:
        reserved = self._ledger.reserve(
            customer=run.customer,
            run_id=run.ctx.run_id,
            amount_usd=amount,
            limits=run.policy.spend,
        )
        if isinstance(reserved, Reservation):
            return reserved
        self._emit(
            call,
            EventKind.CAP_HIT,
            reserved.cap.value,
            subject=reserved.cap.value,
            decision=True,
            details=_cap_details(reserved) | (details or {}),
        )
        return CapReached(cap=reserved.cap)

    def _verdict(self, run: _Run, prepared: _Prepared) -> _Verdict:
        max_label = self.run_max_label(run.ctx.run_id)
        decision = run.engine.decide(
            _request(
                run,
                prepared.config.id,
                prepared.action,
                prepared.document,
                prepared.facts,
                max_label,
            )
        )
        verdict = _Verdict(
            decision.effect, decision.reason.value, decision.rule_ids, decision.policy_version
        )
        written = prepared.facts.write_label
        if verdict.effect is not Effect.DENY and written is not None and max_label is not None:
            written_rank = run.policy.label_rank(written)
            max_rank = run.policy.label_rank(max_label)
            # Writing what the run has read into a less sensitive document would launder it.
            if written_rank is None or max_rank is None or written_rank < max_rank:
                return _Verdict(Effect.DENY, GateReason.WRITE_DOWN, (), decision.policy_version)
        return verdict

    def _deny_verdict(
        self, call: _Call, prepared: _Prepared, verdict: _Verdict, subject: str | None
    ) -> Denied:
        # Any denial of a call that names a document reads as not_found to the model, the
        # same as a missing id, so ids (which reveal titles) cannot be probed even within
        # the tenant. The audit keeps the real reason.
        model_reason = (
            GateReason.NOT_FOUND.value
            if prepared.document_id is not None
            else _model_reason(verdict.reason)
        )
        return self._deny(
            call,
            verdict.reason,
            model_reason=model_reason,
            subject=subject,
            rule_ids=verdict.rule_ids,
            policy_version=verdict.policy_version,
        )

    def _escalation(self, run: _Run, prepared: _Prepared) -> GateReason | None:
        if prepared.config.customer_facing and self._reaches_outside(run, prepared):
            return GateReason.ESCALATED_CUSTOMER_FACING
        if self._risk(run, prepared) >= run.policy.approvals.risk_threshold:
            return GateReason.ESCALATED_RISK
        return None

    def _reaches_outside(self, run: _Run, prepared: _Prepared) -> bool:
        """Whether a customer-facing call reaches someone outside the company.

        A call with no recipient facts (a refund, say) always reaches a customer. One with
        recipients does when any of them is outside the customer's domains, so an internal
        email the policy allows runs without approval.
        """
        recipients = prepared.facts.recipients
        if not recipients:
            return True
        return self._any_external(run, recipients)

    def _any_external(self, run: _Run, recipients: tuple[str, ...]) -> bool:
        domains = frozenset(domain.lower() for domain in run.customer.domains)
        return any(_is_external(raw, domains) for raw in recipients)

    def _risk(self, run: _Run, prepared: _Prepared) -> int:
        risk = prepared.config.base_risk
        if self._any_external(run, prepared.facts.recipients):
            risk += EXTERNAL_RECIPIENT_RISK
        amount = prepared.facts.amount_usd
        if amount is not None and amount.is_finite() and amount > 0:
            risk += min(MAX_SPEND_RISK, int(amount // 100) * SPEND_RISK_PER_100_USD)
        return risk

    def _execute(
        self,
        run: _Run,
        call: _Call,
        prepared: _Prepared,
        reservation: Reservation,
        verdict: _Verdict,
        *,
        subject: str | None,
        allow_recorded: bool = False,
    ) -> Outcome:
        """Record the allow, unless the caller already has, then run the tool."""
        if not allow_recorded:
            try:
                # The allow is on record before anything happens, or nothing happens.
                self._record_allow(call, prepared, verdict, subject, None)
            except Exception:
                self._ledger.release(reservation)
                return Denied(GateReason.AUDIT_UNAVAILABLE.value)

        env = ToolEnv(store=self._store, index=self._index, clock=self._clock)
        result = self._run_adapter(call, prepared, env, run, reservation, subject)
        if isinstance(result, Denied):
            return result
        # The label the decision saw, as well as whatever the tool says it returned.
        labels = (
            []
            if prepared.document_id is None or prepared.document is None
            else [prepared.document.label]
        )
        return self._finish(
            run, call, prepared, reservation, result, [*labels, *result.returned_labels], subject
        )

    def _record_allow(
        self,
        call: _Call,
        prepared: _Prepared,
        verdict: _Verdict,
        subject: str | None,
        details: dict[str, JsonValue] | None,
    ) -> None:
        self._emit(
            call,
            EventKind.ALLOW,
            verdict.reason,
            subject=subject,
            rule_ids=verdict.rule_ids,
            policy_version=verdict.policy_version,
            decision=True,
            details=details,
        )

    def _search(self, run: _Run, call: _Call, prepared: _Prepared) -> Outcome:
        reservation = self._reserve(run, call, prepared.config.cost_usd)
        if isinstance(reservation, CapReached):
            return reservation
        max_label = self.run_max_label(run.ctx.run_id)
        searches: list[tuple[RetrievalResult, int]] = []

        def search(query: str, limit: int) -> RetrievalResult:
            # The policy check per document is the retrieval filter itself.
            found = self._index.search(
                run.engine, run.ctx, query, run_max_label=max_label, tool=SEARCH_TOOL, limit=limit
            )
            searches.append((found, call.latency_us()))
            return found

        env = ToolEnv(store=self._store, index=self._index, clock=self._clock, search=search)
        try:
            result = prepared.adapter.run(env, run.ctx, prepared.args)
        except RetrievalInvariantError:
            self._ledger.release(reservation)
            self._emit(
                call, EventKind.INVARIANT_VIOLATION, GateReason.RETRIEVAL_INVARIANT, decision=True
            )
            return Denied(GateReason.TOOL_ERROR.value)
        except Exception as exc:
            return self._tool_failed(call, reservation, None, exc)
        if len(searches) != 1:
            return self._tool_failed(call, reservation, None, None)

        found, latency = searches[0]
        trace = found.trace
        try:
            # Nothing reaches the model until the retrieval decision is on record.
            self._emit(
                call,
                EventKind.RETRIEVAL,
                GateReason.RETRIEVAL_FILTER,
                rule_ids=trace.permitting_rules,
                policy_version=trace.policy_version,
                latency_us=latency,
                details={
                    "query_terms": len(trace.query_terms),
                    "candidates_in_tenant": trace.candidates_in_tenant,
                    "permitted": trace.permitted,
                    "filtered_by_reason": dict(trace.filtered_by_reason),
                    "returned_ids": list(trace.returned_ids),
                    "run_max_label": max_label,
                },
            )
        except Exception:
            self._ledger.release(reservation)
            return Denied(GateReason.AUDIT_UNAVAILABLE.value)
        labels = [doc.label for doc in found.documents]
        return self._finish(
            run, call, prepared, reservation, result, [*labels, *result.returned_labels], None
        )

    def _run_adapter(
        self,
        call: _Call,
        prepared: _Prepared,
        env: ToolEnv,
        run: _Run,
        reservation: Reservation,
        subject: str | None,
    ) -> ToolResult | Denied:
        try:
            result = prepared.adapter.run(env, run.ctx, prepared.args)
        except Exception as exc:
            return self._tool_failed(call, reservation, subject, exc)
        if not result.cost_usd.is_finite() or result.cost_usd < 0:
            return self._tool_failed(call, reservation, subject, None)
        return result

    def _tool_failed(
        self,
        call: _Call,
        reservation: Reservation,
        subject: str | None,
        exc: Exception | None,
    ) -> Denied:
        self._ledger.release(reservation)
        # The exception type only: messages can quote arguments or document text.
        error = "invalid_result" if exc is None else type(exc).__name__
        return self._deny(call, GateReason.TOOL_ERROR, subject=subject, details={"error": error})

    def _finish(
        self,
        run: _Run,
        call: _Call,
        prepared: _Prepared,
        reservation: Reservation,
        result: ToolResult,
        labels: Iterable[str],
        subject: str | None,
    ) -> Executed:
        cost = prepared.config.cost_usd + result.cost_usd
        self._ledger.settle(reservation, cost)
        max_label = self._raise_label(run, labels)
        self._emit(
            call,
            EventKind.EXECUTED,
            GateReason.EXECUTED,
            subject=subject,
            details={"cost_usd": str(cost), "run_max_label": max_label},
        )
        return Executed(tool=prepared.config.id, content=result.content, cost_usd=cost)

    def _raise_label(self, run: _Run, labels: Iterable[str]) -> str | None:
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

    def _run_role(self, run_id: str) -> Role | None:
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT role FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else Role(row["role"])

    # Audit

    def _identity_denied(self, call: _Call, reason: GateReason, message: str) -> NoReturn:
        self._emit(call, EventKind.DENY, reason, decision=True)
        raise IdentityError(reason.value, message)

    def _deny(
        self,
        call: _Call,
        reason: str,
        *,
        model_reason: str | None = None,
        subject: str | None = None,
        rule_ids: tuple[str, ...] = (),
        policy_version: str | None = None,
        details: dict[str, JsonValue] | None = None,
    ) -> Denied:
        self._emit(
            call,
            EventKind.DENY,
            reason,
            subject=subject,
            rule_ids=rule_ids,
            policy_version=policy_version,
            decision=True,
            details=details,
        )
        return Denied(str(model_reason or reason))

    def _emit(
        self,
        call: _Call,
        kind: EventKind,
        reason: str,
        *,
        subject: str | None = None,
        rule_ids: tuple[str, ...] = (),
        policy_version: str | None = None,
        decision: bool = False,
        latency_us: int | None = None,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        if latency_us is None and decision:
            latency_us = call.latency_us()
        self._sink.record(
            AuditEvent(
                timestamp=self._clock.now(),
                kind=kind,
                customer_id=call.customer_id,
                user_id=call.user_id,
                role=call.role,
                run_id=call.run_id,
                tool=call.tool,
                action=call.action,
                subject=subject,
                reason=str(reason),
                rule_ids=rule_ids,
                policy_version=policy_version or call.policy_version,
                latency_us=latency_us,
                details=details or {},
            )
        )


class _Unrecorded(Exception):
    """An audit event inside an approval transaction could not be recorded."""


def _unrecorded_on_failure(record: Callable[[Approval], None]) -> Callable[[Approval], None]:
    """The hook, with any failure to record turned into _Unrecorded, so the caller can tell
    a sink error from the queue's own errors."""

    def wrapped(approval: Approval) -> None:
        try:
            record(approval)
        except Exception as exc:
            raise _Unrecorded from exc

    return wrapped


def _bind(call: _Call, run: _Run) -> _Call:
    return replace(
        call,
        customer_id=run.ctx.customer_id,
        user_id=run.ctx.user_id,
        role=run.ctx.role,
        run_id=run.ctx.run_id,
        policy_version=run.engine.version,
    )


def _request(
    run: _Run,
    tool: str,
    action: Action,
    document: DocumentFacts | None,
    facts: ToolFacts,
    run_max_label: str | None,
) -> Request:
    return Request(
        customer_id=run.ctx.customer_id,
        user_id=run.ctx.user_id,
        role=run.ctx.role,
        tool=tool,
        action=action,
        document=document,
        recipients=facts.recipients,
        amount_usd=facts.amount_usd,
        run_max_label=run_max_label,
    )


def _validate_args(
    adapter: ToolAdapter, tool: str, action: Action, args: Mapping[str, JsonValue]
) -> BaseModel | dict[str, JsonValue]:
    """The validated args, or audit details saying why they were refused (never values)."""
    if not isinstance(args, Mapping):
        return {"error_types": ["not_an_object"]}
    try:
        # args_hash is what an approval binds to, so args it cannot hash are refused here.
        args_hash(tool, action, args)
        raw = json.dumps(dict(args), allow_nan=False)
    except (TypeError, ValueError):
        return {"error_types": ["not_json"]}
    model = adapter.args_model
    try:
        return model.model_validate_json(raw, strict=True)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False, include_context=False)
        known = set(model.model_fields)
        # Field names the model chose (an unknown key) are not echoed, only known ones.
        fields = sorted({str(e["loc"][0]) for e in errors if e["loc"] and e["loc"][0] in known})
        types = sorted({e["type"] for e in errors})
        return {"error_types": list(types), "fields": list(fields)}


def _model_reason(reason: str) -> str:
    if reason == Reason.DOCUMENT_CUSTOMER_MISMATCH:
        return GateReason.NOT_FOUND.value
    return reason


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


def _is_external(raw: str, domains: frozenset[str]) -> bool:
    match = _ADDRESS.fullmatch(raw.strip().lower())
    # Anything that is not one plain address counts as leaving the company.
    return match is None or match.group(1) not in domains


def _safe_role(value: object) -> Role | None:
    try:
        return Role(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _safe_action(value: object) -> Action | None:
    try:
        return Action(value) if isinstance(value, str) else None
    except ValueError:
        return None
