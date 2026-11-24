"""The Gate: the one object a host holds to decide, record and run what agent runs do.

It composes the parts of this package and is their only public interface:

- `runs.RunRegistry`: starts runs and checks every call's run against the store.
- `tool_calls.ToolCalls`: the tool-call pipeline, from argument validation to the tool.
- `model_calls.ModelAdmission`: model requests through the proxy, and what they cost.
- `decisions.ApprovalDecisions`: approving, rejecting and expiring queued approvals.
- `operator_log.OperatorLog`: policy loads, operator actions and the stale spend sweep.
- `recorder.AuditRecorder`: the audit events all of the above record.
"""

from collections.abc import Callable, Mapping
from datetime import timedelta
from decimal import Decimal

from pydantic import JsonValue

from agentgate.approvals import Approval, ApprovalQueue
from agentgate.audit.base import AuditSink, OperatorAction
from agentgate.clock import Clock
from agentgate.context import RunAssurance, RunContext
from agentgate.domain import Action, User
from agentgate.gate.decisions import ApprovalDecisions
from agentgate.gate.model_calls import ModelAdmission
from agentgate.gate.operator_log import OperatorLog
from agentgate.gate.outcomes import CapReached, Denied, GateReason, Outcome
from agentgate.gate.recorder import AuditRecorder
from agentgate.gate.runs import RunRegistry
from agentgate.gate.tool_calls import ToolCalls
from agentgate.model.admission import Caps, ModelCall, ModelExchange
from agentgate.model.pricing import RateTable
from agentgate.model.tokens import TokenRecord
from agentgate.policy.engine import PolicyEngine
from agentgate.retrieval import DocumentIndex
from agentgate.spend import Reservation, SpendLedger
from agentgate.store import Store
from agentgate.tools.base import ToolAdapter
from agentgate.tools.builtin import default_tools


class Gate:
    """Decides, records and runs the tool and model calls of agent runs for every customer."""

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
        self._engines = dict(engines)
        self._tools = adapters
        self._ledger = SpendLedger(store, clock)
        self._approvals = ApprovalQueue(store, clock)
        known_tools = frozenset(adapters) | frozenset(
            tool_id for engine in self._engines.values() for tool_id in engine.bundle.tools
        )
        self._recorder = AuditRecorder(store=store, sink=sink, clock=clock, known_tools=known_tools)
        self._runs = RunRegistry(
            store=store,
            engines=self._engines,
            ledger=self._ledger,
            recorder=self._recorder,
            clock=clock,
        )
        self._decisions = ApprovalDecisions(
            store=store,
            engines=self._engines,
            approvals=self._approvals,
            runs=self._runs,
            recorder=self._recorder,
        )
        self._tool_calls = ToolCalls(
            store=store,
            index=index,
            clock=clock,
            tools=self._tools,
            approvals=self._approvals,
            ledger=self._ledger,
            runs=self._runs,
            recorder=self._recorder,
            decisions=self._decisions,
        )
        self._models = ModelAdmission(
            store=store,
            engines=self._engines,
            ledger=self._ledger,
            runs=self._runs,
            recorder=self._recorder,
        )
        self._operator = OperatorLog(
            store=store, engines=self._engines, ledger=self._ledger, recorder=self._recorder
        )

    @property
    def approvals(self) -> ApprovalQueue:
        return self._approvals

    @property
    def ledger(self) -> SpendLedger:
        return self._ledger

    def audit_reference(self, value: str) -> str:
        """The reference the durable audit log stores for this value in place of it.

        Stable per store database, so an operator can ask whether a value was ever probed.
        """
        return self._recorder.reference(value)

    # Runs

    def start_run(
        self,
        customer_id: str,
        user_id: str,
        *,
        record: Callable[[RunContext], None] | None = None,
        assurance: RunAssurance | None = None,
    ) -> RunContext:
        """Start a run for a user of a customer, or raise IdentityError (recorded).

        `record`, if given, runs inside the transaction that stores the run, before the run's
        start is recorded, so what it writes (a multi-run proxy's caps and ownership of the
        run) exists exactly when the run does: if it raises, there is no run.

        `assurance`, for a user agentgate verified from a signed assertion, is stored on the
        run (`verified:<issuer>`) and recorded on its start, or on its refusal, with the
        issuer and the reference to the assertion's jti. The checks are the same either way.
        """
        return self._runs.start(customer_id, user_id, record=record, assurance=assurance)

    def refuse_run_start(
        self,
        customer_id: str,
        reason: str,
        *,
        user_id: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Record a run start refused before the gate resolved a user: an identity assertion
        that failed verification (agentgate.identity reason codes), or a user named where a
        verified assertion is required. `details` holds identifiers and references only,
        never the assertion. Raises if not recorded."""
        self._runs.refuse_start(
            customer_id, reason, user_id=user_id, details=None if details is None else dict(details)
        )

    def run_assurance(self, run_id: str) -> str | None:
        """How the run's user was identified: `verified:<issuer>`, or None when the host
        named the user."""
        return self._runs.assurance(run_id)

    def run_max_label(self, run_id: str) -> str | None:
        """The most sensitive label returned to the model in this run, or None."""
        return self._runs.max_label(run_id)

    # Tool calls

    def call_tool(
        self, ctx: RunContext, tool: str, action: Action, args: Mapping[str, JsonValue]
    ) -> Outcome:
        """Decide a tool call for the run and, if allowed, record it and run the tool."""
        return self._tool_calls.call(ctx, tool, action, args)

    def execute_approved(
        self,
        ctx: RunContext,
        approval_id: str,
        tool: str,
        action: Action,
        args: Mapping[str, JsonValue],
    ) -> Outcome:
        """Run an approved call, deciding again with the run's current label and policy."""
        return self._tool_calls.execute_approved(ctx, approval_id, tool, action, args)

    def refuse(
        self,
        ctx: RunContext,
        *,
        tool: str | None = None,
        reason: str = GateReason.INVALID_ARGS.value,
        details: Mapping[str, JsonValue] | None = None,
    ) -> Denied:
        """Refuse, on record, a request a front end could not turn into a gate call at all.

        For a front end such as the MCP gateway, whose own request wrapper can be malformed
        (a missing field, a wrong type, an extra key): the refusal is still the gate's, so
        it is audited like any other. The run and user are checked as for call_tool, so a
        forged context or a suspended user is denied with that reason instead; otherwise
        one DENY with the bound run's identity and `reason`, which must be a gate reason
        code, is recorded. `tool` is recorded only in the tool-id shape. `details` may hold
        only `error_types` and `fields`, as lists of strings: reason codes and field names
        the front end defines, never a caller's key names or values. Nothing is reserved.
        """
        return self._tool_calls.refuse(ctx, tool=tool, reason=reason, details=details)

    # Model calls

    def admit_model(
        self,
        ctx: RunContext,
        call: ModelCall,
        *,
        rates: RateTable | None = None,
        caps: Caps | None = None,
    ) -> Reservation | Denied | CapReached:
        """Decide a model request for the run and, if allowed, reserve its worst-case cost.

        In order: the run and its user are checked as for a tool call; a body that could not
        be read strictly, or that has a top-level field the provider's list does not name, is
        denied; the policy's models section decides (deny by default, exact model, role,
        output limit, one completion, body size, server-side features); the reservation is
        priced from the rate table (the policy's fixed `reserve` is a floor on that, and the
        price itself for a model the table lacks); the run's caps, as
        tightened by `caps`, and the customer's day cap are checked; and the allow is
        recorded before the reservation is handed back. A model the table does not price is
        denied unless the policy sets a fixed reserve. If the allow cannot be recorded the
        reservation is released and the call is denied.

        The caller ends the reservation with `finish_model` once the exchange is over.
        """
        return self._models.admit(ctx, call, rates=rates, caps=caps)

    def finish_model(
        self, ctx: RunContext, reservation: Reservation, exchange: ModelExchange
    ) -> Decimal:
        """End a model call's reservation by what happened to the call, and return the
        amount charged (0 for a release).

        A client that drops the connection, a stream that breaks and a reply that cannot be
        priced do not make a call free: the provider may have billed all of it. So:

        - the request never reached the provider: released;
        - the provider answered 4xx or 5xx and produced no output: released;
        - otherwise: settled at the priced cost when the reply is complete and the cost is a
          usable one (see ModelExchange), else at the full reservation.
        """
        return self._models.finish(ctx, reservation, exchange)

    def refuse_token(
        self, run: RunContext, token_reason: str, *, exchange_id: str | None = None
    ) -> Denied:
        """Record, on the run a refused token belongs to, that a model request presented it.

        For a token that named a run (revoked, expired, for another run, its user suspended),
        so the refusal is on that run's record. The run and user are checked as for any call;
        a suspended user's refusal is recorded as that. `token_reason` is a model.tokens
        TokenReason value, recorded in details as `refused_because` (a key with "token" in
        its name is redacted by the log).
        """
        return self._models.refuse_token(run, token_reason, exchange_id=exchange_id)

    def charge_model(self, ctx: RunContext, estimated_usd: Decimal) -> Reservation | CapReached:
        """Reserve a call of the in-process reference agent (agent.loop) against the run and
        customer-day caps, at the amount its host estimates.

        This is not the model gateway's path: the host trusts its own model adapter for the
        estimate and for what `settle_model` is later given. A gateway call goes through
        `admit_model` and `finish_model`, which price it from the request and the exchange.

        Raises IdentityError, after recording a DENY, for an invalid run or user.
        """
        return self._models.charge(ctx, estimated_usd)

    def settle_model(self, ctx: RunContext, reservation: Reservation, actual_usd: Decimal) -> None:
        """Settle a `charge_model` reservation at the cost its host reports. Refuses one made
        by `admit_model`: those are ended only by `finish_model`."""
        self._models.settle(ctx, reservation, actual_usd)

    # Approvals

    def approve(self, approval_id: str, approver_user_id: str, note: str | None = None) -> Approval:
        """Approve a queued call as `approver_user_id`, on record, or raise ApprovalError."""
        return self._decisions.approve(approval_id, approver_user_id, note)

    def reject(self, approval_id: str, approver_user_id: str, note: str | None = None) -> Approval:
        """Reject a queued call as `approver_user_id`, on record, or raise ApprovalError."""
        return self._decisions.reject(approval_id, approver_user_id, note)

    def expire_due(self) -> list[Approval]:
        """Expire every approval past its expiry. If any EXPIRED event cannot be recorded,
        none is expired and the sink's error propagates."""
        return self._decisions.expire_due()

    # Policies and operator actions

    def record_policy_loaded(
        self, customer_id: str, *, bundle_sha256: str, details: Mapping[str, JsonValue]
    ) -> None:
        """Record that this gate decides for a customer with its current policy: one
        POLICY_LOADED event carrying the version and the sha256 of the canonical bundle
        snapshot (store.PolicySnapshot) it was computed over. Raises if not recorded."""
        self._operator.record_policy_loaded(
            customer_id, bundle_sha256=bundle_sha256, details=details
        )

    def record_operator_action(
        self,
        action: OperatorAction,
        *,
        customer_id: str,
        actor: User | None,
        target: str | None = None,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Record one OPERATOR_ACTION event in a customer's log. `actor` is the acting user,
        named only when they belong to that customer (None for the system, or an operator
        of another customer, recorded as such); `target` is the id acted on, which must be
        one the store knows. Raises if not recorded."""
        self._operator.record_operator_action(
            action, customer_id=customer_id, actor=actor, target=target, details=details
        )

    def record_token_event(
        self,
        action: OperatorAction,
        token: TokenRecord,
        *,
        customer_id: str,
        actor: User | None,
    ) -> None:
        """Record that a run token was issued (TOKEN_ISSUED) or revoked (TOKEN_REVOKED) as an
        operator action. A revocation also records why: `operator`, or `run_ended` when the
        run's gateway ended it, which has no actor. Raises if not recorded."""
        self._operator.record_token_event(action, token, customer_id=customer_id, actor=actor)

    def release_stale(
        self, older_than: timedelta, *, actor: User | None = None, record: bool = True
    ) -> int:
        """Reclaim spend reservations left open longer than older_than
        (SpendLedger.release_stale) and record what was released: one OPERATOR_ACTION per
        customer with reservations released, with their count, total and run ids. With an
        actor (an operator command), the actor's own customer also gets an event when
        nothing of theirs was released. Returns the number released. With record False
        the sweep records nothing, as SpendLedger.release_stale alone.

        The sweep commits before its events are recorded; if the sink fails, the error
        propagates with the reservations already released."""
        return self._operator.release_stale(older_than, actor=actor, record=record)
