"""Tool calls: every call an agent run proposes is checked, decided, recorded, then run.

A call goes through the same steps every time: the run check, strict argument validation,
the facts of the call (a document is loaded and checked for tenant first), a spend
reservation, the policy decision with the run's max label, escalation to approval of allows
that are risky or that reach someone outside the company through a customer-facing tool, and
only for an allow, an audit record and then the tool. If the allow cannot be recorded the
tool does not run.

So that an incident can be reconstructed from the log alone, a decision on a call whose
arguments validated also carries values derived from them, never the text: `args_hash`
(approvals.args_hash of the tool, the action and the validated args, as the tool receives
them), for a send the number of recipients and how many are inside and outside the
customer's domains, and for a spend its amount. EXECUTED adds `effect_ref`, the id of the
row the tool wrote (ToolResult.effect_ref), so a store row can be matched to its event.
Every event about an approval carries the approval row's own `approval_args_hash`, so
`agentgate audit check-store` can tell an approval row rewritten after the fact. A search
carries no args hash (its query is free text whose hash could be tested against guesses),
nor does a call naming a document the store does not hold.

An approved call is decided again when it runs (`execute_approved`), with the run's current
label and policy, so a deny now wins over the earlier approval. Mail to an internal
recipient who may not read the run's max label, or who is no known user, goes for approval.
A call sent for approval gives its spend back but counts as one of the run's calls, so a
run cannot queue more approvals than its call cap.

Model-facing reasons are the engine's reason codes, with one exception: every denial of a
call that names a document (a missing id, another customer's document, or one of the run's
own customer that the run may not read) reads as `not_found`, so document ids, which
reveal titles, cannot be probed within or across tenants. Such calls are decided before
spend, so a cap cannot tell them apart either. The audit event keeps the real reason.
"""

import contextlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from decimal import Decimal

from pydantic import BaseModel, JsonValue, ValidationError

from agentgate.approvals import (
    Approval,
    ApprovalError,
    ApprovalQueue,
    args_hash,
)
from agentgate.audit.base import (
    EventKind,
)
from agentgate.clock import Clock
from agentgate.context import RunContext
from agentgate.domain import Action, Tool, UserStatus
from agentgate.gate._values import (
    as_text,
    get_approval,
    is_external,
    is_lookupable,
    safe_action,
    safe_approval_id,
)
from agentgate.gate.decisions import ApprovalDecisions
from agentgate.gate.outcomes import (
    CapReached,
    Denied,
    Executed,
    GateReason,
    Outcome,
    PendingApproval,
)
from agentgate.gate.recorder import AuditRecorder, Call
from agentgate.gate.runs import CheckedRun, RunRegistry, bind
from agentgate.policy.addresses import parse_address
from agentgate.policy.engine import DocumentFacts, Reason, Request
from agentgate.policy.schema import Effect
from agentgate.retrieval import DocumentIndex, RetrievalInvariantError, RetrievalResult
from agentgate.spend import Reservation, SpendLedger
from agentgate.store import Store
from agentgate.tools.base import ToolAdapter, ToolEnv, ToolFacts, ToolResult

SEARCH_TOOL = "docs.search"
READ_TOOL = "docs.read"
# The tags of a note being written, as the engine sees it when deciding the write.
NOTE_TAGS = frozenset({"note"})

# Risk added to a tool's base risk: any recipient outside the customer's domains, and 5 per
# full 100 USD of spend up to 40.
EXTERNAL_RECIPIENT_RISK = 25
SPEND_RISK_PER_100_USD = 5
MAX_SPEND_RISK = 40


@dataclass(frozen=True)
class Prepared:
    adapter: ToolAdapter
    config: Tool
    action: Action
    args: BaseModel
    facts: ToolFacts
    document: DocumentFacts | None
    # The id of a document the call reads, for event subjects.
    document_id: str | None
    # What the call's decision events record about its args: hash, counts, amount.
    values: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    effect: Effect
    reason: str
    rule_ids: tuple[str, ...]
    policy_version: str


class ToolCalls:
    """Decides, records and runs the tool calls of agent runs."""

    def __init__(
        self,
        *,
        store: Store,
        index: DocumentIndex,
        clock: Clock,
        tools: Mapping[str, ToolAdapter],
        approvals: ApprovalQueue,
        ledger: SpendLedger,
        runs: RunRegistry,
        recorder: AuditRecorder,
        decisions: ApprovalDecisions,
    ) -> None:
        self._store = store
        self._index = index
        self._clock = clock
        self._tools = tools
        self._approvals = approvals
        self._ledger = ledger
        self._runs = runs
        self._recorder = recorder
        self._decisions = decisions

    def call(
        self, ctx: RunContext, tool: str, action: Action, args: Mapping[str, JsonValue]
    ) -> Outcome:
        call = self._runs.begin(ctx, tool, action)
        run = self._runs.check(ctx, call)
        if isinstance(run, Denied):
            return run
        call = bind(call, run)
        prepared = self._prepare(run, call, tool, action, args)
        if isinstance(prepared, Denied):
            return prepared
        if prepared.config.id == SEARCH_TOOL:
            return self._search(run, call, prepared)

        subject = prepared.document_id
        verdict: Verdict | None = None
        if subject is not None:
            # A call naming a document is decided before spend, so a denied id and a
            # missing one take the same path: neither reaches the ledger, so a cap cannot
            # tell them apart either.
            verdict = self._verdict(run, prepared)
            if verdict.effect is Effect.DENY:
                return self._deny_verdict(call, prepared, verdict, subject)
        reservation = self._runs.reserve(run, call, prepared.config.cost_usd)
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
            risk = self._risk(run, prepared)

            def record_request(created: Approval) -> None:
                self._recorder.emit(
                    call,
                    EventKind.APPROVAL_REQUIRED,
                    verdict.reason,
                    subject=created.id,
                    rule_ids=verdict.rule_ids,
                    policy_version=verdict.policy_version,
                    decision=True,
                    details={
                        "risk_score": risk,
                        "document_id": subject,
                        **prepared.values,
                        "approval_args_hash": created.args_hash,
                    },
                )

            # A request the audit log cannot record raises here and leaves no approval.
            try:
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
                    ttl=timedelta(minutes=run.policy.approval_settings().ttl_minutes),
                    record=record_request,
                )
            except BaseException:
                self._ledger.release(reservation)
                raise
            # The request is one of the run's calls, at no cost, so a run cannot queue more
            # approvals than max_calls_per_run allows; the spend goes back to the caps.
            self._ledger.settle(reservation, Decimal(0))
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
        call = self._runs.begin(ctx, tool, action)
        run = self._runs.check(ctx, call)
        if isinstance(run, Denied):
            return run
        call = bind(call, run)
        safe_id = safe_approval_id(approval_id)
        approval = get_approval(self._approvals, approval_id)
        if approval is None:
            return self._recorder.deny(
                call,
                GateReason.APPROVAL_NOT_FOUND,
                model_reason=GateReason.NOT_FOUND,
                subject=safe_id,
                unknown_subject=as_text(approval_id),
            )
        # Another run's approval, even the same user's, looks exactly like a missing one.
        if approval.customer_id != run.ctx.customer_id or approval.run_id != run.ctx.run_id:
            return self._recorder.deny(
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
        stored_hash: dict[str, JsonValue] = {"approval_args_hash": approval.args_hash}
        if verdict.effect is Effect.DENY:
            return self._deny_verdict(call, prepared, verdict, approval.id, extra=stored_hash)
        reservation = self._runs.reserve(run, call, prepared.config.cost_usd)
        if isinstance(reservation, CapReached):
            return reservation

        def record_allow(claimed: Approval) -> None:
            details: dict[str, JsonValue] = {
                "approved_by": claimed.decided_by,
                "approval_args_hash": claimed.args_hash,
            }
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
                expired=_unrecorded_on_failure(self._decisions.record_expired),
            )
        except _Unrecorded:
            self._ledger.release(reservation)
            return Denied(GateReason.AUDIT_UNAVAILABLE.value)
        except ApprovalError as err:
            self._ledger.release(reservation)
            return self._recorder.deny(
                call,
                err.reason.value,
                subject=approval.id,
                rule_ids=approval.rule_ids,
                policy_version=verdict.policy_version,
                details={**prepared.values, **stored_hash},
            )
        return self._execute(
            run,
            call,
            prepared,
            reservation,
            verdict,
            subject=approval.id,
            allow_recorded=True,
            extra=stored_hash,
        )

    def refuse(
        self,
        ctx: RunContext,
        *,
        tool: str | None = None,
        reason: str = GateReason.INVALID_ARGS.value,
        details: Mapping[str, JsonValue] | None = None,
    ) -> Denied:
        code = GateReason(reason).value
        checked = _refusal_details(details)
        call = self._runs.begin(ctx, tool, None)
        run = self._runs.check(ctx, call)
        if isinstance(run, Denied):
            return run
        return self._recorder.deny(bind(call, run), code, details=checked)

    def _prepare(
        self,
        run: CheckedRun,
        call: Call,
        tool: str,
        action: Action | str,
        args: Mapping[str, JsonValue],
    ) -> Prepared | Denied:
        adapter = self._tools.get(tool)
        config = run.engine.bundle.tools.get(tool)
        if adapter is None or config is None:
            return self._recorder.deny(call, GateReason.UNKNOWN_TOOL)
        checked_action = safe_action(action)
        if checked_action is None:
            return self._recorder.deny(call, GateReason.ACTION_NOT_SUPPORTED)
        validated = _validate_args(adapter, config.id, checked_action, args)
        if isinstance(validated, dict):
            return self._recorder.deny(call, GateReason.INVALID_ARGS, details=validated)
        if tool == SEARCH_TOOL and checked_action is not Action.READ:
            # Search never consults the engine per call, so its built-in check lives here.
            return self._recorder.deny(call, Reason.ACTION_NOT_SUPPORTED)

        facts = adapter.facts(validated)
        values = self._call_values(run, config.id, checked_action, validated, facts)
        document: DocumentFacts | None = None
        if facts.document_id is not None:
            stored = (
                self._store.get_document(facts.document_id)
                if is_lookupable(facts.document_id)
                else None
            )
            if stored is None:
                return self._recorder.deny(
                    call,
                    GateReason.DOCUMENT_NOT_FOUND,
                    model_reason=GateReason.NOT_FOUND,
                    subject=facts.document_id,
                    unknown_subject=facts.document_id,
                )
            document = DocumentFacts.from_document(stored)
            if stored.customer_id != run.ctx.customer_id:
                # Decided here, before spend, so a missing id and a foreign one take the
                # same path and cannot be told apart by a cap or by timing of the steps.
                decision = run.engine.decide(
                    _request(run, config.id, checked_action, document, facts, None)
                )
                return self._recorder.deny(
                    call,
                    decision.reason.value,
                    model_reason=GateReason.NOT_FOUND,
                    subject=facts.document_id,
                    rule_ids=decision.rule_ids,
                    policy_version=decision.policy_version,
                    details=values,
                )
        elif facts.write_label is not None:
            document = DocumentFacts(
                customer_id=run.ctx.customer_id, label=facts.write_label, tags=NOTE_TAGS
            )
        return Prepared(
            adapter=adapter,
            config=config,
            action=checked_action,
            args=validated,
            facts=facts,
            document=document,
            document_id=facts.document_id,
            values=values,
        )

    def _call_values(
        self, run: CheckedRun, tool: str, action: Action, args: BaseModel, facts: ToolFacts
    ) -> dict[str, JsonValue]:
        """What the decision events of a validated call record about its args: never a
        value a person wrote, only its hash, counts and the amount."""
        values: dict[str, JsonValue] = {}
        if tool != SEARCH_TOOL:
            # An adapter's args the hash cannot represent are recorded without one.
            with contextlib.suppress(TypeError, ValueError):
                values["args_hash"] = args_hash(tool, action, args.model_dump(mode="json"))
        if facts.recipients:
            domains = frozenset(domain.lower() for domain in run.customer.domains)
            external = sum(1 for raw in facts.recipients if is_external(raw, domains))
            values["recipients"] = len(facts.recipients)
            values["recipients_internal"] = len(facts.recipients) - external
            values["recipients_external"] = external
        if facts.amount_usd is not None:
            values["amount_usd"] = str(facts.amount_usd)
        return values

    def _verdict(self, run: CheckedRun, prepared: Prepared) -> Verdict:
        max_label = self._runs.max_label(run.ctx.run_id)
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
        verdict = Verdict(
            decision.effect, decision.reason.value, decision.rule_ids, decision.policy_version
        )
        written = prepared.facts.write_label
        if verdict.effect is not Effect.DENY and written is not None and max_label is not None:
            written_rank = run.policy.label_rank(written)
            max_rank = run.policy.label_rank(max_label)
            # Writing what the run has read into a less sensitive document would launder it.
            if written_rank is None or max_rank is None or written_rank < max_rank:
                return Verdict(Effect.DENY, GateReason.WRITE_DOWN, (), decision.policy_version)
        return verdict

    def _deny_verdict(
        self,
        call: Call,
        prepared: Prepared,
        verdict: Verdict,
        subject: str | None,
        *,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> Denied:
        # Any denial of a call that names a document reads as not_found to the model, the
        # same as a missing id, so ids (which reveal titles) cannot be probed even within
        # the tenant. The audit keeps the real reason.
        model_reason = (
            GateReason.NOT_FOUND.value
            if prepared.document_id is not None
            else _model_reason(verdict.reason)
        )
        return self._recorder.deny(
            call,
            verdict.reason,
            model_reason=model_reason,
            subject=subject,
            rule_ids=verdict.rule_ids,
            policy_version=verdict.policy_version,
            details={**prepared.values, **(extra or {})},
        )

    def _escalation(self, run: CheckedRun, prepared: Prepared) -> GateReason | None:
        if prepared.config.customer_facing and self._reaches_outside(run, prepared):
            return GateReason.ESCALATED_CUSTOMER_FACING
        if self._uncleared_recipient(run, prepared.facts.recipients):
            return GateReason.ESCALATED_RECIPIENT_CLEARANCE
        if self._risk(run, prepared) >= run.policy.approval_settings().risk_threshold:
            return GateReason.ESCALATED_RISK
        return None

    def _reaches_outside(self, run: CheckedRun, prepared: Prepared) -> bool:
        """Whether a customer-facing call reaches someone outside the company.

        A call with no recipient facts (a refund, say) always reaches a customer. One with
        recipients does when any of them is outside the customer's domains, so an internal
        email the policy allows runs without approval.
        """
        recipients = prepared.facts.recipients
        if not recipients:
            return True
        return self._any_external(run, recipients)

    def _uncleared_recipient(self, run: CheckedRun, recipients: tuple[str, ...]) -> bool:
        """Whether an internal recipient may not read what the run has already been shown.

        Once a run has seen a document at some label, mail to an internal address goes out
        without approval only when every internal recipient is an active user of the
        customer whose role may read a document at that label. The engine is asked about a
        document with that label alone, no tags and not assigned to anyone, so a grant that
        needs more than the label does not count. External recipients are left to the
        policy and to the customer-facing escalation.
        """
        if not recipients:
            return False
        label = self._runs.max_label(run.ctx.run_id)
        if label is None:
            return False
        domains = frozenset(domain.lower() for domain in run.customer.domains)
        users = {user.email.lower(): user for user in self._store.list_users(run.ctx.customer_id)}
        for raw in recipients:
            parsed = parse_address(raw)
            if parsed is None or parsed[1] not in domains:
                continue
            user = users.get(parsed[0])
            if user is None or user.status is not UserStatus.ACTIVE:
                return True
            decision = run.engine.decide(
                Request(
                    customer_id=run.ctx.customer_id,
                    user_id=user.id,
                    role=user.role,
                    tool=READ_TOOL,
                    action=Action.READ,
                    document=DocumentFacts(customer_id=run.ctx.customer_id, label=label),
                )
            )
            if decision.effect is not Effect.ALLOW:
                return True
        return False

    def _any_external(self, run: CheckedRun, recipients: tuple[str, ...]) -> bool:
        domains = frozenset(domain.lower() for domain in run.customer.domains)
        return any(is_external(raw, domains) for raw in recipients)

    def _risk(self, run: CheckedRun, prepared: Prepared) -> int:
        risk = prepared.config.base_risk
        if self._any_external(run, prepared.facts.recipients):
            risk += EXTERNAL_RECIPIENT_RISK
        amount = prepared.facts.amount_usd
        if amount is not None and amount.is_finite() and amount > 0:
            risk += min(MAX_SPEND_RISK, int(amount // 100) * SPEND_RISK_PER_100_USD)
        return risk

    def _execute(
        self,
        run: CheckedRun,
        call: Call,
        prepared: Prepared,
        reservation: Reservation,
        verdict: Verdict,
        *,
        subject: str | None,
        allow_recorded: bool = False,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> Outcome:
        """Record the allow, unless the caller already has, then run the tool. `extra` is
        added to the EXECUTED event's details."""
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
            run,
            call,
            prepared,
            reservation,
            result,
            [*labels, *result.returned_labels],
            subject,
            extra=extra,
        )

    def _record_allow(
        self,
        call: Call,
        prepared: Prepared,
        verdict: Verdict,
        subject: str | None,
        details: dict[str, JsonValue] | None,
    ) -> None:
        details = {**prepared.values, **(details or {})}
        self._recorder.emit(
            call,
            EventKind.ALLOW,
            verdict.reason,
            subject=subject,
            rule_ids=verdict.rule_ids,
            policy_version=verdict.policy_version,
            decision=True,
            details=details,
        )

    def _search(self, run: CheckedRun, call: Call, prepared: Prepared) -> Outcome:
        reservation = self._runs.reserve(run, call, prepared.config.cost_usd)
        if isinstance(reservation, CapReached):
            return reservation
        max_label = self._runs.max_label(run.ctx.run_id)
        searches: list[tuple[RetrievalResult, int]] = []

        def search(query: str, limit: int) -> RetrievalResult:
            # The policy check per document is the retrieval filter itself.
            found = self._index.search(
                run.engine,
                run.ctx,
                query,
                run_max_label=max_label,
                tool=SEARCH_TOOL,
                limit=limit,
                principals=run.user.principals,
            )
            searches.append((found, call.latency_us()))
            return found

        env = ToolEnv(store=self._store, index=self._index, clock=self._clock, search=search)
        try:
            result = prepared.adapter.run(env, run.ctx, prepared.args)
        except RetrievalInvariantError:
            self._ledger.release(reservation)
            self._recorder.emit(
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
            self._recorder.emit(
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
        call: Call,
        prepared: Prepared,
        env: ToolEnv,
        run: CheckedRun,
        reservation: Reservation,
        subject: str | None,
    ) -> ToolResult | Denied:
        values = prepared.values
        try:
            result = prepared.adapter.run(env, run.ctx, prepared.args)
        except Exception as exc:
            return self._tool_failed(call, reservation, subject, exc, values=values)
        if not result.cost_usd.is_finite() or result.cost_usd < 0:
            return self._tool_failed(call, reservation, subject, None, values=values)
        return result

    def _tool_failed(
        self,
        call: Call,
        reservation: Reservation,
        subject: str | None,
        exc: Exception | None,
        *,
        values: Mapping[str, JsonValue] | None = None,
    ) -> Denied:
        self._ledger.release(reservation)
        # The exception type only: messages can quote arguments or document text.
        error = "invalid_result" if exc is None else type(exc).__name__
        details: dict[str, JsonValue] = {**(values or {}), "error": error}
        return self._recorder.deny(call, GateReason.TOOL_ERROR, subject=subject, details=details)

    def _finish(
        self,
        run: CheckedRun,
        call: Call,
        prepared: Prepared,
        reservation: Reservation,
        result: ToolResult,
        labels: Iterable[str],
        subject: str | None,
        *,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> Executed:
        cost = prepared.config.cost_usd + result.cost_usd
        self._ledger.settle(reservation, cost)
        max_label = self._runs.raise_label(run, labels)
        details: dict[str, JsonValue] = {
            "cost_usd": str(cost),
            "run_max_label": max_label,
            **prepared.values,
            **(extra or {}),
        }
        if result.effect_ref is not None:
            details["effect_ref"] = self._recorder.effect_ref(result.effect_ref)
        self._recorder.emit(
            call,
            EventKind.EXECUTED,
            GateReason.EXECUTED,
            subject=subject,
            details=details,
        )
        return Executed(tool=prepared.config.id, content=result.content, cost_usd=cost)


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


def _request(
    run: CheckedRun,
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
        # From the user re-read from the store for this call, never from the caller.
        principals=run.user.principals,
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


_REFUSAL_DETAIL_KEYS = frozenset({"error_types", "fields"})


def _refusal_details(details: Mapping[str, JsonValue] | None) -> dict[str, JsonValue] | None:
    """Refusal details as `refuse` accepts them, or ValueError."""
    if details is None:
        return None
    if not set(details) <= _REFUSAL_DETAIL_KEYS:
        raise ValueError("refusal details may only hold error_types and fields")
    checked: dict[str, JsonValue] = {}
    for key, value in details.items():
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"refusal detail {key} must be a list of strings")
        checked[key] = [*sorted(str(item) for item in value)]
    return checked


def _model_reason(reason: str) -> str:
    if reason == Reason.DOCUMENT_CUSTOMER_MISMATCH:
        return GateReason.NOT_FOUND.value
    return reason
