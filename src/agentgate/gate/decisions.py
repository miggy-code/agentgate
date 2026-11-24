"""Approval decisions: a person approving, rejecting or letting an approval expire.

Each state change is recorded inside the transaction that makes it. An approval is
approved, rejected or expired only if its event (APPROVED, REJECTED, EXPIRED) is recorded;
if the sink refuses the event, the change is rolled back. Creating an approval and claiming
one for execution follow the same rule, in tool_calls.
"""

import time
from collections.abc import Mapping
from dataclasses import replace

from agentgate.approvals import Approval, ApprovalError, ApprovalErrorReason, ApprovalQueue
from agentgate.audit.base import EventKind
from agentgate.gate._values import (
    APPROVAL_ID,
    UNKNOWN_CUSTOMER,
    as_text,
    get_approval,
    get_user,
    is_lookupable,
)
from agentgate.gate.outcomes import GateReason
from agentgate.gate.recorder import AuditRecorder, Call
from agentgate.gate.runs import RunRegistry
from agentgate.policy.engine import PolicyEngine
from agentgate.store import Store


class ApprovalDecisions:
    """Records and applies an approver's decision on a queued approval."""

    def __init__(
        self,
        *,
        store: Store,
        engines: Mapping[str, PolicyEngine],
        approvals: ApprovalQueue,
        runs: RunRegistry,
        recorder: AuditRecorder,
    ) -> None:
        self._store = store
        self._engines = engines
        self._approvals = approvals
        self._runs = runs
        self._recorder = recorder

    def approve(self, approval_id: str, approver_user_id: str, note: str | None = None) -> Approval:
        return self._decide_approval(approval_id, approver_user_id, note, approve=True)

    def reject(self, approval_id: str, approver_user_id: str, note: str | None = None) -> Approval:
        return self._decide_approval(approval_id, approver_user_id, note, approve=False)

    def expire_due(self) -> list[Approval]:
        return self._approvals.expire_due(record=self.record_expired)

    def record_expired(self, approval: Approval) -> None:
        call = Call(
            time.perf_counter_ns(),
            approval.customer_id,
            user_id=approval.requested_by,
            role=self._runs.role(approval.run_id),
            run_id=approval.run_id,
            tool=approval.tool,
            action=approval.action,
            policy_version=approval.policy_version,
        )
        self._recorder.emit(
            call,
            EventKind.EXPIRED,
            GateReason.EXPIRED,
            subject=approval.id,
            rule_ids=approval.rule_ids,
            details={"approval_args_hash": approval.args_hash},
        )

    def _decide_approval(
        self, approval_id: str, approver_user_id: str, note: str | None, *, approve: bool
    ) -> Approval:
        started = time.perf_counter_ns()
        approver = get_user(self._store, approver_user_id)
        approval = get_approval(self._approvals, approval_id)
        safe_id = approval_id if APPROVAL_ID.fullmatch(approval_id) else None
        # An id the queue does not know is the caller's value, recorded as a reference.
        unknown_id = as_text(approval_id) if approval is None else None
        if approver is None:
            customer_id = UNKNOWN_CUSTOMER if approval is None else approval.customer_id
            claimed_user = as_text(approver_user_id)
            call = Call(started, customer_id, user_id=claimed_user, claimed=frozenset({"user_id"}))
            err = ApprovalError(ApprovalErrorReason.APPROVER_NOT_ALLOWED, "unknown approver")
            self._recorder.emit(
                call,
                EventKind.DENY,
                err.reason.value,
                subject=safe_id,
                unknown_subject=unknown_id,
                decision=True,
            )
            raise err

        call = Call(started, approver.customer_id, user_id=approver.id, role=approver.role)
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
            self._recorder.emit(
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
                    "approval_args_hash": decided.args_hash,
                },
            )

        try:
            if engine is None:
                raise ApprovalError(
                    ApprovalErrorReason.APPROVER_NOT_ALLOWED, "approver's customer has no policy"
                )
            if not is_lookupable(approval_id):
                # The queue would look it up in SQLite, which cannot bind it.
                raise ApprovalError(ApprovalErrorReason.NOT_FOUND, "approval not found")
            settings = engine.bundle.policy.approvals
            if settings is None:
                # A models-only policy names no approvers.
                raise ApprovalError(
                    ApprovalErrorReason.APPROVER_NOT_ALLOWED, "approver's policy has no approvals"
                )
            # The decision commits only if its event is recorded; a sink error propagates
            # and leaves the approval as it was.
            decide = self._approvals.approve if approve else self._approvals.reject
            return decide(
                approval_id,
                approver,
                settings,
                note,
                record_decision,
                self.record_expired,
            )
        except ApprovalError as err:
            self._recorder.emit(
                call,
                EventKind.DENY,
                err.reason.value,
                subject=safe_id,
                unknown_subject=unknown_id,
                decision=True,
            )
            raise
