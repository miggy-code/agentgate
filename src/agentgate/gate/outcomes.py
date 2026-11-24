"""What a gate call returns, and the reason codes the gate adds to the policy engine's."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import JsonValue

from agentgate.spend import CapKind


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
    # An internal recipient may not read the most sensitive label the run has seen.
    ESCALATED_RECIPIENT_CLEARANCE = "escalated_recipient_clearance"
    AUDIT_UNAVAILABLE = "audit_unavailable"
    # A model request whose body could not be read strictly (see model.admission).
    MODEL_REQUEST_INVALID = "model_request_invalid"
    # The body has a top-level field the provider's list does not name.
    MODEL_UNKNOWN_FIELD = "model_unknown_field"
    # The policy reserves a computed worst case and the rate table cannot give one.
    MODEL_UNPRICED = "model_unpriced"
    # A model request whose token was refused (unknown, revoked, expired, for another run).
    TOKEN_REFUSED = "token_refused"
    # How a model call's reservation ended, on its EXECUTED event.
    MODEL_SETTLED = "model_settled"
    MODEL_RELEASED = "model_released"
    TOOL_ERROR = "tool_error"
    RETRIEVAL_FILTER = "retrieval_filter"
    RETRIEVAL_INVARIANT = "retrieval_invariant"
    EXECUTED = "executed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    POLICY_LOADED = "policy_loaded"


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
