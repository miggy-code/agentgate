"""The gate: every tool call and model request an agent run makes is checked, decided and
recorded here before anything happens.

`Gate` (in `core`) is the whole interface. The modules behind it each own one part: runs,
tool calls, model calls, approval decisions, operator records and the audit recorder they
share. Nothing outside this package imports those modules.
"""

from agentgate.gate._values import UNKNOWN_CUSTOMER
from agentgate.gate.core import Gate
from agentgate.gate.outcomes import (
    CapReached,
    Denied,
    Executed,
    GateReason,
    IdentityError,
    Outcome,
    PendingApproval,
)
from agentgate.gate.recorder import store_audit_reference

__all__ = [
    "UNKNOWN_CUSTOMER",
    "CapReached",
    "Denied",
    "Executed",
    "Gate",
    "GateReason",
    "IdentityError",
    "Outcome",
    "PendingApproval",
    "store_audit_reference",
]
