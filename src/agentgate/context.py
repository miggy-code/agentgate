"""The trusted identity of one agent run.

A RunContext is created only by the gate, from an identity the host application has already
authenticated. Nothing the model writes, and nothing inside a document or tool result, can
create or change one. Every other fact about the run (spend so far, the most sensitive label
it has read) lives in the gate's own records, keyed by run_id.
"""

from dataclasses import dataclass
from datetime import datetime

from agentgate.models import Role


@dataclass(frozen=True)
class RunContext:
    run_id: str
    customer_id: str
    user_id: str
    role: Role
    started_at: datetime
