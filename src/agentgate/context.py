"""The trusted identity of one agent run.

A RunContext is created only by the gate, from an identity the host application has already
authenticated, or that agentgate verified itself from a signed assertion (RunAssurance).
Nothing the model writes, and nothing inside a document or tool result, can create or change
one. Every other fact about the run (spend so far, the most sensitive label
it has read) lives in the gate's own records, keyed by run_id.
"""

from dataclasses import dataclass
from datetime import datetime

from agentgate.domain import Role


@dataclass(frozen=True)
class RunContext:
    run_id: str
    customer_id: str
    user_id: str
    role: Role
    started_at: datetime


@dataclass(frozen=True)
class RunAssurance:
    """How a run's user was identified when agentgate verified it itself: the trusted issuer
    that vouched for the user (agentgate.identity), and a keyed reference to the assertion's
    `jti`, never the assertion. A run started without one was named by its host (`--as`, or a
    multi-run proxy's user_id form)."""

    # The issuer's configured name, not its URL.
    issuer: str
    # `ref:` plus 16 hex characters (the gate's audit reference), or None without a jti.
    jti_ref: str | None = None

    @property
    def label(self) -> str:
        """What the run's `assurance` column and its run-start event hold."""
        return f"verified:{self.issuer}"
