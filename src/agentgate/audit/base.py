"""Audit events and the sink interface.

Every gate decision becomes one AuditEvent: who asked, for what, what the gate decided, why,
and under which policy version. Events carry ids, labels, counts and reason codes, never
document bodies or message text, so the audit log does not become a second copy of the
data it protects.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agentgate.models import Action, Role


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    ALLOW = "allow"
    DENY = "deny"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    CAP_HIT = "cap_hit"
    RETRIEVAL = "retrieval"
    INVARIANT_VIOLATION = "invariant_violation"


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    kind: EventKind
    customer_id: str
    user_id: str | None = None
    role: Role | None = None
    run_id: str | None = None
    tool: str | None = None
    action: Action | None = None
    # What the decision was about: a document id, an approval id, a cap name.
    subject: str | None = None
    # A stable reason code, such as an engine Reason value or "run_cap".
    reason: str
    rule_ids: tuple[str, ...] = ()
    policy_version: str | None = None
    # Time the gate spent deciding, excluding the tool itself.
    latency_us: int | None = Field(default=None, ge=0)
    details: dict[str, JsonValue] = {}

    def model_post_init(self, __context: object) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("audit timestamps must be timezone-aware")


class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None:
        """Durably store one event, or raise. The gate denies an action it cannot record."""
        ...


class MemorySink:
    """Keeps events in a list. For unit tests and teams that forward events elsewhere."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)
