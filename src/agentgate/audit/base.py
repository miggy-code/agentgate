"""Audit events and the sink interface.

Every gate decision becomes one AuditEvent: who asked, for what, what the gate decided, why,
and under which policy version. Events carry ids, labels, counts and reason codes, never
document bodies or message text, so the audit log does not become a second copy of the
data it protects.

A value a caller chose (a document id, an approval id, a tool name, a claimed run, user or
customer) is kept on the event as given, so in-process checks can see what was asked. When
it names nothing the store knows, or a durable log would refuse or rewrite it, the gate also
attaches a Reference: a keyed hash of the value. A durable sink stores `recorded()`, in which
each referenced field holds its reference, so a probe value never lands in an append-only
log it could not be erased from.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

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


# The fields of an event that can hold a caller's value, and so can be referenced.
ReferencedField = Literal["customer_id", "user_id", "run_id", "tool", "subject"]
REFERENCED_FIELDS: tuple[ReferencedField, ...] = (
    "customer_id",
    "user_id",
    "run_id",
    "tool",
    "subject",
)


class Reference(BaseModel):
    """What a durable log stores in place of a field's value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # "ref:" and the first 16 hex characters of an HMAC-SHA256 of the value.
    ref: str = Field(pattern=r"^ref:[0-9a-f]{16}$")
    # "unknown": the value names nothing the store knows (a probe, a typo, a forged claim).
    # "unstorable": a real id the durable log would refuse or rewrite as written.
    why: Literal["unknown", "unstorable"]


class AuditRejected(ValueError):
    """A sink refused an event for its content and stored nothing. The gate then records a
    reference-only version of the event instead; any other sink error means the sink is
    down."""


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
    # Fields a durable log must store as a reference instead of the value above.
    references: dict[ReferencedField, Reference] = {}

    def model_post_init(self, __context: object) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("audit timestamps must be timezone-aware")

    def recorded(self) -> "AuditEvent":
        """The event as a durable sink stores it: each referenced field holds its reference."""
        if not self.references:
            return self
        return self.model_copy(update={name: r.ref for name, r in self.references.items()})


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
