"""The model interface the reference agent talks to.

A model only proposes: it returns text and tool calls, and the agent loop hands every call to
the gate. Nothing a model returns can name the user, the customer or the role; those come
from the run context the gate created. The same interface fits the offline mock models and
an optional real model, so the demo and the red-team suite run without keys.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import JsonValue

from agentgate.models import Action


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the model sees it. The action is fixed per tool, not chosen by the model."""

    tool: str
    action: Action
    description: str
    input_schema: dict[str, JsonValue]


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool: str
    args: dict[str, JsonValue]


OutcomeStatus = Literal["executed", "denied", "pending_approval", "cap_reached", "error"]


@dataclass(frozen=True)
class ToolOutcome:
    """What the model is told about one call. `content` is everything it gets to see."""

    call_id: str
    tool: str
    status: OutcomeStatus
    content: dict[str, JsonValue]


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantMessage:
    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    # Provider-native content to send back unchanged on the next request, when a real model
    # needs it (for example thinking blocks). Mock models leave it None.
    native: object | None = None


@dataclass(frozen=True)
class ToolResultsMessage:
    results: tuple[ToolOutcome, ...]


Message = UserMessage | AssistantMessage | ToolResultsMessage


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ModelResponse:
    message: AssistantMessage
    usage: Usage
    cost_usd: Decimal
    # "end_turn" when the model is done, "tool_use" when it proposed calls, or a provider
    # reason such as "max_tokens" or "refusal".
    stop_reason: str


class ModelError(RuntimeError):
    """The model could not produce a response. The agent loop stops the run."""


class Model(Protocol):
    @property
    def name(self) -> str: ...

    def estimate_cost(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Decimal:
        """An upper estimate of the next call's cost, reserved against spend caps first."""
        ...

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse: ...
