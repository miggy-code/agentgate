"""The tool interface: what the gate needs from a tool to decide on a call and run it.

An adapter validates the model's arguments with its own strict args model, states the facts
the policy engine decides on (a document, a written label, recipients, an amount) and runs
the call. Adapters never decide anything themselves. The gate validates, decides, records
and only then calls run, so a tool that is reachable only through the gate cannot act on
arguments the gate has not seen.
"""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, JsonValue

from agentgate.clock import Clock
from agentgate.context import RunContext
from agentgate.retrieval import DocumentIndex, RetrievalResult
from agentgate.store import Store

# A permission-filtered search bound by the gate to one run, its policy and its current
# max label: (query, limit) -> result. Search tools get this instead of a raw query path.
Searcher = Callable[[str, int], RetrievalResult]


@dataclass(frozen=True)
class ToolFacts:
    """What the engine decides on for one call, taken from the validated args."""

    # A document the call reads; the gate loads it and checks its customer and label.
    document_id: str | None = None
    # For a write: the label of the document being written.
    write_label: str | None = None
    recipients: tuple[str, ...] = ()
    amount_usd: Decimal | None = None


@dataclass(frozen=True)
class ToolResult:
    # What the model sees.
    content: dict[str, JsonValue]
    # Metered cost beyond the per-call cost in tools.yaml. Built-in tools add nothing.
    cost_usd: Decimal = Decimal(0)
    # Labels of every document whose content is in `content`, so the gate can raise the
    # run's max label from what actually reached the model.
    returned_labels: tuple[str, ...] = ()
    # The id of the side effect the call made (an outbox row, a parts order, a refund, a
    # note), for the EXECUTED event, so a store row can be matched to its event. An id the
    # tool chose, never content; None for a call with no side effect.
    effect_ref: str | None = None


@dataclass(frozen=True)
class ToolEnv:
    store: Store
    index: DocumentIndex
    clock: Clock
    # Set by the gate for search tools only.
    search: Searcher | None = None


class ToolAdapter(Protocol):
    @property
    def tool_id(self) -> str: ...

    @property
    def args_model(self) -> type[BaseModel]:
        """A strict, frozen pydantic model with extra="forbid"."""
        ...

    def facts(self, args: BaseModel) -> ToolFacts:
        """The facts of a call, from args already validated with args_model."""
        ...

    def run(self, env: ToolEnv, ctx: RunContext, args: BaseModel) -> ToolResult:
        """Perform the call. Called only after the gate has allowed and recorded it."""
        ...
