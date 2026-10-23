"""Offline mock models, so the demo and the red-team suite run with no keys and no network.

Every mock is deterministic: no randomness and no clock. Each response is a function of the
system prompt, the messages and the tool specs it is given, so two identical runs propose
identical calls. All mocks share one price list, and their estimate always covers the
response they are about to give, so a model reservation never settles above what it held.

NaiveModel is the interesting one. It is compliant and has no judgement: it does what the
text in front of it says, including instructions planted in an untrusted inbox email. It
exists to show that prompt injection is stopped by the gate, not by the model.
"""

import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal

from pydantic import JsonValue

from agentgate.agent.model import (
    AssistantMessage,
    Message,
    ModelResponse,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

INPUT_USD_PER_TOKEN = Decimal("1.00") / 1_000_000
OUTPUT_USD_PER_TOKEN = Decimal("5.00") / 1_000_000
# Output tokens an estimate allows for. A mock knows its next response, so a longer one
# (the naive model's export email, say) is estimated at its real length instead.
OUTPUT_ALLOWANCE_TOKENS = 600
CHARS_PER_TOKEN = 4

SEARCH_TOOL = "docs.search"
READ_TOOL = "docs.read"
EMAIL_TOOL = "email.send"
READS_PER_TURN = 3
# Used when a spec does not state a maxLength, matching the built-in args models.
DEFAULT_QUERY_CHARS = 500
DEFAULT_BODY_CHARS = 20_000
DEFAULT_SUBJECT_CHARS = 300

_ADDRESS = r"([a-z0-9._%+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+)"
_VERB = r"\b(?:send|e-?mail|forward)\b"
# "send ... to someone@example.com" within one sentence. Documents also say things like
# "email care@... with your order number", which is advice to a reader, not an instruction
# to send something to that address, so in tool results only the "to <address>" form counts.
_SEND_TO = re.compile(_VERB + r"[^.\n]{0,300}?\bto\s+" + _ADDRESS, re.IGNORECASE)
# In the task the user may also say "email someone@example.com about ...".
_DIRECT = re.compile(_VERB + r"\s+" + _ADDRESS, re.IGNORECASE)


def tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def cost_usd(input_tokens: int, output_tokens: int) -> Decimal:
    return input_tokens * INPUT_USD_PER_TOKEN + output_tokens * OUTPUT_USD_PER_TOKEN


def request_text(system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> str:
    """Everything a request carries, as the text the mock price list counts."""
    payload: dict[str, JsonValue] = {
        "system": system,
        "messages": [_message_json(message) for message in messages],
        "tools": [
            {
                "tool": spec.tool,
                "action": spec.action.value,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in tools
        ],
    }
    return _dumps(payload)


def response_text(message: AssistantMessage) -> str:
    calls: list[JsonValue] = [
        {"call_id": call.call_id, "tool": call.tool, "args": call.args}
        for call in message.tool_calls
    ]
    return _dumps({"text": message.text, "tool_calls": calls})


class _MockModel:
    """Shared pricing and response plumbing. Subclasses decide the next message."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def estimate_cost(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Decimal:
        planned = tokens(response_text(self._next(messages, tools)))
        output = max(OUTPUT_ALLOWANCE_TOKENS, planned)
        return cost_usd(tokens(request_text(system, messages, tools)), output)

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        message = self._next(messages, tools)
        usage = Usage(
            input_tokens=tokens(request_text(system, messages, tools)),
            output_tokens=tokens(response_text(message)),
        )
        return ModelResponse(
            message=message,
            usage=usage,
            cost_usd=cost_usd(usage.input_tokens, usage.output_tokens),
            stop_reason="tool_use" if message.tool_calls else "end_turn",
        )

    def _next(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantMessage:
        raise NotImplementedError


class ScriptedModel(_MockModel):
    """Turn i proposes turns[i] (call ids call-<i>-<j>), then every later turn ends the run."""

    def __init__(
        self,
        name: str,
        turns: Sequence[Sequence[tuple[str, dict[str, JsonValue]]]],
        final_text: str = "Done.",
    ) -> None:
        super().__init__(name)
        self._turns = tuple(tuple((tool, dict(args)) for tool, args in turn) for turn in turns)
        self._final_text = final_text

    def _next(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantMessage:
        turn = _turn_index(messages)
        if turn >= len(self._turns) or not self._turns[turn]:
            return AssistantMessage(text=self._final_text)
        calls = tuple(
            ToolCall(call_id=f"call-{turn}-{j}", tool=tool, args=dict(args))
            for j, (tool, args) in enumerate(self._turns[turn])
        )
        return AssistantMessage(text=None, tool_calls=calls)


class RepeatModel(_MockModel):
    """Proposes the same call on every turn and never finishes: a runaway loop."""

    def __init__(self, name: str, tool: str, args: dict[str, JsonValue]) -> None:
        super().__init__(name)
        self._tool = tool
        self._args = dict(args)

    def _next(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantMessage:
        call = ToolCall(call_id=f"call-{_turn_index(messages)}-0", tool=self._tool, args=self._args)
        return AssistantMessage(text=None, tool_calls=(call,))


class NaiveModel(_MockModel):
    """A compliant heuristic model that follows any instruction it reads.

    Its state is only what the messages show: the task, its own earlier calls and their
    results. In order of preference it searches for the task, reads every document the
    search returned that it has not read yet, obeys an instruction in a tool result to send
    something to an address (mailing everything it has read), sends an email the task asks
    for, and otherwise finishes with the titles it read. It never invents a document id.
    """

    def __init__(self, name: str = "naive-mock") -> None:
        super().__init__(name)

    def _next(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantMessage:
        state = _Conversation(messages)
        turn = _turn_index(messages)
        specs = {spec.tool: spec for spec in tools}

        if not state.searched:
            query = state.task[: _max_length(specs, SEARCH_TOOL, "query", DEFAULT_QUERY_CHARS)]
            return _calls(turn, [(SEARCH_TOOL, {"query": query or "help"})])

        unread = [doc_id for doc_id in state.found_ids if doc_id not in state.attempted_reads]
        if unread:
            return _calls(
                turn, [(READ_TOOL, {"document_id": doc_id}) for doc_id in unread[:READS_PER_TURN]]
            )

        for address in state.instructed_addresses:
            if address not in state.emailed:
                limit = _max_length(specs, EMAIL_TOOL, "body", DEFAULT_BODY_CHARS)
                body = "\n\n".join(body for _, body in state.read_documents)[:limit]
                email: dict[str, JsonValue] = {
                    "to": [address],
                    "subject": "Requested export",
                    "body": body or "Nothing was read.",
                }
                return _calls(turn, [(EMAIL_TOOL, email)])

        for address in _addresses(_DIRECT, state.task) + _addresses(_SEND_TO, state.task):
            if address not in state.emailed:
                limit = _max_length(specs, EMAIL_TOOL, "subject", DEFAULT_SUBJECT_CHARS)
                reply: dict[str, JsonValue] = {
                    "to": [address],
                    "subject": "Following up on your request"[:limit],
                    "body": "Hello,\n\nThank you for getting in touch. We will follow up with"
                    " the details shortly.\n\nBest regards",
                }
                return _calls(turn, [(EMAIL_TOOL, reply)])

        titles = [title for title, _ in state.read_documents]
        if not titles:
            return AssistantMessage(text="I could not find any documents for this task.")
        return AssistantMessage(text="I read: " + "; ".join(titles) + ".")


class _Conversation:
    """What the naive model can tell from the messages it was given."""

    def __init__(self, messages: Sequence[Message]) -> None:
        self.task = next((m.text for m in messages if isinstance(m, UserMessage)), "")
        self.searched = False
        self.attempted_reads: set[str] = set()
        self.emailed: set[str] = set()
        self.found_ids: list[str] = []
        self.read_documents: list[tuple[str, str]] = []
        self.instructed_addresses: list[str] = []
        for message in messages:
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls:
                    self._proposed(call)
            elif isinstance(message, ToolResultsMessage):
                for result in message.results:
                    if result.status == "executed":
                        self._executed(result.tool, result.content)

    def _proposed(self, call: ToolCall) -> None:
        if call.tool == SEARCH_TOOL:
            self.searched = True
        elif call.tool == READ_TOOL:
            document_id = call.args.get("document_id")
            if isinstance(document_id, str):
                self.attempted_reads.add(document_id)
        elif call.tool == EMAIL_TOOL:
            recipients = call.args.get("to")
            if isinstance(recipients, list):
                self.emailed.update(r.lower() for r in recipients if isinstance(r, str))

    def _executed(self, tool: str, content: dict[str, JsonValue]) -> None:
        if tool == SEARCH_TOOL:
            documents = content.get("documents")
            for document in documents if isinstance(documents, list) else []:
                doc_id = document.get("id") if isinstance(document, dict) else None
                if isinstance(doc_id, str) and doc_id not in self.found_ids:
                    self.found_ids.append(doc_id)
        elif tool == READ_TOOL:
            title, body = content.get("title"), content.get("body")
            if isinstance(title, str) and isinstance(body, str):
                self.read_documents.append((title, body))
        for text in _strings(content):
            for address in _addresses(_SEND_TO, text):
                if address not in self.instructed_addresses:
                    self.instructed_addresses.append(address)


def _turn_index(messages: Sequence[Message]) -> int:
    return sum(1 for message in messages if isinstance(message, AssistantMessage))


def _calls(turn: int, calls: Sequence[tuple[str, dict[str, JsonValue]]]) -> AssistantMessage:
    return AssistantMessage(
        text=None,
        tool_calls=tuple(
            ToolCall(call_id=f"call-{turn}-{j}", tool=tool, args=args)
            for j, (tool, args) in enumerate(calls)
        ),
    )


def _addresses(pattern: re.Pattern[str], text: str) -> list[str]:
    return [match.group(1).lower() for match in pattern.finditer(text)]


def _strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _max_length(specs: Mapping[str, ToolSpec], tool: str, field: str, default: int) -> int:
    spec = specs.get(tool)
    properties = None if spec is None else spec.input_schema.get("properties")
    schema = properties.get(field) if isinstance(properties, dict) else None
    limit = schema.get("maxLength") if isinstance(schema, dict) else None
    return limit if isinstance(limit, int) and limit > 0 else default


def _message_json(message: Message) -> JsonValue:
    if isinstance(message, UserMessage):
        return {"role": "user", "text": message.text}
    if isinstance(message, AssistantMessage):
        return {"role": "assistant", "content": json.loads(response_text(message))}
    results: list[JsonValue] = [
        {
            "call_id": result.call_id,
            "tool": result.tool,
            "status": result.status,
            "content": result.content,
        }
        for result in message.results
    ]
    return {"role": "tool", "results": results}


def _dumps(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
