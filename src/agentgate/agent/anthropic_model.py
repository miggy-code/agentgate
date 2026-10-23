"""An optional real model: Claude through the official Anthropic Python SDK.

Nothing in agentgate needs this module. The demo, the red-team suite and CI run on the mock
models; this adapter is used only when AGENTGATE_MODEL=anthropic and the `anthropic` extra is
installed. The client is created on the first call, so constructing the adapter needs no key
and never touches the network.

The adapter only translates. Tool names are mapped to the provider's allowed alphabet
(`docs.search` becomes `docs_search`) and back; a name that does not map back is passed
through unchanged, so the gate denies it as an unknown tool. Tool schemas are sent without
`strict`, because the gate validates every argument itself and refuses what does not match.
Assistant turns keep the provider's content as `native` and send it back unchanged, since it
can carry thinking blocks the API expects to see again.

Errors from the SDK become ModelError carrying the exception class name and HTTP status, never
the request or response body, so a failed call cannot copy document text into logs.
"""

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import JsonValue, TypeAdapter, ValidationError

try:
    import anthropic
    from anthropic.types import Message as SdkMessage
    from anthropic.types import TextBlock, ToolUseBlock
    from anthropic.types.beta import (
        BetaFallbackMessageIterationUsage,
        BetaMessage,
        BetaMessageIterationUsage,
        BetaTextBlock,
        BetaToolUseBlock,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the Anthropic model needs the optional extra: pip install 'agentgate[anthropic]'"
    ) from exc

from agentgate.agent.model import (
    AssistantMessage,
    Message,
    ModelError,
    ModelResponse,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

DEFAULT_MODEL = "claude-opus-5-5"
MODEL_ENV = "AGENTGATE_ANTHROPIC_MODEL"

EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})

# Server-side refusal fallbacks: on a policy decline the API reruns the request on the model
# Anthropic recommends for that refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_ERROR_STATUSES = frozenset({"denied", "error", "cap_reached"})
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_MILLION = Decimal(1_000_000)
# Writes to the 5-minute prompt cache cost 1.25 times the base input price.
_CACHE_WRITE_MULTIPLIER = Decimal("1.25")
_ARGS = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: Decimal
    output: Decimal
    cache_read: Decimal

    @property
    def cache_write(self) -> Decimal:
        return self.input * _CACHE_WRITE_MULTIPLIER


# The models this adapter can be configured with.
PRICES: Mapping[str, Price] = {
    "claude-opus-5-5": Price(Decimal("4"), Decimal("20"), Decimal("0.20")),
    "claude-sonnet-5-5": Price(Decimal("2"), Decimal("10"), Decimal("0.20")),
    # Cache read assumed, not published here: 0.1x input for cache reads.
    "claude-haiku-5-5": Price(Decimal("0.10"), Decimal("0.50"), Decimal("0.01")),
}

# Claude Haiku 5.5 charges more once the prompt is over 100K tokens.
_LONG_PROMPT_TOKENS = 100_000
_LONG_PROMPT_PRICES: Mapping[str, Price] = {
    # Input and output are the published above-100K rates. Cache read assumed: 0.1x input
    # for cache reads.
    "claude-haiku-5-5": Price(Decimal("0.50"), Decimal("2.50"), Decimal("0.05")),
}

# Models a server-side fallback can serve from, priced so a fallback-served response is
# charged at the rate it was billed at. Claude Haiku 5.5 has no server-side fallback.
FALLBACK_TARGETS: Mapping[str, tuple[str, ...]] = {
    "claude-opus-5-5": ("claude-opus-5", "claude-opus-4-8"),
    "claude-sonnet-5-5": ("claude-sonnet-5",),
}
# Input and output are the published rates; every cache read here is assumed: 0.1x input
# for cache reads.
_FALLBACK_PRICES: Mapping[str, Price] = {
    "claude-opus-5": Price(Decimal("5"), Decimal("25"), Decimal("0.50")),  # cache read assumed
    "claude-opus-4-8": Price(Decimal("5"), Decimal("25"), Decimal("0.50")),  # cache read assumed
    "claude-sonnet-5": Price(Decimal("2"), Decimal("10"), Decimal("0.20")),  # cache read assumed
}

_KNOWN_PRICES = (*PRICES.values(), *_LONG_PROMPT_PRICES.values(), *_FALLBACK_PRICES.values())
# Used for a model this table does not know, so an unexpected model is never under-charged.
_CEILING_PRICE = Price(
    max(p.input for p in _KNOWN_PRICES),
    max(p.output for p in _KNOWN_PRICES),
    max(p.cache_read for p in _KNOWN_PRICES),
)


def wire_name(tool: str) -> str:
    """The provider-side name for a tool id: `docs.search` becomes `docs_search`."""
    name = _INVALID_NAME_CHARS.sub("_", tool)[:64]
    if not _TOOL_NAME.fullmatch(name):
        raise ModelError("a tool id cannot be mapped to a provider tool name")
    return name


def _price(model: str, prompt_tokens: int) -> Price:
    if prompt_tokens > _LONG_PROMPT_TOKENS and model in _LONG_PROMPT_PRICES:
        return _LONG_PROMPT_PRICES[model]
    return PRICES.get(model) or _FALLBACK_PRICES.get(model) or _CEILING_PRICE


@dataclass(frozen=True)
class _Tokens:
    model: str
    input: int
    cache_write: int
    cache_read: int
    output: int

    @property
    def prompt(self) -> int:
        return self.input + self.cache_write + self.cache_read

    def cost(self) -> Decimal:
        price = _price(self.model, self.prompt)
        dollars = (
            self.input * price.input
            + self.cache_write * price.cache_write
            + self.cache_read * price.cache_read
            + self.output * price.output
        )
        return dollars / _MILLION


class AnthropicModel:
    """Claude as an agentgate Model. One request per `complete`, no streaming, no retries
    beyond the SDK's own."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        max_tokens: int = 2048,
        effort: str = "medium",
        fallbacks: bool = True,
        cache: bool = True,
    ) -> None:
        if model not in PRICES:
            raise ValueError(f"unknown Anthropic model {model!r}; known: {sorted(PRICES)}")
        if effort not in EFFORTS:
            raise ValueError(f"unknown effort {effort!r}; known: {sorted(EFFORTS)}")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self._model = model
        self._client = client
        self._max_tokens = max_tokens
        self._effort = effort
        # A model with no server-side fallback (Claude Haiku 5.5) never sends the parameter.
        self._fallbacks = fallbacks and model in FALLBACK_TARGETS
        self._cache = cache

    @classmethod
    def from_env(cls, *, client: Any | None = None) -> "AnthropicModel":
        """Read the model id from AGENTGATE_ANTHROPIC_MODEL. Credentials stay with the SDK."""
        return cls(model=os.environ.get(MODEL_ENV) or DEFAULT_MODEL, client=client)

    @property
    def name(self) -> str:
        return self._model

    def estimate_cost(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Decimal:
        # Meant to over-reserve against spend caps: about three characters per token is
        # more tokens than the tokenizer produces for ordinary text, the whole request JSON
        # is counted, every input token is priced as a cache write (the dearest input rate),
        # and the full max_tokens of output is assumed. With fallbacks on, a declined hop
        # and the fallback hop that serves the response are both reserved.
        request = self._request(system, messages, tools)
        chars = len(json.dumps(request, default=_dump, sort_keys=True))
        input_tokens = math.ceil(chars / 3)
        models = [self._model]
        if self._fallbacks:
            models.append(max(FALLBACK_TARGETS[self._model], key=_hop_ceiling))
        total = Decimal(0)
        for model in models:
            price = _price(model, input_tokens)
            total += (
                input_tokens * max(price.input, price.cache_write) + self._max_tokens * price.output
            )
        return total / _MILLION

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        request = self._request(system, messages, tools)
        wire_to_tool = _tool_names(tools)
        response = self._send(request)
        if not isinstance(response, SdkMessage | BetaMessage):
            raise ModelError(f"unexpected response type {type(response).__name__}")
        tokens = self._tokens(response)
        usage = Usage(
            input_tokens=sum(t.prompt for t in tokens),
            output_tokens=sum(t.output for t in tokens),
        )
        cost = sum((t.cost() for t in tokens), Decimal(0))
        stop_reason = response.stop_reason or "end_turn"
        return ModelResponse(
            message=_parse(response, wire_to_tool),
            usage=usage,
            cost_usd=cost,
            stop_reason=stop_reason,
        )

    def _request(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": _messages(messages),
            # Thinking is always on for Claude Opus 5.5 and cannot be disabled; effort is the
            # only depth control, and its default differs by model, so it is always explicit.
            "output_config": {"effort": self._effort},
        }
        if system:
            request["system"] = system
        if tools:
            _tool_names(tools)
            request["tools"] = [
                {
                    "name": wire_name(spec.tool),
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                }
                for spec in tools
            ]
        if self._cache:
            # Caches the stable prefix: tools, system and the conversation so far.
            request["cache_control"] = {"type": "ephemeral"}
        return request

    def _send(self, request: dict[str, Any]) -> object:
        try:
            if self._client is None:
                self._client = anthropic.Anthropic()
            if self._fallbacks:
                return self._client.beta.messages.create(
                    **request, betas=[FALLBACK_BETA], fallbacks="default"
                )
            return self._client.messages.create(**request)
        except anthropic.RateLimitError as exc:
            raise ModelError(_describe(exc, exc.status_code)) from None
        except anthropic.APIStatusError as exc:
            raise ModelError(_describe(exc, exc.status_code)) from None
        except anthropic.APITimeoutError as exc:
            raise ModelError(_describe(exc, None)) from None
        except anthropic.APIConnectionError as exc:
            raise ModelError(_describe(exc, None)) from None
        except anthropic.AnthropicError as exc:
            raise ModelError(_describe(exc, None)) from None
        except TypeError as exc:
            # The SDK raises TypeError when it finds no credentials at request time.
            raise ModelError(_describe(exc, None) + "; check Anthropic credentials") from None

    def _tokens(self, response: SdkMessage | BetaMessage) -> list[_Tokens]:
        usage = response.usage
        iterations = getattr(usage, "iterations", None) or ()
        hops = [
            it
            for it in iterations
            if isinstance(it, BetaMessageIterationUsage | BetaFallbackMessageIterationUsage)
        ]
        if hops:
            # A fallback-served response reports each hop with the model that ran it.
            return [
                _Tokens(
                    model=hop.model or self._model,
                    input=hop.input_tokens,
                    cache_write=hop.cache_creation_input_tokens,
                    cache_read=hop.cache_read_input_tokens,
                    output=hop.output_tokens,
                )
                for hop in hops
            ]
        return [
            _Tokens(
                model=response.model,
                input=usage.input_tokens,
                cache_write=usage.cache_creation_input_tokens or 0,
                cache_read=usage.cache_read_input_tokens or 0,
                output=usage.output_tokens,
            )
        ]


def _hop_ceiling(model: str) -> tuple[Decimal, Decimal]:
    price = _price(model, 0)
    return price.output, price.input


def _tool_names(tools: Sequence[ToolSpec]) -> dict[str, str]:
    wire_to_tool: dict[str, str] = {}
    for spec in tools:
        name = wire_name(spec.tool)
        if wire_to_tool.setdefault(name, spec.tool) != spec.tool:
            raise ModelError("two tool ids map to the same provider tool name")
    return wire_to_tool


def _messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        match message:
            case UserMessage(text=text):
                converted.append({"role": "user", "content": text})
            case AssistantMessage():
                content = _assistant_content(message)
                if content:
                    converted.append({"role": "assistant", "content": content})
            case ToolResultsMessage(results=results):
                if results:
                    converted.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": outcome.call_id,
                                    "content": json.dumps(outcome.content, sort_keys=True),
                                    "is_error": outcome.status in _ERROR_STATUSES,
                                }
                                for outcome in results
                            ],
                        }
                    )
    return converted


def _assistant_content(message: AssistantMessage) -> list[Any]:
    if isinstance(message.native, list) and message.native:
        # Sent back exactly as received: it can hold thinking blocks the API checks.
        return message.native
    content: list[Any] = []
    if message.text:
        content.append({"type": "text", "text": message.text})
    content.extend(
        {"type": "tool_use", "id": call.call_id, "name": wire_name(call.tool), "input": call.args}
        for call in message.tool_calls
    )
    return content


def _parse(response: SdkMessage | BetaMessage, wire_to_tool: Mapping[str, str]) -> AssistantMessage:
    if response.stop_reason == "refusal":
        details = response.stop_details
        category = (details.category if details else None) or "unspecified"
        return AssistantMessage(text=f"The model declined to respond (refusal: {category}).")
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in response.content:
        if isinstance(block, TextBlock | BetaTextBlock):
            if block.text:
                texts.append(block.text)
        elif isinstance(block, ToolUseBlock | BetaToolUseBlock):
            try:
                args = _ARGS.validate_python(block.input)
            except ValidationError:
                raise ModelError("a tool call's input is not a JSON object") from None
            calls.append(
                ToolCall(call_id=block.id, tool=wire_to_tool.get(block.name, block.name), args=args)
            )
    text = "\n".join(texts) or None
    if response.stop_reason == "max_tokens":
        # Output was cut off: the last tool call may be incomplete and a partial plan should
        # not run, so every call is dropped. The content is not kept as native either, since
        # tool_use blocks sent back without results would be rejected.
        return AssistantMessage(text=text)
    return AssistantMessage(text=text, tool_calls=tuple(calls), native=response.content)


def _describe(exc: BaseException, status: int | None) -> str:
    suffix = f" (status {status})" if status is not None else ""
    return f"anthropic request failed: {type(exc).__name__}{suffix}"


def _dump(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)
