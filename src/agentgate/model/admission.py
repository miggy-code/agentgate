"""What the gate needs to decide a model call: the request's facts, read strictly, and the
caps the process serving it may tighten.

`parse_model_call` reads the facts the policy's envelope bounds (the model, the declared
output limit, the number of completions) out of a request body. It reads like the strictest
parser it has to agree with, because a gap between what is checked and what the provider
acts on is a bypass: a body that is not one JSON object, that repeats a key, that nests
deeper than the parser allows, or whose model, limit or count has the wrong type is not
guessed at but marked invalid, and the gate denies it. Where a provider has several spellings
of an output limit (`max_tokens` and `max_completion_tokens`), the largest one declared is the
limit, so naming the small one cannot hide the large one.

Only fields the code knows are allowed. Each provider has a list of the top-level body fields
its API takes; a body with any other (`best_of`, OpenRouter's `models` fallback list, a
differently cased `MAX_TOKENS`) is denied, because a field the parser does not read can still
change what the provider does and what it charges. Fields that start server-side work the
policy entry must allow (`allow_server_features`) are listed apart: state kept by the
provider between calls, tools the provider runs itself, and prompt-cache writes. A body must
be UTF-8; the JSON parser would otherwise accept UTF-16 and UTF-32 too.
"""

import json
from dataclasses import dataclass, replace
from decimal import Decimal

from agentgate._json import no_duplicate_keys
from agentgate.policy.schema import ModelProvider, SpendLimits

# A body larger than this is not read at all. The policy's max_request_bytes cannot exceed it.
MAX_BODY_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _WireFormat:
    """What one provider's request body looks like, as far as the gate reads it."""

    # Every top-level field its API takes. Any other is denied.
    fields: frozenset[str]
    # The fields that declare an output limit. Gemini's sit under generationConfig, which the
    # REST API reads in either spelling.
    limit_fields: tuple[str, ...]
    # The fields that ask for more than one completion.
    count_fields: tuple[str, ...]
    # Top-level fields that start work or keep state on the provider's side.
    server_state_fields: frozenset[str]


_OPENAI = _WireFormat(
    fields=frozenset(
        {
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "n",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "seed",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "user",
            "reasoning_effort",
            "functions",
            "function_call",
            "input",
            "instructions",
            "text",
            "reasoning",
            "truncation",
            "include",
            "metadata",
            "previous_response_id",
            "conversation",
            "background",
        }
    ),
    limit_fields=("max_tokens", "max_completion_tokens", "max_output_tokens"),
    count_fields=("n",),
    server_state_fields=frozenset({"previous_response_id", "conversation", "background"}),
)
_FORMATS: dict[ModelProvider, _WireFormat] = {
    ModelProvider.OPENAI: _OPENAI,
    # The same wire protocol under another URL, with the sampling fields some servers add.
    ModelProvider.OPENAI_COMPATIBLE: replace(_OPENAI, fields=_OPENAI.fields | {"top_k", "min_p"}),
    ModelProvider.ANTHROPIC: _WireFormat(
        fields=frozenset(
            {
                "model",
                "messages",
                "max_tokens",
                "system",
                "temperature",
                "top_p",
                "top_k",
                "stop_sequences",
                "stream",
                "tools",
                "tool_choice",
                "metadata",
                "thinking",
                "mcp_servers",
                "container",
            }
        ),
        limit_fields=("max_tokens",),
        count_fields=(),
        server_state_fields=frozenset({"mcp_servers", "container"}),
    ),
    ModelProvider.GEMINI: _WireFormat(
        fields=frozenset(
            {
                "contents",
                "generationConfig",
                "generation_config",
                "systemInstruction",
                "system_instruction",
                "safetySettings",
                "safety_settings",
                "tools",
                "toolConfig",
                "tool_config",
                "cachedContent",
                "cached_content",
            }
        ),
        limit_fields=("maxOutputTokens", "max_output_tokens"),
        count_fields=("candidateCount", "candidate_count"),
        server_state_fields=frozenset({"cachedContent", "cached_content"}),
    ),
}
SERVER_STATE = "server_state"
SERVER_TOOL = "server_tool"
CACHE_WRITE = "cache_write"


@dataclass(frozen=True)
class ModelCall:
    """One model request, as the gate decides it. Nothing here is the request's text."""

    provider: ModelProvider
    # The model as the request names it; "" when it names none, or the body is invalid.
    model: str
    # The declared output limit, or None when none is declared.
    max_tokens: int | None
    # The completions asked for, or None when the request leaves the default of one.
    n: int | None
    request_bytes: int
    # What the client said it is. Recorded as asserted, never relied on.
    correlation_id: str | None = None
    # False when the body could not be read as a request (see parse_model_call).
    valid: bool = True
    # How many top-level fields the provider's list does not name. Their names are the
    # client's text and are not kept.
    unknown_fields: int = 0
    # Which kinds of server-side work the body asks for (SERVER_STATE, SERVER_TOOL,
    # CACHE_WRITE), sorted.
    server_features: tuple[str, ...] = ()


def parse_model_call(
    provider: ModelProvider,
    body: bytes,
    *,
    path_model: str | None = None,
    correlation_id: str | None = None,
) -> ModelCall:
    """The facts of the request in body. It never raises: a body it cannot read strictly is
    returned as an invalid call.

    Gemini names its model in the URL, not the body, so the caller passes it as path_model;
    a model field in a Gemini body is not read.
    """
    invalid = ModelCall(provider, "", None, None, len(body), correlation_id, valid=False)
    if len(body) > MAX_BODY_BYTES:
        return invalid
    try:
        # Decoded here, so only UTF-8 is read: json.loads would take UTF-16 and UTF-32 bytes.
        data = json.loads(body.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    except (ValueError, RecursionError):
        # Not UTF-8, not JSON, a repeated key, or nested too deep to parse.
        return invalid
    if not isinstance(data, dict):
        return invalid

    sources: list[dict[str, object]] = [data]
    model = data.get("model")
    if provider is ModelProvider.GEMINI:
        model = path_model
        sources = []
        for key in ("generationConfig", "generation_config"):
            config = data.get(key)
            if isinstance(config, dict):
                sources.append(config)
            elif config is not None:
                return invalid
    if not isinstance(model, str):
        return invalid

    limits = _integers(sources, _FORMATS[provider].limit_fields)
    counts = _integers(sources, _FORMATS[provider].count_fields)
    if limits is None or counts is None or len(set(counts)) > 1:
        return invalid
    return ModelCall(
        provider=provider,
        model=model,
        max_tokens=max(limits) if limits else None,
        n=counts[0] if counts else None,
        request_bytes=len(body),
        correlation_id=correlation_id,
        unknown_fields=sum(1 for name in data if name not in _FORMATS[provider].fields),
        server_features=_server_features(provider, data),
    )


def _server_features(provider: ModelProvider, data: dict[str, object]) -> tuple[str, ...]:
    """The kinds of server-side work the body asks for. A field it cannot read is counted."""
    found: set[str] = set()
    if any(name in data for name in _FORMATS[provider].server_state_fields):
        found.add(SERVER_STATE)
    tools = data.get("tools")
    if tools is not None and not isinstance(tools, list):
        found.add(SERVER_TOOL)
    for tool in tools if isinstance(tools, list) else ():
        if not isinstance(tool, dict) or _is_server_tool(provider, tool):
            found.add(SERVER_TOOL)
    if _contains_key(data, "cache_control"):
        found.add(CACHE_WRITE)
    return tuple(sorted(found))


def _is_server_tool(provider: ModelProvider, tool: dict[str, object]) -> bool:
    if provider is ModelProvider.GEMINI:
        # Gemini's own tools (search, code execution, URL context) are keys of their own.
        return any(key not in {"functionDeclarations", "function_declarations"} for key in tool)
    if provider is ModelProvider.ANTHROPIC:
        return tool.get("type", "custom") != "custom"
    return tool.get("type") != "function"


def _contains_key(value: object, name: str) -> bool:
    """Whether any object in the parsed body, at any depth, has this key."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if name in item:
                return True
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _integers(sources: list[dict[str, object]], names: tuple[str, ...]) -> list[int] | None:
    """The integer values of the named fields, in every source, that are present and not
    null, or None if one is not a whole number (a boolean, float, string or list is not)."""
    found: list[int] = []
    for source in sources:
        for name in names:
            value = source.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            found.append(value)
    return found


@dataclass(frozen=True)
class ModelExchange:
    """How a model call ended, as the proxy saw it. It decides what the call is charged.

    The four flags have no defaults. A default would be a guess about a call the proxy did
    not look at, and every guess here favours the client: a truncated stream read as complete
    is charged at its price, and a 5xx that produced output read as empty is released.
    """

    # Whether any bytes of the request reached the provider. False only for a call that was
    # refused or failed before it was sent (a rejection, a hook error, a connect error).
    request_sent: bool
    # The provider answered with a 4xx or 5xx.
    provider_error: bool
    # Whether the reply was relayed and read in full.
    complete: bool
    # Whether the provider reported any output tokens, or sent any of a stream.
    produced_output: bool
    # The priced cost of the reply, or None when it cannot be priced. The cost of a provider
    # error is always read as None (see below). A value that is not a
    # finite, non-negative Decimal (a float, NaN, infinity, a negative) is not a usable cost
    # and is read as None, so the call is charged its full reservation. It is not an error:
    # the call was already sent, and an exception here would leave the reservation open.
    cost_usd: Decimal | None = None

    def __post_init__(self) -> None:
        for name in ("request_sent", "provider_error", "complete", "produced_output"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        cost = self.cost_usd
        usable = isinstance(cost, Decimal) and cost.is_finite() and cost >= 0
        # A provider error is priced at zero by Panoptes ("an error reply is not billed"),
        # but one that produced output may have been billed in part. Its cost is unknown.
        if cost is not None and (not usable or self.provider_error):
            object.__setattr__(self, "cost_usd", None)


@dataclass(frozen=True)
class Caps:
    """Caps a process serving one run puts on top of the policy's. They can only tighten."""

    max_usd: Decimal | None = None
    max_calls: int | None = None

    def __post_init__(self) -> None:
        # Types are checked at runtime too: a float would carry binary rounding into the
        # ledger, and 1.5 calls or True calls would pass a bare comparison.
        if self.max_usd is not None:
            if not isinstance(self.max_usd, Decimal):
                raise TypeError("max_usd must be a Decimal")
            if not (self.max_usd.is_finite() and self.max_usd > 0):
                raise ValueError("max_usd must be a positive amount")
        if self.max_calls is not None:
            if isinstance(self.max_calls, bool) or not isinstance(self.max_calls, int):
                raise TypeError("max_calls must be an int")
            if self.max_calls < 1:
                raise ValueError("max_calls must be at least 1")

    def tighten(self, limits: SpendLimits) -> SpendLimits:
        """limits with the run's spend and call caps lowered to these, never raised. The
        customer's day cap is the policy's alone."""
        usd = limits.per_run_usd if self.max_usd is None else min(limits.per_run_usd, self.max_usd)
        calls = (
            limits.max_calls_per_run
            if self.max_calls is None
            else min(limits.max_calls_per_run, self.max_calls)
        )
        return limits.model_copy(update={"per_run_usd": usd, "max_calls_per_run": calls})
