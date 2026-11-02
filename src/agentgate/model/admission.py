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
"""

import json
from dataclasses import dataclass
from decimal import Decimal

from agentgate._json import no_duplicate_keys
from agentgate.policy.schema import ModelProvider, SpendLimits

# A body larger than this is not read at all. The policy's max_request_bytes cannot exceed it.
MAX_BODY_BYTES = 16 * 1024 * 1024

# The body fields that declare an output limit, per provider. Gemini's sit under
# generationConfig, which the REST API reads in either spelling.
_LIMIT_FIELDS: dict[ModelProvider, tuple[str, ...]] = {
    ModelProvider.ANTHROPIC: ("max_tokens",),
    ModelProvider.OPENAI: ("max_tokens", "max_completion_tokens", "max_output_tokens"),
    ModelProvider.OPENAI_COMPATIBLE: ("max_tokens", "max_completion_tokens", "max_output_tokens"),
    ModelProvider.GEMINI: ("maxOutputTokens", "max_output_tokens"),
}
_COUNT_FIELDS: dict[ModelProvider, tuple[str, ...]] = {
    ModelProvider.ANTHROPIC: (),
    ModelProvider.OPENAI: ("n",),
    ModelProvider.OPENAI_COMPATIBLE: ("n",),
    ModelProvider.GEMINI: ("candidateCount", "candidate_count"),
}


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
        data = json.loads(body, object_pairs_hook=no_duplicate_keys)
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

    limits = _integers(sources, _LIMIT_FIELDS[provider])
    counts = _integers(sources, _COUNT_FIELDS[provider])
    if limits is None or counts is None or len(set(counts)) > 1:
        return invalid
    return ModelCall(
        provider=provider,
        model=model,
        max_tokens=max(limits) if limits else None,
        n=counts[0] if counts else None,
        request_bytes=len(body),
        correlation_id=correlation_id,
    )


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
class Caps:
    """Caps a process serving one run puts on top of the policy's. They can only tighten."""

    max_usd: Decimal | None = None
    max_calls: int | None = None

    def __post_init__(self) -> None:
        if self.max_usd is not None and not (self.max_usd.is_finite() and self.max_usd > 0):
            raise ValueError("max_usd must be a positive amount")
        if self.max_calls is not None and (isinstance(self.max_calls, bool) or self.max_calls < 1):
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
