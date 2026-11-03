"""Reading a model request strictly, and the caps a serving process may tighten."""

import json
from decimal import Decimal
from typing import Any

import pytest

from agentgate.model.admission import MAX_BODY_BYTES, Caps, ModelCall, parse_model_call
from agentgate.policy.schema import ModelProvider, SpendLimits

OPENAI = ModelProvider.OPENAI_COMPATIBLE


def _body(**fields: Any) -> bytes:
    return json.dumps({"model": "deepseek-flash", "messages": [], **fields}).encode()


def _parse(body: bytes, provider: ModelProvider = OPENAI, **kwargs: Any) -> ModelCall:
    return parse_model_call(provider, body, **kwargs)


def test_a_plain_request_is_read() -> None:
    body = _body(max_tokens=4000)
    call = _parse(body, correlation_id="trial-7")
    assert call == ModelCall(OPENAI, "deepseek-flash", 4000, None, len(body), "trial-7")


def test_a_request_with_no_limit_declares_none() -> None:
    call = _parse(_body())
    assert call.valid and call.max_tokens is None and call.n is None


def test_null_fields_are_not_declarations() -> None:
    call = _parse(_body(max_tokens=None, n=None))
    assert call.valid and call.max_tokens is None and call.n is None


def test_the_largest_declared_limit_is_the_limit() -> None:
    call = _parse(_body(max_tokens=10, max_completion_tokens=999_999, max_output_tokens=5))
    assert call.max_tokens == 999_999


def test_the_completions_asked_for_are_read() -> None:
    assert _parse(_body(max_tokens=1, n=3)).n == 3
    assert _parse(_body(max_tokens=1, n=1)).n == 1


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[]",
        b'"model"',
        b"null",
        b"123",
        b"\xff\xfe",
        b'{"model": "a", "model": "b"}',
        b'{"model": "a", "max_tokens": 1, "max_tokens": 9999}',
        b'{"model": "a", "messages": {"x": 1, "x": 2}}',
        b'{"model": "a"',
        b"[" * 100_000,
        b'{"a":' * 100_000,
        b'{"model": 5}',
        b'{"model": null}',
        b'{"model": ["a"]}',
        b"{}",
        b'{"model": "a", "max_tokens": "100"}',
        b'{"model": "a", "max_tokens": 1.5}',
        b'{"model": "a", "max_tokens": 100.0}',
        b'{"model": "a", "max_tokens": true}',
        b'{"model": "a", "max_tokens": [100]}',
        b'{"model": "a", "max_tokens": NaN}',
        b'{"model": "a", "max_tokens": Infinity}',
        b'{"model": "a", "max_tokens": 1e999}',
        b'{"model": "a", "max_completion_tokens": "x", "max_tokens": 5}',
        b'{"model": "a", "max_tokens": 5, "n": "2"}',
        b'{"model": "a", "max_tokens": 5, "n": 1.0}',
        b'{"model": "a", "max_tokens": 5, "n": false}',
        b'{"model": "a", "max_tokens": ' + b"9" * 5000 + b"}",
    ],
)
def test_a_body_that_cannot_be_read_strictly_is_invalid(body: bytes) -> None:
    call = _parse(body)
    assert not call.valid and call.model == "" and call.max_tokens is None
    assert call.request_bytes == len(body)


def test_a_body_over_the_read_limit_is_invalid_without_being_parsed() -> None:
    assert not _parse(b" " * (MAX_BODY_BYTES + 1)).valid


def test_the_model_is_taken_exactly_as_sent() -> None:
    for name in ("deepseek-flash ", "DeepSeek-Flash", "deepseek-flash\u200b", ""):
        call = _parse(_body(model=name, max_tokens=1))
        assert call.valid and call.model == name


def test_an_anthropic_request_reads_max_tokens_only() -> None:
    body = _body(max_tokens=100, max_completion_tokens=1, n=5)
    call = _parse(body, ModelProvider.ANTHROPIC)
    assert call.valid and call.max_tokens == 100 and call.n is None


def test_gemini_names_its_model_in_the_path_and_its_limits_in_the_config() -> None:
    body = json.dumps(
        {
            "model": "ignored-body-model",
            "generationConfig": {"maxOutputTokens": 50, "candidateCount": 1},
            "generation_config": {"max_output_tokens": 900},
        }
    ).encode()
    call = _parse(body, ModelProvider.GEMINI, path_model="gemini-2.5-flash")
    assert call.valid and call.model == "gemini-2.5-flash"
    assert call.max_tokens == 900 and call.n == 1


def test_gemini_without_a_path_model_or_with_a_bad_config_is_invalid() -> None:
    body = json.dumps({"generationConfig": {"maxOutputTokens": 5}}).encode()
    assert not _parse(body, ModelProvider.GEMINI).valid
    bad = json.dumps({"generationConfig": [1]}).encode()
    assert not _parse(bad, ModelProvider.GEMINI, path_model="m").valid
    two = json.dumps(
        {"generationConfig": {"candidateCount": 1}, "generation_config": {"candidate_count": 2}}
    )
    assert not _parse(two.encode(), ModelProvider.GEMINI, path_model="m").valid


# Caps


def _limits(run: str = "0.50", calls: int = 40) -> SpendLimits:
    return SpendLimits(
        per_run_usd=Decimal(run), per_customer_day_usd=Decimal("20"), max_calls_per_run=calls
    )


def test_caps_lower_the_policys_limits() -> None:
    tight = Caps(max_usd=Decimal("0.10"), max_calls=3).tighten(_limits())
    assert tight.per_run_usd == Decimal("0.10") and tight.max_calls_per_run == 3
    assert tight.per_customer_day_usd == Decimal("20")


def test_caps_never_raise_them() -> None:
    wide = Caps(max_usd=Decimal("100"), max_calls=10_000).tighten(_limits())
    assert wide == _limits()


def test_no_caps_leave_the_limits_alone() -> None:
    assert Caps().tighten(_limits()) == _limits()


def test_caps_equal_to_the_limits_change_nothing() -> None:
    assert Caps(max_usd=Decimal("0.50"), max_calls=40).tighten(_limits()) == _limits()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_usd": Decimal(0)},
        {"max_usd": Decimal("-1")},
        {"max_usd": Decimal("nan")},
        {"max_usd": Decimal("inf")},
        {"max_calls": 0},
        {"max_calls": -1},
        {"max_calls": True},
    ],
)
def test_invalid_caps_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Caps(**kwargs)


# Fields the parser does not read


@pytest.mark.parametrize(
    "extra",
    [
        {"MAX_TOKENS": 100_000},
        {"Max_Tokens": 100_000},
        {"max_tokens ": 100_000},
        {"max_tokens\u200b": 100_000},
        {"\uff4dax_tokens": 100_000},
        {"best_of": 20},
        {"models": ["gpt-5"]},
        {"route": "fallback"},
        {"provider": {"order": ["x"]}},
        {"service_tier": "priority"},
        {"prompt_cache_retention": "24h"},
    ],
)
def test_a_field_the_provider_list_does_not_name_is_counted(extra: dict[str, Any]) -> None:
    call = _parse(_body(max_tokens=10, **extra))
    assert call.valid and call.max_tokens == 10 and call.unknown_fields == 1


def test_every_unknown_field_is_counted_and_known_ones_are_not() -> None:
    call = _parse(_body(max_tokens=10, temperature=0.1, stream=False, a=1, b=2, c=3))
    assert call.unknown_fields == 3


@pytest.mark.parametrize(
    ("provider", "body"),
    [
        (ModelProvider.ANTHROPIC, {"model": "m", "max_tokens": 5, "n": 2}),
        (ModelProvider.ANTHROPIC, {"model": "m", "max_tokens": 5, "max_completion_tokens": 9}),
        (ModelProvider.GEMINI, {"contents": [], "model": "m"}),
        (ModelProvider.GEMINI, {"contents": [], "max_tokens": 5}),
    ],
)
def test_each_provider_has_its_own_list(provider: ModelProvider, body: dict[str, Any]) -> None:
    call = parse_model_call(provider, json.dumps(body).encode(), path_model="m")
    assert call.valid and call.unknown_fields == 1


def test_the_known_fields_of_each_provider_are_not_counted() -> None:
    anthropic = {"model": "m", "max_tokens": 5, "messages": [], "system": "s", "tools": []}
    gemini = {"contents": [], "generationConfig": {"maxOutputTokens": 5}, "tools": []}
    assert (
        parse_model_call(ModelProvider.ANTHROPIC, json.dumps(anthropic).encode()).unknown_fields
        == 0
    )
    assert (
        parse_model_call(
            ModelProvider.GEMINI, json.dumps(gemini).encode(), path_model="m"
        ).unknown_fields
        == 0
    )


@pytest.mark.parametrize(
    "text",
    [
        '{"model": "m", "max_tokens": 5}',
    ],
)
@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-8-sig"])
def test_only_utf8_is_read(text: str, encoding: str) -> None:
    assert _parse(text.encode(encoding)).valid is False
    assert _parse(text.encode("utf-8")).valid is True


def test_a_deep_but_well_formed_body_hits_the_recursion_limit() -> None:
    # The premise of the red-team case: this body is valid JSON that the parser cannot read.
    deep = b'{"model":"m","max_tokens":1,"messages":' + b"[" * 100_000 + b"]" * 100_000 + b"}"
    with pytest.raises(RecursionError):
        json.loads(deep)
    call = _parse(deep)
    assert not call.valid


@pytest.mark.parametrize(
    ("provider", "fields", "expected"),
    [
        (OPENAI, {"previous_response_id": "r"}, ("server_state",)),
        (OPENAI, {"conversation": "c"}, ("server_state",)),
        (OPENAI, {"background": True}, ("server_state",)),
        (OPENAI, {"tools": [{"type": "web_search"}]}, ("server_tool",)),
        (OPENAI, {"tools": [{"type": "function", "function": {}}]}, ()),
        (OPENAI, {"tools": [{"function": {}}]}, ("server_tool",)),
        (OPENAI, {"tools": "web_search"}, ("server_tool",)),
        (OPENAI, {"tools": ["web_search"]}, ("server_tool",)),
        (OPENAI, {"messages": [{"c": [{"cache_control": {}}]}]}, ("cache_write",)),
        (
            OPENAI,
            {"previous_response_id": "r", "tools": [{"type": "mcp"}]},
            ("server_state", "server_tool"),
        ),
        (OPENAI, {"tools": None}, ()),
    ],
)
def test_server_features_are_seen(
    provider: ModelProvider, fields: dict[str, Any], expected: tuple[str, ...]
) -> None:
    assert _parse(_body(max_tokens=5, **fields), provider).server_features == expected


def test_server_features_of_the_other_providers() -> None:
    def seen(provider: ModelProvider, body: dict[str, Any]) -> tuple[str, ...]:
        return parse_model_call(provider, json.dumps(body).encode(), path_model="m").server_features

    custom = {"name": "t", "input_schema": {}}
    assert seen(ModelProvider.ANTHROPIC, {"model": "m", "max_tokens": 1, "tools": [custom]}) == ()
    typed = {"type": "custom", **custom}
    assert seen(ModelProvider.ANTHROPIC, {"model": "m", "max_tokens": 1, "tools": [typed]}) == ()
    search = {"type": "web_search_20250305", "name": "web_search"}
    assert seen(ModelProvider.ANTHROPIC, {"model": "m", "max_tokens": 1, "tools": [search]}) == (
        "server_tool",
    )
    assert seen(ModelProvider.ANTHROPIC, {"model": "m", "max_tokens": 1, "mcp_servers": []}) == (
        "server_state",
    )
    system = [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
    assert seen(ModelProvider.ANTHROPIC, {"model": "m", "max_tokens": 1, "system": system}) == (
        "cache_write",
    )
    functions = {"functionDeclarations": [{"name": "f"}]}
    assert seen(ModelProvider.GEMINI, {"contents": [], "tools": [functions]}) == ()
    assert seen(ModelProvider.GEMINI, {"contents": [], "tools": [{"googleSearch": {}}]}) == (
        "server_tool",
    )
    assert seen(ModelProvider.GEMINI, {"contents": [], "cachedContent": "c"}) == ("server_state",)
