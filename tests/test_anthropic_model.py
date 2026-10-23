"""The Anthropic adapter, against a fake client and the real SDK types. No network, no key."""

import json
import math
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message as SdkMessage
from anthropic.types.beta import BetaMessage

from agentgate.agent import anthropic_model
from agentgate.agent.anthropic_model import FALLBACK_BETA, AnthropicModel, wire_name
from agentgate.agent.model import (
    AssistantMessage,
    Message,
    ModelError,
    ToolCall,
    ToolOutcome,
    ToolResultsMessage,
    ToolSpec,
    UserMessage,
)
from agentgate.models import Action

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["query"],
    "additionalProperties": False,
}
READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"doc_id": {"type": "string"}},
    "required": ["doc_id"],
    "additionalProperties": False,
}
TOOLS = (
    ToolSpec("docs.search", Action.READ, "Search the customer's documents.", SEARCH_SCHEMA),
    ToolSpec("docs.read", Action.READ, "Read one document by id.", READ_SCHEMA),
)
SYSTEM = "You are a support agent. Use the tools."
SECRET = "confidential warranty terms for customer 42"


def usage(
    input_tokens: int = 10, output_tokens: int = 5, write: int = 0, read: int = 0
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": write,
        "cache_read_input_tokens": read,
    }


def payload(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    *,
    model: str = "claude-opus-5-5",
    stop_details: dict[str, Any] | None = None,
    usage_: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": usage_ or usage(),
    }


def beta_message(content: list[dict[str, Any]], stop_reason: str = "end_turn", **kw: Any) -> Any:
    return BetaMessage.model_validate(payload(content, stop_reason, **kw))


def sdk_message(content: list[dict[str, Any]], stop_reason: str = "end_turn", **kw: Any) -> Any:
    return SdkMessage.model_validate(payload(content, stop_reason, **kw))


THINKING = {"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"}
TEXT = {"type": "text", "text": "Let me look that up."}
TOOL_USES = [
    {"type": "tool_use", "id": "toolu_1", "name": "docs_search", "input": {"query": "warranty"}},
    {"type": "tool_use", "id": "toolu_2", "name": "docs_read", "input": {"doc_id": "doc-1"}},
]


class _Endpoint:
    def __init__(self, client: "FakeClient", endpoint: str) -> None:
        self._client = client
        self._endpoint = endpoint

    def create(self, **kwargs: Any) -> Any:
        self._client.calls.append((self._endpoint, kwargs))
        result = self._client.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _Beta:
    def __init__(self, client: "FakeClient") -> None:
        self.messages = _Endpoint(client, "beta")


class FakeClient:
    """Records every create call and returns canned SDK objects or raises canned errors."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.messages = _Endpoint(self, "messages")
        self.beta = _Beta(self)

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1][1]


def ask(text: str = "What does my warranty cover?") -> list[Message]:
    return [UserMessage(text)]


def test_request_shape_with_fallbacks() -> None:
    client = FakeClient(beta_message([TEXT]))
    AnthropicModel(client=client).complete(SYSTEM, ask(), TOOLS)
    endpoint, kwargs = client.calls[0]
    assert endpoint == "beta"
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] == 2048
    assert kwargs["system"] == SYSTEM
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["fallbacks"] == "default"
    for absent in ("thinking", "temperature", "top_p", "top_k", "tool_choice"):
        assert absent not in kwargs
    assert kwargs["tools"] == [
        {
            "name": "docs_search",
            "description": "Search the customer's documents.",
            "input_schema": SEARCH_SCHEMA,
        },
        {
            "name": "docs_read",
            "description": "Read one document by id.",
            "input_schema": READ_SCHEMA,
        },
    ]
    assert all("strict" not in tool for tool in kwargs["tools"])
    assert kwargs["messages"] == [{"role": "user", "content": "What does my warranty cover?"}]


def test_request_without_fallbacks_or_cache_uses_plain_create() -> None:
    client = FakeClient(sdk_message([TEXT]))
    model = AnthropicModel(client=client, fallbacks=False, cache=False, effort="high")
    model.complete(SYSTEM, ask(), TOOLS)
    endpoint, kwargs = client.calls[0]
    assert endpoint == "messages"
    for absent in ("betas", "fallbacks", "cache_control", "thinking"):
        assert absent not in kwargs
    assert kwargs["output_config"] == {"effort": "high"}


def test_haiku_never_sends_fallbacks() -> None:
    client = FakeClient(sdk_message([TEXT], model="claude-haiku-5-5"))
    AnthropicModel(model="claude-haiku-5-5", client=client).complete(SYSTEM, ask(), TOOLS)
    endpoint, kwargs = client.calls[0]
    assert endpoint == "messages"
    assert "fallbacks" not in kwargs
    assert "betas" not in kwargs


def test_no_tools_and_empty_system_are_omitted() -> None:
    client = FakeClient(beta_message([TEXT]))
    AnthropicModel(client=client).complete("", ask(), ())
    assert "tools" not in client.last
    assert "system" not in client.last


def test_message_conversion_rebuilds_tool_use_and_tool_results() -> None:
    history: list[Message] = [
        UserMessage("Order the part and email the customer."),
        AssistantMessage(
            text="Working on it.",
            tool_calls=(
                ToolCall("c1", "docs.search", {"query": "pump"}),
                ToolCall("c2", "parts.order", {"part": "pump", "quantity": 1}),
                ToolCall("c3", "email.send", {"to": "x@example.com"}),
                ToolCall("c4", "refund.issue", {"order_id": "o1"}),
                ToolCall("c5", "notes.write", {"text": "hi"}),
            ),
        ),
        ToolResultsMessage(
            (
                ToolOutcome("c1", "docs.search", "executed", {"results": [], "count": 0}),
                ToolOutcome("c2", "parts.order", "pending_approval", {"approval_id": "a1"}),
                ToolOutcome("c3", "email.send", "denied", {"reason": "external_after_conf"}),
                ToolOutcome("c4", "refund.issue", "cap_reached", {"reason": "spend_cap"}),
                ToolOutcome("c5", "notes.write", "error", {"reason": "tool_error"}),
            )
        ),
    ]
    client = FakeClient(beta_message([TEXT]))
    AnthropicModel(client=client).complete(SYSTEM, history, TOOLS)
    user, assistant, results = client.last["messages"]
    assert user == {"role": "user", "content": "Order the part and email the customer."}
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "Working on it."}
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "docs_search",
        "input": {"query": "pump"},
    }
    assert [block["name"] for block in assistant["content"][1:]] == [
        "docs_search",
        "parts_order",
        "email_send",
        "refund_issue",
        "notes_write",
    ]
    assert results["role"] == "user"
    blocks = results["content"]
    assert [b["type"] for b in blocks] == ["tool_result"] * 5
    assert [b["tool_use_id"] for b in blocks] == ["c1", "c2", "c3", "c4", "c5"]
    assert [b["is_error"] for b in blocks] == [False, False, True, True, True]
    assert blocks[0]["content"] == '{"count": 0, "results": []}'
    assert blocks[0]["content"] == json.dumps({"results": [], "count": 0}, sort_keys=True)


def test_response_parsing_unmaps_names_and_keeps_native() -> None:
    unknown = {"type": "tool_use", "id": "toolu_3", "name": "shell_exec", "input": {"cmd": "ls"}}
    response = beta_message([THINKING, TEXT, *TOOL_USES, unknown], "tool_use")
    client = FakeClient(response)
    result = AnthropicModel(client=client).complete(SYSTEM, ask(), TOOLS)
    assert result.stop_reason == "tool_use"
    message = result.message
    assert message.text == "Let me look that up."
    assert message.tool_calls == (
        ToolCall("toolu_1", "docs.search", {"query": "warranty"}),
        ToolCall("toolu_2", "docs.read", {"doc_id": "doc-1"}),
        # Not one of the offered tools: passed through so the gate denies it as unknown.
        ToolCall("toolu_3", "shell_exec", {"cmd": "ls"}),
    )
    assert message.native is response.content


def test_text_only_response_and_joined_text() -> None:
    second = {"type": "text", "text": "Done."}
    client = FakeClient(beta_message([THINKING, TEXT, second]))
    message = AnthropicModel(client=client).complete(SYSTEM, ask(), TOOLS).message
    assert message.text == "Let me look that up.\nDone."
    assert message.tool_calls == ()


def test_response_with_no_text_has_none() -> None:
    client = FakeClient(beta_message([THINKING, TOOL_USES[0]], "tool_use"))
    message = AnthropicModel(client=client).complete(SYSTEM, ask(), TOOLS).message
    assert message.text is None
    assert len(message.tool_calls) == 1


def test_native_content_is_sent_back_unchanged() -> None:
    first = beta_message([THINKING, TEXT, *TOOL_USES], "tool_use")
    client = FakeClient(first, beta_message([{"type": "text", "text": "All set."}]))
    model = AnthropicModel(client=client)
    turn = model.complete(SYSTEM, ask(), TOOLS)
    history: list[Message] = [
        *ask(),
        turn.message,
        ToolResultsMessage(
            (
                ToolOutcome("toolu_1", "docs.search", "executed", {"results": []}),
                ToolOutcome("toolu_2", "docs.read", "denied", {"reason": "not_found"}),
            )
        ),
    ]
    model.complete(SYSTEM, history, TOOLS)
    assistant = client.last["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is first.content
    assert assistant["content"][0].type == "thinking"
    assert assistant["content"][0].signature == "c2lnbmF0dXJl"
    results = client.last["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results] == ["toolu_1", "toolu_2"]
    assert [b["is_error"] for b in results] == [False, True]


def test_refusal_ends_with_no_tool_calls() -> None:
    refusal = beta_message(
        [TOOL_USES[0]],
        "refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    result = AnthropicModel(client=FakeClient(refusal)).complete(SYSTEM, ask(), TOOLS)
    assert result.stop_reason == "refusal"
    assert result.message.tool_calls == ()
    assert result.message.text is not None
    assert "cyber" in result.message.text
    assert result.message.native is None


def test_refusal_without_category() -> None:
    refusal = beta_message([], "refusal", stop_details=None)
    result = AnthropicModel(client=FakeClient(refusal)).complete(SYSTEM, ask(), TOOLS)
    assert result.message.text is not None
    assert "unspecified" in result.message.text


def test_max_tokens_drops_tool_calls_and_keeps_text() -> None:
    partial = {"type": "tool_use", "id": "toolu_9", "name": "docs_read", "input": {}}
    cut = beta_message([THINKING, TEXT, TOOL_USES[0], partial], "max_tokens")
    result = AnthropicModel(client=FakeClient(cut)).complete(SYSTEM, ask(), TOOLS)
    assert result.stop_reason == "max_tokens"
    assert result.message.text == "Let me look that up."
    assert result.message.tool_calls == ()
    assert result.message.native is None


def test_cost_from_usage_includes_cache_tokens() -> None:
    response = beta_message([TEXT], usage_=usage(1000, 500, write=2000, read=10_000))
    result = AnthropicModel(client=FakeClient(response)).complete(SYSTEM, ask(), TOOLS)
    # 1000 x $4 + 2000 x $5 (1.25 x input) + 10000 x $0.20 + 500 x $20, per million.
    assert result.cost_usd == Decimal("0.026")
    assert result.usage.input_tokens == 13_000
    assert result.usage.output_tokens == 500


def test_cost_per_model() -> None:
    sonnet = sdk_message([TEXT], model="claude-sonnet-5-5", usage_=usage(1_000_000, 1_000_000))
    model = AnthropicModel(model="claude-sonnet-5-5", client=FakeClient(sonnet), fallbacks=False)
    assert model.complete(SYSTEM, ask(), TOOLS).cost_usd == Decimal(12)
    haiku = sdk_message([TEXT], model="claude-haiku-5-5", usage_=usage(50_000, 1_000_000))
    model = AnthropicModel(model="claude-haiku-5-5", client=FakeClient(haiku))
    assert model.complete(SYSTEM, ask(), TOOLS).cost_usd == Decimal("0.505")


def test_haiku_long_prompt_is_priced_at_the_higher_tier() -> None:
    haiku = sdk_message([TEXT], model="claude-haiku-5-5", usage_=usage(200_000, 0))
    model = AnthropicModel(model="claude-haiku-5-5", client=FakeClient(haiku))
    assert model.complete(SYSTEM, ask(), TOOLS).cost_usd == Decimal("0.1")


def test_fallback_served_cost_prices_each_hop_at_its_model() -> None:
    hops = [
        {
            "type": "message",
            "model": "claude-opus-5-5",
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        {
            "type": "fallback_message",
            "model": "claude-opus-5",
            "input_tokens": 1000,
            "output_tokens": 200,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    ]
    response = beta_message(
        [TEXT], model="claude-opus-5", usage_={**usage(2000, 300), "iterations": hops}
    )
    result = AnthropicModel(client=FakeClient(response)).complete(SYSTEM, ask(), TOOLS)
    # Opus 5.5 hop: 1000 x $4 + 100 x $20. Opus 5 hop: 1000 x $5 + 200 x $25.
    assert result.cost_usd == Decimal("0.016")
    assert result.usage.input_tokens == 2000
    assert result.usage.output_tokens == 300


def test_unknown_served_model_is_priced_at_the_ceiling() -> None:
    response = beta_message([TEXT], model="claude-future-9", usage_=usage(1_000_000, 0))
    result = AnthropicModel(client=FakeClient(response)).complete(SYSTEM, ask(), TOOLS)
    assert result.cost_usd == Decimal(5)


def _long_history() -> list[Message]:
    return [
        UserMessage("Summarise every warranty clause. " * 40),
        AssistantMessage(
            text="Searching.", tool_calls=(ToolCall("c1", "docs.search", {"query": "warranty"}),)
        ),
        ToolResultsMessage(
            (ToolOutcome("c1", "docs.search", "executed", {"body": "Clause text. " * 200}),)
        ),
    ]


@pytest.mark.parametrize("fallbacks", [True, False])
@pytest.mark.parametrize("model_id", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
def test_estimate_exceeds_a_max_tokens_response(fallbacks: bool, model_id: str) -> None:
    history = _long_history()
    # A generous true input count: every character of content the request carries at
    # three per token, all of it billed as a cache write, and the full max_tokens of output.
    visible = len(SYSTEM)
    for message in history:
        if isinstance(message, UserMessage):
            visible += len(message.text)
        elif isinstance(message, AssistantMessage):
            visible += len(message.text or "")
            visible += sum(len(json.dumps(call.args)) for call in message.tool_calls)
        else:
            visible += sum(len(json.dumps(r.content)) for r in message.results)
    visible += sum(len(json.dumps(t.input_schema)) + len(t.description) for t in TOOLS)
    prompt = math.ceil(visible / 3)
    response = beta_message(
        [TEXT], "max_tokens", model=model_id, usage_=usage(0, 2048, write=prompt)
    )
    model = AnthropicModel(model=model_id, client=FakeClient(response), fallbacks=fallbacks)
    estimate = model.estimate_cost(SYSTEM, history, TOOLS)
    actual = model.complete(SYSTEM, history, TOOLS).cost_usd
    assert estimate > actual > 0


def test_estimate_reserves_both_hops_when_fallbacks_are_on() -> None:
    client = FakeClient()
    with_fallbacks = AnthropicModel(client=client).estimate_cost(SYSTEM, ask(), TOOLS)
    without = AnthropicModel(client=client, fallbacks=False).estimate_cost(SYSTEM, ask(), TOOLS)
    # The fallback hop is reserved at Claude Opus 5's higher rates.
    assert with_fallbacks - without > 2048 * Decimal(25) / 1_000_000
    assert without > 2048 * Decimal(20) / 1_000_000
    assert client.calls == []


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    body = {"type": "error", "error": {"type": "x", "message": SECRET}}
    return cls(SECRET, response=httpx2.Response(code, request=_request()), body=body)


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: _status(anthropic.RateLimitError, 429), "RateLimitError (status 429)"),
        (lambda: _status(anthropic.InternalServerError, 500), "InternalServerError (status 500)"),
        (lambda: _status(anthropic.BadRequestError, 400), "BadRequestError (status 400)"),
        (lambda: anthropic.APITimeoutError(_request()), "APITimeoutError"),
        (
            lambda: anthropic.APIConnectionError(message=SECRET, request=_request()),
            "APIConnectionError",
        ),
        (lambda: anthropic.AnthropicError(SECRET), "AnthropicError"),
    ],
)
def test_sdk_errors_become_model_error_without_content(
    make: Callable[[], BaseException], expected: str
) -> None:
    model = AnthropicModel(client=FakeClient(make()))
    with pytest.raises(ModelError) as info:
        model.complete(SYSTEM, [UserMessage(SECRET)], TOOLS)
    message = str(info.value)
    assert expected in message
    assert SECRET not in message
    assert "warranty" not in message
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTGATE_ANTHROPIC_MODEL", raising=False)
    assert AnthropicModel.from_env(client=FakeClient()).name == "claude-opus-5-5"
    monkeypatch.setenv("AGENTGATE_ANTHROPIC_MODEL", "claude-sonnet-5-5")
    assert AnthropicModel.from_env(client=FakeClient()).name == "claude-sonnet-5-5"
    monkeypatch.setenv("AGENTGATE_ANTHROPIC_MODEL", "claude-opus-5-5-20260401")
    with pytest.raises(ValueError, match="unknown Anthropic model"):
        AnthropicModel.from_env()


@pytest.mark.parametrize("model_id", ["claude-opus-4-8", "gpt-5", "", "claude-opus-5-5-20260401"])
def test_unknown_model_id_is_rejected(model_id: str) -> None:
    with pytest.raises(ValueError, match="unknown Anthropic model"):
        AnthropicModel(model=model_id)


def test_bad_effort_and_max_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="effort"):
        AnthropicModel(effort="extreme")
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicModel(max_tokens=0)


def test_injected_client_is_used_and_none_is_created(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> None:
        raise AssertionError("a client was created although one was injected")

    monkeypatch.setattr(anthropic_model.anthropic, "Anthropic", forbidden)
    client = FakeClient(beta_message([TEXT]))
    AnthropicModel(client=client).complete(SYSTEM, ask(), TOOLS)
    assert len(client.calls) == 1


def test_client_is_created_lazily_once(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    created: list[FakeClient] = []

    def factory() -> FakeClient:
        client = FakeClient(beta_message([TEXT]), beta_message([TEXT]))
        created.append(client)
        return client

    monkeypatch.setattr(anthropic_model.anthropic, "Anthropic", factory)
    model = AnthropicModel()
    assert model.estimate_cost(SYSTEM, ask(), TOOLS) > 0
    assert created == []
    model.complete(SYSTEM, ask(), TOOLS)
    model.complete(SYSTEM, ask(), TOOLS)
    assert len(created) == 1
    assert len(created[0].calls) == 2


def test_missing_credentials_become_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_credentials() -> None:
        raise TypeError('"Could not resolve authentication method."')

    monkeypatch.setattr(anthropic_model.anthropic, "Anthropic", no_credentials)
    with pytest.raises(ModelError, match="TypeError; check Anthropic credentials"):
        AnthropicModel().complete(SYSTEM, ask(), TOOLS)


def test_wire_names() -> None:
    assert wire_name("docs.search") == "docs_search"
    assert wire_name("refund.issue") == "refund_issue"
    assert len(wire_name("x." * 50)) == 64
    with pytest.raises(ModelError):
        wire_name("")


def test_colliding_tool_names_are_refused() -> None:
    tools = (*TOOLS, ToolSpec("docs_search", Action.READ, "Shadow.", SEARCH_SCHEMA))
    client = FakeClient(beta_message([TEXT]))
    with pytest.raises(ModelError, match="same provider tool name"):
        AnthropicModel(client=client).complete(SYSTEM, ask(), tools)
    assert client.calls == []


def test_real_sdk_accepts_the_request_offline() -> None:
    """The kwargs go through the real SDK and serialise as the API expects. The transport is
    a local mock, so nothing leaves the process and the key is a placeholder."""
    sent: list[httpx2.Request] = []
    replies = [
        payload([THINKING, TEXT, TOOL_USES[0]], "tool_use"),
        payload([{"type": "text", "text": "Done."}]),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=replies[len(sent) - 1])

    client = anthropic.Anthropic(
        api_key="placeholder-not-a-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    model = AnthropicModel(client=client)
    first = model.complete(SYSTEM, ask(), TOOLS)
    history: list[Message] = [
        *ask(),
        first.message,
        ToolResultsMessage((ToolOutcome("toolu_1", "docs.search", "executed", {"n": 0}),)),
    ]
    model.complete(SYSTEM, history, TOOLS)

    assert sent[0].url.path == "/v1/messages"
    assert sent[0].headers["anthropic-beta"] == FALLBACK_BETA
    body = json.loads(sent[0].content)
    assert body["fallbacks"] == "default"
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["tools"][0]["name"] == "docs_search"
    assert "thinking" not in body
    second = json.loads(sent[1].content)
    assert second["messages"][1]["content"] == [THINKING, TEXT, TOOL_USES[0]]
    assert second["messages"][2]["content"][0]["type"] == "tool_result"
