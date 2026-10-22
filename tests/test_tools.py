"""Built-in tool adapters: strict args models, facts, side-effect tables and read helpers."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from agentgate.context import RunContext
from agentgate.models import Document, Role
from agentgate.policy.schema import load_bundles
from agentgate.tools import ToolEnv, ToolFacts, default_tools
from agentgate.tools.builtin import (
    MAX_BODY_CHARS,
    DocsReadArgs,
    DocsSearchArgs,
    EmailSendArgs,
    NotesWriteArgs,
    PartsOrderArgs,
    RefundIssueArgs,
    list_outbox,
    list_parts_orders,
    list_refunds,
)
from conftest import JR, KC, POLICIES, World

STARTED = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)
ARGS_MODELS: list[type[BaseModel]] = [
    DocsSearchArgs,
    DocsReadArgs,
    NotesWriteArgs,
    EmailSendArgs,
    PartsOrderArgs,
    RefundIssueArgs,
]


def _ctx(customer_id: str = KC, user_id: str = "kc-mgr-marisol", role: Role = Role.MANAGER) -> Any:
    return RunContext(
        run_id="run-test", customer_id=customer_id, user_id=user_id, role=role, started_at=STARTED
    )


def _env(world: World) -> ToolEnv:
    return ToolEnv(store=world.store, index=world.index, clock=world.clock)


def _parse(model: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    return model.model_validate_json(json.dumps(data), strict=True)


# Registry


def test_default_tools_cover_every_configured_tool() -> None:
    tools = default_tools()
    configured = {tool_id for b in load_bundles(POLICIES).values() for tool_id in b.tools}
    assert set(tools) == configured
    for tool_id, adapter in tools.items():
        assert adapter.tool_id == tool_id


@pytest.mark.parametrize("model", ARGS_MODELS)
def test_args_models_are_strict_frozen_and_closed(model: type[BaseModel]) -> None:
    config = model.model_config
    assert config.get("strict") is True
    assert config.get("frozen") is True
    assert config.get("extra") == "forbid"


# Args validation


@pytest.mark.parametrize(
    "data",
    [
        {"query": ""},
        {"query": "x" * 501},
        {"query": "ok", "limit": 0},
        {"query": "ok", "limit": 21},
        {"query": "ok", "limit": "5"},
        {"query": "ok", "limit": True},
        {"query": "ok", "limit": 5.0},
        {"query": "ok", "extra": 1},
        {"query": 7},
    ],
)
def test_docs_search_args_reject(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _parse(DocsSearchArgs, data)


def test_docs_search_args_default_limit() -> None:
    args = _parse(DocsSearchArgs, {"query": "furnace"})
    assert isinstance(args, DocsSearchArgs)
    assert args.limit == 5


@pytest.mark.parametrize(
    "doc_id", ["", "KC-CUSTOMER-LIST", "AGCANARY-3F9A0C71B2D4", "a b", "x" * 65]
)
def test_docs_read_args_take_only_slug_ids(doc_id: str) -> None:
    with pytest.raises(ValidationError):
        _parse(DocsReadArgs, {"document_id": doc_id})


@pytest.mark.parametrize(
    "data",
    [
        {"to": [], "subject": "s", "body": "b"},
        {"to": [f"u{i}@kettlecreek.example" for i in range(21)], "subject": "s", "body": "b"},
        {"to": "a@kettlecreek.example", "subject": "s", "body": "b"},
        {"to": ["a@kettlecreek.example"], "subject": "", "body": "b"},
        {"to": ["a@kettlecreek.example"], "subject": "s" * 301, "body": "b"},
        {"to": ["a@kettlecreek.example"], "subject": "s", "body": "b", "cc": ["x@y.example"]},
    ],
)
def test_email_args_reject(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _parse(EmailSendArgs, data)


@pytest.mark.parametrize(
    "data",
    [
        {"part": "igniter", "quantity": 0, "amount_usd": "10.00"},
        {"part": "igniter", "quantity": 1, "amount_usd": "0"},
        {"part": "igniter", "quantity": 1, "amount_usd": "-5.00"},
        {"part": "igniter", "quantity": 1, "amount_usd": "1.001"},
        {"part": "igniter", "quantity": 1, "amount_usd": "NaN"},
        {"part": "igniter", "quantity": "1", "amount_usd": "10.00"},
        {"part": "", "quantity": 1, "amount_usd": "10.00"},
    ],
)
def test_parts_order_args_reject(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _parse(PartsOrderArgs, data)


def test_money_parses_exactly() -> None:
    args = _parse(PartsOrderArgs, {"part": "igniter", "quantity": 2, "amount_usd": "500.01"})
    assert isinstance(args, PartsOrderArgs)
    assert args.amount_usd == Decimal("500.01")


@pytest.mark.parametrize("label", ["Confidential", "", "x" * 33])
def test_notes_args_reject_bad_labels(label: str) -> None:
    with pytest.raises(ValidationError):
        _parse(NotesWriteArgs, {"title": "t", "body": "b", "label": label})


def test_refund_args_reject_long_reason() -> None:
    with pytest.raises(ValidationError):
        _parse(RefundIssueArgs, {"order_id": "o-1", "amount_usd": "5.00", "reason": "r" * 501})


# Facts


def test_facts() -> None:
    tools = default_tools()
    cases: list[tuple[str, dict[str, Any], ToolFacts]] = [
        ("docs.search", {"query": "x"}, ToolFacts()),
        ("docs.read", {"document_id": "kc-service-menu"}, ToolFacts(document_id="kc-service-menu")),
        (
            "notes.write",
            {"title": "t", "body": "b", "label": "internal"},
            ToolFacts(write_label="internal"),
        ),
        (
            "email.send",
            {"to": ["a@x.example", "b@y.example"], "subject": "s", "body": "b"},
            ToolFacts(recipients=("a@x.example", "b@y.example")),
        ),
        (
            "parts.order",
            {"part": "p", "quantity": 1, "amount_usd": "12.50"},
            ToolFacts(amount_usd=Decimal("12.50")),
        ),
        (
            "refund.issue",
            {"order_id": "o-1", "amount_usd": "9.99", "reason": "late"},
            ToolFacts(amount_usd=Decimal("9.99")),
        ),
    ]
    for tool_id, data, expected in cases:
        adapter = tools[tool_id]
        assert adapter.facts(_parse(adapter.args_model, data)) == expected


def test_adapter_refuses_args_of_another_tool() -> None:
    tools = default_tools()
    wrong = _parse(DocsSearchArgs, {"query": "x"})
    with pytest.raises(TypeError):
        tools["docs.read"].facts(wrong)


# Runs


def test_docs_search_refuses_to_run_without_the_gate_search(world: World) -> None:
    adapter = default_tools()["docs.search"]
    with pytest.raises(RuntimeError):
        adapter.run(_env(world), _ctx(), _parse(DocsSearchArgs, {"query": "furnace"}))


def test_docs_read_returns_fields_and_truncates_bodies(world: World) -> None:
    long_doc = Document(
        id="kc-long-doc",
        customer_id=KC,
        title="Long document",
        label="public",
        body="word " * 2000,
    )
    world.store.add_document(long_doc)
    adapter = default_tools()["docs.read"]
    result = adapter.run(_env(world), _ctx(), _parse(DocsReadArgs, {"document_id": "kc-long-doc"}))
    assert set(result.content) == {"id", "title", "label", "body"}
    body = result.content["body"]
    assert isinstance(body, str) and len(body) == MAX_BODY_CHARS
    assert result.returned_labels == ("public",)
    assert result.cost_usd == 0


def test_docs_read_rechecks_the_tenant(world: World) -> None:
    adapter = default_tools()["docs.read"]
    args = _parse(DocsReadArgs, {"document_id": "jr-return-policy"})
    with pytest.raises(LookupError):
        adapter.run(_env(world), _ctx(), args)


def test_notes_write_creates_an_indexed_note(world: World) -> None:
    adapter = default_tools()["notes.write"]
    args = _parse(
        NotesWriteArgs,
        {"title": "Zephyrine coil", "body": "Zephyrine coil notes.", "label": "internal"},
    )
    result = adapter.run(_env(world), _ctx(), args)
    note_id = result.content["document_id"]
    assert isinstance(note_id, str)
    assert note_id.startswith("note-") and len(note_id) == len("note-") + 12
    note = world.store.get_document(note_id)
    assert note is not None
    assert note.customer_id == KC
    assert note.tags == frozenset({"note"})
    assert note.label == "internal"
    found = world.index.search(world.engines[KC], _ctx(), "zephyrine", run_max_label=None)
    assert [doc.id for doc in found.documents] == [note_id]


def test_email_send_only_queues_to_the_outbox(world: World) -> None:
    adapter = default_tools()["email.send"]
    args = _parse(
        EmailSendArgs,
        {"to": ["beatrice.l@mail-sample.example"], "subject": "Visit", "body": "See you Monday."},
    )
    result = adapter.run(_env(world), _ctx(), args)
    assert result.content["status"] == "queued"
    [email] = list_outbox(world.store, KC)
    assert email.recipients == ("beatrice.l@mail-sample.example",)
    assert (email.subject, email.body, email.run_id) == ("Visit", "See you Monday.", "run-test")
    assert list_outbox(world.store, JR) == []
    assert list_outbox(world.store) == [email]


def test_parts_and_refund_rows(world: World) -> None:
    tools = default_tools()
    tools["parts.order"].run(
        _env(world),
        _ctx(),
        _parse(PartsOrderArgs, {"part": "igniter", "quantity": 2, "amount_usd": "84.20"}),
    )
    jr = _ctx(JR, "jr-mgr-daniel")
    tools["refund.issue"].run(
        _env(world),
        jr,
        _parse(RefundIssueArgs, {"order_id": "o-7", "amount_usd": "12.00", "reason": "damaged"}),
    )
    [order] = list_parts_orders(world.store, KC)
    assert (order.part, order.quantity, order.amount_usd) == ("igniter", 2, Decimal("84.20"))
    [refund] = list_refunds(world.store, JR)
    assert (refund.order_id, refund.amount_usd, refund.reason) == (
        "o-7",
        Decimal("12.00"),
        "damaged",
    )
    assert list_parts_orders(world.store, JR) == []
    assert list_refunds(world.store, KC) == []


def test_read_helpers_work_before_any_write(world: World) -> None:
    assert list_outbox(world.store) == []
    assert list_parts_orders(world.store) == []
    assert list_refunds(world.store) == []
