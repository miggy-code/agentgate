"""Built-in tool adapters: strict args models, facts, side-effect tables and read helpers."""

import json
import sqlite3
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from agentgate.clock import FixedClock
from agentgate.context import RunContext
from agentgate.fixtures import load_fixtures
from agentgate.models import Action, Document, Role
from agentgate.policy import addresses
from agentgate.policy import engine as engine_module
from agentgate.policy.engine import Request
from agentgate.policy.schema import Effect, load_bundles
from agentgate.retrieval import DocumentIndex
from agentgate.store import Store
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
from conftest import FIXTURES, JR, KC, POLICIES, World

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


# Recipients: one rule for the engine and the tool

# Internal (kettlecreek.example) addresses as typed, with what must be stored, or None when
# the rule refuses them.
RECIPIENTS: list[tuple[str, str | None]] = [
    ("nadine@kettlecreek.example", "nadine@kettlecreek.example"),
    ("Nadine@KettleCreek.Example", "nadine@kettlecreek.example"),
    ("  nadine@kettlecreek.example ", "nadine@kettlecreek.example"),
    ("nadine.k+jobs@kettlecreek.example", "nadine.k+jobs@kettlecreek.example"),
    (" \tnadine@kettlecreek.example\r\n", None),
    ("nadine@kettlecreek.example\r\n", None),
    ("nadine@kettlecreek.example\nBcc: records@exfil-drop.example", None),
    ("\u202enadine@kettlecreek.example", None),
    ("nadine@kettlecreek.example\u200f", None),
    ("nad\u200bine@kettlecreek.example", None),
    ("records%exfil-drop.example@kettlecreek.example", None),
    ("exfil-drop.example!records@kettlecreek.example", None),
    ('"records@exfil-drop.example"@kettlecreek.example', None),
    ("nadine@\u212aettlecreek.example", None),
    ("\u212aate@kettlecreek.example", None),
    ("nadine@k\u0435ttlecreek.example", None),
    ("nadine@\uff4bettlecreek.example", None),
    ("nadine@kettlecreek.example, records@exfil-drop.example", None),
    ("Nadine <nadine@kettlecreek.example>", None),
    ("nadine@kettlecreek.example@exfil-drop.example", None),
]


def _send_args(*to: str) -> EmailSendArgs:
    args = _parse(EmailSendArgs, {"to": list(to), "subject": "s", "body": "b"})
    assert isinstance(args, EmailSendArgs)
    return args


def test_email_args_normalise_plain_addresses() -> None:
    args = _send_args(" Nadine@KettleCreek.Example ", "BEATRICE.L@mail-sample.example")
    assert args.to == ["nadine@kettlecreek.example", "beatrice.l@mail-sample.example"]
    facts = default_tools()["email.send"].facts(args)
    assert facts.recipients == ("nadine@kettlecreek.example", "beatrice.l@mail-sample.example")


@pytest.mark.parametrize(("raw", "stored"), RECIPIENTS)
def test_the_engine_and_the_email_tool_agree(world: World, raw: str, stored: str | None) -> None:
    assert addresses.normalize_address(raw) == stored
    adapter = default_tools()["email.send"]
    args = _send_args(raw)
    facts = adapter.facts(args)
    # What the engine decides on is what the tool would store.
    parsed = engine_module._parse_address(facts.recipients[0])
    assert (None if parsed is None else parsed[0]) == stored
    request = Request(
        customer_id=KC,
        user_id="kc-staff-nadine",
        role=Role.STAFF,
        tool="email.send",
        action=Action.SEND,
        recipients=facts.recipients,
    )
    decision = world.engines[KC].decide(request)
    if stored is None:
        # Malformed: on no list (so default deny for staff), and the tool refuses it anyway.
        assert decision.effect is Effect.DENY
        with pytest.raises(ValueError):
            adapter.run(_env(world), _ctx(), args)
        assert list_outbox(world.store) == []
    else:
        assert decision.effect is Effect.ALLOW
        assert facts.recipients == (stored,)
        result = adapter.run(_env(world), _ctx(), args)
        assert result.content["to"] == [stored]
        [email] = list_outbox(world.store)
        assert email.recipients == (stored,)


def test_email_send_refuses_a_malformed_recipient_among_good_ones(world: World) -> None:
    adapter = default_tools()["email.send"]
    args = _send_args("terrence@kettlecreek.example", "\u202enadine@kettlecreek.example")
    with pytest.raises(ValueError):
        adapter.run(_env(world), _ctx(), args)
    assert list_outbox(world.store) == []


# notes.write indexes the note alone


def _index_rows(store: Store) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    conn = store.connection
    postings = conn.execute("SELECT * FROM retrieval_postings ORDER BY term, doc_id").fetchall()
    lengths = conn.execute("SELECT * FROM retrieval_lengths ORDER BY doc_id").fetchall()
    return [tuple(r) for r in postings], [tuple(r) for r in lengths]


def test_notes_write_touches_only_the_new_notes_rows(world: World) -> None:
    adapter = default_tools()["notes.write"]
    title, body = "Coil restock", "Restock the coil shelf; coil count is low."
    args = _parse(NotesWriteArgs, {"title": title, "body": body, "label": "internal"})
    terms = {"coil", "restock", "the", "shelf", "count", "is", "low"}
    conn = world.store.connection
    postings_before, lengths_before = _index_rows(world.store)
    changes = conn.total_changes
    result = adapter.run(_env(world), _ctx(), args)
    # One document row, one lengths row and one posting per distinct term, nothing else:
    # no other customer's (or this customer's) rows are deleted or written again.
    assert conn.total_changes - changes == 1 + 1 + len(terms)
    note_id = result.content["document_id"]
    postings_after, lengths_after = _index_rows(world.store)
    assert [r for r in postings_after if r[1] != note_id] == postings_before
    assert [r for r in lengths_after if r[0] != note_id] == lengths_before
    assert {r[0] for r in postings_after if r[1] == note_id} == terms
    # And the index is the one a full rebuild gives.
    world.index.rebuild()
    assert _index_rows(world.store) == (postings_after, lengths_after)


def test_a_note_write_does_not_block_another_connection(tmp_path: Path) -> None:
    """Two stores on one file, as two gateway processes have. One writes notes while the
    other reads and takes the write lock with a short busy timeout. A full index rebuild per
    note held the lock for seconds at a few thousand documents; one note's rows take
    milliseconds, so the other connection never times out."""
    path = tmp_path / "shared.db"
    writer = Store(path)
    load_fixtures(writer, FIXTURES)
    index = DocumentIndex(writer)
    body = " ".join(f"word{i % 997} term{i % 311}" for i in range(150))
    with writer.transaction():
        writer.connection.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(f"jr-bulk-{i}", JR, "Bulk", "internal", "[]", "[]", body) for i in range(1500)],
        )
    index.rebuild()
    reader = Store(path)
    reader.connection.execute("PRAGMA busy_timeout = 250")
    env = ToolEnv(store=writer, index=index, clock=FixedClock(STARTED))
    jr = _ctx(JR, "jr-staff-priya", Role.STAFF)
    adapter = default_tools()["notes.write"]
    notes = 12
    turns = [threading.Event() for _ in range(notes)]
    failures: list[str] = []

    def write_notes() -> None:
        for i in range(notes):
            args = _parse(NotesWriteArgs, {"title": f"n{i}", "body": "shelf", "label": "internal"})
            adapter.run(env, jr, args)
            # Wait for the reader's turn, so its calls land between and during writes.
            turns[i].wait(timeout=30)

    thread = threading.Thread(target=write_notes)
    thread.start()
    completed = 0
    try:
        for i in range(notes):
            try:
                assert reader.get_document("kc-service-menu") is not None
                with reader.transaction():
                    reader.connection.execute("SELECT count(*) FROM documents").fetchone()
                completed += 1
            except sqlite3.OperationalError as exc:
                failures.append(str(exc))
            finally:
                turns[i].set()
    finally:
        thread.join(timeout=60)
        reader.close()
        writer.close()
    assert failures == []
    assert completed == notes
