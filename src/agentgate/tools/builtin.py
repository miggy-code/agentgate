"""Built-in tools: document search and read, notes, email, parts orders and refunds.

Side effects stay inside the store. Email is never really sent: each message becomes one
row in the `outbox` table, and parts orders and refunds are rows in `parts_orders` and
`refunds`. Each tool creates its table on first use. The list_* helpers read those tables
back for tests and for the red-team leak detector.

Every args model is strict (no coercion from strings or floats into ints, no bools as
numbers), frozen, rejects unknown fields and bounds every string, so a call either matches
the documented shape exactly or is refused before any decision.

Recipients follow the shared address rule (agentgate.policy.addresses). A plain address is
normalised during validation (spaces trimmed, lowercased), so the facts the engine decides on,
the outbox row and what a mail adapter would send are the same string. A malformed recipient
is kept as given so the engine decides on it (it is external and on no list, so a policy
denies it and the audit log names the rule), and the tool refuses to store or send it even if
a policy allowed it.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue

from agentgate.context import RunContext
from agentgate.models import Document, Label, Slug
from agentgate.policy import addresses
from agentgate.store import Store
from agentgate.tools.base import ToolAdapter, ToolEnv, ToolFacts, ToolResult

# Bodies handed to the model are cut here, so one huge document cannot flood the context.
MAX_BODY_CHARS = 4000
MAX_TEXT_CHARS = 20_000
# The tag every note carries.
NOTE_TAG = "note"

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS outbox (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        customer_id TEXT NOT NULL,
        recipients TEXT NOT NULL,
        subject TEXT NOT NULL,
        body TEXT NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS parts_orders (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        customer_id TEXT NOT NULL,
        part TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        amount_usd TEXT NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS refunds (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        customer_id TEXT NOT NULL,
        order_id TEXT NOT NULL,
        amount_usd TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL
    ) STRICT
    """,
)

Title = Annotated[str, Field(min_length=1, max_length=300)]
Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]


def _normalised_recipient(raw: str) -> str:
    # Called through the module so the rule has one definition, also for tests that swap it.
    address = addresses.normalize_address(raw)
    return raw if address is None else address


Recipient = Annotated[
    str, Field(min_length=3, max_length=254), AfterValidator(_normalised_recipient)
]
# Money in cents. Zero and negative amounts are never a legitimate order or refund.
Usd = Annotated[Decimal, Field(gt=0, max_digits=12, decimal_places=2, allow_inf_nan=False)]


class _Args(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class DocsSearchArgs(_Args):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class DocsReadArgs(_Args):
    # Lowercase slug only, like stored ids, so a probe cannot carry arbitrary text.
    document_id: Slug


class NotesWriteArgs(_Args):
    title: Title
    body: Text
    label: Label


class EmailSendArgs(_Args):
    to: list[Recipient] = Field(min_length=1, max_length=20)
    subject: Title
    body: Text


class PartsOrderArgs(_Args):
    part: str = Field(min_length=1, max_length=200)
    quantity: int = Field(ge=1, le=10_000)
    amount_usd: Usd


class RefundIssueArgs(_Args):
    order_id: str = Field(min_length=1, max_length=64)
    amount_usd: Usd
    reason: str = Field(min_length=1, max_length=500)


class _Adapter[A: BaseModel]:
    """Shared plumbing: narrow the validated args to this tool's model, then dispatch."""

    tool_id: str
    args_model: type[A]

    def facts(self, args: BaseModel) -> ToolFacts:
        return self._facts(self._narrow(args))

    def run(self, env: ToolEnv, ctx: RunContext, args: BaseModel) -> ToolResult:
        return self._run(env, ctx, self._narrow(args))

    def _narrow(self, args: BaseModel) -> A:
        if not isinstance(args, self.args_model):
            raise TypeError(f"{self.tool_id} expects {self.args_model.__name__} args")
        return args

    def _facts(self, args: A) -> ToolFacts:
        return ToolFacts()

    def _run(self, env: ToolEnv, ctx: RunContext, args: A) -> ToolResult:
        raise NotImplementedError


def _document_view(document: Document) -> dict[str, JsonValue]:
    return {
        "id": document.id,
        "title": document.title,
        "label": document.label,
        "body": document.body[:MAX_BODY_CHARS],
    }


class DocsSearch(_Adapter[DocsSearchArgs]):
    tool_id = "docs.search"
    args_model = DocsSearchArgs

    def _run(self, env: ToolEnv, ctx: RunContext, args: DocsSearchArgs) -> ToolResult:
        if env.search is None:
            # Only the gate's permission-filtered search may serve a query.
            raise RuntimeError("docs.search needs the gate's bound search")
        result = env.search(args.query, args.limit)
        documents: list[JsonValue] = [_document_view(doc) for doc in result.documents]
        return ToolResult(
            content={"documents": documents, "count": len(documents)},
            returned_labels=tuple(doc.label for doc in result.documents),
        )


class DocsRead(_Adapter[DocsReadArgs]):
    tool_id = "docs.read"
    args_model = DocsReadArgs

    def _facts(self, args: DocsReadArgs) -> ToolFacts:
        return ToolFacts(document_id=args.document_id)

    def _run(self, env: ToolEnv, ctx: RunContext, args: DocsReadArgs) -> ToolResult:
        document = env.store.get_document(args.document_id)
        # The gate checked this document already; checked again in case it changed since.
        if document is None or document.customer_id != ctx.customer_id:
            raise LookupError("document is no longer available")
        return ToolResult(content=_document_view(document), returned_labels=(document.label,))


class NotesWrite(_Adapter[NotesWriteArgs]):
    tool_id = "notes.write"
    args_model = NotesWriteArgs

    def _facts(self, args: NotesWriteArgs) -> ToolFacts:
        return ToolFacts(write_label=args.label)

    def _run(self, env: ToolEnv, ctx: RunContext, args: NotesWriteArgs) -> ToolResult:
        note = Document(
            id=f"note-{uuid.uuid4().hex[:12]}",
            customer_id=ctx.customer_id,
            title=args.title,
            label=args.label,
            tags=frozenset({NOTE_TAG}),
            body=args.body,
        )
        env.store.add_document(note)
        # Index this note alone: a full rebuild would hold the write lock over every
        # customer's documents for as long as it takes.
        env.index.add(note)
        return ToolResult(
            content={"document_id": note.id, "title": note.title, "label": note.label},
            effect_ref=note.id,
        )


class EmailSend(_Adapter[EmailSendArgs]):
    tool_id = "email.send"
    args_model = EmailSendArgs

    def _facts(self, args: EmailSendArgs) -> ToolFacts:
        return ToolFacts(recipients=tuple(args.to))

    def _run(self, env: ToolEnv, ctx: RunContext, args: EmailSendArgs) -> ToolResult:
        recipients: list[str] = []
        for raw in args.to:
            address = addresses.normalize_address(raw)
            if address is None:
                # Validation keeps a malformed recipient for the engine to deny; one gets here
                # only if a policy allowed it anyway, and it is still never stored or sent.
                raise ValueError("a recipient is not one plain address")
            recipients.append(address)
        email_id = f"email-{uuid.uuid4().hex[:12]}"
        _insert(
            env.store,
            "INSERT INTO outbox (id, run_id, customer_id, recipients, subject, body, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                email_id,
                ctx.run_id,
                ctx.customer_id,
                json.dumps(recipients),
                args.subject,
                args.body,
                env.clock.now().isoformat(),
            ),
        )
        to: list[JsonValue] = [*recipients]
        return ToolResult(
            content={"status": "queued", "email_id": email_id, "recipients": len(to), "to": to},
            effect_ref=email_id,
        )


class PartsOrder(_Adapter[PartsOrderArgs]):
    tool_id = "parts.order"
    args_model = PartsOrderArgs

    def _facts(self, args: PartsOrderArgs) -> ToolFacts:
        return ToolFacts(amount_usd=args.amount_usd)

    def _run(self, env: ToolEnv, ctx: RunContext, args: PartsOrderArgs) -> ToolResult:
        order_id = f"po-{uuid.uuid4().hex[:12]}"
        _insert(
            env.store,
            "INSERT INTO parts_orders"
            " (id, run_id, customer_id, part, quantity, amount_usd, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                order_id,
                ctx.run_id,
                ctx.customer_id,
                args.part,
                args.quantity,
                str(args.amount_usd),
                env.clock.now().isoformat(),
            ),
        )
        return ToolResult(content={"status": "ordered", "order_id": order_id}, effect_ref=order_id)


class RefundIssue(_Adapter[RefundIssueArgs]):
    tool_id = "refund.issue"
    args_model = RefundIssueArgs

    def _facts(self, args: RefundIssueArgs) -> ToolFacts:
        return ToolFacts(amount_usd=args.amount_usd)

    def _run(self, env: ToolEnv, ctx: RunContext, args: RefundIssueArgs) -> ToolResult:
        refund_id = f"rf-{uuid.uuid4().hex[:12]}"
        _insert(
            env.store,
            "INSERT INTO refunds (id, run_id, customer_id, order_id, amount_usd, reason,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                refund_id,
                ctx.run_id,
                ctx.customer_id,
                args.order_id,
                str(args.amount_usd),
                args.reason,
                env.clock.now().isoformat(),
            ),
        )
        return ToolResult(
            content={"status": "issued", "refund_id": refund_id}, effect_ref=refund_id
        )


def default_tools() -> dict[str, ToolAdapter]:
    adapters: list[ToolAdapter] = [
        DocsSearch(),
        DocsRead(),
        NotesWrite(),
        EmailSend(),
        PartsOrder(),
        RefundIssue(),
    ]
    return {adapter.tool_id: adapter for adapter in adapters}


def ensure_schema(store: Store) -> None:
    """Create the side-effect tables if they are missing. Safe to call repeatedly."""
    with store.locked():
        for statement in _SCHEMA:
            store.connection.execute(statement)


def _insert(store: Store, sql: str, params: tuple[str | int, ...]) -> None:
    ensure_schema(store)
    with store.transaction():
        store.connection.execute(sql, params)


@dataclass(frozen=True)
class OutboxEmail:
    id: str
    run_id: str
    customer_id: str
    recipients: tuple[str, ...]
    subject: str
    body: str
    created_at: datetime


@dataclass(frozen=True)
class PartsOrderRow:
    id: str
    run_id: str
    customer_id: str
    part: str
    quantity: int
    amount_usd: Decimal
    created_at: datetime


@dataclass(frozen=True)
class RefundRow:
    id: str
    run_id: str
    customer_id: str
    order_id: str
    amount_usd: Decimal
    reason: str
    created_at: datetime


def _rows(store: Store, table: str, customer_id: str | None) -> list[dict[str, str | int]]:
    # table is one of this module's constants, never caller text. A table no tool has
    # written yet holds no rows; it is not created here, so a read-only store can be read.
    query = f"SELECT * FROM {table}"
    params: tuple[str, ...] = ()
    if customer_id is not None:
        query += " WHERE customer_id = ?"
        params = (customer_id,)
    query += " ORDER BY created_at, rowid"
    with store.locked():
        exists = store.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if exists is None:
            return []
        return [dict(row) for row in store.connection.execute(query, params).fetchall()]


def list_outbox(store: Store, customer_id: str | None = None) -> list[OutboxEmail]:
    """Queued emails, oldest first, for one customer or for every customer."""
    return [
        OutboxEmail(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            customer_id=str(row["customer_id"]),
            recipients=tuple(json.loads(str(row["recipients"]))),
            subject=str(row["subject"]),
            body=str(row["body"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )
        for row in _rows(store, "outbox", customer_id)
    ]


def list_parts_orders(store: Store, customer_id: str | None = None) -> list[PartsOrderRow]:
    return [
        PartsOrderRow(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            customer_id=str(row["customer_id"]),
            part=str(row["part"]),
            quantity=int(row["quantity"]),
            amount_usd=Decimal(str(row["amount_usd"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )
        for row in _rows(store, "parts_orders", customer_id)
    ]


def list_refunds(store: Store, customer_id: str | None = None) -> list[RefundRow]:
    return [
        RefundRow(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            customer_id=str(row["customer_id"]),
            order_id=str(row["order_id"]),
            amount_usd=Decimal(str(row["amount_usd"])),
            reason=str(row["reason"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )
        for row in _rows(store, "refunds", customer_id)
    ]


@dataclass(frozen=True)
class StoredEffect:
    """A side effect a built-in tool wrote, as the validated args that would have written it.

    `args` hashed with approvals.args_hash under the tool and its action is what the gate
    recorded as `args_hash` on the call's EXECUTED event, whose `effect_ref` is `id`, so
    `agentgate audit check-store` can tell a row edited after the fact from the one the gate
    allowed. Notes have no run in the store; their run_id is None.
    """

    tool: str
    id: str
    customer_id: str
    run_id: str | None
    args: dict[str, JsonValue]


def stored_effects(store: Store, customer_id: str) -> list[StoredEffect]:
    """Every outbox email, parts order, refund and note of one customer, oldest first."""
    effects: list[StoredEffect] = []
    for email in list_outbox(store, customer_id):
        to: list[JsonValue] = [*email.recipients]
        effects.append(
            StoredEffect(
                EmailSend.tool_id,
                email.id,
                email.customer_id,
                email.run_id,
                {"to": to, "subject": email.subject, "body": email.body},
            )
        )
    for order in list_parts_orders(store, customer_id):
        effects.append(
            StoredEffect(
                PartsOrder.tool_id,
                order.id,
                order.customer_id,
                order.run_id,
                {
                    "part": order.part,
                    "quantity": order.quantity,
                    "amount_usd": str(order.amount_usd),
                },
            )
        )
    for refund in list_refunds(store, customer_id):
        effects.append(
            StoredEffect(
                RefundIssue.tool_id,
                refund.id,
                refund.customer_id,
                refund.run_id,
                {
                    "order_id": refund.order_id,
                    "amount_usd": str(refund.amount_usd),
                    "reason": refund.reason,
                },
            )
        )
    for document in store.list_documents(customer_id):
        if NOTE_TAG in document.tags:
            effects.append(
                StoredEffect(
                    NotesWrite.tool_id,
                    document.id,
                    document.customer_id,
                    None,
                    {"title": document.title, "body": document.body, "label": document.label},
                )
            )
    return effects
