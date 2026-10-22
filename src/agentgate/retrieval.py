"""Permission-aware document search.

Documents are filtered before ranking, so nothing the run may not read reaches the ranker
or the model: no bodies, no titles, no counts. The permitted set comes from the policy
engine itself. For each of the run's customer documents, the engine decides a read with the
search tool from the document's metadata alone (label, tags, assignment, never the body),
and only an ALLOW counts; a read that would need approval is not searchable. Read and search
therefore share one decision path and cannot disagree.

Full-text search then runs only over the permitted documents, in the same SQL statement as
the ranking. As defence in depth, every ranked document is decided again from its stored
row before it is returned, and any disagreement raises RetrievalInvariantError.

Costs and limits:
- Each search makes one engine call per document of the customer. That is fine at MVP
  scale; compiling the read rules into a SQL predicate is the place to optimise later.
- bm25 term statistics (document frequency, average length) come from the whole index, not
  just the permitted rows, so the order of permitted results for a multi-term query can
  shift with the content of documents the run cannot read. Membership and counts never do.
  Ranking with statistics computed over the permitted set alone would close that gap.

The trace returned with each search is for the audit log only. The model sees `documents`.
"""

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from agentgate.context import RunContext
from agentgate.models import Action, Document, Role
from agentgate.policy.engine import DocumentFacts, PolicyEngine, Request
from agentgate.policy.schema import Effect
from agentgate.store import Store

MAX_LIMIT = 50
MAX_TERMS = 16
MAX_TERM_LENGTH = 64

_TERM = re.compile(r"[a-z0-9]+")

# A standalone index keyed by document id rather than an external-content table: the
# documents table has no INTEGER PRIMARY KEY, so its rowids are not stable enough to key on.
# remove_diacritics 2 folds accents the same way query sanitisation does.
_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    doc_id UNINDEXED,
    title,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
)
"""

# bm25 column weights in table order: doc_id (unindexed), title, body. A title hit is a
# stronger signal than a body hit in these short documents.
_RANK_SQL = """
SELECT doc_id
FROM documents_fts
WHERE documents_fts MATCH ?
  AND doc_id IN (SELECT value FROM json_each(?))
ORDER BY bm25(documents_fts, 0.0, 2.0, 1.0), doc_id
LIMIT ?
"""


@dataclass(frozen=True)
class RetrievalTrace:
    customer_id: str
    user_id: str
    role: Role
    run_id: str
    tool: str
    # The sanitised terms actually searched.
    query_terms: tuple[str, ...]
    # Documents of this customer.
    candidates_in_tenant: int
    permitted: int
    # Engine Reason value to count, for filtered documents.
    filtered_by_reason: dict[str, int]
    # Sorted unique rule ids that permitted at least one document.
    permitting_rules: tuple[str, ...]
    returned_ids: tuple[str, ...]
    policy_version: str


@dataclass(frozen=True)
class RetrievalResult:
    # Ranked best first, at most limit.
    documents: tuple[Document, ...]
    trace: RetrievalTrace


class RetrievalInvariantError(RuntimeError):
    """A ranked document failed the post-ranking permission check. This should never happen."""


@dataclass(frozen=True)
class _Filter:
    candidates: int
    permitted_ids: tuple[str, ...]
    filtered_by_reason: dict[str, int]
    permitting_rules: tuple[str, ...]


def sanitize_query(query: str) -> tuple[str, ...]:
    """Reduce free text to plain lowercase word terms, so FTS5 syntax cannot be injected.

    Compatibility forms and accents are folded first (fullwidth letters, "cafe" for "café").
    Terms are deduplicated in order, at most MAX_TERMS are kept, and a term longer than
    MAX_TERM_LENGTH is dropped rather than truncated into a different word.
    """
    folded = unicodedata.normalize("NFKD", query).lower()
    terms: list[str] = []
    for term in _TERM.findall(folded):
        if len(term) > MAX_TERM_LENGTH or term in terms:
            continue
        terms.append(term)
        if len(terms) == MAX_TERMS:
            break
    return tuple(terms)


def _match_expression(terms: tuple[str, ...]) -> str:
    # Terms are [a-z0-9]+ only, so quoting each one makes it a literal phrase and no term
    # can read as an operator (NEAR, NOT), a column filter or a prefix query.
    return " OR ".join(f'"{term}"' for term in terms)


class DocumentIndex:
    """A full-text index over every document in a store, searched per run and permission."""

    def __init__(self, store: Store) -> None:
        self._store = store
        with store.locked():
            store.connection.execute(_SCHEMA)

    def rebuild(self) -> None:
        """(Re)index every document in the store. Call after loading fixtures."""
        with self._store.transaction():
            conn = self._store.connection
            conn.execute("DELETE FROM documents_fts")
            conn.execute(
                "INSERT INTO documents_fts (doc_id, title, body)"
                " SELECT id, title, body FROM documents ORDER BY id"
            )

    def search(
        self,
        engine: PolicyEngine,
        ctx: RunContext,
        query: str,
        *,
        run_max_label: str | None,
        tool: str = "docs.search",
        limit: int = 5,
    ) -> RetrievalResult:
        """Return the best permitted matches for query, best first, with an audit trace."""
        if ctx.customer_id != engine.bundle.customer_id:
            # A wiring bug in the caller, not something a user request can cause.
            raise ValueError(
                f"run customer {ctx.customer_id!r} does not match policy customer"
                f" {engine.bundle.customer_id!r}"
            )
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}")

        terms = sanitize_query(query)
        # One lock for the filter, the ranking and the re-check, so all three see the same
        # rows even if another thread writes documents in between.
        with self._store.locked():
            filtered = self._permitted(engine, ctx, tool, run_max_label)
            ranked_ids = self._rank(terms, filtered.permitted_ids, limit)
            documents = tuple(
                self._recheck(engine, ctx, tool, run_max_label, doc_id) for doc_id in ranked_ids
            )

        trace = RetrievalTrace(
            customer_id=ctx.customer_id,
            user_id=ctx.user_id,
            role=ctx.role,
            run_id=ctx.run_id,
            tool=tool,
            query_terms=terms,
            candidates_in_tenant=filtered.candidates,
            permitted=len(filtered.permitted_ids),
            filtered_by_reason=filtered.filtered_by_reason,
            permitting_rules=filtered.permitting_rules,
            returned_ids=tuple(doc.id for doc in documents),
            policy_version=engine.version,
        )
        return RetrievalResult(documents=documents, trace=trace)

    def _permitted(
        self, engine: PolicyEngine, ctx: RunContext, tool: str, run_max_label: str | None
    ) -> _Filter:
        # Metadata only: bodies are not read for documents the run may not see.
        rows = self._store.connection.execute(
            "SELECT id, customer_id, label, tags, assigned_user_ids FROM documents"
            " WHERE customer_id = ? ORDER BY id",
            (ctx.customer_id,),
        ).fetchall()
        permitted: list[str] = []
        reasons: Counter[str] = Counter()
        rules: set[str] = set()
        for row in rows:
            facts = DocumentFacts(
                customer_id=row["customer_id"],
                label=row["label"],
                tags=frozenset(json.loads(row["tags"])),
                assigned_user_ids=frozenset(json.loads(row["assigned_user_ids"])),
            )
            decision = engine.decide(_read_request(ctx, tool, facts, run_max_label))
            if decision.effect is Effect.ALLOW:
                permitted.append(row["id"])
                rules.update(decision.rule_ids)
            else:
                reasons[decision.reason.value] += 1
        return _Filter(
            candidates=len(rows),
            permitted_ids=tuple(permitted),
            filtered_by_reason=dict(sorted(reasons.items())),
            permitting_rules=tuple(sorted(rules)),
        )

    def _rank(
        self, terms: tuple[str, ...], permitted_ids: tuple[str, ...], limit: int
    ) -> list[str]:
        if not terms or not permitted_ids:
            return []
        # The permitted ids travel as one JSON parameter, so the statement has a fixed shape
        # and no bound-variable limit however many documents a customer has.
        rows = self._store.connection.execute(
            _RANK_SQL, (_match_expression(terms), json.dumps(permitted_ids), limit)
        ).fetchall()
        return [row["doc_id"] for row in rows]

    def _recheck(
        self,
        engine: PolicyEngine,
        ctx: RunContext,
        tool: str,
        run_max_label: str | None,
        doc_id: str,
    ) -> Document:
        document = self._store.get_document(doc_id)
        if document is None or document.customer_id != ctx.customer_id:
            raise RetrievalInvariantError(f"ranked document {doc_id!r} is not in the run's tenant")
        facts = DocumentFacts.from_document(document)
        decision = engine.decide(_read_request(ctx, tool, facts, run_max_label))
        if decision.effect is not Effect.ALLOW:
            raise RetrievalInvariantError(
                f"ranked document {doc_id!r} is not readable: {decision.reason.value}"
            )
        return document


def _read_request(
    ctx: RunContext, tool: str, facts: DocumentFacts, run_max_label: str | None
) -> Request:
    return Request(
        customer_id=ctx.customer_id,
        user_id=ctx.user_id,
        role=ctx.role,
        tool=tool,
        action=Action.READ,
        document=facts,
        run_max_label=run_max_label,
    )
