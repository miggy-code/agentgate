"""Permission-aware document search.

Documents are filtered before ranking, so nothing the run may not read reaches the ranker
or the model: no bodies, no titles, no counts. The permitted set comes from the policy
engine itself. For each of the run's customer documents, the engine decides a read with the
search tool from the document's metadata alone (label, tags, assignment, never the body),
and only an ALLOW counts; a read that would need approval is not searchable. Read and search
therefore share one decision path and cannot disagree.

Ranking is BM25F computed in Python over the permitted documents only. The index is a
plain inverted index (per-field term frequencies and field lengths) built with the same
tokeniser as the query. Every index read is restricted to the permitted ids in SQL, so the
term frequencies, document frequencies, average field lengths and document count behind a
score all come from documents the run may read. The order of results therefore cannot
depend on any document the run cannot read, in this customer or another. FTS5 is not used:
its bm25 takes statistics from the whole table, and it exposes no per-document field
lengths, so permitted-only statistics would need a second tokeniser that has to agree
with FTS5's exactly.

As defence in depth, every ranked document is decided again from its stored row before it
is returned, and any disagreement raises RetrievalInvariantError.

Costs and limits:
- Each search makes one engine call per document of the customer, and reads the field
  lengths of every permitted document plus the postings of the query terms. That is fine at
  MVP scale; compiling the read rules into a SQL predicate and caching per-permission-set
  statistics are the places to optimise later.
- The index is kept explicitly. `add(document)` indexes one stored document (or re-indexes
  it after an edit) and `remove(doc_id)` drops one, each in one short transaction that
  touches only that document's rows; `rebuild()` re-indexes every document in the store and
  holds the write lock for as long as that takes, so it is for loading fixtures and repairs,
  not for every write. A document stored without either is not searchable until one runs.
  Both paths write rows with the same function, so an index built up with `add` holds
  exactly the rows a full rebuild would, and ranks identically.

The trace returned with each search is for the audit log only. The model sees `documents`.
"""

import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
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

# BM25F parameters. A title hit is a stronger signal than a body hit in these short documents.
K1 = 1.2
B = 0.75
TITLE_WEIGHT = 2.0
BODY_WEIGHT = 1.0

# Postings hold only terms a query can contain (MAX_TERM_LENGTH or shorter); lengths count
# every token. Keyed by document id because the documents table has no INTEGER PRIMARY KEY,
# so its rowids are not a stable key.
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS retrieval_postings (
        term TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        title_tf INTEGER NOT NULL,
        body_tf INTEGER NOT NULL,
        PRIMARY KEY (term, doc_id)
    ) STRICT, WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval_lengths (
        doc_id TEXT PRIMARY KEY,
        title_len INTEGER NOT NULL,
        body_len INTEGER NOT NULL
    ) STRICT
    """,
    # For dropping one document's postings without scanning every customer's.
    """
    CREATE INDEX IF NOT EXISTS retrieval_postings_doc_id ON retrieval_postings (doc_id)
    """,
)

# Ids and terms travel as one JSON parameter each, so the statements have a fixed shape and
# no bound-variable limit however many documents a customer has.
_LENGTHS_SQL = """
SELECT doc_id, title_len, body_len
FROM retrieval_lengths
WHERE doc_id IN (SELECT value FROM json_each(?))
"""
_POSTINGS_SQL = """
SELECT term, doc_id, title_tf, body_tf
FROM retrieval_postings
WHERE term IN (SELECT value FROM json_each(?))
  AND doc_id IN (SELECT value FROM json_each(?))
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
    """Reduce free text to plain lowercase word terms, so no query syntax can be injected.

    Compatibility forms and accents are folded first (fullwidth letters, "cafe" for "café").
    Terms are deduplicated in order, at most MAX_TERMS are kept, and a term longer than
    MAX_TERM_LENGTH is dropped rather than truncated into a different word.
    """
    terms: list[str] = []
    for term in _tokens(query):
        if len(term) > MAX_TERM_LENGTH or term in terms:
            continue
        terms.append(term)
        if len(terms) == MAX_TERMS:
            break
    return tuple(terms)


def _tokens(text: str) -> list[str]:
    # Queries and documents share this, so a folded query word always meets the same folded
    # word in the index.
    return _TERM.findall(unicodedata.normalize("NFKD", text).lower())


@dataclass(frozen=True)
class _Posting:
    term: str
    doc_id: str
    title_tf: int
    body_tf: int


@dataclass(frozen=True)
class _Lengths:
    title: int
    body: int


def _bm25f(
    terms: Sequence[str],
    lengths: dict[str, _Lengths],
    postings: Iterable[_Posting],
    limit: int,
) -> list[str]:
    """Rank documents by BM25F using statistics from `lengths` and `postings` alone.

    Callers pass only permitted documents, which is what keeps unreadable documents out of
    every statistic. Each field's term frequency is normalised by that field's length
    against its average, weighted, summed, then saturated once per term with K1.
    """
    count = len(lengths)
    if count == 0:
        return []
    avg_title = sum(n.title for n in lengths.values()) / count
    avg_body = sum(n.body for n in lengths.values()) / count

    by_term: dict[str, list[_Posting]] = {}
    for posting in postings:
        if posting.doc_id in lengths:
            by_term.setdefault(posting.term, []).append(posting)

    scores: dict[str, float] = {}
    for term in terms:
        matches = by_term.get(term, [])
        if not matches:
            continue
        # The Lucene form of IDF stays positive, so a term common in a small permitted set
        # still counts for something instead of scoring zero or below.
        idf = math.log(1 + (count - len(matches) + 0.5) / (len(matches) + 0.5))
        for posting in matches:
            size = lengths[posting.doc_id]
            weighted = TITLE_WEIGHT * _normalised(posting.title_tf, size.title, avg_title)
            weighted += BODY_WEIGHT * _normalised(posting.body_tf, size.body, avg_body)
            gain = idf * weighted * (K1 + 1) / (weighted + K1)
            scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + gain
    ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return ranked[:limit]


def _normalised(tf: int, length: int, average: float) -> float:
    if tf == 0:
        return 0.0
    if average == 0:
        return float(tf)
    return tf / (1 - B + B * length / average)


class DocumentIndex:
    """A word index over every document in a store, searched per run and permission."""

    def __init__(self, store: Store) -> None:
        self._store = store
        with store.locked():
            for statement in _SCHEMA:
                store.connection.execute(statement)

    def rebuild(self) -> None:
        """(Re)index every document in the store. Call after loading fixtures.

        This holds the database write lock while it reads and indexes every customer's
        documents. To index one new or changed document, use `add`.
        """
        with self._store.transaction():
            conn = self._store.connection
            conn.execute("DELETE FROM retrieval_postings")
            conn.execute("DELETE FROM retrieval_lengths")
            rows = conn.execute("SELECT id, title, body FROM documents ORDER BY id").fetchall()
            for row in rows:
                _write_rows(conn, row["id"], row["title"], row["body"])

    def add(self, document: Document) -> None:
        """Index one stored document, replacing any rows it had, in one short transaction.

        Only this document's rows are written, so the write lock is held for the time one
        document takes, whatever the store holds. Call it after storing the document, and
        again after changing its title or body.
        """
        with self._store.transaction():
            conn = self._store.connection
            _delete_rows(conn, document.id)
            _write_rows(conn, document.id, document.title, document.body)

    def remove(self, doc_id: str) -> None:
        """Drop one document's rows from the index, in one short transaction."""
        with self._store.transaction():
            _delete_rows(self._store.connection, doc_id)

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
        conn = self._store.connection
        ids = json.dumps(permitted_ids)
        lengths = {
            row["doc_id"]: _Lengths(row["title_len"], row["body_len"])
            for row in conn.execute(_LENGTHS_SQL, (ids,))
        }
        postings = [
            _Posting(row["term"], row["doc_id"], row["title_tf"], row["body_tf"])
            for row in conn.execute(_POSTINGS_SQL, (json.dumps(terms), ids))
        ]
        return _bm25f(terms, lengths, postings, limit)

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


def _delete_rows(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute("DELETE FROM retrieval_postings WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM retrieval_lengths WHERE doc_id = ?", (doc_id,))


def _write_rows(conn: sqlite3.Connection, doc_id: str, title_text: str, body_text: str) -> None:
    """Insert one document's lengths and postings. `rebuild` and `add` both use this."""
    title = _tokens(title_text)
    body = _tokens(body_text)
    conn.execute(
        "INSERT INTO retrieval_lengths (doc_id, title_len, body_len) VALUES (?, ?, ?)",
        (doc_id, len(title), len(body)),
    )
    title_tf = Counter(t for t in title if len(t) <= MAX_TERM_LENGTH)
    body_tf = Counter(t for t in body if len(t) <= MAX_TERM_LENGTH)
    conn.executemany(
        "INSERT INTO retrieval_postings (term, doc_id, title_tf, body_tf) VALUES (?, ?, ?, ?)",
        [
            (term, doc_id, title_tf[term], body_tf[term])
            for term in sorted(title_tf.keys() | body_tf.keys())
        ],
    )


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
