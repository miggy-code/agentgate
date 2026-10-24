"""Permission-aware retrieval against the real fixtures and policies."""

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agentgate.context import RunContext
from agentgate.fixtures import find_canaries, load_fixtures
from agentgate.models import Action, Customer, Document, Role, User
from agentgate.policy.engine import DocumentFacts, PolicyEngine, Request
from agentgate.policy.schema import Effect, load_bundle, load_bundles
from agentgate.retrieval import (
    MAX_TERMS,
    DocumentIndex,
    RetrievalInvariantError,
    RetrievalResult,
    _Filter,
    sanitize_query,
)
from agentgate.store import Store

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
POLICIES = ROOT / "policies"
KC = "kettle-creek-hvac"
JR = "juniper-row-apothecary"
STARTED = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
# "PAYROLL" in fullwidth letters, built from code points so the source stays ASCII here.
FULLWIDTH_PAYROLL = "".join(chr(ord(c) + 0xFEE0) for c in "PAYROLL")


@dataclass(frozen=True)
class World:
    store: Store
    index: DocumentIndex
    engines: dict[str, PolicyEngine]

    def users(self, customer_id: str) -> list[User]:
        return self.store.list_users(customer_id)

    def user(self, user_id: str) -> User:
        user = self.store.get_user(user_id)
        assert user is not None
        return user

    def docs(self, customer_id: str) -> list[Document]:
        return self.store.list_documents(customer_id)

    def search(
        self,
        user_id: str,
        query: str,
        *,
        run_max_label: str | None = None,
        limit: int = 50,
        tool: str = "docs.search",
    ) -> RetrievalResult:
        user = self.user(user_id)
        return self.index.search(
            self.engines[user.customer_id],
            ctx(user),
            query,
            run_max_label=run_max_label,
            tool=tool,
            limit=limit,
        )

    def readable(self, user_id: str, run_max_label: str | None = None) -> set[str]:
        """The permitted set, computed independently of the index."""
        user = self.user(user_id)
        return permitted_ids(
            self.engines[user.customer_id], user, self.docs(user.customer_id), run_max_label
        )


def ctx(user: User, customer_id: str | None = None) -> RunContext:
    return RunContext(
        run_id=f"run-{user.id}",
        customer_id=customer_id or user.customer_id,
        user_id=user.id,
        role=user.role,
        started_at=STARTED,
    )


def permitted_ids(
    engine: PolicyEngine, user: User, docs: list[Document], run_max_label: str | None = None
) -> set[str]:
    return {
        doc.id
        for doc in docs
        if engine.decide(
            Request(
                customer_id=user.customer_id,
                user_id=user.id,
                role=user.role,
                tool="docs.search",
                action=Action.READ,
                document=DocumentFacts.from_document(doc),
                run_max_label=run_max_label,
            )
        ).effect
        is Effect.ALLOW
    }


def covering_query(docs: list[Document]) -> str:
    """At most MAX_TERMS words that, OR'ed, match every one of docs (greedy set cover)."""
    words = {
        doc.id: set(re.findall(r"[a-z0-9]+", f"{doc.title} {doc.body}".lower())) for doc in docs
    }
    uncovered = set(words)
    chosen: list[str] = []
    while uncovered:
        counts: dict[str, int] = {}
        for doc_id in uncovered:
            for word in words[doc_id]:
                counts[word] = counts.get(word, 0) + 1
        best = min(counts, key=lambda word: (-counts[word], word))
        chosen.append(best)
        uncovered = {doc_id for doc_id in uncovered if best not in words[doc_id]}
    assert len(chosen) <= MAX_TERMS
    return " ".join(chosen)


def _build_world() -> World:
    store = Store()
    load_fixtures(store, FIXTURES)
    engines = {
        customer_id: PolicyEngine(bundle, _customer(store, customer_id))
        for customer_id, bundle in load_bundles(POLICIES).items()
    }
    index = DocumentIndex(store)
    index.rebuild()
    return World(store=store, index=index, engines=engines)


def _customer(store: Store, customer_id: str) -> Customer:
    customer = store.get_customer(customer_id)
    assert customer is not None
    return customer


def _all_user_ids() -> list[str]:
    with Store() as store:
        load_fixtures(store, FIXTURES)
        return [u.id for c in store.list_customers() for u in store.list_users(c.id)]


USER_IDS = _all_user_ids()
STAFF_AND_CONTRACTORS = [u for u in USER_IDS if "-staff-" in u or "-contractor-" in u]


@pytest.fixture(scope="module")
def world() -> Iterator[World]:
    w = _build_world()
    yield w
    w.store.close()


def other(customer_id: str) -> str:
    return JR if customer_id == KC else KC


# Agreement with the engine


def test_covering_query_matches_every_document(world: World) -> None:
    for owner_id in ("kc-owner-dale", "jr-owner-marisol"):
        customer_id = world.user(owner_id).customer_id
        docs = world.docs(customer_id)
        # The owner may read everything, so a covering query must return every document.
        result = world.search(owner_id, covering_query(docs))
        assert set(result.trace.returned_ids) == {d.id for d in docs}


@pytest.mark.parametrize("user_id", USER_IDS)
def test_results_are_exactly_the_permitted_set(world: World, user_id: str) -> None:
    customer_id = world.user(user_id).customer_id
    result = world.search(user_id, covering_query(world.docs(customer_id)))
    expected = world.readable(user_id)
    assert set(result.trace.returned_ids) == expected
    assert len(result.documents) == len(expected)
    assert result.trace.permitted == len(expected)


@pytest.mark.parametrize("user_id", USER_IDS)
def test_every_title_query_returns_only_permitted_documents(world: World, user_id: str) -> None:
    user = world.user(user_id)
    engine = world.engines[user.customer_id]
    readable = world.readable(user_id)
    for doc in world.docs(user.customer_id):
        result = world.search(user_id, doc.title)
        returned = {d.id for d in result.documents}
        assert returned <= readable
        assert permitted_ids(engine, user, list(result.documents)) == returned
        # A readable document is always found by its own title.
        assert (doc.id in returned) is (doc.id in readable)


# Tenant isolation and canaries


@pytest.mark.parametrize("user_id", USER_IDS)
def test_other_customers_words_never_return_their_documents(world: World, user_id: str) -> None:
    customer_id = world.user(user_id).customer_id
    foreign = world.docs(other(customer_id))
    queries = [
        covering_query(foreign),
        "harrowfield naturals alder bench labs juniperrow formula",
        "northfield mechanical supply larkspur dunmore furnace harlan",
    ]
    for query in queries:
        result = world.search(user_id, query)
        assert all(doc.customer_id == customer_id for doc in result.documents)
        assert set(result.trace.returned_ids).isdisjoint(d.id for d in foreign)


@pytest.mark.parametrize("user_id", STAFF_AND_CONTRACTORS)
def test_no_canary_from_an_unreadable_document_is_returned(world: World, user_id: str) -> None:
    customer_id = world.user(user_id).customer_id
    readable = world.readable(user_id)
    all_docs = world.docs(customer_id) + world.docs(other(customer_id))
    forbidden = {c for d in all_docs if d.id not in readable for c in find_canaries(d.body)}
    assert forbidden
    for query in (covering_query(world.docs(customer_id)), "canary agcanary", "AGCANARY"):
        result = world.search(user_id, query)
        for doc in result.documents:
            assert forbidden.isdisjoint(find_canaries(doc.body))


# Role scenarios from the policies


def test_contractor_sees_assigned_and_sop_but_not_other_job_tickets(world: World) -> None:
    result = world.search("kc-contractor-hollis", "job ticket sop procedure checklist")
    returned = set(result.trace.returned_ids)
    assert {"kc-job-2041-furnace-igniter", "kc-job-2044-ac-no-cool"} <= returned
    assert {"kc-install-checklist-sop", "kc-refrigerant-handling-sop"} <= returned
    assert "kc-dispatch-procedure" in returned
    assert returned.isdisjoint({"kc-job-2047-heat-pump-tuneup", "kc-job-2049-thermostat-swap"})


def test_jr_contractor_sees_own_manifests_only(world: World) -> None:
    result = world.search("jr-contractor-gideon", "delivery manifest")
    returned = set(result.trace.returned_ids)
    assert {"jr-manifest-2026-10-22-north", "jr-manifest-2026-10-22-south"} <= returned
    assert "jr-manifest-2026-10-22-shop-pickups" not in returned


@pytest.mark.parametrize("customer_id", [KC, JR])
def test_staff_never_see_confidential_or_restricted(world: World, customer_id: str) -> None:
    query = covering_query(world.docs(customer_id))
    for user in world.users(customer_id):
        if user.role is not Role.STAFF:
            continue
        result = world.search(user.id, query)
        assert result.documents
        assert {d.label for d in result.documents} <= {"public", "internal"}


def test_pharmacy_manager_sees_order_records_but_not_payroll(world: World) -> None:
    result = world.search("jr-mgr-daniel", "payroll order records refill synthetic")
    returned = set(result.trace.returned_ids)
    assert {
        "jr-order-records-2026-09",
        "jr-order-records-2026-10",
        "jr-refill-schedule-records",
    } <= returned
    assert "jr-payroll-summary-2026" not in returned
    owner = world.search("jr-owner-marisol", "payroll")
    assert owner.trace.returned_ids[0] == "jr-payroll-summary-2026"


# run_max_label


@pytest.mark.parametrize("user_id", USER_IDS)
def test_run_max_label_does_not_change_reads_under_shipped_policies(
    world: World, user_id: str
) -> None:
    engine = world.engines[world.user(user_id).customer_id]
    read_rules = [r for r in engine.bundle.policy.rules if Action.READ in r.actions]
    assert all(r.where.run_max_label_gte is None for r in read_rules)
    query = covering_query(world.docs(world.user(user_id).customer_id))
    baseline = world.search(user_id, query).trace.returned_ids
    for label in engine.bundle.policy.labels:
        assert world.search(user_id, query, run_max_label=label).trace.returned_ids == baseline


def test_unknown_run_max_label_permits_nothing(world: World) -> None:
    result = world.search("kc-owner-dale", "furnace", run_max_label="top-secret")
    assert result.documents == ()
    assert result.trace.permitted == 0
    assert result.trace.filtered_by_reason == {"unknown_label": result.trace.candidates_in_tenant}


def test_run_max_label_reaches_a_read_rule(world: World, tmp_path: Path) -> None:
    source = POLICIES / KC
    target = tmp_path / KC
    target.mkdir()
    for name in ("policy.yaml", "tools.yaml"):
        (target / name).write_text((source / name).read_text())
    with (target / "policy.yaml").open("a") as policy:
        policy.write(
            "\n"
            "  - id: no-search-after-restricted\n"
            "    effect: deny\n"
            "    description: Test rule.\n"
            "    tools: [docs.search]\n"
            "    actions: [read]\n"
            "    labels: {max: internal}\n"
            "    where:\n"
            "      run_max_label_gte: restricted\n"
        )
    engine = PolicyEngine(load_bundle(target), _customer(world.store, KC))
    owner = world.user("kc-owner-dale")
    docs = world.docs(KC)
    query = covering_query(docs)

    def search(label: str | None) -> RetrievalResult:
        return world.index.search(engine, ctx(owner), query, run_max_label=label, limit=50)

    before = search("confidential")
    assert set(before.trace.returned_ids) == {d.id for d in docs}
    after = search("restricted")
    expected = {d.id for d in docs if d.label in {"confidential", "restricted"}}
    assert set(after.trace.returned_ids) == expected
    assert expected == permitted_ids(engine, owner, docs, "restricted")
    assert after.trace.filtered_by_reason == {"matched_deny": len(docs) - len(expected)}


# Query sanitisation


@pytest.mark.parametrize(
    ("query", "plain"),
    [
        ("title:secret", "title secret"),
        ('"payroll" OR *', "payroll or"),
        ("NEAR(a b)", "near a b"),
        ("a AND NOT b", "a and not b"),
        (")", ""),
        ('"unterminated', "unterminated"),
        ("payroll*", "payroll"),
        ("^furnace -warranty +labor", "furnace warranty labor"),
        ("{title body}: payroll", "title body payroll"),
        ("café ☕ 東京 ПАРОЛЬ", "cafe"),
        (FULLWIDTH_PAYROLL, "payroll"),
        ("x'; DROP TABLE documents; --", "x drop table documents"),
    ],
)
def test_fts5_syntax_is_searched_as_plain_words(world: World, query: str, plain: str) -> None:
    assert sanitize_query(query) == sanitize_query(plain)
    result = world.search("kc-owner-dale", query)
    assert result.trace.returned_ids == world.search("kc-owner-dale", plain).trace.returned_ids


def test_fullwidth_query_finds_the_plain_word(world: World) -> None:
    result = world.search("jr-owner-marisol", FULLWIDTH_PAYROLL)
    assert result.trace.query_terms == ("payroll",)
    assert result.trace.returned_ids[0] == "jr-payroll-summary-2026"


def test_terms_are_capped_deduplicated_and_bounded() -> None:
    words = [f"w{i}" for i in range(30)]
    assert sanitize_query(" ".join(words)) == tuple(words[:MAX_TERMS])
    assert sanitize_query("Payroll payroll PAYROLL") == ("payroll",)
    assert sanitize_query("x" * 65 + " furnace " + "y" * 64) == ("furnace", "y" * 64)


@pytest.mark.parametrize("query", ["", "   ", "!!! ??? ...", "東京", "*"])
def test_query_without_terms_returns_nothing(world: World, query: str) -> None:
    result = world.search("kc-owner-dale", query)
    assert result.documents == ()
    assert result.trace.query_terms == ()
    assert result.trace.returned_ids == ()
    assert result.trace.permitted == result.trace.candidates_in_tenant > 0


# Arguments


@pytest.mark.parametrize("limit", [0, -1, 51])
def test_limit_out_of_range_is_rejected(world: World, limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        world.search("kc-owner-dale", "furnace", limit=limit)


def test_limit_bounds_are_accepted_and_respected(world: World) -> None:
    query = covering_query(world.docs(KC))
    assert len(world.search("kc-owner-dale", query, limit=1).documents) == 1
    assert len(world.search("kc-owner-dale", query, limit=3).documents) == 3
    assert len(world.search("kc-owner-dale", query, limit=50).documents) == len(world.docs(KC))
    assert len(world.search("kc-owner-dale", query).documents) == len(world.docs(KC))


def test_default_limit_is_five(world: World) -> None:
    owner = world.user("kc-owner-dale")
    query = covering_query(world.docs(KC))
    result = world.index.search(world.engines[KC], ctx(owner), query, run_max_label=None)
    assert len(result.documents) == 5


def test_context_and_engine_customer_must_match(world: World) -> None:
    owner = world.user("kc-owner-dale")
    with pytest.raises(ValueError, match="does not match"):
        world.index.search(world.engines[JR], ctx(owner), "furnace", run_max_label=None)
    with pytest.raises(ValueError, match="does not match"):
        world.index.search(world.engines[KC], ctx(owner, JR), "furnace", run_max_label=None)


def test_unknown_tool_permits_nothing(world: World) -> None:
    result = world.search("kc-owner-dale", "furnace", tool="docs.nope")
    assert result.documents == ()
    assert result.trace.filtered_by_reason == {"unknown_tool": len(world.docs(KC))}


# Trace


@pytest.mark.parametrize("user_id", USER_IDS)
def test_trace_is_consistent(world: World, user_id: str) -> None:
    user = world.user(user_id)
    engine = world.engines[user.customer_id]
    result = world.search(user_id, "payroll furnace customer list sop schedule", limit=7)
    trace = result.trace
    assert (trace.customer_id, trace.user_id, trace.role) == (user.customer_id, user.id, user.role)
    assert trace.run_id == f"run-{user.id}"
    assert trace.tool == "docs.search"
    assert trace.query_terms == ("payroll", "furnace", "customer", "list", "sop", "schedule")
    assert trace.candidates_in_tenant == len(world.docs(user.customer_id))
    assert trace.permitted + sum(trace.filtered_by_reason.values()) == trace.candidates_in_tenant
    assert trace.permitted == len(world.readable(user_id))
    allow_rules = {r.id for r in engine.bundle.policy.rules if r.effect is Effect.ALLOW}
    assert set(trace.permitting_rules) <= allow_rules
    assert list(trace.permitting_rules) == sorted(set(trace.permitting_rules))
    assert bool(trace.permitting_rules) is (trace.permitted > 0)
    assert trace.returned_ids == tuple(d.id for d in result.documents)
    assert len(trace.returned_ids) <= 7
    assert trace.policy_version == engine.version


def test_trace_names_the_rules_that_permitted(world: World) -> None:
    trace = world.search("kc-contractor-hollis", "furnace").trace
    assert trace.permitting_rules == (
        "contractor-read-assigned",
        "contractor-read-public",
        "contractor-read-sop",
    )
    assert set(trace.filtered_by_reason) == {"default_deny"}
    manager = world.search("jr-mgr-daniel", "payroll").trace
    assert manager.permitting_rules == ("manager-read", "manager-read-orders")


# Ranking


def test_ranking_prefers_title_hits_and_breaks_ties_by_id() -> None:
    with Store() as store:
        store.add_customer(
            Customer(id=KC, name="Kettle", domains=("kettlecreek.example",), timezone="UTC")
        )
        body = "Routine notes about the van."
        for doc_id, title, text in [
            ("kc-b", "Van checklist", body),
            ("kc-a", "Van checklist", body),
            ("kc-c", "Other notes", body + " Furnace furnace."),
            ("kc-d", "Furnace guide", body),
        ]:
            store.add_document(
                Document(id=doc_id, customer_id=KC, title=title, label="public", body=text)
            )
        index = DocumentIndex(store)
        index.rebuild()
        index.rebuild()  # Rebuilding twice must not duplicate rows.
        engine = PolicyEngine(load_bundles(POLICIES)[KC], _customer(store, KC))
        owner = User(
            id="kc-owner-dale",
            customer_id=KC,
            name="Dale",
            email="dale@kettlecreek.example",
            role=Role.OWNER,
        )
        result = index.search(engine, ctx(owner), "checklist", run_max_label=None)
        assert result.trace.returned_ids == ("kc-a", "kc-b")
        result = index.search(engine, ctx(owner), "furnace", run_max_label=None)
        assert result.trace.returned_ids[0] == "kc-d"


def test_documents_added_after_rebuild_are_not_searchable_until_rebuilt(world: World) -> None:
    store = Store()
    load_fixtures(store, FIXTURES)
    index = DocumentIndex(store)
    store.add_document(
        Document(id="kc-zz-new", customer_id=KC, title="Zeppelin", label="public", body="x")
    )
    owner = world.user("kc-owner-dale")
    assert (
        index.search(world.engines[KC], ctx(owner), "zeppelin", run_max_label=None).documents == ()
    )
    index.rebuild()
    found = index.search(world.engines[KC], ctx(owner), "zeppelin", run_max_label=None)
    assert found.trace.returned_ids == ("kc-zz-new",)
    store.close()


# Defence in depth


def test_invariant_check_fires_when_the_filter_is_wrong(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def permit_everything(
        self: DocumentIndex, engine: PolicyEngine, ctx: RunContext, tool: str, label: str | None
    ) -> _Filter:
        ids = tuple(d.id for d in world.docs(ctx.customer_id))
        return _Filter(len(ids), ids, {}, ())

    monkeypatch.setattr(DocumentIndex, "_permitted", permit_everything)
    # The owner may read everything, so a wrong filter changes nothing for them.
    assert world.search("kc-owner-dale", "payroll").documents
    with pytest.raises(RetrievalInvariantError, match="kc-payroll-summary"):
        world.search("kc-staff-nadine", "payroll")
    with pytest.raises(RetrievalInvariantError, match="kc-job-2047"):
        world.search("kc-contractor-hollis", "heat pump tune-up")


@pytest.mark.parametrize(
    ("user_id", "query"),
    [
        ("kc-staff-nadine", "furnace refrigerant thermostat warranty"),
        ("kc-contractor-hollis", "furnace refrigerant job dispatch"),
        ("jr-staff-priya", "refill shipping cold order"),
        ("jr-contractor-gideon", "delivery shipping cold return"),
    ],
)
def test_ranking_ignores_documents_the_run_cannot_read(user_id: str, query: str) -> None:
    world = _build_world()
    user = world.user(user_id)
    noisy = sanitize_query(query)[0]
    before = world.search(user_id, query)
    assert len(before.trace.returned_ids) >= 4

    # Many unreadable documents that lean hard on one query term: restricted ones in the
    # user's own customer and readable-looking public ones in the other customer.
    foreign = other(user.customer_id)
    for i in range(300):
        for customer_id, label in ((user.customer_id, "restricted"), (foreign, "public")):
            world.store.add_document(
                Document(
                    id=f"{customer_id[:2]}-noise-{label}-{i:03d}",
                    customer_id=customer_id,
                    title=f"{noisy} {noisy} notes",
                    label=label,
                    body=" ".join([noisy] * 40 + ["filler"] * i),
                )
            )
    world.index.rebuild()

    after = world.search(user_id, query)
    assert after.trace.returned_ids == before.trace.returned_ids
    assert after.trace.candidates_in_tenant == before.trace.candidates_in_tenant + 300
    assert after.trace.permitted == before.trace.permitted
    # The noise is indexed and found by someone who may read it.
    owner = next(u for u in world.users(user.customer_id) if u.role is Role.OWNER)
    assert any("-noise-" in i for i in world.search(owner.id, noisy).trace.returned_ids)
    world.store.close()


# Incremental indexing: add and remove give exactly what a full rebuild gives

VOCABULARY = (
    *("furnace", "refrigerant", "thermostat", "warranty", "dispatch", "coil", "filter", "duct"),
    *("heat", "pump", "tune-up", "refill", "shipping", "cold", "order", "delivery", "return"),
    *("payroll", "customer", "list", "contacts", "pricing", "margin", "notes", "shelf"),
    *("restock", "Caf\u00e9", "na\u00efve", "FURNACE", "Coil", "x2", "r-410a"),
)
LONG_TERM = "z" * 65


def _random_documents(seed: int, count: int) -> list[Document]:
    rng = random.Random(seed)
    labels = ["public", "internal", "confidential", "restricted"]
    documents: list[Document] = []
    for i in range(count):
        customer_id = KC if i % 2 else JR
        words = [rng.choice(VOCABULARY) for _ in range(rng.randint(0, 60))]
        if i % 7 == 0:
            words.append(LONG_TERM)
        documents.append(
            Document(
                id=f"{customer_id[:2]}-inc-{i:03d}",
                customer_id=customer_id,
                title=" ".join(rng.choice(VOCABULARY) for _ in range(rng.randint(1, 6))),
                label=rng.choice(labels),
                tags=frozenset({"note"}),
                body=" ".join(words),
            )
        )
    return documents


def _index_rows(store: Store) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    conn = store.connection
    postings = conn.execute("SELECT * FROM retrieval_postings ORDER BY term, doc_id").fetchall()
    lengths = conn.execute("SELECT * FROM retrieval_lengths ORDER BY doc_id").fetchall()
    return [tuple(r) for r in postings], [tuple(r) for r in lengths]


INCREMENTAL_QUERIES = [
    "furnace coil",
    "refill shipping cold order",
    "payroll customer list contacts",
    "cafe naive",
    "notes shelf restock pricing margin",
    "heat pump tune-up warranty dispatch",
]


def _assert_same_searches(first: World, second: World) -> None:
    for user_id in USER_IDS:
        for query in INCREMENTAL_QUERIES:
            assert first.search(user_id, query) == second.search(user_id, query), (user_id, query)


def test_add_ranks_exactly_like_a_full_rebuild() -> None:
    incremental = _build_world()
    rebuilt = _build_world()
    documents = _random_documents(seed=7, count=160)
    for document in documents:
        incremental.store.add_document(document)
        incremental.index.add(document)
        rebuilt.store.add_document(document)
    rebuilt.index.rebuild()
    assert _index_rows(incremental.store) == _index_rows(rebuilt.store)
    _assert_same_searches(incremental, rebuilt)
    # The added documents are really found, not just equally missing.
    found = incremental.search("kc-owner-dale", "furnace coil").trace.returned_ids
    assert any("-inc-" in doc_id for doc_id in found)

    # Changing a document and adding it again replaces its rows.
    changed = documents[1].model_copy(update={"title": "Brand new", "body": "zephyrine coil"})
    for world in (incremental, rebuilt):
        world.store.connection.execute(
            "UPDATE documents SET title = ?, body = ? WHERE id = ?",
            (changed.title, changed.body, changed.id),
        )
    incremental.index.add(changed)
    rebuilt.index.rebuild()
    assert _index_rows(incremental.store) == _index_rows(rebuilt.store)
    _assert_same_searches(incremental, rebuilt)

    # Removing a document drops its rows.
    gone = documents[2].id
    for world in (incremental, rebuilt):
        world.store.connection.execute("DELETE FROM documents WHERE id = ?", (gone,))
    incremental.index.remove(gone)
    rebuilt.index.rebuild()
    assert _index_rows(incremental.store) == _index_rows(rebuilt.store)
    _assert_same_searches(incremental, rebuilt)
    for world in (incremental, rebuilt):
        world.store.close()


def test_add_writes_only_that_documents_rows() -> None:
    world = _build_world()
    conn = world.store.connection
    before = _index_rows(world.store)
    note = Document(
        id="kc-inc-note",
        customer_id=KC,
        title="Coil coil",
        label="internal",
        body="Coil filter, " + LONG_TERM,
    )
    world.store.add_document(note)
    changes = conn.total_changes
    world.index.add(note)
    # One lengths row and two postings (coil, filter); the long term is counted in the body
    # length but never posted. Nothing else is deleted or written.
    assert conn.total_changes - changes == 1 + 2
    postings, lengths = _index_rows(world.store)
    assert [r for r in postings if r[1] != note.id] == before[0]
    assert [r for r in lengths if r[0] != note.id] == before[1]
    assert [r for r in postings if r[1] == note.id] == [
        ("coil", note.id, 2, 1),
        ("filter", note.id, 0, 1),
    ]
    assert [r for r in lengths if r[0] == note.id] == [(note.id, 2, 3)]
    # Adding it again rewrites the same rows: two deletes of postings, one of lengths,
    # and the same inserts.
    changes = conn.total_changes
    world.index.add(note)
    assert conn.total_changes - changes == (2 + 1) * 2
    assert _index_rows(world.store) == (postings, lengths)
    world.store.close()


def test_remove_of_an_unindexed_document_changes_nothing() -> None:
    world = _build_world()
    before = _index_rows(world.store)
    world.index.remove("kc-not-there")
    assert _index_rows(world.store) == before
    world.store.close()
