"""Source ACLs: a document shared through a source system is readable only by the
principals it is shared with, whatever the policy allows."""

import shutil
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentgate.audit.base import EventKind, MemorySink
from agentgate.clock import FixedClock
from agentgate.domain import Action, Customer, Document, Role, User
from agentgate.fixtures import load_fixtures, sync_fixtures
from agentgate.gate import Denied, Executed, Gate
from agentgate.policy.engine import DocumentFacts, PolicyEngine, Reason, Request
from agentgate.policy.schema import Effect, load_bundles
from agentgate.retrieval import DocumentIndex
from agentgate.store import Store
from conftest import FIXTURES, POLICIES, START, World

KC = "kettle-creek-hvac"
DISPATCH = "group:dispatch@kettlecreek.example"
ROSTER = "kc-acl-dispatch-roster"


def request(role: Role, acl: frozenset[str] | None, principals: frozenset[str]) -> Request:
    return Request(
        customer_id=KC,
        user_id="kc-x",
        role=role,
        tool="docs.read",
        action=Action.READ,
        document=DocumentFacts(customer_id=KC, label="internal", acl=acl),
        principals=principals,
    )


@pytest.fixture(scope="module")
def engine() -> PolicyEngine:
    customer = Customer(id=KC, name="Kettle", domains=("kettlecreek.example",), timezone="UTC")
    return PolicyEngine(load_bundles(POLICIES)[KC], customer)


# Engine


def test_a_principal_on_the_acl_lets_the_policy_decide(engine: PolicyEngine) -> None:
    decision = engine.decide(request(Role.STAFF, frozenset({DISPATCH}), frozenset({DISPATCH})))

    assert (decision.effect, decision.rule_ids) == (Effect.ALLOW, ("staff-read",))


def test_the_acl_is_a_ceiling_no_rule_can_widen(engine: PolicyEngine) -> None:
    # owner-read allows every label; the ACL still wins.
    decision = engine.decide(request(Role.OWNER, frozenset({DISPATCH}), frozenset({"anyone"})))

    assert (decision.effect, decision.reason) == (Effect.DENY, Reason.ACL_DENIED)
    assert decision.rule_ids == ()


def test_an_empty_acl_is_shared_with_nobody(engine: PolicyEngine) -> None:
    decision = engine.decide(request(Role.OWNER, frozenset(), frozenset({"anyone", DISPATCH})))

    assert decision.reason is Reason.ACL_DENIED


def test_a_request_without_principals_cannot_read_an_acl_document(engine: PolicyEngine) -> None:
    decision = engine.decide(request(Role.OWNER, frozenset({"anyone"}), frozenset()))

    assert decision.reason is Reason.ACL_DENIED


def test_a_document_without_an_acl_is_decided_by_policy_alone(engine: PolicyEngine) -> None:
    decision = engine.decide(request(Role.STAFF, None, frozenset()))

    assert decision.effect is Effect.ALLOW


# Models


def test_user_principals_derive_from_email_and_memberships() -> None:
    user = User(
        id="kc-x",
        customer_id=KC,
        name="X",
        email="x@kettlecreek.example",
        role=Role.STAFF,
        memberships=frozenset({DISPATCH, "slack:C123"}),
    )

    assert user.principals == {
        "anyone",
        "user:x@kettlecreek.example",
        "domain:kettlecreek.example",
        DISPATCH,
        "slack:C123",
    }


@pytest.mark.parametrize("principal", ["", "Group:x", "group:", "two words:x", "nokind"])
def test_malformed_principals_are_refused(principal: str) -> None:
    with pytest.raises(ValidationError):
        Document(id="kc-x", customer_id=KC, title="T", label="internal", acl=[principal], body="")


# Store


def test_store_keeps_no_acl_and_an_empty_acl_apart(store: Store) -> None:
    store.add_customer(Customer(id=KC, name="K", domains=("kettlecreek.example",), timezone="UTC"))
    for doc_id, acl in [("kc-none", None), ("kc-empty", frozenset()), ("kc-one", {DISPATCH})]:
        store.add_document(
            Document(id=doc_id, customer_id=KC, title="T", label="internal", acl=acl, body="")
        )

    acls = {d.id: d.acl for d in store.list_documents(KC)}

    assert acls == {"kc-empty": frozenset(), "kc-none": None, "kc-one": frozenset({DISPATCH})}


def test_a_store_from_before_acls_opens_with_none_and_no_memberships(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with Store(path) as store:
        load_fixtures(store, FIXTURES)
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE documents DROP COLUMN acl")
    conn.execute("ALTER TABLE users DROP COLUMN memberships")
    conn.commit()
    conn.close()

    with Store(path) as reopened:
        document = reopened.get_document("kc-customer-list")
        user = reopened.get_user("kc-staff-nadine")

    assert document is not None and document.acl is None
    assert user is not None and user.memberships == frozenset()


# Fixtures and the gate, end to end


def acl_fixtures(tmp_path: Path, nadine_memberships: list[str]) -> Path:
    """The shipped fixtures, plus one internal document shared with the dispatch group, and
    Nadine (staff) in that group."""
    root = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, root, dirs_exist_ok=True)
    (root / KC / "docs" / f"{ROSTER}.md").write_text(
        "---\ntitle: Dispatch roster\nlabel: internal\ntags: [dispatch]\n"
        f"acl: [{DISPATCH}]\n---\nOn-call rota for the zephyrine week.\n"
    )
    users = root / KC / "users.yaml"
    text = users.read_text()
    nadine = "  email: nadine@kettlecreek.example\n"
    assert nadine in text
    users.write_text(text.replace(nadine, nadine + f"  memberships: {nadine_memberships}\n"))
    return root


def acl_world(fixtures: Path) -> World:
    store = Store()
    load_fixtures(store, fixtures)
    index = DocumentIndex(store)
    index.rebuild()
    engines = {
        customer_id: PolicyEngine(bundle, store.get_customer(customer_id))  # type: ignore[arg-type]
        for customer_id, bundle in load_bundles(POLICIES).items()
    }
    sink = MemorySink()
    clock = FixedClock(START)
    gate = Gate(store=store, engines=engines, sink=sink, clock=clock, index=index)
    return World(store=store, index=index, engines=engines, sink=sink, clock=clock, gate=gate)


def read(world: World, user_id: str) -> object:
    ctx = world.run(user_id)
    return world.gate.call_tool(ctx, "docs.read", Action.READ, {"document_id": ROSTER})


def search_ids(world: World, user_id: str) -> list[str]:
    ctx = world.run(user_id)
    outcome = world.gate.call_tool(ctx, "docs.search", Action.READ, {"query": "zephyrine"})
    assert isinstance(outcome, Executed)
    return [doc["id"] for doc in outcome.content["documents"]]  # type: ignore[index, union-attr]


def test_a_member_of_the_shared_group_reads_the_document(tmp_path: Path) -> None:
    world = acl_world(acl_fixtures(tmp_path, [DISPATCH]))

    assert isinstance(read(world, "kc-staff-nadine"), Executed)
    assert search_ids(world, "kc-staff-nadine") == [ROSTER]


@pytest.mark.parametrize("user_id", ["kc-staff-colby", "kc-owner-dale"])
def test_anyone_else_gets_not_found_and_the_audit_says_why(tmp_path: Path, user_id: str) -> None:
    world = acl_world(acl_fixtures(tmp_path, [DISPATCH]))

    assert read(world, user_id) == Denied("not_found")
    denial = world.last(EventKind.DENY)
    assert (denial.reason, denial.subject) == ("acl_denied", ROSTER)


@pytest.mark.parametrize("user_id", ["kc-staff-colby", "kc-owner-dale"])
def test_search_never_returns_or_counts_toward_a_document_outside_the_acl(
    tmp_path: Path, user_id: str
) -> None:
    world = acl_world(acl_fixtures(tmp_path, [DISPATCH]))

    assert search_ids(world, user_id) == []
    retrieval = world.last(EventKind.RETRIEVAL)
    assert retrieval.details["filtered_by_reason"]["acl_denied"] == 1  # type: ignore[index]


def test_fixture_sync_picks_up_a_membership_change(tmp_path: Path) -> None:
    fixtures = acl_fixtures(tmp_path, [])
    store = Store()
    load_fixtures(store, fixtures)
    users = fixtures / KC / "users.yaml"
    users.write_text(users.read_text().replace("memberships: []", f"memberships: [{DISPATCH}]"))

    synced = sync_fixtures(store, fixtures)

    assert ("kc-staff-nadine", ("memberships",)) in synced.users_updated
    nadine = store.get_user("kc-staff-nadine")
    assert nadine is not None and DISPATCH in nadine.principals


def test_a_store_from_before_acls_reads_read_only(tmp_path: Path) -> None:
    # Read-only opens never migrate (audit show, check-store), so rows without the new
    # columns must still read: no ACL, no memberships.
    path = tmp_path / "old.db"
    with Store(path) as store:
        load_fixtures(store, FIXTURES)
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE documents DROP COLUMN acl")
    conn.execute("ALTER TABLE users DROP COLUMN memberships")
    conn.commit()
    conn.close()

    with Store(path, readonly=True) as reader:
        document = reader.get_document("kc-customer-list")
        user = reader.get_user("kc-staff-nadine")

    assert document is not None and document.acl is None
    assert user is not None and user.memberships == frozenset()
