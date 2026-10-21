from pathlib import Path

import pytest
import yaml

from agentgate.fixtures import (
    CANARY_RE,
    CustomerCounts,
    FixtureError,
    FixtureSummary,
    find_canaries,
    load_fixtures,
)
from agentgate.models import Customer, Document, Role, User, UserStatus
from agentgate.store import Store

CANARY_A = "AGCANARY-0123456789AB"
CANARY_B = "AGCANARY-BA9876543210"
CANARY_C = "AGCANARY-CCCCCCCCCCCC"

# Small builders that write the fixture format from fixtures/README.md.


def write_customer(root: Path, customer_id: str, **overrides: object) -> Path:
    data: dict[str, object] = {
        "id": customer_id,
        "name": customer_id.title(),
        "domains": [f"{customer_id}.example"],
        "timezone": "America/Chicago",
    }
    path = root / customer_id / "customer.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data | overrides, sort_keys=False))
    return path


def user_entry(user_id: str, customer_id: str, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": user_id,
        "name": user_id.title(),
        "email": f"{user_id}@{customer_id}.example",
        "role": "staff",
    }
    return data | overrides


def write_users(root: Path, customer_id: str, entries: list[dict[str, object]]) -> Path:
    path = root / customer_id / "users.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(entries, sort_keys=False))
    return path


def write_doc_text(root: Path, customer_id: str, doc_id: str, text: str) -> Path:
    path = root / customer_id / "docs" / f"{doc_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_doc(
    root: Path,
    customer_id: str,
    doc_id: str,
    body: str = "Body.\n",
    **front: object,
) -> Path:
    meta: dict[str, object] = {"title": doc_id.title(), "label": "internal"} | front
    text = f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\n{body}"
    return write_doc_text(root, customer_id, doc_id, text)


def secret_body(canary: str) -> str:
    return f"Margins are thin.\n\nCanary: {canary}\n"


def minimal_tree(root: Path, customer_id: str = "acme") -> None:
    write_customer(root, customer_id)
    write_users(root, customer_id, [user_entry(f"{customer_id}-owner", customer_id)])


def load(root: Path) -> FixtureSummary:
    return load_fixtures(Store(), root)


# Happy path


def test_loads_two_customers_and_reads_back_equal_models(tmp_path: Path) -> None:
    root = tmp_path
    (root / "README.md").write_text("Not a customer.\n")

    acme = Customer(
        id="acme",
        name="Acme Heating",
        domains=("acme.example", "mail.acme.example"),
        timezone="America/Chicago",
    )
    beta = Customer(id="beta", name="Beta Shop", domains=("beta.example",), timezone="UTC")
    users = [
        User(
            id="ac-owner",
            customer_id="acme",
            name="Rosa",
            email="rosa@acme.example",
            role=Role.OWNER,
        ),
        User(
            id="ac-tech",
            customer_id="acme",
            name="Tom",
            email="tom@acme.example",
            role=Role.CONTRACTOR,
            status=UserStatus.SUSPENDED,
        ),
        User(
            id="bt-owner",
            customer_id="beta",
            name="Bea",
            email="bea@beta.example",
            role=Role.MANAGER,
        ),
    ]
    docs = [
        Document(
            id="ac-handbook",
            customer_id="acme",
            title="Handbook",
            label="public",
            body="Welcome.\n",
        ),
        Document(
            id="ac-pricing",
            customer_id="acme",
            title="Pricing",
            label="confidential",
            tags=frozenset({"pricing", "assigned"}),
            assigned_user_ids=frozenset({"ac-tech"}),
            body=secret_body(CANARY_A),
        ),
        Document(
            id="bt-payroll",
            customer_id="beta",
            title="Payroll",
            label="restricted",
            tags=frozenset({"hr"}),
            body=secret_body(CANARY_B),
        ),
    ]

    for customer in (acme, beta):
        write_customer(
            root,
            customer.id,
            name=customer.name,
            domains=list(customer.domains),
            timezone=customer.timezone,
        )
        entries = [
            user.model_dump(mode="json", exclude={"customer_id"})
            for user in users
            if user.customer_id == customer.id
        ]
        write_users(root, customer.id, entries)
    for doc in docs:
        write_doc(
            root,
            doc.customer_id,
            doc.id,
            body=doc.body,
            title=doc.title,
            label=doc.label,
            tags=sorted(doc.tags),
            assigned_user_ids=sorted(doc.assigned_user_ids),
        )

    store = Store()
    summary = load_fixtures(store, root)

    assert summary == FixtureSummary(
        customers=(CustomerCounts("acme", 2, 2), CustomerCounts("beta", 1, 1))
    )
    assert summary.total_users == 3
    assert summary.total_documents == 3
    assert store.list_customers() == [acme, beta]
    assert store.list_users("acme") + store.list_users("beta") == users
    assert store.list_documents("acme") + store.list_documents("beta") == docs


def test_customer_without_docs_dir_or_users_loads(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    (tmp_path / "acme" / "users.yaml").write_text("[]\n")
    assert load(tmp_path) == FixtureSummary(customers=(CustomerCounts("acme", 0, 0),))


def test_front_matter_body_is_everything_after_the_closing_line(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    write_doc_text(tmp_path, "acme", "notes", "---\ntitle: Notes\nlabel: internal\n---\nA\n---\nB")
    store = Store()
    load_fixtures(store, tmp_path)
    doc = store.get_document("notes")
    assert doc is not None
    assert doc.body == "A\n---\nB"


# Atomicity


def test_invalid_later_file_leaves_store_empty(tmp_path: Path) -> None:
    minimal_tree(tmp_path, "acme")
    write_doc(tmp_path, "acme", "ac-memo")
    minimal_tree(tmp_path, "zeta")
    write_doc_text(tmp_path, "zeta", "zt-broken", "no front matter\n")

    store = Store()
    with pytest.raises(FixtureError, match=r"zt-broken\.md"):
        load_fixtures(store, tmp_path)
    assert store.list_customers() == []
    assert store.get_document("ac-memo") is None


def test_store_conflict_mid_load_rolls_back_the_whole_load(tmp_path: Path) -> None:
    minimal_tree(tmp_path, "acme")
    write_doc(tmp_path, "acme", "ac-memo")
    minimal_tree(tmp_path, "zeta")

    store = Store()
    existing = Customer(id="zeta", name="Zeta", domains=("zeta.example",), timezone="UTC")
    store.add_customer(existing)

    with pytest.raises(FixtureError, match=r"zeta/customer\.yaml"):
        load_fixtures(store, tmp_path)
    assert store.list_customers() == [existing]
    assert store.get_document("ac-memo") is None


# Structure and identity


def test_missing_customer_yaml(tmp_path: Path) -> None:
    write_users(tmp_path, "acme", [])
    with pytest.raises(FixtureError, match=r"acme/customer\.yaml.*missing"):
        load(tmp_path)


def test_missing_users_yaml(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    with pytest.raises(FixtureError, match=r"acme/users\.yaml.*missing"):
        load(tmp_path)


def test_customer_id_must_match_directory(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme", id="other")
    with pytest.raises(FixtureError, match=r"customer\.yaml.*does not match directory"):
        load(tmp_path)


def test_users_must_not_set_customer_id(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    write_users(tmp_path, "acme", [user_entry("dana", "acme") | {"customer_id": "acme"}])
    with pytest.raises(FixtureError, match=r"users\.yaml.*customer_id"):
        load(tmp_path)


def test_users_yaml_must_be_a_list(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    (tmp_path / "acme" / "users.yaml").write_text("id: dana\n")
    with pytest.raises(FixtureError, match=r"users\.yaml.*list"):
        load(tmp_path)


def test_documents_must_not_set_customer_id(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    text = "---\ntitle: Memo\nlabel: internal\ncustomer_id: acme\n---\nBody\n"
    write_doc_text(tmp_path, "acme", "memo", text)
    with pytest.raises(FixtureError, match=r"memo\.md.*customer_id"):
        load(tmp_path)


def test_documents_must_not_set_id(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "memo", id="other")
    with pytest.raises(FixtureError, match=r"memo\.md.*'id'"):
        load(tmp_path)


def test_duplicate_user_ids_across_customers(tmp_path: Path) -> None:
    minimal_tree(tmp_path, "acme")
    write_customer(tmp_path, "beta")
    write_users(tmp_path, "beta", [user_entry("acme-owner", "beta")])
    with pytest.raises(FixtureError, match=r"beta/users\.yaml.*duplicate user id"):
        load(tmp_path)


def test_duplicate_user_ids_within_a_file(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    entries = [user_entry("dana", "acme"), user_entry("dana", "acme", email="d2@acme.example")]
    write_users(tmp_path, "acme", entries)
    with pytest.raises(FixtureError, match="duplicate user id 'dana'"):
        load(tmp_path)


def test_duplicate_emails(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    entries = [user_entry("dana", "acme"), user_entry("dan", "acme", email="dana@acme.example")]
    write_users(tmp_path, "acme", entries)
    with pytest.raises(FixtureError, match="duplicate email"):
        load(tmp_path)


def test_duplicate_document_ids_across_customers(tmp_path: Path) -> None:
    minimal_tree(tmp_path, "acme")
    minimal_tree(tmp_path, "beta")
    write_doc(tmp_path, "acme", "memo")
    write_doc(tmp_path, "beta", "memo")
    with pytest.raises(FixtureError, match=r"beta/docs/memo\.md.*duplicate document id"):
        load(tmp_path)


def test_unknown_assigned_user(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "job", tags=["assigned"], assigned_user_ids=["ghost"])
    with pytest.raises(FixtureError, match=r"job\.md.*'ghost' does not exist"):
        load(tmp_path)


def test_cross_customer_assigned_user(tmp_path: Path) -> None:
    minimal_tree(tmp_path, "acme")
    minimal_tree(tmp_path, "beta")
    write_doc(tmp_path, "acme", "job", tags=["assigned"], assigned_user_ids=["beta-owner"])
    with pytest.raises(FixtureError, match=r"job\.md.*belongs to customer 'beta'"):
        load(tmp_path)


def test_non_markdown_file_in_docs_is_rejected(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "memo")
    (tmp_path / "acme" / "docs" / "memo.txt").write_text("stray\n")
    with pytest.raises(FixtureError, match=r"memo\.txt"):
        load(tmp_path)


# Parsing


@pytest.mark.parametrize(
    "text",
    [
        "title: Memo\nlabel: internal\n---\nBody\n",
        " ---\ntitle: Memo\nlabel: internal\n---\nBody\n",
        "---\ntitle: Memo\nlabel: internal\nBody without a closing line\n",
        "---\n---\nBody\n",
        "---\n- just\n- a list\n---\nBody\n",
        "---\ntitle: [unclosed\n---\nBody\n",
    ],
    ids=["no-opening", "indented-opening", "no-closing", "empty", "not-mapping", "bad-yaml"],
)
def test_malformed_front_matter(tmp_path: Path, text: str) -> None:
    minimal_tree(tmp_path)
    write_doc_text(tmp_path, "acme", "memo", text)
    with pytest.raises(FixtureError, match=r"memo\.md"):
        load(tmp_path)


def test_duplicate_yaml_keys_in_front_matter(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    text = "---\ntitle: Memo\nlabel: internal\nlabel: public\n---\nBody\n"
    write_doc_text(tmp_path, "acme", "memo", text)
    with pytest.raises(FixtureError, match=r"memo\.md.*duplicate key 'label'"):
        load(tmp_path)


def test_duplicate_yaml_keys_in_customer_yaml(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    path = tmp_path / "acme" / "customer.yaml"
    path.write_text(path.read_text() + "name: Again\n")
    with pytest.raises(FixtureError, match=r"customer\.yaml.*duplicate key 'name'"):
        load(tmp_path)


def test_yaml_tags_for_python_objects_are_rejected(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    (tmp_path / "acme" / "users.yaml").write_text("!!python/object/apply:os.getcwd []\n")
    with pytest.raises(FixtureError, match=r"users\.yaml.*invalid YAML"):
        load(tmp_path)


def test_validation_errors_name_the_file(tmp_path: Path) -> None:
    write_customer(tmp_path, "acme")
    write_users(tmp_path, "acme", [user_entry("dana", "acme", role="admin")])
    with pytest.raises(FixtureError, match=r"users\.yaml.*role"):
        load(tmp_path)


def test_invalid_document_id_from_file_name(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "Bad_Name")
    with pytest.raises(FixtureError, match=r"Bad_Name\.md"):
        load(tmp_path)


def test_assigned_documents_need_the_assigned_tag(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "job", assigned_user_ids=["acme-owner"])
    with pytest.raises(FixtureError, match=r"job\.md.*assigned"):
        load(tmp_path)


# Canaries


@pytest.mark.parametrize("label", ["confidential", "restricted"])
def test_secret_documents_need_a_canary(tmp_path: Path, label: str) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "memo", label=label, body="No canary here.\n")
    with pytest.raises(FixtureError, match=r"memo\.md.*exactly one"):
        load(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        f"Canary: {CANARY_A}\nCanary: {CANARY_B}\n",
        f"Canary: {CANARY_A}\nAlso mentions {CANARY_B} inline.\n",
        "Canary: AGCANARY-0123456789ab\n",
        f"  Canary: {CANARY_A}\n",
        f"Canary: {CANARY_A}0\n",
    ],
    ids=["two-lines", "extra-token", "lowercase-hex", "indented", "too-long"],
)
def test_secret_documents_reject_malformed_or_extra_canaries(tmp_path: Path, body: str) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "memo", label="restricted", body=body)
    with pytest.raises(FixtureError, match=r"memo\.md"):
        load(tmp_path)


@pytest.mark.parametrize("label", ["public", "internal"])
def test_plain_documents_must_not_contain_canaries(tmp_path: Path, label: str) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "memo", label=label, body="Mentions AGCANARY- in passing.\n")
    with pytest.raises(FixtureError, match=rf"memo\.md.*{label} documents must not"):
        load(tmp_path)


def test_canaries_are_unique_across_customers(tmp_path: Path) -> None:
    minimal_tree(tmp_path, "acme")
    minimal_tree(tmp_path, "beta")
    write_doc(tmp_path, "acme", "ac-memo", label="confidential", body=secret_body(CANARY_C))
    write_doc(tmp_path, "beta", "bt-memo", label="restricted", body=secret_body(CANARY_C))
    with pytest.raises(FixtureError, match=r"bt-memo\.md.*also used in .*ac-memo\.md"):
        load(tmp_path)


def test_unknown_labels_are_rejected(tmp_path: Path) -> None:
    minimal_tree(tmp_path)
    write_doc(tmp_path, "acme", "memo", label="secret")
    with pytest.raises(FixtureError, match=r"memo\.md.*unknown label 'secret'"):
        load(tmp_path)


def test_canary_helpers() -> None:
    text = f"Canary: {CANARY_A}\nleaked {CANARY_B}, and again {CANARY_A}.\nAGCANARY-short"
    assert find_canaries(text) == [CANARY_A, CANARY_B, CANARY_A]
    assert CANARY_RE.findall(text) == [CANARY_A]
    assert find_canaries("nothing to see") == []
