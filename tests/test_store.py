import json
import sqlite3
from pathlib import Path

import pytest

from agentgate.models import Customer, Document, Role, User, UserStatus
from agentgate.store import Store


def _customer(customer_id: str = "acme") -> Customer:
    return Customer(
        id=customer_id,
        name=customer_id.title(),
        domains=(f"{customer_id}.example", f"mail.{customer_id}.example"),
        timezone="America/Chicago",
    )


def _user(user_id: str, customer_id: str = "acme", **overrides: object) -> User:
    data: dict[str, object] = {
        "id": user_id,
        "customer_id": customer_id,
        "name": user_id.title(),
        "email": f"{user_id}@{customer_id}.example",
        "role": Role.STAFF,
    }
    return User.model_validate(data | overrides)


def _document(doc_id: str, customer_id: str = "acme", **overrides: object) -> Document:
    data: dict[str, object] = {
        "id": doc_id,
        "customer_id": customer_id,
        "title": doc_id.title(),
        "label": "internal",
        "body": f"Body of {doc_id}.\n",
    }
    return Document.model_validate(data | overrides)


@pytest.fixture
def store() -> Store:
    return Store()


def test_round_trips_every_model(store: Store) -> None:
    customer = _customer()
    user = _user("dana", role=Role.OWNER, status=UserStatus.SUSPENDED)
    doc = _document(
        "plan",
        tags={"zeta", "assigned", "alpha"},
        assigned_user_ids={"dana"},
        label="confidential",
    )
    store.add_customer(customer)
    store.add_user(user)
    store.add_document(doc)

    assert store.get_customer("acme") == customer
    assert store.get_user("dana") == user
    assert store.get_document("plan") == doc


def test_missing_ids_return_none(store: Store) -> None:
    assert store.get_customer("nope") is None
    assert store.get_user("nope") is None
    assert store.get_document("nope") is None


def test_lists_are_scoped_by_customer_and_ordered_by_id(store: Store) -> None:
    for customer_id in ("zed", "acme"):
        store.add_customer(_customer(customer_id))
    for user_id in ("carl", "abe", "bea"):
        store.add_user(_user(user_id))
    store.add_user(_user("zoe", customer_id="zed"))
    for doc_id in ("memo", "contract"):
        store.add_document(_document(doc_id))
    store.add_document(_document("zed-doc", customer_id="zed"))

    assert [c.id for c in store.list_customers()] == ["acme", "zed"]
    assert [u.id for u in store.list_users("acme")] == ["abe", "bea", "carl"]
    assert [d.id for d in store.list_documents("acme")] == ["contract", "memo"]
    assert [d.id for d in store.list_documents("zed")] == ["zed-doc"]
    assert store.list_users("nobody") == []


def test_get_document_is_not_tenant_scoped(store: Store) -> None:
    # Tenant enforcement is the gate's job; the store returns any document by id.
    store.add_customer(_customer("acme"))
    store.add_customer(_customer("other"))
    store.add_document(_document("secret", customer_id="other"))

    doc = store.get_document("secret")
    assert doc is not None
    assert doc.customer_id == "other"


def test_sets_are_stored_as_sorted_json(tmp_path: Path) -> None:
    db = tmp_path / "agentgate.db"
    with Store(db) as store:
        store.add_customer(_customer())
        store.add_user(_user("bo"))
        store.add_user(_user("al"))
        store.add_document(
            _document("plan", tags={"zeta", "assigned", "alpha"}, assigned_user_ids={"bo", "al"})
        )

    conn = sqlite3.connect(db)
    try:
        query = "SELECT tags, assigned_user_ids FROM documents WHERE id = 'plan'"
        tags, assigned = conn.execute(query).fetchone()
    finally:
        conn.close()
    assert tags == json.dumps(["alpha", "assigned", "zeta"])
    assert assigned == json.dumps(["al", "bo"])


def test_data_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "agentgate.db"
    with Store(db) as store:
        store.add_customer(_customer())
    with Store(db) as store:
        assert store.get_customer("acme") == _customer()


def test_foreign_keys_are_enforced(store: Store) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.add_user(_user("dana", customer_id="ghost"))
    with pytest.raises(sqlite3.IntegrityError):
        store.add_document(_document("memo", customer_id="ghost"))


def test_ids_and_emails_are_unique(store: Store) -> None:
    store.add_customer(_customer())
    store.add_user(_user("dana"))
    with pytest.raises(sqlite3.IntegrityError):
        store.add_customer(_customer())
    with pytest.raises(sqlite3.IntegrityError):
        store.add_user(_user("dana", email="other@acme.example"))
    with pytest.raises(sqlite3.IntegrityError):
        store.add_user(_user("dana2", email="dana@acme.example"))


def test_transaction_commits_on_success(store: Store) -> None:
    with store.transaction():
        store.add_customer(_customer())
        store.add_user(_user("dana"))
    assert store.get_user("dana") is not None


def test_transaction_rolls_back_everything_on_error(store: Store) -> None:
    with pytest.raises(sqlite3.IntegrityError), store.transaction():
        store.add_customer(_customer())
        store.add_user(_user("dana"))
        store.add_user(_user("dana"))
    assert store.list_customers() == []
    assert store.get_user("dana") is None


def test_nested_transactions_are_rejected(store: Store) -> None:
    with store.transaction(), pytest.raises(RuntimeError, match="nested"), store.transaction():
        pass


def test_context_manager_closes_connection() -> None:
    with Store() as store:
        store.add_customer(_customer())
    with pytest.raises(sqlite3.ProgrammingError):
        store.list_customers()


def test_transactions_serialize_read_check_write_across_threads() -> None:
    import threading

    store = Store()
    with store.transaction():
        store.connection.execute("CREATE TABLE counter (n INTEGER NOT NULL) STRICT")
        store.connection.execute("INSERT INTO counter VALUES (0)")
    barrier = threading.Barrier(16)

    def bump() -> None:
        barrier.wait()
        for _ in range(25):
            with store.transaction():
                (n,) = store.connection.execute("SELECT n FROM counter").fetchone()
                store.connection.execute("UPDATE counter SET n = ?", (n + 1,))

    threads = [threading.Thread(target=bump) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with store.locked():
        (n,) = store.connection.execute("SELECT n FROM counter").fetchone()
    assert n == 16 * 25
