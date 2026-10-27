import hashlib
import json
import sqlite3
import stat
from pathlib import Path

import pytest

from agentgate.audit.panoptes import PanoptesSink
from agentgate.models import Customer, Document, Role, User, UserStatus
from agentgate.store import Store, StoreError, make_private_dirs, table_names


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


# File permissions


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_a_new_store_file_is_owner_only(tmp_path: Path, loose_umask: None) -> None:
    path = tmp_path / "agentgate.db"
    with Store(path) as store:
        assert store.tightened_from is None
        store.add_customer(_customer())
    assert _mode(path) == 0o600


def test_a_world_readable_store_is_tightened_on_open(tmp_path: Path, loose_umask: None) -> None:
    path = tmp_path / "agentgate.db"
    Store(path).close()
    path.chmod(0o644)
    journal = tmp_path / "agentgate.db-journal"
    journal.write_bytes(b"")
    journal.chmod(0o666)
    with Store(path) as store:
        assert store.tightened_from == 0o644
    assert _mode(path) == 0o600
    assert _mode(journal) == 0o600
    # Already owner-only: nothing to report the second time.
    with Store(path) as store:
        assert store.tightened_from is None


def test_tightening_keeps_a_read_only_store_read_only(tmp_path: Path) -> None:
    path = tmp_path / "agentgate.db"
    Store(path).close()
    path.chmod(0o444)
    try:
        with Store(path) as store:
            assert store.tightened_from == 0o444
            assert _mode(path) == 0o400
            # Reads still work; writes fail as SQLite errors, for callers to report.
            assert store.list_customers() == []
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                store.add_customer(_customer())
    finally:
        path.chmod(0o600)


def test_memory_stores_touch_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with Store() as store, Store("") as temp:
        assert store.tightened_from is None and temp.tightened_from is None
    assert list(tmp_path.iterdir()) == []


def test_make_private_dirs_creates_owner_only_and_leaves_existing_alone(
    tmp_path: Path, loose_umask: None
) -> None:
    existing = tmp_path / "shared"
    existing.mkdir(mode=0o755)
    make_private_dirs(existing / "a" / "b")
    assert _mode(existing) == 0o755
    assert _mode(existing / "a") == 0o700
    assert _mode(existing / "a" / "b") == 0o700
    make_private_dirs(existing)
    assert _mode(existing) == 0o755


# Refusing an audit database


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_panoptes_audit_database_is_refused_before_any_write(tmp_path: Path) -> None:
    audit = tmp_path / "audit.db"
    PanoptesSink(audit).close()
    before = _digest(audit)
    tables = table_names(audit)
    assert tables is not None and "events" in tables
    with pytest.raises(StoreError, match="is a Panoptes audit database, not an agentgate store"):
        Store(audit)
    assert _digest(audit) == before
    assert table_names(audit) == tables


def test_the_store_never_has_an_events_table(tmp_path: Path) -> None:
    path = tmp_path / "agentgate.db"
    Store(path).close()
    tables = table_names(path)
    assert tables is not None and "customers" in tables and "events" not in tables


def test_table_names_of_missing_and_non_sqlite_files(tmp_path: Path) -> None:
    assert table_names(tmp_path / "missing.db") is None
    junk = tmp_path / "junk.db"
    junk.write_text("not a database, just text that is long enough to have a header" * 4)
    assert table_names(junk) is None


# Updates and operator suspension


def _seeded(store: Store) -> None:
    store.add_customer(_customer())
    store.add_user(_user("dana", role=Role.OWNER))
    store.add_user(_user("eli"))
    store.add_document(_document("plan"))


def test_updates_replace_fields_but_never_move_tenants(store: Store) -> None:
    _seeded(store)
    store.add_customer(_customer("beta"))
    renamed = _customer().model_copy(update={"name": "Acme Two", "timezone": "America/Denver"})
    store.update_customer(renamed)
    assert store.get_customer("acme") == renamed

    changed = _user("eli", role=Role.MANAGER, status=UserStatus.SUSPENDED, name="Eli Two")
    store.update_user(changed)
    assert store.get_user("eli") == changed

    doc = _document("plan", label="confidential", tags={"assigned"}, assigned_user_ids={"eli"})
    store.update_document(doc)
    assert store.get_document("plan") == doc

    with pytest.raises(StoreError, match="unknown user 'eli' in customer 'beta'"):
        store.update_user(_user("eli", customer_id="beta"))
    with pytest.raises(StoreError, match="unknown document 'plan' in customer 'beta'"):
        store.update_document(_document("plan", customer_id="beta"))
    with pytest.raises(StoreError, match="unknown customer 'gamma'"):
        store.update_customer(_customer("gamma"))
    assert store.get_user("eli") == changed


def test_suspend_and_reinstate_by_operator(store: Store) -> None:
    _seeded(store)
    assert not store.operator_suspended("eli")
    suspended = store.suspend_user("eli")
    assert suspended is not None and suspended.status is UserStatus.SUSPENDED
    assert store.operator_suspended("eli")
    # A plain update, as fixture sync makes, leaves the flag alone.
    store.update_user(_user("eli", status=UserStatus.ACTIVE))
    assert store.operator_suspended("eli")
    reinstated = store.reinstate_user("eli")
    # Reinstating clears the flag; the status is for fixture sync to set.
    assert reinstated is not None and not store.operator_suspended("eli")
    assert store.suspend_user("nobody") is None
    assert store.reinstate_user("nobody") is None
    assert not store.operator_suspended("nobody")


def test_suspension_reaches_another_connection_at_once(tmp_path: Path) -> None:
    path = tmp_path / "agentgate.db"
    with Store(path) as first, Store(path) as second:
        _seeded(first)
        second.suspend_user("eli")
        user = first.get_user("eli")
        assert user is not None and user.status is UserStatus.SUSPENDED


def test_a_store_without_the_operator_flag_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id TEXT PRIMARY KEY, name TEXT NOT NULL, domains TEXT NOT NULL,
            timezone TEXT NOT NULL) STRICT;
        CREATE TABLE users (id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES
            customers(id), name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, role TEXT NOT NULL,
            status TEXT NOT NULL) STRICT;
        INSERT INTO customers VALUES ('acme', 'Acme', '["acme.example"]', 'America/Chicago');
        INSERT INTO users VALUES ('eli', 'acme', 'Eli', 'eli@acme.example', 'staff', 'active');
        """
    )
    conn.close()
    with Store(path) as store:
        assert store.get_user("eli") == _user("eli", name="Eli")
        assert not store.operator_suspended("eli")
        store.suspend_user("eli")
        assert store.operator_suspended("eli")
