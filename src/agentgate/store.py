"""SQLite store for customers, users and documents.

The store is a thin persistence layer: it saves and returns validated models and enforces
identity constraints (unique ids and emails, foreign keys). It does not decide who may see
what; tenant and permission checks belong to the gate layer above it.

Other components (spend ledger, approvals, retrieval index) keep their own tables in the
same database through `connection`, always inside `transaction()` or `locked()`. One
connection is shared across threads behind a re-entrant lock, because with autocommit off a
statement from a second thread would otherwise land inside the first thread's transaction.

The database holds every document, approval argument and email body, so a store file is
owner-only: a new file is created 0600, and an existing file that other users can read is
tightened to 0600 when it is opened (`tightened_from` keeps the mode it had, for callers
that want to say so). A database that is a Panoptes audit log is refused before anything is
written to it, so a mistyped path cannot mix store tables into the audit log.

A user suspended by an operator (`suspend_user`, behind `agentgate users suspend`) carries a
flag, `suspended_by_operator`. Fixture sync never reactivates a flagged user; only
`reinstate_user` clears the flag, after which the user's status comes from fixtures again.

`policy_snapshots` keeps the canonical JSON of every policy bundle a world has loaded, keyed
by its version, so the rules behind any version an audit event names can be read back
(`agentgate audit policy`). A snapshot is written once and never replaced; its sha256 is
also on the audit chain, in the `policy_loaded` event, so an edited snapshot shows.

`Store(path, readonly=True)` opens an existing store read-only for commands that must not
change it (`agentgate audit show`): no file is created or tightened, no schema is written.
"""

import contextlib
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from agentgate.models import Customer, Document, User

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    domains TEXT NOT NULL,
    timezone TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL,
    status TEXT NOT NULL,
    suspended_by_operator INTEGER NOT NULL DEFAULT 0 CHECK (suspended_by_operator IN (0, 1))
) STRICT;

CREATE INDEX IF NOT EXISTS users_customer_id ON users(customer_id);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    title TEXT NOT NULL,
    label TEXT NOT NULL,
    tags TEXT NOT NULL,
    assigned_user_ids TEXT NOT NULL,
    body TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS documents_customer_id ON documents(customer_id);

CREATE TABLE IF NOT EXISTS policy_snapshots (
    version TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    bundle TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    first_loaded_at TEXT NOT NULL
) STRICT;
"""


# Mode bits a store file may keep: owner read and write. Execute and every group and other
# bit are removed when an existing file is opened.
_FILE_MODE = 0o600
_DIR_MODE = 0o700
# SQLite's side files, which hold copies of pages while a transaction is open.
_SIDE_FILES = ("-journal", "-wal", "-shm")


class StoreError(ValueError):
    """The path does not hold an agentgate store, or a row does not match the store."""


@dataclass(frozen=True)
class PolicySnapshot:
    """One loaded policy bundle as canonical JSON, under the version computed over it."""

    version: str
    # The customer that first loaded it. The version hashes the bundle alone, so customers
    # with identical rules share one snapshot; their policy_loaded events say who loaded it.
    customer_id: str
    # The canonical JSON the version hash is computed over (policy.version.bundle_version).
    bundle: str
    # Hex sha256 of `bundle` as UTF-8, as the policy_loaded event records it.
    sha256: str
    first_loaded_at: datetime


def _sorted_json(values: Iterable[str]) -> str:
    # Sets have no order; sorting keeps rows and any dumps byte-for-byte reproducible.
    return json.dumps(sorted(values))


class Store:
    """Customers, users and documents in one SQLite database."""

    def __init__(self, path: str | Path = ":memory:", *, readonly: bool = False) -> None:
        file_path = None if str(path) in {":memory:", ""} else Path(path)
        # The mode an existing file had before it was tightened to owner-only, or None.
        self.tightened_from: int | None = None
        self.readonly = readonly
        if readonly:
            if file_path is None:
                raise StoreError("a read-only store needs a file")
            if not file_path.is_file():
                raise FileNotFoundError(f"store not found: {file_path}")
            self._conn = sqlite3.connect(
                f"{file_path.resolve().as_uri()}?mode=ro",
                uri=True,
                isolation_level=None,
                check_same_thread=False,
            )
        else:
            if file_path is not None:
                self.tightened_from = _prepare_file(file_path)
            # isolation_level=None turns off sqlite3's implicit transactions, so single
            # writes commit immediately and transaction() alone decides multi-write
            # boundaries.
            self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        try:
            self._conn.row_factory = sqlite3.Row
            self._lock = threading.RLock()
            self._conn.execute("PRAGMA foreign_keys = ON")
            if file_path is not None:
                # Before the schema script, which would write store tables into the file.
                _refuse_audit_database(self._conn, file_path)
            if readonly:
                if "customers" not in _tables(self._conn):
                    raise StoreError(f"{file_path}: is not an agentgate store")
            else:
                self._conn.executescript(_SCHEMA)
                _migrate(self._conn)
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """The shared connection, for components that keep their own tables here.

        Use it only inside `transaction()` or `locked()`.
        """
        return self._conn

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the store lock for a group of reads that must see one consistent state."""
        with self._lock:
            yield

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the enclosed statements atomically: all of them commit or none do.

        The transaction holds the store lock and SQLite's write lock from the start
        (BEGIN IMMEDIATE), so a read-check-write sequence such as a spend reservation cannot
        interleave with another thread or, for a file database, another process.
        """
        with self._lock:
            if self._conn.in_transaction:
                raise RuntimeError("nested transactions are not supported")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def add_customer(self, customer: Customer) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO customers (id, name, domains, timezone) VALUES (?, ?, ?, ?)",
                # Domains are an ordered tuple on the model, so their order is kept as written.
                (customer.id, customer.name, json.dumps(list(customer.domains)), customer.timezone),
            )

    def add_user(self, user: User) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO users (id, customer_id, name, email, role, status)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    user.id,
                    user.customer_id,
                    user.name,
                    user.email,
                    user.role.value,
                    user.status.value,
                ),
            )

    def add_document(self, document: Document) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO documents"
                " (id, customer_id, title, label, tags, assigned_user_ids, body)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document.id,
                    document.customer_id,
                    document.title,
                    document.label,
                    _sorted_json(document.tags),
                    _sorted_json(document.assigned_user_ids),
                    document.body,
                ),
            )

    def update_customer(self, customer: Customer) -> None:
        """Replace a stored customer's name, domains and timezone."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE customers SET name = ?, domains = ?, timezone = ? WHERE id = ?",
                (customer.name, json.dumps(list(customer.domains)), customer.timezone, customer.id),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"unknown customer {customer.id!r}")

    def update_user(self, user: User) -> None:
        """Replace a stored user's name, email, role and status. The customer cannot change,
        and the operator suspension flag is left as it is."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE users SET name = ?, email = ?, role = ?, status = ?"
                " WHERE id = ? AND customer_id = ?",
                (
                    user.name,
                    user.email,
                    user.role.value,
                    user.status.value,
                    user.id,
                    user.customer_id,
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError(f"unknown user {user.id!r} in customer {user.customer_id!r}")

    def update_document(self, document: Document) -> None:
        """Replace a stored document's title, label, tags, assignment and body. The customer
        cannot change."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE documents SET title = ?, label = ?, tags = ?, assigned_user_ids = ?,"
                " body = ? WHERE id = ? AND customer_id = ?",
                (
                    document.title,
                    document.label,
                    _sorted_json(document.tags),
                    _sorted_json(document.assigned_user_ids),
                    document.body,
                    document.id,
                    document.customer_id,
                ),
            )
            if cursor.rowcount != 1:
                raise StoreError(
                    f"unknown document {document.id!r} in customer {document.customer_id!r}"
                )

    def suspend_user(self, user_id: str) -> User | None:
        """Suspend a user as an operator: status suspended, and flagged so that fixture sync
        does not reactivate them. Returns the updated user, or None if there is none.

        Takes effect on the user's next gate call, in this process or any other, since the
        gate reads the user again on every call."""
        with self._lock:
            self._conn.execute(
                "UPDATE users SET status = 'suspended', suspended_by_operator = 1 WHERE id = ?",
                (user_id,),
            )
            return self.get_user(user_id)

    def reinstate_user(self, user_id: str) -> User | None:
        """Clear an operator suspension. The status is left for fixture sync to set, so a
        user the fixtures suspend stays suspended. Returns the user, or None."""
        with self._lock:
            self._conn.execute(
                "UPDATE users SET suspended_by_operator = 0 WHERE id = ?", (user_id,)
            )
            return self.get_user(user_id)

    def operator_suspended(self, user_id: str) -> bool:
        """Whether an operator suspended this user (see suspend_user)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT suspended_by_operator FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            return row is not None and bool(row["suspended_by_operator"])

    def save_policy_snapshot(self, snapshot: PolicySnapshot) -> tuple[PolicySnapshot, bool]:
        """Store a snapshot unless its version is already stored. Returns the stored one and
        whether it was written now.

        A stored snapshot is never replaced, so the result can differ from `snapshot` only
        if the stored row was edited: the caller compares the sha256."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO policy_snapshots"
                " (version, customer_id, bundle, sha256, first_loaded_at) VALUES (?, ?, ?, ?, ?)",
                (
                    snapshot.version,
                    snapshot.customer_id,
                    snapshot.bundle,
                    snapshot.sha256,
                    snapshot.first_loaded_at.isoformat(),
                ),
            )
            stored = self.get_policy_snapshot(snapshot.version)
            assert stored is not None
            return stored, cursor.rowcount == 1

    def get_policy_snapshot(self, version: str) -> PolicySnapshot | None:
        """The snapshot stored for a policy version, or None (also for a store from before
        snapshots, opened read-only)."""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM policy_snapshots WHERE version = ?", (version,)
                ).fetchone()
            except sqlite3.OperationalError:
                if "policy_snapshots" in _tables(self._conn):
                    raise
                return None
            return None if row is None else _snapshot(row)

    def list_policy_snapshots(self, customer_id: str) -> list[PolicySnapshot]:
        """One customer's snapshots, oldest first."""
        with self._lock:
            if "policy_snapshots" not in _tables(self._conn):
                return []
            rows = self._conn.execute(
                "SELECT * FROM policy_snapshots WHERE customer_id = ?"
                " ORDER BY first_loaded_at, version",
                (customer_id,),
            ).fetchall()
            return [_snapshot(row) for row in rows]

    def get_customer(self, customer_id: str) -> Customer | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM customers WHERE id = ?", (customer_id,)
            ).fetchone()
            return None if row is None else _customer(row)

    def list_customers(self) -> list[Customer]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM customers ORDER BY id").fetchall()
            return [_customer(row) for row in rows]

    def get_user(self, user_id: str) -> User | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return None if row is None else _user(row)

    def list_users(self, customer_id: str) -> list[User]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM users WHERE customer_id = ? ORDER BY id", (customer_id,)
            ).fetchall()
            return [_user(row) for row in rows]

    def get_document(self, doc_id: str) -> Document | None:
        """Return a document by id from any customer.

        This lookup is deliberately not tenant-scoped. Callers must not hand the result to a
        user without the gate layer first checking the document's customer and label.
        """
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return None if row is None else _document(row)

    def list_documents(self, customer_id: str) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE customer_id = ? ORDER BY id", (customer_id,)
            ).fetchall()
            return [_document(row) for row in rows]


# Rows go back through model validation, so a hand-edited database cannot smuggle in
# values the models would have rejected.
def _customer(row: sqlite3.Row) -> Customer:
    return Customer.model_validate(
        {
            "id": row["id"],
            "name": row["name"],
            "domains": json.loads(row["domains"]),
            "timezone": row["timezone"],
        }
    )


def _user(row: sqlite3.Row) -> User:
    # The operator flag is the store's bookkeeping, not part of the model.
    fields = dict(row)
    fields.pop("suspended_by_operator", None)
    return User.model_validate(fields)


def _document(row: sqlite3.Row) -> Document:
    return Document.model_validate(
        {
            "id": row["id"],
            "customer_id": row["customer_id"],
            "title": row["title"],
            "label": row["label"],
            "tags": json.loads(row["tags"]),
            "assigned_user_ids": json.loads(row["assigned_user_ids"]),
            "body": row["body"],
        }
    )


def _snapshot(row: sqlite3.Row) -> PolicySnapshot:
    return PolicySnapshot(
        version=row["version"],
        customer_id=row["customer_id"],
        bundle=row["bundle"],
        sha256=row["sha256"],
        first_loaded_at=datetime.fromisoformat(row["first_loaded_at"]),
    )


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row[0]) for row in rows}


def make_private_dirs(directory: Path) -> None:
    """Create directory and any missing parents owner-only (0700, less any umask bits).

    Directories that already exist are left exactly as they are."""
    missing: list[Path] = []
    current = directory
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    for path in reversed(missing):
        # Another process may create it in between; it is theirs to keep as made.
        with contextlib.suppress(FileExistsError):
            path.mkdir(mode=_DIR_MODE)


def table_names(path: Path) -> set[str] | None:
    """The table names in the SQLite file at path, opened read-only, or None when there is
    no such file or it is not an SQLite database."""
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return None
    return {str(name) for (name,) in rows}


def _prepare_file(path: Path) -> int | None:
    """Create path owner-only if it is missing; tighten it if other users can read it.

    Returns the mode an existing file had when it was tightened, else None."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _FILE_MODE)
    except FileExistsError:
        tightened = _tighten(path)
        for suffix in _SIDE_FILES:
            side = path.with_name(path.name + suffix)
            if side.exists():
                _tighten(side)
        return tightened
    os.close(fd)
    return None


def _tighten(path: Path) -> int | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    # Anything but a regular file is left for SQLite to refuse.
    if not stat.S_ISREG(info.st_mode):
        return None
    mode = stat.S_IMODE(info.st_mode)
    if mode & ~_FILE_MODE == 0:
        return None
    try:
        os.chmod(path, mode & _FILE_MODE)
    except OSError:
        # Not the owner: nothing this process can change, and SQLite decides access.
        return None
    return mode


def _refuse_audit_database(conn: sqlite3.Connection, path: Path) -> None:
    tables = _tables(conn)
    # The store never has an events table; Panoptes keeps its hash chain beside one.
    if "events" in tables:
        raise StoreError(f"{path}: is a Panoptes audit database, not an agentgate store")


def _migrate(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "suspended_by_operator" not in columns:
        # A store written before operator suspensions existed.
        conn.execute(
            "ALTER TABLE users ADD COLUMN suspended_by_operator INTEGER NOT NULL DEFAULT 0"
            " CHECK (suspended_by_operator IN (0, 1))"
        )
