"""SQLite store for customers, users and documents.

The store is a thin persistence layer: it saves and returns validated models and enforces
identity constraints (unique ids and emails, foreign keys). It does not decide who may see
what; tenant and permission checks belong to the gate layer above it.
"""

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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
    status TEXT NOT NULL
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
"""


def _sorted_json(values: Iterable[str]) -> str:
    # Sets have no order; sorting keeps rows and any dumps byte-for-byte reproducible.
    return json.dumps(sorted(values))


class Store:
    """Customers, users and documents in one SQLite database."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        # isolation_level=None turns off sqlite3's implicit transactions, so single writes
        # commit immediately and transaction() alone decides multi-write boundaries.
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

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
        """Run the enclosed writes atomically: all of them commit or none do."""
        if self._conn.in_transaction:
            raise RuntimeError("nested transactions are not supported")
        self._conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def add_customer(self, customer: Customer) -> None:
        self._conn.execute(
            "INSERT INTO customers (id, name, domains, timezone) VALUES (?, ?, ?, ?)",
            # Domains are an ordered tuple on the model, so their order is kept as written.
            (customer.id, customer.name, json.dumps(list(customer.domains)), customer.timezone),
        )

    def add_user(self, user: User) -> None:
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

    def get_customer(self, customer_id: str) -> Customer | None:
        row = self._conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return None if row is None else _customer(row)

    def list_customers(self) -> list[Customer]:
        rows = self._conn.execute("SELECT * FROM customers ORDER BY id").fetchall()
        return [_customer(row) for row in rows]

    def get_user(self, user_id: str) -> User | None:
        row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return None if row is None else _user(row)

    def list_users(self, customer_id: str) -> list[User]:
        rows = self._conn.execute(
            "SELECT * FROM users WHERE customer_id = ? ORDER BY id", (customer_id,)
        ).fetchall()
        return [_user(row) for row in rows]

    def get_document(self, doc_id: str) -> Document | None:
        """Return a document by id from any customer.

        This lookup is deliberately not tenant-scoped. Callers must not hand the result to a
        user without the gate layer first checking the document's customer and label.
        """
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return None if row is None else _document(row)

    def list_documents(self, customer_id: str) -> list[Document]:
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
    return User.model_validate(dict(row))


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
