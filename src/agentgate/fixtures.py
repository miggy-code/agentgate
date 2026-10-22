"""Load the synthetic fixture tree described in fixtures/README.md into a Store.

Loading parses and cross-checks every file before writing anything, then inserts all rows
in one store transaction. A fixture mistake therefore fails with the offending file named
and leaves the store unchanged.
"""

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from pydantic import ValidationError

from agentgate._yaml import parse_yaml
from agentgate.models import Customer, Document, User
from agentgate.store import Store

# A fixture canary line. The red-team suite reuses this to recognise the planted secrets.
CANARY_RE = re.compile(r"^Canary: (AGCANARY-[0-9A-F]{12})$", re.MULTILINE)
# A bare canary token. Leaks rarely keep the "Canary: " prefix or sit on their own line.
CANARY_TOKEN_RE = re.compile(r"AGCANARY-[0-9A-F]{12}")
_CANARY_MARKER = "AGCANARY-"

_CANARY_LABELS = frozenset({"confidential", "restricted"})
_PLAIN_LABELS = frozenset({"public", "internal"})


class FixtureError(ValueError):
    """A fixture file is missing, malformed or inconsistent with the others."""


@dataclass(frozen=True)
class CustomerCounts:
    customer_id: str
    users: int
    documents: int


@dataclass(frozen=True)
class FixtureSummary:
    customers: tuple[CustomerCounts, ...]

    @property
    def total_users(self) -> int:
        return sum(c.users for c in self.customers)

    @property
    def total_documents(self) -> int:
        return sum(c.documents for c in self.customers)


def find_canaries(text: str) -> list[str]:
    """Return every canary token in text, in order of appearance, with repeats."""
    return CANARY_TOKEN_RE.findall(text)


def load_fixtures(store: Store, root: Path) -> FixtureSummary:
    """Load every customer directory under root, in sorted order, as one transaction."""
    bundles = [
        _load_customer_dir(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    ]
    _check_unique_ids(bundles)
    _check_assignments(bundles)
    _check_canaries(bundles)

    with store.transaction():
        for bundle in bundles:
            _insert(store, bundle)
    return FixtureSummary(
        customers=tuple(
            CustomerCounts(b.customer.id, len(b.users), len(b.documents)) for b in bundles
        )
    )


@dataclass(frozen=True)
class _CustomerBundle:
    customer: Customer
    customer_path: Path
    users_path: Path
    users: list[User] = field(default_factory=list)
    documents: list[tuple[Path, Document]] = field(default_factory=list)


def _validate[M: (Customer, User, Document)](model: type[M], data: object, path: Path) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        # One line per problem, without echoing input values, so a document body never
        # ends up in an error log.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or model.__name__}: {err['msg']}"
            for err in exc.errors(include_url=False, include_input=False)
        )
        raise FixtureError(f"{path}: invalid {model.__name__}: {problems}") from exc


def _load_customer_dir(directory: Path) -> _CustomerBundle:
    customer_path = directory / "customer.yaml"
    if not customer_path.is_file():
        raise FixtureError(f"{customer_path}: missing customer.yaml")
    customer = _validate(
        Customer, parse_yaml(customer_path.read_text(), customer_path, FixtureError), customer_path
    )
    if customer.id != directory.name:
        raise FixtureError(
            f"{customer_path}: id {customer.id!r} does not match directory {directory.name!r}"
        )

    users_path = directory / "users.yaml"
    bundle = _CustomerBundle(customer, customer_path, users_path)
    bundle.users.extend(_load_users(users_path, customer.id))

    docs_dir = directory / "docs"
    if docs_dir.is_dir():
        for doc_path in sorted(docs_dir.iterdir()):
            if doc_path.name.startswith("."):
                continue
            if doc_path.suffix != ".md" or not doc_path.is_file():
                raise FixtureError(f"{doc_path}: expected only .md files in docs/")
            bundle.documents.append((doc_path, _load_document(doc_path, customer.id)))
    return bundle


def _load_users(path: Path, customer_id: str) -> list[User]:
    if not path.is_file():
        raise FixtureError(f"{path}: missing users.yaml")
    entries = parse_yaml(path.read_text(), path, FixtureError)
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise FixtureError(f"{path}: expected a list of users")
    users = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise FixtureError(f"{path}: each user must be a mapping")
        if "customer_id" in entry:
            raise FixtureError(f"{path}: users must not set customer_id; it comes from the dir")
        users.append(_validate(User, entry | {"customer_id": customer_id}, path))
    return users


def _split_front_matter(text: str, path: Path) -> tuple[str, str]:
    lines = text.split("\n")
    if lines[0] != "---":
        raise FixtureError(f"{path}: front matter must start with a '---' line")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise FixtureError(f"{path}: front matter has no closing '---' line") from None
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1 :])


def _load_document(path: Path, customer_id: str) -> Document:
    front_text, body = _split_front_matter(path.read_text(), path)
    front = parse_yaml(front_text, path, FixtureError)
    if not isinstance(front, dict):
        raise FixtureError(f"{path}: front matter must be a mapping")
    for key in ("id", "customer_id", "body"):
        # These come from the file name, the directory and the file body respectively.
        if key in front:
            raise FixtureError(f"{path}: front matter must not set {key!r}")
    data = front | {"id": path.stem, "customer_id": customer_id, "body": body}
    return _validate(Document, data, path)


def _check_unique_ids(bundles: list[_CustomerBundle]) -> None:
    user_ids: dict[str, Path] = {}
    emails: dict[str, Path] = {}
    for bundle in bundles:
        for user in bundle.users:
            if user.id in user_ids:
                raise FixtureError(
                    f"{bundle.users_path}: duplicate user id {user.id!r}"
                    f" (also in {user_ids[user.id]})"
                )
            user_ids[user.id] = bundle.users_path
            if user.email in emails:
                raise FixtureError(
                    f"{bundle.users_path}: duplicate email {user.email!r}"
                    f" (also in {emails[user.email]})"
                )
            emails[user.email] = bundle.users_path

    doc_ids: dict[str, Path] = {}
    for bundle in bundles:
        for path, doc in bundle.documents:
            if doc.id in doc_ids:
                raise FixtureError(
                    f"{path}: duplicate document id {doc.id!r} (also {doc_ids[doc.id]})"
                )
            doc_ids[doc.id] = path


def _check_assignments(bundles: list[_CustomerBundle]) -> None:
    owner_of = {user.id: user.customer_id for b in bundles for user in b.users}
    for bundle in bundles:
        for path, doc in bundle.documents:
            for user_id in sorted(doc.assigned_user_ids):
                owner = owner_of.get(user_id)
                if owner is None:
                    raise FixtureError(f"{path}: assigned user {user_id!r} does not exist")
                if owner != doc.customer_id:
                    raise FixtureError(
                        f"{path}: assigned user {user_id!r} belongs to customer {owner!r}"
                    )


def _check_canaries(bundles: list[_CustomerBundle]) -> None:
    seen: dict[str, Path] = {}
    for bundle in bundles:
        for path, doc in bundle.documents:
            marker_count = doc.body.count(_CANARY_MARKER)
            if doc.label in _PLAIN_LABELS:
                if marker_count:
                    raise FixtureError(f"{path}: {doc.label} documents must not contain a canary")
                continue
            if doc.label not in _CANARY_LABELS:
                raise FixtureError(f"{path}: unknown label {doc.label!r}")
            lines = CANARY_RE.findall(doc.body)
            # Any extra or malformed marker would be an untracked secret, so require exactly
            # one well-formed line and nothing else.
            if len(lines) != 1 or marker_count != 1:
                raise FixtureError(
                    f"{path}: {doc.label} documents need exactly one 'Canary: AGCANARY-' line"
                    f" and no other canary text"
                )
            canary = lines[0]
            if canary in seen:
                raise FixtureError(f"{path}: canary {canary} is also used in {seen[canary]}")
            seen[canary] = path


def _insert(store: Store, bundle: _CustomerBundle) -> None:
    # Checks above catch fixture-internal clashes; this catches clashes with rows that
    # were already in the store before loading.
    writes: list[tuple[Path, Callable[[], None]]] = [
        (bundle.customer_path, partial(store.add_customer, bundle.customer)),
        *((bundle.users_path, partial(store.add_user, user)) for user in bundle.users),
        *((path, partial(store.add_document, doc)) for path, doc in bundle.documents),
    ]
    for path, write in writes:
        try:
            write()
        except sqlite3.IntegrityError as exc:
            raise FixtureError(f"{path}: conflicts with existing store data: {exc}") from exc
