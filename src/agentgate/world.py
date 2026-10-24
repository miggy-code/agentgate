"""Open everything a gate needs in one call: store, fixtures, index, policies, audit, gate.

The CLI, the demo and tests all build their gate here, so they agree on how a world is put
together. open_world keeps its state on disk and can be reopened: fixtures are loaded into a
database that has no customers yet, and synced into one that has (fixtures.sync_fixtures:
new customers, users and documents are added, changed ones updated, nothing deleted, a user
removed from the fixtures suspended), so a suspension, role change or relabelling in the
fixtures takes effect on the next open. Approvals, runs and spend from earlier sessions are
still there. The search index is rebuilt whenever the sync added or changed a document.
`World.fixture_sync` says what the sync did.

On every open, open_world also reclaims spend reservations left open longer than the lease
(15 minutes by default; `reservation_lease=None` turns it off), so a host that died between
reserve and settle does not hold the customer's daily budget (spend.SpendLedger.
release_stale). `World.stale_released` is the count.

open_world refuses a store and an audit log that are the same file, and an audit path that
holds an agentgate store, before it writes anything; Store itself refuses a store path that
holds a Panoptes audit log. Directories it creates for the store are owner-only (0700), and
the store file is 0600 (see Store).

Every policy must name a customer the database holds; a policy for a missing customer is
refused rather than silently ignored. A customer without a policy gets no engine, and the
gate denies its runs with `no_policy`.

Every open is on the audit chain. For each customer with a policy, the canonical bundle the
policy version is computed over is kept in the store's `policy_snapshots` (written once per
version) and a `policy_loaded` event records the version and the snapshot's sha256, so the
rules behind any version an event names can be read back and checked against the chain
(`agentgate audit policy`). What the fixture load or sync changed, per customer, and what the
stale spend sweep released are recorded as `operator_action` events, as are the operator
commands `suspend_user` and `reinstate_user`. The fixture sync and the sweep commit before
their events are recorded; if the sink then fails, the open fails with the change made.

`check_store` compares a customer's approval rows and side-effect rows (outbox, parts
orders, refunds, notes) with the audit chain: each approval's args and args_hash with the
`approval_args_hash` its events carry, each side-effect row, hashed as the args that would
have written it, with the `args_hash` of the EXECUTED event that names it as its
`effect_ref`. A row edited after the fact, or one no event names, is a mismatch. It reads
only; `agentgate audit check-store` runs it.
"""

import contextlib
import hashlib
import hmac
import json
import logging
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import JsonValue

from agentgate.approvals import args_hash
from agentgate.audit.base import AuditEvent, AuditSink, EventKind, MemorySink, OperatorAction
from agentgate.audit.panoptes import PanoptesSink
from agentgate.clock import Clock, SystemClock
from agentgate.fixtures import FixtureError, FixtureSync, load_fixtures, sync_fixtures
from agentgate.gate import Gate
from agentgate.models import Action, Role, User, UserStatus
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import PolicyBundle
from agentgate.policy.testing import load_policy_bundles
from agentgate.policy.version import _canonical
from agentgate.retrieval import DocumentIndex
from agentgate.store import PolicySnapshot, Store, make_private_dirs, table_names
from agentgate.tools.builtin import StoredEffect, stored_effects

DEFAULT_DB = Path("data/agentgate.db")
DEFAULT_AUDIT_DB = Path("data/audit.db")
DEFAULT_POLICIES = Path("policies")
DEFAULT_FIXTURES = Path("fixtures")
# How long a spend reservation may stay open before open_world reclaims it.
DEFAULT_RESERVATION_LEASE = timedelta(minutes=15)

_log = logging.getLogger(__name__)


class WorldError(ValueError):
    """The database, policies and fixtures do not fit together."""


class OpenRecord(StrEnum):
    """How much of opening a world is recorded on the audit chain."""

    # Nothing: the default, for hosts and tests that keep the log to their own events.
    NONE = "none"
    # What the open changed: policy_loaded for a version stored for the first time,
    # fixture_sync when the fixtures changed something, the stale sweep when it released
    # anything. For read-only commands, so reading leaves the log as it was.
    CHANGES = "changes"
    # Also a policy_loaded event for every customer on every open.
    ALL = "all"


@dataclass
class World:
    store: Store
    gate: Gate
    index: DocumentIndex
    sink: AuditSink
    engines: dict[str, PolicyEngine]
    clock: Clock
    # Whether close() also closes the sink: true only for a sink open_world opened itself.
    owns_sink: bool = False
    # What syncing the fixtures into an existing database changed; None when the fixtures
    # were loaded into an empty one.
    fixture_sync: FixtureSync | None = None
    # Stale spend reservations reclaimed when the world was opened.
    stale_released: int = 0
    # The policy and fixtures directories the world was built from.
    policies: Path | None = None
    fixtures: Path | None = None

    def close(self) -> None:
        """Close the store, and the audit sink if this world opened it."""
        try:
            if self.owns_sink and isinstance(self.sink, PanoptesSink):
                self.sink.close()
        finally:
            self.store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_world(
    *,
    db_path: Path,
    audit_path: Path,
    policies: Path,
    fixtures: Path,
    clock: Clock | None = None,
    sink: AuditSink | None = None,
    reservation_lease: timedelta | None = DEFAULT_RESERVATION_LEASE,
    record_open: "OpenRecord | None" = None,
) -> World:
    """Open (or create) the database at db_path and build a gate over it.

    The audit log is a Panoptes database at audit_path unless a sink is given, in which case
    audit_path is not touched and the caller keeps ownership of the sink.

    record_open says how much of the open itself goes on the audit chain (OpenRecord):
    nothing by default, as for a host that records its own way, or what the open changed, or
    everything including a policy_loaded event per customer. Policy snapshots are stored
    whatever it says.
    """
    if sink is None:
        check_paths(db_path, audit_path)
    make_private_dirs(db_path.parent)
    store = Store(db_path)
    try:
        return _build(
            store,
            policies,
            fixtures,
            clock,
            sink,
            audit_path,
            reservation_lease,
            OpenRecord.NONE if record_open is None else record_open,
        )
    except BaseException:
        store.close()
        raise


def check_paths(db_path: Path, audit_path: Path) -> None:
    """Refuse a store and audit log that are one file, or an audit path holding a store."""
    if _same_file(db_path, audit_path):
        raise WorldError(f"{db_path}: the store and the audit log must be different files")
    tables = table_names(audit_path)
    if tables is not None and "customers" in tables and "events" not in tables:
        raise WorldError(f"{audit_path}: is an agentgate store, not a Panoptes audit database")


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        # One of them does not exist yet: compare where they would be.
        return a.resolve() == b.resolve()


def open_memory_world(
    *,
    policies: Path,
    fixtures: Path,
    clock: Clock | None = None,
    sink: AuditSink | None = None,
) -> World:
    """A world in an in-memory database, with a MemorySink unless a sink is given."""
    store = Store()
    try:
        return _build(
            store,
            policies,
            fixtures,
            clock,
            MemorySink() if sink is None else sink,
            None,
            None,
            OpenRecord.NONE,
        )
    except BaseException:
        store.close()
        raise


def _build(
    store: Store,
    policies: Path,
    fixtures: Path,
    clock: Clock | None,
    sink: AuditSink | None,
    audit_path: Path | None,
    reservation_lease: timedelta | None,
    record_open: "OpenRecord",
) -> World:
    # Needed on every open, not only the first: the sync is how fixture changes, a
    # suspension among them, reach an existing database.
    if not fixtures.is_dir():
        raise FixtureError(f"{fixtures}: fixtures directory not found")
    synced: FixtureSync | None = None
    initial = not store.list_customers()
    if initial:
        load_fixtures(store, fixtures)
    else:
        synced = sync_fixtures(store, fixtures)
        for line in synced.lines():
            _log.info("fixture sync: %s", line)
    index = DocumentIndex(store)
    # Also rebuilt when the index is empty, for a database populated some other way or one
    # whose earlier session stopped between loading and indexing.
    if synced is None or synced.documents_changed or _index_is_empty(store):
        index.rebuild()

    bundles = load_policy_bundles(policies)
    engines: dict[str, PolicyEngine] = {}
    for customer_id in sorted(bundles):
        customer = store.get_customer(customer_id)
        if customer is None:
            raise WorldError(
                f"{policies / customer_id}: policy customer {customer_id!r} is not in the database"
            )
        engines[customer_id] = PolicyEngine(bundles[customer_id], customer)

    # Opened last, so a world that fails to build creates no audit log.
    owns_sink = sink is None
    if sink is None:
        assert audit_path is not None
        sink = PanoptesSink(audit_path)
    try:
        world_clock = SystemClock() if clock is None else clock
        gate = Gate(store=store, engines=engines, sink=sink, clock=world_clock, index=index)
        record_policies(store, gate, engines, world_clock, record=record_open)
        recording = record_open is not OpenRecord.NONE
        if recording and initial:
            _record_initial_load(store, gate)
        elif recording and synced is not None:
            record_fixture_sync(store, gate, synced, actor=None)
        released = (
            0
            if reservation_lease is None
            else gate.release_stale(reservation_lease, record=recording)
        )
    except BaseException:
        if owns_sink and isinstance(sink, PanoptesSink):
            with contextlib.suppress(Exception):
                sink.close()
        raise
    return World(
        store=store,
        gate=gate,
        index=index,
        sink=sink,
        engines=engines,
        clock=world_clock,
        owns_sink=owns_sink,
        fixture_sync=synced,
        stale_released=released,
        policies=policies,
        fixtures=fixtures,
    )


def canonical_bundle(bundle: PolicyBundle) -> str:
    """The canonical JSON of a bundle: the text policy.version.bundle_version hashes."""
    canonical = _canonical(bundle.model_dump(mode="python", by_alias=True))
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def bundle_sha256(text: str) -> str:
    """Hex sha256 of a snapshot's text, as the policy_loaded event records it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_policies(
    store: Store,
    gate: Gate,
    engines: dict[str, PolicyEngine],
    clock: Clock,
    *,
    record: OpenRecord,
) -> None:
    """Snapshot each customer's bundle under its version (once) and record policy_loaded
    as `record` says.

    A new snapshot is written in the transaction that records its event, so a recorded open
    stores a version only once the chain has it. A stored snapshot is only read, so a
    read-only store opens. A version that is not the hash of its own canonical bundle gets
    no snapshot (it could not be checked), and its event says so."""
    for customer_id in sorted(engines):
        engine = engines[customer_id]
        text = canonical_bundle(engine.bundle)
        digest = bundle_sha256(text)
        details: dict[str, JsonValue] = {
            "rule_count": len(engine.bundle.policy.rules),
            "tool_count": len(engine.bundle.tools),
        }
        if engine.version != "sha256:" + digest[:16]:
            _log.warning("policy version of %s is not the hash of its bundle", customer_id)
            if record is not OpenRecord.NONE:
                gate.record_policy_loaded(
                    customer_id,
                    bundle_sha256=digest,
                    details={**details, "snapshot": "none", "version_matches": False},
                )
            continue
        stored = store.get_policy_snapshot(engine.version)
        if stored is not None:
            if record is OpenRecord.ALL:
                gate.record_policy_loaded(
                    customer_id,
                    bundle_sha256=digest,
                    details={
                        **details,
                        "snapshot": "kept",
                        # False only if the stored snapshot was edited since it was written.
                        "snapshot_matches": hmac.compare_digest(stored.sha256, digest)
                        and stored.bundle == text,
                    },
                )
            continue
        snapshot = PolicySnapshot(engine.version, customer_id, text, digest, clock.now())
        with store.transaction():
            stored, written = store.save_policy_snapshot(snapshot)
            if record is not OpenRecord.NONE:
                gate.record_policy_loaded(
                    customer_id,
                    bundle_sha256=digest,
                    details={
                        **details,
                        "snapshot": "written" if written else "kept",
                        "snapshot_matches": hmac.compare_digest(stored.sha256, digest)
                        and stored.bundle == text,
                    },
                )


def _record_initial_load(store: Store, gate: Gate) -> None:
    """One fixture_sync event per customer loaded into an empty database."""
    for customer in store.list_customers():
        users: list[JsonValue] = [user.id for user in store.list_users(customer.id)]
        documents: list[JsonValue] = [doc.id for doc in store.list_documents(customer.id)]
        gate.record_operator_action(
            OperatorAction.FIXTURE_SYNC,
            customer_id=customer.id,
            actor=None,
            details={
                "initial": True,
                "counts": {"users_added": len(users), "documents_added": len(documents)},
                "users_added": users,
                "documents_added": documents,
            },
        )


def record_fixture_sync(
    store: Store, gate: Gate, synced: FixtureSync, *, actor: User | None
) -> None:
    """One fixture_sync event per customer the sync changed: counts and ids of the users
    and documents added, updated and suspended (never names, emails or document text)."""
    lists: dict[str, dict[str, list[str]]] = {}
    fields_by_user: dict[str, dict[str, JsonValue]] = {}

    def add(customer_id: str | None, key: str, item_id: str) -> None:
        if customer_id is not None:
            lists.setdefault(customer_id, {}).setdefault(key, []).append(item_id)

    def user_customer(user_id: str) -> str | None:
        user = store.get_user(user_id)
        return None if user is None else user.customer_id

    def document_customer(document_id: str) -> str | None:
        document = store.get_document(document_id)
        return None if document is None else document.customer_id

    for customer_id in synced.customers_added:
        add(customer_id, "customer_added", customer_id)
    for customer_id in synced.customers_updated:
        add(customer_id, "customer_updated", customer_id)
    for user_id in synced.users_added:
        add(user_customer(user_id), "users_added", user_id)
    for user_id, changed in synced.users_updated:
        owner = user_customer(user_id)
        add(owner, "users_updated", user_id)
        if owner is not None:
            fields_by_user.setdefault(owner, {})[user_id] = [*changed]
    for user_id in synced.users_suspended:
        add(user_customer(user_id), "users_suspended", user_id)
    for document_id in synced.documents_added:
        add(document_customer(document_id), "documents_added", document_id)
    for document_id in synced.documents_updated:
        add(document_customer(document_id), "documents_updated", document_id)
    kept: dict[str, list[str]] = {}
    for user_id in synced.users_kept_suspended:
        owner = user_customer(user_id)
        if owner is not None:
            kept.setdefault(owner, []).append(user_id)

    for customer_id in sorted(lists):
        changes = lists[customer_id]
        details: dict[str, JsonValue] = {
            "initial": False,
            "counts": {key: len(ids) for key, ids in sorted(changes.items())},
        }
        for key, ids in sorted(changes.items()):
            if key not in ("customer_added", "customer_updated"):
                details[key] = [*ids]
            else:
                details[key] = True
        if customer_id in fields_by_user:
            details["user_fields_updated"] = fields_by_user[customer_id]
        if customer_id in kept:
            details["users_kept_suspended"] = [*kept[customer_id]]
        gate.record_operator_action(
            OperatorAction.FIXTURE_SYNC, customer_id=customer_id, actor=actor, details=details
        )


def _index_is_empty(store: Store) -> bool:
    with store.locked():
        row = store.connection.execute("SELECT 1 FROM retrieval_lengths LIMIT 1").fetchone()
    return row is None


class UserAdminError(ValueError):
    """An operator change to a user was refused."""


class OperatorReason(StrEnum):
    """Why an operator suspended or reinstated a user: a code, never free text, since it
    goes into the audit log."""

    UNSPECIFIED = "unspecified"
    SECURITY = "security"
    INVESTIGATION = "investigation"
    OFFBOARDING = "offboarding"
    POLICY_VIOLATION = "policy_violation"
    RESOLVED = "resolved"
    MISTAKE = "mistake"
    OTHER = "other"


def _owner_of(world: World, acting_user_id: str) -> User:
    acting = world.store.get_user(acting_user_id)
    if acting is None:
        raise UserAdminError(f"unknown user {acting_user_id!r}")
    if acting.status is not UserStatus.ACTIVE:
        raise UserAdminError(f"user {acting_user_id!r} is not active")
    if acting.role is not Role.OWNER:
        raise UserAdminError(f"role {acting.role.value!r} may not change users; owners only")
    return acting


def _target(world: World, acting: User, user_id: str) -> User:
    target = world.store.get_user(user_id)
    # Another customer's user reads exactly like a missing one.
    if target is None or target.customer_id != acting.customer_id:
        raise UserAdminError(f"unknown user {user_id!r}")
    return target


def suspend_user(
    world: World,
    user_id: str,
    *,
    acting_user_id: str,
    reason: OperatorReason = OperatorReason.UNSPECIFIED,
) -> User:
    """Suspend a user of the acting owner's customer, as `agentgate users suspend` does.

    The suspension is sticky: fixture sync keeps the user suspended even if the fixtures
    say active, until reinstate_user. It applies to the user's next gate call in any process
    over the same database, live runs included. Owners cannot suspend themselves, so a
    customer cannot lose its last owner this way.

    A user_suspended event names the actor, the target and the reason code. The suspension
    is made first and stands even if the event cannot be recorded (the sink's error then
    propagates): an audit outage must not keep a user active."""
    acting = _owner_of(world, acting_user_id)
    target = _target(world, acting, user_id)
    if target.id == acting.id:
        raise UserAdminError("an owner cannot suspend themselves")
    suspended = world.store.suspend_user(target.id)
    assert suspended is not None
    world.gate.record_operator_action(
        OperatorAction.USER_SUSPENDED,
        customer_id=acting.customer_id,
        actor=acting,
        target=target.id,
        details={"reason": OperatorReason(reason).value, "status": suspended.status.value},
    )
    return suspended


def reinstate_user(
    world: World,
    user_id: str,
    *,
    acting_user_id: str,
    reason: OperatorReason = OperatorReason.UNSPECIFIED,
) -> User:
    """Lift an operator suspension, as `agentgate users reinstate` does, and apply the
    fixtures again, so the user's status is what the fixtures say (a user the fixtures
    suspend, or no longer list, stays suspended).

    The flag is cleared only if its user_reinstated event is recorded, in one transaction;
    what the fixture sync then changes is recorded as a fixture_sync event with the same
    actor."""
    acting = _owner_of(world, acting_user_id)
    target = _target(world, acting, user_id)
    if not world.store.operator_suspended(target.id):
        raise UserAdminError(f"user {user_id!r} was not suspended by an operator")
    with world.store.transaction():
        world.store.reinstate_user(target.id)
        world.gate.record_operator_action(
            OperatorAction.USER_REINSTATED,
            customer_id=acting.customer_id,
            actor=acting,
            target=target.id,
            details={"reason": OperatorReason(reason).value},
        )
    if world.fixtures is not None:
        synced = sync_fixtures(world.store, world.fixtures)
        record_fixture_sync(world.store, world.gate, synced, actor=acting)
    reinstated = world.store.get_user(target.id)
    assert reinstated is not None
    return reinstated


# Checking the store against the audit chain


@dataclass(frozen=True)
class StoreMismatch:
    """One row the chain does not vouch for. `problem` is a fixed phrase, never content."""

    # "approval", or the tool whose side effect the row is (email.send, parts.order, ...).
    kind: str
    id: str
    problem: str


@dataclass(frozen=True)
class StoreCheck:
    customer_id: str
    # Rows checked, by kind.
    checked: dict[str, int] = field(default_factory=dict)
    # Rows whose events were recorded before the chain carried hashes, so only partly
    # checked: their presence and ids were, their args could not be.
    unchecked: tuple[str, ...] = ()
    mismatches: tuple[StoreMismatch, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatches


# The approval status each event leaves an approval in. An ALLOW naming the approval is its
# claim for execution, which marks it executed before the tool runs.
_APPROVAL_STATE: dict[EventKind, str] = {
    EventKind.APPROVAL_REQUIRED: "pending",
    EventKind.APPROVED: "approved",
    EventKind.REJECTED: "rejected",
    EventKind.EXPIRED: "expired",
    EventKind.ALLOW: "executed",
}
APPROVAL = "approval"
_EFFECT_TOOLS = frozenset({"email.send", "parts.order", "refund.issue", "notes.write"})


def check_store(store: Store, events: Sequence[AuditEvent], customer_id: str) -> StoreCheck:
    """Compare one customer's approval and side-effect rows with the audit events."""
    mine = [event for event in events if event.customer_id == customer_id]
    mismatches: list[StoreMismatch] = []
    unchecked: list[str] = []
    approvals = _approval_rows(store, customer_id)
    for row in approvals:
        mismatches += _approval_problems(row, mine, unchecked)
    effects = stored_effects(store, customer_id)
    executed: dict[str, AuditEvent] = {}
    for event in mine:
        ref = event.details.get("effect_ref")
        if event.kind is EventKind.EXECUTED and isinstance(ref, str):
            executed.setdefault(ref, event)
    for effect in effects:
        mismatches += _effect_problems(effect, executed.get(effect.id), unchecked)
    stored_ids = {effect.id for effect in effects}
    for ref, event in executed.items():
        if event.tool in _EFFECT_TOOLS and ref not in stored_ids:
            mismatches.append(
                StoreMismatch(str(event.tool), ref, "row named by an executed event is missing")
            )
    checked: dict[str, int] = {APPROVAL: len(approvals)}
    for effect in effects:
        checked[effect.tool] = checked.get(effect.tool, 0) + 1
    return StoreCheck(
        customer_id=customer_id,
        checked=checked,
        unchecked=tuple(unchecked),
        mismatches=tuple(mismatches),
    )


@dataclass(frozen=True)
class _ApprovalRow:
    id: str
    run_id: str
    requested_by: str
    tool: str
    action: str
    args: object
    args_hash: str
    status: str
    decided_by: str | None


def _approval_rows(store: Store, customer_id: str) -> list[_ApprovalRow]:
    with store.locked():
        try:
            rows = store.connection.execute(
                "SELECT * FROM approvals WHERE customer_id = ? ORDER BY created_at, rowid",
                (customer_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            if "approvals" in (_tables(store)):
                raise
            return []
    parsed: list[_ApprovalRow] = []
    for row in rows:
        try:
            args: object = json.loads(row["args"])
        except ValueError:
            args = None
        parsed.append(
            _ApprovalRow(
                id=row["id"],
                run_id=row["run_id"],
                requested_by=row["requested_by"],
                tool=row["tool"],
                action=row["action"],
                args=args,
                args_hash=row["args_hash"],
                status=row["status"],
                decided_by=row["decided_by"],
            )
        )
    return parsed


def _tables(store: Store) -> set[str]:
    rows = store.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {str(row[0]) for row in rows.fetchall()}


def _rehash(tool: str, action: object, args: object) -> str | None:
    """args_hash of a row's args, or None if they are not a JSON object it can hash."""
    if not isinstance(args, dict):
        return None
    try:
        return args_hash(tool, Action(str(action)), args)
    except (TypeError, ValueError):
        return None


def _approval_problems(
    row: _ApprovalRow, events: Sequence[AuditEvent], unchecked: list[str]
) -> list[StoreMismatch]:
    about = [event for event in events if event.subject == row.id]
    requested = [event for event in about if event.kind is EventKind.APPROVAL_REQUIRED]
    problems: list[str] = []
    if not requested:
        return [StoreMismatch(APPROVAL, row.id, "no approval_required event names it")]
    first = requested[0]
    if _rehash(row.tool, row.action, row.args) != row.args_hash:
        problems.append("args do not match the row's args_hash")
    hashes = [
        event.details["approval_args_hash"]
        for event in about
        if "approval_args_hash" in event.details
    ]
    if not hashes:
        unchecked.append(row.id)
    elif any(recorded != row.args_hash for recorded in hashes):
        problems.append("args_hash differs from the chain")
    if first.tool != row.tool or (first.action is not None and first.action.value != row.action):
        problems.append("tool or action differs from the chain")
    if first.run_id != row.run_id:
        problems.append("run differs from the chain")
    if first.user_id != row.requested_by:
        problems.append("requester differs from the chain")
    approved = [event for event in about if event.kind is EventKind.APPROVED]
    if row.decided_by is not None and approved and approved[-1].user_id != row.decided_by:
        problems.append("approver differs from the chain")
    states = [_APPROVAL_STATE[event.kind] for event in about if event.kind in _APPROVAL_STATE]
    if states and states[-1] != row.status:
        problems.append("status differs from the chain")
    return [StoreMismatch(APPROVAL, row.id, problem) for problem in problems]


def _effect_problems(
    effect: StoredEffect, event: AuditEvent | None, unchecked: list[str]
) -> list[StoreMismatch]:
    if event is None:
        return [StoreMismatch(effect.tool, effect.id, "no executed event names this row")]
    problems: list[str] = []
    if event.tool != effect.tool:
        problems.append("tool differs from the chain")
    if effect.run_id is not None and event.run_id != effect.run_id:
        problems.append("run differs from the chain")
    recorded = event.details.get("args_hash")
    if recorded is None or event.action is None:
        unchecked.append(effect.id)
    elif _rehash(effect.tool, event.action, effect.args) != recorded:
        problems.append("content differs from the chain")
    to = effect.args.get("to")
    count = event.details.get("recipients")
    if isinstance(to, list) and count is not None and count != len(to):
        problems.append("recipient count differs from the chain")
    amount = event.details.get("amount_usd")
    if amount is not None and "amount_usd" in effect.args and amount != effect.args["amount_usd"]:
        problems.append("amount differs from the chain")
    return [StoreMismatch(effect.tool, effect.id, problem) for problem in problems]
