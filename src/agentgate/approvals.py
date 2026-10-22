"""Approval queue: actions that wait for a human before they run.

An approval is bound to the exact action it was requested for. Its args_hash covers the tool,
the action and the canonical JSON of the arguments, and claim_for_execution recomputes it from
what the caller is about to run, so approving a harmless email and then changing the
recipient voids the approval. Expired and rejected approvals never run, and an approved one
runs at most once: the claim is a single transaction that flips it to EXECUTED.

Every state change is a read-check-write inside one `store.transaction()`, so concurrent
approvers or executors are serialised. Errors that also record a state change (an approval
found past its expiry is marked EXPIRED) are raised only after that change commits.

Check orders are chosen so that a caller from the wrong customer learns nothing beyond what
it already knew:

- approve and reject check the approver alone first (active, an approver role), then look up
  the approval. A missing approval and one belonging to another customer both raise
  NOT_FOUND with the same message, so an approver cannot probe another customer's ids. Then
  self approval, then expiry, then status.
- claim_for_execution is called by the gate with the run's own customer, so it reports
  NOT_FOUND and WRONG_CUSTOMER separately (ids are unguessable, so this reveals nothing
  useful), but WRONG_CUSTOMER comes before any other check and before any write. Then the
  args hash, then executed, then expiry, then status.
"""

import builtins
import hashlib
import hmac
import json
import math
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import JsonValue

from agentgate.clock import Clock
from agentgate.models import Action, User, UserStatus
from agentgate.policy.schema import ApprovalSettings
from agentgate.store import Store

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS approvals (
        id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        requested_by TEXT NOT NULL,
        tool TEXT NOT NULL,
        action TEXT NOT NULL,
        args TEXT NOT NULL,
        args_hash TEXT NOT NULL,
        risk_score INTEGER NOT NULL,
        rule_ids TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        decided_by TEXT,
        decided_at TEXT,
        note TEXT,
        executed_at TEXT
    ) STRICT
    """,
    "CREATE INDEX IF NOT EXISTS approvals_customer ON approvals(customer_id, status)",
    "CREATE INDEX IF NOT EXISTS approvals_due ON approvals(status, expires_at)",
)


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"


class ApprovalErrorReason(StrEnum):
    NOT_FOUND = "not_found"
    WRONG_CUSTOMER = "wrong_customer"
    # approve or reject on an approval that is already decided.
    NOT_PENDING = "not_pending"
    # execute while pending or rejected.
    NOT_APPROVED = "not_approved"
    EXPIRED = "expired"
    ALREADY_EXECUTED = "already_executed"
    ARGS_MISMATCH = "args_mismatch"
    # Suspended, or no approver role. An approver from another customer gets NOT_FOUND.
    APPROVER_NOT_ALLOWED = "approver_not_allowed"
    SELF_APPROVAL = "self_approval"


class ApprovalError(Exception):
    def __init__(self, reason: ApprovalErrorReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def args_hash(tool: str, action: Action, args: Mapping[str, JsonValue]) -> str:
    """Return "sha256:" plus 64 hex chars over the canonical JSON of tool, action and args.

    Raises ValueError for values JSON cannot represent exactly (NaN, infinities, non-string
    keys, lone surrogates) and TypeError for non-JSON types.
    """
    payload = {"tool": tool, "action": Action(action).value, "args": _json_object(args)}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _json_object(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    _check_json(args)
    return dict(args)


def _check_json(value: object) -> None:
    # json.dumps would quietly turn int keys into strings and tuples into lists, so two
    # different argument sets could share a hash. Only plain JSON types are accepted.
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("args must not contain NaN or infinite numbers")
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"args keys must be strings, got {type(key).__name__}")
            _check_json(item)
        return
    raise TypeError(f"args must be JSON values, got {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


@dataclass(frozen=True)
class Approval:
    id: str
    customer_id: str
    run_id: str
    # User id of the person whose run asked for the action.
    requested_by: str
    tool: str
    action: Action
    args: dict[str, JsonValue]
    args_hash: str
    risk_score: int
    rule_ids: tuple[str, ...]
    policy_version: str
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    decided_by: str | None
    decided_at: datetime | None
    note: str | None
    executed_at: datetime | None


class ApprovalQueue:
    """Pending, decided and executed approvals for every customer, in the shared store."""

    def __init__(self, store: Store, clock: Clock) -> None:
        self._store = store
        self._clock = clock
        with store.transaction():
            for statement in _SCHEMA:
                store.connection.execute(statement)

    def create(
        self,
        *,
        customer_id: str,
        run_id: str,
        requested_by: str,
        tool: str,
        action: Action,
        args: Mapping[str, JsonValue],
        risk_score: int,
        rule_ids: Sequence[str],
        policy_version: str,
        ttl: timedelta,
    ) -> Approval:
        if ttl <= timedelta(0):
            raise ValueError("approval ttl must be positive")
        if risk_score < 0:
            raise ValueError("risk_score must not be negative")
        stored_args = _json_object(args)
        digest = args_hash(tool, action, stored_args)
        now = self._now()
        # The prefix keeps ids from starting with "-", which a CLI would read as an option.
        approval_id = "apr_" + secrets.token_urlsafe(24)
        with self._store.transaction():
            self._store.connection.execute(
                "INSERT INTO approvals (id, customer_id, run_id, requested_by, tool, action,"
                " args, args_hash, risk_score, rule_ids, policy_version, status, created_at,"
                " expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    approval_id,
                    customer_id,
                    run_id,
                    requested_by,
                    tool,
                    Action(action).value,
                    _canonical_json(stored_args),
                    digest,
                    risk_score,
                    json.dumps(list(rule_ids)),
                    policy_version,
                    ApprovalStatus.PENDING.value,
                    _ts(now),
                    _ts(now + ttl),
                ),
            )
            approval = self._load(approval_id)
        assert approval is not None
        return approval

    def get(self, approval_id: str) -> Approval | None:
        """Return an approval by id from any customer.

        Like `Store.get_document`, this is not tenant-scoped; callers check the customer.
        """
        with self._store.locked():
            return self._load(approval_id)

    def approve(
        self,
        approval_id: str,
        approver: User,
        settings: ApprovalSettings,
        note: str | None = None,
    ) -> Approval:
        return self._decide(approval_id, approver, settings, note, ApprovalStatus.APPROVED)

    def reject(
        self,
        approval_id: str,
        approver: User,
        settings: ApprovalSettings,
        note: str | None = None,
    ) -> Approval:
        return self._decide(approval_id, approver, settings, note, ApprovalStatus.REJECTED)

    def claim_for_execution(
        self,
        approval_id: str,
        *,
        customer_id: str,
        tool: str,
        action: Action,
        args: Mapping[str, JsonValue],
    ) -> Approval:
        """Mark an approved approval EXECUTED and return it, exactly once.

        The caller passes the tool, action and args it is about to run, never the stored ones,
        so any change since approval is caught here.
        """
        with self._store.transaction():
            outcome = self._claim(approval_id, customer_id, tool, action, args)
        if isinstance(outcome, ApprovalError):
            raise outcome
        return outcome

    def expire_due(self) -> builtins.list[Approval]:
        """Mark every pending or approved approval past its expiry as EXPIRED and return them."""
        now = _ts(self._now())
        live = (ApprovalStatus.PENDING.value, ApprovalStatus.APPROVED.value)
        with self._store.transaction():
            # Timestamps are fixed-width UTC strings, so text order is time order.
            ids = [
                row["id"]
                for row in self._store.connection.execute(
                    "SELECT id FROM approvals WHERE status IN (?, ?) AND expires_at <= ?"
                    " ORDER BY created_at, rowid",
                    (*live, now),
                ).fetchall()
            ]
            expired: builtins.list[Approval] = []
            for approval_id in ids:
                self._set_status(approval_id, ApprovalStatus.EXPIRED)
                approval = self._load(approval_id)
                assert approval is not None
                expired.append(approval)
        return expired

    # This method shadows the builtin inside the class body, so annotations here spell out
    # builtins.list.
    def list(
        self, customer_id: str, status: ApprovalStatus | None = None
    ) -> builtins.list[Approval]:
        """One customer's approvals, oldest first, optionally only those with one status."""
        query = "SELECT * FROM approvals WHERE customer_id = ?"
        params: builtins.list[str] = [customer_id]
        if status is not None:
            query += " AND status = ?"
            params.append(ApprovalStatus(status).value)
        query += " ORDER BY created_at, rowid"
        with self._store.locked():
            rows = self._store.connection.execute(query, params).fetchall()
            return [_approval(row) for row in rows]

    def _decide(
        self,
        approval_id: str,
        approver: User,
        settings: ApprovalSettings,
        note: str | None,
        decision: ApprovalStatus,
    ) -> Approval:
        with self._store.transaction():
            outcome = self._decide_locked(approval_id, approver, settings, note, decision)
        if isinstance(outcome, ApprovalError):
            raise outcome
        return outcome

    def _decide_locked(
        self,
        approval_id: str,
        approver: User,
        settings: ApprovalSettings,
        note: str | None,
        decision: ApprovalStatus,
    ) -> Approval | ApprovalError:
        # Approver-only checks come first: they need nothing from the row, so they reveal
        # nothing about it.
        if approver.status is not UserStatus.ACTIVE:
            return ApprovalError(
                ApprovalErrorReason.APPROVER_NOT_ALLOWED, f"approver {approver.id!r} is not active"
            )
        if approver.role not in settings.approver_roles:
            return ApprovalError(
                ApprovalErrorReason.APPROVER_NOT_ALLOWED,
                f"role {approver.role.value!r} may not decide approvals",
            )
        approval = self._load(approval_id)
        # Another customer's approval looks exactly like a missing one.
        if approval is None or approval.customer_id != approver.customer_id:
            return _not_found(approval_id)
        if approver.id == approval.requested_by:
            return ApprovalError(
                ApprovalErrorReason.SELF_APPROVAL, "an approval cannot be decided by its requester"
            )
        expired = self._expire_if_due(approval)
        if expired is not None:
            return expired
        if approval.status is not ApprovalStatus.PENDING:
            return ApprovalError(
                ApprovalErrorReason.NOT_PENDING,
                f"approval {approval_id!r} is already {approval.status.value}",
            )
        now = self._now()
        self._store.connection.execute(
            "UPDATE approvals SET status = ?, decided_by = ?, decided_at = ?, note = ?"
            " WHERE id = ?",
            (decision.value, approver.id, _ts(now), note, approval_id),
        )
        decided = self._load(approval_id)
        assert decided is not None
        return decided

    def _claim(
        self,
        approval_id: str,
        customer_id: str,
        tool: str,
        action: Action,
        args: Mapping[str, JsonValue],
    ) -> Approval | ApprovalError:
        approval = self._load(approval_id)
        if approval is None:
            return _not_found(approval_id)
        # Before anything else, so a caller from another customer learns no status, expiry or
        # args fact, and causes no write.
        if approval.customer_id != customer_id:
            return ApprovalError(
                ApprovalErrorReason.WRONG_CUSTOMER,
                f"approval {approval_id!r} belongs to another customer",
            )
        try:
            requested = args_hash(tool, action, args)
        except (TypeError, ValueError):
            # Stored args are always valid JSON, so args that cannot be hashed cannot match.
            requested = ""
        if not hmac.compare_digest(requested, approval.args_hash):
            return ApprovalError(
                ApprovalErrorReason.ARGS_MISMATCH,
                "tool, action or args differ from what was approved",
            )
        if approval.status is ApprovalStatus.EXECUTED:
            return ApprovalError(
                ApprovalErrorReason.ALREADY_EXECUTED,
                f"approval {approval_id!r} was already executed",
            )
        expired = self._expire_if_due(approval)
        if expired is not None:
            return expired
        if approval.status is not ApprovalStatus.APPROVED:
            return ApprovalError(
                ApprovalErrorReason.NOT_APPROVED,
                f"approval {approval_id!r} is {approval.status.value}, not approved",
            )
        self._store.connection.execute(
            "UPDATE approvals SET status = ?, executed_at = ? WHERE id = ? AND status = ?",
            (
                ApprovalStatus.EXECUTED.value,
                _ts(self._now()),
                approval_id,
                ApprovalStatus.APPROVED.value,
            ),
        )
        executed = self._load(approval_id)
        assert executed is not None
        return executed

    def _expire_if_due(self, approval: Approval) -> ApprovalError | None:
        """EXPIRED for an expired approval, marking a live one past its expiry on the way."""
        if approval.status is ApprovalStatus.EXPIRED:
            return _expired(approval.id)
        live = approval.status in (ApprovalStatus.PENDING, ApprovalStatus.APPROVED)
        if live and self._now() >= approval.expires_at:
            self._set_status(approval.id, ApprovalStatus.EXPIRED)
            return _expired(approval.id)
        return None

    def _set_status(self, approval_id: str, status: ApprovalStatus) -> None:
        self._store.connection.execute(
            "UPDATE approvals SET status = ? WHERE id = ?", (status.value, approval_id)
        )

    def _load(self, approval_id: str) -> Approval | None:
        row = self._store.connection.execute(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        ).fetchone()
        return None if row is None else _approval(row)

    def _now(self) -> datetime:
        now = self._clock.now()
        if now.tzinfo is None:
            raise ValueError("the clock must return timezone-aware times")
        return now.astimezone(UTC)


def _not_found(approval_id: str) -> ApprovalError:
    return ApprovalError(ApprovalErrorReason.NOT_FOUND, f"approval {approval_id!r} not found")


def _expired(approval_id: str) -> ApprovalError:
    return ApprovalError(ApprovalErrorReason.EXPIRED, f"approval {approval_id!r} has expired")


def _ts(value: datetime) -> str:
    # Fixed width UTC with microseconds, so stored times sort and compare as text.
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_ts(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _approval(row: sqlite3.Row) -> Approval:
    created_at = _parse_ts(row["created_at"])
    expires_at = _parse_ts(row["expires_at"])
    assert created_at is not None and expires_at is not None
    return Approval(
        id=row["id"],
        customer_id=row["customer_id"],
        run_id=row["run_id"],
        requested_by=row["requested_by"],
        tool=row["tool"],
        action=Action(row["action"]),
        args=json.loads(row["args"]),
        args_hash=row["args_hash"],
        risk_score=row["risk_score"],
        rule_ids=tuple(json.loads(row["rule_ids"])),
        policy_version=row["policy_version"],
        status=ApprovalStatus(row["status"]),
        created_at=created_at,
        expires_at=expires_at,
        decided_by=row["decided_by"],
        decided_at=_parse_ts(row["decided_at"]),
        note=row["note"],
        executed_at=_parse_ts(row["executed_at"]),
    )
