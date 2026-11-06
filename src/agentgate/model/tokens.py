"""Run tokens: what an agent presents to the model gateway in place of a provider key.

A token is bound to one agentgate run. It is 256 random bits behind a prefix; only its
sha256 is stored, in the agentgate store, so a copy of the store cannot be used to call the
gateway. The token is shown once, when it is issued.

A token carries no authority of its own. It names a run, and resolving it gives back the run
as the store holds it, so what the run may do is still decided by the gate: the policy, the
user's status and the spend caps. A token is refused when:

- it is unknown, expired or revoked;
- it belongs to another run than the one the gateway serves;
- its run is unknown, or its user is unknown, of another customer or suspended.

A token dies with its run. `end_run` revokes every token of a run, which a gateway serving
one run does when it stops, and marks the run ended so no token is issued for it again; a
token also expires on its own after its lifetime. A
refusal says why in a reason code and never shows the token.
"""

import hashlib
import os
import secrets
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from agentgate.clock import Clock
from agentgate.context import RunContext
from agentgate.model.keys import PRESENTED_CREDENTIAL_HEADERS, TOKEN_PREFIX
from agentgate.models import Role, UserStatus
from agentgate.store import Store

DEFAULT_TTL = timedelta(hours=8)
MAX_TTL = timedelta(days=30)
# Live tokens one run may hold, so a loop of issue calls cannot grow the table without bound.
MAX_LIVE_TOKENS_PER_RUN = 16

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS run_tokens (
        token_id TEXT PRIMARY KEY,
        token_hash TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL,
        issued_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        revoked_at TEXT,
        revoked_reason TEXT
    ) STRICT
    """,
    "CREATE INDEX IF NOT EXISTS run_tokens_run ON run_tokens(run_id)",
    # A run whose gateway has ended it. A run has no end state of its own, so this is the
    # record that no token may be issued for it any more.
    """
    CREATE TABLE IF NOT EXISTS ended_runs (
        run_id TEXT PRIMARY KEY,
        ended_at TEXT NOT NULL
    ) STRICT
    """,
)


class TokenReason(StrEnum):
    """Why a token was refused, or why it was revoked."""

    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    EXPIRED = "expired"
    REVOKED = "revoked"
    WRONG_RUN = "wrong_run"
    RUN_INVALID = "run_invalid"
    USER_SUSPENDED = "user_suspended"
    # Revocation reasons.
    OPERATOR = "operator"
    RUN_ENDED = "run_ended"


class TokenError(ValueError):
    """A token could not be issued or revoked."""


class TokenRejected(Exception):
    """A request did not present exactly one valid token for the run it was sent to."""

    def __init__(self, reason: TokenReason, run_id: str | None = None) -> None:
        super().__init__(reason.value)
        self.reason = reason
        # Set when the token itself was recognised, so a refusal can be tied to its run.
        self.run_id = run_id


@dataclass(frozen=True)
class TokenRecord:
    """Everything stored about a token except its hash."""

    token_id: str
    run_id: str
    issued_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    revoked_reason: str | None

    def live(self, now: datetime) -> bool:
        return self.revoked_at is None and now < self.expires_at


@dataclass(frozen=True)
class IssuedToken:
    # Shown once: the store keeps only its hash.
    token: str
    record: TokenRecord

    def __repr__(self) -> str:
        return f"IssuedToken(record={self.record!r})"


def token_hash(token: str) -> str:
    # The token is 256 random bits, so a fast digest is enough and a lookup by hash is safe.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenStore:
    """Issue, revoke and resolve run tokens in the agentgate store."""

    def __init__(self, store: Store, clock: Clock) -> None:
        self._store = store
        self._clock = clock
        with store.locked():
            for statement in _SCHEMA:
                store.connection.execute(statement)

    def issue(
        self,
        run_id: str,
        ttl: timedelta = DEFAULT_TTL,
        *,
        record: Callable[[TokenRecord], None],
    ) -> IssuedToken:
        """A new token for a run the store holds. The run's user must be active.

        `record` is required: it runs inside the transaction that stores the token, so a
        token whose issue cannot be recorded does not exist (if it raises, nothing is
        stored) and none is issued unrecorded.
        """
        if not timedelta(0) < ttl <= MAX_TTL:
            raise TokenError(f"lifetime must be above zero and at most {MAX_TTL.days} days")
        now = self._clock.now()
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        issued = TokenRecord(
            token_id=secrets.token_hex(8),
            run_id=run_id,
            issued_at=now,
            expires_at=now + ttl,
            revoked_at=None,
            revoked_reason=None,
        )
        with self._store.transaction():
            self._run(run_id)
            if self._store.connection.execute(
                "SELECT 1 FROM ended_runs WHERE run_id = ?", (run_id,)
            ).fetchone():
                raise TokenError("the run has ended")
            live = self._store.connection.execute(
                "SELECT COUNT(*) FROM run_tokens WHERE run_id = ? AND revoked_at IS NULL"
                " AND expires_at > ?",
                (run_id, now.isoformat()),
            ).fetchone()[0]
            if live >= MAX_LIVE_TOKENS_PER_RUN:
                raise TokenError(f"the run already has {MAX_LIVE_TOKENS_PER_RUN} live tokens")
            self._store.connection.execute(
                "INSERT INTO run_tokens (token_id, token_hash, run_id, issued_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    issued.token_id,
                    token_hash(token),
                    run_id,
                    now.isoformat(),
                    issued.expires_at.isoformat(),
                ),
            )
            record(issued)
        return IssuedToken(token, issued)

    def revoke(
        self,
        token_id: str,
        reason: TokenReason = TokenReason.OPERATOR,
        *,
        record: Callable[[TokenRecord], None],
    ) -> bool:
        """Revoke a token by id. False if no live token had that id.

        `record` is required: it runs inside the transaction with the revoked token, so a
        revocation that cannot be recorded is not made, and none is made unrecorded.
        """
        with self._store.transaction():
            cursor = self._store.connection.execute(
                "UPDATE run_tokens SET revoked_at = ?, revoked_reason = ?"
                " WHERE token_id = ? AND revoked_at IS NULL",
                (self._clock.now().isoformat(), reason.value, token_id),
            )
            if cursor.rowcount == 0:
                return False
            revoked = self.get(token_id)
            if revoked is None:
                raise TokenError("a token disappeared while it was being revoked")
            record(revoked)
            return True

    def end_run(self, run_id: str, *, record: Callable[[TokenRecord], None]) -> int:
        """Revoke every live token of a run, and return how many.

        `record` is required: it runs inside the transaction once for each token revoked, with
        the token as revoked, so a revocation that cannot be recorded is not made (if it
        raises, no token of the run is revoked) and none is made unrecorded.
        """
        with self._store.transaction():
            ids = [
                row["token_id"]
                for row in self._store.connection.execute(
                    "SELECT token_id FROM run_tokens WHERE run_id = ? AND revoked_at IS NULL"
                    " ORDER BY issued_at, token_id",
                    (run_id,),
                ).fetchall()
            ]
            self._store.connection.execute(
                "UPDATE run_tokens SET revoked_at = ?, revoked_reason = ?"
                " WHERE run_id = ? AND revoked_at IS NULL",
                (self._clock.now().isoformat(), TokenReason.RUN_ENDED.value, run_id),
            )
            self._store.connection.execute(
                "INSERT OR IGNORE INTO ended_runs (run_id, ended_at) VALUES (?, ?)",
                (run_id, self._clock.now().isoformat()),
            )
            for token_id in ids:
                revoked = self.get(token_id)
                if revoked is None:
                    raise TokenError("a token disappeared while its run was ending")
                record(revoked)
            return len(ids)

    def get(self, token_id: str) -> TokenRecord | None:
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT * FROM run_tokens WHERE token_id = ?", (token_id,)
            ).fetchone()
        return None if row is None else _record(row)

    def list_tokens(self, run_id: str | None = None) -> list[TokenRecord]:
        """Token metadata, newest first. Hashes are never returned."""
        query = "SELECT * FROM run_tokens"
        params: tuple[str, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        with self._store.locked():
            rows = self._store.connection.execute(
                query + " ORDER BY issued_at DESC, token_id", params
            ).fetchall()
        return [_record(row) for row in rows]

    def resolve(self, token: str | None, *, run_id: str | None = None) -> RunContext:
        """The run a live token belongs to, as the store holds it, or TokenRejected.

        With run_id, a token of any other run is refused (`wrong_run`): a gateway serves one
        run, and a token from another run is not a way into it.
        """
        if not token:
            raise TokenRejected(TokenReason.MISSING)
        if not token.startswith(TOKEN_PREFIX):
            raise TokenRejected(TokenReason.UNKNOWN)
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT * FROM run_tokens WHERE token_hash = ?", (token_hash(token),)
            ).fetchone()
        if row is None:
            raise TokenRejected(TokenReason.UNKNOWN)
        record = _record(row)
        if record.revoked_at is not None:
            raise TokenRejected(TokenReason.REVOKED, record.run_id)
        if self._clock.now() >= record.expires_at:
            raise TokenRejected(TokenReason.EXPIRED, record.run_id)
        if run_id is not None and record.run_id != run_id:
            raise TokenRejected(TokenReason.WRONG_RUN, record.run_id)
        try:
            return self._run(record.run_id)
        except _SuspendedUser:
            raise TokenRejected(TokenReason.USER_SUSPENDED, record.run_id) from None
        except TokenError:
            raise TokenRejected(TokenReason.RUN_INVALID, record.run_id) from None

    def run(self, run_id: str) -> RunContext | None:
        """The run as its row says, whatever its user's status, or None if there is none."""
        with self._store.locked():
            try:
                row = self._store.connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
            except sqlite3.OperationalError:
                # No gate has opened this store, so no run exists.
                return None
        if row is None:
            return None
        return RunContext(
            run_id=row["run_id"],
            customer_id=row["customer_id"],
            user_id=row["user_id"],
            role=Role(row["role"]),
            started_at=datetime.fromisoformat(row["started_at"]),
        )

    def _run(self, run_id: str) -> RunContext:
        """The run, if its user is known, of the run's customer and active. A user who left
        or changed role is the gate's to refuse on the next call."""
        run = self.run(run_id)
        if run is None:
            raise TokenError("unknown run")
        user = self._store.get_user(run.user_id)
        if user is None or user.customer_id != run.customer_id:
            raise TokenError("the run's user is not valid")
        if user.status is not UserStatus.ACTIVE:
            raise _SuspendedUser("the run's user is suspended")
        return run


class _SuspendedUser(TokenError):
    pass


def _record(row: sqlite3.Row) -> TokenRecord:
    revoked = row["revoked_at"]
    return TokenRecord(
        token_id=row["token_id"],
        run_id=row["run_id"],
        issued_at=datetime.fromisoformat(row["issued_at"]),
        expires_at=datetime.fromisoformat(row["expires_at"]),
        revoked_at=None if revoked is None else datetime.fromisoformat(revoked),
        revoked_reason=row["revoked_reason"],
    )


def presented_token(
    headers: Iterable[tuple[str, str]], query_values: Sequence[str] = ()
) -> str | None:
    """The one credential a client presented, a Bearer scheme removed.

    Several different values are ambiguous (`TokenRejected`): none of them is chosen over
    the others, so a client cannot put one value where the gateway reads and another where
    something else might. The same value in two places is one credential.
    """
    wanted = set(PRESENTED_CREDENTIAL_HEADERS)
    presented = list(query_values)
    for name, value in headers:
        if name.lower() in wanted:
            text = value.strip()
            scheme, _, rest = text.partition(" ")
            presented.append(rest.strip() if scheme.lower() == "bearer" else text)
    distinct = {value for value in presented if value}
    if len(distinct) > 1:
        raise TokenRejected(TokenReason.AMBIGUOUS)
    return next(iter(distinct), None)


def write_token_file(path: Path, token: str) -> None:
    """Write a token to a new file readable by its owner only.

    The file is created exclusively and without following a symlink, with the mode set at
    creation, so there is no moment when another user could read it and a planted link
    cannot redirect it. An existing file is an error rather than overwritten.
    """
    write_private_file(path, token + "\n")


def write_private_file(path: Path, text: str) -> None:
    """Write text to a new file readable by its owner only (see write_token_file)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
