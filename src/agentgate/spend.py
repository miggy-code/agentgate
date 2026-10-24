"""Spend ledger: per-run and per-customer-day caps enforced before each model or tool call.

Every paid call goes through a reservation. The gate reserves the estimated cost first; the
reservation is refused with a `CapHit` if it would take the run's call count, the run's spend
or the customer's spend for the day past its limit. After the call the gate settles the
reservation with the actual cost, or releases it if the call never happened. An open
reservation counts at its reserved amount, so concurrent calls cannot all squeeze under the
same remaining budget.

The ledger keeps one row per reservation and never deletes rows: status, reserved and actual
amounts and timestamps stay behind as an audit trail. Money is `Decimal` throughout and is
stored as TEXT, because summing floats would let rounding drift past a cap.

The customer-day bucket is the customer's local calendar date when the reservation was
made. A reservation keeps that day when it is settled, even after local midnight, so a call
that straddles midnight is charged to the day that allowed it.

Reservations are leases. A host killed between reserve and settle (a deploy restart, an OOM
kill) leaves its reservation open, and an open reservation counts against the customer's
day, so enough crashed runs would lock the customer out until local midnight.
`release_stale(older_than)` reclaims every reservation still open after that long:
`open_world` calls it with a 15 minute lease on every open, and `agentgate spend
release-stale` lets an operator choose the age. A reclaimed row is `released` with
`reclaimed_at` set, which keeps it apart from a release the gate asked for:

- money: it no longer counts toward the run or the customer day, like any release;
- calls: it still counts toward the run's call limit, since the call may well have
  happened, and a run is never handed back a call by the sweep;
- late settle: if the host was only slow and settles after the sweep, the actual cost is
  recorded (the row becomes `settled`, `reclaimed_at` stays as the trail) and added back to
  the day, because that money was spent;
- late release: a release after the sweep is a no-op, as the budget is already free.
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from zoneinfo import ZoneInfo

from agentgate.clock import Clock
from agentgate.models import Customer
from agentgate.policy.schema import SpendLimits
from agentgate.store import Store

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS spend_reservations (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        customer_id TEXT NOT NULL,
        day TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('open', 'settled', 'released')),
        reserved_usd TEXT NOT NULL,
        actual_usd TEXT,
        reserved_at TEXT NOT NULL,
        finished_at TEXT,
        reclaimed_at TEXT
    ) STRICT
    """,
    "CREATE INDEX IF NOT EXISTS spend_reservations_run ON spend_reservations(run_id)",
    """
    CREATE INDEX IF NOT EXISTS spend_reservations_customer_day
        ON spend_reservations(customer_id, day)
    """,
    # The stale sweep reads only open rows, which are few however long the ledger grows.
    """
    CREATE INDEX IF NOT EXISTS spend_reservations_open
        ON spend_reservations(reserved_at) WHERE status = 'open'
    """,
    # A running total per customer day, kept in the same transaction as every reservation
    # change, so the daily cap check reads one row instead of summing the whole day.
    """
    CREATE TABLE IF NOT EXISTS spend_day_totals (
        customer_id TEXT NOT NULL,
        day TEXT NOT NULL,
        total_usd TEXT NOT NULL,
        PRIMARY KEY (customer_id, day)
    ) STRICT
    """,
)

_OPEN = "open"
_SETTLED = "settled"
_RELEASED = "released"


class CapKind(StrEnum):
    RUN = "run_cap"
    CUSTOMER_DAY = "customer_day_cap"
    CALLS = "max_calls"


@dataclass(frozen=True)
class Reservation:
    id: str
    run_id: str
    customer_id: str
    amount_usd: Decimal
    # The customer's local calendar day when reserved.
    day: date


@dataclass(frozen=True)
class CapHit:
    """Why a reservation was refused. Nothing is written when a cap is hit."""

    cap: CapKind
    limit: Decimal | int
    # Spend or calls already counted, before this request.
    used: Decimal | int
    requested: Decimal


class SpendError(ValueError):
    """Ledger misuse: a bad amount, or an unknown or already finished reservation."""


class SpendLedger:
    """Reservations against spend caps, kept in the store's database."""

    def __init__(self, store: Store, clock: Clock) -> None:
        self._store = store
        self._clock = clock
        with store.locked():
            columns = {
                str(row[1])
                for row in store.connection.execute(
                    "PRAGMA table_info(spend_reservations)"
                ).fetchall()
            }
            if columns and "reclaimed_at" not in columns:
                # A ledger written before reservations were leases. Added before the
                # schema below, whose index names the column's table.
                store.connection.execute(
                    "ALTER TABLE spend_reservations ADD COLUMN reclaimed_at TEXT"
                )
            for statement in _SCHEMA:
                # Each CREATE commits on its own; IF NOT EXISTS makes a second ledger on the
                # same database, or a reopened file, a no-op.
                store.connection.execute(statement)
        self._backfill_day_totals()

    def customer_day(self, customer: Customer) -> date:
        """Today in the customer's timezone, according to the clock."""
        return self._clock.now().astimezone(ZoneInfo(customer.timezone)).date()

    def reserve(
        self,
        *,
        customer: Customer,
        run_id: str,
        amount_usd: Decimal,
        limits: SpendLimits,
    ) -> Reservation | CapHit:
        """Reserve amount_usd for one call, or return the first cap it would break.

        Caps are checked in order: calls per run, spend per run, spend per customer day.
        Reaching a cap exactly is allowed; going one unit past it is not.
        """
        _check_amount(amount_usd)
        day = self.customer_day(customer)
        # BEGIN IMMEDIATE takes SQLite's write lock before the reads, so the totals checked
        # here are still the totals when the row is inserted, across threads and connections.
        with self._store.transaction():
            calls = self.run_calls(run_id)
            if calls >= limits.max_calls_per_run:
                return CapHit(CapKind.CALLS, limits.max_calls_per_run, calls, amount_usd)
            run_spent = self.run_spent(run_id)
            if run_spent + amount_usd > limits.per_run_usd:
                return CapHit(CapKind.RUN, limits.per_run_usd, run_spent, amount_usd)
            day_spent = self.customer_day_spent(customer.id, day)
            if day_spent + amount_usd > limits.per_customer_day_usd:
                return CapHit(
                    CapKind.CUSTOMER_DAY, limits.per_customer_day_usd, day_spent, amount_usd
                )
            reservation = Reservation(
                id=uuid.uuid4().hex,
                run_id=run_id,
                customer_id=customer.id,
                amount_usd=amount_usd,
                day=day,
            )
            self._store.connection.execute(
                "INSERT INTO spend_reservations"
                " (id, run_id, customer_id, day, status, reserved_usd, reserved_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    reservation.id,
                    run_id,
                    customer.id,
                    day.isoformat(),
                    _OPEN,
                    str(amount_usd),
                    self._clock.now().isoformat(),
                ),
            )
            self._add_to_day(customer.id, day.isoformat(), amount_usd)
            return reservation

    def settle(self, reservation: Reservation, actual_usd: Decimal) -> None:
        """Replace an open reservation with the actual cost of the call.

        The actual figure is recorded even when it exceeds the estimate or pushes a total
        past its cap, because that money was spent. Later reservations are refused instead.
        """
        _check_amount(actual_usd)
        self._finish(reservation, _SETTLED, str(actual_usd))

    def release(self, reservation: Reservation) -> None:
        """Drop an open reservation whose call never happened, freeing its budget and call."""
        self._finish(reservation, _RELEASED, None)

    def release_stale(self, older_than: timedelta) -> int:
        """Reclaim every reservation still open older_than after it was made, and return
        how many. Each one stops counting toward its run's and customer day's spend in the
        same transaction, so the running day total stays equal to a recount."""
        if older_than < timedelta(0):
            raise SpendError(f"older_than must not be negative, got {older_than}")
        now = self._clock.now()
        cutoff = now - older_than
        with self._store.transaction():
            rows = self._store.connection.execute(
                "SELECT id, customer_id, day, reserved_usd, reserved_at FROM spend_reservations"
                " WHERE status = 'open' ORDER BY reserved_at, id"
            ).fetchall()
            # Compared as datetimes, not as text, so a different UTC offset cannot misorder.
            stale = [row for row in rows if datetime.fromisoformat(row["reserved_at"]) < cutoff]
            for row in stale:
                self._store.connection.execute(
                    "UPDATE spend_reservations SET status = ?, finished_at = ?, reclaimed_at = ?"
                    " WHERE id = ? AND status = ?",
                    (_RELEASED, now.isoformat(), now.isoformat(), row["id"], _OPEN),
                )
                self._add_to_day(row["customer_id"], row["day"], -Decimal(row["reserved_usd"]))
        return len(stale)

    def run_spent(self, run_id: str) -> Decimal:
        """Settled actuals plus open reservations for one run."""
        with self._store.locked():
            rows = self._store.connection.execute(
                "SELECT status, reserved_usd, actual_usd FROM spend_reservations"
                " WHERE run_id = ? AND status != ?",
                (run_id, _RELEASED),
            ).fetchall()
        return _total(rows)

    def run_calls(self, run_id: str) -> int:
        """Reservations for one run that were not released by the gate: open, settled, or
        reclaimed by the stale sweep (whose call may have happened)."""
        with self._store.locked():
            (count,) = self._store.connection.execute(
                "SELECT COUNT(*) FROM spend_reservations"
                " WHERE run_id = ? AND (status != ? OR reclaimed_at IS NOT NULL)",
                (run_id, _RELEASED),
            ).fetchone()
        return int(count)

    def customer_day_spent(self, customer_id: str, day: date) -> Decimal:
        """Settled actuals plus open reservations for one customer on one local day."""
        with self._store.locked():
            row = self._store.connection.execute(
                "SELECT total_usd FROM spend_day_totals WHERE customer_id = ? AND day = ?",
                (customer_id, day.isoformat()),
            ).fetchone()
        return Decimal(0) if row is None else Decimal(row["total_usd"])

    def _finish(self, reservation: Reservation, status: str, actual: str | None) -> None:
        with self._store.transaction():
            row = self._store.connection.execute(
                "SELECT run_id, customer_id, day, status, reserved_usd, reclaimed_at"
                " FROM spend_reservations WHERE id = ?",
                (reservation.id,),
            ).fetchone()
            if row is None:
                raise SpendError(f"unknown reservation {reservation.id!r}")
            # A Reservation is a plain dataclass a caller could build by hand; refuse one
            # whose id points at another run's or customer's row.
            if row["run_id"] != reservation.run_id or row["customer_id"] != reservation.customer_id:
                raise SpendError(f"reservation {reservation.id!r} does not match the ledger")
            if row["status"] == _RELEASED and row["reclaimed_at"] is not None:
                self._finish_reclaimed(reservation, row, status, actual)
                return
            if row["status"] != _OPEN:
                raise SpendError(f"reservation {reservation.id!r} is already {row['status']}")
            self._store.connection.execute(
                "UPDATE spend_reservations SET status = ?, actual_usd = ?, finished_at = ?"
                " WHERE id = ? AND status = ?",
                (status, actual, self._clock.now().isoformat(), reservation.id, _OPEN),
            )
            # The open reservation counted at its reserved amount; swap in the actual cost,
            # or remove it entirely on release.
            reserved = Decimal(row["reserved_usd"])
            change = (Decimal(0) if actual is None else Decimal(actual)) - reserved
            self._add_to_day(row["customer_id"], row["day"], change)

    def _finish_reclaimed(
        self, reservation: Reservation, row: sqlite3.Row, status: str, actual: str | None
    ) -> None:
        # The sweep already took the reserved amount off the day. A late release has nothing
        # left to free; a late settle records the money that was spent after all.
        if status != _SETTLED or actual is None:
            return
        self._store.connection.execute(
            "UPDATE spend_reservations SET status = ?, actual_usd = ?, finished_at = ?"
            " WHERE id = ? AND status = ?",
            (_SETTLED, actual, self._clock.now().isoformat(), reservation.id, _RELEASED),
        )
        self._add_to_day(row["customer_id"], row["day"], Decimal(actual))

    def _add_to_day(self, customer_id: str, day: str, change: Decimal) -> None:
        # Only called inside a store transaction, alongside the reservation row it mirrors.
        row = self._store.connection.execute(
            "SELECT total_usd FROM spend_day_totals WHERE customer_id = ? AND day = ?",
            (customer_id, day),
        ).fetchone()
        total = (Decimal(0) if row is None else Decimal(row["total_usd"])) + change
        self._store.connection.execute(
            "INSERT INTO spend_day_totals (customer_id, day, total_usd) VALUES (?, ?, ?)"
            " ON CONFLICT (customer_id, day) DO UPDATE SET total_usd = excluded.total_usd",
            (customer_id, day, str(total)),
        )

    def _backfill_day_totals(self) -> None:
        # A database written before spend_day_totals existed has reservations but no totals.
        with self._store.transaction():
            (totals,) = self._store.connection.execute(
                "SELECT COUNT(*) FROM spend_day_totals"
            ).fetchone()
            if totals:
                return
            rows = self._store.connection.execute(
                "SELECT customer_id, day, status, reserved_usd, actual_usd"
                " FROM spend_reservations WHERE status != ?",
                (_RELEASED,),
            ).fetchall()
            by_day: dict[tuple[str, str], list[sqlite3.Row]] = {}
            for row in rows:
                by_day.setdefault((row["customer_id"], row["day"]), []).append(row)
            for (customer_id, day), day_rows in by_day.items():
                self._store.connection.execute(
                    "INSERT INTO spend_day_totals (customer_id, day, total_usd) VALUES (?, ?, ?)",
                    (customer_id, day, str(_total(day_rows))),
                )


def _check_amount(amount: Decimal) -> None:
    # Checked at runtime too: a float from an untyped caller would carry binary rounding
    # error into the ledger, and NaN would make every cap comparison meaningless.
    if not isinstance(amount, Decimal):
        raise SpendError(f"amount must be a Decimal, got {type(amount).__name__}")
    if not amount.is_finite():
        raise SpendError(f"amount must be finite, got {amount}")
    if amount < 0:
        raise SpendError(f"amount must not be negative, got {amount}")


def _total(rows: list[sqlite3.Row]) -> Decimal:
    # Summed in Python, not with SQL SUM, which would convert the TEXT amounts to floats.
    total = Decimal(0)
    for row in rows:
        amount = row["actual_usd"] if row["status"] == _SETTLED else row["reserved_usd"]
        total += Decimal(amount)
    return total
