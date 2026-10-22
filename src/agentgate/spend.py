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
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date
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
        finished_at TEXT
    ) STRICT
    """,
    "CREATE INDEX IF NOT EXISTS spend_reservations_run ON spend_reservations(run_id)",
    """
    CREATE INDEX IF NOT EXISTS spend_reservations_customer_day
        ON spend_reservations(customer_id, day)
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
            for statement in _SCHEMA:
                # Each CREATE commits on its own; IF NOT EXISTS makes a second ledger on the
                # same database, or a reopened file, a no-op.
                store.connection.execute(statement)

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
        """Reservations for one run that were not released, open or settled."""
        with self._store.locked():
            (count,) = self._store.connection.execute(
                "SELECT COUNT(*) FROM spend_reservations WHERE run_id = ? AND status != ?",
                (run_id, _RELEASED),
            ).fetchone()
        return int(count)

    def customer_day_spent(self, customer_id: str, day: date) -> Decimal:
        """Settled actuals plus open reservations for one customer on one local day."""
        with self._store.locked():
            rows = self._store.connection.execute(
                "SELECT status, reserved_usd, actual_usd FROM spend_reservations"
                " WHERE customer_id = ? AND day = ? AND status != ?",
                (customer_id, day.isoformat(), _RELEASED),
            ).fetchall()
        return _total(rows)

    def _finish(self, reservation: Reservation, status: str, actual: str | None) -> None:
        with self._store.transaction():
            row = self._store.connection.execute(
                "SELECT run_id, customer_id, status FROM spend_reservations WHERE id = ?",
                (reservation.id,),
            ).fetchone()
            if row is None:
                raise SpendError(f"unknown reservation {reservation.id!r}")
            # A Reservation is a plain dataclass a caller could build by hand; refuse one
            # whose id points at another run's or customer's row.
            if row["run_id"] != reservation.run_id or row["customer_id"] != reservation.customer_id:
                raise SpendError(f"reservation {reservation.id!r} does not match the ledger")
            if row["status"] != _OPEN:
                raise SpendError(f"reservation {reservation.id!r} is already {row['status']}")
            self._store.connection.execute(
                "UPDATE spend_reservations SET status = ?, actual_usd = ?, finished_at = ?"
                " WHERE id = ? AND status = ?",
                (status, actual, self._clock.now().isoformat(), reservation.id, _OPEN),
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
