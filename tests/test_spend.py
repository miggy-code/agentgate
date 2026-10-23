import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agentgate.clock import FixedClock
from agentgate.models import Customer
from agentgate.policy.schema import SpendLimits
from agentgate.spend import CapHit, CapKind, Reservation, SpendError, SpendLedger
from agentgate.store import Store

CHICAGO = ZoneInfo("America/Chicago")
# 12:00 in Chicago (CDT, UTC-5), well away from midnight.
NOON_CHICAGO = datetime(2026, 10, 22, 12, 0, tzinfo=CHICAGO)
TODAY = date(2026, 10, 22)


def _customer(customer_id: str = "acme") -> Customer:
    return Customer(
        id=customer_id,
        name=customer_id.title(),
        domains=(f"{customer_id}.example",),
        timezone="America/Chicago",
    )


def _limits(run: str = "1.00", day: str = "5.00", calls: int = 100) -> SpendLimits:
    return SpendLimits(
        per_run_usd=Decimal(run), per_customer_day_usd=Decimal(day), max_calls_per_run=calls
    )


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(NOON_CHICAGO)


@pytest.fixture
def ledger(clock: FixedClock) -> SpendLedger:
    return SpendLedger(Store(), clock)


def _ok(result: Reservation | CapHit) -> Reservation:
    assert isinstance(result, Reservation), result
    return result


def _hit(result: Reservation | CapHit) -> CapHit:
    assert isinstance(result, CapHit), result
    return result


def _reserve(
    ledger: SpendLedger,
    amount: str,
    *,
    run_id: str = "run-1",
    customer: Customer | None = None,
    limits: SpendLimits | None = None,
) -> Reservation | CapHit:
    return ledger.reserve(
        customer=customer or _customer(),
        run_id=run_id,
        amount_usd=Decimal(amount),
        limits=limits or _limits(),
    )


def test_reservation_records_run_customer_amount_and_local_day(ledger: SpendLedger) -> None:
    reservation = _ok(_reserve(ledger, "0.25"))
    assert reservation.run_id == "run-1"
    assert reservation.customer_id == "acme"
    assert reservation.amount_usd == Decimal("0.25")
    assert reservation.day == TODAY
    assert ledger.run_spent("run-1") == Decimal("0.25")
    assert ledger.run_calls("run-1") == 1
    assert ledger.customer_day_spent("acme", TODAY) == Decimal("0.25")


def test_run_cap_allows_reaching_it_exactly(ledger: SpendLedger) -> None:
    _ok(_reserve(ledger, "0.60"))
    _ok(_reserve(ledger, "0.40"))
    assert ledger.run_spent("run-1") == Decimal("1.00")


def test_run_cap_refuses_one_cent_over(ledger: SpendLedger) -> None:
    _ok(_reserve(ledger, "0.60"))
    hit = _hit(_reserve(ledger, "0.41"))
    assert hit == CapHit(CapKind.RUN, Decimal("1.00"), Decimal("0.60"), Decimal("0.41"))


def test_day_cap_allows_reaching_it_exactly(ledger: SpendLedger) -> None:
    limits = _limits(run="10", day="2.00")
    _ok(_reserve(ledger, "1.50", run_id="a", limits=limits))
    _ok(_reserve(ledger, "0.50", run_id="b", limits=limits))
    assert ledger.customer_day_spent("acme", TODAY) == Decimal("2.00")


def test_day_cap_refuses_one_cent_over(ledger: SpendLedger) -> None:
    limits = _limits(run="10", day="2.00")
    _ok(_reserve(ledger, "1.50", run_id="a", limits=limits))
    hit = _hit(_reserve(ledger, "0.51", run_id="b", limits=limits))
    assert hit == CapHit(CapKind.CUSTOMER_DAY, Decimal("2.00"), Decimal("1.50"), Decimal("0.51"))


def test_cap_hit_writes_nothing(ledger: SpendLedger) -> None:
    _hit(_reserve(ledger, "1.01"))
    assert ledger.run_spent("run-1") == 0
    assert ledger.run_calls("run-1") == 0
    assert ledger.customer_day_spent("acme", TODAY) == 0


def test_max_calls_counts_open_and_settled_but_not_released(ledger: SpendLedger) -> None:
    limits = _limits(calls=3)
    settled = _ok(_reserve(ledger, "0.10", limits=limits))
    ledger.settle(settled, Decimal("0.10"))
    released = _ok(_reserve(ledger, "0.10", limits=limits))
    ledger.release(released)
    _ok(_reserve(ledger, "0.10", limits=limits))
    assert ledger.run_calls("run-1") == 2

    _ok(_reserve(ledger, "0.10", limits=limits))
    hit = _hit(_reserve(ledger, "0", limits=limits))
    assert hit == CapHit(CapKind.CALLS, 3, 3, Decimal("0"))


def test_free_calls_count_toward_max_calls(ledger: SpendLedger) -> None:
    limits = _limits(calls=2)
    _ok(_reserve(ledger, "0", limits=limits))
    _ok(_reserve(ledger, "0", limits=limits))
    assert _hit(_reserve(ledger, "0", limits=limits)).cap is CapKind.CALLS
    assert ledger.run_spent("run-1") == 0


def test_caps_are_checked_calls_then_run_then_day(ledger: SpendLedger) -> None:
    limits = _limits(run="1.00", day="1.00", calls=1)
    _ok(_reserve(ledger, "1.00", limits=limits))
    # Every cap is now full; the call cap is reported first.
    assert _hit(_reserve(ledger, "0.01", limits=limits)).cap is CapKind.CALLS
    more_calls = _limits(run="1.00", day="1.00", calls=5)
    assert _hit(_reserve(ledger, "0.01", limits=more_calls)).cap is CapKind.RUN
    other_run = _hit(_reserve(ledger, "0.01", run_id="run-2", limits=more_calls))
    assert other_run == CapHit(
        CapKind.CUSTOMER_DAY, Decimal("1.00"), Decimal("1.00"), Decimal("0.01")
    )


def test_settle_below_estimate_frees_the_difference(ledger: SpendLedger) -> None:
    reservation = _ok(_reserve(ledger, "0.80"))
    ledger.settle(reservation, Decimal("0.30"))
    assert ledger.run_spent("run-1") == Decimal("0.30")
    assert ledger.customer_day_spent("acme", TODAY) == Decimal("0.30")
    assert ledger.run_calls("run-1") == 1
    _ok(_reserve(ledger, "0.70"))


def test_settle_above_estimate_records_actual_and_blocks_later_reserves(
    ledger: SpendLedger,
) -> None:
    reservation = _ok(_reserve(ledger, "0.90"))
    ledger.settle(reservation, Decimal("1.25"))
    assert ledger.run_spent("run-1") == Decimal("1.25")
    assert ledger.customer_day_spent("acme", TODAY) == Decimal("1.25")
    hit = _hit(_reserve(ledger, "0"))
    assert hit == CapHit(CapKind.RUN, Decimal("1.00"), Decimal("1.25"), Decimal("0"))


def test_release_frees_budget(ledger: SpendLedger) -> None:
    reservation = _ok(_reserve(ledger, "1.00"))
    _hit(_reserve(ledger, "0.01"))
    ledger.release(reservation)
    assert ledger.run_spent("run-1") == 0
    assert ledger.customer_day_spent("acme", TODAY) == 0
    _ok(_reserve(ledger, "1.00"))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("settle", "settle"),
        ("release", "release"),
        ("release", "settle"),
        ("settle", "release"),
    ],
)
def test_finishing_a_reservation_twice_is_an_error(
    ledger: SpendLedger, first: str, second: str
) -> None:
    reservation = _ok(_reserve(ledger, "0.50"))
    finish: dict[str, Callable[[Reservation], None]] = {
        "settle": lambda r: ledger.settle(r, Decimal("0.40")),
        "release": ledger.release,
    }
    finish[first](reservation)
    with pytest.raises(SpendError, match="already"):
        finish[second](reservation)
    expected = Decimal("0.40") if first == "settle" else Decimal(0)
    assert ledger.run_spent("run-1") == expected


def test_unknown_reservation_is_an_error(ledger: SpendLedger) -> None:
    ghost = Reservation("nope", "run-1", "acme", Decimal("0.10"), TODAY)
    with pytest.raises(SpendError, match="unknown"):
        ledger.settle(ghost, Decimal("0.10"))
    with pytest.raises(SpendError, match="unknown"):
        ledger.release(ghost)


def test_forged_reservation_for_another_run_is_an_error(ledger: SpendLedger) -> None:
    real = _ok(_reserve(ledger, "0.10"))
    forged = Reservation(real.id, "run-2", real.customer_id, real.amount_usd, real.day)
    with pytest.raises(SpendError, match="does not match"):
        ledger.release(forged)
    assert ledger.run_calls("run-1") == 1


@pytest.mark.parametrize("amount", ["-0.01", "NaN", "sNaN", "Infinity", "-Infinity"])
def test_invalid_amounts_are_rejected(ledger: SpendLedger, amount: str) -> None:
    with pytest.raises(SpendError):
        _reserve(ledger, amount)
    reservation = _ok(_reserve(ledger, "0.10"))
    with pytest.raises(SpendError):
        ledger.settle(reservation, Decimal(amount))
    # The failed settle left the reservation open.
    ledger.release(reservation)


def test_float_amounts_are_rejected(ledger: SpendLedger) -> None:
    with pytest.raises(SpendError, match="Decimal"):
        ledger.reserve(
            customer=_customer(),
            run_id="run-1",
            amount_usd=0.1,  # type: ignore[arg-type]
            limits=_limits(),
        )


def test_day_rolls_over_at_local_midnight() -> None:
    clock = FixedClock(datetime(2026, 10, 22, 23, 59, tzinfo=CHICAGO))
    ledger = SpendLedger(Store(), clock)
    limits = _limits(run="10", day="1.00")
    late = _ok(_reserve(ledger, "1.00", run_id="a", limits=limits))
    assert late.day == date(2026, 10, 22)
    # Still the 22nd in Chicago although it is already the 23rd in UTC.
    assert clock.now().astimezone(UTC).date() == date(2026, 10, 23)
    assert _hit(_reserve(ledger, "0.01", run_id="b", limits=limits)).cap is CapKind.CUSTOMER_DAY

    clock.advance(timedelta(minutes=2))
    assert ledger.customer_day(_customer()) == date(2026, 10, 23)
    early = _ok(_reserve(ledger, "1.00", run_id="b", limits=limits))
    assert early.day == date(2026, 10, 23)

    ledger.settle(late, Decimal("0.70"))
    assert ledger.customer_day_spent("acme", date(2026, 10, 22)) == Decimal("0.70")
    assert ledger.customer_day_spent("acme", date(2026, 10, 23)) == Decimal("1.00")


def test_customers_do_not_share_a_day_budget(ledger: SpendLedger) -> None:
    limits = _limits(run="10", day="1.00")
    _ok(_reserve(ledger, "1.00", run_id="a", customer=_customer("acme"), limits=limits))
    _ok(_reserve(ledger, "1.00", run_id="b", customer=_customer("globex"), limits=limits))
    assert ledger.customer_day_spent("acme", TODAY) == Decimal("1.00")
    assert ledger.customer_day_spent("globex", TODAY) == Decimal("1.00")


def test_runs_of_one_customer_share_the_day_but_not_the_run_budget(
    ledger: SpendLedger,
) -> None:
    limits = _limits(run="1.00", day="1.50", calls=1)
    _ok(_reserve(ledger, "1.00", run_id="a", limits=limits))
    # Run b has its own run budget and call count, but only 0.50 of the day is left.
    _ok(_reserve(ledger, "0.50", run_id="b", limits=_limits(run="1.00", day="1.50", calls=2)))
    hit = _hit(_reserve(ledger, "0.01", run_id="b", limits=_limits(run="1.00", day="1.50")))
    assert hit.cap is CapKind.CUSTOMER_DAY
    assert ledger.run_spent("a") == Decimal("1.00")
    assert ledger.run_spent("b") == Decimal("0.50")


def test_rows_keep_an_audit_trail_with_text_amounts(tmp_path: Path, clock: FixedClock) -> None:
    db = tmp_path / "agentgate.db"
    with Store(db) as store:
        ledger = SpendLedger(store, clock)
        settled = _ok(_reserve(ledger, "0.10"))
        clock.advance(timedelta(seconds=5))
        ledger.settle(settled, Decimal("0.123456"))
        released = _ok(_reserve(ledger, "0.20"))
        ledger.release(released)
        _ok(_reserve(ledger, "0.30"))

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT status, reserved_usd, actual_usd, typeof(reserved_usd), day,"
            " reserved_at, finished_at FROM spend_reservations ORDER BY reserved_usd"
        ).fetchall()
    finally:
        conn.close()
    noon = NOON_CHICAGO.astimezone(UTC)
    later = (noon + timedelta(seconds=5)).isoformat()
    assert rows == [
        ("settled", "0.10", "0.123456", "text", "2026-10-22", noon.isoformat(), later),
        ("released", "0.20", None, "text", "2026-10-22", later, later),
        ("open", "0.30", None, "text", "2026-10-22", later, None),
    ]


def test_ledger_reopens_an_existing_database(tmp_path: Path, clock: FixedClock) -> None:
    db = tmp_path / "agentgate.db"
    with Store(db) as store:
        _ok(_reserve(SpendLedger(store, clock), "0.40"))
    with Store(db) as store:
        ledger = SpendLedger(store, clock)
        assert ledger.run_spent("run-1") == Decimal("0.40")


CAP = Decimal("1.00")
AMOUNT = Decimal("0.07")
THREADS = 20
ATTEMPTS = 5


def _hammer(ledgers: list[SpendLedger]) -> list[Reservation]:
    """Each thread reserves repeatedly, on its own run, against one customer-day cap."""
    limits = _limits(run="100", day=str(CAP))
    customer = _customer()
    barrier = threading.Barrier(len(ledgers))
    successes: list[Reservation] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def work(index: int, ledger: SpendLedger) -> None:
        try:
            barrier.wait()
            for _ in range(ATTEMPTS):
                result = ledger.reserve(
                    customer=customer, run_id=f"run-{index}", amount_usd=AMOUNT, limits=limits
                )
                if isinstance(result, Reservation):
                    with results_lock:
                        successes.append(result)
        except BaseException as exc:
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=work, args=(i, ledger)) for i, ledger in enumerate(ledgers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    return successes


def test_concurrent_reserves_through_one_ledger_never_exceed_the_cap(
    tmp_path: Path, clock: FixedClock
) -> None:
    with Store(tmp_path / "agentgate.db") as store:
        ledger = SpendLedger(store, clock)
        successes = _hammer([ledger] * THREADS)
        spent = ledger.customer_day_spent("acme", TODAY)
    assert THREADS * ATTEMPTS * AMOUNT > CAP
    assert len(successes) == int(CAP // AMOUNT)
    assert spent == sum((r.amount_usd for r in successes), Decimal(0))
    assert spent <= CAP


def test_concurrent_reserves_across_connections_never_exceed_the_cap(
    tmp_path: Path, clock: FixedClock
) -> None:
    db = tmp_path / "agentgate.db"
    # One Store, and so one connection, per thread: only SQLite's BEGIN IMMEDIATE write lock
    # serialises them, since the in-process store locks are not shared.
    stores = [Store(db) for _ in range(THREADS)]
    try:
        ledgers = [SpendLedger(store, clock) for store in stores]
        successes = _hammer(ledgers)
        spent = ledgers[0].customer_day_spent("acme", TODAY)
    finally:
        for store in stores:
            store.close()
    assert len(successes) == int(CAP // AMOUNT)
    assert spent == sum((r.amount_usd for r in successes), Decimal(0))
    assert spent <= CAP


def _recount(store: Store, customer_id: str, day: date) -> Decimal:
    # The day total computed the slow way, straight from the reservation rows.
    with store.locked():
        rows = store.connection.execute(
            "SELECT status, reserved_usd, actual_usd FROM spend_reservations"
            " WHERE customer_id = ? AND day = ? AND status != 'released'",
            (customer_id, day.isoformat()),
        ).fetchall()
    return sum(
        (Decimal(r["actual_usd"] if r["status"] == "settled" else r["reserved_usd"]) for r in rows),
        Decimal(0),
    )


def test_running_day_total_matches_a_full_recount(clock: FixedClock) -> None:
    store = Store()
    ledger = SpendLedger(store, clock)
    acme, beta = _customer("acme"), _customer("beta")
    limits = _limits(run="100", day="1000", calls=1000)
    # A fixed mix of reserve, settle above and below the estimate, and release.
    for i in range(300):
        customer = acme if i % 3 else beta
        reservation = _ok(
            _reserve(
                ledger, f"0.{i % 97:02d}", run_id=f"run-{i % 7}", customer=customer, limits=limits
            )
        )
        if i % 5 == 0:
            ledger.release(reservation)
        elif i % 2 == 0:
            ledger.settle(reservation, reservation.amount_usd * 3)
        elif i % 7 == 0:
            ledger.settle(reservation, Decimal(0))
        if i == 150:
            clock.advance(timedelta(days=1))
    for customer in (acme, beta):
        for day in (TODAY, TODAY + timedelta(days=1)):
            assert ledger.customer_day_spent(customer.id, day) == _recount(store, customer.id, day)


def test_day_totals_are_rebuilt_for_a_ledger_written_before_they_existed(
    tmp_path: Path, clock: FixedClock
) -> None:
    path = tmp_path / "ledger.db"
    first = SpendLedger(Store(path), clock)
    kept = _ok(_reserve(first, "0.40"))
    first.settle(kept, Decimal("0.55"))
    _ok(_reserve(first, "0.10"))
    first.release(_ok(_reserve(first, "0.30")))
    # Simulate the older schema: the totals table did not exist yet.
    store = Store(path)
    with store.transaction():
        store.connection.execute("DELETE FROM spend_day_totals")
    reopened = SpendLedger(store, clock)
    assert reopened.customer_day_spent("acme", TODAY) == Decimal("0.65")
