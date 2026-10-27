"""Mutation tests for the spend runaway scenarios: each fails once the cap it relies on is
raised in a copy of the policy bundle (through Arena.regate) or the ledger is broken on
purpose.
"""

from datetime import UTC
from decimal import Decimal
from typing import Any

import pytest

from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import SpendLimits
from agentgate.redteam.core import Arena, Category
from agentgate.spend import SpendLedger
from conftest import Mutation, assert_breaks, assert_every_attack_has_a_mutation, run_mutated


def _raised(**update: Any) -> Any:
    """Raise one cap in a copy of every bundle, then regate the arena with it."""

    def regate(arena: Arena, _: pytest.MonkeyPatch) -> None:
        engines: dict[str, PolicyEngine] = {}
        for customer_id, engine in arena.engines.items():
            policy = engine.bundle.policy
            spend = policy.spend.model_copy(update=update)
            bundle = engine.bundle.model_copy(
                update={"policy": policy.model_copy(update={"spend": spend})}
            )
            customer = arena.store.get_customer(customer_id)
            assert customer is not None
            engines[customer_id] = PolicyEngine(bundle, customer)
        arena.regate(engines=engines)
        for engine in arena.engines.values():
            assert engine.bundle.policy.spend.model_dump(include=set(update)) == update

    return regate


_RESERVE = SpendLedger.reserve
_NO_LIMITS = SpendLimits(
    per_run_usd=Decimal("999999"),
    per_customer_day_usd=Decimal("999999"),
    max_calls_per_run=10_000,
)


def _ignore_limits(self: SpendLedger, **kwargs: Any) -> Any:
    return _RESERVE(self, **(kwargs | {"limits": _NO_LIMITS}))


def _utc_day(self: SpendLedger, customer: object) -> Any:
    return self._clock.now().astimezone(UTC).date()


DAY_CAP_RAISED = Mutation(
    "day cap raised",
    ("sr-005", "sr-006", "sr-007", "sr-012"),
    arena=_raised(per_customer_day_usd=Decimal("1000.00")),
)

MUTATIONS = [
    Mutation(
        "call cap raised",
        ("sr-001", "sr-008", "sr-009", "sr-010", "sr-011", "sr-012", "sr-013"),
        arena=_raised(max_calls_per_run=1000),
    ),
    Mutation(
        "run cap raised",
        ("sr-002", "sr-003", "sr-004", "sr-012", "sr-013", "sr-014"),
        arena=_raised(per_run_usd=Decimal("100.00")),
    ),
    DAY_CAP_RAISED,
    Mutation(
        "release keeps the reservation",
        ("sr-009", "sr-010"),
        replace=((SpendLedger, "release", lambda self, reservation: None),),
    ),
    Mutation("the day is UTC", ("sr-006",), replace=((SpendLedger, "customer_day", _utc_day),)),
    Mutation(
        "the ledger ignores limits",
        tuple(f"sr-{n:03d}" for n in range(1, 15)),
        replace=((SpendLedger, "reserve", _ignore_limits),),
    ),
]


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.name)
def test_mutation_fails_its_scenarios(mutation: Mutation, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_breaks(mutation, monkeypatch)


def test_every_attack_has_a_mutation() -> None:
    assert_every_attack_has_a_mutation(MUTATIONS, Category.SPEND_RUNAWAY)


def test_raised_caps_still_pass_unrelated_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    # Raising the day cap leaves the call-limit loop exactly as it was: the mutation tests
    # above fail because of the cap, not because regating breaks scenarios.
    ids = ("sr-001", "sr-002", "sr-008", "sr-011")
    suite = run_mutated(DAY_CAP_RAISED, monkeypatch, ids=ids)
    assert [r.id for r in suite.failed] == []
