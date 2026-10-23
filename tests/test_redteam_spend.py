"""The spend runaway scenarios: they all pass against the real gate and policies, they are
deterministic, and they are not vacuous. Each one fails once the cap it relies on is raised
in a copy of the policy bundle (through Arena.regate) or the ledger is broken on purpose.
"""

from collections.abc import Callable, Iterator
from datetime import UTC
from decimal import Decimal
from typing import Any

import pytest

from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import SpendLimits
from agentgate.redteam import core
from agentgate.redteam.core import (
    Arena,
    Category,
    RegisteredScenario,
    ScenarioResult,
    all_scenarios,
    run_scenario,
    run_suite,
)
from agentgate.spend import SpendLedger
from conftest import FIXTURES, POLICIES

MODULE = "agentgate.redteam.scenarios.spend_runaway"
EXPECTED_IDS = [f"sr-{n:03d}" for n in range(1, 15)]
# Em and en dashes, written by code point so this file holds neither.
DASHES = (chr(0x2014), chr(0x2013))


def _spend_scenarios() -> list[RegisteredScenario]:
    return [s for s in all_scenarios() if s.category is Category.SPEND_RUNAWAY]


def _by_id(scenario_id: str) -> RegisteredScenario:
    return next(s for s in all_scenarios() if s.id == scenario_id)


def _run(scenario_id: str) -> ScenarioResult:
    return run_scenario(_by_id(scenario_id), policies=POLICIES, fixtures=FIXTURES)


def _failures(result: ScenarioResult) -> str:
    failed = [f"{c.name} ({c.detail})" for c in result.checks if not c.passed]
    return f"{result.id}: error={result.error} leaks={list(result.leaks)} failed={failed}"


# The suite


def test_every_spend_runaway_scenario_passes() -> None:
    suite = run_suite(categories=[Category.SPEND_RUNAWAY], policies=POLICIES, fixtures=FIXTURES)
    assert [r.id for r in suite.results] == EXPECTED_IDS
    assert suite.all_passed, "\n".join(_failures(r) for r in suite.failed)
    assert suite.totals == {Category.SPEND_RUNAWAY: (len(EXPECTED_IDS), len(EXPECTED_IDS))}
    assert all(len(r.checks) >= 5 for r in suite.results)
    assert suite.benign_rate is None


def test_spend_runaway_results_are_deterministic() -> None:
    first = run_suite(categories=[Category.SPEND_RUNAWAY], policies=POLICIES, fixtures=FIXTURES)
    second = run_suite(categories=[Category.SPEND_RUNAWAY], policies=POLICIES, fixtures=FIXTURES)
    assert first == second


def test_ids_prefixes_and_titles() -> None:
    scenarios = _spend_scenarios()
    assert [s.id for s in scenarios] == EXPECTED_IDS
    for s in scenarios:
        assert s.module == MODULE
        assert s.id.startswith(Category.SPEND_RUNAWAY.prefix + "-")
        assert not s.benign
        title = s.title
        assert title == title.strip() and title[0].isupper()
        assert len(title) <= 80, title
        assert not title.endswith("."), title
        assert not any(dash in title for dash in DASHES), title
    titles = [s.title for s in scenarios]
    assert len(set(titles)) == len(titles)
    # Nothing else registered from this module, under another category.
    others = [s for s in all_scenarios() if s.module == MODULE and s not in scenarios]
    assert others == []


def test_module_has_no_em_dashes() -> None:
    source = _by_id("sr-001").fn.__code__.co_filename
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert not any(dash in text for dash in DASHES)


# Mutations: raise one cap in a copy of every bundle, then regate the arena with it.


def _raised(**update: Any) -> Callable[..., None]:
    original = Arena.__init__

    def init(self: Arena, **kwargs: Any) -> None:
        original(self, **kwargs)
        engines: dict[str, PolicyEngine] = {}
        for customer_id, engine in self.engines.items():
            policy = engine.bundle.policy
            spend = policy.spend.model_copy(update=update)
            bundle = engine.bundle.model_copy(
                update={"policy": policy.model_copy(update={"spend": spend})}
            )
            customer = self.store.get_customer(customer_id)
            assert customer is not None
            engines[customer_id] = PolicyEngine(bundle, customer)
        self.regate(engines=engines)
        for engine in self.engines.values():
            assert engine.bundle.policy.spend.model_dump(include=set(update)) == update

    return init


POLICY_MUTATIONS = {
    "max_calls": {"max_calls_per_run": 1000},
    "run_cap": {"per_run_usd": Decimal("100.00")},
    "day_cap": {"per_customer_day_usd": Decimal("1000.00")},
}

# Which scenarios each raised cap must break. Every one of the three kinds has several.
POLICY_CASES = [
    ("max_calls", "sr-001"),
    ("max_calls", "sr-008"),
    ("max_calls", "sr-009"),
    ("max_calls", "sr-010"),
    ("max_calls", "sr-011"),
    ("max_calls", "sr-012"),
    ("max_calls", "sr-013"),
    ("run_cap", "sr-002"),
    ("run_cap", "sr-003"),
    ("run_cap", "sr-004"),
    ("run_cap", "sr-012"),
    ("run_cap", "sr-013"),
    ("run_cap", "sr-014"),
    ("day_cap", "sr-005"),
    ("day_cap", "sr-006"),
    ("day_cap", "sr-007"),
    ("day_cap", "sr-012"),
]


@pytest.mark.parametrize(("mutation", "scenario_id"), POLICY_CASES)
def test_scenario_fails_when_its_cap_is_raised(
    monkeypatch: pytest.MonkeyPatch, mutation: str, scenario_id: str
) -> None:
    monkeypatch.setattr(core.Arena, "__init__", _raised(**POLICY_MUTATIONS[mutation]))
    result = _run(scenario_id)
    assert not result.passed
    # It fails on its own checks, not by crashing or leaking.
    assert result.error is None
    assert result.leaks == ()
    assert any(not c.passed for c in result.checks)


def test_raised_caps_still_pass_unrelated_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    # Raising the day cap leaves the call-limit loop exactly as it was: the mutation tests
    # above fail because of the cap, not because regating breaks scenarios.
    monkeypatch.setattr(core.Arena, "__init__", _raised(**POLICY_MUTATIONS["day_cap"]))
    for scenario_id in ("sr-001", "sr-002", "sr-008", "sr-011"):
        result = _run(scenario_id)
        assert result.passed, _failures(result)


# Mutations of the ledger itself.

_RESERVE = SpendLedger.reserve
_NO_LIMITS = SpendLimits(
    per_run_usd=Decimal("999999"),
    per_customer_day_usd=Decimal("999999"),
    max_calls_per_run=10_000,
)


def _ignore_limits(self: SpendLedger, **kwargs: Any) -> Any:
    return _RESERVE(self, **(kwargs | {"limits": _NO_LIMITS}))


@pytest.fixture
def released_calls_kept(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A ledger whose release does nothing, so a denied call keeps its reservation."""
    monkeypatch.setattr(SpendLedger, "release", lambda self, reservation: None)
    yield


@pytest.mark.parametrize("scenario_id", ["sr-009", "sr-010"])
@pytest.mark.usefixtures("released_calls_kept")
def test_release_scenarios_fail_when_release_does_nothing(scenario_id: str) -> None:
    result = _run(scenario_id)
    assert not result.passed
    assert result.error is None


def test_midnight_scenario_fails_when_the_day_is_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    def utc_day(self: SpendLedger, customer: object) -> Any:
        return self._clock.now().astimezone(UTC).date()

    monkeypatch.setattr(SpendLedger, "customer_day", utc_day)
    result = _run("sr-006")
    assert not result.passed
    assert result.error is None


@pytest.mark.parametrize("scenario_id", EXPECTED_IDS)
def test_every_scenario_fails_when_the_ledger_ignores_limits(
    monkeypatch: pytest.MonkeyPatch, scenario_id: str
) -> None:
    monkeypatch.setattr(SpendLedger, "reserve", _ignore_limits)
    result = _run(scenario_id)
    assert not result.passed
    assert result.error is None
