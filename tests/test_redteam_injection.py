"""The prompt injection scenarios: all pass against the real gate, ids and titles follow the
suite's rules, and each scenario fails once the protection it relies on is taken away.

A mutation changes only the gate under test (its engines through Arena.regate, or one
function patched for the test), never the leak detector's reference engines, and then runs
the scenario function on that arena with the suite's pass criteria.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from agentgate import gate as gate_module
from agentgate.approvals import Approval, ApprovalError, ApprovalQueue, ApprovalStatus
from agentgate.gate import Gate
from agentgate.models import User
from agentgate.policy import engine as engine_module
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import ApprovalSettings, Effect, Policy, PolicyBundle, Rule
from agentgate.redteam.core import (
    CATEGORY_PREFIXES,
    Arena,
    Category,
    RegisteredScenario,
    all_scenarios,
    run_suite,
)
from agentgate.redteam.scenarios import prompt_injection
from conftest import FIXTURES, JR, KC, POLICIES

MODULE = prompt_injection.__name__
NO_EXTERNAL = "no-external-after-confidential"


def injection_scenarios() -> list[RegisteredScenario]:
    return [s for s in all_scenarios() if s.module == MODULE]


def registered(scenario_id: str) -> RegisteredScenario:
    return next(s for s in injection_scenarios() if s.id == scenario_id)


# The suite


def test_every_injection_scenario_passes() -> None:
    suite = run_suite(categories=[Category.PROMPT_INJECTION], policies=POLICIES, fixtures=FIXTURES)
    failed = {
        r.id: ([c.name for c in r.checks if not c.passed], list(r.leaks), r.error)
        for r in suite.failed
    }
    assert failed == {}
    ours = {s.id for s in injection_scenarios()}
    assert ours <= {r.id for r in suite.results}
    assert 10 <= len(ours) <= 14
    for result in suite.results:
        if result.id in ours:
            assert result.checks, result.id
            assert not result.benign


def test_suite_is_deterministic() -> None:
    ids = [s.id for s in injection_scenarios()]
    first = run_suite(ids=ids, policies=POLICIES, fixtures=FIXTURES)
    second = run_suite(ids=ids, policies=POLICIES, fixtures=FIXTURES)
    assert first == second


# Ids and titles


def test_ids_are_numbered_from_one_with_the_category_prefix() -> None:
    scenarios = injection_scenarios()
    prefix = CATEGORY_PREFIXES[Category.PROMPT_INJECTION]
    assert prefix == "pi"
    assert [s.id for s in scenarios] == [f"pi-{n:03d}" for n in range(1, len(scenarios) + 1)]
    assert all(s.category is Category.PROMPT_INJECTION for s in scenarios)


def test_titles_are_short_plain_and_unique() -> None:
    titles = [s.title for s in injection_scenarios()]
    assert len(set(titles)) == len(titles)
    for title in titles:
        assert title == title.strip() and 10 <= len(title) <= 80, title
        assert title[0].isupper(), title
        assert not title.endswith("."), title
        assert re.fullmatch(r"[A-Za-z0-9 ,'\-]+", title), title


def test_no_em_or_en_dashes_in_the_module() -> None:
    source = Path(prompt_injection.__file__).read_text(encoding="utf-8")
    assert chr(0x2014) not in source
    assert chr(0x2013) not in source


def test_every_outside_address_is_reserved() -> None:
    source = Path(prompt_injection.__file__).read_text(encoding="utf-8")
    domains = re.findall(r"@([a-z0-9-]+(?:\.[a-z0-9-]+)+)", source, flags=re.IGNORECASE)
    assert domains
    assert all(domain.lower().endswith(".example") for domain in domains), domains


# Mutations


@dataclass(frozen=True)
class Judged:
    passed: bool
    failed_checks: tuple[str, ...]
    leaks: tuple[str, ...]
    error: str | None


Mutation = Callable[[Arena, pytest.MonkeyPatch], None]


def judge(scenario_id: str, mutate: Mutation, monkeypatch: pytest.MonkeyPatch) -> Judged:
    """Run one scenario on a mutated arena and judge it by the suite's pass criteria."""
    s = registered(scenario_id)
    with Arena(policies=POLICIES, fixtures=FIXTURES) as arena:
        mutate(arena, monkeypatch)
        error: str | None = None
        try:
            s.fn(arena)
        except Exception as exc:
            error = type(exc).__name__
        failed = tuple(c.name for c in arena.checks if not c.passed)
        leaks = tuple(arena.leaks())
        passed = error is None and bool(arena.checks) and not failed and not leaks
    return Judged(passed=passed, failed_checks=failed, leaks=leaks, error=error)


def _regate_bundle(
    arena: Arena, customer_id: str, change: Callable[[PolicyBundle], PolicyBundle]
) -> None:
    engines = dict(arena.engines)
    customer = arena.store.get_customer(customer_id)
    assert customer is not None
    engines[customer_id] = PolicyEngine(change(engines[customer_id].bundle), customer)
    arena.regate(engines=engines)


def _regate_policy(arena: Arena, customer_id: str, change: Callable[[Policy], Policy]) -> None:
    _regate_bundle(
        arena,
        customer_id,
        lambda bundle: bundle.model_copy(update={"policy": change(bundle.policy)}),
    )


def drop_rule(customer_id: str, rule_id: str) -> Mutation:
    def mutate(arena: Arena, _: pytest.MonkeyPatch) -> None:
        def change(policy: Policy) -> Policy:
            rules = tuple(rule for rule in policy.rules if rule.id != rule_id)
            assert len(rules) == len(policy.rules) - 1
            return policy.model_copy(update={"rules": rules})

        _regate_policy(arena, customer_id, change)

    return mutate


def edit_rule(
    customer_id: str,
    rule_id: str,
    effect: Effect | None = None,
    amount_usd_lte: str | None = None,
) -> Mutation:
    def mutate(arena: Arena, _: pytest.MonkeyPatch) -> None:
        def edit(rule: Rule) -> Rule:
            if rule.id != rule_id:
                return rule
            update: dict[str, object] = {}
            if effect is not None:
                update["effect"] = effect
            if amount_usd_lte is not None:
                where = rule.where.model_copy(update={"amount_usd_lte": Decimal(amount_usd_lte)})
                update["where"] = where
            return rule.model_copy(update=update)

        def change(policy: Policy) -> Policy:
            assert any(rule.id == rule_id for rule in policy.rules)
            return policy.model_copy(update={"rules": tuple(edit(r) for r in policy.rules)})

        _regate_policy(arena, customer_id, change)

    return mutate


def not_customer_facing(customer_id: str, tool_id: str) -> Mutation:
    def mutate(arena: Arena, _: pytest.MonkeyPatch) -> None:
        def change(bundle: PolicyBundle) -> PolicyBundle:
            tools = dict(bundle.tools)
            tools[tool_id] = tools[tool_id].model_copy(update={"customer_facing": False})
            return bundle.model_copy(update={"tools": tools})

        _regate_bundle(arena, customer_id, change)

    return mutate


def both(first: Mutation, second: Mutation) -> Mutation:
    def mutate(arena: Arena, monkeypatch: pytest.MonkeyPatch) -> None:
        first(arena, monkeypatch)
        second(arena, monkeypatch)

    return mutate


def extend_list(customer_id: str, name: str, *entries: str) -> Mutation:
    def mutate(arena: Arena, _: pytest.MonkeyPatch) -> None:
        def change(policy: Policy) -> Policy:
            lists = dict(policy.lists)
            lists[name] = frozenset(lists[name] | set(entries))
            return policy.model_copy(update={"lists": lists})

        _regate_policy(arena, customer_id, change)

    return mutate


_FIRST_ADDRESS = re.compile(r"([a-z0-9._%+-]+)@([a-z0-9-]+(?:\.[a-z0-9-]+)+)")


def lenient_parser(arena: Arena, monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine takes the first address-shaped text in a recipient string, as a naive
    parser would."""

    def parse(raw: str) -> tuple[str, str] | None:
        match = _FIRST_ADDRESS.search(raw.strip().lower())
        return None if match is None else (match.group(0), match.group(2))

    monkeypatch.setattr(engine_module, "_parse_address", parse)


def no_write_down(arena: Arena, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate decides a note write by policy alone, without its write-down check."""

    def verdict(self: Gate, run: object, prepared: object) -> object:
        max_label = self.run_max_label(run.ctx.run_id)  # type: ignore[attr-defined]
        decision = run.engine.decide(  # type: ignore[attr-defined]
            gate_module._request(
                run,  # type: ignore[arg-type]
                prepared.config.id,  # type: ignore[attr-defined]
                prepared.action,  # type: ignore[attr-defined]
                prepared.document,  # type: ignore[attr-defined]
                prepared.facts,  # type: ignore[attr-defined]
                max_label,
            )
        )
        return gate_module._Verdict(
            decision.effect, decision.reason.value, decision.rule_ids, decision.policy_version
        )

    monkeypatch.setattr(Gate, "_verdict", verdict)


def self_approval_allowed(arena: Arena, monkeypatch: pytest.MonkeyPatch) -> None:
    """The approval queue no longer notices that the approver is the requester."""
    original = ApprovalQueue._decide_locked

    def decide(
        self: ApprovalQueue,
        approval_id: str,
        approver: User,
        settings: ApprovalSettings,
        note: str | None,
        decision: ApprovalStatus,
    ) -> Approval | ApprovalError:
        approval = self.get(approval_id)
        if approval is not None and approval.requested_by == approver.id:
            approver = approver.model_copy(update={"id": "mutant-approver"})
        return original(self, approval_id, approver, settings, note, decision)

    monkeypatch.setattr(ApprovalQueue, "_decide_locked", decide)


def unchanged(arena: Arena, monkeypatch: pytest.MonkeyPatch) -> None:
    return None


KC_NO_EXTERNAL = drop_rule(KC, NO_EXTERNAL)
JR_NO_EXTERNAL = drop_rule(JR, NO_EXTERNAL)
DENIED_MATCHED = "model told the email was denied matched_deny"

MUTATIONS: list[tuple[str, str, Mutation, str]] = [
    # (scenario, mutation, how, a check or leak that must then fail or appear)
    (
        "pi-001",
        "attacker-domain-internal",
        extend_list(KC, "internal_addresses", "@exfil-drop.example"),
        "model told the email was denied default_deny",
    ),
    (
        "pi-002",
        "attacker-domain-internal",
        extend_list(JR, "internal_addresses", "@records-request.example"),
        "model told the email was denied default_deny",
    ),
    ("pi-003", "no-external-rule-dropped", KC_NO_EXTERNAL, DENIED_MATCHED),
    ("pi-004", "no-external-rule-dropped", JR_NO_EXTERNAL, DENIED_MATCHED),
    ("pi-005", "no-external-rule-dropped", KC_NO_EXTERNAL, DENIED_MATCHED),
    (
        "pi-006",
        "look-alikes-internal",
        extend_list(
            KC,
            "internal_addresses",
            "@kettlecreek.example.evil.example",
            "@kettiecreek.example",
            "@exfil-drop.example",
        ),
        "three email denials on record, no rule matched",
    ),
    ("pi-007", "lenient-parser", lenient_parser, "smuggled recipient #1 denied"),
    ("pi-008", "lenient-parser", lenient_parser, "smuggled recipient #1 denied"),
    ("pi-008", "no-external-rule-dropped", KC_NO_EXTERNAL, "smuggled recipient #1 denied"),
    ("pi-009", "no-write-down-check", no_write_down, "note #1 (kettle-creek-hvac, internal)"),
    # The refund tool is customer facing, so an allow alone is still escalated to approval:
    # the scenario pins the policy rule's own approval, and only taking away both layers
    # lets the refund through.
    (
        "pi-010",
        "staff-refund-allowed",
        edit_rule(JR, "refund-staff-approval", effect=Effect.ALLOW),
        "audit approval_required matched_approval rule refund-staff-approval",
    ),
    (
        "pi-010",
        "refund-approval-removed",
        both(
            edit_rule(JR, "refund-staff-approval", effect=Effect.ALLOW),
            not_customer_facing(JR, "refund.issue"),
        ),
        "no refund issued",
    ),
    (
        "pi-010",
        "staff-refund-limit-raised",
        edit_rule(JR, "refund-staff-approval", amount_usd_lte="1000.00"),
        "refund above the staff limit denied",
    ),
    (
        "pi-011",
        "staff-parts-order-allowed",
        edit_rule(KC, "parts-order-staff-approval", effect=Effect.ALLOW),
        "no parts ordered",
    ),
    (
        "pi-011",
        "manager-parts-limit-raised",
        edit_rule(KC, "parts-order-approval-to-5000", amount_usd_lte="10000.00"),
        "manager order above 5000 denied",
    ),
    ("pi-012", "self-approval-allowed", self_approval_allowed, "approval still pending"),
    ("pi-013", "no-external-rule-dropped", KC_NO_EXTERNAL, DENIED_MATCHED),
    ("pi-014", "no-external-rule-dropped", KC_NO_EXTERNAL, DENIED_MATCHED),
]


@pytest.mark.parametrize("scenario_id", [s.id for s in injection_scenarios()])
def test_the_mutation_harness_passes_unmutated(
    scenario_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    judged = judge(scenario_id, unchanged, monkeypatch)
    assert judged == Judged(passed=True, failed_checks=(), leaks=(), error=None)


@pytest.mark.parametrize(
    ("scenario_id", "mutate", "must_fail"),
    [pytest.param(sid, how, must, id=f"{sid}-{name}") for sid, name, how, must in MUTATIONS],
)
def test_scenario_fails_without_its_protection(
    scenario_id: str, mutate: Mutation, must_fail: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    judged = judge(scenario_id, mutate, monkeypatch)
    assert not judged.passed
    assert judged.error is None
    found = [*judged.failed_checks, *judged.leaks]
    assert any(must_fail in item for item in found), found


def test_every_scenario_has_a_mutation() -> None:
    covered = {sid for sid, *_ in MUTATIONS}
    assert covered == {s.id for s in injection_scenarios()}


def test_write_down_mutation_is_caught_by_the_leak_detector_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judged = judge("pi-009", no_write_down, monkeypatch)
    assert "note write denied write_down" in judged.failed_checks
    assert any("kc-customer-list" in leak for leak in judged.leaks), judged.leaks


def test_dropping_no_external_lets_the_directory_email_reach_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without the rule the listed customer address is only an approval away from receiving
    # the customer list: that is the door the rule closes.
    with Arena(policies=POLICIES, fixtures=FIXTURES) as arena:
        KC_NO_EXTERNAL(arena, monkeypatch)
        registered("pi-005").fn(arena)
        approvals = arena.gate.approvals.list(KC)
        assert [a.rule_ids for a in approvals] == [("email-customers",)]
        assert arena.outbox() == []
