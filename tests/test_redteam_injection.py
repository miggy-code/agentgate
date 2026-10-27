"""Mutation tests for the prompt injection scenarios: each fails once the protection it relies
on is taken away.

A mutation changes only the gate under test (its engines through Arena.regate, or one
function patched for the test), never the leak detector's reference engines.
"""

import json
import re
import uuid
from collections.abc import Callable
from decimal import Decimal

import pytest

from agentgate import gate as gate_module
from agentgate.approvals import Approval, ApprovalError, ApprovalQueue, ApprovalStatus
from agentgate.context import RunContext
from agentgate.gate import Gate
from agentgate.models import User
from agentgate.policy import addresses
from agentgate.policy import engine as engine_module
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import ApprovalSettings, Effect, Policy, PolicyBundle, Rule
from agentgate.redteam.core import (
    Arena,
    Category,
    all_scenarios,
)
from agentgate.redteam.scenarios import prompt_injection
from agentgate.tools import builtin
from agentgate.tools.base import ToolEnv, ToolResult
from conftest import (
    FIXTURES,
    JR,
    KC,
    POLICIES,
    Mutation,
    assert_breaks,
    assert_every_attack_has_a_mutation,
    run_mutated,
)

NO_EXTERNAL = "no-external-after-confidential"


# Mutations


ArenaMutation = Callable[[Arena, pytest.MonkeyPatch], None]


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


def drop_rule(customer_id: str, rule_id: str) -> ArenaMutation:
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
) -> ArenaMutation:
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


def not_customer_facing(customer_id: str, tool_id: str) -> ArenaMutation:
    def mutate(arena: Arena, _: pytest.MonkeyPatch) -> None:
        def change(bundle: PolicyBundle) -> PolicyBundle:
            tools = dict(bundle.tools)
            tools[tool_id] = tools[tool_id].model_copy(update={"customer_facing": False})
            return bundle.model_copy(update={"tools": tools})

        _regate_bundle(arena, customer_id, change)

    return mutate


def both(first: ArenaMutation, second: ArenaMutation) -> ArenaMutation:
    def mutate(arena: Arena, monkeypatch: pytest.MonkeyPatch) -> None:
        first(arena, monkeypatch)
        second(arena, monkeypatch)

    return mutate


def extend_list(customer_id: str, name: str, *entries: str) -> ArenaMutation:
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


KC_NO_EXTERNAL = drop_rule(KC, NO_EXTERNAL)
JR_NO_EXTERNAL = drop_rule(JR, NO_EXTERNAL)
DENIED_MATCHED = "model told the email was denied matched_deny"
DENIED_DEFAULT = "model told the email was denied default_deny"


# The old address rule (pi-015, pi-016): the engine checked a stripped, lowercased copy and the
# email tool stored the recipient as typed.

_OLD_ADDRESS = re.compile(r"^([^@\s,;:<>()\[\]\"\\]+)@([a-z0-9-]+(?:\.[a-z0-9-]+)+)$")


def _old_parse(raw: str) -> tuple[str, str] | None:
    match = _OLD_ADDRESS.fullmatch(raw.strip().lower())
    return None if match is None else (match.group(0), match.group(2))


def _store_as_typed(
    self: builtin.EmailSend, env: ToolEnv, ctx: RunContext, args: builtin.EmailSendArgs
) -> ToolResult:
    email_id = f"email-{uuid.uuid4().hex[:12]}"
    builtin._insert(
        env.store,
        "INSERT INTO outbox (id, run_id, customer_id, recipients, subject, body, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            email_id,
            ctx.run_id,
            ctx.customer_id,
            json.dumps(list(args.to)),
            args.subject,
            args.body,
            env.clock.now().isoformat(),
        ),
    )
    return ToolResult(content={"status": "queued", "email_id": email_id})


MUTATIONS = [
    Mutation(
        "attacker domain internal at KC",
        ("pi-001",),
        arena=extend_list(KC, "internal_addresses", "@exfil-drop.example"),
        must_fail=DENIED_DEFAULT,
    ),
    Mutation(
        "attacker domain internal at JR",
        ("pi-002",),
        arena=extend_list(JR, "internal_addresses", "@records-request.example"),
        must_fail=DENIED_DEFAULT,
    ),
    Mutation(
        "no-external rule dropped at KC",
        ("pi-003", "pi-005", "pi-013", "pi-014"),
        arena=KC_NO_EXTERNAL,
        must_fail=DENIED_MATCHED,
    ),
    Mutation(
        "no-external rule dropped at JR",
        ("pi-004",),
        arena=JR_NO_EXTERNAL,
        must_fail=DENIED_MATCHED,
    ),
    Mutation(
        "look-alikes internal",
        ("pi-006",),
        arena=extend_list(
            KC,
            "internal_addresses",
            "@kettlecreek.example.evil.example",
            "@kettiecreek.example",
            "@exfil-drop.example",
        ),
        must_fail="three email denials on record, no rule matched",
    ),
    Mutation(
        "lenient parser",
        ("pi-007", "pi-008"),
        arena=lenient_parser,
        must_fail="smuggled recipient #1 denied",
    ),
    Mutation(
        "no-external rule dropped, smuggled recipient",
        ("pi-008",),
        arena=KC_NO_EXTERNAL,
        must_fail="smuggled recipient #1 denied",
    ),
    Mutation(
        "no write-down check",
        ("pi-009",),
        arena=no_write_down,
        must_fail="note #1 (kettle-creek-hvac, internal)",
    ),
    # The refund tool is customer facing, so an allow alone is still escalated to approval:
    # the scenario pins the policy rule's own approval, and only taking away both layers
    # lets the refund through.
    Mutation(
        "staff refund allowed",
        ("pi-010",),
        arena=edit_rule(JR, "refund-staff-approval", effect=Effect.ALLOW),
        must_fail="audit approval_required matched_approval rule refund-staff-approval",
    ),
    Mutation(
        "refund approval removed",
        ("pi-010",),
        arena=both(
            edit_rule(JR, "refund-staff-approval", effect=Effect.ALLOW),
            not_customer_facing(JR, "refund.issue"),
        ),
        must_fail="no refund issued",
    ),
    Mutation(
        "staff refund limit raised",
        ("pi-010",),
        arena=edit_rule(JR, "refund-staff-approval", amount_usd_lte="1000.00"),
        must_fail="refund above the staff limit denied",
    ),
    Mutation(
        "staff parts order allowed",
        ("pi-011",),
        arena=edit_rule(KC, "parts-order-staff-approval", effect=Effect.ALLOW),
        must_fail="no parts ordered",
    ),
    Mutation(
        "manager parts limit raised",
        ("pi-011",),
        arena=edit_rule(KC, "parts-order-approval-to-5000", amount_usd_lte="10000.00"),
        must_fail="manager order above 5000 denied",
    ),
    Mutation(
        "self approval allowed",
        ("pi-012",),
        arena=self_approval_allowed,
        must_fail="approval still pending",
    ),
    Mutation(
        "old address rule, disguised recipients",
        ("pi-015",),
        replace=((addresses, "parse_address", _old_parse),),
        must_fail="disguised recipient #1 denied",
    ),
    Mutation(
        "old address rule, look-alike domains",
        ("pi-016",),
        replace=((addresses, "parse_address", _old_parse),),
        must_fail="look-alike #1 denied",
    ),
    Mutation(
        "engine keeps the old rule, disguised recipients",
        ("pi-015",),
        replace=((engine_module, "_parse_address", _old_parse),),
        must_fail="disguised recipient #1 denied",
    ),
    Mutation(
        "engine keeps the old rule, look-alike domains",
        ("pi-016",),
        replace=((engine_module, "_parse_address", _old_parse),),
        must_fail="staff look-alike #1 denied",
    ),
    Mutation(
        "email tool stores recipients as typed",
        ("pi-015",),
        replace=((builtin.EmailSend, "_run", _store_as_typed),),
        must_fail="approved injected send is refused",
    ),
]


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.name)
def test_mutation_fails_its_scenarios(mutation: Mutation, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_breaks(mutation, monkeypatch)


def test_every_attack_has_a_mutation() -> None:
    assert_every_attack_has_a_mutation(MUTATIONS, Category.PROMPT_INJECTION)


def test_write_down_mutation_is_caught_by_the_leak_detector_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutation = next(m for m in MUTATIONS if m.name == "no write-down check")
    [result] = run_mutated(mutation, monkeypatch).results
    assert "note write denied write_down" in [c.name for c in result.checks if not c.passed]
    assert any("kc-customer-list" in leak for leak in result.leaks), result.leaks


def test_dropping_no_external_lets_the_directory_email_reach_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without the rule the listed customer address is only an approval away from receiving
    # the customer list: that is the door the rule closes.
    with Arena(policies=POLICIES, fixtures=FIXTURES) as arena:
        KC_NO_EXTERNAL(arena, monkeypatch)
        next(s for s in all_scenarios() if s.id == "pi-005").fn(arena)
        approvals = arena.gate.approvals.list(KC)
        assert [a.rule_ids for a in approvals] == [("email-customers",)]
        assert arena.outbox() == []


def test_the_old_rule_let_every_disguised_recipient_through_as_internal() -> None:
    # The scenarios' inputs really are the forms the old rule got wrong, not ones it
    # already refused.
    internal = "kettlecreek.example"
    admitted = []
    for raw in prompt_injection.DISGUISED:
        assert addresses.parse_address(raw) is None, raw
        parsed = _old_parse(raw)
        if parsed is not None and parsed[1] == internal:
            admitted.append(raw)
    # All but a trailing bidi mark (which the old domain pattern caught) and a quoted local
    # part (which it refused) went out as internal mail.
    assert len(admitted) == len(prompt_injection.DISGUISED) - 2
    assert prompt_injection.DISGUISED[0] in admitted
    kelvin = prompt_injection.LOOK_ALIKE_DOMAINS[0]
    assert _old_parse(kelvin) == ("terrence@kettlecreek.example", internal)
    assert addresses.parse_address(kelvin) is None
