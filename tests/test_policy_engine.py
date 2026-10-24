from decimal import Decimal
from typing import Any

import pytest

from agentgate.models import Action, Customer, Document, Role
from agentgate.policy import addresses
from agentgate.policy import engine as engine_module
from agentgate.policy.engine import Decision, DocumentFacts, PolicyEngine, Reason, Request
from agentgate.policy.schema import Conditions, Effect, PolicyBundle
from agentgate.policy.version import bundle_version

CUSTOMER = Customer(
    id="acme", name="Acme", domains=("acme.example", "acme-mail.example"), timezone="UTC"
)
LABELS = ["public", "internal", "confidential", "restricted"]
TOOLS: dict[str, dict[str, Any]] = {
    "docs.read": {"actions": ["read"], "base_risk": 0},
    "docs.write": {"actions": ["read", "write"], "base_risk": 30},
    "email.send": {"actions": ["send"], "base_risk": 40, "customer_facing": True},
    "pay.charge": {"actions": ["spend"], "base_risk": 70},
}
LISTS = {"directory": ["pat@home.example", "@partner.example"]}


def _bundle(*rules: dict[str, Any], lists: dict[str, list[str]] | None = None) -> PolicyBundle:
    tools = {
        tool_id: {"id": tool_id, "description": tool_id, "cost_usd": "0"} | spec
        for tool_id, spec in TOOLS.items()
    }
    return PolicyBundle.model_validate(
        {
            "policy": {
                "apiVersion": "agentgate/v1",
                "customer": "acme",
                "labels": LABELS,
                "default": "deny",
                "spend": {"per_run_usd": "1", "per_customer_day_usd": "20", "max_calls_per_run": 9},
                "approvals": {"approver_roles": ["owner"], "ttl_minutes": 60, "risk_threshold": 50},
                "lists": LISTS if lists is None else lists,
                "rules": list(rules),
            },
            "tools": tools,
        }
    )


def _engine(*rules: dict[str, Any]) -> PolicyEngine:
    return PolicyEngine(_bundle(*rules), CUSTOMER)


def _rule(rule_id: str, effect: str = "allow", **fields: Any) -> dict[str, Any]:
    rule: dict[str, Any] = {
        "id": rule_id,
        "effect": effect,
        "roles": ["staff"],
        "tools": ["docs.read"],
        "actions": ["read"],
    }
    return rule | fields


def _send_rule(rule_id: str, effect: str = "allow", **fields: Any) -> dict[str, Any]:
    return _rule(rule_id, effect, tools=["email.send"], actions=["send"], **fields)


def _spend_rule(rule_id: str, effect: str = "allow", **fields: Any) -> dict[str, Any]:
    return _rule(rule_id, effect, tools=["pay.charge"], actions=["spend"], **fields)


def _doc(label: str = "internal", **fields: Any) -> DocumentFacts:
    return DocumentFacts(customer_id=fields.pop("customer_id", "acme"), label=label, **fields)


def _read(**fields: Any) -> Request:
    defaults: dict[str, Any] = {
        "customer_id": "acme",
        "user_id": "dana",
        "role": Role.STAFF,
        "tool": "docs.read",
        "action": Action.READ,
        "document": _doc(),
    }
    return Request(**(defaults | fields))


def _send(*recipients: str, **fields: Any) -> Request:
    return _read(
        tool="email.send", action=Action.SEND, document=None, recipients=recipients, **fields
    )


def _spend(amount: Decimal | None, **fields: Any) -> Request:
    return _read(tool="pay.charge", action=Action.SPEND, document=None, amount_usd=amount, **fields)


def _effect(engine: PolicyEngine, request: Request) -> Effect:
    return engine.decide(request).effect


# Built-in checks


def test_engine_rejects_a_customer_that_does_not_match_the_bundle() -> None:
    other = CUSTOMER.model_copy(update={"id": "other"})
    with pytest.raises(ValueError, match="does not match policy customer"):
        PolicyEngine(_bundle(_rule("r")), other)


# Allows every tool and action the bundle supports, so any deny below can only come from a
# built-in check and its reason shows which one fired.
OPEN = [
    _rule("r", tools=["docs.read"], actions=["read"]),
    _rule("w", tools=["docs.write"], actions=["read", "write"]),
    _send_rule("s"),
    _spend_rule("p"),
]


@pytest.mark.parametrize(
    ("request_", "reason"),
    [
        (_read(customer_id="other"), Reason.CUSTOMER_MISMATCH),
        # Customer mismatch is reported before anything else is wrong with the request.
        (_read(customer_id="other", tool="sms.send"), Reason.CUSTOMER_MISMATCH),
        (_read(tool="sms.send"), Reason.UNKNOWN_TOOL),
        (_read(tool="sms.send", document=None), Reason.UNKNOWN_TOOL),
        (_read(action=Action.WRITE), Reason.ACTION_NOT_SUPPORTED),
        (_read(action=Action.SEND), Reason.ACTION_NOT_SUPPORTED),
        (_read(document=None), Reason.DOCUMENT_REQUIRED),
        (_read(tool="docs.write", action=Action.WRITE, document=None), Reason.DOCUMENT_REQUIRED),
        (_read(document=_doc(customer_id="beta")), Reason.DOCUMENT_CUSTOMER_MISMATCH),
        (_read(document=_doc("secret", customer_id="beta")), Reason.DOCUMENT_CUSTOMER_MISMATCH),
        (_read(document=_doc("secret")), Reason.UNKNOWN_LABEL),
        (_read(run_max_label="secret"), Reason.UNKNOWN_LABEL),
        (_send("a@acme.example", run_max_label="secret"), Reason.UNKNOWN_LABEL),
    ],
)
def test_builtin_checks_deny_with_empty_rule_ids(request_: Request, reason: Reason) -> None:
    engine = _engine(*OPEN)
    decision = engine.decide(request_)
    assert decision == Decision(Effect.DENY, reason, (), engine.version)


def test_requests_that_pass_builtin_checks_reach_the_rules() -> None:
    engine = _engine(*OPEN)
    assert _effect(engine, _read()) is Effect.ALLOW
    assert _effect(engine, _read(tool="docs.write", action=Action.WRITE)) is Effect.ALLOW
    # Send and spend need no document.
    assert _effect(engine, _send("a@acme.example")) is Effect.ALLOW
    assert _effect(engine, _spend(Decimal("1"))) is Effect.ALLOW


# Role, tool, action and labels


def test_role_must_be_listed() -> None:
    engine = _engine(_rule("r", roles=["staff", "manager"]))
    assert _effect(engine, _read(role=Role.STAFF)) is Effect.ALLOW
    assert _effect(engine, _read(role=Role.MANAGER)) is Effect.ALLOW
    decision = engine.decide(_read(role=Role.CONTRACTOR))
    assert (decision.effect, decision.reason) == (Effect.DENY, Reason.DEFAULT_DENY)


@pytest.mark.parametrize("role", list(Role))
def test_deny_rule_without_roles_applies_to_every_role(role: Role) -> None:
    allow_all = _rule("allow", roles=[r.value for r in Role])
    engine = _engine(
        allow_all, {"id": "no", "effect": "deny", "tools": ["docs.read"], "actions": ["read"]}
    )
    assert engine.decide(_read(role=role)).rule_ids == ("no",)


def test_tool_and_action_must_both_be_listed() -> None:
    engine = _engine(_rule("r", tools=["docs.write"], actions=["read"]))
    assert _effect(engine, _read(tool="docs.write")) is Effect.ALLOW
    assert _effect(engine, _read(tool="docs.read")) is Effect.DENY
    assert _effect(engine, _read(tool="docs.write", action=Action.WRITE)) is Effect.DENY


@pytest.mark.parametrize(
    ("label", "effect"),
    [
        ("public", Effect.ALLOW),
        ("internal", Effect.ALLOW),
        ("confidential", Effect.ALLOW),
        ("restricted", Effect.DENY),
    ],
)
def test_label_bound_is_inclusive(label: str, effect: Effect) -> None:
    engine = _engine(_rule("r", labels={"max": "confidential"}))
    assert _effect(engine, _read(document=_doc(label))) is effect


def test_label_bound_at_the_lowest_rank() -> None:
    engine = _engine(_rule("r", labels={"max": "public"}))
    assert _effect(engine, _read(document=_doc("public"))) is Effect.ALLOW
    assert _effect(engine, _read(document=_doc("internal"))) is Effect.DENY


def test_rule_without_label_bound_matches_any_known_label() -> None:
    engine = _engine(_rule("r"))
    assert _effect(engine, _read(document=_doc("restricted"))) is Effect.ALLOW


# recipient_in


@pytest.mark.parametrize(
    ("recipients", "holds"),
    [
        (("pat@home.example",), True),
        ((" PAT@Home.Example ",), True),
        (("anyone@partner.example",), True),
        (("Anyone@PARTNER.example",), True),
        (("pat@home.example", "lee@partner.example"), True),
        (("other@home.example",), False),
        (("pat@sub.home.example",), False),
        (("x@sub.partner.example",), False),
        (("x@partner.example.evil.example",), False),
        (("x@notpartner.example",), False),
        (("pat@home.example", "x@evil.example"), False),
        (("a@partner.example,c@evil.example",), False),
        (("a@partner.example;c@evil.example",), False),
        (("a@partner.example@evil.example",), False),
        (("a b@partner.example",), False),
        (("a@partner .example",), False),
        (("Pat <pat@home.example>",), False),
        (("@partner.example",), False),
        (("pat@",), False),
        (("partner.example",), False),
        (("",), False),
        ((), False),
    ],
)
def test_recipient_in(recipients: tuple[str, ...], holds: bool) -> None:
    engine = _engine(_send_rule("r", where={"recipient_in": "directory"}))
    assert (_effect(engine, _send(*recipients)) is Effect.ALLOW) is holds


def test_recipient_in_with_an_empty_list_never_holds() -> None:
    bundle = _bundle(_send_rule("r", where={"recipient_in": "nobody"}), lists={"nobody": []})
    engine = PolicyEngine(bundle, CUSTOMER)
    assert _effect(engine, _send("pat@home.example")) is Effect.DENY


# recipient_external


@pytest.mark.parametrize(
    ("recipients", "external_true", "external_false"),
    [
        (("a@acme.example",), False, True),
        ((" A@ACME.Example ",), False, True),
        (("a@acme.example", "b@acme-mail.example"), False, True),
        (("a@sub.acme.example",), True, False),
        (("a@acme.example.evil.example",), True, False),
        (("a@acme.example", "b@other.example"), True, False),
        (("b@other.example", "a@acme.example"), True, False),
        (("a@acme.example,b@other.example",), True, False),
        (("a@acme.example@other.example",), True, False),
        (("a@acme.example b@acme.example",), True, False),
        (("not-an-address",), True, False),
        (("@acme.example",), True, False),
        (("",), True, False),
        ((), False, False),
    ],
)
def test_recipient_external(
    recipients: tuple[str, ...], external_true: bool, external_false: bool
) -> None:
    for value, expected in ((True, external_true), (False, external_false)):
        engine = _engine(_send_rule("r", where={"recipient_external": value}))
        assert (_effect(engine, _send(*recipients)) is Effect.ALLOW) is expected, value


def test_malformed_recipient_never_matches_a_list_and_counts_as_external() -> None:
    engine = _engine(
        _send_rule("listed", where={"recipient_in": "directory"}),
        _send_rule("internal", where={"recipient_external": False}),
        {
            "id": "external",
            "effect": "deny",
            "tools": ["email.send"],
            "actions": ["send"],
            "where": {"recipient_external": True},
        },
    )
    decision = engine.decide(_send("pat@home.example,a@acme.example"))
    assert (decision.effect, decision.rule_ids) == (Effect.DENY, ("external",))


# The shared address rule (agentgate.policy.addresses)

ACCEPTED_ADDRESSES = [
    ("a@acme.example", "a@acme.example"),
    ("A@ACME.Example", "a@acme.example"),
    ("  Pat.Lee@Acme.Example  ", "pat.lee@acme.example"),
    ("first.last+tag@acme-mail.example", "first.last+tag@acme-mail.example"),
    ("o'brien_x-y@sub.home.example", "o'brien_x-y@sub.home.example"),
    ("x" * 64 + "@acme.example", "x" * 64 + "@acme.example"),
]
# Each of these is an internal or listed address with something added to it.
REFUSED_ADDRESSES = [
    " \ta@acme.example\r\n",
    "a@acme.example\r\n",
    "a@acme.example\n",
    "\ta@acme.example",
    "a@acme.example\x00",
    "a@acme.example\nBcc: x@evil.example",
    "\u202ea@acme.example",
    "a@acme.example\u200f",
    "p\u200bat@home.example",
    "\ufeffa@acme.example",
    "a\u2066@acme.example",
    "x%evil.example@acme.example",
    "evil.example!x@acme.example",
    '"x@evil.example"@acme.example',
    '"pat"@home.example',
    "pat\\@home.example",
    "a@\u212acme.example",
    "a@\u212ccme.example",
    "\u212aa@acme.example",
    "\u017fam@acme.example",
    "a@\u0430cme.example",
    "a@\uff41cme.example",
    "a@acme.exampl\u00e9",
    "a@acme.example\u00a0",
    "\u3000a@acme.example",
    "p at@home.example",
    ".pat@home.example",
    "pat.@home.example",
    "p..at@home.example",
    "a@acme.example.",
    "a@[192.0.2.1]",
    "a@acme",
    "a@b@acme.example",
    "x" * 65 + "@acme.example",
    "a@" + "x" * 64 + ".example",
    "",
    "   ",
]


@pytest.mark.parametrize(("raw", "normalised"), ACCEPTED_ADDRESSES)
def test_address_rule_accepts_and_normalises_plain_addresses(raw: str, normalised: str) -> None:
    assert addresses.normalize_address(raw) == normalised
    assert addresses.parse_address(raw) == (normalised, normalised.rsplit("@", 1)[1])
    # The normalised form is a fixed point.
    assert addresses.normalize_address(normalised) == normalised


@pytest.mark.parametrize("raw", REFUSED_ADDRESSES)
def test_address_rule_refuses_hostile_forms(raw: str) -> None:
    assert addresses.parse_address(raw) is None
    assert addresses.normalize_address(raw) is None


def test_no_case_mapping_turns_a_refused_character_into_ascii() -> None:
    # The rule checks ASCII before lowercasing, so these must stay refused although their
    # lowercase or casefold is plain ASCII.
    for char in ("\u212a", "\u017f", "\u0130", "\ufb00"):
        folded = char.lower() + char.casefold()
        assert addresses.normalize_address(f"{char}a@acme.example") is None, folded


def test_address_rule_domains() -> None:
    assert addresses.normalize_domain("Acme.Example") == "acme.example"
    assert addresses.normalize_domain(" acme.example ") == "acme.example"
    for raw in ("\u212acme.example", "acme", "acme.example.", "acme .example", "a\tb.example"):
        assert addresses.normalize_domain(raw) is None, raw


@pytest.mark.parametrize("raw", REFUSED_ADDRESSES)
def test_a_hostile_form_is_external_and_on_no_list(raw: str) -> None:
    listed = _engine(_send_rule("r", where={"recipient_in": "directory"}))
    internal = _engine(_send_rule("r", where={"recipient_external": False}))
    external = _engine(_send_rule("r", where={"recipient_external": True}))
    assert _effect(listed, _send(raw)) is Effect.DENY
    assert _effect(internal, _send(raw)) is Effect.DENY
    assert _effect(external, _send(raw)) is Effect.ALLOW


@pytest.mark.parametrize(
    "raw",
    [
        "\u202epat@home.example",
        "pat@home.example\r\n",
        "pat%evil.example@home.example",
        "pat@h\u043eme.example",
        "x@\u0440artner.example",
        "x@partner.exampl\u212a",
    ],
)
def test_a_hostile_form_of_a_listed_address_is_not_listed(raw: str) -> None:
    engine = _engine(_send_rule("r", where={"recipient_in": "directory"}))
    assert _effect(engine, _send(raw)) is Effect.DENY
    assert _effect(engine, _send("pat@home.example")) is Effect.ALLOW


@pytest.mark.parametrize("raw", [raw for raw, _ in ACCEPTED_ADDRESSES if "acme" in raw])
def test_a_plain_internal_address_is_internal_in_any_case(raw: str) -> None:
    internal = _engine(_send_rule("r", where={"recipient_external": False}))
    assert _effect(internal, _send(raw)) is Effect.ALLOW


def test_the_engine_parses_with_the_shared_rule() -> None:
    for raw in [*REFUSED_ADDRESSES, *(raw for raw, _ in ACCEPTED_ADDRESSES)]:
        assert engine_module._parse_address(raw) == addresses.parse_address(raw), raw


@pytest.mark.parametrize(
    "entry", ["pat%evil.example@home.example", '"pat"@home.example', "p..at@home.example"]
)
def test_a_list_entry_the_rule_refuses_stops_the_policy_loading(entry: str) -> None:
    # The schema admits these; the engine refuses them rather than keep a list entry that
    # could never match (which on a deny list would fail open).
    bundle = _bundle(
        _send_rule("r", where={"recipient_in": "directory"}), lists={"directory": [entry]}
    )
    with pytest.raises(ValueError, match="list 'directory'"):
        PolicyEngine(bundle, CUSTOMER)


def test_list_entries_are_normalised_with_the_rule() -> None:
    bundle = _bundle(
        _send_rule("r", where={"recipient_in": "directory"}),
        lists={"directory": ["pat@home.example", "@partner.example"]},
    )
    engine = PolicyEngine(bundle, CUSTOMER)
    assert _effect(engine, _send(" PAT@HOME.EXAMPLE ")) is Effect.ALLOW
    assert _effect(engine, _send("Lee@Partner.Example")) is Effect.ALLOW


# amount_usd_lte


@pytest.mark.parametrize(
    ("amount", "holds"),
    [
        (Decimal("0"), True),
        (Decimal("9.99"), True),
        (Decimal("10"), True),
        (Decimal("10.000000"), True),
        (Decimal("10.000001"), False),
        (Decimal("11"), False),
        (Decimal("-0.01"), False),
        (Decimal("-100"), False),
        (Decimal("NaN"), False),
        (Decimal("sNaN"), False),
        (Decimal("Infinity"), False),
        (Decimal("-Infinity"), False),
        (None, False),
    ],
)
def test_amount_usd_lte(amount: Decimal | None, holds: bool) -> None:
    engine = _engine(_spend_rule("r", where={"amount_usd_lte": "10"}))
    assert (_effect(engine, _spend(amount)) is Effect.ALLOW) is holds


def test_negative_amount_does_not_satisfy_even_a_zero_limit() -> None:
    engine = _engine(_spend_rule("r", where={"amount_usd_lte": "0"}))
    assert _effect(engine, _spend(Decimal("0"))) is Effect.ALLOW
    assert _effect(engine, _spend(Decimal("-1"))) is Effect.DENY


# amount_usd_gt


@pytest.mark.parametrize(
    ("amount", "holds"),
    [
        (Decimal("500"), False),
        (Decimal("500.00"), False),
        (Decimal("500.000001"), True),
        (Decimal("500.01"), True),
        (Decimal("1000000"), True),
        (Decimal("0"), False),
        (Decimal("-1000"), False),
        (Decimal("NaN"), False),
        (Decimal("sNaN"), False),
        (Decimal("Infinity"), False),
        (None, False),
    ],
)
def test_amount_usd_gt(amount: Decimal | None, holds: bool) -> None:
    engine = _engine(_spend_rule("r", where={"amount_usd_gt": "500"}))
    assert (_effect(engine, _spend(amount)) is Effect.ALLOW) is holds


def test_amount_usd_gt_zero_excludes_zero() -> None:
    engine = _engine(_spend_rule("r", where={"amount_usd_gt": "0"}))
    assert _effect(engine, _spend(Decimal("0"))) is Effect.DENY
    assert _effect(engine, _spend(Decimal("0.000001"))) is Effect.ALLOW


# A band: small spends are allowed, 500 < amount <= 5000 needs approval, above is denied.
BAND = (
    _spend_rule("small", where={"amount_usd_lte": "500"}),
    _spend_rule(
        "band", "require_approval", where={"amount_usd_gt": "500", "amount_usd_lte": "5000"}
    ),
)


@pytest.mark.parametrize(
    ("amount", "effect", "rule_ids"),
    [
        (Decimal("0"), Effect.ALLOW, ("small",)),
        (Decimal("500.00"), Effect.ALLOW, ("small",)),
        (Decimal("500.01"), Effect.REQUIRE_APPROVAL, ("band",)),
        (Decimal("5000.00"), Effect.REQUIRE_APPROVAL, ("band",)),
        (Decimal("5000.01"), Effect.DENY, ()),
        (Decimal("-1"), Effect.DENY, ()),
        (None, Effect.DENY, ()),
    ],
)
def test_amount_band(amount: Decimal | None, effect: Effect, rule_ids: tuple[str, ...]) -> None:
    decision = _engine(*BAND).decide(_spend(amount))
    assert (decision.effect, decision.rule_ids) == (effect, rule_ids)


@pytest.mark.parametrize(
    ("amount", "in_band"),
    [
        (Decimal("500.00"), False),
        (Decimal("500.01"), True),
        (Decimal("5000.00"), True),
        (Decimal("5000.01"), False),
        (None, False),
    ],
)
def test_amount_band_on_its_own(amount: Decimal | None, in_band: bool) -> None:
    engine = _engine(BAND[1])
    assert (_effect(engine, _spend(amount)) is Effect.REQUIRE_APPROVAL) is in_band


# run_max_label_gte


@pytest.mark.parametrize(
    ("run_label", "holds"),
    [
        ("restricted", True),
        ("confidential", True),
        ("internal", False),
        ("public", False),
        (None, False),
    ],
)
def test_run_max_label_gte(run_label: str | None, holds: bool) -> None:
    engine = _engine(_send_rule("r", where={"run_max_label_gte": "confidential"}))
    request = _send("a@acme.example", run_max_label=run_label)
    assert (_effect(engine, request) is Effect.ALLOW) is holds


def test_run_max_label_gte_at_the_lowest_rank_needs_a_label() -> None:
    engine = _engine(_send_rule("r", where={"run_max_label_gte": "public"}))
    assert _effect(engine, _send("a@acme.example", run_max_label="public")) is Effect.ALLOW
    assert _effect(engine, _send("a@acme.example")) is Effect.DENY


# doc_tags_any


@pytest.mark.parametrize(
    ("tags", "holds"),
    [
        (frozenset({"billing"}), True),
        (frozenset({"vip", "other"}), True),
        (frozenset({"billing", "vip"}), True),
        (frozenset({"other"}), False),
        (frozenset(), False),
    ],
)
def test_doc_tags_any(tags: frozenset[str], holds: bool) -> None:
    engine = _engine(_rule("r", where={"doc_tags_any": ["billing", "vip"]}))
    assert (_effect(engine, _read(document=_doc(tags=tags))) is Effect.ALLOW) is holds


# doc_assigned_to_user


@pytest.mark.parametrize(
    ("value", "assigned", "holds"),
    [
        (True, frozenset({"dana"}), True),
        (True, frozenset({"dana", "lee"}), True),
        (True, frozenset({"lee"}), False),
        (True, frozenset(), False),
        (False, frozenset({"lee"}), True),
        (False, frozenset(), True),
        (False, frozenset({"dana"}), False),
    ],
)
def test_doc_assigned_to_user(value: bool, assigned: frozenset[str], holds: bool) -> None:
    engine = _engine(_rule("r", where={"doc_assigned_to_user": value}))
    request = _read(document=_doc(assigned_user_ids=assigned))
    assert (_effect(engine, request) is Effect.ALLOW) is holds


# Missing facts


@pytest.mark.parametrize(
    "where",
    [
        Conditions(recipient_in="directory"),
        Conditions(recipient_external=True),
        Conditions(recipient_external=False),
        Conditions(amount_usd_lte=Decimal("10")),
        Conditions(amount_usd_gt=Decimal("0")),
        Conditions(amount_usd_gt=Decimal("0"), amount_usd_lte=Decimal("10")),
        Conditions(run_max_label_gte="public"),
        Conditions(doc_tags_any=frozenset({"billing"})),
        Conditions(doc_assigned_to_user=True),
        Conditions(doc_assigned_to_user=False),
    ],
)
def test_no_condition_holds_on_a_bare_request(where: Conditions) -> None:
    # Document conditions only reach the rules on read and write, where a document is already
    # required, so the missing-document case is checked directly as defence in depth.
    engine = _engine(_rule("r"))
    bare = _send()
    assert engine._conditions_hold(where, bare) is False
    assert engine._conditions_hold(Conditions(), bare) is True


def test_label_bound_without_a_document_does_not_match() -> None:
    bundle = _bundle(_rule("r", labels={"max": "restricted"}))
    engine = PolicyEngine(bundle, CUSTOMER)
    assert engine._matches(bundle.policy.rules[0], _read(document=None)) is False


@pytest.mark.parametrize(
    ("deny_where", "request_"),
    [
        ({"recipient_external": True}, _send()),
        ({"recipient_in": "directory"}, _send()),
        ({"run_max_label_gte": "public"}, _send("a@acme.example")),
    ],
)
def test_missing_facts_do_not_trigger_deny_rules_either(
    deny_where: dict[str, Any], request_: Request
) -> None:
    deny = {
        "id": "deny",
        "effect": "deny",
        "tools": ["email.send"],
        "actions": ["send"],
        "where": deny_where,
    }
    engine = _engine(_send_rule("allow"), deny)
    assert engine.decide(request_).rule_ids == ("allow",)


def test_missing_amount_does_not_trigger_a_deny_rule() -> None:
    deny = {
        "id": "deny",
        "effect": "deny",
        "tools": ["pay.charge"],
        "actions": ["spend"],
        "where": {"amount_usd_lte": "100"},
    }
    engine = _engine(_spend_rule("allow"), deny)
    assert engine.decide(_spend(None)).rule_ids == ("allow",)


def test_missing_amount_does_not_trigger_a_lower_bound_deny_rule() -> None:
    deny = {
        "id": "deny-large",
        "effect": "deny",
        "tools": ["pay.charge"],
        "actions": ["spend"],
        "where": {"amount_usd_gt": "100"},
    }
    engine = _engine(_spend_rule("allow"), deny)
    assert engine.decide(_spend(None)).rule_ids == ("allow",)
    assert engine.decide(_spend(Decimal("100.01"))).rule_ids == ("deny-large",)


def test_all_conditions_must_hold() -> None:
    engine = _engine(
        _send_rule("r", where={"recipient_external": False, "run_max_label_gte": "internal"})
    )
    assert _effect(engine, _send("a@acme.example", run_max_label="internal")) is Effect.ALLOW
    assert _effect(engine, _send("a@acme.example", run_max_label="public")) is Effect.DENY
    assert _effect(engine, _send("a@other.example", run_max_label="internal")) is Effect.DENY


# Resolution


ALLOW_1 = _rule("zeta-allow")
APPROVAL_1 = _rule("beta-approval", "require_approval")
DENY_1 = _rule("omega-deny", "deny")
ALLOW_2 = _rule("alpha-allow")
APPROVAL_2 = _rule("gamma-approval", "require_approval")
DENY_2 = _rule("delta-deny", "deny")
UNMATCHED = _rule("unmatched", roles=["owner"])


def test_deny_overrides_approval_and_allow() -> None:
    engine = _engine(ALLOW_1, APPROVAL_1, DENY_1, UNMATCHED, ALLOW_2, APPROVAL_2, DENY_2)
    decision = engine.decide(_read())
    assert decision == Decision(
        Effect.DENY, Reason.MATCHED_DENY, ("omega-deny", "delta-deny"), engine.version
    )


def test_approval_overrides_allow() -> None:
    engine = _engine(ALLOW_1, APPROVAL_2, UNMATCHED, ALLOW_2, APPROVAL_1)
    decision = engine.decide(_read())
    assert decision == Decision(
        Effect.REQUIRE_APPROVAL,
        Reason.MATCHED_APPROVAL,
        ("gamma-approval", "beta-approval"),
        engine.version,
    )


def test_allow_lists_every_matching_allow_rule_in_file_order() -> None:
    engine = _engine(ALLOW_1, UNMATCHED, ALLOW_2)
    decision = engine.decide(_read())
    assert decision == Decision(
        Effect.ALLOW, Reason.MATCHED_ALLOW, ("zeta-allow", "alpha-allow"), engine.version
    )


def test_default_deny_when_nothing_matches() -> None:
    engine = _engine(UNMATCHED, _send_rule("send"))
    decision = engine.decide(_read())
    assert decision == Decision(Effect.DENY, Reason.DEFAULT_DENY, (), engine.version)


def test_non_matching_deny_does_not_block_allow() -> None:
    engine = _engine(ALLOW_1, _rule("deny-owner", "deny", roles=["owner"]))
    assert engine.decide(_read()).rule_ids == ("zeta-allow",)


# Determinism and versioning


def test_decisions_are_deterministic() -> None:
    rules = (ALLOW_1, APPROVAL_1, DENY_1, ALLOW_2, DENY_2)
    first = _engine(*rules)
    second = _engine(*rules)
    request = _read()
    decisions = {first.decide(request) for _ in range(50)} | {second.decide(request)}
    assert len(decisions) == 1


def test_every_decision_carries_the_bundle_version() -> None:
    bundle = _bundle(*OPEN)
    engine = PolicyEngine(bundle, CUSTOMER)
    assert engine.version == bundle_version(bundle)
    assert engine.bundle is bundle
    for request in (_read(), _read(customer_id="other"), _read(role=Role.OWNER)):
        assert engine.decide(request).policy_version == engine.version


def test_document_facts_from_document() -> None:
    document = Document(
        id="job-1",
        customer_id="acme",
        title="Job",
        label="internal",
        tags=frozenset({"assigned", "hvac"}),
        assigned_user_ids=frozenset({"dana"}),
        body="secret body",
    )
    facts = DocumentFacts.from_document(document)
    assert facts == DocumentFacts(
        customer_id="acme",
        label="internal",
        tags=frozenset({"assigned", "hvac"}),
        assigned_user_ids=frozenset({"dana"}),
    )
