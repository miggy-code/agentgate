import copy
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentgate.fixtures import FixtureError, load_fixtures
from agentgate.models import Action
from agentgate.policy.engine import PolicyEngine, Reason
from agentgate.policy.schema import Effect, PolicyError, load_bundle
from agentgate.policy.testing import (
    PolicyTestCase,
    load_cases,
    load_policy_bundles,
    run_cases,
    run_policy_tests,
)
from agentgate.policy.version import bundle_version
from agentgate.store import Store

REPO = Path(__file__).resolve().parent.parent

POLICY: dict[str, Any] = {
    "apiVersion": "agentgate/v1",
    "customer": "acme",
    "labels": ["public", "internal", "confidential"],
    "default": "deny",
    "spend": {"per_run_usd": "0.50", "per_customer_day_usd": "20", "max_calls_per_run": 40},
    "approvals": {"approver_roles": ["owner"], "ttl_minutes": 60, "risk_threshold": 50},
    "lists": {"customer_directory": ["@partner.example"]},
    "rules": [
        {
            "id": "read-internal",
            "effect": "allow",
            "roles": ["staff", "owner"],
            "tools": ["docs.read"],
            "actions": ["read"],
            "labels": {"max": "internal"},
        },
        {
            "id": "email-partners",
            "effect": "require_approval",
            "roles": ["staff"],
            "tools": ["email.send"],
            "actions": ["send"],
            "where": {"recipient_in": "customer_directory"},
        },
        {
            "id": "small-spend",
            "effect": "allow",
            "roles": ["owner"],
            "tools": ["billing.charge"],
            "actions": ["spend"],
            "where": {"amount_usd_lte": "100"},
        },
    ],
}
TOOLS: list[dict[str, Any]] = [
    {"id": "docs.read", "description": "Read", "actions": ["read"], "base_risk": 0, "cost_usd": 0},
    {
        "id": "email.send",
        "description": "Send",
        "actions": ["send"],
        "base_risk": 40,
        "cost_usd": "0.001",
    },
    {
        "id": "billing.charge",
        "description": "Charge",
        "actions": ["spend"],
        "base_risk": 60,
        "cost_usd": 0,
    },
]
CASES: list[dict[str, Any]] = [
    {
        "name": "staff reads the handbook",
        "user": "acme-staff",
        "tool": "docs.read",
        "action": "read",
        "document": "acme-handbook",
        "expect": "allow",
        "rules": ["read-internal"],
    },
    {
        "name": "staff emails a partner",
        "user": "acme-staff",
        "tool": "email.send",
        "action": "send",
        "recipients": ["ops@partner.example"],
        "expect": "require_approval",
        "rules": ["email-partners"],
    },
    {
        "name": "owner makes a small charge",
        "user": "acme-owner",
        "tool": "billing.charge",
        "action": "spend",
        "amount_usd": "100.00",
        "expect": "allow",
    },
    {
        "name": "owner makes a large charge",
        "user": "acme-owner",
        "tool": "billing.charge",
        "action": "spend",
        "amount_usd": "100.01",
        "expect": "deny",
        "rules": [],
    },
]


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_doc(root: Path, customer_id: str, doc_id: str, label: str) -> None:
    meta = {"title": doc_id.title(), "label": label}
    path = root / customer_id / "docs" / f"{doc_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\nBody.\n")


def write_fixtures(root: Path, customer_ids: tuple[str, ...] = ("acme", "beta")) -> Path:
    """Each customer gets an owner and a staff user plus an internal and a public doc."""
    for customer_id in customer_ids:
        _write_yaml(
            root / customer_id / "customer.yaml",
            {
                "id": customer_id,
                "name": customer_id.title(),
                "domains": [f"{customer_id}.example"],
                "timezone": "America/Chicago",
            },
        )
        _write_yaml(
            root / customer_id / "users.yaml",
            [
                {
                    "id": f"{customer_id}-{role}",
                    "name": f"{customer_id} {role}",
                    "email": f"{role}@{customer_id}.example",
                    "role": role,
                }
                for role in ("owner", "staff")
            ],
        )
        _write_doc(root, customer_id, f"{customer_id}-handbook", "internal")
        _write_doc(root, customer_id, f"{customer_id}-menu", "public")
    return root


def write_policies(
    root: Path, cases: list[dict[str, Any]] | None = None, customer_id: str = "acme"
) -> Path:
    """Write a policy bundle for customer_id; cases=None leaves tests.yaml out."""
    directory = root / customer_id
    _write_yaml(directory / "policy.yaml", copy.deepcopy(POLICY) | {"customer": customer_id})
    _write_yaml(directory / "tools.yaml", TOOLS)
    if cases is not None:
        _write_yaml(directory / "tests.yaml", cases)
    return root


def write_tree(tmp_path: Path, cases: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    """A policies and a fixtures tree side by side; returns (policies, fixtures)."""
    policies = write_policies(tmp_path / "policies", CASES if cases is None else cases)
    fixtures = write_fixtures(tmp_path / "fixtures")
    return policies, fixtures


def case(**changes: Any) -> dict[str, Any]:
    return copy.deepcopy(CASES[0]) | changes


@pytest.fixture
def setup(tmp_path: Path) -> Iterator[tuple[Store, PolicyEngine]]:
    policies, fixtures = write_tree(tmp_path)
    with Store() as store:
        load_fixtures(store, fixtures)
        customer = store.get_customer("acme")
        assert customer is not None
        yield store, PolicyEngine(load_bundle(policies / "acme"), customer)


def _run_one(setup: tuple[Store, PolicyEngine], **changes: Any) -> tuple[bool, str | None]:
    store, engine = setup
    [result] = run_cases(store, engine, [PolicyTestCase.model_validate(case(**changes))])
    assert result.customer_id == "acme"
    return result.passed, result.problem


# Loading cases


def test_load_cases_parses_every_key(tmp_path: Path) -> None:
    path = tmp_path / "tests.yaml"
    path.write_text(
        "- name: all keys\n"
        "  user: acme-staff\n"
        "  tool: email.send\n"
        "  action: send\n"
        "  document: acme-handbook\n"
        "  recipients: [a@partner.example, b@partner.example]\n"
        '  amount_usd: "12.50"\n'
        "  run_max_label: internal\n"
        "  expect: require_approval\n"
        "  rules: [email-partners]\n"
        "- name: defaults\n"
        "  user: acme-staff\n"
        "  tool: docs.read\n"
        "  action: read\n"
        "  expect: deny\n"
    )
    full, minimal = load_cases(path)
    assert full.action is Action.SEND
    assert full.recipients == ("a@partner.example", "b@partner.example")
    assert full.amount_usd == Decimal("12.50")
    assert full.run_max_label == "internal"
    assert full.expect is Effect.REQUIRE_APPROVAL
    assert full.rules == ("email-partners",)
    assert minimal.document is None
    assert minimal.recipients == ()
    assert minimal.amount_usd is None
    assert minimal.rules is None


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (case(rule=["read-internal"]), r"case 1: rule: Extra inputs are not permitted"),
        ({k: v for k, v in CASES[0].items() if k != "expect"}, r"case 1: expect: Field required"),
        (case(expect="maybe"), r"case 1: expect:"),
        (case(action="delete"), r"case 1: action:"),
        (case(amount_usd="lots"), r"case 1: amount_usd:"),
        ("not a mapping", r"case 1:"),
    ],
)
def test_load_cases_rejects_bad_entries(tmp_path: Path, entry: object, message: str) -> None:
    path = tmp_path / "tests.yaml"
    _write_yaml(path, [entry])
    with pytest.raises(PolicyError, match=message) as info:
        load_cases(path)
    assert str(path) in str(info.value)


def test_load_cases_names_the_failing_index(tmp_path: Path) -> None:
    path = tmp_path / "tests.yaml"
    _write_yaml(path, [CASES[0], CASES[1], case(name="third", bogus=1)])
    with pytest.raises(PolicyError, match="case 3: bogus"):
        load_cases(path)


def test_load_cases_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "tests.yaml"
    _write_yaml(path, [CASES[0], CASES[1], case()])
    with pytest.raises(PolicyError, match=r"case 3: duplicate name .*\(also case 1\)"):
        load_cases(path)


@pytest.mark.parametrize("text", ["", "name: not a list\n", "[]\n"])
def test_load_cases_needs_a_non_empty_list(tmp_path: Path, text: str) -> None:
    path = tmp_path / "tests.yaml"
    path.write_text(text)
    with pytest.raises(PolicyError, match="non-empty list of test cases"):
        load_cases(path)


def test_load_cases_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "tests.yaml"
    path.write_text("- name: a\n  name: b\n")
    with pytest.raises(PolicyError, match="duplicate key"):
        load_cases(path)


# Running cases


def test_matching_cases_pass(setup: tuple[Store, PolicyEngine]) -> None:
    store, engine = setup
    cases = [PolicyTestCase.model_validate(entry) for entry in CASES]
    results = run_cases(store, engine, cases)
    assert [result.problem for result in results] == [None] * len(CASES)
    assert all(result.passed for result in results)
    assert [result.case for result in results] == cases
    decision = results[0].decision
    assert decision is not None
    assert decision.rule_ids == ("read-internal",)
    assert decision.policy_version == engine.version


def test_expect_mismatch_reports_the_reason(setup: tuple[Store, PolicyEngine]) -> None:
    assert _run_one(setup, document="acme-menu", user="acme-staff", tool="email.send") == (
        False,
        "expected allow, got deny (action_not_supported)",
    )
    assert _run_one(setup, user="acme-staff", tool="docs.read", expect="deny") == (
        False,
        "expected deny, got allow (matched_allow [read-internal])",
    )


def test_default_deny_mismatch(setup: tuple[Store, PolicyEngine]) -> None:
    passed, problem = _run_one(
        setup, user="acme-owner", tool="billing.charge", action="spend", document=None
    )
    assert not passed
    assert problem == "expected allow, got deny (default_deny)"


def test_rules_mismatch(setup: tuple[Store, PolicyEngine]) -> None:
    assert _run_one(setup, rules=["other-rule"]) == (
        False,
        "expected rules [other-rule], got [read-internal]",
    )
    assert _run_one(setup, rules=[]) == (False, "expected rules [], got [read-internal]")


def test_rules_none_is_not_checked(setup: tuple[Store, PolicyEngine]) -> None:
    assert _run_one(setup, rules=None) == (True, None)


def test_unknown_user_fails_the_case(setup: tuple[Store, PolicyEngine]) -> None:
    store, engine = setup
    [result] = run_cases(store, engine, [PolicyTestCase.model_validate(case(user="nobody"))])
    assert not result.passed
    assert result.decision is None
    assert result.problem == "unknown user 'nobody'"


def test_user_from_another_customer_fails_the_case(setup: tuple[Store, PolicyEngine]) -> None:
    assert _run_one(setup, user="beta-staff") == (
        False,
        "user 'beta-staff' belongs to customer 'beta'",
    )


def test_unknown_document_fails_the_case(setup: tuple[Store, PolicyEngine]) -> None:
    assert _run_one(setup, document="acme-missing") == (
        False,
        "unknown document 'acme-missing'",
    )


def test_cross_customer_document_is_denied(setup: tuple[Store, PolicyEngine]) -> None:
    store, engine = setup
    deny = PolicyTestCase.model_validate(case(document="beta-handbook", expect="deny", rules=[]))
    allow = PolicyTestCase.model_validate(case(document="beta-handbook", rules=None))
    passed, failed = run_cases(store, engine, [deny, allow])
    assert passed.passed
    assert passed.decision is not None
    assert passed.decision.reason is Reason.DOCUMENT_CUSTOMER_MISMATCH
    assert not failed.passed
    assert failed.problem == "expected allow, got deny (document_customer_mismatch)"


def test_one_failure_does_not_stop_the_rest(setup: tuple[Store, PolicyEngine]) -> None:
    store, engine = setup
    cases = [
        PolicyTestCase.model_validate(case(name="a", user="nobody")),
        PolicyTestCase.model_validate(case(name="b")),
    ]
    assert [result.passed for result in run_cases(store, engine, cases)] == [False, True]


# Whole trees


def test_run_policy_tests_reports_per_customer(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path)
    write_policies(policies, [case(user="beta-staff", document="beta-handbook")], "beta")
    (policies / "README.md").write_text("notes")
    report = run_policy_tests(policies, fixtures)
    assert [customer.customer_id for customer in report.customers] == ["acme", "beta"]
    acme, beta = report.customers
    assert acme.passed == len(CASES)
    assert acme.failed == 0
    assert acme.policy_version == bundle_version(load_bundle(policies / "acme"))
    assert beta.passed == 1
    assert (report.passed, report.failed, report.total) == (len(CASES) + 1, 0, len(CASES) + 1)


def test_run_policy_tests_counts_failures(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path, [*CASES, case(name="wrong", expect="deny")])
    report = run_policy_tests(policies, fixtures)
    assert (report.passed, report.failed, report.total) == (len(CASES), 1, len(CASES) + 1)
    [failure] = [r for r in report.customers[0].results if not r.passed]
    assert failure.case.name == "wrong"


def test_missing_tests_file_is_an_error(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path)
    (policies / "acme" / "tests.yaml").unlink()
    with pytest.raises(PolicyError, match=r"tests\.yaml: missing tests\.yaml"):
        run_policy_tests(policies, fixtures)


def test_policy_without_fixture_customer_is_an_error(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path)
    write_policies(policies, CASES, "gamma")
    with pytest.raises(PolicyError, match="customer 'gamma' has no fixtures"):
        run_policy_tests(policies, fixtures)


def test_bad_tests_file_stops_the_run(tmp_path: Path) -> None:
    policies, fixtures = write_tree(tmp_path, [case(expected="allow")])
    with pytest.raises(PolicyError, match=r"acme/tests\.yaml: case 1: expected"):
        run_policy_tests(policies, fixtures)


def test_missing_or_empty_policies_dir_is_an_error(tmp_path: Path) -> None:
    fixtures = write_fixtures(tmp_path / "fixtures")
    with pytest.raises(PolicyError, match="policies directory not found"):
        run_policy_tests(tmp_path / "nope", fixtures)
    (tmp_path / "empty").mkdir()
    with pytest.raises(PolicyError, match="no policy directories found"):
        load_policy_bundles(tmp_path / "empty")


def test_missing_fixtures_dir_is_an_error(tmp_path: Path) -> None:
    policies, _ = write_tree(tmp_path)
    with pytest.raises(FixtureError, match="fixtures directory not found"):
        run_policy_tests(policies, tmp_path / "nope")


def test_real_repo_policies_run() -> None:
    # Only checks that the real tree loads and runs; whether every case passes is the
    # policy test suite's job (make policy), not this unit test's.
    report = run_policy_tests(REPO / "policies", REPO / "fixtures")
    customers = {customer.customer_id for customer in report.customers}
    assert {"kettle-creek-hvac", "juniper-row-apothecary"} <= customers
    assert all(customer.results for customer in report.customers)
