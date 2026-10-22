import copy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentgate.models import Action, Role
from agentgate.policy.schema import Effect, PolicyError, load_bundle, load_bundles

POLICY: dict[str, Any] = {
    "apiVersion": "agentgate/v1",
    "customer": "acme",
    "labels": ["public", "internal", "confidential"],
    "default": "deny",
    "spend": {"per_run_usd": "0.50", "per_customer_day_usd": "20", "max_calls_per_run": 40},
    "approvals": {"approver_roles": ["owner"], "ttl_minutes": 60, "risk_threshold": 50},
    "lists": {"customer_directory": ["pat@home.example", "@partner.example"]},
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
            "id": "email-customers",
            "effect": "require_approval",
            "roles": ["staff"],
            "tools": ["email.send"],
            "actions": ["send"],
            "where": {"recipient_in": "customer_directory"},
        },
        {
            "id": "no-external-after-confidential",
            "effect": "deny",
            "tools": ["email.send"],
            "actions": ["send"],
            "where": {"run_max_label_gte": "confidential", "recipient_external": True},
        },
    ],
}
TOOLS: list[dict[str, Any]] = [
    {
        "id": "docs.read",
        "description": "Read a document",
        "actions": ["read"],
        "base_risk": 0,
        "cost_usd": "0",
    },
    {
        "id": "email.send",
        "description": "Send an email",
        "actions": ["send"],
        "customer_facing": True,
        "base_risk": 40,
        "cost_usd": "0.001",
    },
]


def _write(root: Path, policy: object = None, tools: object = None, name: str = "acme") -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    if policy is not None:
        (directory / "policy.yaml").write_text(yaml.safe_dump(policy))
    if tools is not None:
        (directory / "tools.yaml").write_text(yaml.safe_dump(tools))
    return directory


def _policy(**changes: Any) -> dict[str, Any]:
    return copy.deepcopy(POLICY) | changes


def _with_rule(**rule: Any) -> dict[str, Any]:
    policy = _policy()
    policy["rules"].append(rule)
    return policy


def test_loads_a_valid_bundle(tmp_path: Path) -> None:
    bundle = load_bundle(_write(tmp_path, POLICY, TOOLS))
    assert bundle.customer_id == "acme"
    assert set(bundle.tools) == {"docs.read", "email.send"}
    rule = bundle.policy.rules[0]
    assert rule.effect is Effect.ALLOW
    assert rule.roles == {Role.STAFF, Role.OWNER}
    assert rule.actions == {Action.READ}
    assert bundle.policy.spend.per_run_usd == Decimal("0.50")
    assert bundle.policy.label_rank("confidential") == 2
    assert bundle.policy.label_rank("secret") is None


def test_load_bundles_keys_by_customer_and_skips_files(tmp_path: Path) -> None:
    _write(tmp_path, POLICY, TOOLS)
    _write(tmp_path, _policy(customer="beta"), TOOLS, name="beta")
    (tmp_path / "README.md").write_text("notes")
    assert list(load_bundles(tmp_path)) == ["acme", "beta"]


@pytest.mark.parametrize("missing", ["policy.yaml", "tools.yaml"])
def test_missing_file(tmp_path: Path, missing: str) -> None:
    directory = _write(tmp_path, POLICY, TOOLS)
    (directory / missing).unlink()
    with pytest.raises(PolicyError, match=missing):
        load_bundle(directory)


def test_customer_must_match_directory(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="does not match directory"):
        load_bundle(_write(tmp_path, _policy(customer="other"), TOOLS))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"apiVersion": "agentgate/v2"}, "apiVersion"),
        ({"default": "allow"}, "default"),
        ({"labels": ["public", "public"]}, "labels must be unique"),
        ({"unexpected": True}, "unexpected"),
        ({"rules": []}, "rules"),
    ],
)
def test_policy_level_rejections(tmp_path: Path, changes: dict[str, Any], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        load_bundle(_write(tmp_path, _policy(**changes), TOOLS))


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    directory = _write(tmp_path, POLICY, TOOLS)
    text = (directory / "policy.yaml").read_text()
    (directory / "policy.yaml").write_text(text + "default: deny\n")
    with pytest.raises(PolicyError, match="duplicate key"):
        load_bundle(directory)


BASE_RULE = {"id": "extra", "tools": ["docs.read"], "actions": ["read"]}


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        (BASE_RULE | {"effect": "allow"}, "allow rules must list roles"),
        (BASE_RULE | {"effect": "require_approval"}, "require_approval rules must list roles"),
        (BASE_RULE | {"id": "read-internal", "effect": "deny"}, "duplicate rule id"),
        (BASE_RULE | {"effect": "deny", "labels": {"max": "secret"}}, "unknown label 'secret'"),
        (
            BASE_RULE | {"effect": "deny", "where": {"run_max_label_gte": "secret"}},
            "unknown label 'secret'",
        ),
        (
            {
                "id": "extra",
                "effect": "deny",
                "tools": ["email.send"],
                "actions": ["send"],
                "where": {"recipient_in": "nobody"},
            },
            "unknown list 'nobody'",
        ),
        (
            BASE_RULE | {"effect": "deny", "where": {"recipient_external": True}},
            "recipient_external applies only to send",
        ),
        (
            BASE_RULE | {"effect": "deny", "where": {"amount_usd_lte": "5"}},
            "amount_usd_lte applies only to spend",
        ),
        (
            {
                "id": "extra",
                "effect": "deny",
                "tools": ["email.send"],
                "actions": ["send"],
                "labels": {"max": "internal"},
            },
            "labels apply only to read and write",
        ),
        (
            {
                "id": "extra",
                "effect": "deny",
                "tools": ["email.send"],
                "actions": ["send"],
                "where": {"doc_assigned_to_user": True},
            },
            "doc_assigned_to_user applies only to read, write",
        ),
        (BASE_RULE | {"effect": "deny", "where": {"is_admin": True}}, "is_admin"),
        (BASE_RULE | {"effect": "maybe"}, "effect"),
        (BASE_RULE | {"effect": "deny", "roles": ["admin"]}, "roles"),
        (BASE_RULE | {"effect": "deny", "roles": []}, "roles"),
        (BASE_RULE | {"effect": "deny", "tools": []}, "tools"),
        (BASE_RULE | {"effect": "deny", "actions": ["delete"]}, "actions"),
    ],
)
def test_rule_rejections(tmp_path: Path, rule: dict[str, Any], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        load_bundle(_write(tmp_path, _with_rule(**rule), TOOLS))


def test_rule_must_name_known_tools(tmp_path: Path) -> None:
    policy = _with_rule(id="extra", effect="deny", tools=["sms.send"], actions=["send"])
    with pytest.raises(PolicyError, match=r"unknown tool 'sms\.send'"):
        load_bundle(_write(tmp_path, policy, TOOLS))


def test_rule_actions_must_be_supported_by_each_tool(tmp_path: Path) -> None:
    policy = _with_rule(id="extra", effect="deny", tools=["docs.read"], actions=["read", "write"])
    with pytest.raises(PolicyError, match="does not support write"):
        load_bundle(_write(tmp_path, policy, TOOLS))


def test_list_entries_are_emails_or_domains(tmp_path: Path) -> None:
    policy = _policy(lists={"customer_directory": ["not-an-address"]})
    with pytest.raises(PolicyError, match="customer_directory"):
        load_bundle(_write(tmp_path, policy, TOOLS))


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ([], "non-empty list"),
        ({"docs.read": {}}, "non-empty list"),
        ([TOOLS[0], TOOLS[0], TOOLS[1]], "duplicate tool id"),
        ([TOOLS[0] | {"base_risk": 500}, TOOLS[1]], "base_risk"),
    ],
)
def test_tool_file_rejections(tmp_path: Path, tools: object, message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        load_bundle(_write(tmp_path, POLICY, tools))


@pytest.mark.parametrize(
    "spend",
    [
        {"per_run_usd": "-1", "per_customer_day_usd": "20", "max_calls_per_run": 40},
        {"per_run_usd": "1", "per_customer_day_usd": "20", "max_calls_per_run": 0},
    ],
)
def test_spend_limits_are_bounded(tmp_path: Path, spend: dict[str, Any]) -> None:
    with pytest.raises(PolicyError, match="spend"):
        load_bundle(_write(tmp_path, _policy(spend=spend), TOOLS))
