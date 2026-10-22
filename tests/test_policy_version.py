import copy
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentgate.policy.schema import PolicyBundle, load_bundle
from agentgate.policy.version import bundle_version, git_revision

POLICY: dict[str, Any] = {
    "apiVersion": "agentgate/v1",
    "customer": "acme",
    "labels": ["public", "internal", "confidential"],
    "default": "deny",
    "spend": {"per_run_usd": "0.50", "per_customer_day_usd": "20", "max_calls_per_run": 40},
    "approvals": {"approver_roles": ["owner", "manager"], "ttl_minutes": 60, "risk_threshold": 50},
    "lists": {"directory": ["pat@home.example", "@partner.example", "lee@home.example"]},
    "rules": [
        {
            "id": "read-internal",
            "effect": "allow",
            "roles": ["staff", "owner", "manager", "contractor"],
            "tools": ["docs.read", "docs.search"],
            "actions": ["read"],
            "labels": {"max": "internal"},
            "where": {"doc_tags_any": ["billing", "vip", "hvac", "warranty"]},
        },
        {
            "id": "email-customers",
            "effect": "require_approval",
            "roles": ["staff", "manager"],
            "tools": ["email.send"],
            "actions": ["send"],
            "where": {"recipient_in": "directory"},
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
        "id": "docs.search",
        "description": "Search documents",
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

POLICY_YAML = """\
apiVersion: agentgate/v1
customer: acme
labels: [public, internal, confidential]
default: deny
spend: {per_run_usd: "0.50", per_customer_day_usd: "20", max_calls_per_run: 40}
approvals: {approver_roles: [owner, manager], ttl_minutes: 60, risk_threshold: 50}
lists:
  directory: [pat@home.example, "@partner.example", lee@home.example]
rules:
  - id: read-internal
    effect: allow
    roles: [staff, owner, manager, contractor]
    tools: [docs.read, docs.search]
    actions: [read]
    labels: {max: internal}
    where: {doc_tags_any: [billing, vip, hvac, warranty]}
  - id: email-customers
    effect: require_approval
    roles: [staff, manager]
    tools: [email.send]
    actions: [send]
    where: {recipient_in: directory}
"""

# The same policy: keys reordered, set members reordered, decimals respelled, comments added.
POLICY_YAML_REFORMATTED = """\
# Acme policy, reviewed by the owner.
customer: acme
apiVersion: agentgate/v1   # format version

default: deny
labels:
  - public
  - internal
  - confidential
approvals:
  risk_threshold: 50
  ttl_minutes: 60
  approver_roles: [manager, owner]
spend:
  max_calls_per_run: 40
  per_customer_day_usd: "20.00"
  per_run_usd: "0.5"
lists:
  directory:
    - lee@home.example
    - "@partner.example"
    - pat@home.example
rules:
  - effect: allow
    id: read-internal
    where:
      doc_tags_any: [warranty, hvac, vip, billing]
    labels:
      max: internal
    actions: [read]
    tools: [docs.search, docs.read]
    roles: [contractor, manager, owner, staff]
  # Customer-facing mail always goes through a person.
  - id: email-customers
    actions: [send]
    where:
      recipient_in: directory
    tools: [email.send]
    roles: [manager, staff]
    effect: require_approval
"""


def _write(directory: Path, policy_text: str, tools: object = TOOLS) -> Path:
    directory.mkdir(parents=True)
    (directory / "policy.yaml").write_text(policy_text)
    (directory / "tools.yaml").write_text(yaml.safe_dump(tools, sort_keys=False))
    return directory


def _bundle(policy: dict[str, Any] = POLICY, tools: list[dict[str, Any]] = TOOLS) -> PolicyBundle:
    return PolicyBundle.model_validate(
        {"policy": policy, "tools": {tool["id"]: tool for tool in tools}}
    )


def _changed_policy() -> dict[str, Any]:
    return copy.deepcopy(POLICY)


def test_version_format() -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{16}", bundle_version(_bundle()))


def test_version_ignores_yaml_formatting(tmp_path: Path) -> None:
    reordered_tools = [dict(reversed(tool.items())) for tool in reversed(TOOLS)]
    first = load_bundle(_write(tmp_path / "a" / "acme", POLICY_YAML))
    second = load_bundle(_write(tmp_path / "b" / "acme", POLICY_YAML_REFORMATTED, reordered_tools))
    assert first == second
    assert bundle_version(first) == bundle_version(second)
    assert bundle_version(first) == bundle_version(_bundle())


def _with_rule_effect() -> dict[str, Any]:
    policy = _changed_policy()
    policy["rules"][1]["effect"] = "deny"
    return policy


def _with_rule_role() -> dict[str, Any]:
    policy = _changed_policy()
    policy["rules"][1]["roles"].append("owner")
    return policy


def _with_condition() -> dict[str, Any]:
    policy = _changed_policy()
    policy["rules"][0]["where"]["doc_tags_any"].remove("vip")
    return policy


def _with_label_bound() -> dict[str, Any]:
    policy = _changed_policy()
    policy["rules"][0]["labels"]["max"] = "confidential"
    return policy


def _with_rules_reordered() -> dict[str, Any]:
    policy = _changed_policy()
    policy["rules"].reverse()
    return policy


def _with_labels_reordered() -> dict[str, Any]:
    policy = _changed_policy()
    policy["labels"] = ["internal", "public", "confidential"]
    return policy


def _with_list_entry_added() -> dict[str, Any]:
    policy = _changed_policy()
    policy["lists"]["directory"].append("@evil.example")
    return policy


def _with_list_entry_changed() -> dict[str, Any]:
    policy = _changed_policy()
    policy["lists"]["directory"][1] = "@sub.partner.example"
    return policy


def _with_spend_limit() -> dict[str, Any]:
    policy = _changed_policy()
    policy["spend"]["per_run_usd"] = "0.51"
    return policy


def _with_approvers() -> dict[str, Any]:
    policy = _changed_policy()
    policy["approvals"]["approver_roles"] = ["owner"]
    return policy


@pytest.mark.parametrize(
    "policy",
    [
        _with_rule_effect(),
        _with_rule_role(),
        _with_condition(),
        _with_label_bound(),
        _with_rules_reordered(),
        _with_labels_reordered(),
        _with_list_entry_added(),
        _with_list_entry_changed(),
        _with_spend_limit(),
        _with_approvers(),
    ],
)
def test_version_changes_with_policy_meaning(policy: dict[str, Any]) -> None:
    assert bundle_version(_bundle(policy)) != bundle_version(_bundle())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_risk", 41),
        ("customer_facing", False),
        ("cost_usd", "0.002"),
        ("actions", ["send", "write"]),
    ],
)
def test_version_changes_with_tool_meaning(field: str, value: object) -> None:
    tools = copy.deepcopy(TOOLS)
    tools[2][field] = value
    assert bundle_version(_bundle(tools=tools)) != bundle_version(_bundle())


def test_version_changes_when_a_tool_is_added() -> None:
    extra = TOOLS[0] | {"id": "docs.write", "actions": ["write"]}
    assert bundle_version(_bundle(tools=[*TOOLS, extra])) != bundle_version(_bundle())


_PROBE = """
import sys
from pathlib import Path
from agentgate.policy.schema import load_bundle
from agentgate.policy.version import bundle_version
bundle = load_bundle(Path(sys.argv[1]))
rule = bundle.policy.rules[0]
order = [list(rule.roles or ()), list(rule.tools)]
order.append(list(rule.where.doc_tags_any or ()))
print(bundle_version(bundle), repr(order))
"""


def test_version_is_independent_of_set_iteration_order(tmp_path: Path) -> None:
    # String hashing, and so frozenset iteration order, changes with PYTHONHASHSEED.
    directory = _write(tmp_path / "acme", POLICY_YAML)
    versions: set[str] = set()
    orders: set[str] = set()
    for seed in range(12):
        env = os.environ | {"PYTHONHASHSEED": str(seed)}
        result = subprocess.run(
            [sys.executable, "-c", _PROBE, str(directory)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=60,
        )
        version, order = result.stdout.strip().split(" ", 1)
        versions.add(version)
        orders.add(order)
    assert len(orders) > 1, "hash seeds did not change set order, so this test proves nothing"
    assert versions == {bundle_version(load_bundle(directory))}


# git_revision


@pytest.fixture
def git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate git from the developer's config and from any repository above tmp_path."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    return tmp_path


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.example", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout.strip()


def _repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "acme").mkdir(parents=True)
    (repo / "acme" / "policy.yaml").write_text("customer: acme\n")
    (repo / "other.txt").write_text("other\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "first")
    return repo


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", message)


def test_git_revision_outside_a_repo(git_env: Path) -> None:
    (git_env / "policy.yaml").write_text("customer: acme\n")
    assert git_revision(git_env) is None
    assert git_revision(git_env / "policy.yaml") is None


def test_git_revision_without_git(git_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(git_env / "empty-bin"))
    assert git_revision(git_env) is None


def test_git_revision_for_a_missing_path(git_env: Path) -> None:
    assert git_revision(git_env / "nope" / "policy.yaml") is None


@needs_git
def test_git_revision_of_a_committed_path(git_env: Path) -> None:
    repo = _repo(git_env)
    head = _git(repo, "rev-parse", "--short", "HEAD")
    assert git_revision(repo / "acme") == head
    assert git_revision(repo / "acme" / "policy.yaml") == head


@needs_git
def test_git_revision_is_the_last_commit_touching_the_path(git_env: Path) -> None:
    repo = _repo(git_env)
    first = _git(repo, "rev-parse", "--short", "HEAD")
    (repo / "other.txt").write_text("changed\n")
    _commit_all(repo, "second")
    assert git_revision(repo / "acme") == first
    (repo / "acme" / "policy.yaml").write_text("customer: acme\ndefault: deny\n")
    _commit_all(repo, "third")
    assert git_revision(repo / "acme") == _git(repo, "rev-parse", "--short", "HEAD")


@needs_git
def test_git_revision_is_none_with_uncommitted_changes(git_env: Path) -> None:
    repo = _repo(git_env)
    policy = repo / "acme" / "policy.yaml"
    policy.write_text("customer: acme\ndefault: deny\n")
    assert git_revision(policy) is None
    assert git_revision(repo / "acme") is None
    # Staged but not committed is still not what the commit describes.
    _git(repo, "add", ".")
    assert git_revision(repo / "acme") is None


@needs_git
def test_git_revision_is_none_with_an_untracked_file(git_env: Path) -> None:
    repo = _repo(git_env)
    (repo / "acme" / "tools.yaml").write_text("[]\n")
    assert git_revision(repo / "acme") is None
    # Other paths are unaffected.
    assert git_revision(repo / "other.txt") == _git(repo, "rev-parse", "--short", "HEAD")


@needs_git
def test_git_revision_is_none_for_a_path_never_committed(git_env: Path) -> None:
    repo = _repo(git_env)
    new = repo / "beta"
    new.mkdir()
    assert git_revision(new) is None
