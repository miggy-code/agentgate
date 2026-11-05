"""Models-only policies (no tools.yaml, rules or approvals) and `agentgate init-experiment`,
which writes one: an experiment's "this user may call this model under these caps"."""

import copy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentgate.approvals import ApprovalError, ApprovalErrorReason
from agentgate.cli import app
from agentgate.gate import Denied
from agentgate.models import Action, Customer, Role
from agentgate.policy.engine import ModelRequest, PolicyEngine, Reason, Request
from agentgate.policy.schema import Effect, ModelProvider, PolicyError, load_bundle
from agentgate.policy.version import bundle_version
from agentgate.world import open_memory_world
from test_cli_ops import runner
from test_policy_schema import POLICY, TOOLS

MODELS_ONLY: dict[str, Any] = {
    "apiVersion": "agentgate/v1",
    "customer": "lab",
    "labels": ["public"],
    "default": "deny",
    "spend": {"per_run_usd": "0.25", "per_customer_day_usd": "1", "max_calls_per_run": 10},
    "models": {
        "allow": [
            {
                "id": "deepseek-flash",
                "provider": "openai_compatible",
                "model": "deepseek-flash",
                "roles": ["owner"],
                "max_output_tokens": 8000,
                "max_request_bytes": 120000,
            }
        ],
    },
}
LAB = Customer(id="lab", name="Lab", domains=("lab.example",), timezone="UTC")


def _write(root: Path, policy: dict[str, Any], tools: object = None) -> Path:
    directory = root / policy["customer"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "policy.yaml").write_text(yaml.safe_dump(policy))
    if tools is not None:
        (directory / "tools.yaml").write_text(yaml.safe_dump(tools))
    return directory


# Schema


def test_a_models_only_policy_loads_without_tools_rules_or_approvals(tmp_path: Path) -> None:
    bundle = load_bundle(_write(tmp_path, MODELS_ONLY))
    assert bundle.tools == {} and bundle.policy.rules == ()
    assert bundle.policy.approvals is None
    assert bundle.policy.models is not None and bundle.policy.spend.max_calls_per_run == 10
    assert bundle_version(bundle).startswith("sha256:")


def test_a_models_only_policy_may_still_ship_a_tools_file(tmp_path: Path) -> None:
    bundle = load_bundle(_write(tmp_path, MODELS_ONLY, TOOLS))
    assert set(bundle.tools) == {"docs.read", "email.send"} and bundle.policy.rules == ()


def test_a_policy_that_allows_nothing_is_rejected(tmp_path: Path) -> None:
    policy = {k: v for k, v in MODELS_ONLY.items() if k != "models"}
    with pytest.raises(PolicyError, match="policy allows nothing"):
        load_bundle(_write(tmp_path, policy))


@pytest.mark.parametrize("effect", ["allow", "require_approval"])
def test_approvals_are_required_once_a_rule_can_allow(tmp_path: Path, effect: str) -> None:
    policy = copy.deepcopy(POLICY)
    del policy["approvals"]
    policy["rules"] = [r for r in policy["rules"] if r["effect"] in (effect, "deny")]
    with pytest.raises(PolicyError, match="approvals: required"):
        load_bundle(_write(tmp_path, policy, TOOLS))


def test_deny_only_rules_need_no_approvals(tmp_path: Path) -> None:
    policy = copy.deepcopy(POLICY)
    del policy["approvals"]
    policy["rules"] = [r for r in policy["rules"] if r["effect"] == "deny"]
    assert load_bundle(_write(tmp_path, policy, TOOLS)).policy.approvals is None


def test_rules_still_need_a_tools_file(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match=r"tools\.yaml \(the policy has rules\)"):
        load_bundle(_write(tmp_path, copy.deepcopy(POLICY) | {"customer": "acme"}))


# Engine and gate


@pytest.mark.parametrize(
    ("tools", "reason"), [(None, Reason.UNKNOWN_TOOL), (TOOLS, Reason.DEFAULT_DENY)]
)
def test_the_engine_denies_every_tool_and_allows_the_listed_model(
    tmp_path: Path, tools: object, reason: Reason
) -> None:
    engine = PolicyEngine(load_bundle(_write(tmp_path, MODELS_ONLY, tools)), LAB)
    request = Request(
        customer_id="lab",
        user_id="u",
        role=Role.OWNER,
        tool="email.send",
        action=Action.SEND,
        recipients=("pat@lab.example",),
    )
    tool = engine.decide(request)
    assert (tool.effect, tool.reason) == (Effect.DENY, reason)
    model = engine.decide_model(
        ModelRequest(
            customer_id="lab",
            role=Role.OWNER,
            provider=ModelProvider.OPENAI_COMPATIBLE,
            model="deepseek-flash",
            max_tokens=1000,
            n=None,
            request_bytes=500,
        )
    )
    assert (model.effect, model.reason) == (Effect.ALLOW, Reason.MODEL_ALLOWED)


def test_the_gate_denies_tool_calls_under_a_models_only_policy(tmp_path: Path) -> None:
    _init(tmp_path)
    world = open_memory_world(policies=tmp_path / "policies", fixtures=tmp_path / "fixtures")
    try:
        ctx = world.gate.start_run("lab", "lab-runner")
        outcome = world.gate.call_tool(ctx, "docs.read", Action.READ, {"document_id": "x"})
        assert isinstance(outcome, Denied)
        with pytest.raises(ApprovalError) as raised:
            world.gate.approve("nope", "lab-runner")
        assert raised.value.reason is ApprovalErrorReason.APPROVER_NOT_ALLOWED
    finally:
        world.close()


# init-experiment


def _init(root: Path, *extra: str) -> Any:
    result = runner.invoke(
        app,
        [
            "init-experiment",
            str(root),
            "--model",
            "openai_compatible:deepseek-flash",
            "--max-usd",
            "0.50",
            "--max-calls",
            "40",
            *extra,
        ],
    )
    return result


def test_init_experiment_writes_a_loadable_models_only_policy(tmp_path: Path) -> None:
    result = _init(tmp_path, "--model", "gemini:models/gemini-2.5-pro", "--day-usd", "5")
    assert result.exit_code == 0, result.output
    assert f"policies {tmp_path / 'policies'}" in result.stdout
    assert not (tmp_path / "policies" / "lab" / "tools.yaml").exists()
    policy = load_bundle(tmp_path / "policies" / "lab").policy
    assert policy.rules == () and policy.approvals is None
    assert policy.spend.per_run_usd == Decimal("0.50")
    assert policy.spend.per_customer_day_usd == Decimal("5")
    assert policy.spend.max_calls_per_run == 40
    assert policy.models is not None
    assert [(e.id, e.provider.value, e.model, e.roles) for e in policy.models.allow] == [
        ("deepseek-flash", "openai_compatible", "deepseek-flash", {Role.OWNER}),
        ("models-gemini-2-5-pro", "gemini", "models/gemini-2.5-pro", {Role.OWNER}),
    ]
    users = yaml.safe_load((tmp_path / "fixtures" / "lab" / "users.yaml").read_text())
    assert [(u["id"], u["role"]) for u in users] == [("lab-runner", "owner")]


def test_init_experiment_never_overwrites(tmp_path: Path) -> None:
    assert _init(tmp_path).exit_code == 0
    before = (tmp_path / "policies" / "lab" / "policy.yaml").read_text()
    result = _init(tmp_path, "--max-calls", "1")
    assert result.exit_code == 1 and "already exists" in result.stderr
    assert (tmp_path / "policies" / "lab" / "policy.yaml").read_text() == before


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--model", "deepseek"), "unknown provider"),
        (("--model", "openai:"), "PROVIDER:NAME"),
        (("--max-usd", "lots"), "invalid amount"),
        (("--max-usd", "-1"), "per_run_usd"),
        (("--customer", "Not A Slug"), "id"),
    ],
)
def test_init_experiment_rejects_bad_input_and_writes_nothing(
    tmp_path: Path, extra: tuple[str, ...], message: str
) -> None:
    result = _init(tmp_path, *extra)
    assert result.exit_code == 1 and message in result.stderr, result.output
    assert not (tmp_path / "policies").exists() and not (tmp_path / "fixtures").exists()


def test_init_experiment_never_follows_a_symlink_or_truncates_a_late_arrival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file or symlink that appears after the existence check is neither truncated nor followed,
    and the files this call already created are removed again."""
    from agentgate import experiment

    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    late = tmp_path / "policies" / "lab" / "policy.yaml"
    real_create = experiment._create

    def create(path: Path, text: str) -> tuple[int, int]:
        if path == late:  # simulate a racing process planting a symlink between check and create
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(victim)
        return real_create(path, text)

    monkeypatch.setattr(experiment, "_create", create)
    result = _init(tmp_path)
    assert result.exit_code == 1 and "already exists" in result.stderr
    assert victim.read_text() == "keep me" and late.is_symlink()
    assert not (tmp_path / "fixtures" / "lab" / "customer.yaml").exists()
    assert not (tmp_path / "fixtures" / "lab" / "users.yaml").exists()


def test_init_experiment_rollback_spares_a_file_that_replaced_one_of_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier file is replaced by another process, then a later file appears: the
    rollback removes only files that are still the ones this call created."""
    from agentgate import experiment

    customer = tmp_path / "fixtures" / "lab" / "customer.yaml"
    users = tmp_path / "fixtures" / "lab" / "users.yaml"
    late = tmp_path / "policies" / "lab" / "policy.yaml"
    real_create = experiment._create

    def create(path: Path, text: str) -> tuple[int, int]:
        if path == late:
            customer.unlink()
            customer.write_text("someone else's")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("late arrival")
        return real_create(path, text)

    monkeypatch.setattr(experiment, "_create", create)
    result = _init(tmp_path)
    assert result.exit_code == 1 and "already exists" in result.stderr
    assert customer.read_text() == "someone else's" and late.read_text() == "late arrival"
    assert not users.exists()


def test_init_experiment_server_features_are_opt_in(tmp_path: Path) -> None:
    assert _init(tmp_path / "off").exit_code == 0
    assert _init(tmp_path / "on", "--allow-server-features").exit_code == 0
    off = yaml.safe_load((tmp_path / "off" / "policies" / "lab" / "policy.yaml").read_text())
    on = yaml.safe_load((tmp_path / "on" / "policies" / "lab" / "policy.yaml").read_text())
    assert "allow_server_features" not in off["models"]["allow"][0]
    assert on["models"]["allow"][0]["allow_server_features"] is True
    bundle = load_bundle(tmp_path / "on" / "policies" / "lab")
    assert bundle.policy.models is not None and bundle.policy.models.allow[0].allow_server_features
