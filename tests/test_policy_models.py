"""The policy's models section: its schema, the engine's model decision, the version hash and
the model cases of tests.yaml."""

import copy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentgate.models import Customer, Role
from agentgate.policy.engine import ModelRequest, PolicyEngine, Reason
from agentgate.policy.schema import Effect, ModelProvider, PolicyError, load_bundle
from agentgate.policy.testing import load_cases, run_policy_tests
from agentgate.policy.version import bundle_version, canonical_text
from test_policy_testing import POLICY, TOOLS, write_fixtures

ENTRY: dict[str, Any] = {
    "id": "deepseek-flash",
    "provider": "openai_compatible",
    "model": "deepseek-flash",
    "roles": ["staff", "owner"],
    "max_output_tokens": 8000,
    "max_request_bytes": 120000,
}
MODELS: dict[str, Any] = {"allow": [ENTRY], "reserve": "computed"}
CUSTOMER = Customer(id="acme", name="Acme", domains=("acme.example",), timezone="America/Chicago")


def _bundle(tmp_path: Path, models: object = MODELS, **changes: Any) -> Path:
    policy = copy.deepcopy(POLICY) | changes
    if models is not None:
        policy["models"] = models
    directory = tmp_path / "acme"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "policy.yaml").write_text(yaml.safe_dump(policy))
    (directory / "tools.yaml").write_text(yaml.safe_dump(TOOLS))
    return directory


def _engine(tmp_path: Path, models: object = MODELS) -> PolicyEngine:
    return PolicyEngine(load_bundle(_bundle(tmp_path, models)), CUSTOMER)


def _request(**changes: Any) -> ModelRequest:
    fields: dict[str, Any] = {
        "customer_id": "acme",
        "role": Role.STAFF,
        "provider": ModelProvider.OPENAI_COMPATIBLE,
        "model": "deepseek-flash",
        "max_tokens": 1000,
        "n": None,
        "request_bytes": 5000,
    } | changes
    return ModelRequest(**fields)


# Schema


def test_a_valid_section_loads(tmp_path: Path) -> None:
    models = load_bundle(_bundle(tmp_path)).policy.models
    assert models is not None
    [entry] = models.allow
    assert entry.roles == {Role.STAFF, Role.OWNER}
    assert models.reserve == "computed"


def test_a_fixed_reserve_loads_as_a_decimal(tmp_path: Path) -> None:
    models = load_bundle(_bundle(tmp_path, {**MODELS, "reserve": "0.02"})).policy.models
    assert models is not None
    assert models.reserve == Decimal("0.02")


def test_a_policy_without_the_section_loads(tmp_path: Path) -> None:
    assert load_bundle(_bundle(tmp_path, None)).policy.models is None


@pytest.mark.parametrize(
    "models",
    [
        {"allow": []},
        {"allow": [{**ENTRY, "roles": []}]},
        {"allow": [{k: v for k, v in ENTRY.items() if k != "roles"}]},
        {"allow": [{k: v for k, v in ENTRY.items() if k != "max_output_tokens"}]},
        {"allow": [{k: v for k, v in ENTRY.items() if k != "max_request_bytes"}]},
        {"allow": [{**ENTRY, "max_output_tokens": 0}]},
        {"allow": [{**ENTRY, "max_request_bytes": 0}]},
        {"allow": [{**ENTRY, "provider": "mistral"}]},
        {"allow": [{**ENTRY, "model": "deepseek-*"}]},
        {"allow": [{**ENTRY, "model": ""}]},
        {"allow": [{**ENTRY, "roles": ["root"]}]},
        {"allow": [{**ENTRY, "regex": "deepseek.*"}]},
        {"allow": [ENTRY, ENTRY]},
        {"allow": [ENTRY, {**ENTRY, "id": "second"}]},
        {"allow": [ENTRY], "reserve": "0"},
        {"allow": [ENTRY], "reserve": "-1"},
        {"allow": [ENTRY], "reserve": "cheap"},
        {"allow": [ENTRY], "default": "allow"},
    ],
)
def test_a_bad_section_is_rejected(tmp_path: Path, models: object) -> None:
    with pytest.raises(PolicyError):
        load_bundle(_bundle(tmp_path, models))


# Engine


def test_a_request_inside_the_envelope_is_allowed(tmp_path: Path) -> None:
    decision = _engine(tmp_path).decide_model(_request())
    assert decision.effect is Effect.ALLOW
    assert decision.reason is Reason.MODEL_ALLOWED
    assert decision.rule_ids == ("deepseek-flash",)


def test_the_limits_themselves_are_allowed(tmp_path: Path) -> None:
    decision = _engine(tmp_path).decide_model(_request(max_tokens=8000, n=1, request_bytes=120000))
    assert decision.effect is Effect.ALLOW


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"model": "other"}, Reason.MODEL_NOT_ALLOWED),
        ({"model": "deepseek-flash "}, Reason.MODEL_NOT_ALLOWED),
        ({"model": "DeepSeek-Flash"}, Reason.MODEL_NOT_ALLOWED),
        ({"model": "deepseek-flash-2"}, Reason.MODEL_NOT_ALLOWED),
        ({"model": ""}, Reason.MODEL_NOT_ALLOWED),
        ({"provider": ModelProvider.OPENAI}, Reason.MODEL_NOT_ALLOWED),
        ({"customer_id": "beta"}, Reason.CUSTOMER_MISMATCH),
        ({"role": Role.CONTRACTOR}, Reason.MODEL_ROLE_DENIED),
        ({"role": Role.MANAGER}, Reason.MODEL_ROLE_DENIED),
        ({"max_tokens": None}, Reason.MODEL_MAX_TOKENS_REQUIRED),
        ({"max_tokens": 0}, Reason.MODEL_MAX_TOKENS_REQUIRED),
        ({"max_tokens": -5}, Reason.MODEL_MAX_TOKENS_REQUIRED),
        ({"max_tokens": 8001}, Reason.MODEL_MAX_TOKENS_EXCEEDED),
        ({"n": 2}, Reason.MODEL_N_NOT_ONE),
        ({"n": 0}, Reason.MODEL_N_NOT_ONE),
        ({"n": -1}, Reason.MODEL_N_NOT_ONE),
        ({"request_bytes": 120001}, Reason.MODEL_REQUEST_TOO_LARGE),
    ],
)
def test_a_request_outside_the_entry_is_denied(
    tmp_path: Path, changes: dict[str, Any], reason: Reason
) -> None:
    decision = _engine(tmp_path).decide_model(_request(**changes))
    assert decision.effect is Effect.DENY
    assert decision.reason is reason


def test_a_policy_without_the_section_denies_every_model(tmp_path: Path) -> None:
    decision = _engine(tmp_path, None).decide_model(_request())
    assert (decision.effect, decision.reason) == (Effect.DENY, Reason.MODEL_NOT_ALLOWED)
    assert decision.rule_ids == ()


def test_each_entry_keeps_its_own_envelope(tmp_path: Path) -> None:
    other = {**ENTRY, "id": "small", "model": "small", "max_output_tokens": 10, "roles": ["owner"]}
    engine = _engine(tmp_path, {"allow": [ENTRY, other]})
    assert engine.decide_model(_request(model="small", role=Role.OWNER, max_tokens=10)).effect is (
        Effect.ALLOW
    )
    assert engine.decide_model(_request(model="small", role=Role.OWNER, max_tokens=11)).reason is (
        Reason.MODEL_MAX_TOKENS_EXCEEDED
    )
    assert engine.decide_model(_request(model="small", role=Role.STAFF)).reason is (
        Reason.MODEL_ROLE_DENIED
    )


# Version


def test_a_policy_without_the_section_keeps_its_earlier_canonical_form(tmp_path: Path) -> None:
    bundle = load_bundle(_bundle(tmp_path, None))
    assert '"models"' not in canonical_text(bundle)


@pytest.mark.parametrize(
    "models",
    [
        {**MODELS, "allow": [{**ENTRY, "max_output_tokens": 7999}]},
        {**MODELS, "allow": [{**ENTRY, "max_request_bytes": 1}]},
        {**MODELS, "allow": [{**ENTRY, "roles": ["staff"]}]},
        {**MODELS, "allow": [{**ENTRY, "model": "other"}]},
        {**MODELS, "allow": [{**ENTRY, "provider": "openai"}]},
        {**MODELS, "reserve": "0.02"},
        None,
    ],
)
def test_every_part_of_the_section_changes_the_version(tmp_path: Path, models: object) -> None:
    base = bundle_version(load_bundle(_bundle(tmp_path)))
    assert bundle_version(load_bundle(_bundle(tmp_path, models))) != base


def test_equal_sections_hash_equal_whatever_the_spelling(tmp_path: Path) -> None:
    first = bundle_version(load_bundle(_bundle(tmp_path, {**MODELS, "reserve": "0.020"})))
    second = bundle_version(load_bundle(_bundle(tmp_path, {**MODELS, "reserve": "0.02"})))
    reordered = {**MODELS, "allow": [{**ENTRY, "roles": ["owner", "staff"]}], "reserve": "0.02"}
    assert first == second == bundle_version(load_bundle(_bundle(tmp_path, reordered)))


# tests.yaml


def _cases(tmp_path: Path, cases: list[dict[str, Any]]) -> Path:
    path = tmp_path / "tests.yaml"
    path.write_text(yaml.safe_dump(cases))
    return path


MODEL_CASE: dict[str, Any] = {
    "name": "staff calls the model",
    "user": "acme-staff",
    "model": {"provider": "openai_compatible", "name": "deepseek-flash", "request_bytes": 10},
    "expect": "deny",
}


def test_a_model_case_loads(tmp_path: Path) -> None:
    [loaded] = load_cases(_cases(tmp_path, [MODEL_CASE | {"reason": "model_max_tokens_required"}]))
    assert loaded.model is not None
    assert loaded.model.max_tokens is None
    assert loaded.tool is None


@pytest.mark.parametrize(
    "changes",
    [
        {"tool": "docs.read", "action": "read"},
        {"document": "acme-handbook"},
        {"recipients": ["a@b.example"]},
        {"model": {"provider": "openai_compatible", "name": "x"}},
        {"model": {"provider": "nope", "name": "x", "request_bytes": 1}},
    ],
)
def test_a_malformed_model_case_is_rejected(tmp_path: Path, changes: dict[str, Any]) -> None:
    with pytest.raises(PolicyError):
        load_cases(_cases(tmp_path, [MODEL_CASE | changes]))


def test_a_case_with_neither_tool_nor_model_is_rejected(tmp_path: Path) -> None:
    bare = {k: v for k, v in MODEL_CASE.items() if k != "model"}
    with pytest.raises(PolicyError):
        load_cases(_cases(tmp_path, [bare]))


def _run(tmp_path: Path, models: object, cases: list[dict[str, Any]]) -> tuple[int, int]:
    policies = tmp_path / "policies"
    directory = _bundle(policies, models)
    (directory / "tests.yaml").write_text(yaml.safe_dump(cases))
    report = run_policy_tests(policies, write_fixtures(tmp_path / "fixtures", ("acme",)))
    return report.passed, report.failed


def test_model_cases_run_against_the_engine(tmp_path: Path) -> None:
    ok = MODEL_CASE | {"reason": "model_max_tokens_required", "rules": ["deepseek-flash"]}
    assert _run(tmp_path, MODELS, [ok]) == (1, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"expect": "allow"},
        {"reason": "model_allowed"},
        {"rules": []},
    ],
)
def test_a_model_case_that_disagrees_fails(tmp_path: Path, changes: dict[str, Any]) -> None:
    case = MODEL_CASE | {"reason": "model_max_tokens_required"} | changes
    assert _run(tmp_path, MODELS, [case]) == (0, 1)


def test_the_reason_is_checked_for_tool_cases_too(tmp_path: Path) -> None:
    tool_case = {
        "name": "read",
        "user": "acme-staff",
        "tool": "docs.read",
        "action": "read",
        "document": "acme-handbook",
        "expect": "allow",
    }
    assert _run(tmp_path, None, [tool_case | {"reason": "matched_allow"}]) == (1, 0)
    assert _run(tmp_path, None, [tool_case | {"reason": "default_deny"}]) == (0, 1)
