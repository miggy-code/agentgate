"""The reconstruction hardening scenarios (agentgate.redteam.scenarios.hardening_reconstruct).

1. They pass under both transports, and under `mcp` their agent-side calls go over MCP.
2. Their ids are new: in no baseline and no other scenario module, with the ai prefix.
3. Mutation tests: each protection a scenario covers is removed, by replacing an attribute or
   recompiling one function from its own source with one edit, and the scenario must then
   fail. Every scenario in the module has at least one mutation.
"""

import inspect
import json
import textwrap
from dataclasses import dataclass
from typing import Any

import pytest

import agentgate.world as world_module
from agentgate.gate import Gate
from agentgate.redteam.core import (
    CATEGORY_PREFIXES,
    Category,
    ScenarioResult,
    all_scenarios,
    run_suite,
)
from agentgate.redteam.transport import Transport
from agentgate.store import PolicySnapshot, Store
from conftest import FIXTURES, POLICIES, ROOT

MODULE = "agentgate.redteam.scenarios.hardening_reconstruct"
IDS = ["ai-020", "ai-021", "ai-022", "ai-023", "ai-024"]


def run(ids: list[str], transport: Transport = Transport.GATE) -> dict[str, ScenarioResult]:
    suite = run_suite(ids=ids, policies=POLICIES, fixtures=FIXTURES, transport=transport)
    return {result.id: result for result in suite.results}


def _failures(result: ScenarioResult) -> list[str]:
    return [check.name for check in result.checks if not check.passed]


# 1. The scenarios pass


@pytest.mark.parametrize("transport", list(Transport))
def test_the_scenarios_pass(transport: Transport) -> None:
    results = run(IDS, transport)
    assert sorted(results) == IDS
    for result in results.values():
        assert result.passed, (result.id, result.error, result.leaks, _failures(result))
        assert len(result.checks) >= 5
        if transport is Transport.MCP:
            assert result.mcp_calls > 0 and result.direct_calls == 0, result.id
        else:
            assert result.mcp_calls == 0


def test_results_are_deterministic() -> None:
    assert run(IDS) == run(IDS)


# 2. The ids are new


def test_ids_are_registered_new_and_prefixed() -> None:
    scenarios = all_scenarios()
    mine = {s.id: s for s in scenarios if s.module == MODULE}
    assert sorted(mine) == IDS
    others = {s.id for s in scenarios if s.module != MODULE}
    assert others.isdisjoint(mine)
    baseline = json.loads((ROOT / "redteam" / "baseline.json").read_text())
    known = {scenario_id for ids in baseline["categories"].values() for scenario_id in ids}
    assert known.isdisjoint(mine)
    for scenario_id, registered in mine.items():
        assert registered.category is Category.AUDIT_INTEGRITY
        assert scenario_id.startswith(CATEGORY_PREFIXES[registered.category] + "-")
        assert 10 <= len(registered.title) <= 72
        assert registered.title[0].isupper() and not registered.title.endswith(".")


# 3. Mutations

Edit = tuple[str, str]


@dataclass(frozen=True)
class Mutation:
    name: str
    owner: Any
    attribute: str
    # Either source edits of the attribute, or a replacement for it.
    edits: tuple[Edit, ...] = ()
    replacement: Any = None
    breaks: tuple[str, ...] = ()


def mutant(owner: Any, attribute: str, edits: tuple[Edit, ...]) -> Any:
    """The attribute recompiled from its own source with each edit applied exactly once."""
    original = inspect.getattr_static(owner, attribute)
    source = textwrap.dedent(inspect.getsource(original))
    for old, new in edits:
        assert source.count(old) == 1, f"{attribute}: edit target found {source.count(old)} times"
        source = source.replace(old, new)
    namespace = dict(original.__globals__)
    exec(compile(source, f"<mutant {attribute}>", "exec"), namespace)
    return namespace[attribute]


def _unsaved(self: Store, snapshot: PolicySnapshot) -> tuple[PolicySnapshot, bool]:
    # Says it stored the snapshot, stores nothing.
    return snapshot, True


def _nothing(*args: object, **kwargs: object) -> None:
    return None


def _no_problems(*args: object, **kwargs: object) -> list[object]:
    return []


def _no_values(*args: object, **kwargs: object) -> dict[str, object]:
    return {}


MUTATIONS = (
    Mutation(
        "the approval request carries no approval hash",
        Gate,
        "call_tool",
        edits=(('"approval_args_hash": created.args_hash,', ""),),
        breaks=("ai-020",),
    ),
    Mutation(
        "check-store skips approval rows",
        world_module,
        "_approval_problems",
        replacement=_no_problems,
        breaks=("ai-020",),
    ),
    Mutation(
        "executed events name no side-effect row",
        Gate,
        "_finish",
        edits=(("if result.effect_ref is not None:", "if False:"),),
        breaks=("ai-021", "ai-024"),
    ),
    Mutation(
        "check-store skips side-effect rows",
        world_module,
        "_effect_problems",
        replacement=_no_problems,
        breaks=("ai-021",),
    ),
    Mutation(
        "policy snapshots are not stored",
        Store,
        "save_policy_snapshot",
        replacement=_unsaved,
        breaks=("ai-022",),
    ),
    Mutation(
        "policy loads are not recorded",
        Gate,
        "record_policy_loaded",
        replacement=_nothing,
        breaks=("ai-022",),
    ),
    Mutation(
        "operator actions are not recorded",
        Gate,
        "record_operator_action",
        replacement=_nothing,
        breaks=("ai-023",),
    ),
    Mutation(
        "the stale sweep records nothing",
        Gate,
        "release_stale",
        edits=(("if not record:", "if True:"),),
        breaks=("ai-023",),
    ),
    Mutation(
        "the fixture sync records nothing",
        world_module,
        "record_fixture_sync",
        replacement=_nothing,
        breaks=("ai-023",),
    ),
    Mutation(
        "decision events carry no values",
        Gate,
        "_call_values",
        replacement=_no_values,
        breaks=("ai-020", "ai-021", "ai-024"),
    ),
    Mutation(
        "allow events drop the call's values",
        Gate,
        "_record_allow",
        edits=(("details = {**prepared.values, **(details or {})}", "details = details or {}"),),
        breaks=("ai-024",),
    ),
)


def _apply(mutation: Mutation, monkeypatch: pytest.MonkeyPatch) -> None:
    replacement = (
        mutant(mutation.owner, mutation.attribute, mutation.edits)
        if mutation.edits
        else mutation.replacement
    )
    monkeypatch.setattr(mutation.owner, mutation.attribute, replacement)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_removing_a_protection_fails_its_scenarios(
    mutation: Mutation, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply(mutation, monkeypatch)
    results = run(list(mutation.breaks))
    still_passing = [r.id for r in results.values() if r.passed]
    assert still_passing == [], f"{mutation.name}: passed without their protection"
    for result in results.values():
        # A failed check, not only a crash, shows what the protection is for.
        assert _failures(result) or result.error is not None


def test_a_mutation_fails_over_mcp_too(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply(MUTATIONS[2], monkeypatch)
    result = run(["ai-021"], Transport.MCP)["ai-021"]
    assert not result.passed
    assert "a raised order is caught" in _failures(result)


def test_every_scenario_has_a_mutation() -> None:
    covered = {scenario_id for m in MUTATIONS for scenario_id in m.breaks}
    registered = {s.id for s in all_scenarios() if s.module == MODULE}
    assert covered == registered == set(IDS)


def test_mutations_leave_the_real_code_alone() -> None:
    assert all(r.passed for r in run(IDS).values())
    assert Gate._finish.__module__ == "agentgate.gate"
