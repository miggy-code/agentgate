"""The operational hardening scenarios (agentgate.redteam.scenarios.hardening_ops).

1. They pass under both transports, and under `mcp` their file-world calls really go over
   MCP.
2. Their ids are new: none is in the baseline, and each has its category's prefix.
3. Mutation tests: each fix a scenario covers is removed with monkeypatch, and the scenario
   must then fail. A scenario that still passed without its fix would be testing nothing.
"""

import json
import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import agentgate.store as store_module
import agentgate.world as world_module
from agentgate.fixtures import FixtureSync
from agentgate.models import User
from agentgate.redteam.core import CATEGORY_PREFIXES, ScenarioResult, all_scenarios, run_suite
from agentgate.redteam.transport import Transport
from agentgate.spend import SpendLedger
from agentgate.store import Store
from conftest import FIXTURES, POLICIES, ROOT

MODULE = "agentgate.redteam.scenarios.hardening_ops"
IDS = ["fc-015", "fc-016", "fc-017", "id-015", "id-016"]
# Scenarios that make agent-side calls on a file world, which must go over MCP under mcp.
OVER_MCP = {"fc-015", "id-015", "id-016"}


def run(ids: list[str], transport: Transport = Transport.GATE) -> dict[str, ScenarioResult]:
    suite = run_suite(ids=ids, policies=POLICIES, fixtures=FIXTURES, transport=transport)
    return {result.id: result for result in suite.results}


def _failures(result: ScenarioResult) -> list[str]:
    return [check.name for check in result.checks if not check.passed]


@pytest.fixture
def loose_umask() -> Iterator[None]:
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


# 1. The scenarios pass


@pytest.mark.parametrize("transport", [Transport.GATE, Transport.MCP])
def test_the_scenarios_pass(transport: Transport, loose_umask: None) -> None:
    results = run(IDS, transport)
    assert sorted(results) == IDS
    for result in results.values():
        assert result.passed, (result.id, result.error, result.leaks, _failures(result))
        assert result.checks


def test_file_world_calls_go_over_mcp() -> None:
    results = run(sorted(OVER_MCP), Transport.MCP)
    for scenario_id in OVER_MCP:
        result = results[scenario_id]
        assert result.mcp_calls > 0, scenario_id
        assert result.direct_calls == 0, (scenario_id, result.direct_by_reason)
    in_process = run(sorted(OVER_MCP))
    for scenario_id in OVER_MCP:
        assert in_process[scenario_id].mcp_calls == 0
        assert in_process[scenario_id].direct_calls == results[scenario_id].mcp_calls


def test_results_are_deterministic() -> None:
    assert run(IDS) == run(IDS)


# 2. The ids are new


def test_ids_are_registered_new_and_prefixed() -> None:
    registered = {s.id: s for s in all_scenarios() if s.module == MODULE}
    assert sorted(registered) == IDS
    baseline = json.loads((ROOT / "redteam" / "baseline.json").read_text())
    known = {scenario_id for ids in baseline["categories"].values() for scenario_id in ids}
    assert known.isdisjoint(registered)
    for scenario_id, registered_scenario in registered.items():
        assert scenario_id.startswith(CATEGORY_PREFIXES[registered_scenario.category] + "-")


# 3. Mutations


def _no_sweep(self: SpendLedger, older_than: object) -> int:
    return 0


def _no_late_settle(self: SpendLedger, *args: object) -> None:
    return None


def _flag_only(self: Store, user_id: str) -> User | None:
    # Records the operator flag but leaves the user active.
    with self.locked():
        self.connection.execute(
            "UPDATE users SET suspended_by_operator = 1 WHERE id = ?", (user_id,)
        )
    return self.get_user(user_id)


def _never_flagged(self: Store, user_id: str) -> bool:
    return False


def _no_sync(store: Store, root: object) -> FixtureSync:
    return FixtureSync()


def _nothing(*args: object, **kwargs: object) -> None:
    return None


def _no_prepare(path: object) -> None:
    return None


Mutation = tuple[str, object, str, Callable[..., Any]]

MUTATIONS: list[tuple[str, Mutation]] = [
    ("fc-015", ("no stale sweep", SpendLedger, "release_stale", _no_sweep)),
    ("fc-015", ("late settle dropped", SpendLedger, "_finish_reclaimed", _no_late_settle)),
    ("id-015", ("suspension leaves the user active", Store, "suspend_user", _flag_only)),
    ("id-015", ("sync ignores the operator flag", Store, "operator_suspended", _never_flagged)),
    ("id-016", ("no fixture sync on reopen", world_module, "sync_fixtures", _no_sync)),
    ("fc-016", ("audit log accepted as a store", store_module, "_refuse_audit_database", _nothing)),
    ("fc-016", ("paths not checked", world_module, "check_paths", _nothing)),
    ("fc-017", ("store file left to sqlite", store_module, "_prepare_file", _no_prepare)),
]


@pytest.mark.parametrize(
    ("scenario_id", "mutation"), MUTATIONS, ids=[f"{s}-{m[0]}" for s, m in MUTATIONS]
)
def test_each_scenario_fails_without_its_fix(
    scenario_id: str, mutation: Mutation, monkeypatch: pytest.MonkeyPatch, loose_umask: None
) -> None:
    _, owner, attribute, replacement = mutation
    assert run([scenario_id])[scenario_id].passed
    monkeypatch.setattr(owner, attribute, replacement)
    result = run([scenario_id])[scenario_id]
    assert not result.passed, scenario_id
    # A failed check, not a crash, shows what the fix protects.
    assert _failures(result) or result.error is not None


def test_a_suspension_mutation_fails_over_mcp_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Store, "suspend_user", _flag_only)
    result = run(["id-015"], Transport.MCP)["id-015"]
    assert not result.passed
    assert "docs.read after suspension" in _failures(result)
    assert "the other connection's live run is refused" in _failures(result)
