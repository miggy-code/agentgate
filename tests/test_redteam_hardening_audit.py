"""The audit hardening red-team scenarios (ai-013, ai-014).

1. They pass on the real gate, over both transports.
2. Mutation tests: each protection they target is removed by editing the source of the one
   function that holds it, and the scenarios that target it must then fail.
"""

import inspect
import textwrap
from dataclasses import dataclass
from typing import Any

import pytest

import agentgate.audit.panoptes as panoptes_module
from agentgate.gate import Gate
from agentgate.redteam.core import (
    CATEGORY_PREFIXES,
    Category,
    ScenarioResult,
    all_scenarios,
    run_suite,
)
from agentgate.redteam.transport import Transport
from conftest import FIXTURES, POLICIES

MODULE = "agentgate.redteam.scenarios.hardening_audit"
IDS = ["ai-013", "ai-014"]


def run(ids: list[str], transport: Transport = Transport.GATE) -> list[ScenarioResult]:
    suite = run_suite(ids=ids, policies=POLICIES, fixtures=FIXTURES, transport=transport)
    return list(suite.results)


@pytest.mark.parametrize("transport", list(Transport))
def test_scenarios_pass(transport: Transport) -> None:
    results = run(IDS, transport)
    failing = [
        (r.id, r.error, r.leaks, [c.name for c in r.checks if not c.passed])
        for r in results
        if not r.passed
    ]
    assert failing == []
    assert [r.id for r in results] == IDS
    assert all(len(r.checks) >= 8 for r in results)
    if transport is Transport.MCP:
        # The agent-side calls really went over MCP.
        assert all(r.mcp_calls >= 3 for r in results)


def test_ids_titles_and_category() -> None:
    mine = [s for s in all_scenarios() if s.module == MODULE]
    assert sorted(s.id for s in mine) == IDS
    for s in mine:
        assert s.category is Category.AUDIT_INTEGRITY
        assert s.id.startswith(CATEGORY_PREFIXES[s.category] + "-")
        assert 10 <= len(s.title) <= 72
        assert s.title[0].isupper() and not s.title.endswith(".")


# Mutation tests

Edit = tuple[str, str]


@dataclass(frozen=True)
class Mutation:
    name: str
    owner: Any
    attribute: str
    edits: tuple[Edit, ...]
    breaks: tuple[str, ...]


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


MUTATIONS = (
    Mutation(
        "the gate attaches no references",
        Gate,
        "_emit",
        (("references=self._references(call, subject, unknown_subject),", "references={},"),),
        breaks=("ai-013", "ai-014"),
    ),
    Mutation(
        "the Panoptes sink stores caller values as given",
        panoptes_module,
        "to_panoptes_event",
        (("event = event.recorded()", "event = event"),),
        breaks=("ai-013", "ai-014"),
    ),
    Mutation(
        "a missing document id is treated as a known one",
        Gate,
        "_prepare",
        (("unknown_subject=facts.document_id,", ""),),
        breaks=("ai-013", "ai-014"),
    ),
    Mutation(
        "a missing approval id is treated as a known one",
        Gate,
        "execute_approved",
        (("unknown_subject=_text(approval_id),", ""),),
        breaks=("ai-013",),
    ),
    Mutation(
        "an unknown approver is treated as a known one",
        Gate,
        "_decide_approval",
        (('claimed=frozenset({"user_id"})', "claimed=frozenset()"),),
        breaks=("ai-014",),
    ),
    Mutation(
        "an unknown user at run start is treated as a known one",
        Gate,
        "start_run",
        (('claimed=frozenset({"customer_id", "user_id"}),', "claimed=frozenset(),"),),
        breaks=("ai-014",),
    ),
)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
@pytest.mark.parametrize("transport", list(Transport))
def test_removing_a_protection_fails_its_scenarios(
    mutation: Mutation, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mutation.owner,
        mutation.attribute,
        mutant(mutation.owner, mutation.attribute, mutation.edits),
    )
    results = run(list(mutation.breaks), transport)
    still_passing = [r.id for r in results if r.passed]
    assert still_passing == [], f"{mutation.name}: passed without their protection"


def test_every_scenario_has_a_mutation() -> None:
    covered = {scenario_id for m in MUTATIONS for scenario_id in m.breaks}
    assert covered == set(IDS)


def test_mutations_leave_the_real_gate_alone() -> None:
    assert all(r.passed for r in run(IDS))
    assert Gate._emit.__module__ == "agentgate.gate"
