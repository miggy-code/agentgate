"""The red-team suite: scripted attacks against the gate, judged strictly, with a leak
detector run after every scenario. See agentgate.redteam.core for the pass criteria.
"""

from agentgate.redteam.core import (
    CATEGORY_PREFIXES,
    REDTEAM_START,
    Arena,
    Category,
    Check,
    RegisteredScenario,
    ScenarioFn,
    ScenarioResult,
    SuiteResult,
    all_scenarios,
    describe,
    returned_ids,
    run_scenario,
    run_suite,
    scenario,
    transcript_calls,
)

__all__ = [
    "CATEGORY_PREFIXES",
    "REDTEAM_START",
    "Arena",
    "Category",
    "Check",
    "RegisteredScenario",
    "ScenarioFn",
    "ScenarioResult",
    "SuiteResult",
    "all_scenarios",
    "describe",
    "returned_ids",
    "run_scenario",
    "run_suite",
    "scenario",
    "transcript_calls",
]
