"""The red-team baseline: which scenarios must exist and pass, so none quietly disappears.

`redteam/baseline.json` lists, per category, the sorted ids of every scenario in the suite
when the baseline was written (`agentgate redteam --write-baseline`, which refuses while any
scenario fails). A later run is checked against it: a baseline scenario that did not run
(deleted or renamed), a baseline scenario that failed and a new scenario that failed are
problems. New scenarios that pass are fine; write the baseline again to make them required.
The same baseline applies to every transport: a run over `mcp` must have the same scenarios
and pass them, and its problems name the transport.

The file holds no timestamp or commit, so it changes only when the set of scenarios does.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from agentgate.redteam.core import Category, SuiteResult
from agentgate.redteam.transport import Transport

DEFAULT_BASELINE = Path("redteam/baseline.json")
BASELINE_FORMAT = 1
WRITE_COMMAND = "agentgate redteam --write-baseline"


class BaselineError(ValueError):
    """The baseline file is missing pieces or is not a baseline at all."""


@dataclass(frozen=True)
class Baseline:
    # Sorted scenario ids per category, in Category order; only categories with scenarios.
    scenarios: Mapping[Category, tuple[str, ...]]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(i for ids in self.scenarios.values() for i in ids)


def baseline_from(result: SuiteResult) -> Baseline:
    """The baseline for a suite run in which every scenario passed."""
    if not result.all_passed:
        failed = ", ".join(r.id for r in result.failed)
        raise BaselineError(f"cannot write a baseline while scenarios fail: {failed}")
    scenarios: dict[Category, tuple[str, ...]] = {}
    for category in Category:
        ids = sorted(r.id for r in result.results if r.category is category)
        if ids:
            scenarios[category] = tuple(ids)
    return Baseline(scenarios=scenarios)


def baseline_json(baseline: Baseline) -> dict[str, JsonValue]:
    categories: dict[str, JsonValue] = {
        category.value: list(ids) for category, ids in baseline.scenarios.items()
    }
    return {
        "format": BASELINE_FORMAT,
        "about": (
            "Red-team scenarios that must exist and pass, per category. Written by"
            f" `{WRITE_COMMAND}`; checked by `agentgate redteam`."
        ),
        "categories": categories,
    }


def write_baseline(path: Path, baseline: Baseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(baseline_json(baseline), indent=2, ensure_ascii=True)
    path.write_text(text + "\n", encoding="utf-8")


def load_baseline(path: Path) -> Baseline:
    """Read a baseline file. Raises FileNotFoundError, or BaselineError if malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: not valid JSON") from exc
    if not isinstance(data, dict) or data.get("format") != BASELINE_FORMAT:
        raise BaselineError(f"{path}: not a format {BASELINE_FORMAT} red-team baseline")
    raw = data.get("categories")
    if not isinstance(raw, dict):
        raise BaselineError(f"{path}: categories must be an object")
    scenarios: dict[Category, tuple[str, ...]] = {}
    for name, ids in raw.items():
        try:
            category = Category(name)
        except ValueError:
            raise BaselineError(f"{path}: unknown category {name!r}") from None
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise BaselineError(f"{path}: {name} must be a list of scenario ids")
        scenarios[category] = tuple(sorted(ids))
    ordered = {c: scenarios[c] for c in Category if c in scenarios}
    return Baseline(scenarios=ordered)


def check_against_baseline(
    result: SuiteResult,
    baseline: Baseline,
    *,
    categories: Iterable[Category] | None = None,
) -> list[str]:
    """Problems with this run against the baseline, or an empty list.

    With categories, only the baseline's entries for those categories must have run, for a
    run limited with --category. Problems of a run over any transport but `gate` end with
    the transport, as `(transport mcp)`.
    """
    wanted = None if categories is None else {Category(c) for c in categories}
    by_id = {r.id: r for r in result.results}
    problems: list[str] = []
    for category, ids in baseline.scenarios.items():
        if wanted is not None and category not in wanted:
            continue
        for scenario_id in ids:
            found = by_id.get(scenario_id)
            if found is None:
                problems.append(
                    f"{scenario_id} ({category.value}) is in the baseline but did not run:"
                    " deleted or renamed"
                )
            elif not found.passed:
                problems.append(f"{scenario_id} ({category.value}) is in the baseline and failed")
    required = baseline.ids
    for r in result.results:
        if r.id not in required and not r.passed:
            problems.append(f"{r.id} ({r.category.value}) is new and failed")
    if result.transport is not Transport.GATE:
        return [f"{problem} (transport {result.transport.value})" for problem in problems]
    return problems
