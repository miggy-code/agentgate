"""Red-team report formatting, run metadata and the baseline check."""

import dataclasses
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from agentgate.redteam.baseline import (
    DEFAULT_BASELINE,
    Baseline,
    BaselineError,
    baseline_from,
    check_against_baseline,
    load_baseline,
    write_baseline,
)
from agentgate.redteam.core import Category, Check, ScenarioResult, SuiteResult
from agentgate.redteam.report import (
    IN_PROCESS_NOTE,
    RunMeta,
    collect_meta,
    format_rate,
    git_commit,
    to_json,
    to_markdown,
)
from agentgate.redteam.transport import Transport
from conftest import ROOT

META = RunMeta(
    command="agentgate redteam",
    commit="abc1234",
    python="3.12.0",
    platform="TestOS-1.0",
    machine="x86_64",
    cpu_count=8,
    agentgate_version="0.0.1",
    timestamp="2026-10-22T15:00:00Z",
)


def result(
    scenario_id: str,
    category: Category,
    passed: bool = True,
    *,
    checks: tuple[Check, ...] | None = None,
    leaks: tuple[str, ...] = (),
    error: str | None = None,
) -> ScenarioResult:
    if checks is None:
        checks = (Check("first", True), Check("second", passed))
    return ScenarioResult(
        id=scenario_id,
        category=category,
        title=f"Title of {scenario_id}",
        benign=category is Category.BENIGN,
        passed=passed,
        checks=checks,
        leaks=leaks,
        error=error,
    )


def suite(*results: ScenarioResult) -> SuiteResult:
    """A SuiteResult with totals computed the way run_suite computes them."""
    ordered = tuple(sorted(results, key=lambda r: r.id))
    totals: dict[Category, tuple[int, int]] = {}
    for category in Category:
        of_category = [r for r in ordered if r.category is category]
        if of_category:
            totals[category] = (sum(r.passed for r in of_category), len(of_category))
    benign = [r for r in ordered if r.benign]
    rate = sum(r.passed for r in benign) / len(benign) if benign else None
    return SuiteResult(results=ordered, totals=totals, benign_rate=rate)


PASSING = suite(
    result("xt-001", Category.CROSS_TENANT),
    result("xt-002", Category.CROSS_TENANT),
    result("pe-001", Category.PRIVILEGE_ESCALATION),
    result("bn-001", Category.BENIGN),
    result("bn-002", Category.BENIGN),
)


def table(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("|")]


# Rates


@pytest.mark.parametrize(
    ("passed", "total", "expected"),
    [
        (14, 14, "14/14 (100%)"),
        (1, 1, "1/1 (100%)"),
        (1999, 2000, "1999/2000 (99.9%)"),
        (19999, 20000, "19999/20000 (99.9%)"),
        (2, 3, "2/3 (66.6%)"),
        (1, 3, "1/3 (33.3%)"),
        (1, 2, "1/2 (50.0%)"),
        (0, 3, "0/3 (0.0%)"),
        (0, 0, "0/0 (n/a)"),
    ],
)
def test_format_rate(passed: int, total: int, expected: str) -> None:
    assert format_rate(passed, total) == expected


def test_a_rate_short_of_complete_never_reads_100() -> None:
    for total in range(1, 3000, 7):
        assert "100%" not in format_rate(total - 1, total)
        assert format_rate(total, total).endswith("(100%)")


# Markdown


def test_markdown_table_in_category_order_with_attack_total() -> None:
    markdown = to_markdown(PASSING, META)
    assert table(markdown) == [
        "| Category | Scenarios | Passed | Pass rate |",
        "| --- | ---: | ---: | ---: |",
        "| cross_tenant | 2 | 2 | 2/2 (100%) |",
        "| privilege_escalation | 1 | 1 | 1/1 (100%) |",
        "| **attacks total** | 3 | 3 | 3/3 (100%) |",
        "| benign_control | 2 | 2 | 2/2 (100%) |",
    ]
    assert "Benign control rate: 2/2 (100%)." in markdown
    assert markdown.endswith("## Failures\n\nNone.\n")


def test_markdown_states_how_it_was_produced() -> None:
    markdown = to_markdown(PASSING, META)
    assert "- Command: `agentgate redteam`" in markdown
    assert "- Commit: `abc1234`" in markdown
    assert "- agentgate 0.0.1, Python 3.12.0" in markdown
    assert "- Platform: TestOS-1.0, machine x86_64, 8 CPUs" in markdown
    assert "- Generated: 2026-10-22T15:00:00Z" in markdown


def test_markdown_with_failures() -> None:
    attacks = [result(f"xt-{n:03}", Category.CROSS_TENANT) for n in range(1, 2000)]
    failed = result(
        "xt-999",
        Category.CROSS_TENANT,
        passed=False,
        checks=(Check("denied", True), Check("outbox empty", False, "1 emails")),
        leaks=("run#1 kc-staff-nadine saw AGCANARY-000000000000 (kc-payroll-summary)",),
    )
    raised = result("pe-001", Category.PRIVILEGE_ESCALATION, passed=False, checks=(), error="E")
    benign = [result("bn-001", Category.BENIGN), result("bn-002", Category.BENIGN, passed=False)]
    attacks = [a for a in attacks if a.id != "xt-999"]
    outcome = suite(*attacks, failed, raised, *benign)
    markdown = to_markdown(outcome, META)

    assert table(markdown)[2:] == [
        "| cross_tenant | 1999 | 1998 | 1998/1999 (99.9%) |",
        "| privilege_escalation | 1 | 0 | 0/1 (0.0%) |",
        "| **attacks total** | 2000 | 1998 | 1998/2000 (99.9%) |",
        "| benign_control | 2 | 1 | 1/2 (50.0%) |",
    ]
    assert "Benign control rate: 1/2 (50.0%)." in markdown
    failures = markdown.split("## Failures\n\n", 1)[1]
    assert failures.splitlines() == [
        "- `bn-002` (benign_control) Title of bn-002",
        "  - failed checks: second",
        "- `pe-001` (privilege_escalation) Title of pe-001",
        "  - no checks recorded",
        "  - error: E",
        "- `xt-999` (cross_tenant) Title of xt-999",
        "  - failed checks: outbox empty",
        "  - leak: run#1 kc-staff-nadine saw AGCANARY-000000000000 (kc-payroll-summary)",
    ]


def test_markdown_without_benign_or_attack_scenarios() -> None:
    only_benign = to_markdown(suite(result("bn-001", Category.BENIGN)), META)
    assert "| **attacks total** | 0 | 0 | 0/0 (n/a) |" in only_benign
    only_attacks = to_markdown(suite(result("pe-001", Category.PRIVILEGE_ESCALATION)), META)
    assert "Benign control rate: no benign scenarios ran." in only_attacks
    assert "benign_control |" not in only_attacks


def test_reports_have_no_em_dashes() -> None:
    assert "\u2014" not in to_markdown(PASSING, META)
    assert "\u2014" not in json.dumps(to_json(PASSING, META), ensure_ascii=False)


# JSON


def test_json_round_trips_and_keeps_every_scenario() -> None:
    failed = result(
        "pe-002", Category.PRIVILEGE_ESCALATION, passed=False, leaks=("leak one",), error="E"
    )
    outcome = suite(*PASSING.results, failed)
    data = to_json(outcome, META)
    assert json.loads(json.dumps(data)) == data
    assert RunMeta.from_json(data["meta"]) == META
    assert data["summary"] == {
        "all_passed": False,
        "scenarios": 6,
        "passed": 5,
        "attacks": {"scenarios": 4, "passed": 3},
        "benign": {"scenarios": 2, "passed": 2, "rate": 1.0},
        "categories": [
            {"category": "cross_tenant", "scenarios": 2, "passed": 2},
            {"category": "privilege_escalation", "scenarios": 2, "passed": 1},
            {"category": "benign_control", "scenarios": 2, "passed": 2},
        ],
    }
    scenarios = data["scenarios"]
    assert isinstance(scenarios, list)
    assert [s["id"] for s in scenarios] == [r.id for r in outcome.results]
    assert scenarios[3] == {
        "id": "pe-002",
        "category": "privilege_escalation",
        "title": "Title of pe-002",
        "passed": False,
        "checks": [{"name": "first", "passed": True}, {"name": "second", "passed": False}],
        "leaks": ["leak one"],
        "error": "E",
    }


# Run metadata


def test_collect_meta_describes_this_process() -> None:
    meta = collect_meta("agentgate redteam --verbose")
    assert meta.command == "agentgate redteam --verbose"
    assert re.fullmatch(r"[0-9a-f]{4,40}(-dirty)?|unknown", meta.commit)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta.timestamp)
    assert meta.python.startswith("3.")
    assert meta.agentgate_version


def test_git_commit_outside_a_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    assert git_commit(tmp_path) == "unknown"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_commit_marks_a_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    git("init", "-q")
    (tmp_path / "a.txt").write_text("a")
    git("add", "a.txt")
    git("commit", "-q", "-m", "init")
    sha = git("rev-parse", "--short", "HEAD")
    assert git_commit(tmp_path) == sha
    (tmp_path / "new.txt").write_text("untracked")
    assert git_commit(tmp_path) == f"{sha}-dirty"


# Baseline

BASELINE = Baseline(
    scenarios={
        Category.CROSS_TENANT: ("xt-001", "xt-002"),
        Category.PRIVILEGE_ESCALATION: ("pe-001",),
        Category.BENIGN: ("bn-001", "bn-002"),
    }
)


def test_baseline_matches_a_passing_run() -> None:
    assert baseline_from(PASSING) == BASELINE
    assert check_against_baseline(PASSING, BASELINE) == []


def test_baseline_scenario_missing() -> None:
    renamed = suite(*(r for r in PASSING.results if r.id != "xt-002"))
    assert check_against_baseline(renamed, BASELINE) == [
        "xt-002 (cross_tenant) is in the baseline but did not run: deleted or renamed"
    ]


def test_baseline_scenario_failing() -> None:
    results = [r for r in PASSING.results if r.id != "pe-001"]
    failing = suite(*results, result("pe-001", Category.PRIVILEGE_ESCALATION, passed=False))
    assert check_against_baseline(failing, BASELINE) == [
        "pe-001 (privilege_escalation) is in the baseline and failed"
    ]


def test_new_scenario_failing() -> None:
    grown = suite(*PASSING.results, result("ab-001", Category.APPROVAL_BYPASS, passed=False))
    assert check_against_baseline(grown, BASELINE) == ["ab-001 (approval_bypass) is new and failed"]


def test_new_scenario_passing_is_fine() -> None:
    grown = suite(*PASSING.results, result("ab-001", Category.APPROVAL_BYPASS))
    assert check_against_baseline(grown, BASELINE) == []


def test_baseline_check_limited_to_the_categories_that_ran() -> None:
    benign_only = suite(*(r for r in PASSING.results if r.benign))
    assert check_against_baseline(benign_only, BASELINE, categories=[Category.BENIGN]) == []
    assert len(check_against_baseline(benign_only, BASELINE)) == 3


def test_baseline_refuses_a_failing_run() -> None:
    failing = suite(*PASSING.results, result("pe-002", Category.PRIVILEGE_ESCALATION, False))
    with pytest.raises(BaselineError, match="pe-002"):
        baseline_from(failing)


def test_baseline_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "redteam" / "baseline.json"
    write_baseline(path, BASELINE)
    assert load_baseline(path) == BASELINE
    data = json.loads(path.read_text())
    assert data["format"] == 1
    assert data["categories"] == {
        "cross_tenant": ["xt-001", "xt-002"],
        "privilege_escalation": ["pe-001"],
        "benign_control": ["bn-001", "bn-002"],
    }
    # No timestamp or commit: the file changes only when the scenarios do.
    first = path.read_text()
    write_baseline(path, BASELINE)
    assert path.read_text() == first


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"categories": {}}', "not a format 1"),
        ('{"format": 1, "categories": []}', "categories must be an object"),
        ('{"format": 1, "categories": {"nope": []}}', "unknown category 'nope'"),
        ('{"format": 1, "categories": {"cross_tenant": [1]}}', "list of scenario ids"),
    ],
)
def test_malformed_baseline(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(text)
    with pytest.raises(BaselineError, match=message):
        load_baseline(path)


def test_missing_baseline_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "missing.json")


def test_repository_baseline_is_well_formed() -> None:
    path = ROOT / DEFAULT_BASELINE
    if not path.is_file():
        pytest.skip("redteam/baseline.json has not been written yet")
    baseline = load_baseline(path)
    assert baseline.ids
    for category, ids in baseline.scenarios.items():
        assert all(i.startswith(category.prefix + "-") for i in ids)


# Transports


def over_mcp(
    base: SuiteResult, calls: dict[str, tuple[int, int, tuple[tuple[str, int], ...]]]
) -> SuiteResult:
    """The suite as an mcp run, with (mcp, direct, direct by reason) calls per scenario id."""
    results = tuple(
        dataclasses.replace(r, mcp_calls=m, direct_calls=d, direct_by_reason=reasons)
        for r in base.results
        for m, d, reasons in [calls.get(r.id, (0, 0, ()))]
    )
    return dataclasses.replace(base, results=results, transport=Transport.MCP)


MCP_PASSING = over_mcp(
    PASSING,
    {
        "xt-001": (4, 1, (("forged_context", 1),)),
        "xt-002": (0, 2, (("action_override", 2),)),
        "pe-001": (3, 0, ()),
        "bn-001": (5, 0, ()),
        "bn-002": (2, 0, ()),
    },
)


def test_the_header_names_the_transport() -> None:
    gate = to_markdown(PASSING, META)
    assert gate.startswith("# agentgate red-team report (transport: gate)\n")
    assert "- Transport: `gate`, every call goes to the gate in-process" in gate
    assert "## MCP transport" not in gate
    assert "calls over MCP" not in gate
    mcp = to_markdown(MCP_PASSING, META)
    assert mcp.startswith("# agentgate red-team report (transport: mcp)\n")
    assert (
        "- Transport: `mcp`, agent-side tool calls go through an MCP client session to the"
        " gateway" in mcp
    )


def test_mcp_markdown_counts_calls_and_coverage() -> None:
    markdown = to_markdown(MCP_PASSING, META)
    section = markdown.split("## MCP transport\n\n", 1)[1].split("## Failures", 1)[0]
    lines = section.splitlines()
    assert lines[0] == "calls over MCP: 14, direct: 3"
    assert lines[2].startswith("Direct calls by reason: action_override 2, forged_context 1.")
    assert [line for line in lines if line.startswith("|")] == [
        "| Category | Scenarios | With an MCP call | Calls over MCP | Direct calls |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| cross_tenant | 2 | 1 | 4 | 3 |",
        "| privilege_escalation | 1 | 1 | 3 | 0 |",
        "| benign_control | 2 | 2 | 7 | 0 |",
    ]
    assert IN_PROCESS_NOTE in section
    # The pass-rate table is unchanged, and failures still come last.
    assert table(to_markdown(PASSING, META)) == table(markdown)[:6]
    assert markdown.endswith("## Failures\n\nNone.\n")


def test_mcp_json_counts_calls_per_scenario_and_category() -> None:
    data = to_json(MCP_PASSING, META)
    assert json.loads(json.dumps(data)) == data
    assert data["transport"] == "mcp"
    assert data["mcp"] == {
        "calls_over_mcp": 14,
        "direct_calls": 3,
        "direct_by_reason": {"action_override": 2, "forged_context": 1},
        "categories": [
            {
                "category": "cross_tenant",
                "scenarios": 2,
                "with_mcp_call": 1,
                "mcp_calls": 4,
                "direct_calls": 3,
            },
            {
                "category": "privilege_escalation",
                "scenarios": 1,
                "with_mcp_call": 1,
                "mcp_calls": 3,
                "direct_calls": 0,
            },
            {
                "category": "benign_control",
                "scenarios": 2,
                "with_mcp_call": 2,
                "mcp_calls": 7,
                "direct_calls": 0,
            },
        ],
        "in_process": IN_PROCESS_NOTE,
    }
    scenarios = data["scenarios"]
    assert isinstance(scenarios, list)
    by_id = {s["id"]: s for s in scenarios if isinstance(s, dict)}
    assert by_id["xt-002"]["calls"] == {
        "mcp": 0,
        "direct": 2,
        "direct_by_reason": {"action_override": 2},
    }
    gate = to_json(PASSING, META)
    assert gate["transport"] == "gate"
    assert "mcp" not in gate
    assert all("calls" not in s for s in gate["scenarios"] if isinstance(s, dict))


def test_the_baseline_applies_to_the_mcp_transport() -> None:
    assert check_against_baseline(MCP_PASSING, BASELINE) == []
    assert baseline_from(MCP_PASSING) == BASELINE
    results = [r for r in MCP_PASSING.results if r.id != "pe-001"]
    failing = dataclasses.replace(
        suite(*results, result("pe-001", Category.PRIVILEGE_ESCALATION, passed=False)),
        transport=Transport.MCP,
    )
    missing = dataclasses.replace(
        suite(*(r for r in PASSING.results if r.id != "xt-002")), transport=Transport.MCP
    )
    assert check_against_baseline(failing, BASELINE) == [
        "pe-001 (privilege_escalation) is in the baseline and failed (transport mcp)"
    ]
    assert check_against_baseline(missing, BASELINE) == [
        "xt-002 (cross_tenant) is in the baseline but did not run: deleted or renamed"
        " (transport mcp)"
    ]
