"""`agentgate redteam` and `agentgate bench`: the red-team suite and the latency bench."""

import json
import shlex
import sqlite3
from pathlib import Path
from typing import Annotated

import typer

from agentgate.cli._shared import (
    FixturesOption,
    PoliciesOption,
    fail,
)
from agentgate.redteam import bench as redteam_bench
from agentgate.redteam import report as redteam_report
from agentgate.redteam.baseline import (
    DEFAULT_BASELINE,
    BaselineError,
    baseline_from,
    check_against_baseline,
    load_baseline,
    write_baseline,
)
from agentgate.redteam.core import Category, SuiteResult, run_suite
from agentgate.redteam.transport import Transport
from agentgate.world import (
    DEFAULT_FIXTURES,
    DEFAULT_POLICIES,
)

DEFAULT_REDTEAM_JSON = Path("reports/redteam.json")


DEFAULT_REDTEAM_MD = Path("reports/redteam.md")


# Where a run over each transport writes its reports unless --json and --md say otherwise.
REDTEAM_REPORTS: dict[Transport, tuple[Path, Path]] = {
    Transport.GATE: (DEFAULT_REDTEAM_JSON, DEFAULT_REDTEAM_MD),
    Transport.MCP: (Path("reports/redteam-mcp.json"), Path("reports/redteam-mcp.md")),
}


DEFAULT_BENCH_JSON = Path("reports/bench.json")


DEFAULT_BENCH_MD = Path("reports/bench.md")


JsonOption = Annotated[Path, typer.Option("--json", help="Write the JSON report here.")]


MdOption = Annotated[Path, typer.Option("--md", help="Write the markdown report here.")]


def _command(name: str, *options: tuple[str, object | None, object]) -> str:
    """The command as typed, from its options: each (flag, value, default) whose value is
    not the default. A True value is a bare flag; a list repeats the flag."""
    parts = ["agentgate", name]
    for flag, value, default in options:
        if value is None or value == default:
            continue
        if value is True:
            parts.append(flag)
        elif isinstance(value, list):
            for item in value:
                parts += [flag, str(item)]
        else:
            parts += [flag, str(value)]
    return shlex.join(parts)


def _write_reports(json_path: Path, md_path: Path, data: object, markdown: str) -> None:
    try:
        redteam_report.write_text(json_path, json.dumps(data, indent=2, ensure_ascii=True) + "\n")
        redteam_report.write_text(md_path, markdown)
    except OSError as exc:
        fail(str(exc))
    typer.echo(f"reports written: {json_path}, {md_path}")


def _scenario_lines(result: SuiteResult) -> list[str]:
    lines: list[str] = []
    for scenario in result.results:
        mark = "ok  " if scenario.passed else "FAIL"
        lines.append(f"{mark} {scenario.id}  {scenario.title}")
        for check in scenario.checks:
            if not check.passed:
                detail = f": {check.detail}" if check.detail else ""
                lines.append(f"       check failed: {check.name}{detail}")
        lines += [f"       leak: {leak}" for leak in scenario.leaks]
        if scenario.error is not None:
            lines.append(f"       error: {scenario.error}")
    return lines


def redteam(
    category: Annotated[
        list[Category] | None,
        typer.Option("--category", help="Only this category; repeat for more."),
    ] = None,
    transport: Annotated[
        Transport,
        typer.Option(
            "--transport",
            help=(
                "gate: call the gate in-process. mcp: send every agent-side tool call through"
                " an MCP client session to the gateway."
            ),
        ),
    ] = Transport.GATE,
    json_path: Annotated[
        Path | None,
        typer.Option(
            "--json",
            help="Write the JSON report here. (default: reports/redteam.json, or"
            " reports/redteam-mcp.json with --transport mcp)",
            show_default=False,
        ),
    ] = None,
    md_path: Annotated[
        Path | None,
        typer.Option(
            "--md",
            help="Write the markdown report here. (default: reports/redteam.md, or"
            " reports/redteam-mcp.md with --transport mcp)",
            show_default=False,
        ),
    ] = None,
    baseline: Annotated[
        Path, typer.Option("--baseline", help="Scenarios that must exist and pass.")
    ] = DEFAULT_BASELINE,
    write: Annotated[
        bool,
        typer.Option(
            "--write-baseline",
            help="Write the baseline from this run. Refused if any scenario failed.",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="List every scenario and its failed checks.")
    ] = False,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Run the red-team suite. Exits 1 if any scenario failed or the baseline check found
    problems. The baseline applies to both transports."""
    if write and category:
        fail("--write-baseline needs the whole suite; drop --category")
    default_json, default_md = REDTEAM_REPORTS[transport]
    command = _command(
        "redteam",
        ("--category", [c.value for c in category or []], []),
        ("--transport", transport.value, Transport.GATE.value),
        ("--json", json_path, None),
        ("--md", md_path, None),
        ("--baseline", baseline, DEFAULT_BASELINE),
        ("--write-baseline", write, False),
        ("--verbose", verbose, False),
        ("--policies", policies, DEFAULT_POLICIES),
        ("--fixtures", fixtures, DEFAULT_FIXTURES),
    )
    json_path = default_json if json_path is None else json_path
    md_path = default_md if md_path is None else md_path
    try:
        result = run_suite(
            categories=category or None,
            policies=policies,
            fixtures=fixtures,
            transport=transport,
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        fail(str(exc))
    meta = redteam_report.collect_meta(command)
    markdown = redteam_report.to_markdown(result, meta)
    if verbose:
        for line in _scenario_lines(result):
            typer.echo(line)
        typer.echo("")
    typer.echo(markdown, nl=False)
    typer.echo("")
    _write_reports(json_path, md_path, redteam_report.to_json(result, meta), markdown)

    if write:
        if not result.all_passed:
            fail(f"refusing to write {baseline}: {len(result.failed)} scenarios failed")
        _write_baseline(baseline, result)
        return

    problems = _baseline_problems(baseline, result, category)
    if not result.all_passed or problems:
        raise typer.Exit(code=1)


def _baseline_problems(
    path: Path, result: SuiteResult, categories: list[Category] | None
) -> list[str] | None:
    if not path.is_file():
        typer.echo(f"warning: baseline {path} not found; baseline check skipped", err=True)
        return None
    try:
        loaded = load_baseline(path)
    except (OSError, BaselineError) as exc:
        fail(str(exc))
    problems = check_against_baseline(result, loaded, categories=categories or None)
    for problem in problems:
        typer.echo(f"BASELINE {problem}")
    typer.echo(f"baseline {path}: {len(problems)} problems")
    return problems


def _write_baseline(path: Path, result: SuiteResult) -> None:
    previous = None
    if path.is_file():
        try:
            previous = load_baseline(path)
        except (OSError, BaselineError):
            previous = None
    written = baseline_from(result)
    try:
        write_baseline(path, written)
    except OSError as exc:
        fail(str(exc))
    if previous is not None:
        added = sorted(written.ids - previous.ids)
        removed = sorted(previous.ids - written.ids)
        typer.echo(f"baseline added: {', '.join(added) or 'none'}")
        typer.echo(f"baseline removed: {', '.join(removed) or 'none'}")
    typer.echo(f"baseline written: {path}  {len(written.ids)} scenarios")


def bench(
    iterations: Annotated[
        int, typer.Option("--iterations", min=1, help="Timed calls per subject.")
    ] = redteam_bench.DEFAULT_ITERATIONS,
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Untimed calls per subject first.")
    ] = redteam_bench.DEFAULT_WARMUP,
    busy_day: Annotated[
        int,
        typer.Option(
            "--busy-day",
            min=0,
            help="Reservations already in the customer day for the busy-day section; 0 skips.",
        ),
    ] = redteam_bench.DEFAULT_BUSY_DAY,
    json_path: JsonOption = DEFAULT_BENCH_JSON,
    md_path: MdOption = DEFAULT_BENCH_MD,
    policies: PoliciesOption = DEFAULT_POLICIES,
    fixtures: FixturesOption = DEFAULT_FIXTURES,
) -> None:
    """Measure the latency the gate adds to a tool call. Times are machine-dependent."""
    command = _command(
        "bench",
        ("--iterations", iterations, redteam_bench.DEFAULT_ITERATIONS),
        ("--warmup", warmup, redteam_bench.DEFAULT_WARMUP),
        ("--busy-day", busy_day, redteam_bench.DEFAULT_BUSY_DAY),
        ("--json", json_path, DEFAULT_BENCH_JSON),
        ("--md", md_path, DEFAULT_BENCH_MD),
        ("--policies", policies, DEFAULT_POLICIES),
        ("--fixtures", fixtures, DEFAULT_FIXTURES),
    )
    try:
        result = redteam_bench.run_bench(
            iterations=iterations,
            warmup=warmup,
            busy_day=busy_day,
            policies=policies,
            fixtures=fixtures,
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        fail(str(exc))
    meta = redteam_report.collect_meta(command)
    markdown = redteam_bench.to_markdown(result, meta)
    typer.echo(markdown, nl=False)
    typer.echo("")
    _write_reports(json_path, md_path, redteam_bench.to_json(result, meta), markdown)
