"""Red-team reports: the suite's result as JSON and as markdown, with how it was produced.

Every number in the README comes from a run like this, so a report carries its RunMeta: the
exact command, the git commit (with `-dirty` when the tree had changes, or `unknown`), the
Python version, platform, machine, CPU count, agentgate version and a UTC timestamp.

Rates are printed as `14/14 (100%)` when every scenario passed and otherwise with one
decimal place, truncated rather than rounded, so 1999/2000 reads 99.9% and a rate is never
shown higher than it was. The attack total covers every category except benign_control,
whose rate is printed on its own line: it measures how often the gate gets out of the way,
not how often it stops an attack.
"""

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

import agentgate
from agentgate.redteam.core import Category, ScenarioResult, SuiteResult

_GIT_TIMEOUT_SECONDS = 5
UNKNOWN_COMMIT = "unknown"


@dataclass(frozen=True)
class RunMeta:
    command: str
    commit: str
    python: str
    platform: str
    machine: str
    cpu_count: int | None
    agentgate_version: str
    # ISO 8601 in UTC, to the second, with a Z suffix.
    timestamp: str

    def to_json(self) -> dict[str, JsonValue]:
        return dict(asdict(self))

    @classmethod
    def from_json(cls, data: dict[str, JsonValue]) -> "RunMeta":
        cpu_count = data["cpu_count"]
        return cls(
            command=str(data["command"]),
            commit=str(data["commit"]),
            python=str(data["python"]),
            platform=str(data["platform"]),
            machine=str(data["machine"]),
            cpu_count=cpu_count if isinstance(cpu_count, int) else None,
            agentgate_version=str(data["agentgate_version"]),
            timestamp=str(data["timestamp"]),
        )

    def lines(self) -> list[str]:
        """The provenance block both reports start with, as markdown lines."""
        cpus = "unknown" if self.cpu_count is None else str(self.cpu_count)
        return [
            f"- Command: `{self.command}`",
            f"- Commit: `{self.commit}`",
            f"- agentgate {self.agentgate_version}, Python {self.python}",
            f"- Platform: {self.platform}, machine {self.machine}, {cpus} CPUs",
            f"- Generated: {self.timestamp}",
        ]


def collect_meta(command: str, *, repo: Path | None = None) -> RunMeta:
    """Describe this process: the command it ran, its commit and where it ran."""
    return RunMeta(
        command=command,
        commit=git_commit(repo),
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine() or "unknown",
        cpu_count=os.cpu_count(),
        agentgate_version=agentgate.__version__,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def git_commit(repo: Path | None = None) -> str:
    """Short sha of HEAD, with `-dirty` if the tree has any change, or `unknown`.

    Untracked files count as changes: a new scenario module that is not committed yet is
    part of what ran. The default repository is the one holding the agentgate source.
    """
    cwd = Path(agentgate.__file__).resolve().parent if repo is None else repo
    sha = _git(cwd, "rev-parse", "--short", "HEAD")
    if sha is None or not sha.strip():
        return UNKNOWN_COMMIT
    status = _git(cwd, "status", "--porcelain")
    if status is None:
        return UNKNOWN_COMMIT
    return sha.strip() + ("-dirty" if status.strip() else "")


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def format_rate(passed: int, total: int) -> str:
    """`14/14 (100%)`, or one decimal place truncated toward zero, so never above the truth.

    A total of zero reads `0/0 (n/a)`.
    """
    if total <= 0:
        return f"{passed}/{total} (n/a)"
    if passed == total:
        return f"{passed}/{total} (100%)"
    tenths = passed * 1000 // total
    return f"{passed}/{total} ({tenths // 10}.{tenths % 10}%)"


def attack_totals(result: SuiteResult) -> tuple[int, int]:
    """(passed, total) over every attack scenario, that is every category but benign."""
    attacks = [r for r in result.results if not r.benign]
    return sum(1 for r in attacks if r.passed), len(attacks)


def benign_totals(result: SuiteResult) -> tuple[int, int]:
    benign = [r for r in result.results if r.benign]
    return sum(1 for r in benign if r.passed), len(benign)


def _ordered_totals(result: SuiteResult) -> list[tuple[Category, int, int]]:
    """(category, passed, total) in Category order: attacks first, benign last."""
    return [
        (category, *result.totals[category]) for category in Category if category in result.totals
    ]


def scenario_json(result: ScenarioResult) -> dict[str, JsonValue]:
    return {
        "id": result.id,
        "category": result.category.value,
        "title": result.title,
        "passed": result.passed,
        "checks": [{"name": check.name, "passed": check.passed} for check in result.checks],
        "leaks": list(result.leaks),
        "error": result.error,
    }


def to_json(result: SuiteResult, meta: RunMeta) -> dict[str, JsonValue]:
    attacks_passed, attacks_total = attack_totals(result)
    benign_passed, benign_total = benign_totals(result)
    categories: list[JsonValue] = [
        {"category": category.value, "scenarios": total, "passed": passed}
        for category, passed, total in _ordered_totals(result)
    ]
    return {
        "meta": meta.to_json(),
        "summary": {
            "all_passed": result.all_passed,
            "scenarios": result.total,
            "passed": result.passed,
            "attacks": {"scenarios": attacks_total, "passed": attacks_passed},
            "benign": {
                "scenarios": benign_total,
                "passed": benign_passed,
                "rate": result.benign_rate,
            },
            "categories": categories,
        },
        "scenarios": [scenario_json(r) for r in result.results],
    }


def to_markdown(result: SuiteResult, meta: RunMeta) -> str:
    attacks_passed, attacks_total = attack_totals(result)
    benign_passed, benign_total = benign_totals(result)
    lines = ["# agentgate red-team report", "", *meta.lines(), ""]
    lines += [
        "| Category | Scenarios | Passed | Pass rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, passed, total in _ordered_totals(result):
        if category is not Category.BENIGN:
            lines.append(
                f"| {category.value} | {total} | {passed} | {format_rate(passed, total)} |"
            )
    lines.append(
        f"| **attacks total** | {attacks_total} | {attacks_passed}"
        f" | {format_rate(attacks_passed, attacks_total)} |"
    )
    if Category.BENIGN in result.totals:
        lines.append(
            f"| {Category.BENIGN.value} | {benign_total} | {benign_passed}"
            f" | {format_rate(benign_passed, benign_total)} |"
        )
    lines.append("")
    if benign_total:
        lines.append(
            f"Benign control rate: {format_rate(benign_passed, benign_total)}. Legitimate"
            " tasks the gate let through; not counted in the attack total."
        )
    else:
        lines.append("Benign control rate: no benign scenarios ran.")
    lines += ["", "A scenario passes only when it ran to the end, recorded at least one check,"]
    lines += ["every check passed and the leak detector found nothing.", ""]
    lines += failure_lines(result)
    return "\n".join(lines) + "\n"


def failure_lines(result: SuiteResult) -> list[str]:
    """The failures section: each failed scenario's id, title, failed checks and leaks."""
    lines = ["## Failures", ""]
    if not result.failed:
        return [*lines, "None."]
    for failed in result.failed:
        lines.append(f"- `{failed.id}` ({failed.category.value}) {failed.title}")
        names = [check.name for check in failed.checks if not check.passed]
        if names:
            lines.append(f"  - failed checks: {'; '.join(names)}")
        if not failed.checks:
            lines.append("  - no checks recorded")
        for leak in failed.leaks:
            lines.append(f"  - leak: {leak}")
        if failed.error is not None:
            lines.append(f"  - error: {failed.error}")
    return lines


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
