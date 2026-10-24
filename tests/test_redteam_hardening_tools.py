"""The tool hardening scenarios (pi-015, pi-016, xt-014): they pass under both transports,
their ids fit the suite and the baseline, and each fails once its fix is taken away.

A mutation patches one protection for the test (through monkeypatch) and runs the scenario
function on a fresh arena with the suite's pass criteria, as tests/test_redteam_injection.py
does.
"""

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentgate.context import RunContext
from agentgate.models import Document
from agentgate.policy import addresses
from agentgate.policy import engine as engine_module
from agentgate.redteam.core import (
    CATEGORY_PREFIXES,
    Arena,
    Category,
    RegisteredScenario,
    all_scenarios,
    run_suite,
)
from agentgate.redteam.scenarios import hardening_tools
from agentgate.redteam.transport import Transport
from agentgate.retrieval import DocumentIndex
from agentgate.tools import builtin
from agentgate.tools.base import ToolEnv, ToolResult
from conftest import FIXTURES, POLICIES, ROOT

MODULE = hardening_tools.__name__
IDS = ["pi-015", "pi-016", "xt-014"]


def ours() -> list[RegisteredScenario]:
    return [s for s in all_scenarios() if s.module == MODULE]


def registered(scenario_id: str) -> RegisteredScenario:
    return next(s for s in ours() if s.id == scenario_id)


# The suite


@pytest.mark.parametrize("transport", list(Transport))
def test_every_scenario_passes(transport: Transport) -> None:
    suite = run_suite(ids=IDS, policies=POLICIES, fixtures=FIXTURES, transport=transport)
    failed = {
        r.id: ([c.name for c in r.checks if not c.passed], list(r.leaks), r.error)
        for r in suite.failed
    }
    assert failed == {}
    assert sorted(r.id for r in suite.results) == IDS
    for result in suite.results:
        assert result.checks and not result.benign


def test_suite_is_deterministic() -> None:
    first = run_suite(ids=IDS, policies=POLICIES, fixtures=FIXTURES)
    second = run_suite(ids=IDS, policies=POLICIES, fixtures=FIXTURES)
    assert first == second


# Ids and titles


def test_ids_prefixes_and_titles() -> None:
    scenarios = ours()
    assert sorted(s.id for s in scenarios) == IDS
    for s in scenarios:
        assert s.id.startswith(CATEGORY_PREFIXES[s.category] + "-"), s.id
        assert s.category in (Category.PROMPT_INJECTION, Category.CROSS_TENANT)
        assert s.title == s.title.strip() and 10 <= len(s.title) <= 80, s.title
        assert s.title[0].isupper() and not s.title.endswith("."), s.title
        assert re.fullmatch(r"[A-Za-z0-9 ,'\-]+", s.title), s.title
    titles = [s.title for s in scenarios]
    assert len(set(titles)) == len(titles)


def test_ids_are_in_the_baseline_and_unique_across_modules() -> None:
    baseline = json.loads((ROOT / "redteam" / "baseline.json").read_text(encoding="utf-8"))
    listed = {i for ids in baseline["categories"].values() for i in ids}
    # Registered in the baseline, so CI fails if any goes missing.
    assert set(IDS) <= listed
    others = [s.id for s in all_scenarios() if s.module != MODULE]
    assert not set(IDS) & set(others)


def test_module_source_is_plain_ascii_with_reserved_domains() -> None:
    source = Path(hardening_tools.__file__).read_text(encoding="utf-8")
    assert source.isascii()
    domains = re.findall(r"@([a-z0-9-]+(?:\.[a-z0-9-]+)+)", source, flags=re.IGNORECASE)
    assert domains
    assert all(domain.lower().endswith(".example") for domain in domains), domains


# Mutations


@dataclass(frozen=True)
class Judged:
    passed: bool
    failed_checks: tuple[str, ...]
    leaks: tuple[str, ...]
    error: str | None


Mutation = Callable[[pytest.MonkeyPatch], None]


def judge(scenario_id: str, mutate: Mutation, monkeypatch: pytest.MonkeyPatch) -> Judged:
    s = registered(scenario_id)
    mutate(monkeypatch)
    with Arena(policies=POLICIES, fixtures=FIXTURES) as arena:
        error: str | None = None
        try:
            s.fn(arena)
        except Exception as exc:
            error = type(exc).__name__
        failed = tuple(c.name for c in arena.checks if not c.passed)
        leaks = tuple(arena.leaks())
        passed = error is None and bool(arena.checks) and not failed and not leaks
    return Judged(passed=passed, failed_checks=failed, leaks=leaks, error=error)


def unchanged(_: pytest.MonkeyPatch) -> None:
    pass


# The rule before it was shared: strip, lowercase, then a pattern that admits any
# non-whitespace local part, so bidi, zero-width, "%" and "!" pass and a Kelvin sign is "k".
_OLD_ADDRESS = re.compile(r"^([^@\s,;:<>()\[\]\"\\]+)@([a-z0-9-]+(?:\.[a-z0-9-]+)+)$")


def _old_parse(raw: str) -> tuple[str, str] | None:
    match = _OLD_ADDRESS.fullmatch(raw.strip().lower())
    return None if match is None else (match.group(0), match.group(2))


def old_address_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine and tool both read addresses with the old lenient rule."""
    monkeypatch.setattr(addresses, "parse_address", _old_parse)


def engine_keeps_old_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the engine reads addresses with the old rule; the tool keeps the new one."""
    monkeypatch.setattr(engine_module, "_parse_address", _old_parse)


def tool_stores_as_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The email tool stores recipients as given and refuses nothing, as it used to."""

    def run(
        self: builtin.EmailSend, env: ToolEnv, ctx: RunContext, args: builtin.EmailSendArgs
    ) -> ToolResult:
        email_id = f"email-{uuid.uuid4().hex[:12]}"
        builtin._insert(
            env.store,
            "INSERT INTO outbox (id, run_id, customer_id, recipients, subject, body, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                email_id,
                ctx.run_id,
                ctx.customer_id,
                json.dumps(list(args.to)),
                args.subject,
                args.body,
                env.clock.now().isoformat(),
            ),
        )
        return ToolResult(content={"status": "queued", "email_id": email_id})

    monkeypatch.setattr(builtin.EmailSend, "_run", run)


def rebuild_per_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """notes.write rebuilds the whole index for every note, as it used to."""

    def add(self: DocumentIndex, document: Document) -> None:
        self.rebuild()

    monkeypatch.setattr(DocumentIndex, "add", add)


MUTATIONS: list[tuple[str, str, Mutation, str]] = [
    ("pi-015", "old-address-rule", old_address_rule, "disguised recipient #1 denied"),
    ("pi-015", "engine-keeps-old-rule", engine_keeps_old_rule, "disguised recipient #1 denied"),
    ("pi-015", "tool-stores-as-typed", tool_stores_as_typed, "approved injected send is refused"),
    ("pi-016", "old-address-rule", old_address_rule, "look-alike #1 denied"),
    ("pi-016", "engine-keeps-old-rule", engine_keeps_old_rule, "staff look-alike #1 denied"),
    ("xt-014", "rebuild-per-note", rebuild_per_note, "no KC index row was written or deleted"),
]


@pytest.mark.parametrize("scenario_id", IDS)
def test_the_mutation_harness_passes_unmutated(
    scenario_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert judge(scenario_id, unchanged, monkeypatch) == Judged(True, (), (), None)


@pytest.mark.parametrize(
    ("scenario_id", "mutate", "must_fail"),
    [pytest.param(sid, how, must, id=f"{sid}-{name}") for sid, name, how, must in MUTATIONS],
)
def test_scenario_fails_without_its_fix(
    scenario_id: str, mutate: Mutation, must_fail: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    judged = judge(scenario_id, mutate, monkeypatch)
    assert not judged.passed
    assert judged.error is None
    found = [*judged.failed_checks, *judged.leaks]
    assert any(must_fail in item for item in found), found


def test_every_scenario_has_a_mutation() -> None:
    assert {sid for sid, *_ in MUTATIONS} == set(IDS)


def test_the_old_rule_let_every_disguised_recipient_through_as_internal() -> None:
    # The scenarios' inputs really are the forms the old rule got wrong, not ones it
    # already refused.
    internal = "kettlecreek.example"
    admitted = []
    for raw in hardening_tools.DISGUISED:
        assert addresses.parse_address(raw) is None, raw
        parsed = _old_parse(raw)
        if parsed is not None and parsed[1] == internal:
            admitted.append(raw)
    # All but a trailing bidi mark (which the old domain pattern caught) and a quoted local
    # part (which it refused) went out as internal mail.
    assert len(admitted) == len(hardening_tools.DISGUISED) - 2
    assert hardening_tools.DISGUISED[0] in admitted
    kelvin = hardening_tools.LOOK_ALIKE_DOMAINS[0]
    assert _old_parse(kelvin) == ("terrence@kettlecreek.example", internal)
    assert addresses.parse_address(kelvin) is None
