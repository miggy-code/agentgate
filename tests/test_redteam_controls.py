"""The approval bypass, fail closed, audit integrity and benign red-team scenarios.

Three kinds of test:

1. The scenarios of these four categories pass on the real gate, except the ones listed in
   KNOWN_GATE_BUGS, which must fail until the gate is fixed.
2. Mutation tests: each protection an attack scenario targets is removed, by editing the
   source of the one method that holds it (or replacing the method), and the scenarios that
   target it must then fail. A scenario that still passed with its protection removed would
   be testing nothing.
3. Tightening tests, the same check the other way round for benign scenarios: the rule a
   legitimate task relies on is removed from a copy of the policy, and the scenario must
   then fail. A benign scenario that still passed would not show the gate letting work
   through.
"""

import inspect
import shutil
import textwrap
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

import agentgate.approvals as approvals_module
import agentgate.audit.panoptes as panoptes_module
import agentgate.gate as gate_module
import agentgate.policy.engine as engine_module
import agentgate.retrieval as retrieval_module
from agentgate.approvals import ApprovalQueue
from agentgate.audit.panoptes import PanoptesSink
from agentgate.gate import CapReached, Denied, Gate
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import Effect
from agentgate.redteam.core import (
    CATEGORY_PREFIXES,
    Arena,
    Category,
    ScenarioResult,
    all_scenarios,
    run_suite,
)
from conftest import FIXTURES, JR, KC, POLICIES

MINE = (
    Category.APPROVAL_BYPASS,
    Category.FAIL_CLOSED,
    Category.AUDIT_INTEGRITY,
    Category.BENIGN,
)
MODULES = ("approval_bypass", "fail_closed", "audit_integrity", "benign")
# Em and en dash, which none of these files may contain.
DASHES = frozenset({chr(0x2014), chr(0x2013)})
SCENARIO_DIR = Path(__file__).resolve().parent.parent / "src/agentgate/redteam/scenarios"

# Scenarios that assert correct behaviour the gate does not have yet. Each must fail on the
# real gate; when one starts passing, the bug is fixed and it comes off this list.
KNOWN_GATE_BUGS: dict[str, set[str]] = {}


def run(ids: list[str], policies: Path = POLICIES) -> list[ScenarioResult]:
    return list(run_suite(ids=ids, policies=policies, fixtures=FIXTURES).results)


# 1. The scenarios pass


def test_scenarios_pass_except_known_gate_bugs() -> None:
    suite = run_suite(categories=MINE, policies=POLICIES, fixtures=FIXTURES)
    by_id = {result.id: result for result in suite.results}
    unexpected = [
        (r.id, r.error, r.leaks, [c.name for c in r.checks if not c.passed])
        for r in suite.results
        if not r.passed and r.id not in KNOWN_GATE_BUGS
    ]
    assert unexpected == []
    for scenario_id, failing in KNOWN_GATE_BUGS.items():
        result = by_id[scenario_id]
        assert not result.passed, f"{scenario_id} passes now: remove it from KNOWN_GATE_BUGS"
        assert result.error is None
        assert result.leaks == ()
        assert {c.name for c in result.checks if not c.passed} == failing
    assert all(result.checks for result in suite.results)


def test_category_sizes() -> None:
    counts: dict[Category, int] = defaultdict(int)
    for s in all_scenarios():
        counts[s.category] += 1
    assert counts[Category.APPROVAL_BYPASS] >= 10
    assert counts[Category.FAIL_CLOSED] >= 10
    assert counts[Category.AUDIT_INTEGRITY] >= 10
    assert counts[Category.BENIGN] >= 13


def test_ids_titles_and_text() -> None:
    mine = [s for s in all_scenarios() if s.module.rsplit(".", 1)[-1] in MODULES]
    assert {s.module.rsplit(".", 1)[-1] for s in mine} == set(MODULES)
    for s in mine:
        assert s.id.startswith(CATEGORY_PREFIXES[s.category] + "-"), s.id
        assert s.category in MINE
        assert 10 <= len(s.title) <= 72, s.id
        assert s.title == s.title.strip() and not s.title.endswith("."), s.id
        assert s.title[0].isupper(), s.id
    for name in MODULES:
        text = (SCENARIO_DIR / f"{name}.py").read_text()
        assert DASHES.isdisjoint(text), name
    text = Path(__file__).read_text()
    assert DASHES.isdisjoint(text)


# 2. Mutation tests

Edit = tuple[str, str]


@dataclass(frozen=True)
class Mutation:
    name: str
    # (owner, attribute, source edits): the attribute's source with each edit applied.
    edits: tuple[tuple[Any, str, tuple[Edit, ...]], ...] = ()
    # (owner, attribute, replacement): the attribute replaced outright.
    replace: tuple[tuple[Any, str, Any], ...] = ()
    names: dict[str, Any] = field(default_factory=dict)
    # Every one of these must fail with the mutation in place.
    breaks: tuple[str, ...] = ()


def mutant(owner: Any, attribute: str, edits: tuple[Edit, ...], names: dict[str, Any]) -> Any:
    """The attribute recompiled from its own source with each edit applied exactly once."""
    original = inspect.getattr_static(owner, attribute)
    source = textwrap.dedent(inspect.getsource(original))
    for old, new in edits:
        assert source.count(old) == 1, f"{attribute}: edit target found {source.count(old)} times"
        source = source.replace(old, new)
    namespace = {**original.__globals__, **names}
    exec(compile(source, f"<mutant {attribute}>", "exec"), namespace)
    return namespace[attribute]


@pytest.fixture(scope="module")
def shipped_engines() -> Iterator[dict[str, PolicyEngine]]:
    with Arena(policies=POLICIES, fixtures=FIXTURES) as arena:
        yield dict(arena.engines)


def execute_ignoring_approval(
    self: Gate, ctx: Any, approval_id: str, tool: str, action: Any, args: Any
) -> Any:
    """execute_approved with every approval check removed: decide, then run."""
    call = self._begin(ctx, tool, action)
    checked = self._check_run(ctx, call)
    if isinstance(checked, Denied):
        return checked
    call = gate_module._bind(call, checked)
    prepared = self._prepare(checked, call, tool, action, args)
    if isinstance(prepared, Denied):
        return prepared
    verdict = self._verdict(checked, prepared)
    if verdict.effect is Effect.DENY:
        return self._deny_verdict(call, prepared, verdict, None)
    reservation = self._reserve(checked, call, prepared.config.cost_usd)
    if isinstance(reservation, CapReached):
        return reservation
    return self._execute(checked, call, prepared, reservation, verdict, subject=None)


def lax_parse_address(raw: str) -> tuple[str, str] | None:
    address = raw.strip().lower()
    return address, address.rsplit("@", 1)[-1]


def lax_is_external(raw: str, domains: frozenset[str]) -> bool:
    return raw.strip().lower().rsplit("@", 1)[-1] not in domains


REAL_CREATE = ApprovalQueue.create
REAL_APPROVE = ApprovalQueue.approve


def create_then_record(self: ApprovalQueue, *, record: Any = None, **kwargs: Any) -> Any:
    """The bug ai-012 found: the approval row commits, then the event is attempted."""
    approval = REAL_CREATE(self, **kwargs)
    if record is not None:
        record(approval)
    return approval


def approve_then_record(
    self: ApprovalQueue, *args: Any, record: Any = None, expired: Any = None
) -> Any:
    decided = REAL_APPROVE(self, *args, expired=expired)
    if record is not None:
        record(decided)
    return decided


def mutations(fallback: dict[str, PolicyEngine]) -> list[Mutation]:
    tools = {tool_id: t for e in fallback.values() for tool_id, t in e.bundle.tools.items()}
    claim = (ApprovalQueue, "_claim")
    status_off = ("if approval.status is not ApprovalStatus.APPROVED:", "if False:")
    return [
        # Approval bypass
        Mutation(
            "approvals ignored",
            replace=((Gate, "execute_approved", execute_ignoring_approval),),
            breaks=(
                "ab-001",
                "ab-002",
                "ab-003",
                "ab-004",
                "ab-005",
                "ab-006",
                "ab-007",
                "ab-008",
                "ab-009",
                "ab-010",
                "ab-011",
                "ab-012",
                "ab-015",
            ),
        ),
        Mutation(
            "approval not bound to its run",
            edits=(
                (
                    Gate,
                    "execute_approved",
                    (
                        (
                            "approval.customer_id != run.ctx.customer_id or approval.run_id"
                            " != run.ctx.run_id",
                            "approval.customer_id != run.ctx.customer_id",
                        ),
                    ),
                ),
            ),
            breaks=("ab-002",),
        ),
        Mutation(
            "approval not bound to its args",
            edits=(
                (
                    *claim,
                    (("if not hmac.compare_digest(requested, approval.args_hash):", "if False:"),),
                ),
            ),
            breaks=("ab-006", "ab-007", "ab-008"),
        ),
        Mutation(
            "approval status not checked",
            edits=((*claim, (status_off,)),),
            breaks=("ab-003", "ab-004", "ab-005", "ab-010"),
        ),
        Mutation(
            "approval runs more than once",
            edits=(
                (
                    *claim,
                    (("if approval.status is ApprovalStatus.EXECUTED:", "if False:"), status_off),
                ),
            ),
            breaks=("ab-009",),
        ),
        Mutation(
            "approvals never expire",
            edits=(
                (
                    ApprovalQueue,
                    "_expire_if_due",
                    (("if live and self._now() >= approval.expires_at:", "if False:"),),
                ),
            ),
            replace=((ApprovalQueue, "expire_due", lambda self: []),),
            breaks=("ab-011", "ab-012"),
        ),
        Mutation(
            "requester may approve",
            edits=(
                (
                    ApprovalQueue,
                    "_decide_locked",
                    (("if approver.id == approval.requested_by:", "if False:"),),
                ),
            ),
            breaks=("ab-004",),
        ),
        Mutation(
            "any role may approve",
            edits=(
                (
                    ApprovalQueue,
                    "_decide_locked",
                    (("if approver.role not in settings.approver_roles:", "if False:"),),
                ),
            ),
            breaks=("ab-005",),
        ),
        Mutation(
            "approved calls not decided again",
            edits=(
                (Gate, "execute_approved", (("if verdict.effect is Effect.DENY:", "if False:"),)),
            ),
            breaks=("ab-013", "ab-014"),
        ),
        Mutation(
            "approval lookup built with string formatting",
            edits=(
                (
                    ApprovalQueue,
                    "_load",
                    (
                        (
                            '"SELECT * FROM approvals WHERE id = ?", (approval_id,)',
                            "f\"SELECT * FROM approvals WHERE id = '{approval_id}'\"",
                        ),
                    ),
                ),
            ),
            breaks=("ab-015",),
        ),
        Mutation(
            "raw approval ids written to the audit",
            edits=(
                (
                    Gate,
                    "_decide_approval",
                    (
                        (
                            "safe_id = approval_id if _APPROVAL_ID.fullmatch(approval_id)"
                            " else None",
                            "safe_id = approval_id",
                        ),
                    ),
                ),
            ),
            breaks=("ab-015",),
        ),
        # Fail closed
        Mutation(
            "model-chosen tool names written to the audit",
            edits=(
                (
                    Gate,
                    "_begin",
                    (
                        (
                            "tool=tool if tool is not None and _TOOL_ID.fullmatch(tool) else None,",
                            "tool=tool,",
                        ),
                    ),
                ),
            ),
            breaks=("fc-001",),
        ),
        Mutation(
            "tools configured for any customer accepted",
            edits=(
                (
                    Gate,
                    "_prepare",
                    (
                        (
                            "config = run.engine.bundle.tools.get(tool)",
                            "config = run.engine.bundle.tools.get(tool) or _ANY_TOOL.get(tool)",
                        ),
                    ),
                ),
                (
                    PolicyEngine,
                    "_builtin_check",
                    (("return Reason.UNKNOWN_TOOL", "tool = _ANY_TOOL[request.tool]"),),
                ),
            ),
            names={"_ANY_TOOL": tools},
            breaks=("fc-002",),
        ),
        Mutation(
            "engine skips the action check",
            edits=(
                (PolicyEngine, "_builtin_check", (("return Reason.ACTION_NOT_SUPPORTED", "pass"),)),
            ),
            breaks=("fc-003",),
        ),
        Mutation(
            "arguments not validated",
            edits=(
                (
                    gate_module,
                    "_validate_args",
                    (
                        (
                            "return model.model_validate_json(raw, strict=True)",
                            "return model.model_construct(**json.loads(raw))",
                        ),
                    ),
                ),
            ),
            breaks=("fc-004", "fc-005"),
        ),
        Mutation(
            "missing policy falls back to the shipped one",
            edits=(
                (
                    Gate,
                    "_check_run",
                    (
                        (
                            'engine = self._engines.get(row["customer_id"])',
                            'engine = self._engines.get(row["customer_id"])'
                            ' or _FALLBACK.get(row["customer_id"])',
                        ),
                    ),
                ),
            ),
            names={"_FALLBACK": fallback},
            breaks=("fc-006",),
        ),
        Mutation(
            "unknown labels ranked as public",
            edits=(
                (
                    PolicyEngine,
                    "__init__",
                    (
                        (
                            "self._ranks = {label: rank for rank, label in"
                            " enumerate(bundle.policy.labels)}",
                            "self._ranks = defaultdict(int, {label: rank for rank, label in"
                            " enumerate(bundle.policy.labels)})",
                        ),
                    ),
                ),
                (
                    PolicyEngine,
                    "_builtin_check",
                    (
                        ("if document.label not in self._ranks:", "if False:"),
                        (
                            "if request.run_max_label is not None and request.run_max_label"
                            " not in self._ranks:",
                            "if False:",
                        ),
                    ),
                ),
            ),
            names={"defaultdict": defaultdict},
            breaks=("fc-007", "fc-008"),
        ),
        Mutation(
            "tool runs when its allow cannot be recorded",
            edits=(
                (
                    Gate,
                    "_execute",
                    (
                        ("self._ledger.release(reservation)", "pass"),
                        ("return Denied(GateReason.AUDIT_UNAVAILABLE.value)", "pass"),
                    ),
                ),
            ),
            breaks=("fc-009",),
        ),
        Mutation(
            "search returns results its retrieval event could not record",
            edits=(
                (
                    Gate,
                    "_search",
                    (
                        (
                            "self._ledger.release(reservation)\n"
                            "        return Denied(GateReason.AUDIT_UNAVAILABLE.value)",
                            "pass",
                        ),
                    ),
                ),
            ),
            breaks=("fc-010",),
        ),
        Mutation(
            "failed tool keeps its spend reservation",
            edits=((Gate, "_tool_failed", (("self._ledger.release(reservation)", "pass"),)),),
            breaks=("fc-011",),
        ),
        Mutation(
            "ranked results not checked again",
            edits=(
                (
                    retrieval_module.DocumentIndex,
                    "_recheck",
                    (("if decision.effect is not Effect.ALLOW:", "if False:"),),
                ),
            ),
            breaks=("fc-012",),
        ),
        Mutation(
            "recipients split on the last at sign",
            replace=(
                (engine_module, "_parse_address", lax_parse_address),
                (gate_module, "_is_external", lax_is_external),
            ),
            breaks=("fc-013",),
        ),
        Mutation(
            "a deny loses to a matching allow or approval",
            edits=(
                (
                    PolicyEngine,
                    "decide",
                    (
                        (
                            "if matched[Effect.DENY]:",
                            "if matched[Effect.DENY] and not (matched[Effect.ALLOW]"
                            " or matched[Effect.REQUIRE_APPROVAL]):",
                        ),
                    ),
                ),
            ),
            breaks=("fc-014",),
        ),
        # Audit integrity
        Mutation(
            "events lose their policy version",
            edits=(
                (
                    Gate,
                    "_emit",
                    (
                        (
                            "policy_version=policy_version or call.policy_version,",
                            "policy_version=None,",
                        ),
                    ),
                ),
            ),
            breaks=("ai-001", "ai-002", "ai-009"),
        ),
        Mutation(
            "Panoptes copy drops rule ids",
            edits=(
                (
                    panoptes_module,
                    "to_panoptes_event",
                    (('"rule_ids": list(event.rule_ids),', '"rule_ids": [],'),),
                ),
            ),
            breaks=("ai-005",),
        ),
        Mutation(
            "denials lose their rule ids",
            edits=((Gate, "_deny_verdict", (("rule_ids=verdict.rule_ids,", "rule_ids=(),"),)),),
            breaks=("ai-002",),
        ),
        Mutation(
            "allow events log the arguments",
            edits=(
                (
                    Gate,
                    "_record_allow",
                    (
                        (
                            "details=details,",
                            'details={**(details or {}), "args":'
                            ' prepared.args.model_dump(mode="json")},',
                        ),
                    ),
                ),
            ),
            breaks=("ai-003", "ai-004"),
        ),
        Mutation(
            "approval changes commit before their event",
            replace=(
                (ApprovalQueue, "create", create_then_record),
                (ApprovalQueue, "approve", approve_then_record),
            ),
            breaks=("ai-012",),
        ),
        Mutation(
            "verification always passes",
            replace=((PanoptesSink, "verify", lambda self: {"valid": True}),),
            breaks=("ai-006", "ai-007"),
        ),
        Mutation(
            "executed events dropped",
            edits=((Gate, "_finish", (("self._emit(", "(lambda *a, **k: None)("),)),),
            breaks=("ai-008",),
        ),
        Mutation(
            "policy version not derived from the policy",
            replace=((engine_module, "bundle_version", lambda bundle: "sha256:" + "0" * 16),),
            breaks=("ai-009",),
        ),
        Mutation(
            "refused identities not recorded",
            edits=(
                (
                    Gate,
                    "_identity_denied",
                    (("self._emit(call, EventKind.DENY, reason, decision=True)", "pass"),),
                ),
            ),
            breaks=("ai-010",),
        ),
        Mutation(
            "retrieval events log the query terms",
            edits=(
                (
                    Gate,
                    "_search",
                    (
                        (
                            '"query_terms": len(trace.query_terms),',
                            '"query_terms": list(trace.query_terms),',
                        ),
                    ),
                ),
            ),
            breaks=("ai-011",),
        ),
    ]


def apply(monkeypatch: pytest.MonkeyPatch, mutation: Mutation) -> None:
    for owner, attribute, edits in mutation.edits:
        monkeypatch.setattr(owner, attribute, mutant(owner, attribute, edits, mutation.names))
    for owner, attribute, replacement in mutation.replace:
        monkeypatch.setattr(owner, attribute, replacement)


MUTATION_NAMES = [m.name for m in mutations({})]


@pytest.mark.parametrize("name", MUTATION_NAMES)
def test_removing_a_protection_fails_its_scenarios(
    name: str, monkeypatch: pytest.MonkeyPatch, shipped_engines: dict[str, PolicyEngine]
) -> None:
    mutation = next(m for m in mutations(shipped_engines) if m.name == name)
    apply(monkeypatch, mutation)
    results = run(list(mutation.breaks))
    still_passing = [r.id for r in results if r.passed]
    assert still_passing == [], f"{name}: these scenarios passed without their protection"


def test_every_attack_scenario_has_an_automated_mutation() -> None:
    covered = {scenario_id for m in mutations({}) for scenario_id in m.breaks}
    # Scenarios in other modules (the hardening_* ones) carry their mutations in their own
    # test files; this file covers its own modules.
    attacks = {
        s.id
        for s in all_scenarios()
        if s.module.rsplit(".", 1)[-1] in MODULES
        and s.category is not Category.BENIGN
        and s.id not in KNOWN_GATE_BUGS
    }
    assert attacks - covered == set()


def test_mutations_leave_the_real_gate_alone() -> None:
    # The edits only ever patch through monkeypatch, so the shipped code still passes.
    results = run(["ab-002", "fc-012", "ai-008"])
    assert all(r.passed for r in results)
    assert approvals_module.ApprovalQueue._claim.__module__ == "agentgate.approvals"


# 3. Policy edits: removing a rule a scenario relies on must fail it


def edited_policies(tmp_path: Path, customer: str, edit: Callable[[dict[str, Any]], None]) -> Path:
    root = tmp_path / "policies"
    shutil.copytree(POLICIES, root)
    path = root / customer / "policy.yaml"
    data = yaml.safe_load(path.read_text())
    edit(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return root


def drop(*rule_ids: str) -> Callable[[dict[str, Any]], None]:
    def edit(data: dict[str, Any]) -> None:
        before = len(data["rules"])
        data["rules"] = [rule for rule in data["rules"] if rule["id"] not in rule_ids]
        assert len(data["rules"]) == before - len(rule_ids)

    return edit


def deny_staff_reads(data: dict[str, Any]) -> None:
    data["rules"].append(
        {
            "id": "staff-read-frozen",
            "effect": "deny",
            "roles": ["staff"],
            "tools": ["docs.search", "docs.read"],
            "actions": ["read"],
        }
    )


def approvers_owner_only(data: dict[str, Any]) -> None:
    data["approvals"]["approver_roles"] = ["owner"]


def tiny_daily_cap(data: dict[str, Any]) -> None:
    data["spend"]["per_customer_day_usd"] = "0.05"


TIGHTENINGS: list[tuple[str, str, str, Callable[[dict[str, Any]], None]]] = [
    ("bn-001", KC, "staff-read removed", drop("staff-read")),
    ("bn-001", KC, "staff reads denied", deny_staff_reads),
    ("bn-002", JR, "staff-read removed", drop("staff-read")),
    ("bn-003", KC, "contractor-read-assigned removed", drop("contractor-read-assigned")),
    ("bn-004", JR, "contractor-read-sop removed", drop("contractor-read-sop")),
    ("bn-005", KC, "write-notes-confidential removed", drop("write-notes-confidential")),
    ("bn-006", JR, "manager-read removed", drop("manager-read")),
    ("bn-007", JR, "manager-read-orders removed", drop("manager-read-orders")),
    ("bn-008", KC, "owner-read removed", drop("owner-read")),
    ("bn-009", JR, "owner-read removed", drop("owner-read")),
    ("bn-010", KC, "email-internal removed", drop("email-internal")),
    ("bn-010", JR, "email-internal removed", drop("email-internal")),
    ("bn-011", KC, "email-customers removed", drop("email-customers")),
    ("bn-011", KC, "managers may not approve", approvers_owner_only),
    ("bn-012", JR, "email-customers removed", drop("email-customers")),
    ("bn-013", KC, "parts-order-approval-to-5000 removed", drop("parts-order-approval-to-5000")),
    ("bn-013", KC, "parts-order-up-to-500 removed", drop("parts-order-up-to-500")),
    ("bn-014", KC, "parts-order-staff-approval removed", drop("parts-order-staff-approval")),
    ("bn-015", JR, "refund-staff-approval removed", drop("refund-staff-approval")),
    ("bn-016", JR, "refund-owner-approval removed", drop("refund-owner-approval")),
    ("bn-017", JR, "write-notes-internal removed", drop("write-notes-internal")),
    ("bn-018", KC, "staff reads denied", deny_staff_reads),
    ("bn-019", KC, "daily cap cut to 0.05", tiny_daily_cap),
    # An attack scenario whose protection is a policy rule, the same way round.
    (
        "ab-014",
        KC,
        "no-external-after-confidential removed",
        drop("no-external-after-confidential"),
    ),
]


@pytest.mark.parametrize(
    ("scenario_id", "customer", "change", "edit"),
    TIGHTENINGS,
    ids=[f"{s}-{c[:2]}-{change}" for s, c, change, _ in TIGHTENINGS],
)
def test_policy_edits_fail_the_scenarios_that_rely_on_them(
    tmp_path: Path,
    scenario_id: str,
    customer: str,
    change: str,
    edit: Callable[[dict[str, Any]], None],
) -> None:
    policies = edited_policies(tmp_path, customer, edit)
    (result,) = run([scenario_id], policies=policies)
    assert not result.passed, f"{scenario_id} still passes with {change}"
    assert result.error is None


def test_every_benign_scenario_is_tightened() -> None:
    benign = {
        s.id
        for s in all_scenarios()
        if s.category is Category.BENIGN and s.module.endswith(".benign")
    }
    assert benign - {scenario_id for scenario_id, *_ in TIGHTENINGS} == set()
