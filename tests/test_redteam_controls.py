"""The approval bypass, fail closed, audit integrity and benign red-team scenarios.

Two kinds of test (that the scenarios pass is tested once, for the whole suite, in
test_redteam_core.py):

1. Mutation tests: each protection an attack scenario targets is removed, by editing the
   source of the one method that holds it (or replacing the method), and the scenarios that
   target it must then fail. A scenario that still passed with its protection removed would
   be testing nothing.
2. Tightening tests, the same check the other way round for benign scenarios: the rule a
   legitimate task relies on is removed from a copy of the policy, and the scenario must
   then fail. A benign scenario that still passed would not show the gate letting work
   through.
"""

import shutil
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

import agentgate.audit.panoptes as panoptes_module
import agentgate.gate as gate_module
import agentgate.policy.engine as engine_module
import agentgate.retrieval as retrieval_module
import agentgate.store as store_module
import agentgate.world as world_module
from agentgate.approvals import ApprovalQueue
from agentgate.audit.panoptes import PanoptesSink
from agentgate.gate import CapReached, Denied, Gate
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import Effect
from agentgate.redteam.core import (
    Arena,
    Category,
    all_scenarios,
    run_suite,
)
from agentgate.redteam.transport import Transport
from agentgate.spend import SpendLedger
from agentgate.store import PolicySnapshot, Store
from conftest import (
    FIXTURES,
    JR,
    KC,
    POLICIES,
    Mutation,
    assert_breaks,
    assert_every_attack_has_a_mutation,
    run_mutated,
)

# 1. Mutation tests


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


def _code_mutations() -> list[Mutation]:
    with Arena(policies=POLICIES, fixtures=FIXTURES) as arena:
        fallback = dict(arena.engines)
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
            crashes=True,
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
            crashes=True,
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
            crashes=True,
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
            crashes=True,
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


# Caller-chosen values in the durable log (ai-013, ai-014), checked over both transports.
REFERENCE_MUTATIONS = [
    Mutation(
        "the gate attaches no references",
        ("ai-013", "ai-014"),
        edits=(
            (
                Gate,
                "_emit",
                (
                    (
                        "references=self._references(call, subject, unknown_subject),",
                        "references={},",
                    ),
                ),
            ),
        ),
    ),
    Mutation(
        "the Panoptes sink stores caller values as given",
        ("ai-013", "ai-014"),
        edits=(
            (
                panoptes_module,
                "to_panoptes_event",
                (("event = event.recorded()", "event = event"),),
            ),
        ),
    ),
    Mutation(
        "a missing document id is treated as a known one",
        ("ai-013", "ai-014"),
        edits=((Gate, "_prepare", (("unknown_subject=facts.document_id,", ""),)),),
    ),
    Mutation(
        "a missing approval id is treated as a known one",
        ("ai-013",),
        edits=((Gate, "execute_approved", (("unknown_subject=_text(approval_id),", ""),)),),
    ),
    Mutation(
        "an unknown approver is treated as a known one",
        ("ai-014",),
        edits=(
            (
                Gate,
                "_decide_approval",
                (('claimed=frozenset({"user_id"})', "claimed=frozenset()"),),
            ),
        ),
    ),
    Mutation(
        "an unknown user at run start is treated as a known one",
        ("ai-014",),
        edits=(
            (
                Gate,
                "start_run",
                (('claimed=frozenset({"customer_id", "user_id"}),', "claimed=frozenset(),"),),
            ),
        ),
    ),
]


# Reconstructing an incident from the log (ai-020 to ai-024).
def _nothing(*args: object, **kwargs: object) -> None:
    return None


def _unsaved(self: Store, snapshot: PolicySnapshot) -> tuple[PolicySnapshot, bool]:
    # Says it stored the snapshot, stores nothing.
    return snapshot, True


def _no_problems(*args: object, **kwargs: object) -> list[object]:
    return []


def _no_values(*args: object, **kwargs: object) -> dict[str, object]:
    return {}


RECONSTRUCT_MUTATIONS = [
    Mutation(
        "the approval request carries no approval hash",
        ("ai-020",),
        edits=((Gate, "call_tool", (('"approval_args_hash": created.args_hash,', ""),)),),
    ),
    Mutation(
        "check-store skips approval rows",
        ("ai-020",),
        replace=((world_module, "_approval_problems", _no_problems),),
    ),
    Mutation(
        "executed events name no side-effect row",
        ("ai-021", "ai-024"),
        edits=((Gate, "_finish", (("if result.effect_ref is not None:", "if False:"),)),),
    ),
    Mutation(
        "check-store skips side-effect rows",
        ("ai-021",),
        replace=((world_module, "_effect_problems", _no_problems),),
    ),
    Mutation(
        "policy snapshots are not stored",
        ("ai-022",),
        replace=((Store, "save_policy_snapshot", _unsaved),),
    ),
    Mutation(
        "policy loads are not recorded",
        ("ai-022",),
        replace=((Gate, "record_policy_loaded", _nothing),),
    ),
    Mutation(
        "operator actions are not recorded",
        ("ai-023",),
        replace=((Gate, "record_operator_action", _nothing),),
    ),
    Mutation(
        "the stale sweep records nothing",
        ("ai-023",),
        edits=((Gate, "release_stale", (("if not record:", "if True:"),)),),
    ),
    Mutation(
        "the fixture sync records nothing",
        ("ai-023",),
        replace=((world_module, "record_fixture_sync", _nothing),),
    ),
    Mutation(
        "decision events carry no values",
        ("ai-020", "ai-021", "ai-024"),
        replace=((Gate, "_call_values", _no_values),),
    ),
    Mutation(
        "allow events drop the call's values",
        ("ai-024",),
        edits=(
            (
                Gate,
                "_record_allow",
                (("details = {**prepared.values, **(details or {})}", "details = details or {}"),),
            ),
        ),
    ),
]


# Running agentgate rather than attacking it (fc-015 to fc-017).
def _no_sweep(self: SpendLedger, older_than: object) -> int:
    return 0


def _no_late_settle(self: SpendLedger, *args: object) -> None:
    return None


def _no_prepare(path: object) -> None:
    return None


OPERATIONS_MUTATIONS = [
    Mutation("no stale sweep", ("fc-015",), replace=((SpendLedger, "release_stale", _no_sweep),)),
    Mutation(
        "late settle dropped",
        ("fc-015",),
        replace=((SpendLedger, "_finish_reclaimed", _no_late_settle),),
    ),
    Mutation(
        "audit log accepted as a store",
        ("fc-016",),
        replace=((store_module, "_refuse_audit_database", _nothing),),
    ),
    Mutation(
        "paths not checked",
        ("fc-016",),
        replace=((world_module, "check_paths", _nothing),),
        crashes=True,
    ),
    Mutation(
        "store file left to sqlite",
        ("fc-017",),
        replace=((store_module, "_prepare_file", _no_prepare),),
    ),
]

MUTATIONS = [
    *_code_mutations(),
    *REFERENCE_MUTATIONS,
    *RECONSTRUCT_MUTATIONS,
    *OPERATIONS_MUTATIONS,
]


# A loose umask, so fc-017 shows the store file being tightened rather than created private.
@pytest.mark.usefixtures("loose_umask")
@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.name)
def test_mutation_fails_its_scenarios(mutation: Mutation, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_breaks(mutation, monkeypatch)


@pytest.mark.parametrize("mutation", REFERENCE_MUTATIONS, ids=lambda m: m.name)
def test_reference_mutations_fail_over_mcp_too(
    mutation: Mutation, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert_breaks(mutation, monkeypatch, Transport.MCP)


def test_a_reconstruct_mutation_fails_over_mcp_too(monkeypatch: pytest.MonkeyPatch) -> None:
    unreferenced = next(
        m for m in RECONSTRUCT_MUTATIONS if m.name == "executed events name no side-effect row"
    )
    [result] = run_mutated(unreferenced, monkeypatch, Transport.MCP, ids=["ai-021"]).results
    assert not result.passed
    assert "a raised order is caught" in [c.name for c in result.checks if not c.passed]


def test_every_attack_has_a_mutation() -> None:
    assert_every_attack_has_a_mutation(
        MUTATIONS, Category.APPROVAL_BYPASS, Category.FAIL_CLOSED, Category.AUDIT_INTEGRITY
    )


@pytest.mark.usefixtures("loose_umask")
@pytest.mark.parametrize("transport", list(Transport))
def test_file_world_scenarios_pass_under_a_loose_umask(transport: Transport) -> None:
    ids = ["fc-015", "fc-016", "fc-017", "id-015", "id-016"]
    suite = run_suite(ids=ids, policies=POLICIES, fixtures=FIXTURES, transport=transport)
    assert [r.id for r in suite.failed] == []


# 2. Policy edits: removing a rule a scenario relies on must fail it


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
    (result,) = run_suite(ids=[scenario_id], policies=policies, fixtures=FIXTURES).results
    assert not result.passed, f"{scenario_id} still passes with {change}"
    assert result.error is None


def test_every_benign_scenario_is_tightened() -> None:
    benign = {
        s.id
        for s in all_scenarios()
        if s.category is Category.BENIGN and s.module.endswith(".benign")
    }
    assert benign - {scenario_id for scenario_id, *_ in TIGHTENINGS} == set()
