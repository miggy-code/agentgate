"""Audit integrity: every decision is on record, says who and why, and carries no data.

Each gate call must leave at least one decision event naming the customer, user, role, run,
policy version and a reason code; denials name the rules that decided them and cap hits the
cap. No event may carry a document body, an email subject or body, a note, a search query,
an approval note or a canary. Events stored in Panoptes must verify, and editing or
deleting a stored row must break that verification.

The arena records into a memory sink. The Panoptes scenarios copy the arena's events, in
order, into a Panoptes log in a temporary directory, which is the same mapping the gate's
default sink uses, and then tamper with that file the way someone with disk access would.
"""

import json
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from pydantic import JsonValue

from agentgate.audit.base import AuditEvent, EventKind
from agentgate.audit.panoptes import PanoptesSink
from agentgate.context import RunContext
from agentgate.gate import PendingApproval
from agentgate.models import Role
from agentgate.policy.engine import PolicyEngine
from agentgate.policy.schema import Policy, PolicyBundle
from agentgate.redteam.core import (
    REDTEAM_START,
    Arena,
    AuditDown,
    Category,
    describe,
    returned_ids,
    scenario,
)

KC = "kettle-creek-hvac"
JR = "juniper-row-apothecary"
AI = Category.AUDIT_INTEGRITY

BEATRICE = "beatrice.l@mail-sample.example"
COLLEAGUE = "terrence@kettlecreek.example"
DISPATCH = "kc-dispatch-procedure"
PAYROLL = "kc-payroll-summary"
DECISIONS = frozenset(
    {
        EventKind.ALLOW,
        EventKind.DENY,
        EventKind.APPROVAL_REQUIRED,
        EventKind.RETRIEVAL,
        EventKind.CAP_HIT,
    }
)


def _customer_email(subject: str = "Your furnace visit") -> dict[str, JsonValue]:
    return {
        "to": [BEATRICE],
        "subject": subject,
        "body": "Hello, our technician will arrive Monday between 9 and 11.",
    }


def _internal_email(
    subject: str = "Van 3 inspection", body: str = "Due Friday."
) -> dict[str, JsonValue]:
    return {"to": [COLLEAGUE], "subject": subject, "body": body}


def _audit_text(arena: Arena) -> str:
    return "\n".join(event.model_dump_json() for event in arena.events())


def _pending_id(arena: Arena, ctx: RunContext, tool: str, args: dict[str, JsonValue]) -> str:
    outcome = arena.call(ctx, tool, args)
    arena.expect_pending(outcome, f"{tool} needs approval")
    return outcome.approval_id if isinstance(outcome, PendingApproval) else ""


def _mixed_flow(arena: Arena) -> RunContext:
    """A staff run with allows, denies, an approval and an execution: events of every kind."""
    ctx = arena.start("kc-staff-nadine")
    arena.call(ctx, "docs.search", {"query": "dispatch procedure"})
    arena.call(ctx, "docs.read", {"document_id": DISPATCH})
    arena.call(ctx, "docs.read", {"document_id": PAYROLL})
    arena.call(ctx, "email.send", _internal_email())
    email = _customer_email()
    approval_id = _pending_id(arena, ctx, "email.send", email)
    arena.approve(approval_id, "kc-mgr-terrence")
    arena.execute(ctx, approval_id, "email.send", email)
    arena.call(ctx, "shell.exec", {"command": "ls"})
    return ctx


@contextmanager
def _panoptes_copy(arena: Arena) -> Iterator[tuple[PanoptesSink, Path]]:
    """The arena's events recorded, in order, into a Panoptes log in a temporary directory."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audit.db"
        sink = PanoptesSink(path)
        try:
            for event in arena.events():
                sink.record(event)
            yield sink, path
        finally:
            sink.close()


def _tamper(path: Path, sql: str, *params: object) -> int:
    conn = sqlite3.connect(path)
    try:
        changed = conn.execute(sql, params).rowcount
        conn.commit()
    finally:
        conn.close()
    return changed


def _stored_fields(path: Path) -> list[dict[str, JsonValue]]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT context FROM events ORDER BY rowid").fetchall()
    finally:
        conn.close()
    fields: list[dict[str, JsonValue]] = []
    for (context,) in rows:
        loaded = json.loads(context)
        fields.append(loaded.get("agentgate", {}) if isinstance(loaded, dict) else {})
    return fields


def _first(events: list[AuditEvent], kind: EventKind) -> AuditEvent | None:
    return next((event for event in events if event.kind is kind), None)


@scenario("ai-001", AI, "Every call in a mixed run leaves a complete decision event")
def every_call_audited(arena: Arena) -> None:
    ctx = arena.start("kc-staff-nadine")
    calls: tuple[tuple[str, str, dict[str, JsonValue]], ...] = (
        ("search", "docs.search", {"query": "dispatch procedure"}),
        ("internal read", "docs.read", {"document_id": DISPATCH}),
        ("restricted read", "docs.read", {"document_id": PAYROLL}),
        ("missing read", "docs.read", {"document_id": "kc-no-such-document"}),
        ("internal email", "email.send", _internal_email()),
        ("customer email", "email.send", _customer_email()),
        ("parts order", "parts.order", {"part": "Igniter", "quantity": 1, "amount_usd": "150.00"}),
        ("unknown tool", "shell.exec", {"command": "ls"}),
        ("bad args", "email.send", {"to": COLLEAGUE}),
        ("note", "notes.write", {"title": "Reminder", "body": "Check vans.", "label": "internal"}),
    )
    for name, tool, args in calls:
        before = len(arena.sink.events)
        arena.call(ctx, tool, args)
        new = arena.sink.events[before:]
        kinds = sorted(event.kind.value for event in new)
        decided = any(event.kind in DECISIONS for event in new)
        arena.expect(f"{name} left a decision event", decided, ", ".join(kinds) or "no events")
    pending = [e.subject for e in arena.events(EventKind.APPROVAL_REQUIRED, ctx)]
    if pending and pending[0] is not None:
        before = len(arena.sink.events)
        arena.approve(pending[0], "kc-mgr-terrence")
        approved = arena.sink.events[before:]
        arena.expect(
            "approval decision audited",
            [e.kind for e in approved] == [EventKind.APPROVED]
            and approved[0].user_id == "kc-mgr-terrence"
            and approved[0].role is Role.MANAGER
            and approved[0].run_id == ctx.run_id,
            ", ".join(e.kind.value for e in approved),
        )
    version = arena.engines[KC].version
    run_events = [e for e in arena.events() if e.run_id == ctx.run_id]
    incomplete = [
        e.kind.value
        for e in run_events
        if e.customer_id != KC
        or e.role is None
        or e.user_id is None
        or e.policy_version != version
        or not e.reason
    ]
    arena.expect(
        "every event names customer, user, role, version, reason",
        not incomplete,
        f"{len(incomplete)} incomplete of {len(run_events)}",
    )
    staff_events = [e for e in run_events if e.kind is not EventKind.APPROVED]
    arena.expect(
        "the run's own events name the staff user",
        all(e.user_id == "kc-staff-nadine" and e.role is Role.STAFF for e in staff_events),
    )
    latencies = [e for e in run_events if e.kind in DECISIONS and e.latency_us is None]
    arena.expect("every decision records its latency", not latencies, f"{len(latencies)} without")


@scenario("ai-002", AI, "Deny, approval and cap events name the rule or cap that decided")
def rule_ids_on_decisions(arena: Arena) -> None:
    manager = arena.start("kc-mgr-marisol")
    arena.call(manager, "docs.read", {"document_id": "kc-customer-list"})
    arena.call(manager, "email.send", _customer_email())
    deny = _first(arena.events(EventKind.DENY, manager), EventKind.DENY)
    arena.expect(
        "rule deny names its rule",
        deny is not None and deny.rule_ids == ("no-external-after-confidential",),
        "none" if deny is None else ", ".join(deny.rule_ids),
    )
    order: dict[str, JsonValue] = {"part": "Condenser", "quantity": 1, "amount_usd": "1200.00"}
    arena.call(manager, "parts.order", order)
    staff = arena.start("kc-staff-nadine")
    arena.call(staff, "email.send", _customer_email())
    arena.call(staff, "docs.read", {"document_id": DISPATCH})
    arena.call(staff, "docs.read", {"document_id": PAYROLL})
    jr_staff = arena.start("jr-staff-priya")
    refund: dict[str, JsonValue] = {
        "order_id": "JR-26-1061",
        "amount_usd": "40.00",
        "reason": "Late.",
    }
    arena.call(jr_staff, "refund.issue", refund)
    expected: tuple[tuple[str, RunContext, EventKind, tuple[str, ...]], ...] = (
        ("parts approval", manager, EventKind.APPROVAL_REQUIRED, ("parts-order-approval-to-5000",)),
        ("email approval", staff, EventKind.APPROVAL_REQUIRED, ("email-customers",)),
        ("allowed read", staff, EventKind.ALLOW, ("staff-read",)),
        ("refund approval", jr_staff, EventKind.APPROVAL_REQUIRED, ("refund-staff-approval",)),
    )
    for name, ctx, kind, rules in expected:
        event = _first(arena.events(kind, ctx), kind)
        found = () if event is None else event.rule_ids
        arena.expect(f"{name} names its rule", found == rules, ", ".join(found) or "none")
    default = [e for e in arena.events(EventKind.DENY, staff) if e.subject == PAYROLL]
    arena.expect(
        "default deny says so and names no rule",
        len(default) == 1 and default[0].reason == "default_deny" and default[0].rule_ids == (),
        f"{len(default)} events",
    )

    # Thirty calls is the Juniper Row cap per run; the thirty-first is refused.
    runaway = arena.start("jr-staff-tomas")
    for _ in range(30):
        arena.call(runaway, "docs.search", {"query": "cold-chain shipping"})
    capped = arena.call(runaway, "docs.search", {"query": "cold-chain shipping"})
    arena.expect_cap(capped, "max_calls")
    hits = arena.events(EventKind.CAP_HIT, runaway)
    hit = hits[0] if len(hits) == 1 else None
    arena.expect(
        "cap event names the cap, limit and use",
        hit is not None
        and hit.reason == "max_calls"
        and hit.subject == "max_calls"
        and hit.details.get("limit") == 30
        and hit.details.get("used") == 30
        and hit.policy_version == arena.engines[JR].version,
        f"{len(hits)} cap events",
    )


@scenario("ai-003", AI, "No event carries document text, email text, notes or queries")
def no_content_in_events(arena: Arena) -> None:
    owner = arena.start("kc-owner-dale")
    query = "succession valuation payroll quillmark"
    arena.call(owner, "docs.search", {"query": query})
    documents = arena.store.list_documents(KC)
    for document in documents:
        arena.expect_executed(
            arena.call(owner, "docs.read", {"document_id": document.id}),
            f"owner reads {document.id}",
        )
    note: dict[str, JsonValue] = {
        "title": "Quillmark succession summary",
        "body": "Quillmark: the broker estimate and the payout plan are in the owner notes.",
        "label": "confidential",
    }
    arena.call(owner, "notes.write", note)
    email = _internal_email("Quillmark board prep", "Quillmark: bring the margin sheet Friday.")
    arena.expect_executed(arena.call(owner, "email.send", email), "internal email sent")
    text = _audit_text(arena)
    lines = {
        line.strip()
        for document in documents
        for line in document.body.splitlines()
        if len(line.strip()) >= 25
    }
    quoted = sorted(line for line in lines if line in text)
    arena.expect("no document line in any event", not quoted, f"{len(quoted)} lines found")
    titles = [d.title for d in documents if len(d.title) >= 12 and d.title in text]
    arena.expect("no document title in any event", not titles, f"{len(titles)} titles found")
    arena.expect("no query, note, subject or body", "uillmark" not in text)
    arena.expect("no canary in any event", "AGCANARY-" not in text)


@scenario("ai-004", AI, "Approval and refund events carry no message, note or reason text")
def no_content_in_approval_events(arena: Arena) -> None:
    staff = arena.start("kc-staff-nadine")
    email = _customer_email("Bramblewick visit")
    approval_id = _pending_id(arena, staff, "email.send", email)
    jr_staff = arena.start("jr-staff-priya")
    refund: dict[str, JsonValue] = {
        "order_id": "JR-26-1061",
        "amount_usd": "40.00",
        "reason": "Bramblewick jar arrived cracked in transit.",
    }
    refund_id = _pending_id(arena, jr_staff, "refund.issue", refund)
    if not approval_id or not refund_id:
        return
    arena.gate.approve(approval_id, "kc-mgr-terrence", note="Bramblewick wording looks fine.")
    arena.gate.approve(refund_id, "jr-mgr-daniel", note="Bramblewick refund ok.")
    arena.expect_executed(arena.execute(staff, approval_id, "email.send", email), "email runs")
    arena.expect_executed(arena.execute(jr_staff, refund_id, "refund.issue", refund), "refund runs")
    text = _audit_text(arena)
    arena.expect("no subject, body, reason or note text", "ramblewick" not in text)
    arena.expect("no recipient address", BEATRICE not in text)
    approved = arena.events(EventKind.APPROVED)
    arena.expect(
        "approval events say only that a note exists",
        len(approved) == 2 and all(e.details.get("has_note") is True for e in approved),
        f"{len(approved)} events",
    )
    allows = [e for e in arena.events(EventKind.ALLOW) if e.subject in (approval_id, refund_id)]
    approvers = sorted(str(e.details.get("approved_by")) for e in allows)
    arena.expect(
        "executions name their approver",
        approvers == ["jr-mgr-daniel", "kc-mgr-terrence"],
        ", ".join(approvers),
    )


@scenario("ai-005", AI, "A Panoptes copy of a mixed flow verifies and matches the events")
def panoptes_verifies(arena: Arena) -> None:
    _mixed_flow(arena)
    events = arena.events()
    with _panoptes_copy(arena) as (sink, path):
        result = sink.verify()
        arena.expect("chain verifies", result.get("valid") is True, str(result.get("error")))
        arena.expect("every event stored", sink.count() == len(events), f"{sink.count()} stored")
        stored = _stored_fields(path)
        expected = [
            {
                "kind": e.kind.value,
                "reason": e.reason,
                "run_id": e.run_id,
                "user_id": e.user_id,
                "policy_version": e.policy_version,
                "rule_ids": list(e.rule_ids),
            }
            for e in events
        ]
        got = [{key: row.get(key) for key in expected[0]} for row in stored] if expected else []
        arena.expect("stored rows match the events in order", got == expected)


@scenario("ai-006", AI, "Editing one stored audit row breaks verification")
def panoptes_edit_detected(arena: Arena) -> None:
    ctx = _mixed_flow(arena)
    target = _first(arena.events(EventKind.DENY, ctx), EventKind.DENY)
    if not arena.expect("flow has a deny to rewrite", target is not None) or target is None:
        return
    edits = (
        ("deny rewritten as success", "outcome", "$.status", "success"),
        ("reason rewritten", "context", "$.agentgate.reason", "matched_allow"),
        ("user rewritten", "context", "$.agentgate.user_id", "kc-owner-dale"),
    )
    for name, column, json_path, value in edits:
        with _panoptes_copy(arena) as (sink, path):
            before = sink.verify().get("valid")
            changed = _tamper(
                path,
                f"UPDATE events SET {column} = json_set({column}, ?, ?) WHERE event_id = ?",
                json_path,
                value,
                target.event_id,
            )
            result = sink.verify()
            arena.expect(
                f"{name}: detected",
                before is True and changed == 1 and result.get("valid") is False,
                str(result.get("error")),
            )
            arena.expect(
                f"{name}: the edited event is named",
                result.get("broken_event_id") == target.event_id,
            )


@scenario("ai-007", AI, "Deleting one stored audit row breaks verification")
def panoptes_delete_detected(arena: Arena) -> None:
    _mixed_flow(arena)
    events = arena.events()
    if not arena.expect("flow has events", len(events) >= 3, f"{len(events)} events"):
        return
    for name, event in (
        ("first", events[0]),
        ("a deny", _first(events, EventKind.DENY) or events[1]),
        ("last", events[-1]),
    ):
        with _panoptes_copy(arena) as (sink, path):
            before = sink.verify().get("valid")
            changed = _tamper(path, "DELETE FROM events WHERE event_id = ?", event.event_id)
            result = sink.verify()
            arena.expect(
                f"deleting the {name} event is detected",
                before is True and changed == 1 and result.get("valid") is False,
                str(result.get("error")),
            )


@scenario("ai-008", AI, "Events are recorded in the order the calls were made")
def event_order(arena: Arena) -> None:
    ctx = _mixed_flow(arena)
    got = [
        (e.kind.value, e.tool, e.reason, e.user_id)
        for e in arena.events()
        if e.run_id == ctx.run_id
    ]
    nadine, terrence = "kc-staff-nadine", "kc-mgr-terrence"
    expected = [
        ("run_started", None, "run_started", nadine),
        ("retrieval", "docs.search", "retrieval_filter", nadine),
        ("executed", "docs.search", "executed", nadine),
        ("allow", "docs.read", "matched_allow", nadine),
        ("executed", "docs.read", "executed", nadine),
        ("deny", "docs.read", "default_deny", nadine),
        ("allow", "email.send", "matched_allow", nadine),
        ("executed", "email.send", "executed", nadine),
        ("approval_required", "email.send", "matched_approval", nadine),
        ("approved", "email.send", "approved", terrence),
        ("allow", "email.send", "matched_approval", nadine),
        ("executed", "email.send", "executed", nadine),
        ("deny", "shell.exec", "unknown_tool", nadine),
    ]
    arena.expect("event sequence matches the calls", got == expected, f"{len(got)} events")
    stamps = [e.timestamp for e in arena.events()]
    arena.expect("timestamps never go backwards", stamps == sorted(stamps))


@scenario("ai-009", AI, "Events carry the new policy version after a policy change")
def policy_version_changes(arena: Arena) -> None:
    ctx = arena.start("kc-staff-nadine")
    email = _customer_email()
    approval_id = _pending_id(arena, ctx, "email.send", email)
    arena.call(ctx, "docs.read", {"document_id": DISPATCH})
    old = arena.engines[KC].version
    before = len(arena.sink.events)

    engines: dict[str, PolicyEngine] = dict(arena.engines)
    current = engines[KC].bundle
    raw = current.policy.model_dump(by_alias=True, exclude_none=True)
    raw["rules"] = [
        *raw["rules"],
        {
            "id": "no-contractor-parts",
            "effect": "deny",
            "roles": ["contractor"],
            "tools": ["parts.order"],
            "actions": ["spend"],
        },
    ]
    customer = arena.store.get_customer(KC)
    if customer is None:
        return
    bundle = PolicyBundle(policy=Policy.model_validate(raw), tools=current.tools)
    engines[KC] = PolicyEngine(bundle, customer)
    arena.regate(engines=engines)
    new = arena.engines[KC].version
    arena.expect("the change gives a new version", new != old)

    arena.approve(approval_id, "kc-mgr-terrence")
    arena.call(ctx, "docs.read", {"document_id": DISPATCH})
    arena.call(ctx, "docs.read", {"document_id": PAYROLL})
    arena.expect_executed(arena.execute(ctx, approval_id, "email.send", email), "email runs")
    earlier = [e for e in arena.sink.events[:before] if e.run_id == ctx.run_id]
    later = [e for e in arena.sink.events[before:] if e.kind is not EventKind.APPROVED]
    arena.expect(
        "earlier events carry the old version", all(e.policy_version == old for e in earlier)
    )
    arena.expect(
        "later decisions carry the new version",
        bool(later) and all(e.policy_version == new for e in later),
        f"{len(later)} events",
    )
    approval = arena.gate.approvals.get(approval_id)
    arena.expect(
        "the approval keeps the version it was requested under",
        approval is not None and approval.policy_version == old,
    )


@scenario("ai-010", AI, "Refused runs, forged contexts and refused approvers are audited")
def refusals_audited(arena: Arena) -> None:
    forged = RunContext(
        run_id="forged-run",
        customer_id=KC,
        user_id="kc-owner-dale",
        role=Role.OWNER,
        started_at=REDTEAM_START,
    )
    staff = arena.start("kc-staff-nadine")
    approval_id = _pending_id(arena, staff, "email.send", _customer_email())
    attempts: tuple[tuple[str, str, str | None, Role | None], ...] = (
        ("suspended user", "suspended_user", "kc-staff-gavin", Role.STAFF),
        ("unknown user", "unknown_user", "nobody-at-all", None),
        ("other customer's user", "wrong_customer", "jr-staff-priya", None),
        ("forged context", "invalid_run", "kc-owner-dale", Role.OWNER),
        ("staff approver", "approver_not_allowed", "kc-staff-colby", Role.STAFF),
    )
    for name, reason, user, role in attempts:
        before = len(arena.sink.events)
        if reason == "invalid_run":
            arena.expect_denied(arena.call(forged, "docs.read", {"document_id": PAYROLL}))
        elif reason == "approver_not_allowed":
            arena.approve(approval_id, "kc-staff-colby")
        elif reason == "wrong_customer":
            arena.try_start("jr-staff-priya", KC)
        else:
            arena.try_start(str(user))
        new = arena.sink.events[before:]
        event = new[0] if len(new) == 1 else None
        arena.expect(
            f"{name} leaves one deny with the reason",
            event is not None and event.kind is EventKind.DENY and event.reason == reason,
            ", ".join(f"{e.kind.value} {e.reason}" for e in new) or "no events",
        )
        arena.expect(
            f"{name} event names who asked",
            event is not None and event.user_id == user and event.role is role,
        )
    run_ids = {e.run_id for e in arena.events(EventKind.DENY) if e.reason == "invalid_run"}
    arena.expect("forged run id recorded as claimed", run_ids == {"forged-run"})


@scenario("ai-011", AI, "Retrieval events record ids and counts, never the query")
def retrieval_event_contents(arena: Arena) -> None:
    for user in ("kc-staff-nadine", "kc-contractor-hollis", "jr-mgr-daniel"):
        ctx = arena.start(user)
        query = "zanthorpe dispatch procedure furnace order records"
        outcome = arena.call(ctx, "docs.search", {"query": query})
        shown = returned_ids(outcome)
        events = arena.events(EventKind.RETRIEVAL, ctx)
        details = events[0].details if len(events) == 1 else {}
        logged = details.get("returned_ids")
        arena.expect(f"{user} event lists what the model got", logged == shown, describe(outcome))
        filtered = details.get("filtered_by_reason")
        total = (
            sum(v for v in filtered.values() if isinstance(v, int))
            if isinstance(filtered, dict)
            else -1
        )
        candidates = details.get("candidates_in_tenant")
        permitted = details.get("permitted")
        tenant = len(arena.store.list_documents(arena.customer_of(user)))
        arena.expect(
            f"{user} counts add up",
            isinstance(permitted, int) and candidates == tenant and permitted + total == tenant,
            f"{permitted} permitted, {total} filtered, {candidates} candidates",
        )
        arena.expect(
            f"{user} query recorded as a count", isinstance(details.get("query_terms"), int)
        )
    arena.expect("the query never reaches the audit", "zanthorpe" not in _audit_text(arena))


@scenario("ai-012", AI, "Approval state changes while the audit log is down")
def approval_changes_without_audit(arena: Arena) -> None:
    ctx = arena.start("kc-staff-nadine")
    email = _customer_email()
    # A request the log cannot record must not leave a pending approval behind.
    arena.break_audit(lambda event: event.kind is EventKind.APPROVAL_REQUIRED)
    with suppress(AuditDown):
        arena.call(ctx, "email.send", email)
    arena.restore_audit()
    unrecorded = arena.gate.approvals.list(KC)
    arena.expect(
        "an unrecorded request leaves no approval",
        not unrecorded,
        ", ".join(a.status.value for a in unrecorded) or "none",
    )

    # A decision the log cannot record must not take effect.
    second = _customer_email("Your furnace visit, rescheduled")
    approval_id = _pending_id(arena, ctx, "email.send", second)
    arena.break_audit(lambda event: event.kind is EventKind.APPROVED)
    with suppress(AuditDown):
        arena.approve(approval_id, "kc-mgr-terrence")
    arena.restore_audit()
    approval = arena.gate.approvals.get(approval_id)
    status = "missing" if approval is None else approval.status.value
    arena.expect("an unrecorded approval stays pending", status == "pending", f"status {status}")
    outcome = arena.execute(ctx, approval_id, "email.send", second)
    arena.expect_denied(outcome, "not_approved", name="an unrecorded approval cannot run")
    approved = arena.events(EventKind.APPROVED)
    sent = len(arena.outbox())
    arena.expect(
        "nothing sent without an approval on record",
        sent == 0 or bool(approved),
        f"{sent} sent, {len(approved)} approvals recorded",
    )
