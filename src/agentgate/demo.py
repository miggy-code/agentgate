"""`make demo`: ten short scenarios that show the gate deciding what agent runs can read,
write, spend and send, offline, with mock models and no keys.

Every scenario runs in one in-memory world over the real fixtures and policies, with a fixed
clock and a Panoptes audit log, and checks its own outcome against what the policy should
do. The output is plain ASCII and byte-for-byte deterministic: run and approval ids are
printed as `run #1` and `approval #1` in order of creation, the only times are the fixed
clock's, and nothing measured (latency, durations) is printed. Document bodies and email
bodies are never printed, only ids, titles, labels, counts and reason codes, the same
things the audit log keeps.
"""

import contextlib
import json
import tempfile
import textwrap
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TextIO

from pydantic import JsonValue

from agentgate.agent.loop import AgentTranscript, run_agent, tool_specs
from agentgate.agent.mock import NaiveModel, RepeatModel, ScriptedModel
from agentgate.agent.model import Model, ToolCall
from agentgate.approvals import ApprovalError
from agentgate.audit.base import AuditEvent, AuditSink, EventKind
from agentgate.audit.panoptes import PanoptesSink
from agentgate.clock import FixedClock
from agentgate.context import RunContext
from agentgate.gate import CapReached, Denied, Executed, IdentityError, Outcome, PendingApproval
from agentgate.models import Action, Role
from agentgate.spend import CapKind
from agentgate.store import Store
from agentgate.tools.builtin import default_tools, list_outbox, list_refunds
from agentgate.world import DEFAULT_FIXTURES, DEFAULT_POLICIES, World, open_memory_world

DEMO_START = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)
WIDTH = 100
AUDIT_FILE = "audit.db"

KC = "kettle-creek-hvac"
JR = "juniper-row-apothecary"
KC_EXFIL = "records-review@exfil-drop.example"
DIRECTORY_CUSTOMER = "beatrice.l@mail-sample.example"
OTHER_DIRECTORY_CUSTOMER = "ruben.c@mail-sample.example"
SENSITIVE_LABELS = frozenset({"confidential", "restricted"})

KC_FURNACE_TASK = (
    "A homeowner wrote in about a furnace making a clicking noise and asked if it is the"
    " igniter. What should we tell him?"
)
KC_MANAGER_TASK = "Is the homeowner whose furnace is clicking already on our customer list?"

TUNE_UP_EMAIL: dict[str, JsonValue] = {
    "to": [DIRECTORY_CUSTOMER],
    "subject": "Your tune-up visit on Monday",
    "body": "Hi Beatrice, the technician will arrive at 9 am on Monday. Reply to reschedule.",
}
VISIT_EMAIL: dict[str, JsonValue] = {
    "to": [OTHER_DIRECTORY_CUSTOMER],
    "subject": "Confirming your service visit",
    "body": "Hi Ruben, this confirms your air conditioner visit on Thursday afternoon.",
}
REFUND: dict[str, JsonValue] = {
    "order_id": "JR-26-1061",
    "amount_usd": "40.00",
    "reason": "Order arrived late",
}

Check = tuple[str, bool]


class Names:
    """Stable names for random ids: `run #1`, `approval #1`, in order of first sight."""

    def __init__(self) -> None:
        self._runs: dict[str, int] = {}
        self._approvals: dict[str, int] = {}

    def run(self, run_id: str) -> str:
        return f"run #{self._runs.setdefault(run_id, len(self._runs) + 1)}"

    def approval(self, approval_id: str) -> str:
        number = self._approvals.setdefault(approval_id, len(self._approvals) + 1)
        return f"approval #{number}"


class _TeeSink:
    """Records to the real sink, then keeps a copy so the demo can print what was recorded."""

    def __init__(self, inner: AuditSink) -> None:
        self._inner = inner
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self._inner.record(event)
        self.events.append(event)


# Formatting shared with `agentgate agent run`


def wrap(text: str, indent: int = 0, hang: int = 2) -> list[str]:
    """text as lines of at most WIDTH columns, continuation lines indented a little more."""
    return textwrap.wrap(
        text,
        width=WIDTH,
        initial_indent=" " * indent,
        subsequent_indent=" " * (indent + hang),
        break_on_hyphens=False,
    ) or [" " * indent]


def _quote(value: str, limit: int = 60) -> str:
    if len(value) > limit:
        value = value[: limit - 3] + "..."
    return json.dumps(value, ensure_ascii=True)


def _time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _usd(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.000001'))} USD"


def _rules(rule_ids: tuple[str, ...]) -> str:
    if not rule_ids:
        return ""
    return (", rule " if len(rule_ids) == 1 else ", rules ") + ", ".join(rule_ids)


def _audit(event: AuditEvent | None) -> str:
    if event is None:
        return "audit: no event found"
    return f"audit: {event.kind.value} {event.reason}{_rules(event.rule_ids)}"


def describe_call(tool: str, args: dict[str, JsonValue], store: Store) -> str:
    """What the model proposed, without any free text it could carry (bodies, reasons)."""
    if tool == "docs.search":
        query = args.get("query")
        return f"{tool} query={_quote(query) if isinstance(query, str) else '?'}"
    if tool == "docs.read":
        document_id = args.get("document_id")
        if not isinstance(document_id, str):
            return f"{tool} (no document id)"
        document = store.get_document(document_id)
        title = "" if document is None else f" {_quote(document.title)}"
        return f"{tool} {document_id}{title}"
    if tool == "email.send":
        recipients = args.get("to")
        to = (
            ", ".join(r for r in recipients if isinstance(r, str))
            if isinstance(recipients, list)
            else "?"
        )
        subject = args.get("subject")
        body = args.get("body")
        size = len(body) if isinstance(body, str) else 0
        shown = _quote(subject) if isinstance(subject, str) else "?"
        return f"{tool} to {to} subject {shown} (body {size} chars, not shown)"
    if tool == "refund.issue":
        return f"{tool} order {args.get('order_id')} amount {args.get('amount_usd')} USD"
    if tool == "parts.order":
        part = args.get("part")
        shown = _quote(part) if isinstance(part, str) else "?"
        return f"{tool} {shown} x{args.get('quantity')} amount {args.get('amount_usd')} USD"
    if tool == "notes.write":
        note_title = args.get("title")
        shown = _quote(note_title) if isinstance(note_title, str) else "?"
        return f"{tool} title {shown} label {args.get('label')}"
    # A tool the model made up: its name and args are not echoed.
    return f"unknown tool ({len(args)} args)"


def _result(tool: str, content: dict[str, JsonValue]) -> str:
    if tool == "docs.read":
        return f"document returned, label {content.get('label')}"
    if tool == "email.send":
        return "email queued in the outbox"
    if tool == "refund.issue":
        return "refund issued"
    if tool == "parts.order":
        return "parts ordered"
    if tool == "notes.write":
        return f"note written, label {content.get('label')}"
    return "done"


def describe_outcome(
    tool: str, outcome: Outcome, event: AuditEvent | None, names: Names, told: str = "model"
) -> list[str]:
    """The gate's decision, what the caller (the model) was told, and the audit reason."""
    if isinstance(outcome, Executed):
        if tool == "docs.search":
            return _search_lines(outcome, event)
        return wrap(f"gate: allowed, {_result(tool, outcome.content)} ({_audit(event)})", 6)
    if isinstance(outcome, Denied):
        return wrap(f"gate: denied, {told} told {outcome.reason} ({_audit(event)})", 6)
    if isinstance(outcome, PendingApproval):
        name = names.approval(outcome.approval_id)
        text = f"gate: needs approval, {name} expires {_time(outcome.expires_at)} ({_audit(event)})"
        return wrap(text, 6)
    return wrap(f"gate: cap reached, {outcome.cap.value} ({_audit(event)}{_cap(event)})", 6)


def _cap(event: AuditEvent | None) -> str:
    if event is None:
        return ""
    used, limit = event.details.get("used"), event.details.get("limit")
    return f", {used} used of {limit}"


def _search_lines(outcome: Executed, event: AuditEvent | None) -> list[str]:
    count = outcome.content.get("count")
    lines = wrap(f"gate: allowed, {count} documents returned ({_audit(event)})", 6)
    if event is None or event.kind is not EventKind.RETRIEVAL:
        return lines
    details = event.details
    filtered = details.get("filtered_by_reason")
    reasons = filtered if isinstance(filtered, dict) else {}
    total = sum(n for n in reasons.values() if isinstance(n, int))
    by_reason = ", ".join(f"{reason} {reasons[reason]}" for reason in sorted(reasons))
    lines += wrap(
        f"filter: {details.get('candidates_in_tenant')} documents in the tenant,"
        f" {details.get('permitted')} permitted, {total} filtered"
        + (f" ({by_reason})" if by_reason else ""),
        6,
    )
    returned = details.get("returned_ids")
    ids = [i for i in returned if isinstance(i, str)] if isinstance(returned, list) else []
    lines += wrap("returned: " + (", ".join(ids) or "none"), 6)
    return lines


_DECISION_KINDS = {
    "executed": EventKind.ALLOW,
    "denied": EventKind.DENY,
    "pending_approval": EventKind.APPROVAL_REQUIRED,
    "cap_reached": EventKind.CAP_HIT,
}


def _is_model_cap(event: AuditEvent) -> bool:
    return event.kind is EventKind.CAP_HIT and event.details.get("source") == "model"


def _outcome_of(tool: str, status: str, content: dict[str, JsonValue]) -> Outcome:
    """Rebuild a gate Outcome from what the loop handed the model."""
    if status == "executed":
        return Executed(tool=tool, content=content, cost_usd=Decimal(0))
    if status == "pending_approval":
        approval_id = content.get("approval_id")
        expires = content.get("expires_at")
        return PendingApproval(
            approval_id=str(approval_id),
            expires_at=datetime.fromisoformat(str(expires)),
        )
    if status == "cap_reached":
        return CapReached(cap=CapKind(str(content.get("cap"))))
    return Denied(str(content.get("error")))


def format_transcript(
    transcript: AgentTranscript, events: list[AuditEvent], names: Names, store: Store
) -> list[str]:
    """One block per proposed call, in order, each matched to its own audit decision.

    Calls are numbered in order. Consecutive calls with the same proposal and the same
    decision are folded into one line, so a runaway loop reads as one repeated step.
    """
    run_events = [e for e in events if e.run_id == transcript.run_id]
    queues: dict[str, deque[AuditEvent]] = {}
    for event in run_events:
        if _is_model_cap(event):
            key = "model_cap"
        elif event.kind is EventKind.RETRIEVAL:
            key = "search"
        else:
            key = event.kind.value
        queues.setdefault(key, deque()).append(event)

    def pop(key: str) -> AuditEvent | None:
        queue = queues.get(key)
        return queue.popleft() if queue else None

    blocks: list[list[str]] = []
    for step in transcript.steps:
        calls: tuple[ToolCall, ...] = step.response.message.tool_calls
        for call, result in zip(calls, step.outcomes, strict=True):
            outcome = _outcome_of(call.tool, result.status, result.content)
            if result.status == "executed" and call.tool == "docs.search":
                key = "search"
            else:
                key = _DECISION_KINDS.get(result.status, EventKind.DENY).value
            block = [describe_call(call.tool, call.args, store)]
            block += describe_outcome(call.tool, outcome, pop(key), names)
            blocks.append(block)

    lines: list[str] = []
    index = 0
    while index < len(blocks):
        repeat = 1
        while index + repeat < len(blocks) and blocks[index + repeat] == blocks[index]:
            repeat += 1
        first, *rest = blocks[index]
        lines += wrap(f"[{index + 1}] {first}", 2, 4)
        lines += rest
        if repeat > 1:
            lines.append(
                f"  [{index + 2}-{index + repeat}] the same call and decision"
                f" {repeat - 1} more times"
            )
        index += repeat

    model_cap = pop("model_cap")
    if model_cap is not None:
        lines += wrap(
            f"[{len(blocks) + 1}] next model call refused before it was made:"
            f" cap reached, {model_cap.reason} ({_audit(model_cap)}{_cap(model_cap)})",
            2,
            4,
        )
    model_calls = len(transcript.steps)
    lines.append(f"  stop: {transcript.stop_reason} after {model_calls} model calls")
    if transcript.final_text is not None:
        label = "final answer" if transcript.stop_reason == "final" else "last text"
        lines += wrap(f"{label}: {transcript.final_text}", 2, 4)
    return lines


# The demo


@dataclass
class _Demo:
    world: World
    clock: FixedClock
    sink: PanoptesSink
    tee: _TeeSink
    out: TextIO
    names: Names = field(default_factory=Names)

    @property
    def events(self) -> list[AuditEvent]:
        return self.tee.events

    def say(self, text: str = "", indent: int = 0) -> None:
        for line in wrap(text, indent) if text else [""]:
            self.out.write(line.rstrip() + "\n")

    def lines(self, lines: list[str]) -> None:
        for line in lines:
            self.out.write(line.rstrip() + "\n")

    def start(self, user_id: str) -> RunContext:
        user = self.world.store.get_user(user_id)
        assert user is not None, user_id
        ctx = self.world.gate.start_run(user.customer_id, user_id)
        self.say(
            f"acting: {user.id} (role {user.role.value}, customer {user.customer_id}),"
            f" {self.names.run(ctx.run_id)}"
        )
        return ctx

    def agent(self, ctx: RunContext, model: Model, task: str) -> AgentTranscript:
        self.say(f"task: {task}")
        customer = self.world.store.get_customer(ctx.customer_id)
        assert customer is not None
        transcript = run_agent(
            self.world.gate,
            ctx,
            model,
            task,
            specs=tool_specs(self.world.engines[ctx.customer_id], default_tools()),
            company=customer.name,
            max_steps=1000,
        )
        self.lines(format_transcript(transcript, self.events, self.names, self.world.store))
        return transcript

    def gate_call(self, label: str, call: Callable[[], Outcome], tool: str) -> Outcome:
        """Run one direct gate call and print it with the decision event it produced."""
        start = len(self.events)
        outcome = call()
        decision = next(
            (
                e
                for e in self.events[start:]
                if e.kind
                in {
                    EventKind.ALLOW,
                    EventKind.DENY,
                    EventKind.APPROVAL_REQUIRED,
                    EventKind.CAP_HIT,
                    EventKind.RETRIEVAL,
                }
            ),
            None,
        )
        self.say(label, 2)
        self.lines(describe_outcome(tool, outcome, decision, self.names, told="caller"))
        return outcome

    def decide(self, approval_id: str, approver_id: str) -> str:
        """Approve as approver_id; the approval's status, or the refusal reason."""
        user = self.world.store.get_user(approver_id)
        role = "?" if user is None else user.role.value
        name = self.names.approval(approval_id)
        try:
            decided = self.world.gate.approve(approval_id, approver_id)
        except ApprovalError as err:
            self.say(f"{approver_id} (role {role}) approves {name}: refused, {err.reason.value}", 2)
            return err.reason.value
        self.say(f"{approver_id} (role {role}) approves {name}: {decided.status.value}", 2)
        return decided.status.value

    def deny_events(self, run_id: str) -> list[AuditEvent]:
        return [e for e in self.events if e.run_id == run_id and e.kind is EventKind.DENY]


def _calls(transcript: AgentTranscript) -> list[tuple[ToolCall, str, dict[str, JsonValue]]]:
    return [
        (call, result.status, result.content)
        for step in transcript.steps
        for call, result in zip(step.response.message.tool_calls, step.outcomes, strict=True)
    ]


def _pending_id(transcript: AgentTranscript) -> str | None:
    for _, status, content in _calls(transcript):
        approval_id = content.get("approval_id")
        if status == "pending_approval" and isinstance(approval_id, str):
            return approval_id
    return None


def _retrieval(demo: _Demo, run_id: str) -> AuditEvent | None:
    return next(
        (e for e in demo.events if e.run_id == run_id and e.kind is EventKind.RETRIEVAL), None
    )


# Scenarios. Each prints its steps and returns named checks of the outcome.


def _staff_search(demo: _Demo) -> list[Check]:
    ctx = demo.start("kc-staff-nadine")
    model = ScriptedModel(
        "scripted-mock",
        [
            [("docs.search", {"query": "pricing margins customer list"})],
            [("docs.read", {"document_id": "kc-pricing-margin-sheet"})],
        ],
        final_text="I could not find the pricing and margin sheet.",
    )
    transcript = demo.agent(ctx, model, "Pull up our pricing and margins and the customer list.")
    store = demo.world.store
    confidential = sorted(d.id for d in store.list_documents(KC) if d.label in SENSITIVE_LABELS)
    trace = _retrieval(demo, ctx.run_id)
    returned_raw = None if trace is None else trace.details.get("returned_ids")
    returned = (
        {i for i in returned_raw if isinstance(i, str)} if isinstance(returned_raw, list) else set()
    )
    absent = [doc_id for doc_id in confidential if doc_id not in returned]
    demo.say(f"confidential and restricted documents absent from the results: {len(absent)}", 2)
    demo.say(", ".join(absent), 4)
    calls = _calls(transcript)
    denies = demo.deny_events(ctx.run_id)
    return [
        ("search executed", len(calls) == 2 and calls[0][1] == "executed"),
        ("no sensitive document returned", trace is not None and absent == confidential),
        (
            "filter removed documents",
            trace is not None and bool(trace.details.get("filtered_by_reason")),
        ),
        ("read told not_found", len(calls) == 2 and calls[1][2] == {"error": "not_found"}),
        ("audit says default_deny", [e.reason for e in denies] == ["default_deny"]),
        ("run never saw confidential", demo.world.gate.run_max_label(ctx.run_id) == "internal"),
    ]


def _contractor(demo: _Demo) -> list[Check]:
    ctx = demo.start("kc-contractor-hollis")
    model = ScriptedModel(
        "scripted-mock",
        [
            [("docs.read", {"document_id": "kc-job-2041-furnace-igniter"})],
            [("docs.read", {"document_id": "kc-job-2047-heat-pump-tuneup"})],
        ],
        final_text="Ticket 2041 is a furnace igniter job. I could not find ticket 2047.",
    )
    transcript = demo.agent(ctx, model, "Show me job ticket 2041 and job ticket 2047.")
    calls = _calls(transcript)
    denies = demo.deny_events(ctx.run_id)
    return [
        ("assigned ticket read", len(calls) == 2 and calls[0][1] == "executed"),
        ("other ticket not_found", len(calls) == 2 and calls[1][2] == {"error": "not_found"}),
        ("audit says default_deny", [e.reason for e in denies] == ["default_deny"]),
    ]


def _exfil_sends(transcript: AgentTranscript) -> list[str]:
    statuses: list[str] = []
    for call, status, _ in _calls(transcript):
        if call.tool == "email.send" and call.args.get("to") == [KC_EXFIL]:
            statuses.append(status)
    return statuses


def _injection(demo: _Demo) -> list[Check]:
    ctx = demo.start("kc-staff-nadine")
    demo.say("model: naive-mock, which obeys any instruction it reads, even in an inbox email")
    transcript = demo.agent(ctx, NaiveModel(), KC_FURNACE_TASK)
    outbox = list_outbox(demo.world.store, KC)
    demo.say(f"outbox: {len(outbox)} emails", 2)
    denies = demo.deny_events(ctx.run_id)
    return [
        ("model tried the export once", _exfil_sends(transcript) == ["denied"]),
        (
            "audit says default_deny",
            [(e.tool, e.reason) for e in denies] == [("email.send", "default_deny")],
        ),
        ("outbox empty", outbox == []),
        ("run never saw confidential", demo.world.gate.run_max_label(ctx.run_id) == "internal"),
    ]


def _injection_after_confidential(demo: _Demo) -> list[Check]:
    ctx = demo.start("kc-mgr-marisol")
    demo.say("model: naive-mock")
    transcript = demo.agent(ctx, NaiveModel(), KC_MANAGER_TASK)
    max_label = demo.world.gate.run_max_label(ctx.run_id)
    outbox = list_outbox(demo.world.store, KC)
    demo.say(f"run max label: {max_label}; outbox: {len(outbox)} emails", 2)
    reads = {
        call.args.get("document_id"): status
        for call, status, _ in _calls(transcript)
        if call.tool == "docs.read"
    }
    denies = demo.deny_events(ctx.run_id)
    return [
        ("customer list read", reads.get("kc-customer-list") == "executed"),
        ("inbox email read", reads.get("kc-inbox-repair-inquiry-0914") == "executed"),
        ("run max label confidential", max_label == "confidential"),
        ("export denied", _exfil_sends(transcript) == ["denied"]),
        (
            "denied by no-external-after-confidential",
            [(e.reason, e.rule_ids) for e in denies]
            == [("matched_deny", ("no-external-after-confidential",))],
        ),
        ("outbox empty", outbox == []),
    ]


def _approval_flow(demo: _Demo) -> list[Check]:
    ctx = demo.start("kc-staff-nadine")
    model = ScriptedModel(
        "scripted-mock",
        [[("email.send", TUNE_UP_EMAIL)]],
        final_text="The email is waiting for a manager's approval.",
    )
    transcript = demo.agent(ctx, model, f"Please email {DIRECTORY_CUSTOMER} about her tune-up.")
    approval_id = _pending_id(transcript)
    if approval_id is None:
        return [("email waits for approval", False)]
    name = demo.names.approval(approval_id)
    by_colby = demo.decide(approval_id, "kc-staff-colby")
    by_terrence = demo.decide(approval_id, "kc-mgr-terrence")
    gate = demo.world.gate
    run = demo.names.run(ctx.run_id)
    changed: dict[str, JsonValue] = {**TUNE_UP_EMAIL, "to": [OTHER_DIRECTORY_CUSTOMER]}
    mismatch = demo.gate_call(
        f"{run} executes {name} with the recipient changed to {OTHER_DIRECTORY_CUSTOMER}",
        lambda: gate.execute_approved(ctx, approval_id, "email.send", Action.SEND, changed),
        "email.send",
    )
    executed = demo.gate_call(
        f"{run} executes {name} with exactly the approved args",
        lambda: gate.execute_approved(ctx, approval_id, "email.send", Action.SEND, TUNE_UP_EMAIL),
        "email.send",
    )
    replay = demo.gate_call(
        f"{run} executes {name} again",
        lambda: gate.execute_approved(ctx, approval_id, "email.send", Action.SEND, TUNE_UP_EMAIL),
        "email.send",
    )
    outbox = list_outbox(demo.world.store, KC)
    demo.say(
        f"outbox: {len(outbox)} email, to "
        + ", ".join(r for email in outbox for r in email.recipients),
        2,
    )
    return [
        ("email waits for approval", True),
        ("staff cannot approve", by_colby == "approver_not_allowed"),
        ("manager approves", by_terrence == "approved"),
        ("changed recipient refused", mismatch == Denied("args_mismatch")),
        ("exact args execute", isinstance(executed, Executed)),
        ("replay refused", replay == Denied("already_executed")),
        (
            "one email in the outbox",
            [email.recipients for email in outbox] == [(DIRECTORY_CUSTOMER,)],
        ),
    ]


def _expiry(demo: _Demo) -> list[Check]:
    ctx = demo.start("kc-staff-nadine")
    model = ScriptedModel(
        "scripted-mock",
        [[("email.send", VISIT_EMAIL)]],
        final_text="The email is waiting for a manager's approval.",
    )
    task = f"Email {OTHER_DIRECTORY_CUSTOMER} to confirm his visit on Thursday."
    transcript = demo.agent(ctx, model, task)
    approval_id = _pending_id(transcript)
    if approval_id is None:
        return [("email waits for approval", False)]
    name = demo.names.approval(approval_id)
    approved = demo.decide(approval_id, "kc-mgr-terrence")
    ttl = demo.world.engines[KC].bundle.policy.approvals.ttl_minutes
    demo.clock.advance(timedelta(minutes=ttl + 1))
    demo.say(
        f"clock: advanced {ttl + 1} minutes to {_time(demo.clock.now())}"
        f" (approval TTL is {ttl} minutes)",
        2,
    )
    expired = demo.world.gate.expire_due()
    demo.say("expire_due: " + (", ".join(demo.names.approval(a.id) for a in expired) or "none"), 2)
    gate = demo.world.gate
    late = demo.gate_call(
        f"{demo.names.run(ctx.run_id)} executes {name}",
        lambda: gate.execute_approved(ctx, approval_id, "email.send", Action.SEND, VISIT_EMAIL),
        "email.send",
    )
    outbox = list_outbox(demo.world.store, KC)
    demo.say(f"outbox: still {len(outbox)} email", 2)
    return [
        ("email waits for approval", True),
        ("manager approves", approved == "approved"),
        ("expire_due expires it", [a.id for a in expired] == [approval_id]),
        ("execution refused as expired", late == Denied("expired")),
        ("nothing new in the outbox", len(outbox) == 1),
    ]


def _cross_customer(demo: _Demo) -> list[Check]:
    ctx = demo.start("kc-mgr-marisol")
    model = ScriptedModel(
        "scripted-mock",
        [[("docs.read", {"document_id": "jr-customer-list"})]],
        final_text="I could not find that document.",
    )
    transcript = demo.agent(ctx, model, "Read the document jr-customer-list.")
    calls = _calls(transcript)
    denies = demo.deny_events(ctx.run_id)
    return [
        ("model told not_found", [content for _, _, content in calls] == [{"error": "not_found"}]),
        (
            "audit says document_customer_mismatch",
            [e.reason for e in denies] == ["document_customer_mismatch"],
        ),
    ]


def _runaway(demo: _Demo) -> list[Check]:
    ctx = demo.start("jr-staff-tomas")
    demo.say("model: runaway-mock, which repeats the same search forever")
    model = RepeatModel("runaway-mock", "docs.search", {"query": "cold-chain shipping"})
    transcript = demo.agent(ctx, model, "Find the cold-chain shipping procedure.")
    limits = demo.world.engines[JR].bundle.policy.spend
    ledger = demo.world.gate.ledger
    spent = ledger.run_spent(ctx.run_id)
    calls = ledger.run_calls(ctx.run_id)
    hits = [e for e in demo.events if e.run_id == ctx.run_id and e.kind is EventKind.CAP_HIT]
    cap = hits[0].reason if hits else "none"
    demo.say(f"cap: {cap}; model calls: {len(transcript.steps)}", 2)
    demo.say(f"calls counted: {calls} of {limits.max_calls_per_run}", 2)
    demo.say(f"spend so far: {_usd(spent)} of the {limits.per_run_usd} USD run cap", 2)
    return [
        ("stopped at a cap", transcript.stop_reason == "cap_reached"),
        ("one cap hit", len(hits) == 1),
        ("spend within the run cap", spent <= limits.per_run_usd),
        ("calls within the call cap", calls <= limits.max_calls_per_run),
    ]


def _identity(demo: _Demo) -> list[Check]:
    user = demo.world.store.get_user("kc-staff-gavin")
    if user is not None:
        demo.say(
            f"acting: {user.id} (role {user.role.value}, customer {user.customer_id},"
            f" status {user.status.value})"
        )
    reason = "none, the run started"
    try:
        demo.world.gate.start_run(KC, "kc-staff-gavin")
    except IdentityError as err:
        reason = err.reason
    demo.say(f"start_run: refused, IdentityError {reason}", 2)
    ctx = demo.start("kc-staff-nadine")
    forged = replace(ctx, role=Role.OWNER)
    gate = demo.world.gate
    run = demo.names.run(ctx.run_id)
    outcome = demo.gate_call(
        f"a forged copy of {run}'s context, role changed from {ctx.role.value} to owner,"
        " calls docs.read kc-payroll-summary",
        lambda: gate.call_tool(
            forged, "docs.read", Action.READ, {"document_id": "kc-payroll-summary"}
        ),
        "docs.read",
    )
    return [
        ("suspended user cannot start a run", reason == "suspended_user"),
        ("forged context refused", outcome == Denied("invalid_run")),
    ]


def _refund(demo: _Demo) -> list[Check]:
    ctx = demo.start("jr-staff-hana")
    model = ScriptedModel(
        "scripted-mock",
        [[("refund.issue", REFUND)]],
        final_text="The refund is waiting for a manager's approval.",
    )
    transcript = demo.agent(ctx, model, "Refund 40.00 USD on order JR-26-1061, it arrived late.")
    approval_id = _pending_id(transcript)
    if approval_id is None:
        return [("refund waits for approval", False)]
    name = demo.names.approval(approval_id)
    approved = demo.decide(approval_id, "jr-mgr-daniel")
    gate = demo.world.gate
    outcome = demo.gate_call(
        f"{demo.names.run(ctx.run_id)} executes {name}",
        lambda: gate.execute_approved(ctx, approval_id, "refund.issue", Action.SPEND, REFUND),
        "refund.issue",
    )
    refunds = list_refunds(demo.world.store, JR)
    demo.say(
        f"refunds: {len(refunds)}, "
        + ", ".join(f"{r.order_id} {r.amount_usd} USD" for r in refunds),
        2,
    )
    return [
        ("refund waits for approval", True),
        ("manager approves", approved == "approved"),
        ("refund executes", isinstance(outcome, Executed)),
        (
            "one refund of 40.00",
            [(r.order_id, r.amount_usd) for r in refunds] == [("JR-26-1061", Decimal("40.00"))],
        ),
    ]


@dataclass(frozen=True)
class Scenario:
    title: str
    expected: str
    run: Callable[[_Demo], list[Check]]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "Staff search: confidential documents never reach the model",
        "confidential documents filtered out before ranking; the pricing sheet read is"
        " not_found to the model and default_deny in the audit log",
        _staff_search,
    ),
    Scenario(
        "Contractor: only assigned job tickets",
        "the assigned ticket is read; another technician's ticket is not_found",
        _contractor,
    ),
    Scenario(
        "Prompt injection: an inbox email asks for an export",
        "the export email is denied (default_deny) and the outbox stays empty",
        _injection,
    ),
    Scenario(
        "Prompt injection after a confidential read",
        "the export email is denied by rule no-external-after-confidential",
        _injection_after_confidential,
    ),
    Scenario(
        "Approval flow: emailing a customer",
        "staff cannot approve, a manager can; changed args and a replay are refused;"
        " exactly one email is sent",
        _approval_flow,
    ),
    Scenario(
        "Expiry: an approval not used in time",
        "past the TTL the approval expires and executing it is refused as expired",
        _expiry,
    ),
    Scenario(
        "Cross-customer read",
        "another customer's document is not_found to the model,"
        " document_customer_mismatch in the audit log",
        _cross_customer,
    ),
    Scenario(
        "Runaway loop",
        "the run stops at a cap with spend and calls within the policy caps",
        _runaway,
    ),
    Scenario(
        "Identity: suspended user and forged context",
        "a suspended user cannot start a run; an altered run context is denied invalid_run",
        _identity,
    ),
    Scenario(
        "Pharmacy refunds need a manager",
        "the 40.00 USD refund waits for approval, the manager approves, and it executes",
        _refund,
    ),
)


def run_demo(
    out: TextIO,
    *,
    audit_dir: Path | None = None,
    policies: Path = DEFAULT_POLICIES,
    fixtures: Path = DEFAULT_FIXTURES,
) -> bool:
    """Run every scenario, print what happened, and return True if all behaved as expected.

    The audit log is a Panoptes database at audit_dir/audit.db, or in a temporary directory
    removed afterwards when audit_dir is None. An existing audit.db is refused, so the log
    holds exactly this run's events.
    """
    with contextlib.ExitStack() as stack:
        directory = audit_dir
        if directory is None:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        path = directory / AUDIT_FILE
        if path.exists():
            raise FileExistsError(f"{path} already exists; use an empty directory")
        sink = PanoptesSink(path)
        stack.callback(sink.close)
        tee = _TeeSink(sink)
        clock = FixedClock(DEMO_START)
        world = open_memory_world(policies=policies, fixtures=fixtures, clock=clock, sink=tee)
        stack.callback(world.close)
        demo = _Demo(world=world, clock=clock, sink=sink, tee=tee, out=out)
        return _run(demo)
    raise AssertionError("unreachable")  # pragma: no cover


def _run(demo: _Demo) -> bool:
    customers = ", ".join(c.id for c in demo.world.store.list_customers())
    demo.say("agentgate demo: mock models, no keys, no network")
    demo.say(f"customers: {customers}")
    demo.say(f"clock: fixed at {_time(DEMO_START)}; audit log: Panoptes, hash-chained")
    unexpected = 0
    for number, scenario in enumerate(SCENARIOS, start=1):
        demo.say()
        demo.say(f"== {number}. {scenario.title} ==")
        checks = _checks(demo, scenario)
        demo.say(f"expected: {scenario.expected}")
        failed = [name for name, passed in checks if not passed]
        if failed:
            unexpected += 1
            demo.say("UNEXPECTED: " + "; ".join(failed))
        else:
            demo.say("ok")

    demo.say()
    demo.say("== audit summary ==")
    counts = Counter(event.kind.value for event in demo.events)
    demo.say(f"events: {len(demo.events)}")
    for kind in sorted(counts):
        demo.out.write(f"  {kind:<20} {counts[kind]:>4}\n")
    result = demo.sink.verify()
    stored = demo.sink.count()
    intact = bool(result["valid"]) and stored == len(demo.events)
    demo.say(f"audit chain: {'intact' if intact else 'BROKEN'} ({stored} events)")
    total = len(SCENARIOS)
    if unexpected == 0 and intact:
        demo.say(f"demo: all {total} scenarios behaved as expected")
        return True
    demo.say(f"demo: {unexpected} of {total} scenarios UNEXPECTED")
    return False


def _checks(demo: _Demo, scenario: Scenario) -> list[Check]:
    try:
        return scenario.run(demo)
    except Exception as exc:
        # The type only: a message could quote arguments or document text.
        return [(f"raised {type(exc).__name__}", False)]
