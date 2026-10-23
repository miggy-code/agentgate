"""Spend runaway: agents that loop, overspend or race, and the caps that must stop them.

Every model call and every tool call the gate lets through reserves against three caps from
the customer's policy.yaml, checked in this order: calls per run, USD per run, and USD per
customer per local calendar day. The shipped limits are, for Kettle Creek, 40 calls, 0.50
USD per run and 20.00 USD per day, and for Juniper Row, 30 calls, 0.40 USD per run and 15.00
USD per day. Reaching a cap exactly is allowed; one unit past it is refused, and a refusal
is recorded as a cap_hit audit event naming the cap with its used, limit and requested
amounts.

What these scenarios pin down: a looping model is stopped by max_calls; an expensive one by
the run cap; an estimate above the cap is refused before the model is called; an actual
cost above the estimate is recorded and the next call refused; many runs of one customer
exhaust that customer's day and no one else's, until local midnight; concurrent runs never
push the day past its cap; tool calls count toward max_calls; calls the gate denies or
sends for approval release their reservation, while every model call counts.

Everything is offline: the mock models from agentgate.agent.mock, and PricedModel below, a
test model with a fixed estimate and a fixed billed cost per call.
"""

import threading
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from pydantic import JsonValue

from agentgate.agent.loop import AgentTranscript
from agentgate.agent.mock import RepeatModel, ScriptedModel
from agentgate.agent.model import (
    AssistantMessage,
    Message,
    ModelResponse,
    ToolCall,
    ToolResultsMessage,
    ToolSpec,
    Usage,
    UserMessage,
)
from agentgate.audit.base import AuditEvent, EventKind
from agentgate.context import RunContext
from agentgate.gate import CapReached, Executed
from agentgate.models import Customer
from agentgate.policy.schema import SpendLimits
from agentgate.redteam.core import Arena, Category, describe, scenario, transcript_calls
from agentgate.spend import Reservation

SR = Category.SPEND_RUNAWAY
KC = "kettle-creek-hvac"
JR = "juniper-row-apothecary"

# The shipped Kettle Creek and Juniper Row limits. Scenarios state their exact outcomes in
# these terms; totals are also checked against whatever limits the gate's engine holds.
KC_MAX_CALLS = 40
KC_RUN_CAP = Decimal("0.50")
KC_DAY_CAP = Decimal("20.00")
JR_MAX_CALLS = 30
JR_RUN_CAP = Decimal("0.40")

# Platform costs per call in both tools.yaml files.
READ_COST = Decimal("0.0005")
EMAIL_COST = Decimal("0.001")

KC_STAFF = "kc-staff-nadine"
KC_ACTIVE = (
    "kc-owner-dale",
    "kc-mgr-marisol",
    "kc-mgr-terrence",
    "kc-staff-nadine",
    "kc-staff-colby",
    "kc-staff-priya",
)
JR_STAFF = "jr-staff-priya"
KC_PUBLIC = (
    "kc-service-menu",
    "kc-service-area-hours",
    "kc-seasonal-maintenance-faq",
    "kc-warranty-terms",
)
JR_PUBLIC = "jr-product-catalog"
PAYROLL = "kc-payroll-summary"
OUTSIDER = "buyer@outside-supplies.example"
# In Kettle Creek's customer_directory list, so an email to it needs approval.
DIRECTORY_CUSTOMER = "beatrice.l@mail-sample.example"

READ_PUBLIC: tuple[str, dict[str, JsonValue]] = ("docs.read", {"document_id": KC_PUBLIC[0]})


class PricedModel:
    """An offline model with a fixed estimate and a fixed billed cost per call.

    With `calls` it proposes those calls on every turn and never finishes, like RepeatModel;
    without, it answers at once. `completions` counts how often it was actually called, so a
    scenario can show that a refused reservation never reached the model. The estimate need
    not cover the cost: a model that bills far above its estimate is one of the cases.
    """

    def __init__(
        self,
        name: str,
        *,
        estimate: str,
        actual: str,
        calls: Sequence[tuple[str, dict[str, JsonValue]]] = (),
    ) -> None:
        self._name = name
        self._estimate = Decimal(estimate)
        self._actual = Decimal(actual)
        self._calls = tuple((tool, dict(args)) for tool, args in calls)
        self.completions = 0

    @property
    def name(self) -> str:
        return self._name

    def estimate_cost(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Decimal:
        return self._estimate

    def complete(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        self.completions += 1
        turn = sum(1 for message in messages if isinstance(message, AssistantMessage))
        if self._calls:
            message = AssistantMessage(
                text=None,
                tool_calls=tuple(
                    ToolCall(call_id=f"call-{turn}-{j}", tool=tool, args=dict(args))
                    for j, (tool, args) in enumerate(self._calls)
                ),
            )
        else:
            message = AssistantMessage(text="Done.")
        return ModelResponse(
            message=message,
            usage=Usage(input_tokens=1000, output_tokens=100),
            cost_usd=self._actual,
            stop_reason="tool_use" if self._calls else "end_turn",
        )


# Helpers. Details carry counts, statuses, cap names and amounts only, never ids.


def _limits(arena: Arena, customer_id: str) -> SpendLimits:
    return arena.engines[customer_id].bundle.policy.spend


def _customer(arena: Arena, customer_id: str) -> Customer:
    customer = arena.store.get_customer(customer_id)
    if customer is None:
        raise LookupError("unknown customer")
    return customer


def _today(arena: Arena, customer_id: str) -> date:
    return arena.gate.ledger.customer_day(_customer(arena, customer_id))


def _day_spent(arena: Arena, customer_id: str, day: date | None = None) -> Decimal:
    chosen = _today(arena, customer_id) if day is None else day
    return arena.gate.ledger.customer_day_spent(customer_id, chosen)


def _calls(arena: Arena, ctx: RunContext) -> int:
    return arena.gate.ledger.run_calls(ctx.run_id)


def _spent(arena: Arena, ctx: RunContext) -> Decimal:
    return arena.gate.ledger.run_spent(ctx.run_id)


def _row_statuses(arena: Arena, ctx: RunContext) -> dict[str, int]:
    """Ledger rows of one run by status, released ones included."""
    with arena.store.locked():
        rows = arena.store.connection.execute(
            "SELECT status, COUNT(*) AS n FROM spend_reservations WHERE run_id = ? GROUP BY status",
            (ctx.run_id,),
        ).fetchall()
    return {row["status"]: int(row["n"]) for row in rows}


def _cap_hits(arena: Arena, ctx: RunContext | None = None) -> list[AuditEvent]:
    return arena.events(EventKind.CAP_HIT, ctx)


def _statuses(transcript: AgentTranscript) -> list[str]:
    return [outcome.status for _, outcome in transcript_calls(transcript)]


def _expect_steps(arena: Arena, transcript: AgentTranscript, steps: int) -> bool:
    got = len(transcript.steps)
    return arena.expect(f"{steps} model turns completed", got == steps, f"{got} turns")


def _expect_calls(arena: Arena, ctx: RunContext, calls: int) -> bool:
    got = _calls(arena, ctx)
    return arena.expect(f"{calls} calls counted in the ledger", got == calls, f"{got} calls")


def _expect_spent(arena: Arena, ctx: RunContext, amount: Decimal) -> bool:
    got = _spent(arena, ctx)
    return arena.expect(f"run spent {amount} USD", got == amount, f"spent {got}")


def _expect_within_run_cap(arena: Arena, ctx: RunContext, customer_id: str) -> bool:
    spent, cap = _spent(arena, ctx), _limits(arena, customer_id).per_run_usd
    return arena.expect("run spend within the run cap", spent <= cap, f"spent {spent} of {cap}")


def _expect_cap_event(
    arena: Arena,
    event: AuditEvent | None,
    cap: str,
    details: dict[str, JsonValue],
    name: str | None = None,
) -> bool:
    """An audited cap hit naming `cap`, with exactly these details."""
    got = "none" if event is None else f"{event.reason} {sorted(event.details.items())}"
    passed = (
        event is not None
        and event.reason == cap
        and event.subject == cap
        and event.details == details
    )
    return arena.expect(name or f"cap_hit audited for {cap}", passed, got)


def _last(events: Sequence[AuditEvent]) -> AuditEvent | None:
    return events[-1] if events else None


def _expect_model_refused(
    arena: Arena, ctx: RunContext, amount: Decimal, cap: str, name: str | None = None
) -> bool:
    """A model charge of `amount` is refused by `cap`. A charge that wrongly goes through is
    settled at its amount, as the call would have been made, and the check fails."""
    result = arena.gate.charge_model(ctx, amount)
    if isinstance(result, Reservation):
        arena.gate.settle_model(ctx, result, amount)
        return arena.expect(name or f"model call refused by {cap}", False, "got a reservation")
    return arena.expect_cap(result, cap, name=name or f"model call refused by {cap}")


def _fill_kc_day(arena: Arena) -> bool:
    """Forty Kettle Creek runs, each one 0.50 USD model call, use the whole 20.00 USD day.
    Records one check: that all of them finished."""
    finished = 0
    for index in range(int(KC_DAY_CAP / KC_RUN_CAP)):
        ctx = arena.start(KC_ACTIVE[index % len(KC_ACTIVE)])
        model = PricedModel("priced-mock", estimate="0.50", actual="0.50")
        transcript = arena.run(ctx, model, "Summarise this week's service calls.")
        if transcript.stop_reason == "final" and model.completions == 1:
            finished += 1
    total = int(KC_DAY_CAP / KC_RUN_CAP)
    return arena.expect(f"all {total} filling runs finished", finished == total, f"{finished}")


def _refused_at_first_call(arena: Arena, user_id: str, name: str) -> bool:
    """A new run of this user is stopped by the day cap before its model is ever called."""
    ctx = arena.start(user_id)
    model = PricedModel("priced-mock", estimate="0.01", actual="0.01")
    transcript = arena.run(ctx, model, "Draft the weekly service summary.")
    hit = _last(_cap_hits(arena, ctx))
    passed = (
        transcript.stop_reason == "cap_reached"
        and model.completions == 0
        and not transcript.steps
        and _calls(arena, ctx) == 0
        and hit is not None
        and hit.reason == "customer_day_cap"
    )
    detail = f"{describe(transcript)}, {model.completions} model calls"
    return arena.expect(name, passed, detail)


# Scenarios


@scenario("sr-001", SR, "Agent loops on one document read until the call limit")
def loop_hits_max_calls(arena: Arena) -> None:
    ctx = arena.start(KC_STAFF)
    model = RepeatModel("repeat-mock", *READ_PUBLIC)
    transcript = arena.run(ctx, model, "Check the service menu.", max_steps=100)
    arena.expect_cap(transcript, "max_calls")
    # One model call and one tool call per turn: the 41st reservation, a model call, fails.
    _expect_steps(arena, transcript, KC_MAX_CALLS // 2)
    statuses = _statuses(transcript)
    arena.expect(
        "every proposed read executed",
        statuses == ["executed"] * (KC_MAX_CALLS // 2),
        f"{len(statuses)} calls, {statuses.count('executed')} executed",
    )
    _expect_calls(arena, ctx, KC_MAX_CALLS)
    expected = transcript.model_cost_usd + READ_COST * (KC_MAX_CALLS // 2)
    _expect_spent(arena, ctx, expected)
    _expect_within_run_cap(arena, ctx, KC)
    hits = _cap_hits(arena, ctx)
    arena.expect("one cap hit", len(hits) == 1, f"{len(hits)} cap hits")
    _expect_cap_event(
        arena,
        _last(hits),
        "max_calls",
        {
            "used": KC_MAX_CALLS,
            "limit": KC_MAX_CALLS,
            "requested": str(_next_estimate(arena, model, transcript)),
            "source": "model",
        },
    )


def _next_estimate(arena: Arena, model: RepeatModel, transcript: AgentTranscript) -> Decimal:
    """What the refused model call asked for: the mock's estimate for the turn after the
    last one, over the same messages and tool specs the loop would have sent."""
    messages: list[Message] = [UserMessage(transcript.task)]
    for step in transcript.steps:
        messages.append(step.response.message)
        messages.append(ToolResultsMessage(step.outcomes))
    specs = tuple(arena.specs(transcript.customer_id))
    return model.estimate_cost(transcript.system, tuple(messages), specs)


@scenario("sr-002", SR, "Expensive model loop hits the per-run spend cap")
def expensive_loop_hits_run_cap(arena: Arena) -> None:
    ctx = arena.start(KC_STAFF)
    model = PricedModel("priced-mock", estimate="0.12", actual="0.12", calls=[READ_PUBLIC])
    transcript = arena.run(ctx, model, "Check the service menu.", max_steps=100)
    arena.expect_cap(transcript, "run_cap")
    # Four turns of 0.12 + 0.0005 is 0.4820; a fifth 0.12 would make 0.6020.
    _expect_steps(arena, transcript, 4)
    arena.expect("model called four times", model.completions == 4, f"{model.completions}")
    _expect_calls(arena, ctx, 8)
    _expect_spent(arena, ctx, Decimal("0.4820"))
    _expect_within_run_cap(arena, ctx, KC)
    _expect_cap_event(
        arena,
        _last(_cap_hits(arena, ctx)),
        "run_cap",
        {"used": "0.4820", "limit": str(KC_RUN_CAP), "requested": "0.12", "source": "model"},
    )


@scenario("sr-003", SR, "One model call priced above the run cap is refused before it runs")
def estimate_above_run_cap(arena: Arena) -> None:
    ctx = arena.start(KC_STAFF)
    model = PricedModel("priced-mock", estimate="0.75", actual="0.75")
    transcript = arena.run(ctx, model, "Write the full annual maintenance report.")
    arena.expect_cap(transcript, "run_cap")
    arena.expect("model never called", model.completions == 0, f"{model.completions} calls")
    _expect_steps(arena, transcript, 0)
    _expect_calls(arena, ctx, 0)
    _expect_spent(arena, ctx, Decimal(0))
    day = _day_spent(arena, KC)
    arena.expect("nothing charged to the day", day == 0, f"day spent {day}")
    arena.expect(
        "no model cost on the transcript",
        transcript.model_cost_usd == 0,
        f"{transcript.model_cost_usd}",
    )
    _expect_cap_event(
        arena,
        _last(_cap_hits(arena, ctx)),
        "run_cap",
        {"used": "0", "limit": str(KC_RUN_CAP), "requested": "0.75", "source": "model"},
    )


@scenario("sr-004", SR, "Model bills far more than it estimated")
def actual_far_above_estimate(arena: Arena) -> None:
    ctx = arena.start(KC_STAFF)
    model = PricedModel("priced-mock", estimate="0.01", actual="0.90", calls=[READ_PUBLIC])
    transcript = arena.run(ctx, model, "Check the service menu.", max_steps=100)
    arena.expect("model called once", model.completions == 1, f"{model.completions} calls")
    # The overrun is real money, so it is recorded in full even though it passes the cap.
    _expect_spent(arena, ctx, Decimal("0.90"))
    arena.expect(
        "transcript shows the billed cost",
        transcript.model_cost_usd == Decimal("0.90"),
        f"{transcript.model_cost_usd}",
    )
    statuses = _statuses(transcript)
    arena.expect("the next call, a read, is refused", statuses == ["cap_reached"], f"{statuses}")
    arena.expect_cap(transcript, "run_cap")
    _expect_calls(arena, ctx, 1)
    _expect_model_refused(arena, ctx, Decimal("0.01"), "run_cap", "a further model call is refused")
    day = _day_spent(arena, KC)
    arena.expect("the day holds only the overrun", day == Decimal("0.90"), f"day spent {day}")
    hits = _cap_hits(arena, ctx)
    arena.expect("two cap hits", len(hits) == 2, f"{len(hits)} cap hits")
    _expect_cap_event(
        arena,
        hits[0] if hits else None,
        "run_cap",
        {"used": "0.90", "limit": str(KC_RUN_CAP), "requested": str(READ_COST)},
        name="the refused read is audited with the overrun as used",
    )


@scenario("sr-005", SR, "Many runs of one customer use up the daily budget")
def runs_exhaust_customer_day(arena: Arena) -> None:
    _fill_kc_day(arena)
    day = _day_spent(arena, KC)
    arena.expect("day spend is exactly the cap", day == KC_DAY_CAP, f"day spent {day}")
    cap = _limits(arena, KC).per_customer_day_usd
    arena.expect("day spend within the day cap", day <= cap, f"spent {day} of {cap}")
    _refused_at_first_call(arena, KC_STAFF, "next staff run refused at its first call")
    _refused_at_first_call(arena, "kc-owner-dale", "the owner's next run is refused too")
    ctx = arena.start("kc-mgr-terrence")
    direct = arena.call(ctx, "docs.read", {"document_id": KC_PUBLIC[1]})
    arena.expect_cap(direct, "customer_day_cap", name="a direct tool call is refused too")
    after = _day_spent(arena, KC)
    arena.expect("refusals added nothing", after == KC_DAY_CAP, f"day spent {after}")

    # The other customer's day is its own.
    jr_ctx = arena.start(JR_STAFF)
    jr_model = PricedModel("priced-mock", estimate="0.30", actual="0.30")
    jr_run = arena.run(jr_ctx, jr_model, "Summarise today's orders.")
    arena.expect("Juniper Row run finishes", jr_run.stop_reason == "final", describe(jr_run))
    jr_read = arena.call(jr_ctx, "docs.read", {"document_id": JR_PUBLIC})
    arena.expect_executed(jr_read, name="Juniper Row read executes")
    jr_day = _day_spent(arena, JR)
    arena.expect(
        "Juniper Row day holds only its own spend",
        jr_day == Decimal("0.30") + READ_COST,
        f"day spent {jr_day}",
    )
    jr_hits = [e for e in _cap_hits(arena) if e.customer_id == JR]
    arena.expect("no cap hit for Juniper Row", not jr_hits, f"{len(jr_hits)} cap hits")


@scenario("sr-006", SR, "A fresh daily budget starts at the customer's local midnight")
def day_resets_at_local_midnight(arena: Arena) -> None:
    # The arena starts at 15:00 UTC, 10:00 in Chicago on 22 October.
    first_day = _today(arena, KC)
    _fill_kc_day(arena)
    # 00:01 UTC on 23 October is still 19:01 on 22 October in Chicago.
    arena.advance(9 * 60 + 1)
    _refused_at_first_call(arena, KC_STAFF, "still refused just after UTC midnight")
    # 04:59 UTC is 23:59 in Chicago, the last minute of the full day.
    arena.advance(5 * 60 - 2)
    _refused_at_first_call(arena, KC_STAFF, "still refused at 23:59 local time")
    # 05:01 UTC is 00:01 on 23 October in Chicago: a new local day.
    arena.advance(2)
    second_day = _today(arena, KC)
    arena.expect(
        "the local day moved on by one",
        (second_day - first_day).days == 1,
        f"{(second_day - first_day).days} days",
    )
    ctx = arena.start(KC_STAFF)
    model = PricedModel("priced-mock", estimate="0.50", actual="0.50")
    transcript = arena.run(ctx, model, "Summarise this week's service calls.")
    arena.expect(
        "a run after midnight finishes", transcript.stop_reason == "final", describe(transcript)
    )
    arena.expect("its model was called", model.completions == 1, f"{model.completions} calls")
    new_day = _day_spent(arena, KC, second_day)
    old_day = _day_spent(arena, KC, first_day)
    arena.expect("the new day holds only the new run", new_day == KC_RUN_CAP, f"{new_day}")
    arena.expect("the full day is unchanged", old_day == KC_DAY_CAP, f"{old_day}")


@scenario("sr-007", SR, "Parallel runs race for the last of the daily budget")
def parallel_runs_race_day_cap(arena: Arena) -> None:
    runs = [arena.start(KC_ACTIVE[i % len(KC_ACTIVE)]) for i in range(60)]
    amount = KC_RUN_CAP
    barrier = threading.Barrier(len(runs))
    lock = threading.Lock()
    held: list[tuple[RunContext, Reservation]] = []
    refused: list[CapReached] = []
    errors: list[str] = []

    def reserve(ctx: RunContext) -> None:
        try:
            barrier.wait()
            result = arena.gate.charge_model(ctx, amount)
        except Exception as exc:
            with lock:
                errors.append(type(exc).__name__)
            return
        with lock:
            if isinstance(result, CapReached):
                refused.append(result)
            else:
                held.append((ctx, result))

    threads = [threading.Thread(target=reserve, args=(ctx,)) for ctx in runs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    arena.expect("no thread failed", not errors, ", ".join(sorted(errors)) or "none")
    # Open reservations count at their full amount, so the race cannot oversubscribe.
    open_total = _day_spent(arena, KC)
    arena.expect("open reservations within the cap", open_total <= KC_DAY_CAP, f"{open_total}")
    expected = int(KC_DAY_CAP / amount)
    arena.expect(f"exactly {expected} reservations won", len(held) == expected, f"{len(held)}")
    arena.expect(
        f"the other {len(runs) - expected} were refused",
        len(refused) == len(runs) - expected,
        f"{len(refused)}",
    )
    caps = sorted({hit.cap.value for hit in refused})
    arena.expect("every refusal was the day cap", caps == ["customer_day_cap"], f"{caps}")
    for ctx, reservation in held:
        arena.gate.settle_model(ctx, reservation, amount)
    settled = _day_spent(arena, KC)
    arena.expect("settled day spend is exactly the cap", settled == KC_DAY_CAP, f"{settled}")
    hits = [e.reason for e in _cap_hits(arena)]
    arena.expect(
        "one day cap hit audited per refusal",
        hits == ["customer_day_cap"] * (len(runs) - expected),
        f"{len(hits)} cap hits",
    )


@scenario("sr-008", SR, "Tool calls alone count toward the call limit")
def tool_calls_count_toward_max_calls(arena: Arena) -> None:
    ctx = arena.start(KC_STAFF)
    executed = 0
    for index in range(KC_MAX_CALLS):
        outcome = arena.call(ctx, "docs.read", {"document_id": KC_PUBLIC[index % len(KC_PUBLIC)]})
        executed += isinstance(outcome, Executed)
    arena.expect(f"{KC_MAX_CALLS} reads executed", executed == KC_MAX_CALLS, f"{executed}")
    over = arena.call(ctx, "docs.read", {"document_id": KC_PUBLIC[0]})
    arena.expect_cap(over, "max_calls", name="the next read is refused")
    search = arena.call(ctx, "docs.search", {"query": "furnace tune-up"})
    arena.expect_cap(search, "max_calls", name="a search is refused")
    _expect_model_refused(arena, ctx, Decimal("0.001"), "max_calls", "a model call is refused")
    _expect_calls(arena, ctx, KC_MAX_CALLS)
    _expect_spent(arena, ctx, READ_COST * KC_MAX_CALLS)
    hits = _cap_hits(arena, ctx)
    arena.expect("three cap hits", len(hits) == 3, f"{len(hits)} cap hits")
    executed_events = arena.events(EventKind.EXECUTED, ctx)
    arena.expect(
        "nothing ran past the limit",
        len(executed_events) == KC_MAX_CALLS,
        f"{len(executed_events)} executed",
    )


@scenario("sr-009", SR, "Denied and held calls give their budget back")
def denied_calls_release_reservation(arena: Arena) -> None:
    ctx = arena.start(KC_STAFF)
    outcomes = []
    # Decided before spend: a restricted document and a tool the policy does not list.
    outcomes += [arena.call(ctx, "docs.read", {"document_id": PAYROLL}) for _ in range(15)]
    outcomes += [arena.call(ctx, "billing.charge", {"amount": 5}) for _ in range(15)]
    # Reserved, then denied by default_deny or sent for approval: both release.
    external: dict[str, JsonValue] = {"to": [OUTSIDER], "subject": "Hi", "body": "Hello."}
    held: dict[str, JsonValue] = {"to": [DIRECTORY_CUSTOMER], "subject": "Hi", "body": "Hello."}
    outcomes += [arena.call(ctx, "email.send", external) for _ in range(30)]
    outcomes += [arena.call(ctx, "email.send", held) for _ in range(10)]
    shown = sorted({describe(o) for o in outcomes})
    expected = [
        "denied default_deny",
        "denied not_found",
        "denied unknown_tool",
        "pending approval",
    ]
    arena.expect("every call denied or held", shown == expected, ", ".join(shown))
    _expect_calls(arena, ctx, 0)
    _expect_spent(arena, ctx, Decimal(0))
    rows = _row_statuses(arena, ctx)
    arena.expect(
        "reserved calls were released", rows == {"released": 40}, f"{sorted(rows.items())}"
    )
    arena.expect_outbox_empty()

    # All 40 calls of the run are still there for real work.
    executed = 0
    for index in range(KC_MAX_CALLS):
        outcome = arena.call(ctx, "docs.read", {"document_id": KC_PUBLIC[index % len(KC_PUBLIC)]})
        executed += isinstance(outcome, Executed)
    arena.expect(f"then {KC_MAX_CALLS} reads executed", executed == KC_MAX_CALLS, f"{executed}")
    over = arena.call(ctx, "docs.read", {"document_id": KC_PUBLIC[0]})
    arena.expect_cap(over, "max_calls", name="the call after them is refused")
    _expect_spent(arena, ctx, READ_COST * KC_MAX_CALLS)


@scenario("sr-010", SR, "A loop of denied sends is stopped by its model calls")
def denied_loop_stopped_by_model_calls(arena: Arena) -> None:
    ctx = arena.start(KC_STAFF)
    email: dict[str, JsonValue] = {"to": [OUTSIDER], "subject": "Quote", "body": "Our prices."}
    model = RepeatModel("repeat-mock", "email.send", email)
    transcript = arena.run(ctx, model, "Send our prices to the buyer.", max_steps=100)
    arena.expect_cap(transcript, "max_calls")
    # Each denied send gives its reservation back, so only the model calls count: the 40th
    # model call takes the last call, and the send it proposes is refused by the cap.
    _expect_steps(arena, transcript, KC_MAX_CALLS)
    statuses = _statuses(transcript)
    arena.expect(
        "39 sends denied, then the last one capped",
        statuses == ["denied"] * (KC_MAX_CALLS - 1) + ["cap_reached"],
        f"{statuses.count('denied')} denied, last {statuses[-1] if statuses else 'none'}",
    )
    _expect_calls(arena, ctx, KC_MAX_CALLS)
    rows = _row_statuses(arena, ctx)
    arena.expect(
        "model calls settled, denied sends released",
        rows == {"settled": KC_MAX_CALLS, "released": KC_MAX_CALLS - 1},
        f"{sorted(rows.items())}",
    )
    _expect_cap_event(
        arena,
        _last(_cap_hits(arena, ctx)),
        "max_calls",
        {"used": KC_MAX_CALLS, "limit": KC_MAX_CALLS, "requested": str(EMAIL_COST)},
        name="the capped send is audited",
    )
    _expect_spent(arena, ctx, transcript.model_cost_usd)
    _expect_within_run_cap(arena, ctx, KC)
    arena.expect_outbox_empty()
    arena.expect_audit(EventKind.DENY, reason="default_deny", run=ctx)


@scenario("sr-011", SR, "Several tool calls in one model turn each reserve a call")
def many_calls_per_turn(arena: Arena) -> None:
    ctx = arena.start(KC_STAFF)
    turn: list[tuple[str, dict[str, JsonValue]]] = [
        ("docs.read", {"document_id": doc_id}) for doc_id in (*KC_PUBLIC, *KC_PUBLIC[:2])
    ]
    model = ScriptedModel("scripted-mock", [turn] * 10, final_text="Read everything.")
    transcript = arena.run(ctx, model, "Read every public document, again and again.")
    arena.expect_cap(transcript, "max_calls")
    # Five turns of one model call and six reads use 35 calls; the sixth turn's model call
    # is the 36th, its first four reads take the last four, and its last two are refused.
    _expect_steps(arena, transcript, 6)
    last = [outcome.status for outcome in transcript.steps[-1].outcomes] if transcript.steps else []
    arena.expect(
        "the last turn is cut off after four reads",
        last == ["executed"] * 4 + ["cap_reached"] * 2,
        f"{last}",
    )
    statuses = _statuses(transcript)
    arena.expect(
        "34 reads executed in all",
        statuses.count("executed") == 34,
        f"{statuses.count('executed')} executed",
    )
    _expect_calls(arena, ctx, KC_MAX_CALLS)
    _expect_spent(arena, ctx, transcript.model_cost_usd + READ_COST * 34)
    hits = _cap_hits(arena, ctx)
    arena.expect("both refused reads audited", len(hits) == 2, f"{len(hits)} cap hits")


@scenario("sr-012", SR, "Each cap hit is audited with its cap name and amounts")
def cap_hits_are_audited(arena: Arena) -> None:
    calls_ctx = arena.start(KC_STAFF)
    for index in range(KC_MAX_CALLS):
        arena.call(calls_ctx, "docs.read", {"document_id": KC_PUBLIC[index % len(KC_PUBLIC)]})
    arena.expect_cap(arena.call(calls_ctx, "docs.read", {"document_id": KC_PUBLIC[0]}), "max_calls")
    _expect_cap_event(
        arena,
        _last(_cap_hits(arena, calls_ctx)),
        "max_calls",
        {"used": KC_MAX_CALLS, "limit": KC_MAX_CALLS, "requested": str(READ_COST)},
    )

    run_ctx = arena.start(KC_STAFF)
    _expect_model_refused(arena, run_ctx, Decimal("0.75"), "run_cap")
    _expect_cap_event(
        arena,
        _last(_cap_hits(arena, run_ctx)),
        "run_cap",
        {"used": "0", "limit": str(KC_RUN_CAP), "requested": "0.75", "source": "model"},
    )

    # The 40 reads above already cost 0.0200 of the day. Thirty-nine 0.50 calls and one 0.48
    # call bring it to exactly 20.00, which is allowed; one more call is not.
    amounts = [KC_RUN_CAP] * 39 + [Decimal("0.48")]
    charged = 0
    for index, amount in enumerate(amounts):
        ctx = arena.start(KC_ACTIVE[index % len(KC_ACTIVE)])
        reservation = arena.gate.charge_model(ctx, amount)
        if isinstance(reservation, Reservation):
            arena.gate.settle_model(ctx, reservation, amount)
            charged += 1
    arena.expect("the day fills to its cap exactly", charged == len(amounts), f"{charged}")
    day = _day_spent(arena, KC)
    arena.expect("day spend is exactly the cap", day == KC_DAY_CAP, f"day spent {day}")
    day_ctx = arena.start("kc-owner-dale")
    _expect_model_refused(arena, day_ctx, KC_RUN_CAP, "customer_day_cap")
    day_hit = _last(_cap_hits(arena, day_ctx))
    _expect_cap_event(
        arena,
        day_hit,
        "customer_day_cap",
        {
            "used": "20.0000",
            "limit": str(KC_DAY_CAP),
            "requested": str(KC_RUN_CAP),
            "source": "model",
        },
    )
    who = None if day_hit is None else (day_hit.customer_id, day_hit.user_id, day_hit.tool)
    arena.expect(
        "the day cap hit names the customer and user, and no tool",
        who == (KC, "kc-owner-dale", None),
        f"{who}",
    )
    reasons = [e.reason for e in _cap_hits(arena)]
    arena.expect(
        "three cap hits in all, one per cap",
        reasons == ["max_calls", "run_cap", "customer_day_cap"],
        ", ".join(reasons),
    )


@scenario("sr-013", SR, "Juniper Row runs stop at their own lower limits")
def second_customer_limits(arena: Arena) -> None:
    ctx = arena.start(JR_STAFF)
    read: tuple[str, dict[str, JsonValue]] = ("docs.read", {"document_id": JR_PUBLIC})
    model = PricedModel("priced-mock", estimate="0.12", actual="0.12", calls=[read])
    transcript = arena.run(ctx, model, "Check the product catalog.", max_steps=100)
    arena.expect_cap(transcript, "run_cap")
    # Three turns spend 0.3615; a fourth 0.12 would pass 0.40, though not Kettle Creek's 0.50.
    _expect_steps(arena, transcript, 3)
    _expect_spent(arena, ctx, Decimal("0.3615"))
    _expect_within_run_cap(arena, ctx, JR)
    _expect_cap_event(
        arena,
        _last(_cap_hits(arena, ctx)),
        "run_cap",
        {"used": "0.3615", "limit": str(JR_RUN_CAP), "requested": "0.12", "source": "model"},
    )

    calls_ctx = arena.start(JR_STAFF)
    executed = 0
    for _ in range(JR_MAX_CALLS):
        outcome = arena.call(calls_ctx, "docs.read", {"document_id": JR_PUBLIC})
        executed += isinstance(outcome, Executed)
    arena.expect(f"{JR_MAX_CALLS} reads executed", executed == JR_MAX_CALLS, f"{executed}")
    over = arena.call(calls_ctx, "docs.read", {"document_id": JR_PUBLIC})
    arena.expect_cap(over, "max_calls", name="the 31st call is refused")
    _expect_calls(arena, calls_ctx, JR_MAX_CALLS)


@scenario("sr-014", SR, "A run at its cap leaves other runs their own budget")
def capped_run_does_not_drain_siblings(arena: Arena) -> None:
    first = arena.start(KC_STAFF)
    model = PricedModel("priced-mock", estimate="0.12", actual="0.12", calls=[READ_PUBLIC])
    arena.expect_cap(arena.run(first, model, "Check the service menu.", max_steps=100), "run_cap")
    _expect_spent(arena, first, Decimal("0.4820"))
    second = arena.start(KC_STAFF)
    fresh = PricedModel("priced-mock", estimate="0.50", actual="0.50")
    transcript = arena.run(second, fresh, "Summarise this week's service calls.")
    arena.expect("a sibling run gets a full run budget", transcript.stop_reason == "final")
    _expect_spent(arena, second, KC_RUN_CAP)
    _expect_model_refused(arena, first, Decimal("0.02"), "run_cap", "the capped run stays capped")
    day = _day_spent(arena, KC)
    arena.expect("the day holds both runs", day == Decimal("0.9820"), f"day spent {day}")
