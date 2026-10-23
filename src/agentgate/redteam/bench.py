"""`agentgate bench`: the latency the gate adds to one tool call, measured, not estimated.

Everything runs in one in-memory world over the real fixtures and policies, with a fixed
clock. Each section times its subjects with time.perf_counter_ns, a warmup first and then
the timed iterations, one call per subject per iteration, back to back and alternating
which subject goes first. The sections:

1. `engine.decide` alone, for an allowed read and a denied read.
2. An allowed `docs.read` through `gate.call_tool`, against the same adapter's `run` called
   directly with the same (already validated) args, with a MemorySink.
3. The same with a PanoptesSink (durable, hash-chained audit) in a temporary directory.
4. `docs.search` through the gate, against `DocumentIndex.search` for the same engine, run
   and query, without the gate's validation, audit and spend steps.
5. A denied call through the gate: staff reading a restricted document.
6. The allowed `docs.read` again, on a customer day that already holds many reservations,
   to show the customer-day spend check does not slow down as the day fills up.

Added latency is the difference of the two medians (and of the two p95s), not a percentile
of per-call differences. Percentiles are nearest-rank over the timed samples.

The policies are not changed. So that no spend cap trips, each gated subject starts a fresh
run every max_calls_per_run calls and, in sections 2 to 5, moves the clock forward a day at
the same time, so a timed call never sees more earlier reservations in its run or its
customer day than one run may make. Section 6 says how much a busy day costs on top.
"""

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from time import perf_counter_ns

from pydantic import JsonValue

from agentgate.audit.panoptes import PanoptesSink
from agentgate.clock import FixedClock
from agentgate.context import RunContext
from agentgate.gate import Denied, Executed, Gate
from agentgate.models import Action, Document
from agentgate.policy.engine import Decision, DocumentFacts, PolicyEngine, Request
from agentgate.policy.schema import Effect
from agentgate.redteam.core import REDTEAM_START, describe
from agentgate.redteam.report import RunMeta
from agentgate.tools.base import ToolEnv
from agentgate.tools.builtin import DocsRead, DocsReadArgs
from agentgate.world import DEFAULT_FIXTURES, DEFAULT_POLICIES, World, open_memory_world

DEFAULT_ITERATIONS = 2000
DEFAULT_WARMUP = 200
# Reservations already in the customer's day before section 6 times anything.
DEFAULT_BUSY_DAY = 1000
# Section 6 times at most this many calls, since each one adds a reservation to the day.
BUSY_DAY_SAMPLES = 200

CUSTOMER = "kettle-creek-hvac"
STAFF = "kc-staff-nadine"
ALLOWED_DOC = "kc-dispatch-procedure"
RESTRICTED_DOC = "kc-payroll-summary"
SEARCH_QUERY = "furnace igniter clicking noise"
READ_TOOL = "docs.read"
SEARCH_TOOL = "docs.search"


class BenchError(RuntimeError):
    """A benchmarked call did not take the path it is meant to measure."""


@dataclass(frozen=True)
class Stats:
    samples: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float

    @classmethod
    def of(cls, samples_ns: Sequence[int]) -> "Stats":
        if not samples_ns:
            raise ValueError("no samples")
        ordered = sorted(samples_ns)
        return cls(
            samples=len(ordered),
            p50_us=_us(_nearest_rank(ordered, 50)),
            p95_us=_us(_nearest_rank(ordered, 95)),
            p99_us=_us(_nearest_rank(ordered, 99)),
            mean_us=round(sum(ordered) / len(ordered) / 1000, 3),
        )


def _nearest_rank(ordered: Sequence[int], percent: int) -> int:
    # The smallest sample with at least `percent` percent of samples at or below it.
    rank = -(-percent * len(ordered) // 100)
    return ordered[max(rank, 1) - 1]


def _us(ns: int) -> float:
    return round(ns / 1000, 3)


@dataclass(frozen=True)
class Series:
    label: str
    stats: Stats


@dataclass(frozen=True)
class Added:
    """What the measured subject costs on top of `versus`: p50 minus p50, p95 minus p95."""

    versus: str
    p50_us: float
    p95_us: float

    @classmethod
    def between(cls, subject: Stats, baseline: Stats, versus: str) -> "Added":
        return cls(
            versus=versus,
            p50_us=round(subject.p50_us - baseline.p50_us, 3),
            p95_us=round(subject.p95_us - baseline.p95_us, 3),
        )


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    note: str
    series: tuple[Series, ...]
    added: Added | None = None


@dataclass(frozen=True)
class BenchResult:
    warmup: int
    iterations: int
    busy_day: int
    sections: tuple[Section, ...]
    notes: tuple[str, ...]

    def section(self, key: str) -> Section:
        for section in self.sections:
            if section.key == key:
                return section
        raise KeyError(key)


# Timing


@dataclass
class _Subject:
    label: str
    call: Callable[[], object]
    # Raises BenchError unless the result is the path this subject measures. Not timed.
    check: Callable[[object], None]
    # Setup before each call, such as a fresh run. Not timed.
    before: Callable[[], None] = field(default=lambda: None)


def _measure(subjects: Sequence[_Subject], warmup: int, iterations: int) -> list[list[int]]:
    samples: list[list[int]] = [[] for _ in subjects]
    forward = list(range(len(subjects)))
    for i in range(warmup + iterations):
        for j in forward if i % 2 == 0 else forward[::-1]:
            subject = subjects[j]
            subject.before()
            start = perf_counter_ns()
            value = subject.call()
            elapsed = perf_counter_ns() - start
            subject.check(value)
            if i >= warmup:
                samples[j].append(elapsed)
    return samples


class _Runs:
    """Fresh runs for one gated subject, so caps never trip: a new run every `every` calls,
    and with new_day the clock also moves forward a day, so the customer day starts empty."""

    def __init__(self, gate: Gate, clock: FixedClock, every: int, *, new_day: bool = True) -> None:
        self._gate = gate
        self._clock = clock
        self._every = every
        self._new_day = new_day
        self._calls = 0
        self.ctx: RunContext | None = None

    def next(self) -> None:
        if self.ctx is None or self._calls % self._every == 0:
            if self._new_day:
                self._clock.advance(timedelta(days=1))
            self.ctx = self._gate.start_run(CUSTOMER, STAFF)
        self._calls += 1

    @property
    def current(self) -> RunContext:
        if self.ctx is None:
            raise BenchError("no run started")
        return self.ctx


def _expect_executed(what: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        if not isinstance(value, Executed):
            raise BenchError(f"{what}: expected executed, got {describe(value)}")

    return check


def _expect_not_found(what: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        if not (isinstance(value, Denied) and value.reason == "not_found"):
            raise BenchError(f"{what}: expected denied not_found, got {describe(value)}")

    return check


def _expect_effect(what: str, effect: Effect) -> Callable[[object], None]:
    def check(value: object) -> None:
        if not (isinstance(value, Decision) and value.effect is effect):
            raise BenchError(f"{what}: expected {effect.value}")

    return check


def _expect_documents(what: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        documents = getattr(value, "documents", None)
        if not documents:
            raise BenchError(f"{what}: expected at least one document")

    return check


# Sections


@dataclass
class _Bench:
    world: World
    clock: FixedClock
    engine: PolicyEngine
    max_calls: int
    warmup: int
    iterations: int

    def document(self, doc_id: str) -> Document:
        document = self.world.store.get_document(doc_id)
        if document is None:
            raise BenchError(f"fixture document {doc_id!r} is missing")
        return document

    def env(self) -> ToolEnv:
        return ToolEnv(store=self.world.store, index=self.world.index, clock=self.clock)

    def decide(self) -> Section:
        user = self.world.store.get_user(STAFF)
        if user is None:
            raise BenchError(f"fixture user {STAFF!r} is missing")

        def request(doc_id: str) -> Request:
            return Request(
                customer_id=CUSTOMER,
                user_id=user.id,
                role=user.role,
                tool=READ_TOOL,
                action=Action.READ,
                document=DocumentFacts.from_document(self.document(doc_id)),
            )

        allowed, denied = request(ALLOWED_DOC), request(RESTRICTED_DOC)
        subjects = [
            _Subject(
                f"allowed read ({ALLOWED_DOC}, internal)",
                lambda: self.engine.decide(allowed),
                _expect_effect("allowed decide", Effect.ALLOW),
            ),
            _Subject(
                f"denied read ({RESTRICTED_DOC}, restricted)",
                lambda: self.engine.decide(denied),
                _expect_effect("denied decide", Effect.DENY),
            ),
        ]
        return Section(
            key="decide",
            title="1. engine.decide alone",
            note=f"The policy decision only, for {STAFF} (staff) reading two documents.",
            series=self._series(subjects),
        )

    def read(self, gate: Gate, key: str, title: str, sink: str) -> Section:
        runs = _Runs(gate, self.clock, self.max_calls)
        adapter = DocsRead()
        args = {"document_id": ALLOWED_DOC}
        validated = DocsReadArgs.model_validate(args)
        env = self.env()
        subjects = [
            _Subject(
                "through gate.call_tool",
                lambda: gate.call_tool(runs.current, READ_TOOL, Action.READ, args),
                _expect_executed(f"gated {READ_TOOL}"),
                before=runs.next,
            ),
            _Subject(
                "DocsRead.run called directly",
                lambda: adapter.run(env, runs.current, validated),
                _check_tool_result,
            ),
        ]
        series = self._series(subjects)
        return Section(
            key=key,
            title=title,
            note=(
                f"An allowed {READ_TOOL} of {ALLOWED_DOC} by {STAFF}, audit to a {sink}. The"
                " direct call gets the same args already validated."
            ),
            series=series,
            added=Added.between(series[0].stats, series[1].stats, series[1].label),
        )

    def search(self) -> Section:
        gate = self.world.gate
        runs = _Runs(gate, self.clock, self.max_calls)
        args: dict[str, JsonValue] = {"query": SEARCH_QUERY}
        label: list[str | None] = [None]

        def direct() -> object:
            return self.world.index.search(
                self.engine, runs.current, SEARCH_QUERY, run_max_label=label[0], tool=SEARCH_TOOL
            )

        def before_direct() -> None:
            # The label the gated call would decide with, for the same run.
            label[0] = gate.run_max_label(runs.current.run_id)

        subjects = [
            _Subject(
                "through gate.call_tool",
                lambda: gate.call_tool(runs.current, SEARCH_TOOL, Action.READ, args),
                _expect_executed(f"gated {SEARCH_TOOL}"),
                before=runs.next,
            ),
            _Subject(
                "DocumentIndex.search called directly",
                direct,
                _expect_documents(f"direct {SEARCH_TOOL}"),
                before=before_direct,
            ),
        ]
        series = self._series(subjects)
        return Section(
            key="search",
            title="4. docs.search through the gate vs DocumentIndex.search",
            note=(
                f"Query {SEARCH_QUERY!r} by {STAFF}, limit 5, audit to a MemorySink. The"
                " direct search uses the same engine, run and run label; both filter every"
                " tenant document through the policy."
            ),
            series=series,
            added=Added.between(series[0].stats, series[1].stats, series[1].label),
        )

    def denied(self) -> Section:
        gate = self.world.gate
        runs = _Runs(gate, self.clock, self.max_calls)
        args: dict[str, JsonValue] = {"document_id": RESTRICTED_DOC}
        subjects = [
            _Subject(
                "through gate.call_tool",
                lambda: gate.call_tool(runs.current, READ_TOOL, Action.READ, args),
                _expect_not_found(f"denied {READ_TOOL}"),
                before=runs.next,
            )
        ]
        return Section(
            key="denied",
            title="5. A denied call through the gate",
            note=(
                f"{STAFF} (staff) reads {RESTRICTED_DOC} (restricted). Denied before spend,"
                " told not_found, one deny event to a MemorySink."
            ),
            series=self._series(subjects),
        )

    def busy_day(self, busy: int, quiet: Stats) -> Section:
        gate = self.world.gate
        self.clock.advance(timedelta(days=1))
        runs = _Runs(gate, self.clock, self.max_calls, new_day=False)
        args: dict[str, JsonValue] = {"document_id": ALLOWED_DOC}
        check = _expect_executed(f"busy-day {READ_TOOL}")
        for _ in range(busy):
            runs.next()
            check(gate.call_tool(runs.current, READ_TOOL, Action.READ, args))
        samples = min(self.iterations, BUSY_DAY_SAMPLES)
        subjects = [
            _Subject(
                f"through gate.call_tool, {busy} to {busy + samples - 1} earlier reservations",
                lambda: gate.call_tool(runs.current, READ_TOOL, Action.READ, args),
                check,
                before=runs.next,
            )
        ]
        (timed,) = _measure(subjects, 0, samples)
        series = (Series(subjects[0].label, Stats.of(timed)),)
        return Section(
            key="busy_day",
            title="6. docs.read through the gate on a busy customer day",
            note=(
                f"As in section 2, after {busy} untimed gated calls on the same customer day"
                f" (fresh runs every {self.max_calls} calls, no day change). {samples} timed"
                " calls, no warmup beyond those. Compared with section 2's gated series, which"
                " was measured earlier, not interleaved."
            ),
            series=series,
            added=Added.between(series[0].stats, quiet, "section 2, through gate.call_tool"),
        )

    def _series(self, subjects: Sequence[_Subject]) -> tuple[Series, ...]:
        samples = _measure(subjects, self.warmup, self.iterations)
        return tuple(
            Series(subject.label, Stats.of(timed))
            for subject, timed in zip(subjects, samples, strict=True)
        )


def _check_tool_result(value: object) -> None:
    content = getattr(value, "content", None)
    if not isinstance(content, dict) or content.get("id") != ALLOWED_DOC:
        raise BenchError(f"direct {READ_TOOL}: expected {ALLOWED_DOC}")


def run_bench(
    *,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    busy_day: int = DEFAULT_BUSY_DAY,
    policies: Path = DEFAULT_POLICIES,
    fixtures: Path = DEFAULT_FIXTURES,
) -> BenchResult:
    """Run every section. Raises BenchError if a call takes a path it should not."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if warmup < 0 or busy_day < 0:
        raise ValueError("warmup and busy_day must not be negative")
    clock = FixedClock(REDTEAM_START)
    with open_memory_world(policies=policies, fixtures=fixtures, clock=clock) as world:
        engine = world.engines.get(CUSTOMER)
        if engine is None:
            raise BenchError(f"no policy for {CUSTOMER!r}")
        max_calls = engine.bundle.policy.spend.max_calls_per_run
        bench = _Bench(world, clock, engine, max_calls, warmup, iterations)
        sections = [bench.decide()]
        quiet = bench.read(
            world.gate, "read_memory", "2. docs.read through the gate vs direct", "MemorySink"
        )
        sections.append(quiet)
        with tempfile.TemporaryDirectory(prefix="agentgate-bench-") as tmp:
            sink = PanoptesSink(Path(tmp) / "audit.db")
            try:
                durable = Gate(
                    store=world.store,
                    engines=world.engines,
                    sink=sink,
                    clock=clock,
                    index=world.index,
                )
                sections.append(
                    bench.read(
                        durable,
                        "read_panoptes",
                        "3. docs.read through the gate vs direct, Panoptes audit",
                        "PanoptesSink (SQLite file, hash-chained)",
                    )
                )
                verified = sink.verify()
            finally:
                sink.close()
        if not verified.get("valid"):
            raise BenchError("the Panoptes audit chain did not verify after the run")
        sections.append(bench.search())
        sections.append(bench.denied())
        if busy_day > 0:
            sections.append(bench.busy_day(busy_day, quiet.series[0].stats))
    notes = (
        "Times are wall-clock (time.perf_counter_ns) on the machine in the header and are"
        " machine-dependent: compare them only with runs on the same machine.",
        "Policies and fixtures are the real ones, unchanged. So that no spend cap trips, each"
        f" gated subject starts a fresh run every {max_calls} calls ({CUSTOMER}'s"
        " max_calls_per_run) and, in sections 2 to 5, moves the fixed clock forward one day"
        f" at the same time, so a timed call sees at most {max_calls - 1} earlier"
        " reservations in its run and in its customer day.",
        "The customer-day spend check reads a running daily total, so a busy day should cost"
        " the same as a quiet one; section 6 measures that.",
        "Each iteration calls every subject of a section once, back to back, alternating"
        " which goes first. Warmup iterations are not counted. Percentiles are nearest-rank.",
        "Added latency is the median minus the median (and p95 minus p95), not a percentile"
        " of per-call differences. The gate's figures include argument validation, the run"
        " check, document lookup, the policy decision, spend reservation and settlement, the"
        " audit events and the run label update.",
        "The store is in-memory SQLite. Section 3 writes a Panoptes log in a temporary"
        " directory, verifies its hash chain after the run, then deletes it.",
    )
    return BenchResult(
        warmup=warmup,
        iterations=iterations,
        busy_day=busy_day,
        sections=tuple(sections),
        notes=notes,
    )


# Output


def _stats_json(stats: Stats) -> dict[str, JsonValue]:
    return {
        "samples": stats.samples,
        "p50_us": stats.p50_us,
        "p95_us": stats.p95_us,
        "p99_us": stats.p99_us,
        "mean_us": stats.mean_us,
    }


def to_json(result: BenchResult, meta: RunMeta) -> dict[str, JsonValue]:
    sections: list[JsonValue] = []
    for section in result.sections:
        entry: dict[str, JsonValue] = {
            "key": section.key,
            "title": section.title,
            "note": section.note,
            "series": [
                {"label": series.label, **_stats_json(series.stats)} for series in section.series
            ],
            "added": None,
        }
        if section.added is not None:
            entry["added"] = {
                "versus": section.added.versus,
                "p50_us": section.added.p50_us,
                "p95_us": section.added.p95_us,
            }
        sections.append(entry)
    return {
        "meta": meta.to_json(),
        "settings": {
            "warmup": result.warmup,
            "iterations": result.iterations,
            "busy_day": result.busy_day,
            "unit": "microseconds",
        },
        "sections": sections,
        "notes": list(result.notes),
    }


def _number(value: float) -> str:
    return f"{value:.1f}"


def _signed(value: float) -> str:
    return f"{value:+.1f}"


def to_markdown(result: BenchResult, meta: RunMeta) -> str:
    lines = ["# agentgate gate latency", "", *meta.lines()]
    lines += [
        f"- Warmup {result.warmup}, timed iterations {result.iterations}, times in"
        " microseconds (us)",
        "",
    ]
    for section in result.sections:
        lines += [f"## {section.title}", "", section.note, ""]
        lines += [
            "| Series | Samples | p50 | p95 | p99 | Mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for series in section.series:
            s = series.stats
            lines.append(
                f"| {series.label} | {s.samples} | {_number(s.p50_us)} | {_number(s.p95_us)}"
                f" | {_number(s.p99_us)} | {_number(s.mean_us)} |"
            )
        lines.append("")
        if section.added is not None:
            lines += [
                f"Added versus {section.added.versus}: p50 {_signed(section.added.p50_us)} us,"
                f" p95 {_signed(section.added.p95_us)} us.",
                "",
            ]
    lines += ["## Notes", ""]
    lines += [f"- {note}" for note in result.notes]
    return "\n".join(lines) + "\n"
