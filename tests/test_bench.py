"""The latency bench: every section runs, with sane numbers, at a small N."""

import json

import pytest

from agentgate.redteam.bench import BenchResult, Stats, run_bench, to_json, to_markdown
from agentgate.redteam.report import RunMeta
from conftest import FIXTURES, POLICIES

META = RunMeta(
    command="agentgate bench --iterations 20",
    commit="abc1234",
    python="3.12.0",
    platform="TestOS-1.0",
    machine="arm64",
    cpu_count=4,
    agentgate_version="0.0.1",
    timestamp="2026-10-22T15:00:00Z",
)
SECTIONS = ["decide", "read_memory", "read_panoptes", "search", "denied", "busy_day"]
WITH_ADDED = {"read_memory", "read_panoptes", "search", "busy_day"}


@pytest.fixture(scope="module")
def bench() -> BenchResult:
    return run_bench(iterations=20, warmup=5, busy_day=50, policies=POLICIES, fixtures=FIXTURES)


def test_every_section_runs(bench: BenchResult) -> None:
    assert [section.key for section in bench.sections] == SECTIONS
    for section in bench.sections:
        assert section.series
        for series in section.series:
            stats = series.stats
            assert stats.samples == 20
            values = [stats.p50_us, stats.p95_us, stats.p99_us, stats.mean_us]
            assert all(value >= 0 for value in values), section.key
            assert stats.p50_us <= stats.p95_us <= stats.p99_us
        assert (section.added is not None) == (section.key in WITH_ADDED), section.key


def test_added_latency_is_the_difference_of_medians_and_p95s(bench: BenchResult) -> None:
    for key in ("read_memory", "read_panoptes", "search"):
        section = bench.section(key)
        gated, direct = section.series
        assert section.added is not None
        assert section.added.versus == direct.label
        assert section.added.p50_us == pytest.approx(gated.stats.p50_us - direct.stats.p50_us)
        assert section.added.p95_us == pytest.approx(gated.stats.p95_us - direct.stats.p95_us)
    # The gate does strictly more than the bare adapter read.
    assert bench.section("read_memory").added.p50_us > 0
    assert bench.section("read_panoptes").added.p50_us > 0


def test_json_and_markdown(bench: BenchResult) -> None:
    data = to_json(bench, META)
    assert json.loads(json.dumps(data)) == data
    assert data["settings"] == {
        "warmup": 5,
        "iterations": 20,
        "busy_day": 50,
        "unit": "microseconds",
    }
    sections = data["sections"]
    assert isinstance(sections, list)
    assert [s["key"] for s in sections] == SECTIONS
    read = sections[1]
    assert isinstance(read, dict)
    assert set(read["added"]) == {"versus", "p50_us", "p95_us"}

    markdown = to_markdown(bench, META)
    for section in bench.sections:
        assert f"## {section.title}" in markdown
    assert "machine x86_64" not in markdown
    assert "- Platform: TestOS-1.0, machine arm64, 4 CPUs" in markdown
    assert "machine-dependent" in markdown
    assert "fresh run every 40 calls" in markdown
    assert markdown.count("Added versus") == len(WITH_ADDED)
    assert "\u2014" not in markdown


def test_busy_day_can_be_skipped() -> None:
    result = run_bench(iterations=2, warmup=0, busy_day=0, policies=POLICIES, fixtures=FIXTURES)
    assert [section.key for section in result.sections] == SECTIONS[:-1]


def test_stats_are_nearest_rank() -> None:
    stats = Stats.of([n * 1000 for n in range(100, 0, -1)])
    assert stats == Stats(samples=100, p50_us=50.0, p95_us=95.0, p99_us=99.0, mean_us=50.5)
    single = Stats.of([1500])
    assert (single.p50_us, single.p99_us, single.mean_us) == (1.5, 1.5, 1.5)


@pytest.mark.parametrize("kwargs", [{"iterations": 0}, {"warmup": -1}, {"busy_day": -1}])
def test_bad_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        run_bench(policies=POLICIES, fixtures=FIXTURES, **kwargs)
