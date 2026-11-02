"""Rate tables and the worst-case reservation, in the Panoptes rate-table format."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from agentgate.model.pricing import MAX_RATE, RateTable, RateTableError

RATES: dict[str, Any] = {
    "version": "test-rates-1",
    "source": "unit test",
    "currency": "USD",
    "models": {
        "deepseek-flash": {
            "input_per_million": "0.14",
            "cached_input_per_million": "0.014",
            "output_per_million": "0.28",
        },
        "dear-cache": {
            "input_per_million": 1,
            "cached_input_per_million": 3,
            "output_per_million": 2,
        },
    },
}


def test_the_worst_case_prices_the_output_limit_and_the_prompt_estimate() -> None:
    table = RateTable.from_dict(RATES)
    # 8000 output tokens at 0.28 and 30000 prompt tokens (120000 bytes / 4) at 0.14 per million.
    assert table.worst_case_usd("deepseek-flash", 8000, 120000) == Decimal("0.006440")


def test_the_estimate_rounds_the_prompt_up_and_the_dollars_up() -> None:
    table = RateTable.from_dict(RATES)
    # 1 byte is 1 token; 0.14 / 1e6 rounds up to a whole micro-dollar.
    assert table.worst_case_usd("deepseek-flash", 0, 1) == Decimal("0.000001")
    assert table.worst_case_usd("deepseek-flash", 0, 0) == Decimal("0.000000")
    assert table.worst_case_usd("deepseek-flash", 0, 5) == Decimal("0.000001")
    # 4 bytes are exactly one token and 5 are two, so the cost steps at 5.
    one = table.worst_case_usd("dear-cache", 0, 4_000_000)
    two = table.worst_case_usd("dear-cache", 0, 4_000_001)
    assert one == Decimal("3.000000") and two is not None and two > one


def test_the_prompt_is_priced_at_the_dearer_of_input_and_cached_input() -> None:
    table = RateTable.from_dict(RATES)
    assert table.worst_case_usd("dear-cache", 0, 4_000_000) == Decimal("3.000000")


def test_a_model_the_table_does_not_name_has_no_price() -> None:
    table = RateTable.from_dict(RATES)
    for model in ("other", "", "DeepSeek-Flash", "deepseek-flash "):
        assert table.worst_case_usd(model, 10, 10) is None


def test_negative_inputs_are_refused() -> None:
    table = RateTable.from_dict(RATES)
    with pytest.raises(ValueError):
        table.worst_case_usd("deepseek-flash", -1, 10)
    with pytest.raises(ValueError):
        table.worst_case_usd("deepseek-flash", 1, -10)


def test_the_most_expensive_allowed_request_still_prices() -> None:
    table = RateTable.from_dict(
        {**RATES, "models": {"m": dict.fromkeys(RATES["models"]["dear-cache"], MAX_RATE)}}
    )
    assert table.worst_case_usd("m", 10_000_000, 16 * 1024 * 1024) is not None


@pytest.mark.parametrize(
    "mutation",
    [
        {"currency": "EUR"},
        {"currency": None},
        {"version": ""},
        {"version": 3},
        {"source": None},
        {"models": {}},
        {"models": None},
        {"models": []},
        {"models": {"m": 5}},
        {"models": {"m": {"input_per_million": 1}}},
        {"models": {"m": {**RATES["models"]["dear-cache"], "input_per_million": -1}}},
        {"models": {"m": {**RATES["models"]["dear-cache"], "output_per_million": "nan"}}},
        {"models": {"m": {**RATES["models"]["dear-cache"], "output_per_million": "inf"}}},
        {"models": {"m": {**RATES["models"]["dear-cache"], "output_per_million": "abc"}}},
        {"models": {"m": {**RATES["models"]["dear-cache"], "output_per_million": True}}},
        {"models": {"m": {**RATES["models"]["dear-cache"], "output_per_million": None}}},
        {"models": {"m": {**RATES["models"]["dear-cache"], "output_per_million": MAX_RATE + 1}}},
    ],
)
def test_a_malformed_table_is_refused(mutation: dict[str, Any]) -> None:
    with pytest.raises(RateTableError):
        RateTable.from_dict({**RATES, **mutation})


@pytest.mark.parametrize("raw", [None, [], "text", 3, {}])
def test_a_table_that_is_not_an_object_is_refused(raw: object) -> None:
    with pytest.raises(RateTableError):
        RateTable.from_dict(raw)


def test_a_table_loads_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(RATES))
    table = RateTable.load(path)
    assert (table.version, table.source) == ("test-rates-1", "unit test")
    assert table.models["deepseek-flash"].output_per_million == Decimal("0.28")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"version": "a", "version": "b"}', "not valid JSON"),
        ("[" * 100_000, "not valid JSON"),
        ("x" * (1024 * 1024 + 1), "too large"),
    ],
)
def test_an_unreadable_table_is_refused(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "rates.json"
    path.write_text(content)
    with pytest.raises(RateTableError, match=message):
        RateTable.load(path)


def test_a_missing_table_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RateTableError, match="cannot read"):
        RateTable.load(tmp_path / "absent.json")
