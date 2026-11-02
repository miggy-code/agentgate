"""Rate tables and the worst-case cost a model call reserves before it is sent.

The table is a versioned JSON file in the format Panoptes uses to price observed responses:

    {"version": "2026-11", "source": "provider price page, 2026-11-02", "currency": "USD",
     "models": {"deepseek-flash": {"input_per_million": "0.14",
                                   "cached_input_per_million": "0.014",
                                   "output_per_million": "0.28"}}}

agentgate never guesses a price. A model the table does not name has none, and a call to it
reserves nothing it could justify, so the gate denies it unless the policy sets a fixed
reserve (`models.reserve`).

The worst case is the request's declared `max_tokens` at the output rate, plus an estimate
of the prompt at the dearer of the input and cached-input rates: one token per 4 bytes of
body. That estimate is not an upper bound (text in some scripts, or dense JSON, runs closer
to one token per byte), so a response can cost more than was reserved. The ledger records
what a call actually cost even then, and the next reservation is refused if that put the
run over its cap; see docs/limits.md.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path

from agentgate._json import no_duplicate_keys

MILLION = Decimal(1_000_000)
MICRO = Decimal("0.000001")
BYTES_PER_TOKEN = 4
RATE_FIELDS = ("input_per_million", "cached_input_per_million", "output_per_million")
MAX_TABLE_BYTES = 1024 * 1024
# USD per million tokens. Far above any real price; it keeps a typo from overflowing a reserve.
MAX_RATE = Decimal(10_000_000)


class RateTableError(ValueError):
    """The rate table is missing, malformed or unversioned."""


@dataclass(frozen=True)
class Rate:
    """USD per million tokens."""

    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class RateTable:
    version: str
    source: str
    models: Mapping[str, Rate]

    @classmethod
    def load(cls, path: Path) -> "RateTable":
        try:
            with path.open("rb") as handle:
                content = handle.read(MAX_TABLE_BYTES + 1)
        except OSError as exc:
            raise RateTableError(f"cannot read rate table {path}: {exc.strerror}") from None
        if len(content) > MAX_TABLE_BYTES:
            raise RateTableError(f"rate table {path} is too large")
        raw: object = None
        unreadable = False
        try:
            raw = json.loads(content, object_pairs_hook=no_duplicate_keys)
        except (ValueError, RecursionError):
            unreadable = True
        if unreadable:
            raise RateTableError(f"rate table {path} is not valid JSON")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: object) -> "RateTable":
        if not isinstance(raw, dict) or raw.get("currency") != "USD":
            raise RateTableError("rate table must be a USD object")
        version, source = raw.get("version"), raw.get("source")
        if not isinstance(version, str) or not version or not isinstance(source, str) or not source:
            raise RateTableError("rate table needs a version and a source")
        listed = raw.get("models")
        if not isinstance(listed, dict) or not listed:
            raise RateTableError("rate table names no models")
        models: dict[str, Rate] = {}
        for name, rate in listed.items():
            if not isinstance(rate, dict):
                raise RateTableError(f"rate for {name} must be an object")
            models[str(name)] = Rate(*(_rate(name, rate, field) for field in RATE_FIELDS))
        return cls(version=version, source=source, models=models)

    def worst_case_usd(self, model: str, max_tokens: int, request_bytes: int) -> Decimal | None:
        """What a call to model could cost at most, rounded up to a micro-dollar, or None if
        the table has no price for it."""
        rate = self.models.get(model)
        if rate is None:
            return None
        if max_tokens < 0 or request_bytes < 0:
            raise ValueError("max_tokens and request_bytes must not be negative")
        prompt_tokens = -(-request_bytes // BYTES_PER_TOKEN)
        prompt_rate = max(rate.input_per_million, rate.cached_input_per_million)
        usd = (max_tokens * rate.output_per_million + prompt_tokens * prompt_rate) / MILLION
        return usd.quantize(MICRO, rounding=ROUND_CEILING)


def _rate(model: str, rate: dict[object, object], field: str) -> Decimal:
    value = rate.get(field)
    if value is None or isinstance(value, bool):
        raise RateTableError(f"rate for {model} lacks a valid {field}")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise RateTableError(f"rate for {model} lacks a valid {field}") from None
    if not number.is_finite() or number < 0 or number > MAX_RATE:
        raise RateTableError(f"rate for {model} has an invalid {field}")
    return number
