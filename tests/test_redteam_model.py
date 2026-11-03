"""Mutation and tightening tests for the model boundary scenarios: each protection is removed,
by editing the one function that holds it or replacing it, and the scenarios that rely on it
must then fail. A scenario that still passed would be testing nothing. That the scenarios pass
unmutated is tested once, for the whole suite, in test_redteam_core.py.
"""

import json
import re
import shutil
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

import agentgate.gate as gate_module
import agentgate.model.admission as admission_module
import agentgate.model.keys as keys_module
import agentgate.model.tokens as tokens_module
import agentgate.redteam.scenarios.model_boundary as scenarios_module
from agentgate.gate import Gate
from agentgate.model.admission import Caps, ModelExchange
from agentgate.model.keys import KeyConfigError, ProviderKeys
from agentgate.model.pricing import RateTable
from agentgate.model.tokens import TokenStore
from agentgate.policy.engine import PolicyEngine, Reason
from agentgate.policy.schema import SpendLimits
from agentgate.redteam.core import Category, run_suite
from agentgate.spend import SpendLedger
from conftest import (
    FIXTURES,
    KC,
    POLICIES,
    Mutation,
    assert_breaks,
    assert_every_attack_has_a_mutation,
    mutant,
)

REAL_PARSE_KEYS = keys_module._parse_keys
REAL_RESERVE = SpendLedger.reserve
UNLIMITED = SpendLimits(
    per_run_usd=Decimal("999999"),
    per_customer_day_usd=Decimal("999999"),
    max_calls_per_run=10_000,
)


def everywhere(owner: Any, name: str, replacement: Any) -> tuple[tuple[Any, str, Any], ...]:
    """Replace a function in its module and in the scenarios, which import it by name."""
    return ((owner, name, replacement), (scenarios_module, name, replacement))


def ids(*numbers: int) -> tuple[str, ...]:
    return tuple(f"mb-{n:03d}" for n in numbers)


# Replacements


def _first_entry(self: PolicyEngine, provider: Any, model: str) -> Any:
    return next(iter(self._models.values()), None)


def _entry_by_name_only(self: PolicyEngine, provider: Any, model: str) -> Any:
    return next((e for (_, name), e in self._models.items() if name == model), None)


def _envelope_without(*skipped: Reason) -> Any:
    """The envelope check with some of its reasons never raised."""
    real = PolicyEngine._model_envelope_reason

    def envelope(entry: Any, request: Any) -> Reason | None:
        reason = real(entry, request)
        return None if reason in skipped else reason

    return staticmethod(envelope)


def _ignore_limits(self: SpendLedger, **kwargs: Any) -> Any:
    return REAL_RESERVE(self, **(kwargs | {"limits": UNLIMITED}))


def _caps_ignored(self: Caps, limits: SpendLimits) -> SpendLimits:
    return limits


def _caps_widen(self: Caps, limits: SpendLimits) -> SpendLimits:
    update: dict[str, Any] = {}
    if self.max_usd is not None:
        update["per_run_usd"] = self.max_usd
    if self.max_calls is not None:
        update["max_calls_per_run"] = self.max_calls
    return limits.model_copy(update=update)


def _output_free(
    self: RateTable, model: str, max_tokens: int, request_bytes: int
) -> Decimal | None:
    """A worst case that prices the prompt and forgets the output limit."""
    rate = self.models.get(model)
    if rate is None:
        return None
    tokens = request_bytes
    usd = tokens * max(rate.input_per_million, rate.cached_input_per_million) / 1_000_000
    return usd.quantize(Decimal("0.000001"))


def _unknown_is_free(self: RateTable, model: str, max_tokens: int, request_bytes: int) -> Decimal:
    rate = self.models.get(model)
    if rate is None:
        return Decimal(0)
    return Decimal("0.019040")


def _lax_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    return dict(pairs)


def _coercing_integers(sources: list[dict[str, object]], names: tuple[str, ...]) -> list[int]:
    found: list[int] = []
    for source in sources:
        for name in names:
            value = source.get(name)
            if value is None:
                continue
            try:
                found.append(int(value))  # type: ignore[call-overload]
            except (TypeError, ValueError, OverflowError):
                found.append(0)
    return found


def _chained_parse(content: bytes, path: Path) -> Any:
    try:
        json.loads(content)
    except ValueError as exc:
        raise KeyConfigError(f"keys file {path} is not valid JSON") from exc
    return REAL_PARSE_KEYS(content, path)


def _naming_parse(content: bytes, path: Path) -> Any:
    try:
        data = json.loads(content)
    except ValueError:
        return REAL_PARSE_KEYS(content, path)
    known = {p.value for p in keys_module.ModelProvider}
    unknown = sorted(str(name) for name in data) if isinstance(data, dict) else []
    unknown = [name for name in unknown if name not in known]
    if unknown:
        raise KeyConfigError(f"keys file {path} names unknown providers: {unknown}")
    return REAL_PARSE_KEYS(content, path)


def _show_keys(self: ProviderKeys) -> str:
    return f"ProviderKeys({self._keys})"


def _no_redaction(self: ProviderKeys, data: Any) -> Any:
    return data


def _headers_without(*names: str) -> frozenset[str]:
    return keys_module.CLIENT_CREDENTIAL_HEADERS - set(names)


MUTATIONS = [
    # The allowlist and the envelope
    Mutation(
        "the first entry stands for every model",
        ids(1, 2),
        replace=((PolicyEngine, "model_entry", _first_entry),),
    ),
    Mutation(
        "the provider is not part of the match",
        ids(1, 2),
        replace=((PolicyEngine, "model_entry", _entry_by_name_only),),
    ),
    Mutation(
        "the role is not checked",
        ids(7),
        replace=(
            (PolicyEngine, "_model_envelope_reason", _envelope_without(Reason.MODEL_ROLE_DENIED)),
        ),
    ),
    Mutation(
        "the output limit is not enforced",
        ids(3),
        replace=(
            (
                PolicyEngine,
                "_model_envelope_reason",
                _envelope_without(Reason.MODEL_MAX_TOKENS_EXCEEDED),
            ),
        ),
        # A limit of 2**200 then reaches the pricing, which cannot represent it.
        crashes=("InvalidOperation",),
    ),
    Mutation(
        "a missing output limit is allowed",
        ids(3, 18),
        edits=(
            (
                Gate,
                "admit_model",
                (("if models is None or call.max_tokens is None:", "if False:"),),
            ),
        ),
        replace=(
            (
                PolicyEngine,
                "_model_envelope_reason",
                _envelope_without(Reason.MODEL_MAX_TOKENS_REQUIRED),
            ),
        ),
        crashes=("TypeError",),
    ),
    Mutation(
        "the number of completions is not checked",
        ids(4),
        replace=(
            (PolicyEngine, "_model_envelope_reason", _envelope_without(Reason.MODEL_N_NOT_ONE)),
        ),
    ),
    Mutation(
        "the body size is not checked",
        ids(5),
        replace=(
            (
                PolicyEngine,
                "_model_envelope_reason",
                _envelope_without(Reason.MODEL_REQUEST_TOO_LARGE),
            ),
        ),
    ),
    # Reading the body
    Mutation(
        "an invalid body is decided like any other",
        ids(4, 5, 6),
        edits=(
            (
                Gate,
                "admit_model",
                (
                    (
                        "return self._deny(asked, GateReason.MODEL_REQUEST_INVALID, "
                        "details=values)",
                        "pass",
                    ),
                ),
            ),
        ),
    ),
    Mutation(
        "a repeated key keeps its last value",
        ids(6),
        replace=((admission_module, "no_duplicate_keys", _lax_duplicates),),
    ),
    Mutation(
        "a wrongly typed limit is coerced",
        ids(4, 6),
        replace=((admission_module, "_integers", _coercing_integers),),
    ),
    Mutation(
        "the first spelling of the limit is the limit",
        ids(3),
        replace=everywhere(
            admission_module,
            "parse_model_call",
            mutant(
                admission_module,
                "parse_model_call",
                (
                    (
                        "max_tokens=max(limits) if limits else None",
                        "max_tokens=limits[0] if limits else None",
                    ),
                ),
                {},
            ),
        ),
    ),
    Mutation(
        "the body is decoded leniently",
        ids(6),
        replace=everywhere(
            admission_module,
            "parse_model_call",
            mutant(admission_module, "parse_model_call", (('body.decode("utf-8")', "body"),), {}),
        ),
    ),
    Mutation(
        "a body nested past the parser is not caught",
        ids(6),
        replace=everywhere(
            admission_module,
            "parse_model_call",
            mutant(
                admission_module,
                "parse_model_call",
                (("except (ValueError, RecursionError):", "except ValueError:"),),
                {},
            ),
        ),
        crashes=("RecursionError",),
    ),
    Mutation(
        "unread fields are not counted",
        ids(23),
        replace=everywhere(
            admission_module,
            "parse_model_call",
            mutant(
                admission_module,
                "parse_model_call",
                (
                    (
                        "unknown_fields=sum(1 for name in data if name not in _FIELDS[provider])",
                        "unknown_fields=0",
                    ),
                ),
                {},
            ),
        ),
    ),
    Mutation(
        "the gate lets unread fields through",
        ids(23),
        edits=((Gate, "admit_model", (("if call.unknown_fields:", "if False:"),)),),
    ),
    Mutation(
        "server features are not seen",
        ids(24),
        replace=((admission_module, "_server_features", lambda provider, data: ()),),
    ),
    Mutation(
        "server features are allowed",
        ids(24),
        replace=(
            (
                PolicyEngine,
                "_model_envelope_reason",
                _envelope_without(Reason.MODEL_SERVER_FEATURE),
            ),
        ),
    ),
    # Tokens
    Mutation(
        "a token of a suspended user still resolves",
        ids(8),
        edits=(
            (
                TokenStore,
                "_run",
                (('raise _SuspendedUser("the run\'s user is suspended")', "pass"),),
            ),
        ),
    ),
    Mutation(
        "the gate lets a suspended user's run through",
        ids(8),
        edits=(
            (
                Gate,
                "_check_run",
                (("return self._deny(call, GateReason.SUSPENDED_USER)", "pass"),),
            ),
        ),
    ),
    Mutation(
        "a token works for any run",
        ids(9),
        edits=(
            (
                TokenStore,
                "resolve",
                (("if run_id is not None and record.run_id != run_id:", "if False:"),),
            ),
        ),
    ),
    Mutation(
        "a token can be issued for a run that does not exist",
        ids(8, 9),
        edits=((TokenStore, "issue", (("self._run(run_id)", "pass"),)),),
    ),
    Mutation(
        "two different credentials are not ambiguous",
        ids(9),
        replace=everywhere(
            tokens_module,
            "presented_token",
            mutant(tokens_module, "presented_token", (("if len(distinct) > 1:", "if False:"),), {}),
        ),
    ),
    Mutation(
        "a revoked token still resolves",
        ids(10),
        edits=((TokenStore, "resolve", (("if record.revoked_at is not None:", "if False:"),)),),
    ),
    Mutation(
        "an expired token still resolves",
        ids(10),
        edits=(
            (TokenStore, "resolve", (("if self._clock.now() >= record.expires_at:", "if False:"),)),
        ),
    ),
    Mutation(
        "a token can be issued for a run that has ended",
        ids(10),
        edits=((TokenStore, "issue", (('raise TokenError("the run has ended")', "pass"),)),),
    ),
    Mutation(
        "ending a run records nothing",
        ids(10),
        edits=((TokenStore, "end_run", (("record(revoked)", "pass"),)),),
    ),
    Mutation(
        "ending a run revokes nothing",
        (*ids(10), "bn-020"),
        replace=((TokenStore, "end_run", lambda self, run_id, *, record: 0),),
    ),
    Mutation(
        "a token is stored as it is",
        ids(11),
        replace=((tokens_module, "token_hash", lambda token: token),),
    ),
    # Keys and credentials
    Mutation(
        "a key prints",
        ids(12),
        replace=((ProviderKeys, "__repr__", _show_keys), (ProviderKeys, "__str__", _show_keys)),
    ),
    Mutation(
        "an unusable key is shown in the error",
        ids(12),
        edits=(
            (
                keys_module,
                "validate_key",
                (
                    (
                        'f"{name} key must be visible ASCII with no whitespace"',
                        'f"{name} key {value!r} must be visible ASCII with no whitespace"',
                    ),
                ),
            ),
        ),
    ),
    Mutation(
        "an unknown provider name is printed",
        ids(12),
        replace=((keys_module, "_parse_keys", _naming_parse),),
    ),
    Mutation(
        "a parse error is chained to the key error",
        ids(12),
        replace=((keys_module, "_parse_keys", _chained_parse),),
    ),
    Mutation(
        "an echoed key is not redacted",
        ids(12),
        replace=((ProviderKeys, "redact", _no_redaction),),
    ),
    Mutation(
        "a keys file others can read is accepted",
        ids(12),
        edits=((keys_module, "_read_keys_file", (("if info.st_mode & 0o077:", "if False:"),)),),
    ),
    Mutation(
        "a provider listed twice keeps the last key",
        ids(12),
        replace=((keys_module, "no_duplicate_keys", _lax_duplicates),),
    ),
    Mutation(
        "an Authorization header is forwarded",
        ids(13),
        replace=((keys_module, "CLIENT_CREDENTIAL_HEADERS", _headers_without("authorization")),),
    ),
    Mutation(
        "a cookie is forwarded",
        ids(13),
        replace=((keys_module, "CLIENT_CREDENTIAL_HEADERS", _headers_without("cookie")),),
    ),
    Mutation(
        "a client's key query parameter is forwarded",
        ids(13),
        replace=((keys_module, "CLIENT_CREDENTIAL_QUERY_PARAMS", frozenset()),),
    ),
    Mutation(
        "query parameter names are compared case-sensitively",
        ids(13),
        replace=(
            (
                keys_module,
                "strip_query_credentials",
                mutant(
                    keys_module,
                    "strip_query_credentials",
                    (("unquote_plus(name).casefold()", "unquote_plus(name)"),),
                    {},
                ),
            ),
        ),
    ),
    Mutation(
        "query parameter names are not decoded",
        ids(13),
        replace=(
            (
                keys_module,
                "strip_query_credentials",
                mutant(
                    keys_module,
                    "strip_query_credentials",
                    (("unquote_plus(name).casefold()", "name.casefold()"),),
                    {},
                ),
            ),
        ),
    ),
    # Spend
    Mutation(
        "the ledger ignores limits",
        ids(14, 15, 16),
        replace=((SpendLedger, "reserve", _ignore_limits),),
    ),
    Mutation("caps are ignored", ids(14, 15, 16), replace=((Caps, "tighten", _caps_ignored),)),
    Mutation("caps can widen the policy", ids(15), replace=((Caps, "tighten", _caps_widen),)),
    Mutation(
        "a fixed reserve replaces the computed worst case",
        ids(25),
        edits=(
            (
                Gate,
                "admit_model",
                (
                    (
                        "amount = fixed if computed is None else "
                        "computed if fixed is None else max(fixed, computed)",
                        "amount = fixed if fixed is not None else computed",
                    ),
                ),
            ),
        ),
    ),
    Mutation(
        "release keeps the reservation",
        (*ids(17), "bn-020"),
        replace=((SpendLedger, "release", lambda self, reservation: None),),
    ),
    Mutation(
        "a gateway reservation can be settled by hand",
        ids(17),
        edits=(
            (
                Gate,
                "settle_model",
                (('raise ValueError("a gateway reservation is ended by finish_model")', "pass"),),
            ),
        ),
        # The call is settled by hand, and the scenario's later finish finds it settled.
        crashes=("SpendError",),
    ),
    Mutation(
        "a dropped connection is released",
        (*ids(17, 22),),
        edits=(
            (
                Gate,
                "finish_model",
                (
                    (
                        "if not exchange.request_sent or (",
                        "if not exchange.complete or not exchange.request_sent or (",
                    ),
                ),
            ),
        ),
    ),
    Mutation(
        "an unusable cost is passed to the ledger",
        ids(17),
        replace=((ModelExchange, "__post_init__", lambda self: None),),
        crashes=("SpendError",),
    ),
    Mutation(
        "a provider error keeps its zero price",
        ids(17),
        edits=(
            (
                ModelExchange,
                "__post_init__",
                (("not usable or self.provider_error", "not usable"),),
            ),
        ),
    ),
    Mutation(
        "a provider error that produced output is released",
        ids(17),
        edits=(
            (
                Gate,
                "finish_model",
                (
                    (
                        "exchange.provider_error and not exchange.produced_output",
                        "exchange.provider_error",
                    ),
                ),
            ),
        ),
    ),
    Mutation(
        "a partial price is charged for a dropped connection",
        ids(17),
        edits=(
            (
                Gate,
                "finish_model",
                (
                    (
                        "if exchange.complete and exchange.cost_usd is not None:",
                        "if exchange.cost_usd is not None:",
                    ),
                ),
            ),
        ),
    ),
    Mutation(
        "a provider error with no output is charged",
        (*ids(17), "bn-020"),
        edits=(
            (
                Gate,
                "finish_model",
                (("(exchange.provider_error and not exchange.produced_output)", "False"),),
            ),
        ),
    ),
    Mutation(
        "the worst case forgets the output limit",
        ids(18),
        replace=((RateTable, "worst_case_usd", _output_free),),
    ),
    Mutation(
        "an unpriced model costs nothing",
        ids(18),
        replace=((RateTable, "worst_case_usd", _unknown_is_free),),
    ),
    # Audit and recovery
    Mutation(
        "any correlation id is recorded as written",
        ids(19),
        replace=((gate_module, "_CORRELATION_ID", re.compile(r"(?s).*")),),
    ),
    Mutation(
        "a key-shaped correlation id is recorded as written",
        ids(19),
        replace=((gate_module, "storable_identifier", lambda value: True),),
    ),
    Mutation(
        "an allow that cannot be recorded is admitted anyway",
        ids(20),
        edits=(
            (
                Gate,
                "admit_model",
                (
                    (
                        "        self._ledger.release(reserved)\n"
                        "        return Denied(GateReason.AUDIT_UNAVAILABLE.value)\n",
                        "        pass\n",
                    ),
                ),
            ),
        ),
    ),
    Mutation(
        "no stale sweep",
        ids(21),
        replace=((SpendLedger, "release_stale", lambda self, older_than: 0),),
    ),
]


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.name)
def test_mutation_fails_its_scenarios(mutation: Mutation, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_breaks(mutation, monkeypatch)


def test_every_attack_has_a_mutation() -> None:
    assert_every_attack_has_a_mutation(MUTATIONS, Category.MODEL_BOUNDARY)


# Tightening: the benign control fails once the policy no longer lists the model.


def _edited(tmp_path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    root = tmp_path / "policies"
    shutil.copytree(POLICIES, root)
    path = root / KC / "policy.yaml"
    data = yaml.safe_load(path.read_text())
    edit(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return root


def _drop_models(data: dict[str, Any]) -> None:
    del data["models"]


def _small_envelope(data: dict[str, Any]) -> None:
    data["models"]["allow"][0]["max_output_tokens"] = 100


def _contractors_only(data: dict[str, Any]) -> None:
    data["models"]["allow"][0]["roles"] = ["contractor"]


@pytest.mark.parametrize(
    "edit", [_drop_models, _small_envelope, _contractors_only], ids=lambda e: e.__name__
)
def test_the_benign_model_scenario_is_tightened(tmp_path: Path, edit: Callable[..., None]) -> None:
    policies = _edited(tmp_path, edit)
    (result,) = run_suite(ids=["bn-020"], policies=policies, fixtures=FIXTURES).results
    assert not result.passed and result.error is None
