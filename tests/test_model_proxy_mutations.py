"""Mutation tests for the model proxy: each protection is removed, by editing the one place
that holds it or replacing it, and the HTTP tests that rely on it must then fail. A test that
still passed would be testing nothing. Every test in test_model_proxy.py that guards a
boundary is named by at least one mutation (see test_every_guard_has_a_mutation).
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import agentgate.model.proxy as proxy_module
import test_model_proxy as t
from agentgate.model.proxy import ModelGatewayHooks, ModelProxy, Route
from agentgate.model.tokens import TokenStore
from conftest import mutant

Case = tuple[Callable[..., None], tuple[Any, ...]]


def whole(fn: Callable[..., None]) -> list[Case]:
    return [(fn, ())]


def cases(fn: Callable[..., None], params: list[tuple[Any, ...]]) -> list[Case]:
    return [(fn, tuple(p)) for p in params]


@dataclass(frozen=True)
class ProxyMutation:
    name: str
    # The tests that must fail with the protection gone; at least one case of each must.
    breaks: tuple[list[Case], ...]
    edits: tuple[tuple[Any, str, tuple[tuple[str, str], ...]], ...] = ()
    replace: tuple[tuple[Any, str, Any], ...] = ()
    guards: tuple[str, ...] = field(default=())


def _everything_routes(method: str, target: str) -> Route:
    return Route("openai", None)


def _any_token_resolves(self: TokenStore, token: str | None, *, run_id: str | None = None) -> Any:
    return self.run(run_id) if run_id is not None else None


def _first_credential(headers: Any, query_values: Any = ()) -> str | None:
    for name, value in headers:
        if name.lower() == "authorization":
            return str(value).removeprefix("Bearer ").strip()
    return None


def _keep_client_credentials(keys: Any, provider: Any, headers: Any) -> list[tuple[str, str]]:
    credential = keys.credential(provider)
    kept = [(n, v) for n, v in headers if n.lower() != credential.name]
    return [*kept, (credential.name, credential.value)]


def _rates_unchecked(*args: Any, **kwargs: Any) -> None:
    return None


def _world_readable_token_file(path: Path, token: str) -> None:
    path.write_text(token + "\n")
    path.chmod(0o644)


def _edit(old: str, new: str) -> tuple[str, str]:
    return (old, new)


HOOKS = ModelGatewayHooks
MUTATIONS = [
    ProxyMutation(
        "a request to any route is forwarded",
        (
            cases(
                t.test_only_a_post_to_a_generation_endpoint_reaches_the_provider, t.REFUSED_ROUTES
            ),
        ),
        replace=((proxy_module, "generation_route", _everything_routes),),
    ),
    ProxyMutation(
        "any token is accepted",
        (
            whole(t.test_a_request_without_a_valid_token_is_refused_and_never_forwarded),
            whole(t.test_a_revoked_token_is_refused_on_its_runs_record),
            whole(t.test_a_suspended_user_is_refused),
        ),
        replace=((TokenStore, "resolve", _any_token_resolves),),
    ),
    ProxyMutation(
        "two different credentials are not ambiguous",
        (whole(t.test_two_different_credentials_are_ambiguous),),
        replace=((proxy_module, "presented_token", _first_credential),),
    ),
    ProxyMutation(
        "client credentials travel upstream",
        (whole(t.test_every_client_credential_is_replaced_by_the_providers_key),),
        replace=((proxy_module, "upstream_headers", _keep_client_credentials),),
    ),
    ProxyMutation(
        "the query credentials are not dropped",
        (whole(t.test_a_token_in_the_query_is_read_and_never_forwarded),),
        edits=((HOOKS, "before_forward", (_edit("ex.drop_query_credentials()", "pass"),)),),
    ),
    ProxyMutation(
        "a denial does not stop the request",
        (
            cases(
                t.test_a_request_the_policy_denies_gets_a_reason_code_and_goes_nowhere,
                [(b, r) for b, r in t.DENIED_BODIES],
            ),
        ),
        edits=(
            (
                HOOKS,
                "before_forward",
                (_edit("if isinstance(admitted, Denied):", "if False:"),),
            ),
        ),
    ),
    ProxyMutation(
        "a reached cap does not stop the request",
        (
            whole(t.test_the_run_cap_is_a_429_and_the_next_request_is_not_forwarded),
            whole(t.test_a_cap_below_the_worst_case_refuses_the_first_request),
        ),
        edits=(
            (
                HOOKS,
                "before_forward",
                (_edit("if isinstance(admitted, CapReached):", "if False:"),),
            ),
        ),
    ),
    ProxyMutation(
        "the exchange is not recorded under the run",
        (whole(t.test_a_complete_reply_is_charged_what_it_cost_and_joined_to_its_exchange),),
        edits=((HOOKS, "before_forward", (_edit("ex.identity = identity", "pass"),)),),
    ),
    ProxyMutation(
        "the exchange id is not recorded with the decision",
        (whole(t.test_a_complete_reply_is_charged_what_it_cost_and_joined_to_its_exchange),),
        edits=(
            (
                HOOKS,
                "before_forward",
                (_edit("exchange_id=ex.exchange_id,", ""),),
            ),
        ),
    ),
    ProxyMutation(
        "a provider error is charged",
        (whole(t.test_a_provider_error_with_no_output_is_released),),
        edits=(
            (
                HOOKS,
                "_finish",
                (
                    _edit(
                        "provider_error=status is not None and status >= 400,",
                        "provider_error=False,",
                    ),
                ),
            ),
        ),
    ),
    ProxyMutation(
        "a call that was never sent is charged",
        (whole(t.test_a_failure_after_the_reservation_is_a_hook_error_and_releases_it),),
        edits=(
            (
                HOOKS,
                "_finish",
                (_edit("request_sent=bool(exchange.request_sent),", "request_sent=True,"),),
            ),
        ),
    ),
    ProxyMutation(
        "after_exchange never ends the call",
        (
            whole(t.test_a_failure_after_the_reservation_is_a_hook_error_and_releases_it),
            whole(t.test_a_complete_reply_is_charged_what_it_cost_and_joined_to_its_exchange),
        ),
        edits=((HOOKS, "after_exchange", (_edit("self._finish(exchange)", "pass"),)),),
    ),
    ProxyMutation(
        "an error in the decision lets the request through",
        (whole(t.test_an_error_in_the_decision_is_a_503_and_nothing_is_forwarded),),
        edits=(
            (
                HOOKS,
                "before_forward",
                (_edit("return _reject(503, UNAVAILABLE)", "return None"),),
            ),
        ),
    ),
    ProxyMutation(
        "the proxy starts its own run instead of the gateway's",
        (whole(t.test_a_proxy_started_on_a_gateways_run_shares_its_caps_with_the_tools),),
        edits=(
            (
                ModelProxy,
                "start",
                (
                    _edit(
                        "run if run is not None else",
                        "world.gate.start_run(user.customer_id, user.id) if True else",
                    ),
                ),
            ),
        ),
    ),
    ProxyMutation(
        "stopping leaves the token live",
        (whole(t.test_stopping_drains_ends_the_run_and_revokes_its_token),),
        edits=((ModelProxy, "_end_run", (_edit("if self._run is None:", "if True:"),)),),
    ),
    ProxyMutation(
        "disagreeing rate tables start",
        (whole(t.test_two_rate_tables_that_disagree_do_not_start),),
        replace=(
            (proxy_module, "check_rates", _rates_unchecked),
            (t, "check_rates", _rates_unchecked),
        ),
    ),
    ProxyMutation(
        "a missing provider key does not stop the start",
        (whole(t.test_the_proxy_does_not_start_without_a_key_for_its_upstream),),
        edits=(
            (ModelProxy, "validate", (_edit("if not config.keys.has(provider):", "if False:"),)),
        ),
    ),
    ProxyMutation(
        "the token file is readable by others",
        (whole(t.test_the_token_file_is_private_and_removed_when_the_proxy_stops),),
        replace=((proxy_module, "write_token_file", _world_readable_token_file),),
    ),
]


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: m.name)
def test_mutation_fails_its_tests(
    mutation: ProxyMutation, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for owner, attribute, edits in mutation.edits:
        monkeypatch.setattr(owner, attribute, mutant(owner, attribute, edits, {}))
    for owner, attribute, replacement in mutation.replace:
        monkeypatch.setattr(owner, attribute, replacement)
    for index, group in enumerate(mutation.breaks):
        failed = 0
        for number, (fn, args) in enumerate(group):
            workdir = tmp_path / f"{index}-{number}"
            workdir.mkdir()
            wants_mp = "monkeypatch" in inspect.signature(fn).parameters
            extra: dict[str, Any] = {}
            patches = pytest.MonkeyPatch()
            if wants_mp:
                extra["monkeypatch"] = patches
            try:
                needs_dir = "tmp_path" in inspect.signature(fn).parameters
                fn(*((workdir,) if needs_dir else ()), *args, **extra)
            except (AssertionError, pytest.fail.Exception):
                failed += 1
            finally:
                patches.undo()
        assert failed, f"{mutation.name}: {group[0][0].__name__} passed without its protection"


def test_every_guard_has_a_mutation() -> None:
    named = {fn.__name__ for m in MUTATIONS for group in m.breaks for fn, _ in group}
    # Tests that exercise the process, not one protection, or that a gate test covers.
    exempt = {
        "test_the_generation_routes",
        "test_a_body_that_is_not_one_json_object_is_denied",
        "test_a_connection_header_cannot_remove_the_injected_key",
        # Panoptes returns an unknown cost for it; the gate's charge-in-full rule for an
        # unknown cost has its own mutation (mb-017).
        "test_a_reply_without_usage_is_charged_the_whole_reservation",
        "test_a_provider_error_with_no_output_is_released",
        "test_a_reply_cut_short_is_charged_the_whole_reservation",
        "test_concurrent_requests_are_each_reserved_and_settled",
        "test_a_reservation_left_open_by_a_crash_is_reclaimed_by_the_sweep",
        "test_the_log_stays_one_valid_chain_with_both_writers",
        "test_no_key_or_token_reaches_the_audit_log_or_the_store",
    }
    tests = {
        name
        for name, fn in vars(t).items()
        if name.startswith("test_") and callable(fn) and fn.__module__ == t.__name__
    }
    assert tests - named - exempt == set()
