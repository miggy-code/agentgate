"""The model proxy over real HTTP: a client, the proxy, and a fake provider that records what
reaches it. Each test is one boundary a client or a failure could cross; the mutation tests in
test_model_proxy_mutations.py remove the protection behind each and require these to fail.
"""

import contextlib
import json
import os
import stat
import threading
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from panoptes.pricing import PricingTable

from agentgate.audit.base import EventKind, OperatorAction
from agentgate.audit.panoptes import verify_log
from agentgate.gate import Gate
from agentgate.model.admission import Caps
from agentgate.model.pricing import RateTable
from agentgate.model.proxy import (
    ProxyConfig,
    ProxySetupError,
    check_rates,
    generation_route,
)
from agentgate.model.tokens import TokenReason, TokenRejected
from proxyharness import (
    BODY,
    OK_COST,
    PROVIDER_KEY,
    RATES,
    Provider,
    Script,
    rig,
)


def received(provider: Provider) -> dict[str, str]:
    (seen,) = provider.seen
    return {name.lower(): value for name, value in seen.headers}


# Routes


REFUSED_ROUTES = [
    ("GET", "/v1/chat/completions"),
    ("POST", "/v1/models"),
    ("GET", "/v1/models"),
    ("GET", "/v1/responses/resp_123"),
    ("POST", "/v1/responses/resp_123/cancel"),
    ("POST", "/v1/chat/completions/extra"),
    ("POST", "/v1/messages/count_tokens"),
]


@pytest.mark.parametrize(("method", "path"), REFUSED_ROUTES)
def test_only_a_post_to_a_generation_endpoint_reaches_the_provider(
    tmp_path: Path, method: str, path: str
) -> None:
    with rig(tmp_path) as (made, provider):
        reply = made.post(path, method=method)
        assert reply.status >= 400
        assert provider.seen == []
        assert made.reservations() == []


@pytest.mark.parametrize(
    ("method", "target", "expected"),
    [
        ("POST", "/v1/chat/completions", ("openai", None)),
        ("POST", "/v1/chat/completions?x=1", ("openai", None)),
        ("POST", "/v1/messages", ("anthropic", None)),
        ("POST", "/v1beta/models/gemini-pro:generateContent", ("gemini", "gemini-pro")),
        ("POST", "/v1/models/g:streamGenerateContent?alt=sse", ("gemini", "g")),
        # The model is read from the path as received: an encoded name is another name.
        ("POST", "/v1beta/models/gemini%2Dpro:generateContent", ("gemini", "gemini%2Dpro")),
        ("POST", "/v1beta/models/a/b:generateContent", None),
        ("POST", "/v1beta/models/g:countTokens", None),
        ("POST", "/v1/models", None),
        ("GET", "/v1/messages", None),
        ("POST", "/v1/responses/abc", None),
    ],
)
def test_the_generation_routes(method: str, target: str, expected: Any) -> None:
    found = generation_route(method, target)
    assert (None if found is None else (found.wire, found.path_model)) == expected


# Tokens


def test_a_request_without_a_valid_token_is_refused_and_never_forwarded(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        for token in (False, "agt_not-a-token", "sk-not-ours", ""):
            reply = made.post(token=token)
            assert reply.status == 401 and reply.error == "token_refused"
        assert provider.seen == [] and made.reservations() == []


def test_two_different_credentials_are_ambiguous(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        reply = made.post(headers={"X-Api-Key": "agt_another"})
        assert reply.status == 401
        assert provider.seen == []
        # The same token in two places is one credential.
        assert made.post(headers={"X-Api-Key": made.token}).status == 200


def test_a_token_in_the_query_is_read_and_never_forwarded(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        reply = made.post(f"/v1/chat/completions?key={made.token}&alt=json", token=False)
        assert reply.status == 200, reply.body
        (seen,) = provider.seen
        assert "key=" not in seen.target and "alt=json" in seen.target
        assert made.token not in seen.target


def test_every_client_credential_is_replaced_by_the_providers_key(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        reply = made.post(
            headers={
                "X-Api-Key": made.token,
                "Api-Key": made.token,
                "X-Goog-Api-Key": made.token,
                "Cookie": "session=abc",
                "Proxy-Authorization": "Basic Zm9vOmJhcg==",
            }
        )
        assert reply.status == 200, reply.body
        headers = received(provider)
        assert headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        for name in ("x-api-key", "api-key", "x-goog-api-key", "cookie", "proxy-authorization"):
            assert name not in headers
        (seen,) = provider.seen
        assert made.token not in json.dumps(seen.headers) + seen.target


def test_a_connection_header_cannot_remove_the_injected_key(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        reply = made.post(headers={"Connection": "authorization, x-api-key"})
        assert reply.status == 200, reply.body
        assert received(provider)["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_a_revoked_token_is_refused_on_its_runs_record(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        record = made.proxy._tokens.list_tokens(made.run.run_id)[0]
        made.proxy._tokens.revoke(
            record.token_id,
            TokenReason.OPERATOR,
            record=lambda gone: made.world.gate.record_token_event(
                OperatorAction.TOKEN_REVOKED, gone, customer_id=made.run.customer_id, actor=None
            ),
        )
        reply = made.post()
        assert reply.status == 401 and provider.seen == []
        refusals = [e for e in made.events() if e.reason == "token_refused"]
        assert [e.details["refused_because"] for e in refusals] == ["revoked"]
        assert refusals[0].run_id == made.run.run_id and refusals[0].exchange_id


def test_a_suspended_user_is_refused(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        made.world.store.connection.execute(
            "UPDATE users SET status = 'suspended' WHERE id = ?", (made.run.user_id,)
        )
        assert made.post().status == 401 and provider.seen == []


# Decisions


DENIED_BODIES = [
    ({**BODY, "model": "deepseek-pro"}, "model_not_allowed"),
    ({**BODY, "max_tokens": 9000}, "model_max_tokens_exceeded"),
    ({"model": "deepseek-flash", "messages": []}, "model_max_tokens_required"),
    ({**BODY, "n": 2}, "model_n_not_one"),
    ({**BODY, "previous_response_id": "resp_1"}, "model_server_feature"),
    ({**BODY, "surprise": 1}, "model_unknown_field"),
]


@pytest.mark.parametrize(("body", "reason"), DENIED_BODIES)
def test_a_request_the_policy_denies_gets_a_reason_code_and_goes_nowhere(
    tmp_path: Path, body: dict[str, Any], reason: str
) -> None:
    with rig(tmp_path) as (made, provider):
        reply = made.post(body=body)
        assert reply.status == 403 and reply.json() == {"error": reason}
        assert provider.seen == [] and made.reservations() == []


def test_a_body_that_is_not_one_json_object_is_denied(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        for body in (b'{"model": "deepseek-flash", "model": "x", "max_tokens": 1}', b"[]"):
            assert made.post(body=body).status in (400, 403)
        assert provider.seen == []


def test_the_run_cap_is_a_429_and_the_next_request_is_not_forwarded(tmp_path: Path) -> None:
    with rig(tmp_path, caps=Caps(max_calls=1)) as (made, provider):
        assert made.post().status == 200
        made.reservations()
        reply = made.post()
        assert reply.status == 429 and reply.json() == {"error": "max_calls"}
        assert len(provider.seen) == 1


def test_a_cap_below_the_worst_case_refuses_the_first_request(tmp_path: Path) -> None:
    with rig(tmp_path, caps=Caps(max_usd=Decimal("0.000001"))) as (made, provider):
        assert made.post().status == 429 and provider.seen == []


# Settlement


def test_a_complete_reply_is_charged_what_it_cost_and_joined_to_its_exchange(
    tmp_path: Path,
) -> None:
    with rig(tmp_path) as (made, _):
        reply = made.post(headers={"X-Panoptes-Correlation-Id": "client-corr-1"})
        assert reply.status == 200
        (row,) = made.reservations()
        assert row["status"] == "settled" and Decimal(row["actual_usd"]) == OK_COST
        assert Decimal(row["reserved_usd"]) > OK_COST

        decisions = [e for e in made.events() if e.kind is EventKind.ALLOW]
        ends = [e for e in made.events() if e.reason == "model_settled"]
        assert len(decisions) == len(ends) == 1
        exchange_id = decisions[0].exchange_id
        assert exchange_id and ends[0].exchange_id == exchange_id
        assert decisions[0].correlation_id == "client-corr-1"
        assert decisions[0].details["asserted_correlation_id"] == "client-corr-1"
        assert ends[0].details["charged_usd"] == str(OK_COST)

        observed = made.exchange_events()
        assert observed
        for event in observed:
            assert event["attributes"]["exchange_id"] == exchange_id
            assert event["agent"]["session_id"] == made.run.run_id
            assert event["agent"]["run_id"] == made.run.run_id
            assert event["user"] == {"id": made.run.user_id}
            assert event["attributes"]["identity"]["source"] == "hook"


def test_a_provider_error_with_no_output_is_released(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        provider.script = Script(status=500, body=b'{"error": {"message": "boom"}}')
        assert made.post().status == 500
        (row,) = made.reservations()
        assert row["status"] == "released"
        assert [e.reason for e in made.events() if e.kind is EventKind.EXECUTED] == [
            "model_released"
        ]


def test_a_reply_cut_short_is_charged_the_whole_reservation(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        provider.script = Script(truncate=True)
        with contextlib.suppress(Exception):  # the client may see a broken read; the charge matters
            made.post()
        (row,) = made.reservations()
        assert row["status"] == "settled"
        assert Decimal(row["actual_usd"]) == Decimal(row["reserved_usd"])


def test_a_reply_without_usage_is_charged_the_whole_reservation(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, provider):
        provider.script = Script(body=b'{"id": "c1", "choices": []}')
        assert made.post().status == 200
        (row,) = made.reservations()
        assert Decimal(row["actual_usd"]) == Decimal(row["reserved_usd"])


def test_concurrent_requests_are_each_reserved_and_settled(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, _):
        statuses: list[int] = []

        def one() -> None:
            statuses.append(made.post().status)

        threads = [threading.Thread(target=one) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert statuses == [200] * 8
        rows = made.reservations()
        assert len(rows) == 8 and all(row["status"] == "settled" for row in rows)
        assert made.spent() == OK_COST * 8


# Failing closed


def test_an_error_in_the_decision_is_a_503_and_nothing_is_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("store failed")

    with rig(tmp_path) as (made, provider):
        monkeypatch.setattr(Gate, "admit_model", broken)
        reply = made.post()
        assert reply.status == 503 and reply.json() == {"error": "unavailable"}
        assert provider.seen == []


def test_a_failure_after_the_reservation_is_a_hook_error_and_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from panoptes.hooks import Exchange

    def broken(self: Exchange) -> None:
        raise RuntimeError("cannot drop")

    with rig(tmp_path) as (made, provider):
        monkeypatch.setattr(Exchange, "drop_query_credentials", broken)
        assert made.post().status >= 500
        assert provider.seen == []
        (row,) = made.reservations()
        assert row["status"] == "released"


def test_a_reservation_left_open_by_a_crash_is_reclaimed_by_the_sweep(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, _):
        made.world.store.connection.execute(
            "INSERT INTO spend_reservations (id, run_id, customer_id, day, status, reserved_usd,"
            " reserved_at) VALUES ('stale', ?, ?, '2026-10-22', 'open', '0.01', ?)",
            (made.run.run_id, made.run.customer_id, "2026-10-22T10:00:00+00:00"),
        )
        assert made.world.gate.release_stale(timedelta(minutes=15), record=True) == 1


# The process


def test_the_log_stays_one_valid_chain_with_both_writers(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, _):
        made.post()
        made.post(body={**BODY, "model": "deepseek-pro"})
        made.reservations()
        result, count = verify_log(made.audit_path)
        assert result["valid"], result
        assert count > 4


def test_no_key_or_token_reaches_the_audit_log_or_the_store(tmp_path: Path) -> None:
    with rig(tmp_path) as (made, _):
        made.post(headers={"X-Api-Key": made.token}, path=f"/v1/chat/completions?key={made.token}")
        made.reservations()
        token = made.token
        paths = [made.audit_path, tmp_path / "data" / "agentgate.db"]
    for path in paths:
        content = path.read_bytes()
        assert PROVIDER_KEY.encode() not in content and token.encode() not in content


def test_stopping_drains_ends_the_run_and_revokes_its_token(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    with rig(tmp_path) as (made, _):
        made.proxy.stop()
        record = made.proxy._tokens.list_tokens(made.run.run_id)[0]
        assert record.revoked_reason == "run_ended"
        with pytest.raises(TokenRejected) as refused:
            made.proxy._tokens.resolve(made.token)
        assert refused.value.reason is TokenReason.REVOKED
        revoked = [e for e in made.events() if e.reason == OperatorAction.TOKEN_REVOKED.value]
        assert revoked and revoked[-1].user_id is None
    assert not token_file.exists()


def test_the_token_file_is_private_and_removed_when_the_proxy_stops(tmp_path: Path) -> None:
    from agentgate.model.proxy import ModelProxy
    from conftest import open_file_world
    from proxyharness import STAFF

    provider = Provider()
    world = open_file_world(tmp_path)
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    token_file = tmp_path / "token"
    proxy = ModelProxy(
        world,
        STAFF,
        ProxyConfig(
            upstreams={next(iter(_providers())): provider.origin},
            keys=_keys(),
            rate_table=rates,
            token_file=token_file,
        ),
    )
    try:
        proxy.start()
        assert proxy.token is None
        assert stat.S_IMODE(os.stat(token_file).st_mode) == 0o600
        token = token_file.read_text().strip()
        assert proxy._tokens.resolve(token).run_id == proxy.run.run_id
    finally:
        proxy.stop()
        world.close()
        provider.close()
    assert not token_file.exists()


def _providers() -> list[Any]:
    from agentgate.policy.schema import ModelProvider

    return [ModelProvider.OPENAI_COMPATIBLE]


def _keys() -> Any:
    from agentgate.model.keys import ProviderKeys
    from agentgate.policy.schema import ModelProvider

    return ProviderKeys({ModelProvider.OPENAI_COMPATIBLE: PROVIDER_KEY})


# Starting


def test_two_rate_tables_that_disagree_do_not_start() -> None:
    reserve = RateTable.from_dict(RATES)
    same = PricingTable.from_dict(RATES)
    check_rates(reserve, same, {"deepseek-flash"})

    other = json.loads(json.dumps(RATES))
    other["models"]["deepseek-flash"]["output_per_million"] = "9"
    with pytest.raises(ProxySetupError):
        check_rates(reserve, PricingTable.from_dict(other), {"deepseek-flash"})
    other = {**RATES, "version": "other"}
    with pytest.raises(ProxySetupError):
        check_rates(reserve, PricingTable.from_dict(other), {"deepseek-flash"})
    priced_by_one = json.loads(json.dumps(RATES))
    priced_by_one["models"]["extra"] = priced_by_one["models"]["deepseek-flash"]
    with pytest.raises(ProxySetupError):
        check_rates(reserve, PricingTable.from_dict(priced_by_one), {"extra"})


def test_the_proxy_does_not_start_without_a_key_for_its_upstream(tmp_path: Path) -> None:
    from agentgate.model.keys import ProviderKeys
    from agentgate.model.proxy import ModelProxy
    from agentgate.policy.schema import ModelProvider
    from conftest import open_file_world
    from proxyharness import STAFF

    world = open_file_world(tmp_path)
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    config = ProxyConfig(
        upstreams={ModelProvider.OPENAI_COMPATIBLE: "http://127.0.0.1:1"},
        keys=ProviderKeys({ModelProvider.ANTHROPIC: PROVIDER_KEY}),
        rate_table=rates,
    )
    try:
        with pytest.raises(ProxySetupError, match="no provider key"):
            ModelProxy(world, STAFF, config).start()
    finally:
        world.close()
