"""The OIDC verifier against a real provider: Keycloak, with the test realm in
tests/keycloak/realm-agentgate-test.json imported.

Run only when AGENTGATE_KEYCLOAK_URL is set (the `keycloak` CI job, or locally):

    docker run -d -p 127.0.0.1:8080:8080 -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \\
      -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin -v $PWD/tests/keycloak:/opt/keycloak/data/import \\
      quay.io/keycloak/keycloak:26.8.2 start-dev --import-realm
    AGENTGATE_KEYCLOAK_URL=http://127.0.0.1:8080 uv run pytest -m keycloak

The realm's users match Kettle Creek fixture users by email. Its public client allows the
direct grant (username and password for a token, no browser), for these tests only.
"""

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import yaml

from agentgate.audit.base import EventKind
from agentgate.audit.panoptes import read_events, verify_log
from agentgate.cli import app
from agentgate.identity import IssuerConfig, TrustConfig
from agentgate.model.keys import ProviderKeys
from agentgate.model_proxy.client import ControlClient, ControlError
from agentgate.model_proxy.server import MultiRunProxy, ServerConfig
from agentgate.policy.schema import ModelProvider
from agentgate.world import open_world
from conftest import FIXTURES, KC, POLICIES
from proxyharness import PROVIDER_KEY, RATES
from test_cli_ops import runner

pytestmark = pytest.mark.keycloak

BASE = os.environ.get("AGENTGATE_KEYCLOAK_URL", "").rstrip("/")
REALM = "agentgate-test"
CLIENT = "agentgate-test-cli"

if not BASE:
    pytest.skip("AGENTGATE_KEYCLOAK_URL is not set", allow_module_level=True)

ISSUER = f"{BASE}/realms/{REALM}"


def _issuer() -> IssuerConfig:
    return IssuerConfig.model_validate(
        {
            "name": "kc-keycloak",
            "kind": "oidc",
            "issuer": ISSUER,
            "audience": "agentgate",
            "customer": KC,
            "subject": "email",
            "algorithms": ["RS256", "ES256"],
            "max_lifetime_seconds": 3600,
            "single_use": False,
        }
    )


def access_token(username: str) -> str:
    """A real access token from Keycloak's direct grant, as the test client."""
    reply = httpx.post(
        f"{ISSUER}/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": CLIENT,
            "username": username,
            "password": f"test-only-{username}",
            "scope": "openid",
        },
        timeout=10,
    )
    assert reply.status_code == 200, reply.text
    return str(reply.json()["access_token"])


@contextmanager
def keycloak_server(tmp_path: Path) -> Iterator[tuple[MultiRunProxy, ControlClient, Path]]:
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    audit = tmp_path / "data" / "audit.db"
    world = open_world(
        db_path=tmp_path / "data" / "agentgate.db",
        audit_path=audit,
        policies=POLICIES,
        fixtures=FIXTURES,
    )
    upstream = ModelProvider.OPENAI_COMPATIBLE
    proxy = MultiRunProxy(
        world,
        ServerConfig(
            upstreams={upstream: "http://127.0.0.1:9"},
            keys=ProviderKeys({upstream: PROVIDER_KEY}),
            rate_table=rates,
            control_socket=Path(tempfile.mkdtemp(prefix="ag-kc-")) / "ctl" / "control.sock",
            sweep_interval=timedelta(hours=1),
            identity=TrustConfig((_issuer(),)),
        ),
    )
    _, socket_path = proxy.start()
    client = ControlClient(socket_path)
    try:
        yield proxy, client, audit
    finally:
        client.close()
        proxy.stop()
        world.close()


def test_a_direct_grant_token_starts_a_run_as_that_user(tmp_path: Path) -> None:
    with keycloak_server(tmp_path) as (proxy, client, audit):
        assert proxy.identity_problems == {}
        started = client.start_verified_run(access_token("nadine"))
        assert (started.customer_id, started.user_id) == (KC, "kc-staff-nadine")
        assert proxy.world.gate.run_assurance(started.run_id) == "verified:kc-keycloak"
        client.end_run(started.run_id)
    [event] = [e for e in read_events(audit) if e.kind is EventKind.RUN_STARTED]
    assert event.details["assurance"] == "verified:kc-keycloak"
    assert verify_log(audit)[0]["valid"]


def test_a_token_for_an_unverified_email_is_refused(tmp_path: Path) -> None:
    with keycloak_server(tmp_path) as (proxy, client, audit):
        with pytest.raises(ControlError) as refused:
            client.start_verified_run(access_token("colby"))
        assert (refused.value.status, refused.value.reason) == (403, "email_unverified")
        assert proxy.live_runs() == 0
    denied = [e for e in read_events(audit) if e.reason == "email_unverified"]
    assert len(denied) == 1 and denied[0].details["issuer"] == "kc-keycloak"


def test_a_verified_email_with_no_agentgate_user_is_refused(tmp_path: Path) -> None:
    with keycloak_server(tmp_path) as (_, client, _audit):
        with pytest.raises(ControlError) as refused:
            client.start_verified_run(access_token("stranger"))
        assert refused.value.reason == "subject_unknown"


def test_a_tampered_keycloak_token_is_refused(tmp_path: Path) -> None:
    header, payload, signature = access_token("nadine").split(".")
    flipped = signature[:-2] + ("A" if signature[-2] != "A" else "B") + signature[-1]
    with keycloak_server(tmp_path) as (_, client, _audit):
        with pytest.raises(ControlError) as refused:
            client.start_verified_run(f"{header}.{payload}.{flipped}")
        assert refused.value.reason == "signature_invalid"


def test_check_config_reads_the_realms_keys(tmp_path: Path) -> None:
    config = tmp_path / "identity.yaml"
    config.write_text(
        yaml.safe_dump({"issuers": [_issuer().model_dump(mode="json", exclude_none=True)]})
    )
    result = runner.invoke(app, ["identity", "check-config", str(config)])
    assert result.exit_code == 0, result.output
    assert "kc-keycloak  oidc  customer kettle-creek-hvac" in result.output
