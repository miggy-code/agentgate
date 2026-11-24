"""Verified identity at run start (agentgate.identity) and its use by the multi-run proxy.

Keys are generated in each test. The OIDC issuer is a fake provider behind an httpx
MockTransport; the same verifier against a real provider (Keycloak) is in
test_identity_keycloak.py. Each reason code has its own failing case; the mutation tests at
the end remove the three protections the spec names and require their tests to fail.
"""

import gzip
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from agentgate.audit.base import EventKind
from agentgate.audit.panoptes import read_events, verify_log
from agentgate.cli import app
from agentgate.clock import FixedClock
from agentgate.gate import IdentityError
from agentgate.identity import (
    AssertionReason,
    AssertionRefused,
    AssertionVerifier,
    IdentityConfigError,
    IssuerConfig,
    TrustConfig,
    VerifiedSubject,
    load_identity_config,
    mint_assertion,
    public_jwk,
)
from agentgate.identity import verify as verify_module
from agentgate.identity.mint import jwk_thumbprint, load_signing_key, write_keypair
from agentgate.identity.oidc import REFETCH_INTERVAL_SECONDS
from agentgate.model.keys import ProviderKeys
from agentgate.model_proxy.client import ControlClient, ControlError
from agentgate.model_proxy.proxy import ProxySetupError
from agentgate.model_proxy.server import MultiRunProxy, ServerConfig
from agentgate.policy.schema import ModelProvider
from agentgate.store import Store
from agentgate.world import World, open_world
from conftest import FIXTURES, KC, POLICIES, START, mutant
from conftest import World as MemoryWorld
from proxyharness import PROVIDER_KEY, RATES
from test_cli_ops import runner

STAFF = "kc-staff-nadine"
COLBY = "kc-staff-colby"
JR_OWNER = "jr-owner-marisol"
PERSONAS = "https://brain.local/personas"
OIDC = "https://id.example/realms/kc"
NOW = START.timestamp()


# Keys and assertions


@dataclass
class Signer:
    """An Ed25519 persona signer, as a host holds one."""

    key: Ed25519PrivateKey = field(default_factory=Ed25519PrivateKey.generate)
    # The time assertions are issued at: the fixed test clock's, or the wall clock's for a
    # file world.
    now: float = NOW

    @property
    def jwk(self) -> dict[str, Any]:
        return public_jwk(self.key)

    @property
    def kid(self) -> str:
        return str(self.jwk["kid"])

    def claims(self, **changes: Any) -> dict[str, Any]:
        claims: dict[str, Any] = {
            "iss": PERSONAS,
            "aud": "agentgate:kc",
            "sub": "nadine@kettlecreek.example",
            "iat": int(self.now),
            "exp": int(self.now) + 120,
            "jti": os.urandom(8).hex(),
        }
        claims.update(changes)
        return {k: v for k, v in claims.items() if v is not None}

    def sign(self, headers: dict[str, Any] | None = None, **changes: Any) -> str:
        sent = {"kid": self.kid} | (headers or {})
        return jwt.encode(self.claims(**changes), self.key, algorithm="EdDSA", headers=sent)


def persona_issuer(**changes: Any) -> IssuerConfig:
    fields: dict[str, Any] = {
        "name": "kc-personas",
        "kind": "signed",
        "issuer": PERSONAS,
        "audience": "agentgate:kc",
        "customer": KC,
        "subject": "email",
        "jwks_file": "personas.jwks.json",
        "algorithms": ["EdDSA"],
        "max_lifetime_seconds": 300,
        "single_use": True,
    }
    fields.update(changes)
    return IssuerConfig.model_validate(fields)


def oidc_issuer(**changes: Any) -> IssuerConfig:
    fields: dict[str, Any] = {
        "name": "kc-oidc",
        "kind": "oidc",
        "issuer": OIDC,
        "audience": "agentgate",
        "customer": KC,
        "subject": "email",
        "algorithms": ["ES256"],
        "max_lifetime_seconds": 3600,
        "single_use": False,
    }
    fields.update(changes)
    return IssuerConfig.model_validate(fields)


class FakeProvider:
    """An OIDC provider behind an httpx MockTransport: discovery, a JWKS of EC keys, a count of
    each fetch, and a switch to make it unreachable."""

    def __init__(self, issuer: str = OIDC, now: float = NOW) -> None:
        self.issuer = issuer
        self.now = now
        self.keys = [ec.generate_private_key(ec.SECP256R1())]
        self.fetches: list[str] = []
        self.reachable = True
        self.discovered_issuer = issuer
        self.cache_control: str | None = None
        self.gzip = False
        self.http = httpx.Client(transport=httpx.MockTransport(self._handle))

    def kid(self, index: int = 0) -> str:
        return f"ec-{index}"

    def rotate(self) -> None:
        self.keys.append(ec.generate_private_key(ec.SECP256R1()))

    def token(self, index: int = 0, **changes: Any) -> str:
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "aud": ["agentgate", "account"],
            "sub": "a1b2c3",
            "email": "nadine@kettlecreek.example",
            "email_verified": True,
            "iat": int(self.now),
            "exp": int(self.now) + 300,
            "jti": os.urandom(8).hex(),
        }
        claims.update(changes)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(
            claims, self.keys[index], algorithm="ES256", headers={"kid": self.kid(index)}
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.fetches.append(url)
        if not self.reachable:
            raise httpx.ConnectError("unreachable", request=request)
        if url == f"{self.issuer}/.well-known/openid-configuration":
            return httpx.Response(
                200, json={"issuer": self.discovered_issuer, "jwks_uri": f"{self.issuer}/certs"}
            )
        if url == f"{self.issuer}/certs":
            published = []
            for index, key in enumerate(self.keys):
                jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key.public_key()))
                published.append(jwk | {"kid": self.kid(index), "use": "sig", "alg": "ES256"})
            # An encryption key is published beside them, as Keycloak does.
            published.append({"kid": "enc-1", "kty": "RSA", "use": "enc", "n": "x", "e": "AQAB"})
            headers = {} if self.cache_control is None else {"Cache-Control": self.cache_control}
            body = json.dumps({"keys": published}).encode()
            if self.gzip:
                headers["Content-Encoding"] = "gzip"
                body = gzip.compress(body)
            return httpx.Response(200, content=body, headers=headers)
        return httpx.Response(404)

    def jwks_fetches(self) -> int:
        return sum(1 for url in self.fetches if url.endswith("/certs"))


class Monotonic:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Verifying:
    world: MemoryWorld
    signer: Signer
    provider: FakeProvider
    monotonic: Monotonic
    verifier: AssertionVerifier

    def refusal(self, token: object) -> str:
        try:
            subject = self.verifier.verify(token)
        except AssertionRefused as refused:
            return refused.reason.value
        return f"verified {subject.user_id}"


def make_verifier(
    world: MemoryWorld, *issuers: IssuerConfig, signer: Signer | None = None
) -> Verifying:
    signer = signer or Signer()
    provider = FakeProvider()
    monotonic = Monotonic()
    chosen = issuers or (persona_issuer(), oidc_issuer())
    trust = TrustConfig(
        chosen,
        {i.name: {signer.kid: signer.jwk} for i in chosen if i.kind.value == "signed"},
    )
    verifier = AssertionVerifier(
        trust,
        world.store,
        world.clock,
        reference=world.gate.audit_reference,
        http=provider.http,
        monotonic=monotonic,
    )
    verifier.load()
    return Verifying(world, signer, provider, monotonic, verifier)


@pytest.fixture
def verifying(world: MemoryWorld) -> Verifying:
    return make_verifier(world)


# A valid assertion


def test_a_valid_persona_assertion_names_the_user_in_the_issuers_customer(
    verifying: Verifying,
) -> None:
    subject = verifying.verifier.verify(verifying.signer.sign())
    assert isinstance(subject, VerifiedSubject)
    assert (subject.customer_id, subject.user_id) == (KC, STAFF)
    assert subject.assurance.label == "verified:kc-personas"
    assert subject.assurance.jti_ref is not None and subject.assurance.jti_ref.startswith("ref:")


def test_a_valid_oidc_token_names_the_user_by_its_verified_email(verifying: Verifying) -> None:
    subject = verifying.verifier.verify(verifying.provider.token())
    assert (subject.customer_id, subject.user_id) == (KC, STAFF)
    assert subject.assurance.label == "verified:kc-oidc"


def test_email_is_compared_lowercased(verifying: Verifying) -> None:
    assert verifying.refusal(verifying.signer.sign(sub="Nadine@KettleCreek.Example")) == (
        f"verified {STAFF}"
    )


@pytest.mark.parametrize(
    "lookalike",
    [
        "nadine@\u212aettlecreek.example",  # KELVIN SIGN, which Python lowercases to "k"
        "nad\u0130ne@kettlecreek.example",  # I WITH DOT ABOVE, lowercased to "i" + a mark
        "n\u0430dine@kettlecreek.example",  # CYRILLIC A
    ],
)
def test_a_unicode_lookalike_email_names_no_user(verifying: Verifying, lookalike: str) -> None:
    """Only ASCII letters are case-folded: a provider's verified address that merely folds
    to a user's address is another mailbox."""
    assert verifying.refusal(verifying.signer.sign(sub=lookalike)) == "subject_unknown"


def test_a_user_id_subject_is_matched_to_the_users_id(world: MemoryWorld) -> None:
    case = make_verifier(world, persona_issuer(subject="user_id"))
    assert case.refusal(case.signer.sign(sub=STAFF)) == f"verified {STAFF}"


# Each reason code


def _segments(token: str) -> list[str]:
    return token.split(".")


def _b64(data: dict[str, Any] | bytes) -> str:
    import base64

    raw = data if isinstance(data, bytes) else json.dumps(data).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _with_header(token: str, header: dict[str, Any]) -> str:
    _, payload, signature = _segments(token)
    return f"{_b64(header)}.{payload}.{signature}"


def _hs256_with_public_key(signer: Signer) -> str:
    """The algorithm-confusion forgery: HS256 keyed with the issuer's public key."""
    pem = signer.key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    header = {"alg": "HS256", "typ": "JWT", "kid": signer.kid}
    signing_input = f"{_b64(header)}.{_b64(signer.claims())}"
    import hashlib
    import hmac

    mac = hmac.new(pem, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(mac)}"


CASES: dict[str, Callable[[Verifying], object]] = {
    "assertion_malformed: not a string": lambda v: 12345,
    "assertion_malformed: two segments": lambda v: "a.b",
    "assertion_malformed: header not JSON": lambda v: (
        "bm90IGpzb24." + v.signer.sign().split(".", 1)[1]
    ),
    "assertion_malformed: a repeated claim": lambda v: (
        _segments(v.signer.sign())[0]
        + "."
        + _b64(b'{"iss":"' + PERSONAS.encode() + b'","iss":"x"}')
        + ".c2ln"
    ),
    "assertion_malformed: a crit header": lambda v: v.signer.sign(headers={"crit": ["exp"]}),
    "assertion_malformed: no exp": lambda v: v.signer.sign(exp=None),
    "assertion_malformed: no iat": lambda v: v.signer.sign(iat=None),
    "assertion_malformed: a boolean exp": lambda v: v.signer.sign(exp=True),
    "assertion_malformed: no jti on a single-use issuer": lambda v: v.signer.sign(jti=None),
    "assertion_malformed: over 8 KiB": lambda v: v.signer.sign(pad="x" * 8192),
    "issuer_unknown: an issuer not configured": lambda v: v.signer.sign(iss="https://evil/"),
    "issuer_unknown: no iss": lambda v: v.signer.sign(iss=None),
    "issuer_unknown: a trailing slash apart": lambda v: v.signer.sign(iss=PERSONAS + "/"),
    "algorithm_refused: alg none": lambda v: _with_header(
        v.signer.sign(), {"alg": "none", "kid": v.signer.kid}
    ),
    "algorithm_refused: HS256 keyed with the public key": lambda v: _hs256_with_public_key(
        v.signer
    ),
    "algorithm_refused: an algorithm the issuer does not list": lambda v: _with_header(
        v.signer.sign(), {"alg": "ES256", "kid": v.signer.kid}
    ),
    "key_unknown: no kid": lambda v: _with_header(v.signer.sign(), {"alg": "EdDSA"}),
    "key_unknown: a kid the issuer does not hold": lambda v: v.signer.sign(headers={"kid": "nope"}),
    "signature_invalid: signed by another key": lambda v: Signer().sign(
        headers={"kid": v.signer.kid}
    ),
    "signature_invalid: a claim changed after signing": lambda v: (
        _segments(v.signer.sign())[0]
        + "."
        + _b64(v.signer.claims(sub="dale@kettlecreek.example"))
        + "."
        + _segments(v.signer.sign())[2]
    ),
    "audience_mismatch: another audience": lambda v: v.signer.sign(aud="agentgate:jr"),
    "audience_mismatch: no aud": lambda v: v.signer.sign(aud=None),
    "expired: past exp and leeway": lambda v: v.signer.sign(iat=int(NOW) - 200, exp=int(NOW) - 31),
    "not_yet_valid: iat ahead": lambda v: v.signer.sign(iat=int(NOW) + 60, exp=int(NOW) + 120),
    "not_yet_valid: nbf ahead": lambda v: v.signer.sign(nbf=int(NOW) + 60),
    "lifetime_too_long: exp - iat over the issuer's maximum": lambda v: v.signer.sign(
        exp=int(NOW) + 301
    ),
    "subject_unknown: no such user": lambda v: v.signer.sign(sub="nobody@kettlecreek.example"),
    "subject_unknown: another customer's user": lambda v: v.signer.sign(
        sub="marisol@juniperrow.example"
    ),
    "subject_unknown: no sub": lambda v: v.signer.sign(sub=None),
    "email_unverified: email_verified false": lambda v: v.provider.token(email_verified=False),
    "email_unverified: email_verified missing": lambda v: v.provider.token(email_verified=None),
    "email_unverified: email_verified the string true": lambda v: v.provider.token(
        email_verified="true"
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_failure_is_refused_with_its_reason_code(verifying: Verifying, case: str) -> None:
    expected = case.split(":")[0]
    assert verifying.refusal(CASES[case](verifying)) == expected


def test_every_reason_code_has_a_failing_case() -> None:
    covered = {case.split(":")[0] for case in CASES} | {"replayed", "issuer_unavailable"}
    assert covered == {reason.value for reason in AssertionReason}


def test_replay_is_refused_only_when_the_issuer_is_single_use(world: MemoryWorld) -> None:
    single = make_verifier(world, persona_issuer())
    token = single.signer.sign()
    assert single.refusal(token) == f"verified {STAFF}"
    assert single.refusal(token) == "replayed"

    reusable = make_verifier(world, persona_issuer(single_use=False))
    token = reusable.signer.sign()
    assert reusable.refusal(token) == f"verified {STAFF}"
    assert reusable.refusal(token) == f"verified {STAFF}"


def test_concurrent_uses_of_one_single_use_assertion_start_one_run(tmp_path: Path) -> None:
    """Sixteen control requests race with one assertion: exactly one run starts."""
    with serving(tmp_path) as served:
        token = _minted(served.signer)
        barrier = threading.Barrier(16)
        outcomes: list[str] = []

        def race() -> None:
            barrier.wait()
            with ControlClient(served.client.socket_path) as client:
                try:
                    outcomes.append(client.start_verified_run(token).user_id)
                except ControlError as error:
                    outcomes.append(error.reason)

        threads = [threading.Thread(target=race) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        # A connect beyond the socket's listen backlog can be refused (`unreachable`); every
        # request that reached the server either started the one run or was a replay.
        assert len(outcomes) == 16 and outcomes.count(STAFF) == 1
        assert set(outcomes) - {STAFF} <= {"replayed", "unreachable"}
        assert "replayed" in outcomes
        assert served.proxy.live_runs() == 1


def test_a_replay_names_the_jti_by_reference_only(verifying: Verifying) -> None:
    token = verifying.signer.sign(jti="jti-plain-text")
    verifying.verifier.verify(token)
    with pytest.raises(AssertionRefused) as refused:
        verifying.verifier.verify(token)
    assert refused.value.jti_ref == verifying.world.gate.audit_reference("jti-plain-text")
    assert "jti-plain-text" not in json.dumps(refused.value.details())
    stored = verifying.world.store.connection.execute(
        "SELECT jti_sha256 FROM used_assertions"
    ).fetchall()
    assert [row[0] for row in stored] != ["jti-plain-text"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"iat": int(NOW) - 100, "exp": int(NOW) - 29}, f"verified {STAFF}"),
        ({"iat": int(NOW) - 100, "exp": int(NOW) - 30}, "expired"),
        ({"iat": int(NOW) + 30, "exp": int(NOW) + 90}, f"verified {STAFF}"),
        ({"iat": int(NOW) + 31, "exp": int(NOW) + 90}, "not_yet_valid"),
        ({"nbf": int(NOW) + 30}, f"verified {STAFF}"),
        ({"nbf": int(NOW) + 31}, "not_yet_valid"),
        ({"exp": int(NOW) + 300}, f"verified {STAFF}"),
    ],
)
def test_the_clock_leeway_applies_at_both_edges(
    verifying: Verifying, changes: dict[str, Any], expected: str
) -> None:
    assert verifying.refusal(verifying.signer.sign(**changes)) == expected


def test_a_used_assertion_is_pruned_once_it_could_no_longer_pass(verifying: Verifying) -> None:
    clock = verifying.world.clock
    assert isinstance(clock, FixedClock)
    first = verifying.signer.sign(exp=int(NOW) + 60)
    verifying.verifier.verify(first)
    rows = "SELECT COUNT(*) FROM used_assertions"
    assert verifying.world.store.connection.execute(rows).fetchone()[0] == 1
    clock.advance(timedelta(seconds=91))
    # Past exp and leeway: the old row goes when the next assertion is recorded, and the old
    # assertion itself is refused by its time.
    verifying.verifier.verify(verifying.signer.sign(iat=int(NOW) + 91, exp=int(NOW) + 151))
    assert verifying.world.store.connection.execute(rows).fetchone()[0] == 1
    assert verifying.refusal(first) == "expired"


# The customer binding


@pytest.fixture
def shared_domain_fixtures(tmp_path: Path) -> Path:
    """The fixtures with a Juniper Row user whose address is in Kettle Creek's domain."""
    copied = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, copied)
    users = copied / "juniper-row-apothecary" / "users.yaml"
    users.write_text(
        users.read_text()
        + "\n- id: jr-staff-ana\n  name: Ana Lucero\n  email: ana@kettlecreek.example\n"
        "  role: staff\n  status: active\n"
    )
    return copied


def test_an_issuer_cannot_start_a_run_for_another_customers_user_with_its_domain(
    tmp_path: Path, shared_domain_fixtures: Path
) -> None:
    """Kettle Creek's issuer vouches for ana@kettlecreek.example, a Juniper Row user, and
    for a Juniper Row user id: neither starts a run, in either customer."""
    with serving(tmp_path, fixtures=shared_domain_fixtures) as served:
        user = served.world.store.get_user("jr-staff-ana")
        assert user is not None and user.customer_id == "juniper-row-apothecary"
        refused = []
        for token in (
            served.signer.sign(sub="ana@kettlecreek.example"),
            served.user_id_signer.sign(iss=PERSONAS + "/ids", aud="agentgate:kc", sub=JR_OWNER),
        ):
            try:
                served.proxy.start_verified_run(token)
                refused.append("started")
            except IdentityError as error:
                refused.append(error.reason)
        assert refused == ["subject_unknown", "subject_unknown"]
        assert served.proxy.live_runs() == 0
        assert served.world.store.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


# OIDC keys


def test_a_rotated_key_is_fetched_once_then_refetches_are_rate_limited(
    verifying: Verifying,
) -> None:
    provider, monotonic = verifying.provider, verifying.monotonic
    # The fetch at start counts toward the interval.
    assert provider.jwks_fetches() == 1
    provider.rotate()
    monotonic.now += REFETCH_INTERVAL_SECONDS
    assert verifying.refusal(provider.token(1)) == f"verified {STAFF}"
    assert provider.jwks_fetches() == 2
    # Another new kid inside the interval: no fetch, the kid is unknown.
    provider.rotate()
    monotonic.now += REFETCH_INTERVAL_SECONDS - 1
    assert verifying.refusal(provider.token(2)) == "key_unknown"
    assert verifying.refusal(provider.token(2)) == "key_unknown"
    assert provider.jwks_fetches() == 2
    # After it: one fetch, and the key is found.
    monotonic.now += 1
    assert verifying.refusal(provider.token(2)) == f"verified {STAFF}"
    assert provider.jwks_fetches() == 3
    # A kid the provider never publishes is refetched at most once per interval, and the
    # keys already held keep working.
    monotonic.now += REFETCH_INTERVAL_SECONDS
    for _ in range(5):
        forged = _with_header(provider.token(0), {"alg": "ES256", "kid": "never"})
        assert verifying.refusal(forged) == "key_unknown"
        assert verifying.refusal(provider.token(0)) == f"verified {STAFF}"
    assert provider.jwks_fetches() == 4


def test_the_jwks_is_cached_for_its_max_age_and_no_less_than_the_refetch_interval(
    world: MemoryWorld,
) -> None:
    case = make_verifier(world, oidc_issuer())
    provider, monotonic = case.provider, case.monotonic
    provider.cache_control = "public, max-age=0"
    monotonic.now += 601
    assert case.refusal(provider.token()) == f"verified {STAFF}"
    fetched = provider.jwks_fetches()
    # max-age=0 is held for the refetch interval, not refetched on every assertion.
    monotonic.now += REFETCH_INTERVAL_SECONDS - 1
    assert case.refusal(provider.token()) == f"verified {STAFF}"
    assert provider.jwks_fetches() == fetched
    monotonic.now += 1
    assert case.refusal(provider.token()) == f"verified {STAFF}"
    assert provider.jwks_fetches() == fetched + 1


def test_a_compressed_jwks_is_read_once_decoded(world: MemoryWorld) -> None:
    provider = FakeProvider()
    provider.gzip = True
    verifier = AssertionVerifier(
        TrustConfig((oidc_issuer(),)),
        world.store,
        world.clock,
        reference=world.gate.audit_reference,
        http=provider.http,
    )
    assert verifier.load() == {"kc-oidc": None}
    assert verifier.verify(provider.token()).user_id == STAFF


def test_an_unreachable_issuer_fails_closed_until_its_keys_load(world: MemoryWorld) -> None:
    provider = FakeProvider()
    provider.reachable = False
    monotonic = Monotonic()
    verifier = AssertionVerifier(
        TrustConfig((oidc_issuer(),)),
        world.store,
        world.clock,
        reference=world.gate.audit_reference,
        http=provider.http,
        monotonic=monotonic,
    )
    problems = verifier.load()
    assert problems["kc-oidc"] is not None and "unreachable" in str(problems["kc-oidc"])
    with pytest.raises(AssertionRefused) as refused:
        verifier.verify(provider.token())
    assert refused.value.reason is AssertionReason.ISSUER_UNAVAILABLE
    provider.reachable = True
    # Not before the refetch interval: still refused.
    with pytest.raises(AssertionRefused):
        verifier.verify(provider.token())
    monotonic.now += REFETCH_INTERVAL_SECONDS
    assert verifier.verify(provider.token()).user_id == STAFF


def test_expired_keys_are_not_used_when_the_issuer_stops_answering(verifying: Verifying) -> None:
    verifying.provider.reachable = False
    verifying.monotonic.now += 601
    assert verifying.refusal(verifying.provider.token()) == "issuer_unavailable"


def test_discovery_must_name_the_configured_issuer(world: MemoryWorld) -> None:
    provider = FakeProvider()
    provider.discovered_issuer = OIDC + "/"
    verifier = AssertionVerifier(
        TrustConfig((oidc_issuer(),)),
        world.store,
        world.clock,
        reference=world.gate.audit_reference,
        http=provider.http,
    )
    assert "another issuer" in str(verifier.load()["kc-oidc"])


def test_a_jku_or_x5u_header_is_ignored_and_never_fetched(verifying: Verifying) -> None:
    attacker = Signer()
    token = attacker.sign(
        headers={"jku": "https://evil.example/jwks", "x5u": "https://evil.example/cert"}
    )
    assert verifying.refusal(token) == "key_unknown"
    oidc = _with_header(
        verifying.provider.token(), {"alg": "ES256", "kid": "x", "jku": "https://evil.example/"}
    )
    assert verifying.refusal(oidc) == "key_unknown"
    assert not any("evil" in url for url in verifying.provider.fetches)


# The trust configuration


def _write_config(directory: Path, issuers: list[dict[str, Any]]) -> Path:
    import yaml

    path = directory / "identity.yaml"
    path.write_text(yaml.safe_dump({"issuers": issuers}))
    return path


def _persona_entry(**changes: Any) -> dict[str, Any]:
    entry = persona_issuer().model_dump(mode="json")
    entry.update(changes)
    return {k: v for k, v in entry.items() if v is not None}


def _keys_dir(tmp_path: Path) -> Path:
    signer = Signer()
    (tmp_path / "personas.jwks.json").write_text(json.dumps({"keys": [signer.jwk]}))
    return tmp_path


def test_a_valid_config_loads_with_its_signed_keys(tmp_path: Path) -> None:
    path = _write_config(_keys_dir(tmp_path), [_persona_entry()])
    trust = load_identity_config(path)
    assert [i.name for i in trust.issuers] == ["kc-personas"]
    assert len(trust.signed_keys["kc-personas"]) == 1


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([_persona_entry(algorithms=["HS256"])], "not accepted"),
        ([_persona_entry(algorithms=["none"])], "not accepted"),
        ([_persona_entry(extra="x")], "Extra inputs"),
        ([_persona_entry(jwks_file=None)], "needs jwks_file"),
        ([_persona_entry(algorithms=["EdDSA", "EdDSA"])], "twice"),
        ([_persona_entry(), _persona_entry(issuer=PERSONAS + "/x")], "name"),
        ([_persona_entry(), _persona_entry(name="other", issuer=PERSONAS + "/")], "twice"),
        (
            [
                oidc_issuer().model_dump(mode="json")
                | {"jwks_file": None, "issuer": "http://id.example/realms/kc"}
            ],
            "https",
        ),
    ],
)
def test_an_invalid_config_is_refused_at_load(
    tmp_path: Path, entries: list[dict[str, Any]], message: str
) -> None:
    cleaned = [{k: v for k, v in e.items() if v is not None} for e in entries]
    path = _write_config(_keys_dir(tmp_path), cleaned)
    with pytest.raises(IdentityConfigError, match=message):
        load_identity_config(path)


@pytest.mark.parametrize(
    "issuer", ["http://127.0.0.1:8080/realms/kc", "http://localhost:8080/r", "http://[::1]/r"]
)
def test_plain_http_is_allowed_for_an_oidc_issuer_on_a_loopback_host(issuer: str) -> None:
    assert oidc_issuer(issuer=issuer).issuer == issuer


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ({"kty": "oct", "kid": "s", "k": "c2VjcmV0"}, "only RSA, EC and OKP"),
        ({"kid": "p", "kty": "OKP", "crv": "Ed25519", "x": "AA", "d": "AA"}, "private"),
        ({"kty": "OKP", "crv": "Ed25519", "x": "AA"}, "no kid"),
        ({"kid": "e", "kty": "RSA", "n": "AA", "e": "AQAB", "use": "enc"}, "not a signing"),
    ],
)
def test_a_jwks_file_holds_public_signing_keys_only(
    tmp_path: Path, key: dict[str, Any], message: str
) -> None:
    (tmp_path / "personas.jwks.json").write_text(json.dumps({"keys": [key]}))
    path = _write_config(tmp_path, [_persona_entry()])
    with pytest.raises(IdentityConfigError, match=message):
        load_identity_config(path)


def test_a_config_built_by_hand_still_cannot_verify_with_hmac(world: MemoryWorld) -> None:
    """`model_construct` skips validation: the verifier refuses HS256 whatever the
    configuration says."""
    issuer = persona_issuer().model_copy(update={"algorithms": ("EdDSA", "HS256", "none")})
    case = make_verifier(world, issuer)
    assert case.refusal(_hs256_with_public_key(case.signer)) == "algorithm_refused"
    unsigned = _with_header(case.signer.sign(), {"alg": "none", "kid": case.signer.kid})
    assert case.refusal(unsigned) == "algorithm_refused"


# Host helpers


def _cli(*args: str) -> Any:
    result = runner.invoke(app, list(args))
    assert "Traceback" not in result.output
    return result


def test_keygen_writes_a_private_key_and_its_public_jwks(tmp_path: Path) -> None:
    out = tmp_path / "keys"
    out.mkdir()
    private, jwks, kid = write_keypair(out)
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    published = json.loads(jwks.read_text())["keys"]
    assert len(published) == 1 and "d" not in published[0]
    assert published[0]["kid"] == kid == jwk_thumbprint(published[0])
    key = load_signing_key(private)
    token = mint_assertion(
        key, issuer=PERSONAS, audience="a", subject="s", ttl=timedelta(minutes=1), kid=kid
    )
    assert jwt.get_unverified_header(token)["kid"] == kid
    with pytest.raises(FileExistsError):
        write_keypair(out)


def test_the_rfc_7638_thumbprint_matches_the_rfcs_example() -> None:
    # RFC 7638 section 3.1.
    jwk = {
        "kty": "RSA",
        "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6"
        "tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-6"
        "5YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyr"
        "dkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKn"
        "qDKgw",
        "e": "AQAB",
        "alg": "RS256",
        "kid": "2011-04-29",
    }
    assert jwk_thumbprint(jwk) == "NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs"


def test_cli_keygen_then_check_config(tmp_path: Path) -> None:
    out = tmp_path / "keys"
    result = _cli("identity", "keygen", "--out", str(out))
    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert "kid " in result.output and "BEGIN" not in result.output
    again = _cli("identity", "keygen", "--out", str(out))
    assert again.exit_code == 1 and "already exists" in again.output
    config = _write_config(out, [_persona_entry(jwks_file="jwks.json")])
    checked = _cli("identity", "check-config", str(config))
    assert checked.exit_code == 0, checked.output
    assert "kc-personas  signed  customer kettle-creek-hvac" in checked.output
    assert "1 keys" in checked.output


def test_cli_check_config_fails_on_an_unreachable_oidc_issuer(tmp_path: Path) -> None:
    unreachable = oidc_issuer(issuer="http://127.0.0.1:9/realms/kc").model_dump(mode="json")
    config = _write_config(tmp_path, [{k: v for k, v in unreachable.items() if v is not None}])
    checked = _cli("identity", "check-config", str(config))
    assert checked.exit_code == 1
    assert "kc-oidc  oidc" in checked.output and "unavailable" in checked.output


def test_cli_check_config_names_an_invalid_file(tmp_path: Path) -> None:
    config = _write_config(tmp_path, [_persona_entry(algorithms=["HS256"])])
    checked = _cli("identity", "check-config", str(config))
    assert checked.exit_code == 1 and "error:" in checked.output and "HS256" in checked.output


# Through the multi-run proxy


def _short_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ag-id-"))


@dataclass
class Served:
    world: World
    proxy: MultiRunProxy
    client: ControlClient
    signer: Signer
    user_id_signer: Signer
    provider: FakeProvider
    audit_path: Path

    def events(self) -> list[Any]:
        return read_events(self.audit_path)


@contextmanager
def serving(
    tmp_path: Path, *, fixtures: Path = FIXTURES, identity: bool = True
) -> Iterator[Served]:
    now = time.time()
    signer, user_id_signer, provider = Signer(now=now), Signer(now=now), FakeProvider(now=now)
    issuers = (
        persona_issuer(),
        persona_issuer(name="kc-ids", issuer=PERSONAS + "/ids", subject="user_id"),
        oidc_issuer(),
    )
    trust = TrustConfig(
        issuers,
        {
            "kc-personas": {signer.kid: signer.jwk},
            "kc-ids": {user_id_signer.kid: user_id_signer.jwk},
        },
    )
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    audit_path = tmp_path / "data" / "audit.db"
    world = open_world(
        db_path=tmp_path / "data" / "agentgate.db",
        audit_path=audit_path,
        policies=POLICIES,
        fixtures=fixtures,
    )
    upstream = ModelProvider.OPENAI_COMPATIBLE
    config = ServerConfig(
        upstreams={upstream: "http://127.0.0.1:9"},
        keys=ProviderKeys({upstream: PROVIDER_KEY}),
        rate_table=rates,
        control_socket=_short_dir() / "ctl" / "control.sock",
        sweep_interval=timedelta(hours=1),
        identity=trust if identity else None,
        identity_http=provider.http,
    )
    proxy = MultiRunProxy(world, config)
    _, socket_path = proxy.start()
    client = ControlClient(socket_path)
    try:
        yield Served(world, proxy, client, signer, user_id_signer, provider, audit_path)
    finally:
        client.close()
        proxy.stop()
        world.close()


def _minted(signer: Signer, subject: str = "nadine@kettlecreek.example") -> str:
    return mint_assertion(
        signer.key,
        issuer=PERSONAS,
        audience="agentgate:kc",
        subject=subject,
        ttl=timedelta(minutes=2),
        kid=signer.kid,
    )


def test_a_persona_assertion_starts_a_run_recorded_as_verified(tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        started = served.client.start_verified_run(_minted(served.signer), max_calls=3)
        assert (started.customer_id, started.user_id, started.max_calls) == (KC, STAFF, 3)
        assert served.world.gate.run_assurance(started.run_id) == "verified:kc-personas"
        [event] = [
            e
            for e in served.events()
            if e.kind is EventKind.RUN_STARTED and e.run_id == started.run_id
        ]
        assert event.details["assurance"] == "verified:kc-personas"
        assert event.details["issuer"] == "kc-personas"
        assert str(event.details["jti_ref"]).startswith("ref:")
        served.client.end_run(started.run_id)
    assert verify_log(served.audit_path)[0]["valid"]


def test_an_oidc_token_starts_a_run_through_the_control_socket(tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        started = served.client.start_verified_run(served.provider.token())
        assert started.user_id == STAFF
        assert served.world.gate.run_assurance(started.run_id) == "verified:kc-oidc"


def test_a_server_with_an_identity_config_refuses_the_bare_user_id_form(tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        with pytest.raises(ControlError) as refused:
            served.client.start_run(KC, STAFF)
        assert (refused.value.status, refused.value.reason) == (403, "assertion_required")
        assert served.proxy.live_runs() == 0
        denied = [e for e in served.events() if e.reason == "assertion_required"]
        assert len(denied) == 1 and denied[0].kind is EventKind.DENY


def test_a_server_without_one_keeps_the_user_id_form_and_trusts_no_issuer(
    tmp_path: Path,
) -> None:
    with serving(tmp_path, identity=False) as served:
        started = served.client.start_run(KC, STAFF)
        assert served.world.gate.run_assurance(started.run_id) is None
        with pytest.raises(ControlError) as refused:
            served.client.start_verified_run(_minted(served.signer))
        assert refused.value.reason == "issuer_unknown"


def test_a_refusal_is_recorded_with_its_reason_and_never_the_assertion(tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        token = _minted(served.signer)
        served.client.start_verified_run(token)
        reasons = []
        for sent in (token, served.signer.sign(aud="other"), "garbage"):
            with pytest.raises(ControlError) as refused:
                served.client.start_verified_run(sent)
            assert sent not in str(refused.value)
            reasons.append(refused.value.reason)
        assert reasons == ["replayed", "audience_mismatch", "assertion_malformed"]
        denied = [
            e for e in served.events() if e.kind is EventKind.DENY and e.reason in set(reasons)
        ]
        assert [e.reason for e in denied] == reasons
        assert denied[0].details["issuer"] == "kc-personas"
        assert str(denied[0].details["jti_ref"]).startswith("ref:")
        assert denied[2].customer_id == "unknown"
    audit = served.audit_path.read_bytes()
    for part in token.split("."):
        assert part.encode() not in audit
    assert verify_log(served.audit_path)[0]["valid"]


def test_a_suspended_users_assertion_is_refused_by_the_gate(tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        served.world.store.suspend_user(STAFF)
        with pytest.raises(ControlError) as refused:
            served.client.start_verified_run(_minted(served.signer))
        assert (refused.value.status, refused.value.reason) == (403, "suspended_user")
        [event] = [e for e in served.events() if e.reason == "suspended_user"]
        assert event.details["assurance"] == "verified:kc-personas"


def test_an_unavailable_issuer_is_answered_503_and_the_server_still_serves(
    tmp_path: Path,
) -> None:
    with serving(tmp_path) as served:
        served.provider.reachable = False
        token = served.provider.token()
        # The keys loaded at start; drop them as an expiry would.
        keys = served.proxy.verifier.oidc_keys("kc-oidc") if served.proxy.verifier else None
        assert keys is not None
        keys._keys = None
        with pytest.raises(ControlError) as refused:
            served.client.start_verified_run(token)
        assert (refused.value.status, refused.value.reason) == (503, "issuer_unavailable")
        assert served.client.start_verified_run(_minted(served.signer)).user_id == STAFF


def test_an_assertion_just_over_8_kib_reaches_the_verifier(tmp_path: Path) -> None:
    with serving(tmp_path) as served:
        big = served.signer.sign(pad="x" * 8200)
        assert 8192 < len(big) < 16384
        with pytest.raises(ControlError) as refused:
            served.client.start_verified_run(big)
        assert (refused.value.status, refused.value.reason) == (403, "assertion_malformed")


@pytest.mark.parametrize(
    "body",
    [
        {"assertion": "x", "user_id": STAFF},
        {"assertion": "x", "customer_id": KC},
        {"assertion": "x", "extra": 1},
    ],
)
def test_an_assertion_start_with_other_identity_fields_is_400(
    tmp_path: Path, body: dict[str, Any]
) -> None:
    with serving(tmp_path) as served:
        transport = httpx.HTTPTransport(uds=str(served.client.socket_path))
        with httpx.Client(transport=transport, base_url="http://agentgate-control") as raw:
            response = raw.post("/runs", json=body)
        assert (response.status_code, response.json()) == (400, {"error": "unknown_field"})


def test_the_server_refuses_an_issuer_for_a_customer_without_a_policy(tmp_path: Path) -> None:
    signer = Signer()
    trust = TrustConfig(
        (persona_issuer(customer="no-such-customer"),), {"kc-personas": {signer.kid: signer.jwk}}
    )
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps(RATES))
    world = open_world(
        db_path=tmp_path / "data" / "agentgate.db",
        audit_path=tmp_path / "data" / "audit.db",
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
            identity=trust,
        ),
    )
    try:
        with pytest.raises(ProxySetupError, match="no-such-customer"):
            proxy.start()
    finally:
        proxy.stop()
        world.close()


# Storage


def test_an_old_runs_table_gains_the_assurance_column(tmp_path: Path) -> None:
    path = tmp_path / "data" / "agentgate.db"
    path.parent.mkdir()
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,"
            " user_id TEXT NOT NULL, role TEXT NOT NULL, started_at TEXT NOT NULL,"
            " max_label TEXT) STRICT"
        )
        conn.execute(
            "INSERT INTO runs VALUES ('old-run', ?, ?, 'staff', '2026-10-01T00:00:00+00:00', NULL)",
            (KC, STAFF),
        )
    os.chmod(path, 0o600)
    world = open_world(
        db_path=path,
        audit_path=tmp_path / "data" / "audit.db",
        policies=POLICIES,
        fixtures=FIXTURES,
    )
    try:
        assert world.gate.run_assurance("old-run") is None
        ctx = world.gate.start_run(KC, STAFF)
        assert world.gate.run_assurance(ctx.run_id) is None
    finally:
        world.close()


def test_a_read_only_open_does_not_add_used_assertions(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with Store(path) as store:
        store.connection.execute("DROP TABLE used_assertions")
    with Store(path, readonly=True) as reader:
        tables = {
            row[0] for row in reader.connection.execute("SELECT name FROM sqlite_master").fetchall()
        }
    assert "used_assertions" not in tables


# Mutation proofs: each protection removed, its test must fail.


def _fails(test: Callable[[], None]) -> bool:
    try:
        test()
    except (AssertionError, pytest.fail.Exception):
        return True
    return False


def test_without_the_customer_binding_the_cross_customer_test_fails(
    tmp_path: Path, shared_domain_fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verifier searches every customer for the subject and starts the run in the
    user's own customer: the Kettle Creek issuer then starts Juniper Row runs."""
    edits = (
        (
            "matches = self._store.users_by_email(issuer.customer, value)",
            "matches = [u for c in self._store.list_customers()"
            " for u in self._store.users_by_email(c.id, value)]",
        ),
        (
            "if user is None or user.customer_id != issuer.customer:",
            "if user is None:",
        ),
    )
    cls = verify_module.AssertionVerifier
    monkeypatch.setattr(cls, "_subject", mutant(cls, "_subject", edits, {}))
    monkeypatch.setattr(
        cls,
        "verify",
        mutant(
            cls,
            "verify",
            (
                (
                    "VerifiedSubject(issuer.customer, user.id",
                    "VerifiedSubject(user.customer_id, user.id",
                ),
            ),
            {},
        ),
    )
    assert _fails(
        lambda: test_an_issuer_cannot_start_a_run_for_another_customers_user_with_its_domain(
            tmp_path, shared_domain_fixtures
        )
    )


def test_without_the_jti_check_the_replay_test_fails(
    world: MemoryWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls = verify_module.AssertionVerifier
    edits = (("if issuer.single_use:", "if False:"),)
    monkeypatch.setattr(cls, "verify", mutant(cls, "verify", edits, {}))
    assert _fails(lambda: test_replay_is_refused_only_when_the_issuer_is_single_use(world))
