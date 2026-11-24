"""Identity spoofing at a verified run start: forged, borrowed and stretched identity
assertions against a multi-run proxy that starts runs only from a signed assertion
(`--identity-config`, agentgate.identity).

Each scenario starts a real server on a durable file world, trusting two persona signers:
Kettle Creek's and Juniper Row's, each bound to its own customer. It first starts a run from
a good assertion (the control, so a scenario cannot pass by refusing everything), then makes
its attack and requires the refusal with its own reason code, recorded on the chain, and no
run started. The keys are generated per scenario; the clock is the arena's.

These run in-process under both transports: run starts are not carried over MCP.
"""

import base64
import hashlib
import hmac
import http.server
import json
import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from agentgate.audit.base import AuditEvent, EventKind
from agentgate.gate import IdentityError
from agentgate.identity import IssuerConfig, TrustConfig, mint_assertion, public_jwk
from agentgate.model.keys import ProviderKeys
from agentgate.model_proxy.server import MultiRunProxy, ServerConfig
from agentgate.policy.schema import ModelProvider
from agentgate.redteam.core import Arena, Category, scenario
from agentgate.redteam.files import FileWorlds
from agentgate.world import World

ID = Category.IDENTITY_SPOOFING
KC = "kettle-creek-hvac"
JR = "juniper-row-apothecary"
KC_PERSONAS = "https://brain.local/kettle-creek/personas"
JR_PERSONAS = "https://brain.local/juniper-row/personas"
KC_AUDIENCE = "agentgate:kettle-creek"
JR_AUDIENCE = "agentgate:juniper-row"
NADINE = "nadine@kettlecreek.example"
COLBY = "colby@kettlecreek.example"
PROVIDER = ModelProvider.OPENAI_COMPATIBLE
RATES: dict[str, object] = {
    "version": "redteam-identity",
    "source": "made up for the red-team suite",
    "currency": "USD",
    "models": {
        "deepseek-flash": {
            "input_per_million": "0.14",
            "cached_input_per_million": "0.014",
            "output_per_million": "0.28",
        }
    },
}


@dataclass
class _Signer:
    issuer: str
    audience: str
    key: Ed25519PrivateKey = field(default_factory=Ed25519PrivateKey.generate)

    @property
    def jwk(self) -> dict[str, Any]:
        return public_jwk(self.key)

    @property
    def kid(self) -> str:
        return str(self.jwk["kid"])


def _issuer(name: str, signer: _Signer, customer: str) -> IssuerConfig:
    return IssuerConfig.model_validate(
        {
            "name": name,
            "kind": "signed",
            "issuer": signer.issuer,
            "audience": signer.audience,
            "customer": customer,
            "subject": "email",
            "jwks_file": f"{name}.jwks.json",
            "algorithms": ["EdDSA"],
            "max_lifetime_seconds": 300,
            "single_use": True,
        }
    )


@dataclass
class _Verifying:
    arena: Arena
    disk: FileWorlds
    world: World
    proxy: MultiRunProxy
    kc: _Signer
    jr: _Signer

    def mint(self, signer: _Signer | None = None, **changes: Any) -> str:
        """A good assertion from a signer (Kettle Creek's by default), as a host mints it,
        or one with its claims changed and signed by the same key."""
        signer = signer or self.kc
        if not changes:
            return mint_assertion(
                signer.key,
                issuer=signer.issuer,
                audience=signer.audience,
                subject=NADINE,
                ttl=timedelta(minutes=2),
                kid=signer.kid,
                now=self.arena.clock.now(),
            )
        return self.sign(signer, headers=None, **changes)

    def sign(self, signer: _Signer, headers: dict[str, Any] | None = None, **changes: Any) -> str:
        now = int(self.arena.clock.now().timestamp())
        claims: dict[str, Any] = {
            "iss": signer.issuer,
            "aud": signer.audience,
            "sub": NADINE,
            "iat": now,
            "exp": now + 120,
            "jti": secrets.token_urlsafe(18),
        }
        claims.update(changes)
        sent = {"kid": signer.kid} | (headers or {})
        return jwt.encode(claims, signer.key, algorithm="EdDSA", headers=sent)

    def attempt(self, token: str) -> str:
        """The run start's outcome: `started <user>`, or the refusal's reason code."""
        try:
            started = self.proxy.start_verified_run(token)
        except IdentityError as refused:
            return refused.reason
        return f"started {started.run.customer_id}/{started.run.user_id}"

    def started(self) -> list[str]:
        return [str(e.run_id) for e in self.disk.events() if e.kind is EventKind.RUN_STARTED]

    def control(self) -> None:
        """A good assertion starts a run: the server is verifying, not refusing everything."""
        got = self.attempt(self.mint())
        self.arena.expect(
            "control: a good persona assertion starts a run",
            got == f"started {KC}/kc-staff-nadine",
            got,
        )

    def refused(self, token: str, reason: str, name: str) -> None:
        """The attack is refused with `reason`, recorded, and starts nothing."""
        before = len(self.started())
        denials = len(self._denials(reason))
        got = self.attempt(token)
        self.arena.expect(f"{name} is refused as {reason}", got == reason, got)
        self.arena.expect(f"{name} starts no run", len(self.started()) == before)
        self.arena.expect(
            f"{name}: the refusal is recorded", len(self._denials(reason)) == denials + 1
        )

    def _denials(self, reason: str) -> list[AuditEvent]:
        return [e for e in self.disk.events() if e.kind is EventKind.DENY and e.reason == reason]


@contextmanager
def _verifying(arena: Arena, *, kc_algorithms: tuple[str, ...] = ()) -> Iterator[_Verifying]:
    """A multi-run proxy on a durable file world that starts runs only from assertions of
    the two persona signers. Its upstream is never contacted.

    `kc_algorithms` replaces the Kettle Creek issuer's algorithm list past validation, as a
    configuration built by hand would."""
    kc = _Signer(KC_PERSONAS, KC_AUDIENCE)
    jr = _Signer(JR_PERSONAS, JR_AUDIENCE)
    kc_issuer = _issuer("kc-personas", kc, KC)
    if kc_algorithms:
        kc_issuer = kc_issuer.model_copy(update={"algorithms": kc_algorithms})
    trust = TrustConfig(
        (kc_issuer, _issuer("jr-personas", jr, JR)),
        {"kc-personas": {kc.kid: kc.jwk}, "jr-personas": {jr.kid: jr.jwk}},
    )
    with FileWorlds(arena) as disk:
        world = disk.open(durable=True)
        rates = disk.path("rates.json")
        rates.write_text(json.dumps(RATES))
        proxy = MultiRunProxy(
            world,
            ServerConfig(
                upstreams={PROVIDER: "http://127.0.0.1:9"},
                keys=ProviderKeys({PROVIDER: "ds-fake-redteam-identity-key-0123456789"}),
                rate_table=rates,
                sweep_interval=timedelta(hours=1),
                identity=trust,
            ),
        )
        proxy.start()
        try:
            yield _Verifying(arena, disk, world, proxy, kc, jr)
        finally:
            proxy.stop()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _rehead(token: str, header: dict[str, Any]) -> str:
    _, payload, signature = token.split(".")
    return f"{_b64(json.dumps(header).encode())}.{payload}.{signature}"


@scenario("id-017", ID, "An identity assertion with alg none is refused")
def assertion_alg_none(arena: Arena) -> None:
    """The header says `none` and the signature is empty (and, once, kept): no key is
    consulted and nothing starts."""
    with _verifying(arena) as v:
        v.control()
        good = v.mint()
        payload = good.split(".")[1]
        v.refused(_rehead(good, {"alg": "none", "kid": v.kc.kid}), "algorithm_refused", "alg none")
        v.refused(
            _rehead(v.mint(), {"alg": "None", "kid": v.kc.kid}), "algorithm_refused", "alg None"
        )
        # An empty signature segment is not even a compact JWS.
        unsigned = f"{_b64(json.dumps({'alg': 'none', 'kid': v.kc.kid}).encode())}.{payload}."
        v.refused(unsigned, "assertion_malformed", "an unsigned token")


@scenario("id-018", ID, "HS256 signed with the issuer's public key (algorithm confusion)")
def algorithm_confusion(arena: Arena) -> None:
    """The forger HMACs the assertion with the persona signer's public key, which anyone can
    read. Kettle Creek's issuer here even lists HS256 (a configuration built by hand, past
    validation): the verifier refuses HMAC whatever the configuration says."""
    with _verifying(arena, kc_algorithms=("EdDSA", "HS256")) as v:
        v.control()
        claims = jwt.decode(v.mint(), options={"verify_signature": False})
        claims["sub"] = "dale@kettlecreek.example"
        for public in (
            v.kc.key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            ),
            json.dumps(v.kc.jwk, sort_keys=True).encode(),
        ):
            header = {"alg": "HS256", "typ": "JWT", "kid": v.kc.kid}
            signing_input = (
                f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(claims).encode())}"
            )
            mac = hmac.new(public, signing_input.encode(), hashlib.sha256).digest()
            v.refused(f"{signing_input}.{_b64(mac)}", "algorithm_refused", "HS256 forgery")


@scenario("id-019", ID, "An assertion whose kid names another issuer's key")
def kid_of_another_issuer(arena: Arena) -> None:
    """Juniper Row's persona signer (its key, its kid) writes Kettle Creek's issuer and
    audience and vouches for a Kettle Creek user. Keys belong to one issuer: the kid is
    unknown to Kettle Creek's."""
    with _verifying(arena) as v:
        v.control()
        forged = v.sign(v.jr, iss=KC_PERSONAS, aud=KC_AUDIENCE, sub=COLBY)
        v.refused(forged, "key_unknown", "another issuer's key")


class _JwksHost:
    """An attacker's web server publishing their key, counting what it is asked."""

    def __init__(self, jwk: dict[str, Any]) -> None:
        body = json.dumps({"keys": [jwk]}).encode()
        hits: list[str] = []
        self.hits = hits

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                hits.append(self.path)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@scenario("id-020", ID, "A jku or x5u header pointing at the forger's own keys")
def key_url_headers(arena: Arena) -> None:
    """The forger signs with their own key and points `jku` and `x5u` at a server
    publishing it. Keys come only from the configuration: the headers are ignored, the
    forger's kid is unknown, and their server is never asked."""
    forger = _Signer(KC_PERSONAS, KC_AUDIENCE)
    host = _JwksHost(forger.jwk)
    try:
        with _verifying(arena) as v:
            v.control()
            for headers in (
                {"jku": f"{host.url}/jwks.json"},
                {"x5u": f"{host.url}/cert.pem"},
                {"jku": f"{host.url}/jwks.json", "x5u": f"{host.url}/cert.pem"},
            ):
                forged = v.sign(forger, headers=headers, sub="dale@kettlecreek.example")
                v.refused(forged, "key_unknown", f"a token with {sorted(headers)}")
        arena.expect("the forger's server is never asked", host.hits == [], f"{host.hits}")
    finally:
        host.close()


@scenario("id-021", ID, "A valid assertion for another audience")
def another_audience(arena: Arena) -> None:
    """Kettle Creek's signer made a good assertion for another service (another agentgate
    deployment's audience). It is genuine, and still not for this one."""
    with _verifying(arena) as v:
        v.control()
        v.refused(v.sign(v.kc, aud="agentgate:elsewhere"), "audience_mismatch", "another aud")
        v.refused(
            v.sign(v.kc, aud=["agentgate:elsewhere", "reporting"]),
            "audience_mismatch",
            "a list without this audience",
        )


@scenario("id-022", ID, "An expired assertion replayed after its use was pruned")
def expired_replay(arena: Arena) -> None:
    """A single-use assertion starts a run. Ten minutes later its used-assertion row has
    been pruned, so only its expiry stands between a replay and a second run."""
    with _verifying(arena) as v:
        v.control()
        token = v.mint()
        got = v.attempt(token)
        arena.expect("the assertion starts its run once", got.startswith("started"), got)
        v.refused(token, "replayed", "an immediate replay")
        arena.clock.advance(timedelta(minutes=10))
        # Another good start prunes the first assertion's row.
        fresh = v.attempt(v.mint())
        arena.expect("a fresh assertion still starts a run", fresh.startswith("started"), fresh)
        rows = v.world.store.connection.execute("SELECT COUNT(*) FROM used_assertions").fetchone()[
            0
        ]
        arena.expect("the expired assertion's row was pruned", rows == 1, f"{rows} rows")
        v.refused(token, "expired", "the replay after pruning")


@scenario("id-023", ID, "A persona assertion naming a suspended user")
def suspended_persona(arena: Arena) -> None:
    """A genuine assertion for a user an operator has suspended. Verification passes; the
    gate's run start refuses the user, and the refusal records the verifying issuer."""
    with _verifying(arena) as v:
        v.control()
        v.world.store.suspend_user("kc-staff-colby")
        v.refused(v.mint(v.kc, sub=COLBY), "suspended_user", "a suspended user's assertion")
        refusal = [
            e for e in v.disk.events() if e.kind is EventKind.DENY and e.reason == "suspended_user"
        ]
        arena.expect(
            "the refusal names the issuer",
            bool(refusal) and refusal[-1].details.get("assurance") == "verified:kc-personas",
        )


@scenario("id-024", ID, "An identity assertion over 8 KiB")
def oversized_assertion(arena: Arena) -> None:
    """A genuinely signed assertion padded past 8 KiB is refused before it is parsed."""
    with _verifying(arena) as v:
        v.control()
        big = v.sign(v.kc, padding="x" * 8192)
        arena.expect("the token is over 8 KiB", len(big.encode()) > 8 * 1024)
        v.refused(big, "assertion_malformed", "an 8 KiB+ assertion")


@scenario("id-025", ID, "An issuer URL that differs only by a trailing slash")
def trailing_slash_issuer(arena: Arena) -> None:
    """Kettle Creek's own key signs an assertion whose `iss` has a trailing slash. Issuers
    are matched exactly, so it names no configured issuer."""
    with _verifying(arena) as v:
        v.control()
        v.refused(v.sign(v.kc, iss=KC_PERSONAS + "/"), "issuer_unknown", "iss with a slash")
        v.refused(v.sign(v.kc, iss=KC_PERSONAS.upper()), "issuer_unknown", "iss in capitals")
