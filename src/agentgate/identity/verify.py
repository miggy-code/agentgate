"""Verifying an identity assertion: who agentgate may start a run for, on its own evidence.

`AssertionVerifier.verify(token)` returns a VerifiedSubject (the issuer's customer, the user,
and the run's assurance) or raises AssertionRefused with one reason code. The checks run in
this order, and the first that fails decides the code:

1. Size and shape: at most 8 KiB, three base64url segments, a JSON object header and payload
   (no repeated member), no `crit` or `b64` header. Otherwise `assertion_malformed`.
2. Issuer: the unverified `iss` selects the issuer configuration (`issuer_unknown`).
3. Header: `alg` is one of the issuer's algorithms, and never `none` or HMAC whatever the
   configuration says (`algorithm_refused`). `kid` names a key of this issuer usable with
   that algorithm (`key_unknown`; `issuer_unavailable` when an OIDC issuer's keys are not
   loaded). A `jku`, `x5u` or `jwk` header is ignored: keys come only from the configuration.
4. Signature: PyJWT verifies it with that key and the one algorithm (`signature_invalid`).
5. The verified `iss` equals the issuer's exactly (`issuer_unknown`).
6. Audience: `aud` (a string or a list) contains the issuer's audience (`audience_mismatch`).
7. Times: `exp` and `iat` are required numbers (`assertion_malformed`), with 30 s leeway:
   `expired`, `not_yet_valid` (also for `nbf`), and `lifetime_too_long` when `exp - iat`
   exceeds the issuer's maximum.
8. Replay: for a single-use issuer, `jti` is required and recorded as used; a second use is
   `replayed`.
9. Subject: the subject claim names exactly one user of the issuer's customer, and only of
   that customer (`subject_unknown`). An OIDC email needs `email_verified: true`
   (`email_unverified`).

The token itself is never kept or recorded: a refusal carries the issuer's name and a keyed
reference to the `jti` (from the caller's `reference`, the gate's audit reference), no more.
"""

import base64
import binascii
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
import jwt

from agentgate._json import no_duplicate_keys
from agentgate.clock import Clock
from agentgate.context import RunAssurance
from agentgate.domain import User
from agentgate.identity.config import (
    ACCEPTED_ALGORITHMS,
    IssuerConfig,
    IssuerKind,
    Jwk,
    SubjectField,
    TrustConfig,
)
from agentgate.identity.oidc import IssuerUnavailable, OidcKeys, fetch_client
from agentgate.store import Store

MAX_ASSERTION_BYTES = 8 * 1024
LEEWAY_SECONDS = 30
_MAX_JTI = 256
_MAX_SUBJECT = 254
_SEGMENT = re.compile(r"[A-Za-z0-9_-]+")
# Headers that change how a JWS is read; agentgate understands neither.
_REFUSED_HEADERS = frozenset({"crit", "b64"})


class AssertionReason(StrEnum):
    MALFORMED = "assertion_malformed"
    ISSUER_UNKNOWN = "issuer_unknown"
    ISSUER_UNAVAILABLE = "issuer_unavailable"
    ALGORITHM_REFUSED = "algorithm_refused"
    KEY_UNKNOWN = "key_unknown"
    SIGNATURE_INVALID = "signature_invalid"
    AUDIENCE_MISMATCH = "audience_mismatch"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    LIFETIME_TOO_LONG = "lifetime_too_long"
    REPLAYED = "replayed"
    EMAIL_UNVERIFIED = "email_unverified"
    SUBJECT_UNKNOWN = "subject_unknown"


class AssertionRefused(Exception):
    """An assertion failed verification. Carries no part of the assertion."""

    def __init__(
        self,
        reason: AssertionReason,
        *,
        issuer: IssuerConfig | None = None,
        jti_ref: str | None = None,
    ) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.issuer = issuer
        self.jti_ref = jti_ref

    @property
    def customer_id(self) -> str | None:
        """The customer of the issuer the assertion named, once it is known."""
        return None if self.issuer is None else self.issuer.customer

    def details(self) -> dict[str, str]:
        """What a refusal's audit event records: identifiers and references only."""
        found = {"verification": "assertion"}
        if self.issuer is not None:
            found["issuer"] = self.issuer.name
        if self.jti_ref is not None:
            found["jti_ref"] = self.jti_ref
        return found


@dataclass(frozen=True)
class VerifiedSubject:
    customer_id: str
    user_id: str
    assurance: RunAssurance


class AssertionVerifier:
    """Verifies assertions against a trust configuration. Thread-safe."""

    def __init__(
        self,
        config: TrustConfig,
        store: Store,
        clock: Clock,
        *,
        reference: Callable[[str], str],
        http: httpx.Client | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._clock = clock
        self._reference = reference
        self._issuers: dict[str, IssuerConfig] = {i.issuer: i for i in config.issuers}
        self._signed: Mapping[str, Mapping[str, Jwk]] = config.signed_keys
        self._http = http
        self._oidc: dict[str, OidcKeys] = {}
        oidc = [i for i in config.issuers if i.kind is IssuerKind.OIDC]
        if oidc and self._http is None:
            self._http = fetch_client()
        for issuer in oidc:
            assert self._http is not None
            self._oidc[issuer.name] = OidcKeys(issuer, self._http, monotonic=monotonic)

    @property
    def issuers(self) -> tuple[IssuerConfig, ...]:
        return tuple(self._issuers.values())

    def oidc_keys(self, name: str) -> OidcKeys | None:
        return self._oidc.get(name)

    def load(self) -> dict[str, str | None]:
        """Fetch every OIDC issuer's keys now (at startup). Returns, per OIDC issuer name,
        None if its keys loaded, else why not. Never raises: an issuer that cannot be reached
        is refused until its keys load."""
        return {name: None if keys.load() else keys.problem for name, keys in self._oidc.items()}

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def verify(self, token: object) -> VerifiedSubject:
        header, claims = _shape(token)
        assert isinstance(token, str)
        named = claims.get("iss")
        issuer = self._issuers.get(named) if isinstance(named, str) else None
        if issuer is None:
            raise AssertionRefused(AssertionReason.ISSUER_UNKNOWN)

        def refuse(reason: AssertionReason, jti_ref: str | None = None) -> AssertionRefused:
            return AssertionRefused(reason, issuer=issuer, jti_ref=jti_ref)

        algorithm = header.get("alg")
        if not _acceptable(algorithm) or algorithm not in issuer.algorithms:
            raise refuse(AssertionReason.ALGORITHM_REFUSED)
        assert isinstance(algorithm, str)
        key = self._key(issuer, header.get("kid"), algorithm)

        try:
            verified = jwt.PyJWS().decode_complete(token, key=key, algorithms=[algorithm])
        except jwt.InvalidSignatureError:
            raise refuse(AssertionReason.SIGNATURE_INVALID) from None
        except (jwt.PyJWTError, ValueError, TypeError):
            # Anything else PyJWT cannot read is a malformed assertion, never a traceback.
            raise refuse(AssertionReason.MALFORMED) from None
        signed = _object(verified["payload"])
        if signed is None:
            raise refuse(AssertionReason.MALFORMED)
        # From here only what the signature covers is read.
        claims = signed

        if claims.get("iss") != issuer.issuer:
            raise refuse(AssertionReason.ISSUER_UNKNOWN)
        if not _audience_contains(claims.get("aud"), issuer.audience):
            raise refuse(AssertionReason.AUDIENCE_MISMATCH)
        expires = self._check_times(issuer, claims, refuse)

        jti = claims.get("jti")
        jti_ref = None
        if isinstance(jti, str) and 0 < len(jti) <= _MAX_JTI:
            jti_ref = self._reference(jti)
        if issuer.single_use:
            if jti_ref is None:
                raise refuse(AssertionReason.MALFORMED)
            assert isinstance(jti, str)
            used = self._store.use_assertion(
                issuer.name,
                hashlib.sha256(jti.encode()).hexdigest(),
                math.ceil(expires),
                prune_before=math.floor(self._now() - LEEWAY_SECONDS),
            )
            if not used:
                raise refuse(AssertionReason.REPLAYED, jti_ref)

        user = self._subject(issuer, claims, lambda reason: refuse(reason, jti_ref))
        # The run is always in the issuer's own customer: an issuer is bound to one.
        return VerifiedSubject(issuer.customer, user.id, RunAssurance(issuer.name, jti_ref))

    def _key(self, issuer: IssuerConfig, kid: object, algorithm: str) -> jwt.PyJWK:
        def unknown() -> AssertionRefused:
            return AssertionRefused(AssertionReason.KEY_UNKNOWN, issuer=issuer)

        if not isinstance(kid, str) or not 0 < len(kid) <= 256:
            raise unknown()
        if issuer.kind is IssuerKind.SIGNED:
            jwk = self._signed.get(issuer.name, {}).get(kid)
        else:
            try:
                jwk = self._oidc[issuer.name].key(kid)
            except IssuerUnavailable:
                raise AssertionRefused(AssertionReason.ISSUER_UNAVAILABLE, issuer=issuer) from None
        if jwk is None or jwk.get("alg", algorithm) != algorithm:
            raise unknown()
        try:
            # The key is read for the header's algorithm alone: a key of another type (an
            # RSA key for EdDSA) is no key for this assertion.
            return jwt.PyJWK(dict(jwk), algorithm=algorithm)
        except (jwt.PyJWTError, ValueError, TypeError, NotImplementedError):
            raise unknown() from None

    def _check_times(
        self,
        issuer: IssuerConfig,
        claims: dict[str, Any],
        refuse: Callable[[AssertionReason], AssertionRefused],
    ) -> float:
        """The assertion's expiry, once its times pass."""
        expires, issued = claims.get("exp"), claims.get("iat")
        not_before = claims.get("nbf", issued)
        if not (_number(expires) and _number(issued) and _number(not_before)):
            raise refuse(AssertionReason.MALFORMED)
        assert isinstance(expires, int | float) and isinstance(issued, int | float)
        assert isinstance(not_before, int | float)
        now = self._now()
        if expires <= now - LEEWAY_SECONDS:
            raise refuse(AssertionReason.EXPIRED)
        if issued > now + LEEWAY_SECONDS or not_before > now + LEEWAY_SECONDS:
            raise refuse(AssertionReason.NOT_YET_VALID)
        if expires < issued:
            raise refuse(AssertionReason.MALFORMED)
        if expires - issued > issuer.max_lifetime_seconds:
            raise refuse(AssertionReason.LIFETIME_TOO_LONG)
        return float(expires)

    def _subject(
        self,
        issuer: IssuerConfig,
        claims: dict[str, Any],
        refuse: Callable[[AssertionReason], AssertionRefused],
    ) -> User:
        value = claims.get(issuer.subject_claim)
        if not isinstance(value, str) or not 0 < len(value) <= _MAX_SUBJECT:
            raise refuse(AssertionReason.SUBJECT_UNKNOWN)
        # Exactly true: a provider that has not verified the address has not said whose
        # mailbox it is.
        if issuer.subject_claim == "email" and claims.get("email_verified") is not True:
            raise refuse(AssertionReason.EMAIL_UNVERIFIED)
        # Only the issuer's own customer is searched, so an issuer can never name a user of
        # another customer, whatever it signs.
        if issuer.subject is SubjectField.EMAIL:
            matches = self._store.users_by_email(issuer.customer, value)
            if len(matches) != 1:
                raise refuse(AssertionReason.SUBJECT_UNKNOWN)
            return matches[0]
        user = self._store.get_user(value)
        if user is None or user.customer_id != issuer.customer:
            raise refuse(AssertionReason.SUBJECT_UNKNOWN)
        return user

    def _now(self) -> float:
        return self._clock.now().timestamp()


def _acceptable(algorithm: object) -> bool:
    """Whether an algorithm can verify an identity at all. `none` and HMAC never can: a
    verifier that holds an HMAC secret could mint as well, and a public key used as one is
    the classic algorithm-confusion forgery."""
    if not isinstance(algorithm, str):
        return False
    if algorithm.lower() == "none" or algorithm.upper().startswith("HS"):
        return False
    return algorithm in ACCEPTED_ALGORITHMS


def _shape(token: object) -> tuple[dict[str, Any], dict[str, Any]]:
    """The unverified header and payload of a compact JWS, or AssertionRefused(malformed)."""
    malformed = AssertionRefused(AssertionReason.MALFORMED)
    if not isinstance(token, str) or len(token.encode("utf-8")) > MAX_ASSERTION_BYTES:
        raise malformed
    parts = token.split(".")
    if len(parts) != 3 or not all(_SEGMENT.fullmatch(part) for part in parts):
        raise malformed
    header = _object(_b64(parts[0]))
    claims = _object(_b64(parts[1]))
    if header is None or claims is None or _REFUSED_HEADERS & set(header):
        raise malformed
    return header, claims


def _b64(segment: str) -> bytes | None:
    try:
        return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
    except (binascii.Error, ValueError):
        return None


def _object(raw: bytes | None) -> dict[str, Any] | None:
    """A JSON object read strictly (UTF-8, no repeated member), or None."""
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _audience_contains(audience: object, wanted: str) -> bool:
    if isinstance(audience, str):
        return audience == wanted
    return isinstance(audience, list) and any(
        isinstance(item, str) and item == wanted for item in audience
    )
