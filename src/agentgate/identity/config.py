"""The trust configuration: which issuers agentgate accepts identity assertions from, and for
which customer each one may start runs (`--identity-config`).

```yaml
issuers:
  - name: casa-prado-personas      # recorded on the run
    kind: signed                    # keys from a local JWKS file
    issuer: https://brain.local/personas
    audience: agentgate:casa-prado
    customer: casa-prado            # the only customer this issuer can start runs in
    subject: email                  # `sub` holds the user's email (or: user_id)
    jwks_file: personas.jwks.json   # relative to this file
    algorithms: [EdDSA]
    max_lifetime_seconds: 300
    single_use: true
  - name: casa-prado-keycloak
    kind: oidc                      # keys from discovery + jwks_uri
    issuer: http://127.0.0.1:8080/realms/casa-prado
    audience: agentgate
    customer: casa-prado
    subject: email                  # the `email` claim, with email_verified: true
    algorithms: [RS256, ES256]
    max_lifetime_seconds: 3600
    single_use: false
```

Where the person is in an assertion: a `signed` issuer (a host's own persona signer) puts
them in `sub`, as an email or a user id per `subject`. An `oidc` issuer with `subject: email`
is read from its `email` claim, which must come with `email_verified: true`; with `subject:
user_id`, from `sub`.

Everything is validated strictly: unknown keys, a symmetric or `none` algorithm, a plain-http
OIDC issuer off the loopback host, and two issuers with one name or one URL (a trailing slash
apart included) are refused at load. A `signed` issuer's JWKS file is read at load and holds
public keys only.
"""

import ipaddress
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentgate._json import no_duplicate_keys
from agentgate._yaml import parse_yaml

# Asymmetric JWS algorithms an issuer may list. `none` and every HMAC algorithm are refused
# whatever a configuration says: a verifier holding an HMAC secret could also mint.
ACCEPTED_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
        "EdDSA",
    }
)
# The longest assertion lifetime any issuer may allow: a day.
MAX_LIFETIME_LIMIT = 86_400
# JWK members that hold private or symmetric key material.
_SECRET_MEMBERS = frozenset({"d", "p", "q", "dp", "dq", "qi", "oth", "k"})

IssuerName = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")]
Text = Annotated[str, Field(min_length=1, max_length=512)]


class IdentityConfigError(ValueError):
    """The trust configuration or a key file is not valid."""


class IssuerKind(StrEnum):
    # A host's own signer: public keys from a local JWKS file.
    SIGNED = "signed"
    # An OpenID Connect provider: keys from its discovery document's jwks_uri.
    OIDC = "oidc"


class SubjectField(StrEnum):
    """Which user field the assertion's subject is matched to."""

    EMAIL = "email"
    USER_ID = "user_id"


class IssuerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: IssuerName
    kind: IssuerKind
    issuer: Text
    audience: Text
    customer: Annotated[str, Field(min_length=1, max_length=200)]
    subject: SubjectField
    jwks_file: Path | None = None
    algorithms: Annotated[tuple[str, ...], Field(min_length=1)]
    max_lifetime_seconds: Annotated[int, Field(ge=1, le=MAX_LIFETIME_LIMIT)]
    single_use: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        for algorithm in self.algorithms:
            if algorithm not in ACCEPTED_ALGORITHMS:
                raise ValueError(
                    f"algorithm {algorithm!r} is not accepted"
                    f" (one of {', '.join(sorted(ACCEPTED_ALGORITHMS))})"
                )
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("an algorithm is listed twice")
        if self.kind is IssuerKind.SIGNED:
            if self.jwks_file is None:
                raise ValueError("a signed issuer needs jwks_file")
        else:
            if self.jwks_file is not None:
                raise ValueError("an oidc issuer takes its keys from discovery, not jwks_file")
            check_fetch_url(self.issuer, "issuer")
        return self

    @property
    def subject_claim(self) -> str:
        """The claim that names the person."""
        if self.kind is IssuerKind.OIDC and self.subject is SubjectField.EMAIL:
            return "email"
        return "sub"


class IdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issuers: Annotated[tuple[IssuerConfig, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        names: set[str] = set()
        urls: set[str] = set()
        for issuer in self.issuers:
            if issuer.name in names:
                raise ValueError(f"issuer name {issuer.name!r} is listed twice")
            names.add(issuer.name)
            # Two issuers a trailing slash apart would read as one to a careless eye; the
            # verifier itself matches `iss` exactly.
            url = issuer.issuer.rstrip("/")
            if url in urls:
                raise ValueError(f"issuer {issuer.issuer!r} is listed twice")
            urls.add(url)
        return self


# A public JWK as published: kty, kid and the key's own members.
Jwk = Mapping[str, Any]


@dataclass(frozen=True)
class TrustConfig:
    """A validated configuration with the public keys of its signed issuers."""

    issuers: tuple[IssuerConfig, ...]
    # Per signed issuer name: its public signing keys by kid.
    signed_keys: Mapping[str, Mapping[str, Jwk]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for issuer in self.issuers:
            if issuer.kind is IssuerKind.SIGNED and issuer.name not in self.signed_keys:
                raise IdentityConfigError(f"issuer {issuer.name!r} has no keys loaded")


def check_fetch_url(url: str, what: str) -> None:
    """Refuse a URL agentgate would fetch keys from unless it is https, or http on a loopback
    host. Raises ValueError."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        raise ValueError(f"{what} is not a URL") from None
    if host is None or parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError(f"{what} must be a plain https URL")
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and _loopback(host):
        return
    raise ValueError(f"{what} must use https (plain http only on a loopback host)")


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def load_identity_config(path: Path) -> TrustConfig:
    """Read and validate a trust configuration, and the public keys of its signed issuers.
    Raises IdentityConfigError naming the file, never a key."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IdentityConfigError(f"{path}: cannot read: {exc}") from None
    raw = parse_yaml(text, path, IdentityConfigError)
    try:
        config = IdentityConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in error['loc']) or 'config'}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)
        )
        raise IdentityConfigError(f"{path}: {problems}") from None
    keys: dict[str, Mapping[str, Jwk]] = {}
    for issuer in config.issuers:
        if issuer.jwks_file is not None:
            jwks_path = issuer.jwks_file
            if not jwks_path.is_absolute():
                jwks_path = path.parent / jwks_path
            keys[issuer.name] = load_jwks_file(jwks_path)
    return TrustConfig(config.issuers, keys)


def load_jwks_file(path: Path) -> dict[str, dict[str, Any]]:
    """A local JWKS file's public signing keys by kid. Refuses private or symmetric key
    material, a key without a kid, and a kid listed twice."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) else "not a JSON object"
        raise IdentityConfigError(f"{path}: cannot read the JWKS: {reason}") from None
    try:
        return signing_keys(data, strict=True)
    except ValueError as exc:
        raise IdentityConfigError(f"{path}: {exc}") from None


def signing_keys(data: object, *, strict: bool) -> dict[str, dict[str, Any]]:
    """The public signing keys of a JWKS document, by kid.

    A key for encryption (`use: enc`), a symmetric key and a key with private members are
    never signing keys. With strict (a local file the operator wrote) any of them, or a key
    with no kid, is an error; without (a provider's published set) they are skipped."""
    if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
        raise ValueError("a JWKS is an object with a `keys` list")
    found: dict[str, dict[str, Any]] = {}
    for entry in data["keys"]:
        problem = _key_problem(entry)
        if problem is not None:
            if strict:
                raise ValueError(problem)
            continue
        assert isinstance(entry, dict)
        kid = entry["kid"]
        if kid in found:
            if strict:
                raise ValueError(f"kid {kid!r} is listed twice")
            # A provider listing one kid twice: neither key can be told apart, so neither
            # is used.
            found[kid] = {}
            continue
        found[kid] = dict(entry)
    return {kid: key for kid, key in found.items() if key}


def _key_problem(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return "a key is not an object"
    kid = entry.get("kid")
    if not isinstance(kid, str) or not 0 < len(kid) <= 256:
        return "a key has no kid"
    if entry.get("kty") not in {"RSA", "EC", "OKP"}:
        return f"key {kid!r}: only RSA, EC and OKP public keys can verify an assertion"
    if _SECRET_MEMBERS & set(entry):
        return f"key {kid!r} holds private key material: a JWKS file holds public keys only"
    if entry.get("use", "sig") != "sig":
        return f"key {kid!r} is not a signing key"
    return None
