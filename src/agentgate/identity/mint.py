"""Host helpers for persona assertions: a signing key and the assertions it signs.

agentgate is the verifier; a host holds the private key. `agentgate identity keygen` writes
an Ed25519 key (PEM, mode 0600) and its public JWKS, whose one key has the RFC 7638
thumbprint as its kid. The operator lists the JWKS file under a `signed` issuer in the trust
configuration, and the host signs with `mint_assertion` when a person is chosen.
"""

import base64
import hashlib
import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

PRIVATE_KEY_NAME = "signing-key.pem"
JWKS_NAME = "jwks.json"
# The members RFC 7638 hashes for each key type, in the order it sorts them.
_THUMBPRINT_MEMBERS = {
    "OKP": ("crv", "kty", "x"),
    "EC": ("crv", "kty", "x", "y"),
    "RSA": ("e", "kty", "n"),
}


def generate_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def jwk_thumbprint(jwk: dict[str, Any]) -> str:
    """The RFC 7638 SHA-256 thumbprint of a public JWK, base64url without padding."""
    members = _THUMBPRINT_MEMBERS.get(str(jwk.get("kty")))
    if members is None:
        raise ValueError("a thumbprint needs an OKP, EC or RSA key")
    canonical = json.dumps({m: jwk[m] for m in members}, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def public_jwk(private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """The key's public JWK, with its thumbprint as kid, for EdDSA signatures."""
    jwk: dict[str, Any] = json.loads(jwt.algorithms.OKPAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = jwk_thumbprint(jwk)
    jwk["alg"] = "EdDSA"
    jwk["use"] = "sig"
    return jwk


def mint_assertion(
    private_key: Ed25519PrivateKey,
    *,
    issuer: str,
    audience: str,
    subject: str,
    ttl: timedelta,
    kid: str,
    now: datetime | None = None,
) -> str:
    """A compact JWS (EdDSA) saying that `issuer` vouches for `subject` (the person's email
    or user id, as the issuer's `subject` setting says) to `audience`, for `ttl` from now,
    with a random jti so a single-use issuer accepts it once."""
    if ttl <= timedelta(0):
        raise ValueError("an assertion's lifetime must be above zero")
    issued = int((now or datetime.now(UTC)).timestamp())
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "iat": issued,
        "exp": issued + int(ttl.total_seconds()),
        "jti": secrets.token_urlsafe(18),
    }
    return jwt.encode(claims, private_key, algorithm="EdDSA", headers={"kid": kid, "typ": "JWT"})


def load_signing_key(path: Path) -> Ed25519PrivateKey:
    """An Ed25519 private key from a PEM file written by `write_keypair`."""
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path}: not an Ed25519 private key")
    return key


def write_keypair(directory: Path) -> tuple[Path, Path, str]:
    """Write a new Ed25519 private key (PEM, 0600) and its public JWKS into `directory`,
    never over an existing file. Returns the two paths and the kid."""
    key = generate_signing_key()
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    jwk = public_jwk(key)
    private_path = directory / PRIVATE_KEY_NAME
    jwks_path = directory / JWKS_NAME
    for path in (private_path, jwks_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{path} already exists")
    _write_new(private_path, pem, 0o600)
    try:
        _write_new(jwks_path, (json.dumps({"keys": [jwk]}, indent=2) + "\n").encode(), 0o644)
    except BaseException:
        private_path.unlink(missing_ok=True)
        raise
    return private_path, jwks_path, str(jwk["kid"])


def _write_new(path: Path, data: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as handle:
        # The umask may have narrowed the mode; the public JWKS is meant to be readable.
        os.fchmod(handle.fileno(), mode)
        handle.write(data)
