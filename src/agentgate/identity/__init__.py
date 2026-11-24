"""Verified identity at run start: agentgate checks a signed assertion itself before it
starts a run, instead of trusting a user id its host names.

Two kinds of issuer are trusted, each bound to one customer (`config`): a host's own persona
signer, whose public keys are in a local JWKS file, and any OpenID Connect provider, whose
keys come from discovery (`oidc`). `verify` checks an assertion and names the user;
`mint` holds the host's side (a signing key and the assertions it signs).
"""

from agentgate.identity.config import (
    IdentityConfigError,
    IssuerConfig,
    IssuerKind,
    SubjectField,
    TrustConfig,
    load_identity_config,
)
from agentgate.identity.mint import (
    generate_signing_key,
    load_signing_key,
    mint_assertion,
    public_jwk,
    write_keypair,
)
from agentgate.identity.verify import (
    AssertionReason,
    AssertionRefused,
    AssertionVerifier,
    VerifiedSubject,
)

__all__ = [
    "AssertionReason",
    "AssertionRefused",
    "AssertionVerifier",
    "IdentityConfigError",
    "IssuerConfig",
    "IssuerKind",
    "SubjectField",
    "TrustConfig",
    "VerifiedSubject",
    "generate_signing_key",
    "load_identity_config",
    "load_signing_key",
    "mint_assertion",
    "public_jwk",
    "write_keypair",
]
