"""`agentgate identity`: a persona signing key for a host, and a check of a trust
configuration before a server relies on it."""

from pathlib import Path
from typing import Annotated

import typer

from agentgate.cli._shared import fail
from agentgate.store import make_private_dirs

identity_app = typer.Typer(
    help="Verified identity at run start: signing keys and trust configurations.",
    no_args_is_help=True,
)


@identity_app.command("keygen")
def keygen(
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for the two files (created 0700 if missing)."),
    ],
) -> None:
    """Write a new Ed25519 persona signing key and its public JWKS.

    The private key (signing-key.pem, mode 0600) stays with the host that signs persona
    assertions (agentgate.identity.mint_assertion). The JWKS (jwks.json) goes under a
    `signed` issuer's jwks_file in the trust configuration. Neither file is overwritten."""
    from agentgate.identity import write_keypair

    try:
        make_private_dirs(out)
        private_path, jwks_path, kid = write_keypair(out)
    except FileExistsError as exc:
        fail(str(exc))
    except OSError as exc:
        fail(f"cannot write the key pair in {out}: {exc.strerror}")
    typer.echo(f"private key {private_path} (keep it with the signing host)")
    typer.echo(f"public JWKS {jwks_path}")
    typer.echo(f"kid {kid}")


@identity_app.command("check-config")
def check_config(
    path: Annotated[Path, typer.Argument(help="Trust configuration (YAML).")],
) -> None:
    """Load a trust configuration, fetch every OIDC issuer's discovery document and keys, and
    print each issuer's state. Exits 1 if the configuration is invalid or an issuer's keys
    cannot be loaded."""
    from agentgate.identity import IdentityConfigError, IssuerKind, load_identity_config
    from agentgate.identity.oidc import OidcKeys, fetch_client

    try:
        trust = load_identity_config(path)
    except IdentityConfigError as exc:
        fail(str(exc))
    unavailable = 0
    with fetch_client() as http:
        for issuer in trust.issuers:
            head = (
                f"{issuer.name}  {issuer.kind.value}  customer {issuer.customer}"
                f"  subject {issuer.subject.value}  {issuer.issuer}"
            )
            if issuer.kind is IssuerKind.SIGNED:
                count = len(trust.signed_keys[issuer.name])
                typer.echo(f"{head}  {count} keys")
                continue
            keys = OidcKeys(issuer, http)
            if keys.load():
                typer.echo(f"{head}  {keys.key_count()} keys")
            else:
                unavailable += 1
                typer.echo(f"{head}  unavailable: {keys.problem}")
    if unavailable:
        raise typer.Exit(code=1)
