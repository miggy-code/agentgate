# Changelog

All notable changes to `panoptes-agentgate` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version can break the API.

## [0.3.0] - 2026-11-24

### Added

- Verified run start. `agentgate model-proxy server --identity-config FILE` starts a run
  only for a person agentgate verified itself from a signed identity assertion
  (`ControlClient.start_verified_run`). Without a trust config the server behaves as
  before; with one, the user-id form is refused as `assertion_required`.
  - Two issuer kinds, each bound to one customer: `signed` (a host's own Ed25519 persona
    signer, keys in a local JWKS file, the person in `sub`) and `oidc` (any OpenID Connect
    provider via discovery and a JWKS cache; with `subject: email`, `email_verified` must
    be true).
  - `none` and HMAC algorithms are always refused, keys come only from the configuration
    (`jku`, `x5u` and `jwk` headers are ignored), and single-use assertions are recorded
    by a hash of their `jti` so a replay is refused.
  - Each run records its assurance (`verified:<issuer>`); every refusal is a `deny` audit
    event with one reason code and never the assertion. An unreachable provider answers
    503 and fails closed.
- `agentgate identity keygen`, `agentgate identity check-config` and
  `agentgate.identity.mint_assertion` for hosts that sign their own assertions.
- Red-team scenarios id-017 to id-025 for forged, replayed and misdirected assertions.
- A CI job that runs the OIDC verifier against Keycloak 26.8.2.

### Changed

- New dependency: `pyjwt[crypto]`.
- The control API's request body limit is 16 KiB (was 4 KiB), so an assertion up to the
  8 KiB maximum reaches the verifier.

### Security

- An email address with a Unicode look-alike (for example KELVIN SIGN for `K`) no longer
  matches an ASCII user: only ASCII is case-folded, on both sides.
- Accepted limits of verified run start are listed in `docs/limits.md`.

## [0.2.0] - 2026-11-24

Everything since 0.0.1. Version 0.1.0 was prepared but never published, so its changes
are included here.

### Breaking

- Domain types moved from `agentgate.models` to `agentgate.domain`.
- The model proxy, the multi-run server and the control client moved from
  `agentgate.model` to `agentgate.model_proxy`. `agentgate.model` keeps admission,
  pricing, tokens and keys.
- `Gate` keeps its public interface, but its internals are now a package. Import from
  `agentgate.gate`, not from its submodules.

### Added

- `agentgate model-proxy server`: serves model calls for many runs from one process on
  one loopback port. Runs start and end over a Unix control socket, and each run has its
  own caps. Stopping the server ends its runs, and the next start ends the runs of a
  server that died. `ControlClient` wraps the control API.
- Source ACLs on documents. A document can carry the ACL of the system it came from. The
  engine denies a read with `acl_denied` unless the user holds one of the document's
  principals, so a policy can narrow what the source shares but never widen it.
- Models-only policies. `tools.yaml`, rules and approvals are optional when a policy only
  governs model calls.
- `model-proxy serve --ready-file`: once listening, writes a private JSON file with the
  URL, run and caps, and removes it on stop.
- `agentgate init-experiment`: writes a one-user fixture and a models-only policy.
- The store records each run's limits and which server started it.
- Red-team scenarios mb-026 to mb-029 for the multi-run proxy.

### Changed

- Requires `panoptes-observe` >=0.2.0,<0.4. Panoptes 0.3 prices prompt-cache writes
  separately.
- The CLI is a package with one module per command group. Commands and options are
  unchanged.
- Package layers are enforced by import-linter in `make lint`.

### Fixed

- `init-experiment` and the ready file no longer remove a file another process put in
  their place. They create files exclusively, don't follow symlinks, and keep each file
  open so its inode can't be reused before cleanup.
- A second SIGTERM no longer interrupts shutdown cleanup.
- `audit show` and `audit check-store` read stores created before source ACLs existed.
- `model-proxy server --host` with a non-loopback address prints one error line instead
  of a traceback.

## [0.0.1] - 2026-11-04

First release on PyPI.

[0.3.0]: https://github.com/miggy-code/agentgate/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/miggy-code/agentgate/compare/e5c6cd0...v0.2.0
[0.0.1]: https://github.com/miggy-code/agentgate/commit/e5c6cd0
