# Changelog

All notable changes to `panoptes-agentgate` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version can break the API.

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

[0.2.0]: https://github.com/miggy-code/agentgate/compare/e5c6cd0...v0.2.0
[0.0.1]: https://github.com/miggy-code/agentgate/commit/e5c6cd0
