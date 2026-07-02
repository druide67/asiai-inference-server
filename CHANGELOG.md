# Changelog

All notable changes to asiai-inference-server (the `aisctl` CLI and the
`ais_core` library) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0](https://github.com/druide67/asiai-inference-server/compare/v0.4.0...v0.5.0) — 2026-07-02

Fleet groundwork + hardening on top of the privileged-helper model. This is
the release whose packaged helper matches the security-reviewed sources —
pin `>=0.5.0` when bootstrapping a new machine.

### Added

- The loopback server (`aisctl serve`) and the fleet client (`aisctl fleet
  push`) now consume the shared `asiai.fleet.command_spec` (single source of
  truth for the write-command whitelist and timeouts). The nesting invariant
  `client > edge > loopback` holds by construction; the wrapped `upgrade`
  passes an inner tool deadline so the operator sees the tool's real error,
  not a timeout kill. Requires `asiai>=1.15.0`.

### Fixed

- **Firewall:** `aisctl reinstall`/`uninstall` no longer tears the daemon
  down before discovering it cannot touch pf — a sudo/TTY preflight runs
  first, and anchor installation is idempotent (no-op when already current).
- **Bootstrap:** the helper's log directory is created root:wheel `0755` and
  the audit log is pre-created `0640 root:admin`, so refusal entries are
  operator-readable without sudo.
- **Helper:** a missing log directory now yields a clear refusal pointing at
  `aisctl bootstrap` instead of an opaque internal error; the audit log is
  created group-readable.
- **Lifecycle:** user-supplied model/template paths are resolved
  (`realpath`) before reaching the helper, so a symlinked model file (e.g.
  an `active.gguf` pointer) installs again.

## [0.4.0](https://github.com/druide67/asiai-inference-server/compare/v0.3.2...v0.4.0) — 2026-06-24

Privileged-helper release (retro-added entry).

### Added

- Root-owned privileged helper (`asiai-priv`) replacing wildcard sudoers
  rules: closed action allowlist, plist content validation at load,
  refuse-by-default, append-only audit log, `aisctl bootstrap
  --install/--verify/--rollback` with a recorded pre-bootstrap sudoers
  backup for lossless rollback.
- Reserved-service installs (`asiai web` / `aisctl serve`) through the
  helper with content-validated specs.

## [0.3.2](https://github.com/druide67/asiai-inference-server/compare/v0.3.1...v0.3.2) — 2026-06-13

OSS hygiene. No functional change.

### Changed

- Test suite uses a neutral `testuser` daemon-user placeholder instead of a
  personal login.

## [0.3.1](https://github.com/druide67/asiai-inference-server/compare/v0.3.0...v0.3.1) — 2026-06-13

Audit follow-up (2026-06-11). Correctness and hardening; no breaking changes.

### Fixed

- **Firewall (critical):** `block-port` anchors were a no-op. The generated
  `pf.conf` referenced the anchor but was missing the `load anchor … from …`
  directive, so it was never actually loaded. Added the load directive and an
  IPv6 block rule; a failing `pfctl -f` now raises instead of being swallowed.
- **Lifecycle:** `stop` no longer disables the daemon (it had wrongly passed
  `launchctl … -w`), so a stopped engine still comes back on reboot as
  documented — `disable` remains the durable cold-standby path. `start` and
  `restart` now honour `--dry-run`.
- **Manifest / plist:** `~` resolves against the daemon user rather than the
  invoking user, and the health URL maps a `0.0.0.0`/empty bind to
  `127.0.0.1`.
- Hardened error paths and preconditions across `ais_core` (manifest parsing,
  memory repair/purge, install-record validation) and surfaced previously
  silenced failures in the driver proxy.
- `aisctl fleet` with no subcommand prints a usage error instead of crashing;
  closed `asiai engine` parity gaps.

### Changed

- The test suite is now hermetic — it never touches the developer's real
  config or state — and several weak assertions were tightened.

### Docs

- README, roadmap, presets and engine docs realigned with v0.3.0.

## [0.3.0](https://github.com/druide67/asiai-inference-server/compare/v0.2.0...v0.3.0) — 2026-06-11

### Added

- **`aisctl status --deep`:** a generation-based health probe that catches a
  process which is up but serving a stale/zombie model.
- **`aisctl disable` / `enable`:** durable cold standby that survives reboot
  (`launchctl … -w`).
- **Install records + `aisctl reinstall`:** the preset and options used at
  install time are persisted, so a reinstall can no longer silently degrade
  the configuration.

### Changed

- User presets now resolve from a single location, `engine_manifests/presets/`.

### Fixed

- Deterministic synchronization for the cross-process lock test (a CI flake).

## [0.2.0](https://github.com/druide67/asiai-inference-server/compare/v0.1.0...v0.2.0) — 2026-05-30

### Added

- **`aisctl upgrade <engine>`:** native engine upgrade through a shared Homebrew
  whitelist, run behind the OperationsLock, with `--restart` and `--dry-run`.
- **`asiai.version_sources` provider:** exposes engine and tooling versions to
  `asiai versions`.

## [0.1.0](https://github.com/druide67/asiai-inference-server/releases/tag/v0.1.0) — 2026-05-28

First public release.

### Added

- The `aisctl` lifecycle CLI and the `ais_core` building blocks
  (install/uninstall, start/stop/restart, unload, purge, an fcntl-based
  operations lock and `RegistrationContext`) for managing local inference
  engines on Apple Silicon.
- Engines: Ollama, LM Studio, llama.cpp (llama-server), mlx-lm, vMLX and
  Rapid-MLX, plus the config-driven `llamacpp-aux` family and a TOML preset
  mechanism for tuned Hermes Agent stacks.
- `aisctl serve` loopback command server and Fleet Phase 2 (`fleet push`,
  `install-service`); sudoers bootstrap; pf firewall anchors.
- `asiai.engines` entry points so `asiai` discovers the engines managed here.
- CI matrix Python 3.11/3.12/3.13 × macOS 14/15, with OIDC trusted publishing
  to PyPI.

### Security

- Pre-release audit fixes: H1 (TOCTOU on `/tmp` → a `0700` staging dir),
  H2 (process-pattern validation), M2 (lock mode `0o600`), and removal of the
  `--keep-logs` escape hatch.
