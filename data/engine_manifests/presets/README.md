# Engine manifest presets — bundled examples

A preset is a full, self-contained `EngineManifest` TOML tuned for a
specific model and use case. Pick one at install time:

```
aisctl install <engine> --preset <preset-name>
```

The preset replaces the engine baseline verbatim — there is no field
merge. If you copy a preset to start from, expect to edit every field
that matters to your hardware (model path hard link, port if you run
multiple instances of the same engine, KV cache type, parallel slots,
context size).

## What lives in this directory

Everything here is a **bundled example**. The presets ship with the
package so contributors and adopters can read realistic, end-to-end
tuning side-by-side with the baseline manifests, but **none of them is
required**: the engine baselines work as-is for an out-of-the-box
install.

Presets named with `hermes` in their stem are concrete tuning examples
adapted for the Hermes Agent class of orchestrator —
multi-turn agentic workloads with a system prefix that recurs across
turns and short user payloads that change every turn. They are example
references, not a product configuration: an asiai user adopting a
different orchestrator (Open WebUI, LiteLLM-fronted, a custom agent
framework, ...) will most likely write their own preset with different
tuning. Read the header comments — each preset explains *why* the
chosen flags exist for that scenario, which is the part worth reusing.

The `llamacpp-aux-N` presets demonstrate the multi-instance pattern:
five `llama-server` instances running side-by-side on dedicated ports
8090-8094 with distinct model hard links. The first four split the
general-purpose roles (multi-role, medium, small, vision); `aux-5` is
the worked example of extending the family — a dedicated long-context
single-flow slot (256K, compression workloads) added next to the
existing four. To add a sixth instance, drop `llamacpp-aux-6.toml`
next to the others (see the user-config section below for the
recommended location), then create a matching preset if you want to
ship tuning along with it.

## Writing your own presets without committing them upstream

If you want to add or override presets without modifying the package,
drop your TOMLs in the XDG user-config directory, mirroring the bundled
package tree (presets live next to the manifests they tune):

```
$XDG_CONFIG_HOME/asiai-inference-server/engine_manifests/presets/<name>.toml
```

The default base path on macOS is `~/.config/asiai-inference-server/`.
You can override the whole user-config tree with the
`ASIAI_USER_CONFIG_DIR` environment variable.

Files there are discovered automatically by `list_presets()` and
`aisctl install --preset <name>` resolves them just like bundled
presets. A user preset whose stem matches a bundled one wins the lookup
(with a warning logged at load time so the override is visible to
operators).

> **Upgrading from ≤ v0.2:** user presets used to be read from
> `presets/` at the config root. That location is no longer scanned —
> move your TOMLs to `engine_manifests/presets/`. Any file left behind
> triggers a warning naming it and the new path, so the relocation
> cannot fail silently.

The same XDG mechanism applies to engine manifests themselves: drop
`engine_manifests/<name>.toml` under the user-config directory and
`aisctl status / install / start ...` will pick it up.

## Validation

Every preset is loaded via `load_manifest(<engine>, preset=<name>)`,
which validates that `name` (the engine identifier inside the TOML)
matches the engine the operator asked for. A preset that targets
`llamacpp-aux-3` cannot be installed on `llamacpp-aux-1` — `aisctl`
rejects the mismatch before touching the system.

## Where model and template files must live (security constraint)

`model_path`, `template_path` and `mmproj_path` must reside **under the
home directory of the account the daemon runs as**, on a volume with
real POSIX permissions (APFS). The privileged helper enforces this and
refuses anything else. Two practical consequences:

- **No external/network volumes.** A GGUF on an exFAT/FAT USB disk or
  an SMB/NFS share has no enforced ownership: "readable by the daemon"
  would not imply "not modifiable by another local user", which reopens
  the very content-swap window the helper closes. Keep weights under
  `~/llms/` (or any home subdirectory) on the internal volume.
- **Use a hard link (or a real path), not a symlink.** The helper
  refuses a symlink final component by design (anti swap-TOCTOU), so
  `aisctl install`/`reinstall` resolves symlinks client-side — tilde
  paths included — and pins the plist to the real target of the moment.
  The install then succeeds, **but** the daemon's command line carries
  the real path, so a `process_pattern` written against the symlink
  (`llms/gguf/active.gguf`) no longer matches: state detection and
  `pkill` go blind (`aisctl install` warns about this). The switchable
  slot convention that keeps everything coherent is a **hard link** on
  the same APFS volume: `ln -f ~/llms/gguf/<MyModel>.gguf
  ~/llms/gguf/active.gguf`, then `aisctl restart <engine>` to switch
  models — the pinned path never changes and the pattern always
  matches. (A hard link pins the inode: re-run `ln -f` after replacing
  the source file, or the link silently keeps serving the old bytes.)

## Files in this directory

| Preset | Engine target | Notes |
|---|---|---|
| `qwen3.6-27b-dense-hermes-agent-64gb.toml` | `llamacpp` | Example primary-agent tuning, dense 27B on a 64 GB Apple Silicon host. |
| `qwen3.6-35b-a3b-hermes-agent-64gb.toml` | `llamacpp` | Example primary-agent tuning, MoE 35B-A3B on a 64 GB Apple Silicon host. |
| `qwen3.6-35b-a3b-hermes-fallback-mlx-lm.toml` | `mlx-lm` | Example fallback tuning, MoE 35B-A3B served by mlx-lm. |
| `qwen3-4b-instruct-hermes-aux-1.toml` | `llamacpp-aux-1` | Example multi-role auxiliary, dense 4B on 8090 with parallel=4. |
| `qwen3-1.7b-instruct-hermes-aux-2.toml` | `llamacpp-aux-2` | Example medium auxiliary, dense 1.7B on 8091. |
| `qwen3-0.6b-instruct-hermes-aux-3.toml` | `llamacpp-aux-3` | Example small auxiliary (e.g. title generation), dense 0.6B on 8092. |
| `qwen2.5-vl-7b-instruct-hermes-aux-4.toml` | `llamacpp-aux-4` | Example vision auxiliary with mmproj on 8093. |
| `qwen3-4b-instruct-hermes-aux-5-compression.toml` | `llamacpp-aux-5` | Example dedicated long-context slot (256K single flow, compression workloads) on 8094. |
| `qwen3.6-27b-ud-rapidmlx-hermes.toml` | `rapidmlx` | Example primary-agent tuning, dense 27B UD served by Rapid-MLX. |
| `qwen3.6-35b-a3b-rapidmlx-hermes.toml` | `rapidmlx` | Example primary-agent tuning, MoE 35B-A3B served by Rapid-MLX. |
| `qwopus-27b-v2-rapidmlx-hermes.toml` | `rapidmlx` | Example primary-agent tuning, Qwopus 27B v2 served by Rapid-MLX. |
| `qwopus-35b-a3b-rapidmlx-hermes.toml` | `rapidmlx` | Example primary-agent tuning, Qwopus 35B-A3B served by Rapid-MLX. |
| `qwen3.6-27b-mtplx-hermes-agent.toml` | `mtplx` | Example primary-agent tuning, Qwen3.6-27B Optimized-Speed served by MTPLX (native MTP D3), main slot 8080. |

To use one of these, the operator must have the matching GGUF available
and the conventional `~/llms/gguf/aux<N>/active.gguf` hard link in place
(or override `model_path` in their own copy of the preset).

The MTPLX preset has two extra preconditions of its own: the model must
already be in the MTPLX cache (`mtplx models` to check), and — because
MTPLX refuses a non-localhost bind without an API key, `/health`
included — a key file must exist at `~/.mtplx/api-key` (600 perms)
before installing. The preset's header comment documents both, plus the
`[network].api_key_file` field that lets the aisrv health/generation
probes authenticate.

## API key for non-loopback binds (`llamacpp-aux` presets)

The five `hermes-aux` presets bind `0.0.0.0`, so they run the engine
**authenticated**: `[binary].api_key_file` makes the generated plist pass
`--api-key-file <path>` to `llama-server`, which reads its API key(s)
from that file at startup. Only the *path* ever appears in the manifest,
the plist (which is world-readable), or `ps` output — never the key.

Before installing one of these presets, create the key file yourself
(the tooling never writes it):

```
mkdir -p ~/.config/asiai-inference-server
umask 177 && printf '%s\n' "$(openssl rand -hex 32)" \
  > ~/.config/asiai-inference-server/backend-api-key
```

`aisctl install` refuses to proceed when the declared file is missing
(llama-server would abort at startup and crash-loop) or contains no key
(llama-server parses zero keys and starts with auth *disabled* — a
silent fail-open on a LAN bind).

The same file is declared as `[network].api_key_file` so the aisrv
generation probe can authenticate. `llama-server` keeps `/health`,
`/v1/health`, `/models` and `/v1/models` public even when a key is set,
but `/v1/chat/completions` — the generation probe — is gated. When the
file is shared with the probe it must contain exactly **one** key line
and no comments: the probe sends the whole file content (stripped) as
the Bearer token, whereas `llama-server` itself accepts one key per
line with `#` comments.

The engine baselines (`llamacpp`, `llamacpp-aux-1..5`) ship with both
fields documented but commented out, so a plain out-of-the-box install
keeps working without a key file. Uncomment both fields together to
adopt the same convention there. Clients then call the engine with
`Authorization: Bearer <key>` (or `X-Api-Key: <key>`).
