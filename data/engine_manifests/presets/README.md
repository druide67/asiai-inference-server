# Engine manifest presets — bundled examples

A preset is a full, self-contained `EngineManifest` TOML tuned for a
specific model and use case. Pick one at install time:

```
aisctl install <engine> --preset <preset-name>
```

The preset replaces the engine baseline verbatim — there is no field
merge. If you copy a preset to start from, expect to edit every field
that matters to your hardware (model path symlink, port if you run
multiple instances of the same engine, KV cache type, parallel slots,
context size).

## What lives in this directory

Everything here is a **bundled example**. The presets ship with the
package so contributors and adopters can read realistic, end-to-end
tuning side-by-side with the baseline manifests, but **none of them is
required**: the engine baselines work as-is for an out-of-the-box
install.

Presets named with `hermes` in their stem are concrete tuning examples
adapted for the [Hermes Agent](https://) class of orchestrator —
multi-turn agentic workloads with a system prefix that recurs across
turns and short user payloads that change every turn. They are example
references, not a product configuration: an asiai user adopting a
different orchestrator (Open WebUI, LiteLLM-fronted, a custom agent
framework, ...) will most likely write their own preset with different
tuning. Read the header comments — each preset explains *why* the
chosen flags exist for that scenario, which is the part worth reusing.

The `llamacpp-aux-N` presets demonstrate the multi-instance pattern:
four `llama-server` instances running side-by-side on dedicated ports
8090-8093 with distinct model symlinks. Add a fifth instance by
dropping `llamacpp-aux-5.toml` next to the others (see the user-config
section below for the recommended location), then create a matching
preset if you want to ship tuning along with it.

## Writing your own presets without committing them upstream

If you want to add or override presets without modifying the package,
drop your TOMLs in the XDG user-config directory:

```
$XDG_CONFIG_HOME/asiai-inference-server/presets/<name>.toml
```

The default path on macOS is `~/.config/asiai-inference-server/presets/`.
You can override the whole user-config tree with the `ASIAI_USER_CONFIG_DIR`
environment variable.

Files there are discovered automatically by `list_presets()` and
`aisctl install --preset <name>` resolves them just like bundled
presets. A user preset whose stem matches a bundled one silently wins
the lookup (with a warning logged at load time so the override is
visible to operators).

The same XDG mechanism applies to engine manifests themselves: drop
`engine_manifests/<name>.toml` under the user-config directory and
`aisctl status / install / start ...` will pick it up.

## Validation

Every preset is loaded via `load_manifest(<engine>, preset=<name>)`,
which validates that `name` (the engine identifier inside the TOML)
matches the engine the operator asked for. A preset that targets
`llamacpp-aux-3` cannot be installed on `llamacpp-aux-1` — `aisctl`
rejects the mismatch before touching the system.

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

To use one of these, the operator must have the matching GGUF available
and the conventional `~/llms/gguf/aux<N>/active.gguf` symlink in place
(or override `model_path` in their own copy of the preset).
