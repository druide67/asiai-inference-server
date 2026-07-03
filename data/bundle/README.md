# SMAppService bundle — engines with a name and an icon in Background Items

Raw `/Library/LaunchDaemons` plists show up in macOS Settings > Login Items &
Extensions > "Apps in the background" as unidentified Unix executables
(generic icon, raw binary name — `llama-server` ×N). Building an app bundle
and registering the same daemons through `SMAppService.daemon` gives them one
entry with a proper display name and a custom icon.

## Quick start

```bash
# 1. Build the bundle (one embedded LaunchDaemon per engine)
aisctl bundle build --services llamacpp-aux-1,llamacpp-aux-2 \
    [--sign "My Local Code Signing"]

# 2. Publish the active manifest each daemon will exec from
#    (defaults to the preset recorded at install time)
aisctl bundle activate llamacpp-aux-1
aisctl bundle activate llamacpp-aux-2

# 3. Install + register
cp -R dist/Asiai.app /Applications/
aisctl uninstall llamacpp-aux-1   # the same label must not be loaded twice
aisctl bundle register            # or: aisctl bundle register llamacpp-aux-1
aisctl bundle status
```

`register` enforces both preconditions: it refuses while a legacy
`/Library/LaunchDaemons` plist for a selected service still exists (or its
label is still loaded), and it refuses outside a GUI session (the approval
toggle is GUI-only; `--allow-headless` overrides if you will approve later).

`status: requiresApproval` means registration worked and macOS is waiting for
you to flip the toggle in Settings > Login Items & Extensions.

## How it works (sealed-bundle constraint)

A signed bundle is sealed — changing an embedded plist invalidates the
signature. So nothing tuning-specific lives inside:

```
embedded plist  →  Contents/MacOS/Asiai <service>     (tiny C exec stub)
                →  asiai-launch <service>             (path baked at build)
                →  reads the ACTIVE manifest          (outside the bundle)
                →  exec()s the engine binary
```

The active manifests live in `~/.local/share/asiai-inference-server/active/`
(override: `ASIAI_LAUNCH_MANIFEST_DIR`) and are written by
`aisctl bundle activate`. Changing a model, context size or env var means
re-running `activate` — never rebuilding or re-signing the bundle.

`asiai-launch` refuses to exec anything outside a hardcoded allowlist of
engine binaries, and refuses active manifests with unsafe permissions
(group/world-writable or foreign-owned) — a tampered config file cannot turn
the trusted bundle into a confused deputy.

## Code signing (the icon part)

On macOS 26+ ("Tahoe"), Background Task Management silently rejects custom
icons from ad-hoc-signed bundles: an unsigned bundle works fine but shows the
generic icon. To get the icon, sign with a **locally-trusted code-signing
certificate** (a free self-signed cert in the System keychain, trusted for
Code Signing only — no Apple Developer ID needed for your own machines):

```bash
aisctl bundle build --services ... --sign "My Local Code Signing"
# or sign an existing build:
xattr -cr /Applications/Asiai.app
codesign --force --deep --sign "My Local Code Signing" /Applications/Asiai.app
```

This repo ships no certificate and no pre-signed bundle (a published private
key would let anyone impersonate the signer; a pre-signed bundle is invalid
on any other machine). Each deployer creates their own local cert. Forks
should also use their own reverse-DNS namespace via `--bundle-id`.

## Files here

| File | Role |
|---|---|
| `templates/stub.c` | Exec stub compiled into `Contents/MacOS/<App>` (launcher path baked via `-DASIAI_LAUNCHER_PATH`) |
| `templates/register.swift` | `<App>Register` helper — SMAppService register/unregister/status per service |
| `AppIcon-1024.png` | Icon source (asiai logo), turned into `AppIcon.icns` at build time via sips/iconutil |
