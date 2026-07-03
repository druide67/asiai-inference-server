"""``aisctl`` — fleet manager CLI for local LLM inference engines on Apple Silicon.

Top-level entry point. Subcommand dispatch is delegated to
:mod:`ais_cli.commands` so the parser definition stays focused on argument
shape.
"""

from __future__ import annotations

import argparse
import sys

from ais_cli import commands
from ais_core import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aisctl",
        description=(
            "Fleet manager for local LLM inference engines on Apple Silicon. "
            "Install/start/stop/unload engines, purge memory, manage profiles."
        ),
    )
    parser.add_argument("--version", action="version", version=f"aisctl {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # list
    p_list = sub.add_parser("list", help="List known engines (manifests).")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=commands.cmd_list)

    # status
    p_status = sub.add_parser("status", help="Show engine state.")
    p_status.add_argument("engine", nargs="?", help="Engine name (default: all)")
    p_status.add_argument("--json", action="store_true")
    p_status.add_argument(
        "--deep",
        action="store_true",
        help=(
            "Generation-probe running engines (1 token). Detects GPU-OOM "
            "zombies that /health cannot see (state becomes 'degraded')."
        ),
    )
    p_status.set_defaults(func=commands.cmd_status)

    # install
    p_install = sub.add_parser("install", help="Install + start an engine via LaunchDaemon.")
    p_install.add_argument("engine")
    p_install.add_argument("--binary", help="Override binary path resolution.")
    p_install.add_argument("--user", help="macOS user the daemon runs as.")
    p_install.add_argument(
        "--firewall",
        choices=["lan-only", "none"],
        default="none",
        help="Apply pf anchor restricting to RFC1918 subnets.",
    )
    p_install.add_argument(
        "--preset",
        help="Use a tuned preset manifest from data/engine_manifests/presets/ "
        "(e.g. qwen3.6-35b-a3b-hermes-agent-64gb). Run "
        "'aisctl list-presets' to see what's available.",
    )
    p_install.add_argument("--dry-run", action="store_true")
    p_install.add_argument(
        "--force",
        action="store_true",
        help="Bypass operations lock; also confirms replacing a preset-based "
        "install with the base manifest.",
    )
    p_install.add_argument("--json", action="store_true")
    p_install.set_defaults(func=commands.cmd_install)

    # reinstall
    p_reinstall = sub.add_parser(
        "reinstall",
        help="Uninstall + install replaying the recorded preset (see install).",
    )
    p_reinstall.add_argument("engine")
    p_reinstall.add_argument("--binary", help="Override binary path resolution.")
    p_reinstall.add_argument("--user", help="macOS user the daemon runs as.")
    p_reinstall.add_argument(
        "--firewall",
        choices=["lan-only", "none"],
        default=None,
        help="Override the recorded firewall mode (default: reuse it).",
    )
    p_reinstall.add_argument("--dry-run", action="store_true")
    p_reinstall.add_argument("--force", action="store_true", help="Bypass operations lock.")
    p_reinstall.add_argument("--json", action="store_true")
    p_reinstall.set_defaults(func=commands.cmd_reinstall)

    # list-presets
    p_presets = sub.add_parser(
        "list-presets",
        help="List bundled tuned-manifest presets for use with install --preset.",
    )
    p_presets.add_argument("--json", action="store_true")
    p_presets.set_defaults(func=commands.cmd_list_presets)

    # uninstall
    p_uninstall = sub.add_parser("uninstall", help="Remove the LaunchDaemon + pf anchor.")
    p_uninstall.add_argument("engine")
    p_uninstall.add_argument("--dry-run", action="store_true")
    p_uninstall.add_argument("--force", action="store_true")
    p_uninstall.add_argument("--json", action="store_true")
    p_uninstall.set_defaults(func=commands.cmd_uninstall)

    # start / stop / restart
    for name, help_text in [
        ("start", "Load the LaunchDaemon and wait for health."),
        ("stop", "Stop the LaunchDaemon."),
        ("restart", "Stop then start; wait for health."),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("engine")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--json", action="store_true")
        sp.set_defaults(func=getattr(commands, f"cmd_{name}"))

    # disable / enable — durable cold standby (issue #8)
    p_disable = sub.add_parser(
        "disable",
        help="Stop the engine AND keep it off across reboots (cold standby).",
    )
    p_disable.add_argument("engine")
    p_disable.add_argument("--dry-run", action="store_true")
    p_disable.add_argument("--force", action="store_true")
    p_disable.add_argument("--json", action="store_true")
    p_disable.set_defaults(func=commands.cmd_disable)

    p_enable = sub.add_parser(
        "enable",
        help="Re-enable a disabled engine (rejoins the boot sequence).",
    )
    p_enable.add_argument("engine")
    p_enable.add_argument(
        "--start",
        action="store_true",
        help="Also start it now and wait for health.",
    )
    p_enable.add_argument("--dry-run", action="store_true")
    p_enable.add_argument("--force", action="store_true")
    p_enable.add_argument("--json", action="store_true")
    p_enable.set_defaults(func=commands.cmd_enable)

    # upgrade
    p_upgrade = sub.add_parser(
        "upgrade",
        help="brew-upgrade a whitelisted engine; --restart to reconcile the daemon.",
    )
    p_upgrade.add_argument("engine")
    p_upgrade.add_argument(
        "--restart",
        action="store_true",
        help="Restart the daemon after upgrading so it runs the new binary.",
    )
    p_upgrade.add_argument("--dry-run", action="store_true")
    p_upgrade.add_argument("--force", action="store_true")
    p_upgrade.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Max seconds for the brew upgrade (default: 600).",
    )
    p_upgrade.add_argument("--json", action="store_true")
    p_upgrade.set_defaults(func=commands.cmd_upgrade)

    # unload
    p_unload = sub.add_parser(
        "unload",
        help="Unload a model. Native API if supported, restart fallback otherwise.",
    )
    p_unload.add_argument("engine")
    p_unload.add_argument(
        "model",
        nargs="?",
        help="Model to unload. Omit to force a full daemon restart.",
    )
    p_unload.add_argument("--force", action="store_true")
    p_unload.add_argument("--json", action="store_true")
    p_unload.set_defaults(func=commands.cmd_unload)

    # load
    p_load = sub.add_parser(
        "load",
        help="Warm-load a model so the first inference is hot (Ollama / LM Studio).",
    )
    p_load.add_argument("engine")
    p_load.add_argument("model")
    p_load.add_argument(
        "--keep-alive",
        default="5m",
        help="How long to keep the model resident (Ollama only). Default: 5m.",
    )
    p_load.add_argument("--force", action="store_true")
    p_load.add_argument("--json", action="store_true")
    p_load.set_defaults(func=commands.cmd_load)

    # purge
    p_purge = sub.add_parser(
        "purge",
        help="sudo /usr/sbin/purge + report measured delta in MB.",
    )
    p_purge.add_argument("--dry-run", action="store_true")
    p_purge.add_argument(
        "--force",
        action="store_true",
        help="Bypass operations lock (e.g. while asiai bench is running).",
    )
    p_purge.add_argument("--json", action="store_true")
    p_purge.set_defaults(func=commands.cmd_purge)

    # repair
    p_repair = sub.add_parser(
        "repair",
        help="Clean stale operations.lock + report orphan com.asiai.* plists.",
    )
    p_repair.add_argument("--dry-run", action="store_true")
    p_repair.add_argument("--json", action="store_true")
    p_repair.set_defaults(func=commands.cmd_repair)

    # bundle (SMAppService — Background Items panel identity)
    p_bundle = sub.add_parser(
        "bundle",
        help="Build/manage the SMAppService app bundle (icon + name in Background Items).",
    )
    bundle_sub = p_bundle.add_subparsers(dest="bundle_cmd", required=True, metavar="<action>")

    pb_build = bundle_sub.add_parser(
        "build", help="Build <App>.app embedding one LaunchDaemon per service."
    )
    pb_build.add_argument(
        "--services",
        required=True,
        help="Comma-separated engine names to embed (e.g. llamacpp-aux-1,llamacpp-aux-2).",
    )
    pb_build.add_argument("--output", default="dist", help="Output directory (default: dist).")
    pb_build.add_argument("--user", help="macOS user the daemons run as (default: $USER).")
    pb_build.add_argument(
        "--launcher",
        help="Path to asiai-launch baked into the exec stub (default: resolved from PATH).",
    )
    pb_build.add_argument(
        "--bundle-id",
        default="dev.asiai.engines",
        help="Reverse-DNS bundle identifier (forks: use your own namespace).",
    )
    pb_build.add_argument("--app-name", default="Asiai", help="App bundle name (<name>.app).")
    pb_build.add_argument(
        "--display-name",
        default="asiai Inference Engines",
        help="Label shown in the Background Items panel.",
    )
    pb_build.add_argument(
        "--sign",
        help="Code-signing identity (a locally-trusted cert; required for the "
        "custom icon to show on macOS 26+).",
    )
    pb_build.add_argument("--json", action="store_true")
    pb_build.set_defaults(func=commands.cmd_bundle_build)

    pb_activate = bundle_sub.add_parser(
        "activate",
        help="Publish the active manifest asiai-launch reads for an engine.",
    )
    pb_activate.add_argument("engine")
    pb_activate.add_argument(
        "--preset",
        help="Preset to activate (default: the preset recorded at install time).",
    )
    pb_activate.add_argument("--json", action="store_true")
    pb_activate.set_defaults(func=commands.cmd_bundle_activate)

    for action, help_text in [
        ("register", "Register the bundle's daemons with SMAppService."),
        ("unregister", "Unregister the bundle's daemons."),
        ("status", "Show SMAppService status per service."),
    ]:
        pb = bundle_sub.add_parser(action, help=help_text)
        pb.add_argument("service", nargs="?", help="Service name (default: all).")
        pb.add_argument(
            "--app",
            default="/Applications/Asiai.app",
            help="Installed bundle path (default: /Applications/Asiai.app).",
        )
        if action == "register":
            pb.add_argument(
                "--allow-headless",
                action="store_true",
                help="Register without a GUI session (daemons stay in requiresApproval "
                "until the toggle is flipped in System Settings — know what you do).",
            )
        pb.set_defaults(func=commands.cmd_bundle_ctl, action=action)

    # bootstrap
    p_boot = sub.add_parser(
        "bootstrap",
        help="One-time setup: install the privileged helper + helper-only sudoers fragment.",
    )
    # The four verbs are mutually exclusive: this is a root- and sudoers-mutating command, so a
    # fat-fingered `--rollback --install` must error loudly, never silently run one and drop the
    # other. --dedicated-user / --dry-run are modifiers and stay outside the group.
    p_boot_verb = p_boot.add_mutually_exclusive_group()
    p_boot_verb.add_argument(
        "--install",
        action="store_true",
        help="Full one-time bootstrap: I0 chain check -> install helper "
        "(/Library/PrivilegedHelperTools/asiai-priv root:wheel 0755) -> install sudoers. "
        "Idempotent.",
    )
    p_boot_verb.add_argument(
        "--install-sudoers",
        action="store_true",
        help="Install only the sudoers fragment (granular/legacy), after visudo validation.",
    )
    p_boot_verb.add_argument(
        "--verify",
        action="store_true",
        help="Recompute the installed helper's SHA-256 and compare it to its sidecar (NFR11).",
    )
    p_boot_verb.add_argument(
        "--rollback",
        action="store_true",
        help="Revert (FR8): restore the pre-bootstrap sudoers fragment (anti-lockout, "
        "visudo-validated), then remove the helper and its signature sidecar.",
    )
    p_boot.add_argument(
        "--dedicated-user",
        action="store_true",
        help="With --install: also create the hidden non-admin role account engines run as "
        "(_aisrv, uid 450-499). Recommended; confines an engine RCE to a powerless uid.",
    )
    p_boot.add_argument("--dry-run", action="store_true")
    p_boot.set_defaults(func=commands.cmd_bootstrap)

    # serve (Phase 2) — loopback HTTP server for fleet write commands
    from ais_cli.serve import add_serve_subparser

    add_serve_subparser(sub)

    # fleet (Phase 2) — orchestrator-side push to remote nodes
    from ais_cli.fleet import add_fleet_subparser

    add_fleet_subparser(sub)

    # install-service / uninstall-service (Phase 2) — LaunchDaemons for
    # the two companion services (asiai-web on the LAN edge, aisctl-serve
    # on the loopback).
    from ais_cli.install_service import add_install_service_subparsers

    add_install_service_subparsers(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
