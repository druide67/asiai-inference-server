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
    parser.add_argument(
        "--version", action="version", version=f"aisctl {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # list
    p_list = sub.add_parser("list", help="List known engines (manifests).")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=commands.cmd_list)

    # status
    p_status = sub.add_parser("status", help="Show engine state.")
    p_status.add_argument("engine", nargs="?", help="Engine name (default: all)")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=commands.cmd_status)

    # install
    p_install = sub.add_parser(
        "install", help="Install + start an engine via LaunchDaemon."
    )
    p_install.add_argument("engine")
    p_install.add_argument("--binary", help="Override binary path resolution.")
    p_install.add_argument("--user", help="macOS user the daemon runs as.")
    p_install.add_argument(
        "--firewall",
        choices=["lan-only", "none"],
        default="none",
        help="Apply pf anchor restricting to RFC1918 subnets.",
    )
    p_install.add_argument("--dry-run", action="store_true")
    p_install.add_argument("--force", action="store_true", help="Bypass operations lock.")
    p_install.add_argument("--json", action="store_true")
    p_install.set_defaults(func=commands.cmd_install)

    # uninstall
    p_uninstall = sub.add_parser("uninstall", help="Remove the LaunchDaemon + pf anchor.")
    p_uninstall.add_argument("engine")
    p_uninstall.add_argument("--keep-logs", action="store_true")
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

    # unload
    p_unload = sub.add_parser(
        "unload",
        help="Unload a model. Native API if supported, restart fallback otherwise.",
    )
    p_unload.add_argument("engine")
    p_unload.add_argument(
        "model", nargs="?",
        help="Model to unload. Omit to force a full daemon restart.",
    )
    p_unload.add_argument("--force", action="store_true")
    p_unload.add_argument("--json", action="store_true")
    p_unload.set_defaults(func=commands.cmd_unload)

    # purge
    p_purge = sub.add_parser(
        "purge",
        help="sudo /usr/sbin/purge + report measured delta in MB.",
    )
    p_purge.add_argument("--dry-run", action="store_true")
    p_purge.add_argument(
        "--force", action="store_true",
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

    # bootstrap
    p_boot = sub.add_parser(
        "bootstrap",
        help="One-shot setup. Currently: install /etc/sudoers.d/asiai-inference.",
    )
    p_boot.add_argument(
        "--install-sudoers", action="store_true",
        help="Write the sudoers fragment after visudo validation.",
    )
    p_boot.add_argument("--dry-run", action="store_true")
    p_boot.set_defaults(func=commands.cmd_bootstrap)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
