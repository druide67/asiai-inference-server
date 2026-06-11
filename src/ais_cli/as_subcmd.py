"""Versioned sub-CLI registration hook for the asiai entry-point group.

When ``asiai-inference-server`` is installed alongside ``asiai``, the asiai
CLI calls :func:`register` via the ``asiai.subcommands`` entry-point group.
We expose a versioned dataclass-based contract instead of a positional
``(subparsers, commands)`` tuple — that way asiai can evolve its CLI
internals (add a parent parser, switch to click, …) without silently
breaking every plugin.

Contract versioning (Copilot Q1)
--------------------------------
* :data:`PLUGIN_API_VERSION` is the major contract version we *implement*.
* Asiai exposes its own :data:`PLUGIN_API_VERSION_SUPPORTED` (a set of major
  versions it can talk to).
* :func:`register` validates compatibility and raises early with a clear
  message if asiai is too old or too new.
* The :class:`RegistrationContext` dataclass carries everything a plugin
  needs (subparsers, command map, asiai version, host CLI mode). New
  fields can be added in minor versions; breaking changes bump the major.

Cold-start (Copilot Q4)
-----------------------
``register`` itself stays light: it only adds argparse parsers and stores
*function references* to the handlers in :mod:`ais_cli.commands`. The
handlers are not imported until the user actually runs ``asiai engine
<subcmd>`` — that lazy boundary keeps ``asiai --version`` cold-start under
its current budget even when the plugin is installed.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable
from typing import Any

PLUGIN_API_VERSION: int = 1
"""Major version of the asiai-subcommands contract that this plugin implements."""


@dataclasses.dataclass(frozen=True)
class RegistrationContext:
    """Everything an asiai sub-CLI plugin needs at registration time.

    Attributes
    ----------
    api_version
        Major contract version asiai is offering. Plugin must verify this
        matches its own :data:`PLUGIN_API_VERSION`.
    subparsers
        The argparse ``_SubParsersAction`` from asiai's top-level parser.
    commands
        A mutable mapping ``name -> handler(args) -> int`` that asiai uses
        to dispatch sub-CLI invocations.
    asiai_version
        Asiai's own version string (informational; do not parse).
    cli_mode
        ``"interactive"`` for ``asiai engine ...`` direct invocation,
        ``"plugin"`` for any other host (reserved for future MCP/web hosts).
    """

    api_version: int
    subparsers: argparse._SubParsersAction
    commands: dict[str, Callable[[argparse.Namespace], int]]
    asiai_version: str = ""
    cli_mode: str = "interactive"


class IncompatiblePluginError(RuntimeError):
    """Raised when asiai's PLUGIN_API_VERSION isn't in our supported range."""


def validate_asiai_compat(context_api_version: int) -> None:
    """Raise IncompatiblePluginError if the asiai-side major version disagrees.

    For now we accept the exact same major; v0.x is pre-1.0 and breaking
    changes are still on the table. Once we hit 1.0 we'll widen the range
    to ``[1, ∞)`` until v2.
    """
    if context_api_version != PLUGIN_API_VERSION:
        raise IncompatiblePluginError(
            f"asiai-inference-server expects PLUGIN_API_VERSION={PLUGIN_API_VERSION}, "
            f"asiai offered {context_api_version}. "
            "Upgrade asiai or asiai-inference-server so the major versions match."
        )


def register(*args: Any, **kwargs: Any) -> None:
    """Asiai-side entry point. Accepts both calling conventions.

    Asiai's loader can pass either:
      * the modern ``RegistrationContext`` dataclass (recommended);
      * the legacy ``(subparsers, commands)`` tuple — for the brief window
        during which old asiai versions are still in the wild before they
        ship the dataclass-aware loader.

    The split is invisible to the rest of the plugin: both branches end up
    calling :func:`_attach_engine_subparser` with the same arguments.
    """
    ctx = _coerce_context(args, kwargs)
    validate_asiai_compat(ctx.api_version)
    _attach_engine_subparser(ctx)


def _coerce_context(args: tuple, kwargs: dict) -> RegistrationContext:
    """Accept either RegistrationContext, ``(subparsers, commands)``, or kwargs."""
    if len(args) == 1 and isinstance(args[0], RegistrationContext):
        return args[0]

    if len(args) == 2:
        subparsers, commands = args
        return RegistrationContext(
            api_version=PLUGIN_API_VERSION,
            subparsers=subparsers,
            commands=commands,
        )

    if "context" in kwargs and isinstance(kwargs["context"], RegistrationContext):
        return kwargs["context"]

    raise TypeError(
        "register() expects a RegistrationContext, or (subparsers, commands), "
        f"got args={args!r} kwargs={list(kwargs)}"
    )


def _attach_engine_subparser(ctx: RegistrationContext) -> None:
    """Add the ``engine`` subparser with all aisctl subcommands as children.

    The actual handlers live in :mod:`ais_cli.commands`; we reference them
    via lazy thunks so the heavy imports only run when the user invokes a
    subcommand.
    """
    parser = ctx.subparsers.add_parser(
        "engine",
        help="Manage local inference engines (install/start/stop/unload/purge).",
        description=(
            "asiai-inference-server — sub-CLI auto-injected into asiai. "
            "Same handlers as the standalone `aisctl` binary."
        ),
    )

    engine_sub = parser.add_subparsers(dest="engine_cmd", required=True, metavar="<subcommand>")

    # --- subcommands ----------------------------------------------------
    # Each entry: (name, help, [(arg_name, kwargs), ...], handler_attr)
    specs = [
        (
            "list",
            "List known engines.",
            [
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_list",
        ),
        (
            "status",
            "Show engine state.",
            [
                (("engine",), {"nargs": "?"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_status",
        ),
        (
            "install",
            "Install + start an engine.",
            [
                (("engine",), {}),
                (("--binary",), {}),
                (("--user",), {}),
                (("--firewall",), {"choices": ["lan-only", "none"], "default": "none"}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_install",
        ),
        (
            "reinstall",
            "Uninstall + install replaying the recorded preset.",
            [
                (("engine",), {}),
                (("--binary",), {}),
                (("--user",), {}),
                (("--firewall",), {"choices": ["lan-only", "none"], "default": None}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_reinstall",
        ),
        (
            "uninstall",
            "Remove an engine.",
            [
                (("engine",), {}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_uninstall",
        ),
        (
            "start",
            "Start an engine.",
            [
                (("engine",), {}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_start",
        ),
        (
            "stop",
            "Stop an engine.",
            [
                (("engine",), {}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_stop",
        ),
        (
            "restart",
            "Restart an engine.",
            [
                (("engine",), {}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_restart",
        ),
        (
            "disable",
            "Stop an engine AND keep it off across reboots (cold standby).",
            [
                (("engine",), {}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_disable",
        ),
        (
            "enable",
            "Re-enable a disabled engine (rejoins the boot sequence).",
            [
                (("engine",), {}),
                (("--start",), {"action": "store_true"}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_enable",
        ),
        (
            "upgrade",
            "brew-upgrade a whitelisted engine (--restart to reconcile).",
            [
                (("engine",), {}),
                (("--restart",), {"action": "store_true"}),
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--timeout",), {"type": float, "default": 600.0}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_upgrade",
        ),
        (
            "unload",
            "Unload a model (API or restart fallback).",
            [
                (("engine",), {}),
                (("model",), {"nargs": "?"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_unload",
        ),
        (
            "purge",
            "sudo purge + measured delta.",
            [
                (("--dry-run",), {"action": "store_true"}),
                (("--force",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_purge",
        ),
        (
            "repair",
            "Clean stale lock + report orphan plists.",
            [
                (("--dry-run",), {"action": "store_true"}),
                (("--json",), {"action": "store_true"}),
            ],
            "cmd_repair",
        ),
        (
            "bootstrap",
            "Install /etc/sudoers.d/asiai-inference (visudo-validated).",
            [
                (("--install-sudoers",), {"action": "store_true"}),
                (("--dry-run",), {"action": "store_true"}),
            ],
            "cmd_bootstrap",
        ),
    ]

    for name, help_text, sub_args, handler_attr in specs:
        sp = engine_sub.add_parser(name, help=help_text)
        for flag_or_arg, kwargs in sub_args:
            sp.add_argument(*flag_or_arg, **kwargs)
        sp.set_defaults(func=_lazy_handler(handler_attr))

    # The asiai dispatcher uses ``commands["engine"]`` for first-level routing;
    # we forward to the inner subparser via _route_engine.
    parser.set_defaults(func=_route_engine)
    ctx.commands["engine"] = _route_engine


def _lazy_handler(attr_name: str):
    """Return a thunk that imports ais_cli.commands only when called."""

    def thunk(args: argparse.Namespace) -> int:
        from ais_cli import commands as _c

        return int(getattr(_c, attr_name)(args))

    thunk.__name__ = f"lazy_{attr_name}"
    return thunk


def _route_engine(args: argparse.Namespace) -> int:
    """Dispatch ``asiai engine <sub>`` to the right inner handler.

    argparse already populated ``args.func`` with the subcommand thunk during
    parsing, so we just call it. This wrapper exists to give asiai's command
    map a stable ``"engine"`` key.
    """
    func = getattr(args, "func", None)
    if func is None or func is _route_engine:
        # No subcommand was given (parser will already have errored out, but
        # be defensive).
        return 2
    return int(func(args))
