"""Invoke the root-owned privileged helper (``asiai-priv``).

After AEP-01 the ``sudoers`` fragment grants ``NOPASSWD`` on a **single** binary —
the helper at :data:`ais_core.sudoers.PRIVILEGED_HELPER_PATH` — and nothing else. Every
privileged daemon-lifecycle operation (``lifecycle.py``) and the memory purge route through
this module instead of raw ``sudo launchctl``/``sudo mv``/``sudo /usr/sbin/purge``.

The helper runs under ``python3 -I`` as root and re-validates + *generates*: callers pass
**bounded flags**, never a plist or a free path that becomes a privileged write verbatim.

Invariant #3 (load-bearing): the EXECUTED helper is the ``root:wheel`` copy installed at
``PRIVILEGED_HELPER_PATH`` — never the admin-writable ``site-packages`` source of
``data/helpers/asiai_priv.py``. This module always invokes that absolute installed path, so
``sudo`` resolves the rule against the same inode the operator reviewed and bootstrapped.
"""

from __future__ import annotations

import subprocess

from ais_core.sudoers import PRIVILEGED_HELPER_PATH

# Mirror the helper's stable exit-code contract (asiai_priv.py) so callers get a readable
# reason rather than a bare exit code. The helper itself prints nothing on the happy path
# and audits refusals/errors to its own root-owned log.
_EXIT_REASON = {
    1: "internal error",
    2: "refused (failed a helper-side validation)",
}


class PrivHelperError(RuntimeError):
    """Raised when an ``asiai-priv`` invocation exits non-zero (and ``check=True``)."""


def helper_path() -> str:
    """Absolute path of the privileged helper that is actually executed (invariant #3)."""
    return PRIVILEGED_HELPER_PATH


def run(
    action: str,
    *args: str,
    check: bool = True,
    dry_run: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str] | None:
    """Invoke ``sudo <helper> <action> [args...]`` as an argument list (never a shell).

    ``args`` are passed through verbatim as separate ``argv`` entries — callers build them
    from validated/bounded values; the helper re-validates everything regardless. Returns the
    completed process, or ``None`` in ``dry_run`` mode (the command is printed, nothing runs).

    With ``check=True`` (default) a non-zero exit raises :class:`PrivHelperError` with the
    helper's exit-code meaning. ``check=False`` returns the process so the caller can treat a
    failure as best-effort (e.g. booting out a daemon that was never loaded). ``timeout`` (when
    set) bounds the call; a timeout raises :class:`PrivHelperError` regardless of ``check``.
    """
    argv = ["sudo", PRIVILEGED_HELPER_PATH, action, *args]
    if dry_run:
        print("[dry-run] " + " ".join(argv))
        return None
    try:
        proc = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise PrivHelperError(f"asiai-priv {action} timed out after {timeout}s") from exc
    if check and proc.returncode != 0:
        reason = _EXIT_REASON.get(proc.returncode, f"exit {proc.returncode}")
        detail = (proc.stderr or proc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise PrivHelperError(f"asiai-priv {action} failed — {reason}{suffix}")
    return proc
