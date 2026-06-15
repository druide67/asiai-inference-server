"""Generate, validate, and install ``/etc/sudoers.d/asiai-inference``.

The fragment grants ``NOPASSWD`` on the **privileged helper alone**
(``/Library/PrivilegedHelperTools/asiai-priv``) — no wildcard rules on raw
``launchctl``/``pfctl``/``mv``/``chown``/``chmod``. The helper validates every
operation internally (generate-don't-validate, AEP-01), so one allowlisted
command replaces the entire wildcard surface that an attacker could over-match.

The file is **never** installed silently. ``aisctl bootstrap --install-sudoers``
prints the generated content, runs ``visudo -cf`` to fail-fast on syntax
errors, and only then ``sudo /bin/mv``-s it into place.

Why ``visudo -cf`` matters
--------------------------
A malformed sudoers file at boot can lock out every privileged operation on
the machine, including the ability to remove the broken file. Apple's docs
explicitly recommend ``visudo`` for this reason. The Bash version
(``setup-headless.sh:113``) called it; we keep that habit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ais_core.io import secure_staging_dir

SUDOERS_PATH = "/etc/sudoers.d/asiai-inference"
ADMIN_GROUP = "%admin"
# Canonical path of the EXECUTED helper: the root-owned copy that the bootstrap (story 2.1)
# places at /Library/PrivilegedHelperTools/asiai-priv (root:wheel 0755). The sudoers rule and
# the bootstrap copy MUST point at this same path — never the admin-writable site-packages
# source (invariant #3).
PRIVILEGED_HELPER_PATH = "/Library/PrivilegedHelperTools/asiai-priv"


class SudoersError(RuntimeError):
    """Raised when the sudoers content is malformed or cannot be installed."""


def generate_sudoers_content() -> str:
    """Render the helper-only sudoers fragment. Pure function, no I/O.

    Grants ``NOPASSWD`` on the privileged helper ALONE — zero wildcard rules on raw
    ``launchctl``/``pfctl``/``mv``/``chown``/``chmod``/``rm``. The helper validates every
    operation internally, so one allowlisted command (with any arguments — the helper
    self-validates them) replaces the entire over-matchable wildcard surface (AEP-01).

    Security — ``env_reset``/``SUDO_*``: the helper derives the daemon's non-root run-as
    identity from ``SUDO_UID`` (I2), so that value must be the real invoker's, never
    caller-spoofable. sudo guarantees this itself: it sets ``SUDO_UID``/``SUDO_GID``/
    ``SUDO_USER`` from the authenticated invoker AFTER the env_keep/env_check pass, with
    overwrite — so even a hostile global ``env_keep += SUDO_UID`` cannot shield a caller
    value. The per-command ``Defaults!<helper> env_reset`` is defence-in-depth ONLY for the
    case where a site disabled ``env_reset`` globally. Do NOT add an ``env_delete`` /
    ``env_keep`` for ``SUDO_*``: it is unnecessary and could strip the legitimate sudo-set
    value the helper depends on, breaking I2. (Verified: sudo 1.9.17p2 man pages + env.c.)
    """
    lines = [
        "# /etc/sudoers.d/asiai-inference",
        "# Managed by asiai-inference-server. Do not edit by hand.",
        "# Generated via `aisctl bootstrap --install-sudoers`.",
        "#",
        "# Single privileged surface: the root-owned helper validates everything",
        "# internally; there are NO wildcard rules on launchctl/pfctl/mv/chown/chmod.",
        "# SECURITY: the helper trusts SUDO_UID (set by sudo from the real invoker) to",
        "# derive the daemon's non-root run-as account (I2). env_reset stays on; SUDO_* is",
        "# never env_kept. Do NOT add env_delete/env_keep for SUDO_* (would break I2).",
        "",
        f"Defaults!{PRIVILEGED_HELPER_PATH} env_reset",
        # runas restricted to root: the helper must run as root; (root) drops the needless
        # freedom of (ALL) to target another account (the helper would refuse euid!=0 anyway).
        f"{ADMIN_GROUP} ALL=(root) NOPASSWD: {PRIVILEGED_HELPER_PATH}",
        "",
    ]
    return "\n".join(lines)


def validate_content(content: str) -> None:
    """Run ``visudo -cf`` against the rendered content.

    Writes to a private tempfile (``visudo -cf`` only accepts paths, not stdin)
    and removes it whether validation succeeds or not.
    """
    with secure_staging_dir() as staging:
        tmp_path = staging / "asiai-inference.sudoers"
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.chmod(0o440)
        proc = subprocess.run(
            ["/usr/sbin/visudo", "-cf", str(tmp_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SudoersError(
                f"visudo rejected sudoers content:\n{(proc.stderr or proc.stdout).strip()}"
            )


def install_sudoers(content: str | None = None, *, dry_run: bool = False) -> str:
    """Validate then atomically install the sudoers fragment.

    Returns the destination path. In dry-run mode prints the content and
    returns the destination without touching the filesystem.
    """
    if content is None:
        content = generate_sudoers_content()

    if dry_run:
        print(f"--- {SUDOERS_PATH} ---")
        print(content, end="")
        print("--- end sudoers ---")
        return SUDOERS_PATH

    # Detect non-TTY environments (Claude Code shells, CI pipes, agent runners)
    # before sudo silently fails on password prompt. Surface clear instructions
    # so the operator can run the install manually in a real terminal.
    if not sys.stdin.isatty():
        raise SudoersError(
            "aisctl bootstrap --install-sudoers requires an interactive terminal "
            "the first time (sudo password is needed for /bin/mv into "
            f"{SUDOERS_PATH}). Run this command directly in Terminal.app:\n\n"
            "  cd ~/projets/asiai-inference-server && "
            ".venv/bin/aisctl bootstrap --install-sudoers\n\n"
            "After the sudoers fragment is installed, subsequent privileged "
            "operations (engine install/start/stop/unload/purge) are NOPASSWD "
            "and can run from non-TTY contexts."
        )

    validate_content(content)

    with secure_staging_dir() as staging:
        tmp_path = staging / "asiai-inference.sudoers"
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.chmod(0o440)
        try:
            subprocess.run(["sudo", "/bin/mv", str(tmp_path), SUDOERS_PATH], check=True)
            subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", SUDOERS_PATH], check=True)
            subprocess.run(["sudo", "/bin/chmod", "440", SUDOERS_PATH], check=True)
        except subprocess.CalledProcessError as e:
            raise SudoersError(f"Failed to install {SUDOERS_PATH}: {e}") from e

    return SUDOERS_PATH


def is_installed() -> bool:
    """True iff the sudoers fragment exists at the canonical path."""
    return Path(SUDOERS_PATH).is_file()


def remove_sudoers(*, dry_run: bool = False) -> bool:
    """Remove the sudoers fragment. Returns True if removed, False if absent."""
    if not is_installed():
        return False

    if dry_run:
        print(f"[dry-run] sudo /bin/rm -f {SUDOERS_PATH}")
        return True

    subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_PATH], check=True)
    return True
