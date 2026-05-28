"""Generate, validate, and install ``/etc/sudoers.d/asiai-inference``.

The strict-scope sudoers rule is what makes the unload+purge pipeline
runnable from a non-interactive Claude Code agent: every privileged call we
make is whitelisted by exact path, with the wildcard restricted to
``com.asiai.*`` plist labels and pf anchors.

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
from ais_core.manifest import EngineManifest

SUDOERS_PATH = "/etc/sudoers.d/asiai-inference"
ADMIN_GROUP = "%admin"


class SudoersError(RuntimeError):
    """Raised when the sudoers content is malformed or cannot be installed."""


def generate_sudoers_content(manifests: list[EngineManifest] | None = None) -> str:
    """Render the sudoers content. Pure function, no I/O.

    The wildcard pattern ``com.asiai.*`` is the same one validated by
    :func:`ais_core.manifest.is_valid_plist_label`, so no engine installed
    via this tool can escape the scope.
    """
    # `manifests` is currently unused — the wildcard makes the rules
    # engine-agnostic — but we keep the parameter so a future v0.2 can
    # narrow the scope to specifically-installed engines.
    _ = manifests

    lines = [
        "# /etc/sudoers.d/asiai-inference",
        "# Managed by asiai-inference-server. Do not edit by hand.",
        "# Generated via `aisctl bootstrap --install-sudoers`.",
        "#",
        "# Scope: every wildcard is restricted to com.asiai.* so engines",
        "# installed manually (or by openclaw legacy plists) are NOT covered.",
        "",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /usr/sbin/purge",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/launchctl load -w "
        "/Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/launchctl unload "
        "/Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/launchctl unload -w "
        "/Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/launchctl stop com.asiai.*",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/launchctl start com.asiai.*",
        # Phase 2 LaunchDaemon bootstrap / bootout (modern launchctl).
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/launchctl bootstrap system "
        "/Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/launchctl bootout system/com.asiai.*",
        # Phase 2 install-service uses /usr/bin/install for atomic plist drop.
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /usr/bin/install -m 0644 -o root -g wheel "
        "/tmp/asiai-* /Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /usr/sbin/sysctl iogpu.wired_limit_mb=*",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /sbin/pfctl -f /etc/pf.conf",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /sbin/pfctl -f /etc/pf.anchors/com.asiai.*",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /sbin/pfctl -e",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /sbin/pfctl -nf -",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/mkdir -p /etc/pf.anchors",
        # Source paths are scoped to /tmp/asiai-*/ — a per-invocation 0700
        # staging directory created by ais_core.io.secure_staging_dir().
        # The directory's mode 0700 collapses the symlink-swap TOCTOU window
        # that would otherwise exist between file creation and `sudo /bin/mv`.
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/mv /tmp/asiai-*/com.asiai.*.plist "
        "/Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/mv /tmp/asiai-*/com.asiai.*.anchor "
        "/etc/pf.anchors/com.asiai.*",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/mv /tmp/asiai-*/pf.conf /etc/pf.conf",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/mv "
        "/tmp/asiai-*/asiai-inference.sudoers /etc/sudoers.d/asiai-inference",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /usr/sbin/chown root\\:wheel "
        "/Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /usr/sbin/chown root\\:wheel "
        "/etc/pf.anchors/com.asiai.*",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /usr/sbin/chown root\\:wheel /etc/pf.conf",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/chmod 644 "
        "/Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/chmod 644 /etc/pf.anchors/com.asiai.*",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/chmod 644 /etc/pf.conf",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/rm -f /Library/LaunchDaemons/com.asiai.*.plist",
        f"{ADMIN_GROUP} ALL=(ALL) NOPASSWD: /bin/rm -f /etc/pf.anchors/com.asiai.*",
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
