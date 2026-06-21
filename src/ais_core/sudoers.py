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
SUDOERS_DIR = "/etc/sudoers.d"
# Staging name INSIDE the root-owned sudoers.d, dot-prefixed: sudo ignores any sudoers.d
# file whose name contains a '.', so a half-installed fragment is never parsed as a rule.
SUDOERS_STAGED = f"{SUDOERS_DIR}/.asiai-inference.tmp"
ADMIN_GROUP = "%admin"
# Canonical path of the EXECUTED helper: the root-owned copy that the bootstrap (story 2.1)
# places at /Library/PrivilegedHelperTools/asiai-priv (root:wheel 0755). The sudoers rule and
# the bootstrap copy MUST point at this same path — never the admin-writable site-packages
# source (invariant #3).
PRIVILEGED_HELPER_PATH = "/Library/PrivilegedHelperTools/asiai-priv"

# --- Rollback support (FR8, story 2.3) ------------------------------------------------------
# A rollback must restore EXACTLY the pre-bootstrap state of SUDOERS_PATH so the operator is never
# locked out of sudo. We record that state ONCE — the first time a fragment is installed — as one
# of two DOTTED files in sudoers.d. sudo ignores any sudoers.d entry whose name contains a '.'
# (the same property SUDOERS_STAGED relies on), so NEITHER is ever parsed as a live rule:
#   * SUDOERS_BACKUP_PATH   — the prior fragment's bytes, when one existed (the wildcard fragment
#                             on a parc machine). Rollback re-installs it.
#   * SUDOERS_ABSENT_MARKER — an empty marker, when NO prior fragment existed (clean install).
#                             Rollback removes the live fragment (revert to "no fragment").
# Exactly one is created at first install; both are left untouched on re-runs, so the recorded
# state always remains the ORIGINAL pre-helper one — never a later helper-only fragment that a
# re-bootstrap or cutover writes over it. Existence is decided AS ROOT (see ``_sudo_exists``) so a
# hardened (0750) sudoers.d can't make a present file look absent, and no privileged content read
# is needed for the rollback decision.
SUDOERS_BACKUP_PATH = f"{SUDOERS_DIR}/.asiai-inference.pre-bootstrap.bak"
SUDOERS_ABSENT_MARKER = f"{SUDOERS_DIR}/.asiai-inference.was-absent.marker"
# Staging name for capturing the backup (dotted -> sudo-ignored). Distinct from SUDOERS_STAGED,
# which restore reuses (a restore stages into the same "about-to-be-live fragment" slot install
# uses). Backup never runs concurrently with install.
SUDOERS_BACKUP_STAGED = f"{SUDOERS_DIR}/.asiai-inference.bak.tmp"


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
    """Validate then atomically install the sudoers fragment, root:wheel 0440.

    Returns the destination path. In dry-run mode prints the content and returns the
    destination without touching the filesystem.

    The fragment is root:wheel 0440 at every path sudo PARSES, never invoker-owned-then-
    chowned. All owner/mode work happens on a DOT-PREFIXED staged name that sudo ignores in
    ``sudoers.d``; only an atomic rename touches the final (parsed) name. ``cp`` run as root
    writes the staged file root-owned directly — unlike ``install(1)`` it leaves no
    non-dotted ``INS@`` temp that sudo would parse on a crash (cp's worst case is a partial
    DOTTED file sudo ignores and cleanup removes). This closes the TOCTOU where a process in
    the installer's session could rewrite an invoker-owned file between a ``mv`` and a later
    ``chown`` and freeze attacker content as a root-owned sudoers rule (full root). A final
    ``visudo -c`` validates the whole tree and fails loud if anything no longer parses, so a
    broken sudoers file can never silently lock out ``sudo``. The install is interactive
    (sudo password): in the helper-only model nothing here is NOPASSWD.
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
            "the first time (sudo password is needed to install the fragment into "
            f"{SUDOERS_PATH}). Run this command directly in Terminal.app:\n\n"
            "  cd ~/projets/asiai-inference-server && "
            ".venv/bin/aisctl bootstrap --install-sudoers\n\n"
            "After the sudoers fragment is installed, subsequent privileged "
            "operations (engine install/start/stop/unload/purge) are NOPASSWD "
            "and can run from non-TTY contexts."
        )

    validate_content(content)

    # I0 (defense-in-depth): refuse before any privileged write if /etc/sudoers.d's chain isn't
    # locked — a group/other-writable or non-root ancestor could let a non-root party swap the
    # fragment (root-equivalent). Deferred import: bootstrap imports sudoers (cycle otherwise).
    from ais_core import bootstrap

    bootstrap.assert_chain_locked(SUDOERS_DIR)

    with secure_staging_dir() as staging:
        tmp_path = staging / "asiai-inference.sudoers"
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.chmod(0o440)
        # Owner/mode work happens only on the DOT-PREFIXED staged name (sudo ignores it in
        # sudoers.d); the final parsed name appears via one atomic rename. cp (run as root)
        # writes the staged file root-owned — and unlike install(1) leaves no non-dotted INS@
        # temp that sudo could parse on a crash. chown/chmod are belt-and-suspenders on the
        # already-root, sudo-ignored staged file; visudo -c then fails loud if the resulting
        # tree no longer parses (a broken sudoers file must never silently lock out sudo).
        try:
            # Clear any stale/pre-positioned staged inode first: the staged name is a fixed
            # sibling, and cp (no -P) would follow a symlink left there (e.g. from a SIGKILL'd
            # prior run) and write through it. /etc/sudoers.d is root-only so a non-root party
            # can't plant one, but the stale-symlink case is real — rm -f closes both.
            subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_STAGED], check=True)
            subprocess.run(["sudo", "/bin/cp", str(tmp_path), SUDOERS_STAGED], check=True)
            subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", SUDOERS_STAGED], check=True)
            subprocess.run(["sudo", "/bin/chmod", "0440", SUDOERS_STAGED], check=True)
            subprocess.run(["sudo", "/bin/mv", SUDOERS_STAGED, SUDOERS_PATH], check=True)
            subprocess.run(["sudo", "/usr/sbin/visudo", "-c"], check=True)
        except subprocess.CalledProcessError as e:
            subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_STAGED], check=False)  # best-effort
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


def _sudo_exists(path: str) -> bool:
    """True iff PATH exists, decided AS ROOT so a 0750 sudoers.d (or a 0440 fragment a non-wheel
    invoker cannot stat) can never make a present file look absent — which would silently corrupt
    the rollback decision (restore vs remove). FAIL-CLOSED: only a clean "No such file" counts as
    absent; a sudo/authentication/other error raises rather than being read as "not there".

    ``ls -d -- <path>`` distinguishes the three states unambiguously: rc 0 (exists) / rc!=0 with
    "No such file or directory" on stderr (absent) / anything else (sudo failed, permission, …).
    """
    proc = subprocess.run(
        ["sudo", "/bin/ls", "-d", "--", path],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True
    if "No such file or directory" in (proc.stderr or ""):
        return False
    detail = (proc.stderr or "").strip() or f"ls rc={proc.returncode}"
    raise SudoersError(f"cannot determine existence of {path}: {detail}")


def backup_recorded() -> bool:
    """True iff the pre-bootstrap state has already been recorded (either marker present)."""
    return _sudo_exists(SUDOERS_BACKUP_PATH) or _sudo_exists(SUDOERS_ABSENT_MARKER)


def _fragment_is_helper_model(path: str) -> bool:
    """True iff the sudoers fragment at PATH already references the privileged helper (i.e. it IS
    our helper-only model, not a genuine pre-helper fragment). FAIL-CLOSED: grep rc 0 = match,
    rc 1 = no match, anything else (sudo/read error) raises rather than guessing.

    Used to refuse recording our OWN fragment as the "pre-bootstrap" baseline if the dotted markers
    were ever lost while the live fragment was already migrated — which would silently freeze the
    helper-only fragment as the rollback target and defeat FR8.
    """
    proc = subprocess.run(
        ["sudo", "/usr/bin/grep", "-qF", "--", PRIVILEGED_HELPER_PATH, path],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    raise SudoersError(
        f"cannot inspect {path} to tell the pre-bootstrap fragment from the helper model "
        f"(grep rc={proc.returncode}): {(proc.stderr or '').strip()}"
    )


def backup_existing_sudoers(*, dry_run: bool = False) -> str | None:
    """Record the pre-bootstrap state of SUDOERS_PATH ONCE so a rollback can restore it (FR8).

    Called from the bootstrap BEFORE the helper-only fragment overwrites whatever is there. First
    capture only: if either marker already exists the recorded state is LEFT UNTOUCHED, so a re-run
    (or a later cutover) never overwrites the original pre-helper backup with a helper-only
    fragment. Writes ``SUDOERS_BACKUP_PATH`` (the prior fragment's bytes) when one exists, else an
    empty ``SUDOERS_ABSENT_MARKER``. Both are dotted (sudo-ignored), root:wheel 0440, published
    TOCTOU-safe (dotted staged name cleared + chowned/chmod'd, then ONE atomic mv) — identically
    for both branches.

    Baseline-poisoning guard: if no marker exists yet BUT the live fragment already references the
    helper (so the dotted markers were lost while the model was live), refuse rather than freeze the
    helper-only fragment as the "pre-bootstrap" baseline — that would silently break FR8. The
    operator must re-establish the baseline from a known-good source.

    Returns the path written, or None if a backup already existed / in dry-run.
    """
    if dry_run:
        print(
            f"[dry-run] record pre-bootstrap state of {SUDOERS_PATH} ONCE -> "
            f"{SUDOERS_BACKUP_PATH} (or {SUDOERS_ABSENT_MARKER} if no prior fragment)"
        )
        return None

    if not sys.stdin.isatty():
        raise SudoersError(
            "recording the sudoers backup requires an interactive terminal (sudo password). "
            "Run `aisctl bootstrap --install` directly in Terminal.app."
        )

    if backup_recorded():
        return None  # the first capture is the authoritative pre-helper state — never clobber it.

    # I0 (defence-in-depth; the bootstrap already checked, cheap to re-assert before a root write).
    from ais_core import bootstrap

    bootstrap.assert_chain_locked(SUDOERS_DIR)

    prior_exists = _sudo_exists(SUDOERS_PATH)
    if prior_exists and _fragment_is_helper_model(SUDOERS_PATH):
        # No marker recorded, yet the live fragment is ALREADY the helper model: the baseline was
        # lost. Recording this fragment would make --rollback "restore" the helper-only model — a
        # silent FR8 defeat. Refuse and tell the operator to re-establish the baseline.
        raise SudoersError(
            f"refusing to record the baseline: {SUDOERS_PATH} already references the helper "
            f"({PRIVILEGED_HELPER_PATH}) but no pre-bootstrap marker exists — the rollback baseline "
            "was lost. Re-establish it from a known-good source (or restore the dotted markers) "
            "before re-running; recording the helper-only fragment as 'pre-bootstrap' would break "
            "rollback."
        )

    # Both branches publish via the same TOCTOU-safe pattern: clear a stale/symlinked staged inode,
    # write+lock owner/mode on the dotted staged name, then ONE atomic mv onto the dotted target.
    target = SUDOERS_BACKUP_PATH if prior_exists else SUDOERS_ABSENT_MARKER
    try:
        subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_BACKUP_STAGED], check=True)
        if prior_exists:
            # Copy the prior fragment's bytes (as root — it is 0440) into the staged name.
            subprocess.run(["sudo", "/bin/cp", SUDOERS_PATH, SUDOERS_BACKUP_STAGED], check=True)
        else:
            # No prior fragment: an empty staged marker so rollback reverts to "no fragment".
            subprocess.run(["sudo", "/usr/bin/touch", SUDOERS_BACKUP_STAGED], check=True)
        subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", SUDOERS_BACKUP_STAGED], check=True)
        subprocess.run(["sudo", "/bin/chmod", "0440", SUDOERS_BACKUP_STAGED], check=True)
        subprocess.run(["sudo", "/bin/mv", SUDOERS_BACKUP_STAGED, target], check=True)
        return target
    except subprocess.CalledProcessError as e:
        subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_BACKUP_STAGED], check=False)  # best-effort
        raise SudoersError(f"Failed to back up {SUDOERS_PATH}: {e}") from e


def restore_sudoers(*, dry_run: bool = False) -> str:
    """Restore SUDOERS_PATH to its recorded pre-bootstrap state (FR8). Returns a short description.

    ANTI-LOCKOUT (AC2): never publishes content that does not pass ``visudo -cf``, and re-validates
    the WHOLE tree with ``visudo -c`` afterwards — so a malformed restore can never lock ``%admin``
    out of password sudo. If the backup itself fails ``visudo -cf`` the live fragment is left
    untouched (the helper-only one is valid → sudo keeps working) and the error is surfaced.

    Three cases, decided by root-side existence checks (no privileged content read):
      * backup present  -> validate it, then atomically install it as the live fragment.
      * absent-marker    -> remove the live fragment (revert to no asiai fragment).
      * neither          -> REFUSE (the prior state was never recorded; never guess).

    The markers are INTENTIONALLY retained after a rollback (not deleted here): they record the
    original pre-bootstrap state, so a later ``--install`` round-trips to the same baseline. To
    fully un-asiai a node, remove the dotted markers deliberately — :func:`backup_existing_sudoers`
    refuses to re-record a helper-model fragment as the baseline, so a missing-marker re-install
    fails loud rather than silently freezing the wrong baseline.
    """
    if dry_run:
        print(
            f"[dry-run] restore {SUDOERS_PATH} from its recorded pre-bootstrap state: "
            f"re-install {SUDOERS_BACKUP_PATH} if present, else remove the live fragment if "
            f"{SUDOERS_ABSENT_MARKER} is present, else refuse; then visudo -c the whole tree"
        )
        return "dry-run"

    if not sys.stdin.isatty():
        raise SudoersError(
            "rollback requires an interactive terminal (sudo password). "
            "Run `aisctl bootstrap --rollback` directly in Terminal.app."
        )

    from ais_core import bootstrap

    bootstrap.assert_chain_locked(SUDOERS_DIR)

    if _sudo_exists(SUDOERS_BACKUP_PATH):
        try:
            subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_STAGED], check=True)
            subprocess.run(["sudo", "/bin/cp", SUDOERS_BACKUP_PATH, SUDOERS_STAGED], check=True)
            subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", SUDOERS_STAGED], check=True)
            subprocess.run(["sudo", "/bin/chmod", "0440", SUDOERS_STAGED], check=True)
            # Validate the staged backup BEFORE it becomes the live fragment: a backup that no
            # longer parses must NOT be published (it would lock out sudo). Leave the current
            # (valid) fragment in place and fail loud instead.
            check = subprocess.run(
                ["sudo", "/usr/sbin/visudo", "-cf", SUDOERS_STAGED],
                check=False,
                capture_output=True,
                text=True,
            )
            if check.returncode != 0:
                subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_STAGED], check=False)
                why = (check.stderr or check.stdout).strip()
                raise SudoersError(
                    "refusing to restore: the recorded backup no longer passes visudo -cf "
                    f"({why}); the current sudoers is left intact"
                )
            subprocess.run(["sudo", "/bin/mv", SUDOERS_STAGED, SUDOERS_PATH], check=True)
            subprocess.run(["sudo", "/usr/sbin/visudo", "-c"], check=True)
        except subprocess.CalledProcessError as e:
            subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_STAGED], check=False)
            raise SudoersError(f"Failed to restore {SUDOERS_PATH} from backup: {e}") from e
        return f"restored the pre-bootstrap fragment from {SUDOERS_BACKUP_PATH}"

    if _sudo_exists(SUDOERS_ABSENT_MARKER):
        try:
            subprocess.run(["sudo", "/bin/rm", "-f", SUDOERS_PATH], check=True)
            subprocess.run(["sudo", "/usr/sbin/visudo", "-c"], check=True)
        except subprocess.CalledProcessError as e:
            raise SudoersError(f"Failed to remove {SUDOERS_PATH} during rollback: {e}") from e
        return "removed the live fragment (no asiai fragment existed before bootstrap)"

    raise SudoersError(
        f"no recorded pre-bootstrap state ({SUDOERS_BACKUP_PATH} / {SUDOERS_ABSENT_MARKER} both "
        "absent): the full bootstrap was never run, or the markers were removed. Refusing to guess "
        f"— inspect {SUDOERS_PATH} manually."
    )
