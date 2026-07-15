"""Building blocks for the one-time privileged-helper bootstrap (story 2.1).

The bootstrap copies the helper to ``/Library/PrivilegedHelperTools`` (root:wheel 0755,
invariant #3) and installs the helper-only sudoers fragment, in a strict order, idempotently.
The check helpers (``assert_chain_locked`` — I0) are pure; ``install_helper`` performs the
privileged copy via ``sudo`` and so is interactive (password). Running it on the fleet is an
ops-PROD step performed with the operator present and the security reviewer on the gate; the
CLI orchestration (``aisctl bootstrap --install``) composes these in the strict order
(I0 chain check -> install helper -> install sudoers).
"""

from __future__ import annotations

import hashlib
import os
import plistlib
import pwd
import re
import stat
import subprocess
import sys
from pathlib import Path

from ais_core import plist, sudoers
from ais_core.io import secure_staging_dir

# The helper's audit log + Standard*Path live here; no other ais_core module owns this path.
LOG_DIR = "/Library/Logs/asiai"

# The helper's audit log (mirrors AUDIT_LOG in data/helpers/asiai_priv.py). Pre-created by the
# bootstrap 0640 root:admin so an operator can READ refusals without sudo (write stays
# root-only); the helper's hot path never chowns — group ownership is set here, once.
AUDIT_LOG_PATH = f"{LOG_DIR}/asiai-priv-audit.log"

# Directory chain whose permissions must be verified locked BEFORE any root write (I0). A
# group/other-writable, non-root, setuid/setgid or symlinked ancestor would let a non-root
# party swap or tamper with what root writes below it — making every downstream perms check
# theatre. Sourced from the canonical path constants so a future new root-write target cannot
# silently fall outside I0. Crucially this includes /etc/sudoers.d: the sudoers fragment is
# the single most root-equivalent thing AEP-01 writes.
LOCKED_CHAIN: tuple[str, ...] = (
    str(Path(sudoers.PRIVILEGED_HELPER_PATH).parent),  # /Library/PrivilegedHelperTools
    plist.LAUNCH_DAEMONS_DIR,  # /Library/LaunchDaemons
    LOG_DIR,  # /Library/Logs/asiai
    sudoers.SUDOERS_DIR,  # /etc/sudoers.d (root-equivalent: the sudoers fragment IS root)
)


class BootstrapError(RuntimeError):
    """Raised when a bootstrap precondition fails (e.g. the I0 dir-chain check)."""


def assert_chain_locked(path: str) -> None:
    """I0: every *existing* component of ``path``, from ``/`` down, must be **root:wheel**
    (uid 0, gid 0), a directory (ancestors), not a symlink, not group/other-writable, and not
    setuid/setgid — otherwise raise ``BootstrapError``. Sticky (``S_ISVTX``) is allowed:
    ``/Library/PrivilegedHelperTools`` legitimately carries it.

    The walk is component-by-component from the root so a writable/non-root/setgid *ancestor*
    (not just the leaf) is caught. A not-yet-existing leaf is allowed — the bootstrap creates
    it root-owned — as long as its existing ancestors are locked; the walk stops at the first
    missing component. ``lstat`` (not ``stat``) flags a symlinked component rather than
    following it.

    CONTRACT (load-bearing): the writer MUST act on the EXACT literal path checked here — no
    ``realpath``/``normpath``/``resolve`` and ideally an ``O_NOFOLLOW`` per-component (openat)
    walk — or the check and the write touch different inodes and this becomes theatre. ``..``
    is rejected outright to kill that literal-vs-canonical divergence class. The guarantee is
    point-in-time: it holds only because the chain is root-only once it passes (no non-root
    party can change it before the write); the orchestration must not widen a dir or create a
    leaf under a loose umask between this check and the write.
    """
    target = Path(path)
    if not target.is_absolute():
        raise BootstrapError(f"chain path must be absolute: {path!r}")
    if ".." in target.parts:
        raise BootstrapError(f"chain path must not contain '..': {path!r}")

    components: list[Path] = []
    cur = target
    while True:
        components.append(cur)
        if cur == cur.parent:  # reached the filesystem root "/"
            break
        cur = cur.parent

    ordered = list(reversed(components))  # "/" first, descending toward `path`
    for index, component in enumerate(ordered):
        try:
            st = os.lstat(component)
        except FileNotFoundError:
            return  # this component and everything below it is created by the bootstrap
        if stat.S_ISLNK(st.st_mode):
            # macOS ships /etc, /var, /tmp as ROOT-OWNED symlinks into /private. A symlink whose
            # PARENT was already verified root-owned + non-group/other-writable (walk order:
            # ancestors are checked before their children) cannot have been planted or swapped by
            # a non-root party, so a root-owned symlink is NOT a tamper vector. Accept it and walk
            # through (the next literal component resolves the link and is itself checked); reject
            # only a symlink a non-root party could control. (Symlink mode bits are advisory — the
            # target's perms govern access — so only ownership is checked here.)
            if st.st_uid != 0 or st.st_gid != 0:
                raise BootstrapError(
                    f"chain component is a non-root symlink: {component} "
                    f"(uid={st.st_uid} gid={st.st_gid})"
                )
            continue
        if st.st_uid != 0 or st.st_gid != 0:
            raise BootstrapError(
                f"chain component not root:wheel: {component} (uid={st.st_uid} gid={st.st_gid})"
            )
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise BootstrapError(
                f"chain component is group/other-writable: {component} (mode {st.st_mode:#o})"
            )
        if st.st_mode & (stat.S_ISUID | stat.S_ISGID):
            raise BootstrapError(
                f"chain component is setuid/setgid: {component} (mode {st.st_mode:#o})"
            )
        if index < len(ordered) - 1 and not stat.S_ISDIR(st.st_mode):
            raise BootstrapError(f"chain ancestor is not a directory: {component}")


def assert_fleet_chain_locked() -> None:
    """Run :func:`assert_chain_locked` over every path in ``LOCKED_CHAIN`` (I0, pre-bootstrap)."""
    for path in LOCKED_CHAIN:
        assert_chain_locked(path)


HELPER_STAGED = str(Path(sudoers.PRIVILEGED_HELPER_PATH).with_name(".asiai-priv.tmp"))


def _bundled_helper_path() -> Path:
    """Locate the bundled helper SOURCE (``data/helpers/asiai_priv.py``) in either layout.

    Wheel install copies ``data/helpers/`` to ``ais_core/data/helpers/`` via the
    ``force-include`` in pyproject; an editable install keeps it at the repo root. This is the
    source we copy FROM — the executed copy is the root-owned one at
    ``PRIVILEGED_HELPER_PATH`` (invariant #3).
    """
    here = Path(__file__).resolve().parent  # src/ais_core
    bundled = here / "data" / "helpers" / "asiai_priv.py"
    if bundled.is_file():
        return bundled
    editable = here.parent.parent / "data" / "helpers" / "asiai_priv.py"  # src/ais_core -> repo
    if editable.is_file():
        return editable
    raise BootstrapError(
        f"asiai_priv.py not found under {bundled} or {editable}; install layout is broken"
    )


def install_helper(*, dry_run: bool = False) -> str:
    """Install the bundled helper to ``PRIVILEGED_HELPER_PATH`` as root:wheel 0755 (FR1, I0).

    **Invariant #3**: the EXECUTED helper is this root-owned copy under
    ``/Library/PrivilegedHelperTools`` — never the admin-writable site-packages source.
    :func:`ais_core.privhelper.run` invokes exactly this path, and the sudoers fragment grants
    NOPASSWD on exactly this path; ``install_helper`` is what puts the reviewed bytes there.

    Order/safety:
      * I0 — :func:`assert_chain_locked` on the destination refuses the install if any chain
        component is group/other-writable, non-root, a symlink, or setuid/setgid, BEFORE any
        write (so a tampered parent can't redirect the root copy).
      * the destination dir is ``mkdir -p``'d (idempotent; a freshly created dir is root:wheel
        0755 under root's umask — locked); its perms are otherwise left untouched so the
        standard ``drwxr-xr-t`` sticky bit survives.
      * the helper is ``cp``'d as root to a dot-prefixed staged name in the destination dir,
        ``chown root:wheel`` + ``chmod 0755`` on the staged name, then ONE atomic ``mv`` onto
        the final name — no window where a half-copied or wrong-perm helper sits at the live
        path. Idempotent: re-running overwrites in place.

    Interactive (sudo password); refused on a non-TTY with clear instructions, like
    :func:`ais_core.sudoers.install_sudoers`.
    """
    src = _bundled_helper_path()
    dest = sudoers.PRIVILEGED_HELPER_PATH
    dest_dir = str(Path(dest).parent)

    if dry_run:
        print(f"[dry-run] install helper {src} -> {dest} (root:wheel 0755), I0-checked")
        return dest

    if not sys.stdin.isatty():
        raise BootstrapError(
            "aisctl bootstrap --install requires an interactive terminal the first time "
            "(sudo password is needed to copy the helper into "
            f"{dest}). Run it directly in Terminal.app."
        )

    # I0: refuse before any privileged write if the destination chain is not locked.
    assert_chain_locked(dest)

    try:
        # mkdir -p is idempotent and only creates missing components (root:wheel 0755 under
        # root's umask); we do NOT chmod an existing dir, to preserve its sticky bit.
        subprocess.run(["sudo", "/bin/mkdir", "-p", dest_dir], check=True)
        # Re-assert AFTER mkdir: assert_chain_locked above returns early at a MISSING dir, so a
        # just-created dest_dir was never inspected. A non-default root umask could have made it
        # group/other-writable — re-checking turns "freshly created dir is locked" from an
        # assumption into an enforced invariant, before anything is written into it.
        assert_chain_locked(dest)
        # Clear any stale/pre-positioned staged inode FIRST (in the now-verified-locked dir):
        # the staged name is a fixed sibling NOT covered by the I0 leaf check, and cp/chmod would
        # otherwise follow a symlink left there (by a SIGKILL'd prior run, or — only if the dir
        # were ever writable — a planted one) and write/chmod through it.
        subprocess.run(["sudo", "/bin/rm", "-f", HELPER_STAGED], check=True)
        subprocess.run(["sudo", "/bin/cp", str(src), HELPER_STAGED], check=True)
        subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", HELPER_STAGED], check=True)
        subprocess.run(["sudo", "/bin/chmod", "0755", HELPER_STAGED], check=True)
        subprocess.run(["sudo", "/bin/mv", HELPER_STAGED, dest], check=True)
    except subprocess.CalledProcessError as e:
        subprocess.run(["sudo", "/bin/rm", "-f", HELPER_STAGED], check=False)  # best-effort
        raise BootstrapError(f"Failed to install helper at {dest}: {e}") from e
    return dest


# Integrity sidecar (NFR11). A root-owned SHA-256 of the installed helper, written next to it.
# This is integrity + audit, NOT execution control — that is invariant #3 (only the root-owned
# copy is executed) + I2 (the daemon runs non-root). Honestly: a root-owned hash beside a
# root-owned helper adds no resistance against a ROOT attacker (it would replace both); its job
# is to detect corruption / a wrong version and to record what was installed, verifiable at any
# time via ``aisctl bootstrap --verify`` (and by ``shasum -a 256 -c`` — the file is in that
# format). Stronger provenance (a detached CMS signature over the hash) is a future upgrade.
HELPER_SHA256_PATH = sudoers.PRIVILEGED_HELPER_PATH + ".sha256"
SIDECAR_STAGED = str(Path(HELPER_SHA256_PATH).with_name(".asiai-priv.sha256.tmp"))


def _sha256_hex(path: str) -> str:
    """Stream a file through SHA-256 and return its hex digest (constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_helper_signature(*, dry_run: bool = False) -> str:
    """NFR11: hash the INSTALLED helper and write a root:wheel 0644 ``<helper>.sha256`` sidecar.

    Hashes the on-disk installed copy (not the source) so the sidecar records what is actually
    executed. Same TOCTOU-safe publish as the helper itself (I0 on the dir, dotted staged name
    cleared + written + chowned/chmod'd, then atomic mv). ``shasum -a 256 -c`` compatible.
    Must run AFTER :func:`install_helper` (it reads the installed helper).
    """
    dest = sudoers.PRIVILEGED_HELPER_PATH
    if dry_run:
        print(f"[dry-run] write SHA-256 sidecar {HELPER_SHA256_PATH} for {dest}")
        return HELPER_SHA256_PATH

    if not sys.stdin.isatty():
        raise BootstrapError(
            "aisctl bootstrap --install requires an interactive terminal (sudo password)."
        )
    if not Path(dest).is_file():
        raise BootstrapError(f"cannot sign: helper not installed at {dest} (run install first)")

    # I0 on the sidecar's chain (same dir as the helper) before the privileged write.
    assert_chain_locked(HELPER_SHA256_PATH)
    content = f"{_sha256_hex(dest)}  {Path(dest).name}\n"  # shasum -a 256 -c format
    with secure_staging_dir() as staging:
        tmp = staging / "asiai-priv.sha256"
        tmp.write_text(content, encoding="utf-8")
        try:
            subprocess.run(["sudo", "/bin/rm", "-f", SIDECAR_STAGED], check=True)
            subprocess.run(["sudo", "/bin/cp", str(tmp), SIDECAR_STAGED], check=True)
            subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", SIDECAR_STAGED], check=True)
            subprocess.run(["sudo", "/bin/chmod", "0644", SIDECAR_STAGED], check=True)
            subprocess.run(["sudo", "/bin/mv", SIDECAR_STAGED, HELPER_SHA256_PATH], check=True)
        except subprocess.CalledProcessError as e:
            subprocess.run(["sudo", "/bin/rm", "-f", SIDECAR_STAGED], check=False)
            raise BootstrapError(f"Failed to write signature {HELPER_SHA256_PATH}: {e}") from e
    return HELPER_SHA256_PATH


def verify_helper() -> bool:
    """NFR11: True iff the installed helper's SHA-256 matches its sidecar.

    Raises :class:`BootstrapError` if the helper or its sidecar is missing or the sidecar is
    malformed — those are "cannot verify" states, distinct from a clean mismatch (returns False).
    """
    dest = sudoers.PRIVILEGED_HELPER_PATH
    if not Path(dest).is_file():
        raise BootstrapError(f"helper not installed at {dest}")
    try:
        recorded = Path(HELPER_SHA256_PATH).read_text(encoding="utf-8").split()
    except FileNotFoundError:
        raise BootstrapError(f"no signature sidecar at {HELPER_SHA256_PATH}") from None
    token = recorded[0].lower() if recorded else ""
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise BootstrapError(f"malformed signature sidecar at {HELPER_SHA256_PATH}")
    return _sha256_hex(dest) == token


def remove_helper(*, dry_run: bool = False) -> list[str]:
    """Remove the installed helper and its SHA-256 sidecar (FR8 rollback). Returns the paths.

    Idempotent: ``rm -f`` makes removing an already-absent helper a no-op. Interactive (raw sudo
    password — removing a root file under ``/Library/PrivilegedHelperTools`` is NOT in the NOPASSWD
    helper, by design). I0-checked defensively before the privileged ``rm`` (a missing leaf is fine
    — :func:`assert_chain_locked` validates the parent chain and returns at the absent helper).
    """
    dest = sudoers.PRIVILEGED_HELPER_PATH
    targets = [dest, HELPER_SHA256_PATH]
    if dry_run:
        print(f"[dry-run] remove helper {dest} and signature {HELPER_SHA256_PATH}")
        return targets
    if not sys.stdin.isatty():
        raise BootstrapError(
            "aisctl bootstrap --rollback requires an interactive terminal "
            f"(sudo password is needed to remove {dest})."
        )
    assert_chain_locked(dest)
    try:
        subprocess.run(["sudo", "/bin/rm", "-f", *targets], check=True)
    except subprocess.CalledProcessError as e:
        raise BootstrapError(f"Failed to remove the helper at {dest}: {e}") from e
    return targets


def ensure_log_dir(*, dry_run: bool = False) -> str:
    """Ensure ``/Library/Logs/asiai`` exists root:wheel 0755 (audit finding #3b).

    The helper deliberately does not create its own log dir (a root process
    mkdir'ing on demand would blur the bootstrap/runtime split); a host where it
    is missing used to fail ``install-daemon`` opaquely. The bootstrap owns it:
    I0-check the chain, ``mkdir -p`` (idempotent), pin root:wheel 0755, re-assert.
    """
    if dry_run:
        print(f"[dry-run] ensure log dir {LOG_DIR} (root:wheel 0755)")
        return LOG_DIR
    if not sys.stdin.isatty():
        raise BootstrapError(
            "aisctl bootstrap --install requires an interactive terminal (sudo password)."
        )
    assert_chain_locked(LOG_DIR)
    try:
        subprocess.run(["sudo", "/bin/mkdir", "-p", LOG_DIR], check=True)
        subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", LOG_DIR], check=True)
        subprocess.run(["sudo", "/bin/chmod", "0755", LOG_DIR], check=True)
    except subprocess.CalledProcessError as e:
        raise BootstrapError(f"Failed to create log dir {LOG_DIR}: {e}") from e
    # Re-assert AFTER the mkdir: the pre-check returns early at a missing dir, so a
    # freshly created one was never inspected (same pattern as install_helper).
    assert_chain_locked(LOG_DIR)
    return LOG_DIR


def installed_daemon_log_specs() -> list[tuple[Path, str]]:
    """Discover ``(log_path, run_as_user)`` pairs for every installed com.asiai.* daemon.

    Reads ``StandardOutPath``/``StandardErrorPath`` and ``UserName`` from each
    ``/Library/LaunchDaemons/com.asiai.*.plist``. Only paths directly under
    ``LOG_DIR`` are returned: this function feeds ``sudo touch``/``chown``, so a
    plist pointing anywhere else is deliberately left alone.
    """
    specs: list[tuple[Path, str]] = []
    daemons_dir = Path(plist.LAUNCH_DAEMONS_DIR)
    if not daemons_dir.is_dir():
        return specs
    for pl in sorted(daemons_dir.glob("com.asiai.*.plist")):
        try:
            with pl.open("rb") as fh:
                data = plistlib.load(fh)
        except (OSError, plistlib.InvalidFileException):
            continue  # unreadable or malformed: not ours to fix here
        user = data.get("UserName")
        if not isinstance(user, str) or not user:
            continue
        for key in ("StandardOutPath", "StandardErrorPath"):
            raw = data.get(key)
            if isinstance(raw, str) and Path(raw).parent == Path(LOG_DIR):
                specs.append((Path(raw), user))
    return specs


def missing_daemon_log_files() -> list[tuple[Path, str]]:
    """Read-only check: the subset of ``installed_daemon_log_specs`` that does not exist."""
    return [(p, u) for p, u in installed_daemon_log_specs() if not p.exists()]


def ensure_daemon_log_files(*, dry_run: bool = False) -> list[str]:
    """Pre-create the ``Standard*Path`` log files of installed daemons.

    launchd opens ``Standard*Path`` with the JOB's uid: a daemon running as a
    regular user cannot create its own log file inside the root-owned 0755
    ``LOG_DIR`` and dies with ``EX_CONFIG`` before exec — silently, since the
    very file that would tell you is the one it could not create. The helper
    pre-creates these leaves at install-daemon time, but a macOS system update
    can prune ``LOG_DIR`` entirely (observed with the sealed-system-volume
    post-update migration), leaving every installed daemon unable to respawn
    at the post-update boot. This repairs them all in one pass.

    Files are created root-side then chown'd to the plist's run-as user and
    its primary group, mode 0640 — matching what install-daemon produces.
    """
    missing = missing_daemon_log_files()
    if dry_run:
        for p, u in missing:
            print(f"[dry-run] create {p} ({u}, 0640)")
        return [str(p) for p, _ in missing]
    if not missing:
        return []
    if not sys.stdin.isatty():
        raise BootstrapError(
            "aisctl bootstrap --logs-only requires an interactive terminal (sudo password)."
        )
    created: list[str] = []
    for p, user in missing:
        try:
            gid = pwd.getpwnam(user).pw_gid
        except KeyError:
            print(f"skipping {p}: run-as user {user!r} does not exist", file=sys.stderr)
            continue
        try:
            subprocess.run(["sudo", "/usr/bin/touch", str(p)], check=True)
            subprocess.run(["sudo", "/usr/sbin/chown", f"{user}:{gid}", str(p)], check=True)
            subprocess.run(["sudo", "/bin/chmod", "0640", str(p)], check=True)
        except subprocess.CalledProcessError as e:
            raise BootstrapError(f"Failed to pre-create log file {p}: {e}") from e
        created.append(str(p))
    return created


def ensure_audit_log(*, dry_run: bool = False) -> str:
    """Pre-create/normalize the helper's audit log **0640 root:admin** (audit finding #2).

    Write stays root-only (no group/other write bit); the admin GROUP gains read so an
    operator can diagnose helper refusals without sudo. Group ownership is set HERE, once —
    the helper's ``_audit`` hot path deliberately never chowns/getgrnams (O_APPEND
    atomicity, no lookups), and its degraded ``O_CREAT`` re-create falls back to
    root:wheel 0640 until the next bootstrap run re-normalizes it.

    Idempotent and never truncates: ``touch`` only updates mtime on an existing log.
    Must run AFTER :func:`ensure_log_dir` (the chain check requires the dir).
    """
    if dry_run:
        print(f"[dry-run] ensure audit log {AUDIT_LOG_PATH} (root:admin 0640)")
        return AUDIT_LOG_PATH
    if not sys.stdin.isatty():
        raise BootstrapError(
            "aisctl bootstrap --install requires an interactive terminal (sudo password)."
        )
    # I0: verify the PARENT chain (LOG_DIR), NOT the audit-log leaf. The leaf is
    # deliberately root:ADMIN (gid 80, so an operator reads refusals without sudo) —
    # checking it with assert_chain_locked, which requires root:wheel (gid 0) on every
    # component, rejected the very group ownership this function sets. On a FRESH host the
    # leaf is absent so the check returned early and install passed; on a RE-install over an
    # existing audit log it failed, making `bootstrap --install` non-idempotent and breaking
    # the rollback→reinstall recovery path (caught by the M5 cutover rehearsal, 2026-07-03).
    # LOG_DIR is created + verified root:wheel non-group/other-writable by ensure_log_dir
    # (which runs first), so a non-root party cannot plant or swap the leaf inside it — the
    # parent-chain check is the real, sufficient guard; the leaf's admin group is by design.
    assert_chain_locked(LOG_DIR)
    try:
        subprocess.run(["sudo", "/usr/bin/touch", AUDIT_LOG_PATH], check=True)
        subprocess.run(["sudo", "/usr/sbin/chown", "root:admin", AUDIT_LOG_PATH], check=True)
        subprocess.run(["sudo", "/bin/chmod", "0640", AUDIT_LOG_PATH], check=True)
    except subprocess.CalledProcessError as e:
        raise BootstrapError(f"Failed to prepare audit log {AUDIT_LOG_PATH}: {e}") from e
    return AUDIT_LOG_PATH


# Dedicated daemon account (NFR12, opt-in via --dedicated-user). A hidden, non-login, non-admin
# macOS role account that engine daemons run as, confining a network RCE in an engine to a
# powerless uid (the operator is admin; engines should not be). uid in the sanctioned role range
# [450, 499] => < 500, satisfying NFR12.
DEDICATED_USER = "_aisrv"
_ROLE_UID_MIN = 450
_ROLE_UID_MAX = 499
_DSCL = "/usr/bin/dscl"
_SYSADMINCTL = "/usr/sbin/sysadminctl"
_DSEDITGROUP = "/usr/sbin/dseditgroup"
# `dseditgroup -o checkmember` exit codes (verified on macOS 26): 0 = member, 67 = confirmed
# not-a-member. ANY OTHER code (e.g. 64 = lookup error) is INDETERMINATE -> fail closed.
_DSEDITGROUP_NOT_MEMBER = 67
# Account name must be a safe directory-services record name (no '/', '..', whitespace). Only
# reachable via a programmatic caller — the CLI flag is store_true and always uses DEDICATED_USER
# — but validated so a future change that wires it to user input can't path-traverse dscl.
_ACCOUNT_NAME_RE = re.compile(r"_?[a-z][a-z0-9_-]*")


def _run_ro(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a read-only directory query with a bounded timeout; a hang becomes a clean
    BootstrapError (caught by the bootstrap's exit-2 path) instead of blocking forever."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise BootstrapError(f"directory query timed out: {' '.join(argv)}") from e


def _account_uid(name: str) -> int | None:
    """The account's numeric UniqueID, or None if it does not exist. dscl read needs no sudo."""
    proc = _run_ro([_DSCL, ".", "-read", f"/Users/{name}", "UniqueID"])
    if proc.returncode != 0:
        return None
    parts = proc.stdout.split()
    return int(parts[1]) if len(parts) >= 2 and parts[1].lstrip("-").isdigit() else None


def _dscl_value(name: str, key: str) -> str | None:
    """Last whitespace token of ``dscl -read /Users/<name> <key>`` (the attribute value), or None.

    Works for single-token values (UserShell, NFSHomeDirectory) and macOS's
    ``dsAttrTypeNative:IsHidden: 1`` form alike (the value is always the trailing token).
    """
    proc = _run_ro([_DSCL, ".", "-read", f"/Users/{name}", key])
    if proc.returncode != 0:
        return None
    tokens = proc.stdout.split()
    return tokens[-1] if tokens else None


def _has_auth_authority(name: str) -> bool:
    """True iff the account has an AuthenticationAuthority key (i.e. a usable login/password).

    A ``sysadminctl -roleAccount`` has NONE (``No such key``), which is what we require: the
    daemon account must not be loginable.

    CRITICAL — ``dscl -read`` returns rc 0 whether the key is PRESENT or ABSENT (verified on
    macOS 26: absent -> rc 0, empty stdout, ``No such key: AuthenticationAuthority`` on STDERR;
    present -> rc 0, the value on stdout). So the return code is NOT a present/absent signal — the
    decision must be made on the OUTPUT. FAIL-CLOSED: ``No such key`` -> absent; a non-empty value
    -> present; anything else (a DirectoryServices error, or rc 0 with no value and no
    ``No such key``) is indeterminate -> raise rather than assume the account is non-loginable.
    """
    proc = _run_ro([_DSCL, ".", "-read", f"/Users/{name}", "AuthenticationAuthority"])
    if "No such key" in (proc.stdout or "") or "No such key" in (proc.stderr or ""):
        return False  # key absent -> no login authority (the role-account case)
    if proc.returncode == 0 and (proc.stdout or "").strip():
        return True  # key present with a value -> a usable login authority
    raise BootstrapError(
        f"cannot read AuthenticationAuthority of {name!r} "
        f"(rc={proc.returncode}, no value and no 'No such key'); refusing"
    )


def _in_admin_group(name: str) -> bool:
    """True iff ``name`` is in the admin group. FAIL-CLOSED: an INDETERMINATE result (any rc that
    is neither 0=member nor 67=confirmed-not-member — e.g. a DirectoryServices error) raises
    ``BootstrapError`` so callers refuse rather than assume non-admin on an admin-membership query
    that did not actually answer (the most security-relevant predicate of the safe-fail gate)."""
    rc = _run_ro([_DSEDITGROUP, "-o", "checkmember", "-m", name, "admin"]).returncode
    if rc == 0:
        return True
    if rc == _DSEDITGROUP_NOT_MEMBER:
        return False
    raise BootstrapError(
        f"cannot determine admin membership of {name!r} (dseditgroup rc={rc}); refusing to assume"
    )


def _assert_role_account(name: str) -> None:
    """Assert ``name`` has the full NFR12 role-account shape; raise ``BootstrapError`` on ANY
    mismatch (never auto-fix — the operator decides). Used BOTH right after creation (catch a
    partial ``sysadminctl``) AND before adopting a pre-existing account (so the idempotent no-op
    can't adopt a uid-conforming but login-capable ``_aisrv``)."""
    shell = _dscl_value(name, "UserShell")
    if shell != "/usr/bin/false":
        raise BootstrapError(f"{name}: UserShell={shell!r}, expected /usr/bin/false (non-login)")
    home = _dscl_value(name, "NFSHomeDirectory")
    if home not in ("/var/empty", "/dev/null"):
        raise BootstrapError(f"{name}: NFSHomeDirectory={home!r}, expected /var/empty")
    if _dscl_value(name, "IsHidden") != "1":
        raise BootstrapError(f"{name}: IsHidden not 1 (must be hidden from the login screen)")
    if _has_auth_authority(name):
        raise BootstrapError(f"{name}: has an AuthenticationAuthority (a login) — must be absent")
    if _in_admin_group(name):  # fail-closed: an indeterminate membership raises, not adopts
        raise BootstrapError(f"{name}: is in the admin group — the daemon account must not be")


def _free_role_uid() -> int:
    """First unused uid in [450, 499] (sysadminctl -roleAccount requires an explicit uid there)."""
    proc = _run_ro([_DSCL, ".", "-list", "/Users", "UniqueID"])
    if proc.returncode != 0:
        raise BootstrapError(f"cannot enumerate uids (dscl -list rc={proc.returncode})")
    taken: set[int] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            taken.add(int(parts[1]))
    for uid in range(_ROLE_UID_MIN, _ROLE_UID_MAX + 1):
        if uid not in taken:
            return uid
    raise BootstrapError(f"no free uid in [{_ROLE_UID_MIN}, {_ROLE_UID_MAX}] for {DEDICATED_USER}")


def create_dedicated_user(name: str = DEDICATED_USER, *, dry_run: bool = False) -> dict:
    """NFR12: ensure a hidden, non-login, non-admin role account (uid 450-499) for engine daemons.

    ``sysadminctl -addUser <name> -UID <uid> -roleAccount`` (the uid is REQUIRED in 450-499 —
    verified on macOS 26) sets ``UserShell=/usr/bin/false``, ``NFSHomeDirectory=/var/empty``,
    ``IsHidden=1``, group ``staff``, and NO ``AuthenticationAuthority`` by default — exactly
    NFR12, no extra dscl steps. :func:`_assert_role_account` re-reads and ENFORCES that full shape
    (not just the uid) both after creation and before adopting a pre-existing account.

    Idempotent + safe-fail: an existing account that already has the role-account shape is a
    no-op; one that does NOT (uid 0/>=500, admin, wrong shell/home/hidden, or any login authority)
    is REFUSED, never modified — the operator decides. Reversible via ``sysadminctl -deleteUser``.
    Privileged + interactive (raw sudo password; sysadminctl is not in the NOPASSWD helper).
    """
    if not _ACCOUNT_NAME_RE.fullmatch(name):
        raise BootstrapError(f"invalid account name: {name!r}")

    if dry_run:
        print(
            f"[dry-run] ensure role account {name!r} "
            f"(uid {_ROLE_UID_MIN}-{_ROLE_UID_MAX}, hidden, /usr/bin/false, no password, non-admin)"
        )
        return {"user": name, "created": None, "dry_run": True}

    if not sys.stdin.isatty():
        raise BootstrapError(
            "aisctl bootstrap --install --dedicated-user requires an interactive terminal "
            "(sudo password is needed to create the role account)."
        )

    existing = _account_uid(name)
    if existing is not None:
        if existing == 0 or existing >= 500:
            raise BootstrapError(
                f"refusing to reuse existing account {name!r}: uid={existing} is outside the "
                f"role range [{_ROLE_UID_MIN}, {_ROLE_UID_MAX}] — remove it or choose another name."
            )
        _assert_role_account(name)  # full NFR12 shape (incl. non-admin, no login) or refuse
        return {"user": name, "created": False, "uid": existing}  # idempotent no-op

    uid = _free_role_uid()
    try:
        subprocess.run(
            ["sudo", _SYSADMINCTL, "-addUser", name, "-UID", str(uid), "-roleAccount"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise BootstrapError(
            f"sysadminctl failed to create {name} (uid {uid}): {(e.stderr or '').strip() or e}"
        ) from e
    # sysadminctl can exit 0 on partial failures — verify the uid AND the full role-account shape.
    if _account_uid(name) != uid:
        raise BootstrapError(f"{name} was not created as expected (uid {uid})")
    _assert_role_account(name)
    return {"user": name, "created": True, "uid": uid}
