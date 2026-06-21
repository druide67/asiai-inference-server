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

import os
import stat
import subprocess
import sys
from pathlib import Path

from ais_core import plist, sudoers

# The helper's audit log + Standard*Path live here; no other ais_core module owns this path.
LOG_DIR = "/Library/Logs/asiai"

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
