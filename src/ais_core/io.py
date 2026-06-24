"""Secure tempfile staging for privileged operations that read an invoker-staged source.

A privileged step run as root (``sudo /usr/bin/install`` for the sudoers fragment,
``sudo /bin/mv`` for a plist) reads a source file the non-root invoker wrote. If that
source lived directly in world-writable ``/tmp``, a local attacker could symlink-swap it
between the write and the moment root reads it, redirecting what root copies/moves.

To collapse that window, every such write goes through a per-invocation directory
``/tmp/asiai-<random>/`` created by ``mkdtemp`` (mode 0700, owned by the invoker). No other
local user can write there, so the source cannot be swapped before root consumes it.

Historical note
---------------
An earlier design also relied on a ``/tmp/asiai-*`` *sudoers wildcard* so the privileged
move was NOPASSWD. The helper-only sudoers model (AEP-01) removed every wildcard rule — the
staging dir's sole remaining job is the 0700 anti-swap protection above, and the privileged
step is now an interactive (password) ``sudo``. Do NOT reintroduce a ``/tmp/asiai-*`` sudoers
rule.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

_PREFIX = "asiai-"


@contextlib.contextmanager
def secure_staging_dir() -> Iterator[Path]:
    """Yield a per-invocation 0700 directory under ``/tmp``.

    The directory is created with ``mkdtemp`` (mode 0700, ownership = caller)
    and recursively removed on context exit, regardless of whether the body
    raised. Files written inside are protected from local-user TOCTOU swaps
    until the privileged step (``install``/``mv``) run as root reads them.
    """
    d = Path(tempfile.mkdtemp(prefix=_PREFIX, dir="/tmp"))
    try:
        yield d
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(d, ignore_errors=True)
