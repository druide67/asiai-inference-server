"""Secure tempfile staging for privileged ``sudo /bin/mv`` operations.

The default tempfile directory ``/tmp`` is world-writable. Between the moment
``NamedTemporaryFile`` creates the staging file and the moment ``sudo /bin/mv``
moves it to a privileged destination, a local attacker can substitute the
file with a symlink, redirecting the privileged move's source.

To collapse that TOCTOU window, every privileged write goes through a
per-invocation directory ``/tmp/asiai-<random>/`` created by ``mkdtemp``,
which sets mode 0700 by default. The attacker has no write access to that
directory, so no symlink swap is possible. The sudoers rules are scoped to
``/tmp/asiai-*/...`` accordingly (see :mod:`ais_core.sudoers`).

Why ``/tmp`` and not ``~/.local/share``
---------------------------------------
The sudoers fragment matches absolute paths via wildcards. A path under
``~/.local/share`` would either need a ``/Users/*/.local/share/asiai/...``
wildcard (broad enough to be abused by any local account named to look
plausible) or per-user generation (fragile across machines with different
``$USER``). Staying under ``/tmp`` with a per-process random suffix gives
us a stable sudoers wildcard ``/tmp/asiai-*/...`` whose only writers are
the original ``aisctl`` process and root.
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
    until the privileged ``mv`` consumes them.
    """
    d = Path(tempfile.mkdtemp(prefix=_PREFIX, dir="/tmp"))
    try:
        yield d
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(d, ignore_errors=True)
