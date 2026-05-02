"""Tests for ais_core.io.secure_staging_dir — H1 fix (TOCTOU on /tmp tempfiles)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from ais_core.io import secure_staging_dir


def test_staging_dir_exists_during_context() -> None:
    with secure_staging_dir() as d:
        assert d.is_dir()


def test_staging_dir_removed_on_exit() -> None:
    with secure_staging_dir() as d:
        captured = d
        (d / "x").write_text("hello")
    assert not captured.exists()


def test_staging_dir_removed_even_when_body_raises() -> None:
    captured: Path | None = None
    try:
        with secure_staging_dir() as d:
            captured = d
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert captured is not None
    assert not captured.exists()


def test_staging_dir_is_0700() -> None:
    """The whole point of H1 — directory must not be readable/writable by others."""
    with secure_staging_dir() as d:
        mode = stat.S_IMODE(d.stat().st_mode)
    assert mode == 0o700, f"expected 0o700, got {oct(mode)}"


def test_staging_dir_is_owned_by_caller() -> None:
    with secure_staging_dir() as d:
        owner_uid = d.stat().st_uid
    assert owner_uid == os.getuid()


def test_staging_dir_under_tmp_with_asiai_prefix() -> None:
    """Sudoers wildcard /tmp/asiai-*/ depends on this path layout."""
    with secure_staging_dir() as d:
        assert str(d).startswith("/tmp/asiai-"), f"unexpected path: {d}"


def test_staging_dirs_are_unique_per_invocation() -> None:
    with secure_staging_dir() as a, secure_staging_dir() as b:
        assert a != b
        assert a.exists() and b.exists()


def test_files_inside_staging_dir_inherit_protection() -> None:
    """A file created inside is unreachable by other users because the dir is 0700."""
    with secure_staging_dir() as d:
        f = d / "secret.plist"
        f.write_text("xml")
        # File mode is whatever the caller sets, but the directory wraps it.
        assert f.parent.stat().st_mode & 0o077 == 0
