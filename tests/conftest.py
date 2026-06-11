"""Shared fixtures.

Install records (:mod:`ais_core.install_state`) and user manifests/presets
(:mod:`ais_core.manifest`) default to the real XDG trees; every test runs
against isolated throwaway directories so the suite never reads or writes
the developer's actual install records or ``~/.config`` overrides. Tests
that need a populated user-config tree (test_manifest_user_config.py)
re-point ``ASIAI_USER_CONFIG_DIR`` themselves; the bundled-manifest
fallback stays active either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_install_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "_ais_state"
    monkeypatch.setenv("ASIAI_USER_STATE_DIR", str(state))
    return state


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "_ais_config"
    monkeypatch.setenv("ASIAI_USER_CONFIG_DIR", str(config))
    return config
