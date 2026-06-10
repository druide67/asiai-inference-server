"""Shared fixtures.

Install records (:mod:`ais_core.install_state`) default to the real XDG
state directory; every test runs against an isolated throwaway tree so the
suite never reads or writes the developer's actual install records.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_install_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "_ais_state"
    monkeypatch.setenv("ASIAI_USER_STATE_DIR", str(state))
    return state
