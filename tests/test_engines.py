"""Tests for ais_engines drivers — unload-with-fallback semantics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ais_core.manifest import load_manifest
from ais_engines.base import EngineDriver, RestartOnlyDriver, UnloadOutcome
from ais_engines.llamacpp import LlamaCppDriver
from ais_engines.llamacpp_aux import LlamaCppAuxDriver
from ais_engines.lmstudio import LMStudioDriver
from ais_engines.ollama import OllamaDriver
from ais_engines.omlx import OmlxDriver
from ais_engines.turboquant import TurboquantDriver

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEngineWithUnload:
    """Stand-in for asiai's OllamaEngine / LMStudioEngine."""

    def __init__(self, *, unload_returns: bool = True, raises: Exception | None = None):
        self._unload_returns = unload_returns
        self._raises = raises
        self.calls: list[str] = []

    def unload_model(self, model: str) -> bool:
        self.calls.append(model)
        if self._raises is not None:
            raise self._raises
        return self._unload_returns

    def list_running(self) -> list[object]:
        return [type("M", (), {"name": "fake-model"})()]


class _FakeEngineNoUnload:
    """Stand-in for asiai's OmlxEngine — has no unload_model attribute."""

    def list_running(self) -> list[object]:
        return []


# ---------------------------------------------------------------------------
# EngineDriver — happy path: native API returns True
# ---------------------------------------------------------------------------


def test_native_unload_success_skips_restart() -> None:
    m = load_manifest("ollama")
    fake = _FakeEngineWithUnload(unload_returns=True)
    driver = EngineDriver(m, fake)

    with patch("ais_core.lifecycle.restart") as mock_restart:
        result = driver.unload("llama3.2")

    assert mock_restart.call_count == 0
    assert result == UnloadOutcome(
        engine="ollama",
        model="llama3.2",
        method="api",
        success=True,
        detail="unloaded llama3.2 via native API",
    )
    assert fake.calls == ["llama3.2"]


def test_native_unload_returns_false_falls_back_to_restart() -> None:
    m = load_manifest("ollama")
    fake = _FakeEngineWithUnload(unload_returns=False)
    driver = EngineDriver(m, fake)

    with patch("ais_core.lifecycle.restart") as mock_restart:
        result = driver.unload("llama3.2")

    assert mock_restart.call_count == 1
    assert result.method == "restart"
    assert result.success is True


def test_native_unload_raises_falls_back_to_restart() -> None:
    """A driver that throws on the API path must NOT crash unload()."""
    m = load_manifest("ollama")
    fake = _FakeEngineWithUnload(raises=ConnectionError("boom"))
    driver = EngineDriver(m, fake)

    with patch("ais_core.lifecycle.restart"):
        result = driver.unload("llama3.2")

    assert result.method == "error"
    assert result.success is False
    assert "boom" in result.detail


def test_unload_with_no_engine_falls_back_to_restart() -> None:
    """Driver instantiated without an asiai engine still works (restart-only)."""
    m = load_manifest("ollama")
    driver = EngineDriver(m, asiai_engine=None)

    with patch("ais_core.lifecycle.restart") as mock_restart:
        result = driver.unload("llama3.2")

    assert mock_restart.call_count == 1
    assert result.method == "restart"


def test_unload_with_model_none_forces_restart() -> None:
    """``unload(None)`` is the 'unload everything' shortcut: skip the API."""
    m = load_manifest("ollama")
    fake = _FakeEngineWithUnload(unload_returns=True)
    driver = EngineDriver(m, fake)

    with patch("ais_core.lifecycle.restart") as mock_restart:
        result = driver.unload(None)

    assert fake.calls == []  # API path not exercised
    assert mock_restart.call_count == 1
    assert result.method == "restart"


def test_restart_only_driver_never_calls_native_unload() -> None:
    m = load_manifest("omlx")
    fake = _FakeEngineWithUnload(unload_returns=True)
    driver = RestartOnlyDriver(m, fake)

    with patch("ais_core.lifecycle.restart"):
        result = driver.unload("anything")

    assert fake.calls == []
    assert result.method == "restart"


def test_restart_failure_reports_error_not_raise() -> None:
    """If lifecycle.restart raises, unload() returns an error outcome cleanly."""
    m = load_manifest("ollama")
    driver = EngineDriver(m, asiai_engine=None)

    with patch("ais_core.lifecycle.restart", side_effect=RuntimeError("launchctl angry")):
        result = driver.unload("llama3.2")

    assert result.method == "error"
    assert result.success is False
    assert "launchctl angry" in result.detail


# ---------------------------------------------------------------------------
# list_loaded_models
# ---------------------------------------------------------------------------


def test_list_loaded_models_uses_asiai_engine() -> None:
    m = load_manifest("ollama")
    fake = _FakeEngineWithUnload()
    driver = EngineDriver(m, fake)
    assert driver.list_loaded_models() == ["fake-model"]


def test_list_loaded_models_empty_when_no_engine() -> None:
    m = load_manifest("ollama")
    driver = EngineDriver(m, asiai_engine=None)
    assert driver.list_loaded_models() == []


def test_list_loaded_models_empty_when_engine_lacks_method() -> None:
    m = load_manifest("omlx")
    driver = EngineDriver(m, _FakeEngineNoUnload())
    # _FakeEngineNoUnload has list_running() but it returns [] anyway.
    assert driver.list_loaded_models() == []


# ---------------------------------------------------------------------------
# from_manifest factories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, expected_name",
    [
        (OllamaDriver.from_manifest, "ollama"),
        (LMStudioDriver.from_manifest, "lmstudio"),
        (OmlxDriver.from_manifest, "omlx"),
        (TurboquantDriver.from_manifest, "turboquant"),
        (LlamaCppDriver.from_manifest, "llamacpp"),
        (LlamaCppAuxDriver.from_manifest, "llamacpp-aux"),
    ],
)
def test_factories_load_correct_manifest(factory, expected_name: str) -> None:
    driver = factory()
    assert driver.manifest.name == expected_name


def test_omlx_driver_is_restart_only_subclass() -> None:
    driver = OmlxDriver.from_manifest()
    assert isinstance(driver, RestartOnlyDriver)


def test_turboquant_driver_is_restart_only_subclass() -> None:
    driver = TurboquantDriver.from_manifest()
    assert isinstance(driver, RestartOnlyDriver)


def test_llamacpp_driver_is_restart_only_subclass() -> None:
    """llama-server has no native unload API — must use restart-only path."""
    driver = LlamaCppDriver.from_manifest()
    assert isinstance(driver, RestartOnlyDriver)
    assert driver.name == "llamacpp"
    # _try_native_unload always returns False on RestartOnlyDriver
    assert driver._try_native_unload("any-model") is False


def test_llamacpp_aux_driver_is_restart_only_subclass() -> None:
    driver = LlamaCppAuxDriver.from_manifest()
    assert isinstance(driver, RestartOnlyDriver)
    assert driver.name == "llamacpp-aux"
    assert driver._try_native_unload("any-model") is False


def test_ollama_driver_is_not_restart_only() -> None:
    """OllamaDriver must keep the API path enabled (it's the cheaper unload)."""
    driver = OllamaDriver.from_manifest()
    assert not isinstance(driver, RestartOnlyDriver)
    assert isinstance(driver, EngineDriver)


def test_factories_dont_explode_when_asiai_engine_missing() -> None:
    """If asiai is not installed, factories must still return a working driver."""
    with patch("ais_engines.ollama.make_asiai_engine_proxy", return_value=None):
        driver = OllamaDriver.from_manifest()
    assert driver.asiai_engine is None
    # restart fallback still works:
    with patch("ais_core.lifecycle.restart") as mock_restart:
        result = driver.unload("foo")
    assert mock_restart.called
    assert result.method == "restart"


def test_factories_handle_engine_constructor_typeerror() -> None:
    """If asiai's engine constructor signature changes, fall back to no-arg."""
    from ais_engines.base import make_asiai_engine_proxy

    fake_engine_cls = MagicMock(
        side_effect=[TypeError("base_url unsupported"), _FakeEngineWithUnload()]
    )
    with patch.dict(
        "sys.modules",
        {"asiai.engines.ollama": MagicMock(OllamaEngine=fake_engine_cls)},
    ):
        engine = make_asiai_engine_proxy(
            load_manifest("ollama"),
            module="asiai.engines.ollama",
            class_name="OllamaEngine",
        )

    assert fake_engine_cls.call_count == 2
    assert engine is not None


def test_make_asiai_engine_proxy_returns_none_when_module_missing() -> None:
    """ImportError path is silent (documented dev environment without asiai)."""
    from ais_engines.base import make_asiai_engine_proxy

    engine = make_asiai_engine_proxy(
        load_manifest("ollama"),
        module="nonexistent_package_xyz",
        class_name="Whatever",
    )
    assert engine is None


def test_make_asiai_engine_proxy_logs_warning_on_construction_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Q4 audit fix — silent constructor failures must surface in logs."""
    import logging

    from ais_engines.base import make_asiai_engine_proxy

    fake_cls = MagicMock(side_effect=[TypeError("base_url"), RuntimeError("noargs too")])
    with (
        caplog.at_level(logging.WARNING, logger="ais_engines.base"),
        patch.dict(
            "sys.modules",
            {"asiai.engines.ollama": MagicMock(OllamaEngine=fake_cls)},
        ),
    ):
        engine = make_asiai_engine_proxy(
            load_manifest("ollama"),
            module="asiai.engines.ollama",
            class_name="OllamaEngine",
        )

    assert engine is None
    assert any("restart-only fallback" in rec.message for rec in caplog.records)
