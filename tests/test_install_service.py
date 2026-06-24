"""Tests for ``aisctl install-service`` after the AEP-01 migration to the privileged helper.

The client no longer does raw ``sudo mv`` / ``launchctl``; it calls the content-validated
``install-reserved-service`` / ``uninstall-reserved-service`` helper actions (B-only scope:
aisctl-serve is shipped; asiai-web is deferred to the Fleet epic)."""

from __future__ import annotations

import argparse
from unittest.mock import patch

from ais_cli import install_service
from ais_core import privhelper


def _args(service, *, port=None, host=None, dry_run=False, as_json=False):
    return argparse.Namespace(service=service, port=port, host=host, dry_run=dry_run, json=as_json)


def test_install_serve_routes_to_helper():
    with patch.object(install_service.privhelper, "run") as run:
        rc = install_service.cmd_install_service(_args("aisctl-serve"))
    assert rc == 0
    run.assert_called_once()
    args, _ = run.call_args
    assert args[0] == "install-reserved-service"
    assert "--service" in args and "aisctl-serve" in args
    # No --user / --host / --label / --binary smuggled to the helper.
    assert "--user" not in args and "--host" not in args and "--label" not in args


def test_install_serve_passes_port():
    with patch.object(install_service.privhelper, "run") as run:
        install_service.cmd_install_service(_args("aisctl-serve", port=9999))
    args, _ = run.call_args
    assert "--port" in args and "9999" in args


def test_install_serve_dry_run_propagated():
    with patch.object(install_service.privhelper, "run") as run:
        install_service.cmd_install_service(_args("aisctl-serve", dry_run=True))
    _, kwargs = run.call_args
    assert kwargs.get("dry_run") is True


def test_install_web_is_deferred_not_installed():
    """asiai-web daemon is deferred to the Fleet epic: never touches the helper, exits 0."""
    with patch.object(install_service.privhelper, "run") as run:
        rc = install_service.cmd_install_service(_args("asiai-web"))
    assert rc == 0
    run.assert_not_called()


def test_install_unknown_service_rejected():
    with patch.object(install_service.privhelper, "run") as run:
        rc = install_service.cmd_install_service(_args("com.asiai.evil"))
    assert rc == 2
    run.assert_not_called()


def test_install_serve_helper_failure_returns_1():
    with patch.object(
        install_service.privhelper,
        "run",
        side_effect=privhelper.PrivHelperError("asiai-priv install-reserved-service failed"),
    ):
        rc = install_service.cmd_install_service(_args("aisctl-serve"))
    assert rc == 1


def test_uninstall_serve_routes_to_helper():
    with patch.object(install_service.privhelper, "run") as run:
        rc = install_service.cmd_uninstall_service(_args("aisctl-serve"))
    assert rc == 0
    args, _ = run.call_args
    assert args[0] == "uninstall-reserved-service"
    assert "aisctl-serve" in args


def test_uninstall_web_is_noop():
    with patch.object(install_service.privhelper, "run") as run:
        rc = install_service.cmd_uninstall_service(_args("asiai-web"))
    assert rc == 0
    run.assert_not_called()
