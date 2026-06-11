"""Tests for ``asiai-launch`` — the secure generic launcher (ais_core.launch).

Focus: the confused-deputy guards (binary allowlist + manifest permissions +
service-name validation) and faithful argv construction from manifest params.
"""

from __future__ import annotations

import os
import stat

import pytest

from ais_core import launch
from ais_core.launch import (
    LaunchError,
    LaunchSecurityError,
    resolve_launch,
)


def _manifest_toml(
    *,
    name: str = "llamacpp-aux-test",
    binary_path: str,
    program_args: tuple[str, ...] = (),
    model_path: str | None = None,
    template_path: str | None = None,
    mmproj_path: str | None = None,
    wrapper_needed: bool = False,
    bind: str = "0.0.0.0",
    port: int = 8090,
) -> str:
    args = ", ".join(f'"{a}"' for a in program_args)
    lines = [
        f'name = "{name}"',
        'display = "test engine"',
        "",
        "[binary]",
        f'candidates = ["{binary_path}"]',
        'process_pattern = "llms/gguf/test/active.gguf"',
        f"program_args = [{args}]",
    ]
    if model_path:
        lines.append(f'model_path = "{model_path}"')
    if template_path:
        lines.append(f'template_path = "{template_path}"')
    if mmproj_path:
        lines.append(f'mmproj_path = "{mmproj_path}"')
    lines += [
        "",
        "[plist]",
        'name = "com.asiai.llamacpp-aux-test"',
        "throttle_interval = 10",
        "timeout = 30",
        "",
        "[network]",
        f"port = {port}",
        f'bind = "{bind}"',
        'health_endpoint = "/health"',
        "health_timeout = 30",
        "",
        "[firewall]",
        "supported = true",
        'anchor_name = "com.asiai.llamacpp-aux-test"',
        "",
        "[logs]",
        'dir = "~/Library/Logs/asiai/test"',
        'stdout = "t.log"',
        'stderr = "t.err"',
        "",
        "[wrapper]",
        f"needed = {'true' if wrapper_needed else 'false'}",
    ]
    if wrapper_needed:
        lines.append('install_path = "/usr/local/bin/turboquant-start"')
        lines.append('template = "wrapper-start.sh.tpl"')
    return "\n".join(lines) + "\n"


def _make_bin(tmp_path, name: str) -> str:
    """Create an executable-ish file so BinarySpec.resolve() finds it."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text("#!/bin/sh\n")
    return str(p)


def _write_manifest(tmp_path, service: str, body: str):
    mdir = tmp_path / "active"
    mdir.mkdir(exist_ok=True)
    path = mdir / f"{service}.toml"
    path.write_text(body)
    os.chmod(path, 0o644)  # owner-writable only
    return mdir, path


# --- argv construction --------------------------------------------------------


class TestBuildArgv:
    def test_full_argv_order(self, tmp_path):
        binary = _make_bin(tmp_path, "llama-server")
        body = _manifest_toml(
            binary_path=binary,
            program_args=("--flash-attn", "on", "--ctx-size", "131072"),
            model_path="/models/active.gguf",
            mmproj_path="/models/mmproj.gguf",
            bind="0.0.0.0",
            port=8090,
        )
        mdir, _ = _write_manifest(tmp_path, "aux1", body)
        argv = resolve_launch("aux1", manifest_dir=mdir)
        assert argv == [
            binary,
            "--flash-attn",
            "on",
            "--ctx-size",
            "131072",
            "--model",
            "/models/active.gguf",
            "--mmproj",
            "/models/mmproj.gguf",
            "--host",
            "0.0.0.0",
            "--port",
            "8090",
        ]

    def test_empty_bind_omits_host_port(self, tmp_path):
        binary = _make_bin(tmp_path, "llama-server")
        mdir, _ = _write_manifest(tmp_path, "aux1", _manifest_toml(binary_path=binary, bind=""))
        argv = resolve_launch("aux1", manifest_dir=mdir)
        assert "--host" not in argv and "--port" not in argv

    def test_template_included(self, tmp_path):
        binary = _make_bin(tmp_path, "llama-server")
        body = _manifest_toml(binary_path=binary, template_path="/t/active.jinja")
        mdir, _ = _write_manifest(tmp_path, "aux1", body)
        argv = resolve_launch("aux1", manifest_dir=mdir)
        assert "--chat-template-file" in argv
        assert argv[argv.index("--chat-template-file") + 1] == "/t/active.jinja"


# --- allowlist (binary path never trusted from the manifest) ------------------


class TestBinaryAllowlist:
    def test_allowlisted_binary_ok(self, tmp_path):
        binary = _make_bin(tmp_path, "llama-server-turboquant")
        mdir, _ = _write_manifest(tmp_path, "aux5", _manifest_toml(binary_path=binary))
        argv = resolve_launch("aux5", manifest_dir=mdir)
        assert argv[0] == binary

    def test_non_allowlisted_binary_rejected(self, tmp_path):
        binary = _make_bin(tmp_path, "evil")  # basename not in allowlist
        mdir, _ = _write_manifest(tmp_path, "aux1", _manifest_toml(binary_path=binary))
        with pytest.raises(LaunchSecurityError, match="not in the launcher allowlist"):
            resolve_launch("aux1", manifest_dir=mdir)

    def test_missing_binary_candidate(self, tmp_path):
        # candidate path that does not exist on disk
        ghost = str(tmp_path / "bin" / "llama-server")
        mdir, _ = _write_manifest(tmp_path, "aux1", _manifest_toml(binary_path=ghost))
        with pytest.raises(LaunchError, match="no binary candidate exists"):
            resolve_launch("aux1", manifest_dir=mdir)


# --- manifest permission guard (confused-deputy) ------------------------------


class TestManifestPerms:
    def test_world_writable_rejected(self, tmp_path):
        binary = _make_bin(tmp_path, "llama-server")
        mdir, path = _write_manifest(tmp_path, "aux1", _manifest_toml(binary_path=binary))
        os.chmod(path, 0o666)  # world-writable
        with pytest.raises(LaunchSecurityError, match="group/world-writable"):
            resolve_launch("aux1", manifest_dir=mdir)

    def test_group_writable_rejected(self, tmp_path):
        binary = _make_bin(tmp_path, "llama-server")
        mdir, path = _write_manifest(tmp_path, "aux1", _manifest_toml(binary_path=binary))
        os.chmod(path, 0o664)  # group-writable
        with pytest.raises(LaunchSecurityError, match="group/world-writable"):
            resolve_launch("aux1", manifest_dir=mdir)

    def test_foreign_owner_rejected(self, tmp_path, monkeypatch):
        binary = _make_bin(tmp_path, "llama-server")
        mdir, _ = _write_manifest(tmp_path, "aux1", _manifest_toml(binary_path=binary))
        # File is owned by the current uid; pretend the daemon runs as someone else.
        monkeypatch.setattr(os, "geteuid", lambda: 999_999)
        with pytest.raises(LaunchSecurityError, match="must be root or the daemon user"):
            resolve_launch("aux1", manifest_dir=mdir)

    def test_safe_perms_pass(self, tmp_path):
        binary = _make_bin(tmp_path, "llama-server")
        mdir, path = _write_manifest(tmp_path, "aux1", _manifest_toml(binary_path=binary))
        os.chmod(path, 0o600)
        argv = resolve_launch("aux1", manifest_dir=mdir)  # no raise
        assert argv[0] == binary


# --- service name validation + wrapper rejection ------------------------------


class TestGuards:
    @pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "", ".hidden"])
    def test_bad_service_name(self, tmp_path, bad):
        with pytest.raises(LaunchSecurityError, match="invalid service name"):
            resolve_launch(bad, manifest_dir=tmp_path / "active")

    def test_wrapper_engine_rejected(self, tmp_path):
        binary = _make_bin(tmp_path, "llama-server-turboquant")
        body = _manifest_toml(binary_path=binary, wrapper_needed=True)
        mdir, _ = _write_manifest(tmp_path, "turboquant", body)
        with pytest.raises(LaunchError, match="boot prelude"):
            resolve_launch("turboquant", manifest_dir=mdir)


# --- main() -------------------------------------------------------------------


class TestMain:
    def test_usage_when_no_service(self):
        assert launch.main(["asiai-launch"]) == 2

    def test_launch_error_returns_1(self, monkeypatch):
        def boom(_service):
            raise LaunchSecurityError("nope")

        monkeypatch.setattr(launch, "resolve_launch", boom)
        assert launch.main(["asiai-launch", "aux1"]) == 1

    def test_execs_resolved_command(self, monkeypatch):
        monkeypatch.setattr(launch, "resolve_launch", lambda _s: ["/bin/echo", "hi"])
        recorded = {}

        def fake_execv(path, args):
            recorded["path"] = path
            recorded["args"] = args
            # real execv replaces the process; the fake returns so we can assert

        monkeypatch.setattr(os, "execv", fake_execv)
        rc = launch.main(["asiai-launch", "aux1"])
        assert recorded == {"path": "/bin/echo", "args": ["/bin/echo", "hi"]}
        assert rc == 127  # only reached because the fake execv returned


def test_assert_safe_manifest_perms_on_missing_file(tmp_path):
    from ais_core.launch import assert_safe_manifest_perms

    with pytest.raises(LaunchSecurityError, match="cannot stat manifest"):
        assert_safe_manifest_perms(tmp_path / "nope.toml")


def test_non_regular_file_rejected(tmp_path):
    from ais_core.launch import assert_safe_manifest_perms

    with pytest.raises(LaunchSecurityError, match="not a regular file"):
        assert_safe_manifest_perms(tmp_path)  # a directory


# stat import kept meaningful (mode constants referenced indirectly via chmod)
assert stat.S_IWOTH and stat.S_IWGRP
