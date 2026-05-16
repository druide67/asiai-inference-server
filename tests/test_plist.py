"""Tests for ais_core.plist — pure rendering logic."""

from __future__ import annotations

import plistlib
from dataclasses import replace

import pytest

from ais_core.manifest import (
    BinarySpec,
    EngineManifest,
    FirewallSpec,
    LogSpec,
    NetworkSpec,
    PlistSpec,
    WrapperSpec,
    load_manifest,
)
from ais_core.plist import (
    LAUNCH_DAEMONS_DIR,
    PlistError,
    build_plist_dict,
    plist_path,
    render_plist_xml,
)


def test_plist_path_uses_launch_daemons_dir() -> None:
    m = load_manifest("ollama")
    assert plist_path(m) == f"{LAUNCH_DAEMONS_DIR}/com.asiai.ollama.plist"


def test_ollama_plist_no_wrapper_no_bind_args() -> None:
    """Ollama uses env vars, not --host/--port; no wrapper."""
    m = load_manifest("ollama")
    d = build_plist_dict(m, user="jmn", binary_path="/opt/homebrew/bin/ollama")

    assert d["Label"] == "com.asiai.ollama"
    assert d["UserName"] == "jmn"
    assert d["ProgramArguments"] == ["/opt/homebrew/bin/ollama", "serve"]
    assert "--host" not in d["ProgramArguments"]
    assert d["EnvironmentVariables"]["OLLAMA_HOST"] == "0.0.0.0:11434"
    assert d["EnvironmentVariables"]["OLLAMA_KEEP_ALIVE"] == "-1"
    assert d["EnvironmentVariables"]["PATH"].startswith("/opt/homebrew/bin")
    assert d["KeepAlive"] == {"Crashed": True, "SuccessfulExit": False}
    assert d["RunAtLoad"] is True
    assert d["ThrottleInterval"] == 10
    assert d["TimeOut"] == 30
    import os

    expected_log = os.path.expanduser("~/Library/Logs/asiai/ollama/ollama.log")
    assert d["StandardOutPath"] == expected_log


def test_lmstudio_plist_uses_wrapper_path_not_binary() -> None:
    """When wrapper.needed, ProgramArguments[0] is the wrapper, not lms."""
    m = load_manifest("lmstudio")
    d = build_plist_dict(m, user="jmn", binary_path="/opt/homebrew/bin/lms")

    assert d["ProgramArguments"] == ["/usr/local/bin/lmstudio-server-start"]
    # Wrapper handles its own args; no --host/--port appended.
    assert len(d["ProgramArguments"]) == 1


def test_omlx_plist_appends_host_port_when_bind_set() -> None:
    """Engines with explicit bind get --host/--port appended."""
    m = load_manifest("omlx")
    d = build_plist_dict(m, user="jmn", binary_path="/opt/homebrew/bin/omlx")

    assert d["ProgramArguments"] == [
        "/opt/homebrew/bin/omlx",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8800",
    ]


def test_llamacpp_plist_injects_model_path_expanded() -> None:
    """llama-server needs --model <abs_path> bound at launch.

    The manifest stores a tilde path (``~/llms/gguf/active.gguf``) so it
    stays Mac-portable; the plist renderer must expand it to an absolute
    path before emitting ProgramArguments.
    """
    import os

    m = load_manifest("llamacpp")
    d = build_plist_dict(m, user="jmn", binary_path="/opt/homebrew/bin/llama-server")
    args = d["ProgramArguments"]

    # First element is the binary path
    assert args[0] == "/opt/homebrew/bin/llama-server"
    # --model must be present with an expanded absolute path
    assert "--model" in args
    model_idx = args.index("--model")
    expanded = args[model_idx + 1]
    assert expanded.startswith("/")  # absolute path, no tilde left
    assert expanded == os.path.expanduser("~/llms/gguf/active.gguf")
    # --host/--port still appended at the end from [network]
    assert "--host" in args
    assert "0.0.0.0" in args
    assert "--port" in args
    assert "8080" in args
    # No duplicate --host or --port from program_args (manifest hygiene)
    assert args.count("--host") == 1
    assert args.count("--port") == 1


def test_plist_no_model_arg_when_model_path_unset() -> None:
    """Engines that don't bind a model at launch must not get --model injected."""
    m = load_manifest("ollama")
    d = build_plist_dict(m, user="jmn", binary_path="/opt/homebrew/bin/ollama")
    assert "--model" not in d["ProgramArguments"]


def test_llamacpp_plist_injects_template_path_expanded() -> None:
    """Same contract as model_path: tilde path in manifest, absolute in plist.

    The Qwen3.6 official template uses Jinja ``|items`` which crashes the
    C++ minijinja engine; we ship a froggeric-fixed template via a user
    symlink at ``~/llms/templates/active.jinja`` and pass it to llama-server
    with ``--chat-template-file``.
    """
    import os

    m = load_manifest("llamacpp")
    d = build_plist_dict(m, user="jmn", binary_path="/opt/homebrew/bin/llama-server")
    args = d["ProgramArguments"]

    assert "--chat-template-file" in args
    tpl_idx = args.index("--chat-template-file")
    expanded = args[tpl_idx + 1]
    assert expanded.startswith("/")
    assert expanded == os.path.expanduser("~/llms/templates/active.jinja")
    # --jinja (which would re-enable the model's embedded template and
    # defeat the override) must be gone
    assert "--jinja" not in args


def test_plist_no_template_arg_when_template_path_unset() -> None:
    """Engines without template_path must not get --chat-template-file injected."""
    m = load_manifest("ollama")
    d = build_plist_dict(m, user="jmn", binary_path="/opt/homebrew/bin/ollama")
    assert "--chat-template-file" not in d["ProgramArguments"]


def test_render_plist_xml_is_valid_plist() -> None:
    m = load_manifest("ollama")
    xml_bytes = render_plist_xml(m, user="jmn", binary_path="/opt/homebrew/bin/ollama")
    # Round-trip: parsing it back must yield the same dict.
    parsed = plistlib.loads(xml_bytes)
    assert parsed["Label"] == "com.asiai.ollama"
    assert parsed["EnvironmentVariables"]["OLLAMA_HOST"] == "0.0.0.0:11434"


def test_render_plist_xml_escapes_ampersand_in_env_values() -> None:
    """Bash heredoc would have rendered & raw and corrupted the XML."""
    m = load_manifest("ollama")
    m_with_amp = replace(
        m,
        env_vars=tuple([*list(m.env_vars), "WITH_AMP=foo&bar"]),
    )
    xml_bytes = render_plist_xml(m_with_amp, user="jmn", binary_path="/opt/homebrew/bin/ollama")
    assert b"&amp;" in xml_bytes
    parsed = plistlib.loads(xml_bytes)
    assert parsed["EnvironmentVariables"]["WITH_AMP"] == "foo&bar"


def test_invalid_plist_label_refused_at_render() -> None:
    """Defense in depth: even if a malformed manifest sneaks past validation,
    the renderer must still refuse a non-com.asiai label."""
    m = load_manifest("ollama")
    bad = replace(m, plist=replace(m.plist, name="com.evilcorp.ollama"))
    with pytest.raises(PlistError, match=r"com\.asiai"):
        build_plist_dict(bad, user="jmn", binary_path="/opt/homebrew/bin/ollama")


def test_wrapper_needed_without_install_path_refused_at_render() -> None:
    m = load_manifest("ollama")
    bad = replace(m, wrapper=WrapperSpec(needed=True, install_path=None))
    with pytest.raises(PlistError, match="install_path"):
        build_plist_dict(bad, user="jmn", binary_path="/opt/homebrew/bin/ollama")


def test_keepalive_keeps_unsuccessful_exits_alive() -> None:
    """Critical for prod headless: a healthy daemon must restart on crash but
    NOT loop on SuccessfulExit (otherwise an intentional shutdown bounces)."""
    m = load_manifest("ollama")
    d = build_plist_dict(m, user="jmn", binary_path="/opt/homebrew/bin/ollama")
    assert d["KeepAlive"]["Crashed"] is True
    assert d["KeepAlive"]["SuccessfulExit"] is False


def _minimal_manifest() -> EngineManifest:
    """Hand-built manifest used to test edge cases without touching disk."""
    return EngineManifest(
        name="dummy",
        display="Dummy",
        binary=BinarySpec(
            candidates=("/usr/local/bin/dummy",),
            process_pattern="dummy",
            program_args=("serve",),
        ),
        plist=PlistSpec(name="com.asiai.dummy", throttle_interval=10, timeout=30),
        network=NetworkSpec(
            port=9999,
            bind="127.0.0.1",
            health_endpoint="/health",
            health_timeout=10,
        ),
        firewall=FirewallSpec(supported=True, anchor_name="com.asiai.dummy"),
        logs=LogSpec(dir="/tmp", stdout="dummy.out", stderr="dummy.err"),
        wrapper=WrapperSpec(needed=False),
    )


def test_minimal_manifest_renders_cleanly() -> None:
    m = _minimal_manifest()
    d = build_plist_dict(m, user="jmn", binary_path="/usr/local/bin/dummy")
    assert d["ProgramArguments"] == [
        "/usr/local/bin/dummy",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "9999",
    ]
