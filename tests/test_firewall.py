"""Tests for ais_core.firewall — pure rendering + state-line manipulation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ais_core.firewall import (
    DEFAULT_SUBNETS,
    PF_ANCHORS_DIR,
    FirewallError,
    _anchor_conf_lines,
    anchor_path,
    build_anchor_content,
)
from ais_core.manifest import load_manifest


def test_anchor_path_uses_pf_anchors_dir() -> None:
    m = load_manifest("ollama")
    assert anchor_path(m) == f"{PF_ANCHORS_DIR}/com.asiai.ollama"


def test_anchor_content_includes_localhost_and_lan_pass_rules() -> None:
    m = load_manifest("ollama")
    content = build_anchor_content(m)

    assert "pass in quick on lo0 proto tcp to any port 11434" in content
    assert "192.168.0.0/16" in content
    assert "10.0.0.0/8" in content
    assert "172.16.0.0/12" in content
    assert "block in quick inet proto tcp to any port 11434" in content


def test_anchor_content_uses_engine_specific_port() -> None:
    m = load_manifest("lmstudio")
    content = build_anchor_content(m)
    assert "port 1234" in content
    assert "port 11434" not in content


def test_anchor_content_with_custom_subnets() -> None:
    m = load_manifest("ollama")
    custom = ("192.168.1.0/24",)
    content = build_anchor_content(m, subnets=custom)
    assert "192.168.1.0/24" in content
    assert "10.0.0.0/8" not in content


def test_anchor_content_refuses_empty_subnets() -> None:
    """Refuse to write a wide-open anchor by accident."""
    m = load_manifest("ollama")
    with pytest.raises(FirewallError, match="empty"):
        build_anchor_content(m, subnets=())


def test_anchor_content_refuses_non_asiai_label() -> None:
    """Defense in depth even if a malformed manifest sneaks past validation."""
    m = load_manifest("ollama")
    bad = replace(m, firewall=replace(m.firewall, anchor_name="com.evilcorp.x"))
    with pytest.raises(FirewallError, match=r"com\.asiai"):
        build_anchor_content(bad)


def test_default_subnets_match_rfc1918() -> None:
    assert set(DEFAULT_SUBNETS) == {
        "192.168.0.0/16",
        "10.0.0.0/8",
        "172.16.0.0/12",
    }


def test_anchor_content_block_rule_is_quick() -> None:
    """Without 'quick', a later allow rule could shadow the deny."""
    m = load_manifest("ollama")
    content = build_anchor_content(m)
    block_lines = [ln for ln in content.splitlines() if ln.startswith("block")]
    assert all("quick" in ln for ln in block_lines)


def test_anchor_content_blocks_ipv6_too() -> None:
    """inet-only block left the port open to any IPv6 LAN peer."""
    m = load_manifest("ollama")
    content = build_anchor_content(m)
    assert "block in quick inet6 proto tcp to any port 11434" in content


def test_pf_conf_lines_include_load_directive() -> None:
    """The bug the audit caught: an `anchor` line without `load anchor`
    declares an attachment point but never reads the rules file — the
    firewall silently enforces nothing."""
    decl, load = _anchor_conf_lines("com.asiai.ollama")
    assert decl == 'anchor "com.asiai.ollama" all'
    assert load == f'load anchor "com.asiai.ollama" from "{PF_ANCHORS_DIR}/com.asiai.ollama"'


# ---------------------------------------------------------------------------
# S0 (2026-07-01 half-uninstall retex): preflight + idempotence predicates
# ---------------------------------------------------------------------------


def test_preflight_sudo_no_op_with_tty(monkeypatch) -> None:
    """A TTY means sudo can prompt — preflight passes without invoking sudo at all."""
    import ais_core.firewall as fw

    monkeypatch.setattr(fw.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        fw.subprocess, "run", lambda *a, **k: pytest.fail("preflight must not call sudo on a TTY")
    )
    fw.preflight_sudo("test-op")


def test_preflight_sudo_accepts_cached_ticket(monkeypatch) -> None:
    """No TTY but `sudo -nv` succeeds (cached ticket / NOPASSWD) — passes."""
    import types

    import ais_core.firewall as fw

    monkeypatch.setattr(fw.sys.stdin, "isatty", lambda: False)
    calls: list[list[str]] = []

    def fake_run(argv, **k):
        calls.append(argv)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(fw.subprocess, "run", fake_run)
    fw.preflight_sudo("test-op")
    assert calls == [["sudo", "-n", "-v"]]


def test_preflight_sudo_raises_without_tty_or_ticket(monkeypatch) -> None:
    """No TTY and no ticket: raise BEFORE any daemon mutation, with actionable advice."""
    import types

    import ais_core.firewall as fw

    monkeypatch.setattr(fw.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(fw.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1))
    with pytest.raises(FirewallError, match="--firewall none"):
        fw.preflight_sudo("test-op")


def _fw_tmp(monkeypatch, tmp_path):
    """Point the module's pf paths at tmp_path; returns (module, anchors_dir, pf_conf)."""
    import ais_core.firewall as fw

    anchors = tmp_path / "pf.anchors"
    anchors.mkdir()
    pf_conf = tmp_path / "pf.conf"
    monkeypatch.setattr(fw, "PF_ANCHORS_DIR", str(anchors))
    monkeypatch.setattr(fw, "PF_CONF_PATH", str(pf_conf))
    return fw, anchors, pf_conf


def test_anchor_is_current_true_only_on_byte_match(monkeypatch, tmp_path) -> None:
    fw, anchors, _ = _fw_tmp(monkeypatch, tmp_path)
    m = load_manifest("llamacpp")
    assert fw.anchor_is_current(m) is False  # no file yet
    (anchors / m.firewall.anchor_name).write_text(fw.build_anchor_content(m), encoding="utf-8")
    assert fw.anchor_is_current(m) is True
    (anchors / m.firewall.anchor_name).write_text("# stale\n", encoding="utf-8")
    assert fw.anchor_is_current(m) is False


def test_anchor_present_sees_file_or_pf_conf_lines(monkeypatch, tmp_path) -> None:
    fw, anchors, pf_conf = _fw_tmp(monkeypatch, tmp_path)
    m = load_manifest("llamacpp")
    name = m.firewall.anchor_name
    assert fw.anchor_present(m) is False
    # pf.conf line alone (file already deleted) still counts — remove_anchor must scrub it.
    pf_conf.write_text(f'anchor "{name}" all\n', encoding="utf-8")
    assert fw.anchor_present(m) is True
    pf_conf.write_text("", encoding="utf-8")
    (anchors / name).write_text("# anything\n", encoding="utf-8")
    assert fw.anchor_present(m) is True


def test_anchor_up_to_date_needs_match_and_both_pf_lines(monkeypatch, tmp_path) -> None:
    fw, anchors, pf_conf = _fw_tmp(monkeypatch, tmp_path)
    m = load_manifest("llamacpp")
    name = m.firewall.anchor_name
    (anchors / name).write_text(fw.build_anchor_content(m), encoding="utf-8")
    assert fw.anchor_up_to_date(m) is False  # file matches but pf.conf not wired
    # Generate the lines via the module so the monkeypatched anchors dir is embedded
    # in the load directive.
    decl, load = fw._anchor_conf_lines(name)
    pf_conf.write_text(f"{decl}\n", encoding="utf-8")
    assert fw.anchor_up_to_date(m) is False  # declaration alone is not wired (no load line)
    pf_conf.write_text(f"{decl}\n{load}\n", encoding="utf-8")
    assert fw.anchor_up_to_date(m) is True


def test_install_anchor_skips_all_sudo_when_up_to_date(monkeypatch, tmp_path) -> None:
    """The idempotence guard: current anchor + wired pf.conf → return without any sudo."""
    fw, anchors, pf_conf = _fw_tmp(monkeypatch, tmp_path)
    m = load_manifest("llamacpp")
    name = m.firewall.anchor_name
    (anchors / name).write_text(fw.build_anchor_content(m), encoding="utf-8")
    decl, load = fw._anchor_conf_lines(name)
    pf_conf.write_text(f"{decl}\n{load}\n", encoding="utf-8")
    monkeypatch.setattr(
        fw.subprocess, "run", lambda *a, **k: pytest.fail("no sudo when anchor is up to date")
    )
    assert fw.install_anchor(m) == fw.anchor_path(m)
