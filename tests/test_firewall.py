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
