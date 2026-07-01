"""pf anchor management for engine firewalls.

Ports ``lib-engine.sh:540-622`` (``engine_setup_firewall`` /
``engine_remove_firewall``) to Python.

Concretely we generate a small pf anchor file under ``/etc/pf.anchors/`` that
allows the engine port from localhost + RFC1918 subnets and blocks everything
else, then we add a single ``anchor "<name>" all`` line to ``/etc/pf.conf``
and reload pf. ``com.asiai.*`` is the only label pattern accepted.

Password-gated, NOT in the helper (AEP-01)
------------------------------------------
These pf operations deliberately stay raw ``sudo`` (no ``-n``) and are NOT in the
NOPASSWD ``asiai-priv`` surface: firewall setup is opt-in (``enable_firewall``,
off by default), install-time, and operator-present, so prompting for a password
is acceptable. Editing ``/etc/pf.conf`` is root-equivalent and delicate (cf. the
2026-06-11 load-anchor audit), so it is kept out of a generate-don't-validate
helper. No autonomous path (cron / Hermes) ever triggers it.

Atomic write
------------
We write the anchor file via ``sudo tee`` from a stdin buffer (no shell heredoc
race) and validate the syntax with ``pfctl -nf <file>`` before activating it.
The Bash version skipped the validation step and would silently push a broken
anchor into pf.conf.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from ais_core.io import secure_staging_dir
from ais_core.manifest import ANCHOR_NAME_RE, EngineManifest

PF_CONF_PATH = "/etc/pf.conf"
PF_ANCHORS_DIR = "/etc/pf.anchors"

# RFC1918 private subnets — the only sources allowed by default.
DEFAULT_SUBNETS: tuple[str, ...] = (
    "192.168.0.0/16",
    "10.0.0.0/8",
    "172.16.0.0/12",
)


class FirewallError(RuntimeError):
    """Raised when an anchor cannot be written, validated, or activated."""


def anchor_path(manifest: EngineManifest) -> str:
    return f"{PF_ANCHORS_DIR}/{manifest.firewall.anchor_name}"


def preflight_sudo(what: str) -> None:
    """Fail fast when a pf operation will need sudo but sudo cannot prompt.

    pf operations are deliberately raw ``sudo`` (password-gated, see the module
    docstring). That contract assumes an operator at a terminal. Without a TTY,
    ``sudo`` cannot prompt and fails midway — 2026-07-01 retex: ``uninstall`` had
    already booted out the daemon when the anchor ``rm`` died, leaving the engine
    half-removed. Callers MUST run this BEFORE any daemon mutation whose sequence
    later includes a pf write.

    A cached sudo timestamp (or blanket NOPASSWD) satisfies ``sudo -nv`` and is
    accepted; so is an interactive stdin. Anything else raises with instructions.
    """
    if sys.stdin.isatty():
        return  # sudo can prompt the operator
    proc = subprocess.run(["sudo", "-n", "-v"], capture_output=True)
    if proc.returncode == 0:
        return  # valid cached ticket / NOPASSWD — no prompt needed
    raise FirewallError(
        f"{what} needs an interactive sudo password and there is no terminal to "
        "prompt on. Re-run from an interactive terminal, or skip firewall changes "
        "(--firewall none)."
    )


def anchor_is_current(
    manifest: EngineManifest,
    *,
    subnets: tuple[str, ...] = DEFAULT_SUBNETS,
) -> bool:
    """True iff the installed anchor file byte-matches what we would generate now.

    Pure read (the anchor file is root:wheel 0644 — world-readable): lets callers
    skip the whole password-gated pf sequence when a reinstall would rewrite the
    identical anchor (same port/subnets — the nominal flag-change reinstall).
    """
    try:
        installed = Path(anchor_path(manifest)).read_text(encoding="utf-8")
    except OSError:
        return False
    return installed == build_anchor_content(manifest, subnets=subnets)


def anchor_present(manifest: EngineManifest) -> bool:
    """True iff any trace of this engine's anchor exists (file or pf.conf lines).

    Used to decide whether an uninstall will need the password-gated pf path at
    all — ``remove_anchor`` also scrubs pf.conf, so a missing file alone does not
    mean there is nothing to do.
    """
    if Path(anchor_path(manifest)).exists():
        return True
    name = manifest.firewall.anchor_name
    stripped = {line.strip() for line in _read_pf_conf().splitlines()}
    return any(line in stripped for line in _anchor_conf_lines(name))


def _pf_conf_wired(name: str) -> bool:
    """True iff BOTH pf.conf lines (anchor declaration + load directive) are present."""
    stripped = {line.strip() for line in _read_pf_conf().splitlines()}
    return all(line in stripped for line in _anchor_conf_lines(name))


def anchor_up_to_date(
    manifest: EngineManifest,
    *,
    subnets: tuple[str, ...] = DEFAULT_SUBNETS,
) -> bool:
    """True iff the anchor byte-matches AND both pf.conf lines are wired.

    The "nothing privileged to do" predicate — pure reads, callers use it to
    decide whether an install/reinstall will need the password-gated pf path.
    """
    return anchor_is_current(manifest, subnets=subnets) and _pf_conf_wired(
        manifest.firewall.anchor_name
    )


def build_anchor_content(
    manifest: EngineManifest,
    *,
    subnets: tuple[str, ...] = DEFAULT_SUBNETS,
) -> str:
    """Render the pf anchor body.

    Pure function — no I/O, no validation, just text. The result is fed to
    ``pfctl -nf`` for syntax validation before any real install.
    """
    if not ANCHOR_NAME_RE.match(manifest.firewall.anchor_name):
        raise FirewallError(
            f"Anchor name {manifest.firewall.anchor_name!r} must match com.asiai.<engine>"
        )
    if not subnets:
        raise FirewallError("subnets is empty; refusing to write a wide-open anchor")

    port = manifest.network.port
    subnet_list = ", ".join(subnets)
    return (
        f"# {manifest.display} firewall rules — generated by asiai-inference-server\n"
        f"# Allow LAN-only access on port {port}\n"
        "\n"
        f"pass in quick on lo0 proto tcp to any port {port} keep state\n"
        f"pass in quick inet proto tcp from {{ {subnet_list} }} "
        f"to any port {port} flags S/SA keep state\n"
        f"block in quick inet proto tcp to any port {port}\n"
        # No private-v6 allowlist is configured, so everything that is not
        # loopback (lo0 passes above) is blocked — an inet-only block left
        # the port wide open to any IPv6 peer on the LAN.
        f"block in quick inet6 proto tcp to any port {port}\n"
    )


def validate_anchor(anchor_text: str) -> None:
    """Run ``pfctl -nf -`` against the anchor body. Raises FirewallError on failure."""
    proc = subprocess.run(
        ["sudo", "/sbin/pfctl", "-nf", "-"],
        input=anchor_text,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise FirewallError(f"pfctl rejected anchor: {proc.stderr.strip() or proc.stdout.strip()}")


def install_anchor(
    manifest: EngineManifest,
    *,
    subnets: tuple[str, ...] = DEFAULT_SUBNETS,
    dry_run: bool = False,
) -> str:
    """Render, validate, and activate the pf anchor for this engine.

    Returns the absolute path of the installed anchor file.

    Sequence:
      1. Render content (pure).
      2. Validate via ``pfctl -nf -`` (fail fast on syntax errors).
      3. Write to a tmpfile, ``sudo /bin/mv`` to ``/etc/pf.anchors/<name>``.
      4. Append ``anchor "<name>" all`` to ``/etc/pf.conf`` if absent.
      5. ``sudo /sbin/pfctl -f /etc/pf.conf`` to load the new ruleset.
    """
    if not manifest.firewall.supported:
        raise FirewallError(
            f"{manifest.name}: firewall.supported=false — refusing to install anchor"
        )

    content = build_anchor_content(manifest, subnets=subnets)
    dst = anchor_path(manifest)

    # Idempotence: when the installed anchor already byte-matches AND both pf.conf
    # lines are wired, there is nothing privileged to do — return without touching
    # sudo. This is what makes the nominal reinstall (flags change, port/subnets
    # don't) run password-free end to end.
    if anchor_up_to_date(manifest, subnets=subnets):
        if dry_run:
            print(f"[dry-run] anchor {dst} already current + wired — nothing to do")
        return dst

    if dry_run:
        print(f"--- anchor {dst} ---")
        print(content, end="")
        print("--- end anchor ---")
        return dst

    preflight_sudo(f"installing the pf anchor {dst}")
    validate_anchor(content)

    with secure_staging_dir() as staging:
        tmp_path = staging / f"{manifest.firewall.anchor_name}.anchor"
        tmp_path.write_text(content, encoding="utf-8")
        try:
            subprocess.run(["sudo", "/bin/mkdir", "-p", PF_ANCHORS_DIR], check=True)
            subprocess.run(["sudo", "/bin/mv", str(tmp_path), dst], check=True)
            subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", dst], check=True)
            subprocess.run(["sudo", "/bin/chmod", "644", dst], check=True)
        except subprocess.CalledProcessError as e:
            raise FirewallError(f"Failed to install anchor {dst}: {e}") from e

    _ensure_anchor_in_pf_conf(manifest.firewall.anchor_name)
    _reload_pf()
    return dst


def remove_anchor(manifest: EngineManifest, *, dry_run: bool = False) -> bool:
    """Delete the anchor file and the matching line from ``/etc/pf.conf``.

    Returns True if anything was changed, False if the anchor was already gone.
    """
    dst = anchor_path(manifest)
    name = manifest.firewall.anchor_name
    changed = False

    if Path(dst).exists():
        if dry_run:
            print(f"[dry-run] sudo /bin/rm -f {dst}")
        else:
            subprocess.run(["sudo", "/bin/rm", "-f", dst], check=True)
        changed = True

    pf_conf_text = _read_pf_conf()
    line_re = re.compile(
        rf'^(?:anchor\s+"{re.escape(name)}"\s+all|load\s+anchor\s+"{re.escape(name)}"\s+from\s+"[^"]*")\s*$',
        re.MULTILINE,
    )
    if line_re.search(pf_conf_text):
        new_text = line_re.sub("", pf_conf_text)
        # Collapse the blank lines the substitutions leave behind.
        new_text = re.sub(r"\n{3,}", "\n\n", new_text)
        if dry_run:
            print(f"[dry-run] would rewrite {PF_CONF_PATH} (anchor + load lines removed)")
        else:
            _atomic_write_pf_conf(new_text)
            _reload_pf()
        changed = True

    return changed


def _read_pf_conf() -> str:
    try:
        return Path(PF_CONF_PATH).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _anchor_conf_lines(name: str) -> tuple[str, str]:
    """The pf.conf line PAIR that makes an anchor effective.

    The ``anchor`` line alone only DECLARES an attachment point — without
    the ``load anchor`` directive the rules file is never read and the
    firewall silently enforces nothing (the original bash had the same
    bug, inherited by the port and caught by the 2026-06-11 audit).
    """
    return (
        f'anchor "{name}" all',
        f'load anchor "{name}" from "{PF_ANCHORS_DIR}/{name}"',
    )


def _ensure_anchor_in_pf_conf(name: str) -> None:
    """Ensure both the anchor declaration and its load directive are in pf.conf."""
    if not Path(PF_CONF_PATH).exists():
        # A macOS host always ships /etc/pf.conf. Recreating it from scratch
        # would replace the system ruleset with our single anchor line.
        raise FirewallError(
            f"{PF_CONF_PATH} is missing — host in unexpected state, refusing "
            "to create it from scratch"
        )
    text = _read_pf_conf()
    stripped = {line.strip() for line in text.splitlines()}
    missing = [line for line in _anchor_conf_lines(name) if line not in stripped]
    if not missing:
        return
    new_text = text.rstrip() + "\n" + "\n".join(missing) + "\n"
    _atomic_write_pf_conf(new_text)


def _atomic_write_pf_conf(new_text: str) -> None:
    with secure_staging_dir() as staging:
        tmp_path = staging / "pf.conf"
        tmp_path.write_text(new_text, encoding="utf-8")
        try:
            subprocess.run(["sudo", "/bin/mv", str(tmp_path), PF_CONF_PATH], check=True)
            subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", PF_CONF_PATH], check=True)
            subprocess.run(["sudo", "/bin/chmod", "644", PF_CONF_PATH], check=True)
        except subprocess.CalledProcessError as e:
            raise FirewallError(f"Failed to update {PF_CONF_PATH}: {e}") from e


def _reload_pf() -> None:
    """Reload pf rules, then make sure pf is enabled.

    The ruleset load must NOT be best-effort: a silent failure here means
    install_anchor reports success while nothing was actually loaded.
    Only ``pfctl -e`` stays best-effort (it fails when pf is already on).
    """
    proc = subprocess.run(
        ["sudo", "/sbin/pfctl", "-f", PF_CONF_PATH],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FirewallError(
            f"pfctl -f {PF_CONF_PATH} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    subprocess.run(["sudo", "/sbin/pfctl", "-e"], check=False, capture_output=True)
