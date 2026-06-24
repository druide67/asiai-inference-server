"""Tests for ais_core.memory — vm_stat parsing, lock semantics, repair."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ais_core import memory, privhelper
from ais_core.memory import (
    OperationsLock,
    PurgeReport,
    VmStat,
    vm_stat_parse,
)

# ---------------------------------------------------------------------------
# vm_stat parsing
# ---------------------------------------------------------------------------

REAL_VM_STAT_OUTPUT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                12345.
Pages active:                             567890.
Pages inactive:                           234567.
Pages speculative:                          1234.
Pages throttled:                               0.
Pages wired down:                         100000.
Pages purgeable:                            5678.
"Translation faults":                  9876543210.
Pages copy-on-write:                       12345.
Pages zero filled:                       1234567.
Pages reactivated:                          1234.
Pages purged:                                  0.
File-backed pages:                        100000.
Anonymous pages:                          800000.
Pages stored in compressor:               300000.
Pages occupied by compressor:             150000.
Decompressions:                              123.
Compressions:                                456.
Pageins:                                     789.
Pageouts:                                      0.
Swapins:                                       0.
Swapouts:                                      0.
"""


def test_vm_stat_parse_extracts_page_size_and_counters() -> None:
    s = vm_stat_parse(REAL_VM_STAT_OUTPUT)
    assert s.page_size_bytes == 16384
    assert s.pages_free == 12345
    assert s.pages_active == 567890
    assert s.pages_wired == 100000
    assert s.pages_compressed == 150000


def test_vm_stat_total_active_bytes_includes_compressor() -> None:
    """The compressor ballast is exactly what we want to see drop after purge."""
    s = vm_stat_parse(REAL_VM_STAT_OUTPUT)
    expected = (567890 + 100000 + 150000) * 16384
    assert s.total_active_bytes == expected


def test_vm_stat_parse_tolerates_missing_keys() -> None:
    s = vm_stat_parse("page size of 16384 bytes\nPages free: 100.")
    assert s.pages_free == 100
    assert s.pages_active == 0  # missing but not an error
    assert s.pages_compressed == 0


def test_vm_stat_parse_falls_back_to_default_page_size() -> None:
    """Older macOS versions might not include the 'page size of N' line."""
    s = vm_stat_parse("Pages free: 100.\n")
    assert s.page_size_bytes == 16384  # PAGE_SIZE_DEFAULT


def test_purge_report_freed_mb_signed_delta() -> None:
    """Negative deltas are reported as-is — caller decides how to display."""
    before = VmStat(16384, 0, 1000, 0, 0, 0)
    after = VmStat(16384, 0, 500, 0, 0, 0)
    rep = PurgeReport(before=before, after=after, pressure_after="normal", elapsed_s=0.1)
    # 500 pages * 16384 / (1024*1024) ≈ 7.8 MB → 7
    assert rep.freed_mb == 7

    # Reverse: more activity after purge (other process allocated).
    rep_neg = PurgeReport(before=after, after=before, pressure_after="normal", elapsed_s=0.1)
    assert rep_neg.freed_mb == -8


# ---------------------------------------------------------------------------
# memory_pressure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("System-wide memory free percentage: 80%\nMemory status: normal\n", "normal"),
        ("Memory status: warning\n", "warn"),
        ("Memory status: critical\n", "critical"),
        ("totally_unparseable_output", "unknown"),
    ],
)
def test_memory_pressure_keywords(stdout: str, expected: str) -> None:
    fake = MagicMock(returncode=0, stdout=stdout, stderr="")
    with patch("ais_core.memory.subprocess.run", return_value=fake):
        assert memory.memory_pressure() == expected


def test_memory_pressure_unknown_when_binary_missing() -> None:
    with patch("ais_core.memory.subprocess.run", side_effect=FileNotFoundError):
        assert memory.memory_pressure() == "unknown"


# ---------------------------------------------------------------------------
# purge_memory — fully mocked subprocess
# ---------------------------------------------------------------------------


def test_purge_memory_runs_helper_purge() -> None:
    fake_vm = VmStat(16384, 1000, 5000, 0, 1000, 2000)
    with (
        patch("ais_core.memory.vm_stat_parse", return_value=fake_vm),
        patch("ais_core.memory.privhelper.run") as mock_run,
        patch("ais_core.memory.memory_pressure", return_value="normal"),
    ):
        rep = memory.purge_memory()

    # Routed through the helper (NOPASSWD on the helper only), not raw `sudo /usr/sbin/purge`.
    assert mock_run.call_args.args[0] == "purge"
    assert mock_run.call_args.kwargs.get("timeout") == 30
    assert rep.before == fake_vm
    assert rep.after == fake_vm
    assert rep.pressure_after == "normal"


def test_purge_memory_dry_run_uses_helper_dry_run() -> None:
    fake_vm = VmStat(16384, 1000, 5000, 0, 1000, 2000)
    with (
        patch("ais_core.memory.vm_stat_parse", return_value=fake_vm),
        patch("ais_core.memory.privhelper.run") as mock_run,
        patch("ais_core.memory.memory_pressure", return_value="normal"),
    ):
        rep = memory.purge_memory(dry_run=True)

    mock_run.assert_called_once_with("purge", dry_run=True)
    assert isinstance(rep, PurgeReport)


def test_purge_memory_wraps_helper_failure() -> None:
    fake_vm = VmStat(16384, 1000, 5000, 0, 1000, 2000)
    with (
        patch("ais_core.memory.vm_stat_parse", return_value=fake_vm),
        patch(
            "ais_core.memory.privhelper.run",
            side_effect=privhelper.PrivHelperError("asiai-priv purge failed — refused"),
        ),
        patch("ais_core.memory.memory_pressure", return_value="normal"),
        pytest.raises(memory.MemoryError_, match="helper purge failed"),
    ):
        memory.purge_memory()


# ---------------------------------------------------------------------------
# OperationsLock — real fcntl, real children
# ---------------------------------------------------------------------------


def test_operations_lock_acquired_and_released(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    lock = OperationsLock(path=lock_path)
    lock.acquire()
    try:
        assert lock_path.exists()
        assert lock_path.read_text().strip() == str(os.getpid())
    finally:
        lock.release()


def test_operations_lock_second_acquire_raises_when_held(tmp_path: Path) -> None:
    """Two locks on the same path: the second must fail with MemoryError_."""
    lock_path = tmp_path / "test.lock"
    lock1 = OperationsLock(path=lock_path)
    lock2 = OperationsLock(path=lock_path)
    lock1.acquire()
    try:
        with pytest.raises(memory.MemoryError_, match="held by another process"):
            lock2.acquire()
    finally:
        lock1.release()


def test_operations_lock_force_bypasses_acquisition(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    lock1 = OperationsLock(path=lock_path)
    forced = OperationsLock(path=lock_path, force=True)
    lock1.acquire()
    try:
        # Should not raise even though lock1 holds the file.
        with forced:
            pass
    finally:
        lock1.release()


def test_operations_lock_released_after_context_exit(tmp_path: Path) -> None:
    """Lock must be released even if the inner block raises."""
    lock_path = tmp_path / "test.lock"
    with pytest.raises(RuntimeError, match="boom"), OperationsLock(path=lock_path):
        raise RuntimeError("boom")
    # A fresh lock on the same path must succeed now.
    OperationsLock(path=lock_path).acquire()


def _hold_lock_until_released(lock_path: str, acquired_evt, release_evt) -> None:
    lock = OperationsLock(path=Path(lock_path))
    lock.acquire()
    acquired_evt.set()
    release_evt.wait(timeout=30.0)
    lock.release()


def test_operations_lock_cross_process(tmp_path: Path) -> None:
    """Real subprocess holding the lock blocks the parent's acquisition.

    Synchronized with Events, not sleeps: macOS multiprocessing uses spawn,
    and the child re-imports the whole module tree before acquiring — on a
    slow CI runner that takes longer than any fixed sleep, letting the
    parent acquire first and turning the expected raise into a flake.
    """
    lock_path = tmp_path / "cross.lock"
    acquired = multiprocessing.Event()
    release = multiprocessing.Event()
    proc = multiprocessing.Process(
        target=_hold_lock_until_released, args=(str(lock_path), acquired, release)
    )
    proc.start()
    try:
        assert acquired.wait(timeout=30.0), "child never acquired the lock"
        with pytest.raises(memory.MemoryError_):
            OperationsLock(path=lock_path).acquire()
    finally:
        release.set()
        proc.join(timeout=30.0)


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------


def test_repair_clears_stale_lock_when_pid_dead(tmp_path: Path, monkeypatch) -> None:
    """A stale lock holding a long-dead PID must be removed."""
    lock_path = tmp_path / "stale.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("999999999\n")  # almost certainly dead PID

    monkeypatch.setattr(memory, "OPERATIONS_LOCK_PATH", lock_path)
    monkeypatch.setattr(memory, "LAUNCH_DAEMONS_DIR", tmp_path / "no_such_dir")

    report = memory.repair()
    assert report.stale_lock_cleared is True
    assert not lock_path.exists()


def test_repair_keeps_live_lock(tmp_path: Path, monkeypatch) -> None:
    """A live PID's lock must NOT be cleared."""
    lock_path = tmp_path / "live.lock"
    lock_path.write_text(f"{os.getpid()}\n")

    monkeypatch.setattr(memory, "OPERATIONS_LOCK_PATH", lock_path)
    monkeypatch.setattr(memory, "LAUNCH_DAEMONS_DIR", tmp_path / "no_such_dir")

    report = memory.repair()
    assert report.stale_lock_cleared is False
    assert lock_path.exists()


def test_repair_lists_orphan_plists(tmp_path: Path, monkeypatch) -> None:
    """A com.asiai.* plist not declared in any manifest should be reported."""
    fake_launchd = tmp_path / "LaunchDaemons"
    fake_launchd.mkdir()
    orphan = fake_launchd / "com.asiai.ghost.plist"
    orphan.write_text("<?xml version='1.0'?>")
    legit = fake_launchd / "com.asiai.ollama.plist"  # known
    legit.write_text("<?xml version='1.0'?>")

    monkeypatch.setattr(memory, "LAUNCH_DAEMONS_DIR", fake_launchd)
    # Use a never-existing path so the lock branch is a no-op.
    monkeypatch.setattr(memory, "OPERATIONS_LOCK_PATH", tmp_path / "no_such_lock_file")

    report = memory.repair()
    assert any("com.asiai.ghost.plist" in p for p in report.orphan_plists)
    assert all("com.asiai.ollama.plist" not in p for p in report.orphan_plists)
