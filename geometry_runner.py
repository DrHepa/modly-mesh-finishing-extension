# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated, bounded Blender subprocess for read-only geometry diagnosis."""

from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import signal
import subprocess
import threading
import time


SCHEMA_VERSION = "1.0.0"
DEFAULT_TIMEOUT_SECONDS = 600
SHUTDOWN_GRACE_SECONDS = 3


class GeometryError(ValueError):
    pass


class GeometryCancelled(GeometryError):
    pass


def _find_windows_blender(configured: str | None, search_path: str) -> Path:
    """Search explicit absolute locations only; never CWD or PATHEXT."""
    if configured:
        requested = PureWindowsPath(configured)
        if not requested.is_absolute() or requested.suffix.lower() != ".exe":
            raise GeometryError("BLENDER_EXECUTABLE must be a fully absolute .exe path on Windows")
        candidates = [requested]
    else:
        candidates = (PureWindowsPath(entry) / "blender.exe" for entry in search_path.split(";")
                      if PureWindowsPath(entry).is_absolute())
    for candidate in candidates:
        try:
            path = Path(str(candidate)).resolve(strict=True)
            if path.is_absolute() and path.suffix.lower() == ".exe" and path.is_file() and os.access(path, os.X_OK):
                return path
        except (OSError, RuntimeError):
            if configured:
                raise GeometryError("BLENDER_EXECUTABLE must resolve to an executable .exe file") from None
        if configured:
            raise GeometryError("BLENDER_EXECUTABLE must resolve to an executable .exe file")
    raise GeometryError("Blender executable not found; set absolute BLENDER_EXECUTABLE or add an absolute PATH directory containing blender.exe")


def find_blender() -> Path:
    configured = os.environ.get("BLENDER_EXECUTABLE")
    if os.name == "nt":
        return _find_windows_blender(configured, os.environ.get("PATH", ""))
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            raise GeometryError("BLENDER_EXECUTABLE must be an absolute executable path")
    else:
        found = shutil.which("blender")
        if not found:
            raise GeometryError("Blender executable not found; set absolute BLENDER_EXECUTABLE or add Blender to PATH")
        path = Path(found)
    path = path.resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise GeometryError("Blender path is not an executable file")
    return path


def _posix_child_exited_without_reaping(pid: int) -> bool:
    """Keep the session leader's PID reserved until its process group is cleaned.

    Popen.poll() reaps the child. If the leader exited but left descendants,
    reaping it before killpg permits numeric PGID reuse and a dangerous signal
    to an unrelated future group. WNOWAIT observes exit without freeing PID.
    """
    while True:
        try:
            status = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            return status is not None and status.si_pid == pid
        except InterruptedError:
            continue


def _live_linux_group_descendants(pgid: int, leader_pid: int) -> set[int] | None:
    """Best-effort warning evidence; group kill remains unconditional.

    Linux /proc exposes live same-PGID descendants. None means visibility was
    incomplete, not that the process group was clean.
    """
    if not Path("/proc").is_dir():
        return None
    members: set[int] = set()
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return None
    incomplete = False
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        if pid == leader_pid:
            continue
        try:
            stat_text = (Path(entry.path) / "stat").read_text(encoding="ascii")
        except FileNotFoundError:
            continue
        except OSError:
            incomplete = True
            continue
        closing = stat_text.rfind(")")
        if closing < 0:
            incomplete = True
            continue
        fields = stat_text[closing + 2:].split()
        if len(fields) < 3:
            incomplete = True
            continue
        if fields[0] != "Z" and int(fields[2]) == pgid:
            members.add(pid)
    return None if incomplete else members


def _terminate_tree(process: subprocess.Popen) -> set[int] | None:
    """Kill the dedicated session group *before* reaping its leader."""
    if os.name == "posix":
        descendants = _live_linux_group_descendants(process.pid, process.pid)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        return descendants
    # Native Windows descendant containment is not established by this runner.
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)
    return None


