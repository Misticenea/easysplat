"""Async subprocess helpers shared by the toolchain, installer and trainer."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

LineCallback = Callable[[str], None]

_LIB_PATH_VARS = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH")


def clean_base_env() -> dict[str, str]:
    """os.environ with PyInstaller's bundled-library paths removed.

    One-file builds extract to a temp dir and point LD_LIBRARY_PATH at it so
    the app itself can load its bundled libraries — but child processes
    (git, ffmpeg, colmap, uv, trainers) must load the *system* libraries, or
    they crash with symbol-version mismatches like
    "libssl.so.3: version 'OPENSSL_3.2.0' not found".
    """
    env = dict(os.environ)
    if not getattr(sys, "frozen", False):
        return env
    meipass = getattr(sys, "_MEIPASS", "")
    for var in _LIB_PATH_VARS:
        # the bootloader saves the pre-launch value in <VAR>_ORIG
        orig_key = f"{var}_ORIG"
        if orig_key in env:
            orig = env.pop(orig_key)
            if orig:
                env[var] = orig
            else:
                env.pop(var, None)
        elif var in env:
            parts = [p for p in env[var].split(os.pathsep) if p and p != meipass]
            if parts:
                env[var] = os.pathsep.join(parts)
            else:
                env.pop(var)
    return env


class CommandError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, tail: str = ""):
        self.cmd = list(cmd)
        self.returncode = returncode
        self.tail = tail
        super().__init__(
            f"command failed ({returncode}): {' '.join(str(c) for c in cmd)}"
            + (f"\n{tail}" if tail else "")
        )


async def run_streaming(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_line: LineCallback | None = None,
    check: bool = True,
) -> int:
    """Run a command, streaming merged stdout/stderr line by line to ``on_line``."""
    process = await asyncio.create_subprocess_exec(
        *[str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env if env is not None else clean_base_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
    )
    tail: list[str] = []
    assert process.stdout is not None
    while True:
        raw = await process.stdout.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        tail.append(line)
        if len(tail) > 30:
            tail.pop(0)
        if on_line is not None:
            on_line(line)
    returncode = await process.wait()
    if check and returncode != 0:
        raise CommandError(cmd, returncode, "\n".join(tail))
    return returncode


def subprocess_env(extra_paths: Sequence[Path] = (), **overrides: str) -> dict[str, str]:
    """Clean child-process environment with extra directories prepended to PATH."""
    env = clean_base_env()
    if extra_paths:
        env["PATH"] = os.pathsep.join(
            [str(p) for p in extra_paths] + [env.get("PATH", "")]
        )
    env.update(overrides)
    return env
