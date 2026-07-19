"""Async subprocess helpers shared by the toolchain, installer and trainer."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable, Sequence

LineCallback = Callable[[str], None]


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
        env=env,
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
    """Copy of os.environ with extra directories prepended to PATH."""
    env = dict(os.environ)
    if extra_paths:
        env["PATH"] = os.pathsep.join(
            [str(p) for p in extra_paths] + [env.get("PATH", "")]
        )
    env.update(overrides)
    return env
