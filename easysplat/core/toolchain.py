"""Self-provisioning toolchain.

Nothing is assumed to exist on the host system: on first use EasySplat
downloads static builds of ``uv`` (standalone Python + venvs) and
``micromamba`` (conda-forge installs of colmap/ffmpeg/git) into
``~/.easysplat``. A system copy of a tool is preferred when present, so on a
normal dev box no downloads happen at all. This is what makes the app work on
immutable OSes with no Python, compilers or package manager access.
"""

from __future__ import annotations

import platform
import shutil
from pathlib import Path
from typing import Awaitable, Callable

from easysplat.core import paths
from easysplat.core.downloader import download_file, extract_archive, install_binary
from easysplat.core.proc import LineCallback, run_streaming, subprocess_env

# status_cb(message, fraction_or_None) — fraction is 0..1 for byte downloads
StatusCallback = Callable[[str, float | None], None]

_EXE = ".exe" if platform.system() == "Windows" else ""


def _uv_url() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Linux":
        target, ext = ("aarch64-unknown-linux-gnu" if arm else "x86_64-unknown-linux-gnu"), "tar.gz"
    elif system == "Darwin":
        target, ext = ("aarch64-apple-darwin" if arm else "x86_64-apple-darwin"), "tar.gz"
    elif system == "Windows":
        target, ext = ("aarch64-pc-windows-msvc" if arm else "x86_64-pc-windows-msvc"), "zip"
    else:
        raise RuntimeError(f"unsupported platform: {system}")
    return f"https://github.com/astral-sh/uv/releases/latest/download/uv-{target}.{ext}"


def _micromamba_url() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    plat = {
        "Linux": "linux-aarch64" if arm else "linux-64",
        "Darwin": "osx-arm64" if arm else "osx-64",
        "Windows": "win-64",
    }.get(system)
    if plat is None:
        raise RuntimeError(f"unsupported platform: {system}")
    return f"https://micro.mamba.pm/api/micromamba/{plat}/latest"


def find_tool(name: str) -> Path | None:
    """Look for a tool on the system PATH first, then in our managed prefix."""
    system = shutil.which(name)
    if system:
        return Path(system)
    for bin_dir in [paths.bin_dir(), *paths.tools_bin_dirs()]:
        candidate = bin_dir / f"{name}{_EXE}"
        if candidate.exists():
            return candidate
    return None


async def _fetch_binary(
    url: str, binary_name: str, status_cb: StatusCallback, archive_name: str
) -> Path:
    paths.ensure_dirs()
    archive = paths.cache_dir() / archive_name
    status_cb(f"Downloading {binary_name}…", 0.0)

    def on_bytes(done: int, total: int | None) -> None:
        status_cb(
            f"Downloading {binary_name}…", (done / total) if total else None
        )

    await download_file(url, archive, on_bytes)
    status_cb(f"Extracting {binary_name}…", None)
    extract_dir = paths.cache_dir() / f"{binary_name}-extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_archive(archive, extract_dir)
    dest = paths.bin_dir() / f"{binary_name}{_EXE}"
    install_binary(extract_dir, f"{binary_name}{_EXE}", dest)
    shutil.rmtree(extract_dir, ignore_errors=True)
    archive.unlink(missing_ok=True)
    return dest


async def ensure_uv(status_cb: StatusCallback) -> Path:
    existing = find_tool("uv")
    if existing:
        return existing
    ext = "zip" if platform.system() == "Windows" else "tar.gz"
    return await _fetch_binary(_uv_url(), "uv", status_cb, f"uv.{ext}")


async def ensure_micromamba(status_cb: StatusCallback) -> Path:
    existing = find_tool("micromamba")
    if existing:
        return existing
    return await _fetch_binary(
        _micromamba_url(), "micromamba", status_cb, "micromamba.tar.bz2"
    )


async def ensure_conda_tool(
    name: str,
    status_cb: StatusCallback,
    on_line: LineCallback | None = None,
) -> Path:
    """Return a usable ``name`` binary, installing it from conda-forge if needed."""
    existing = find_tool(name)
    if existing:
        return existing
    mamba = await ensure_micromamba(status_cb)
    status_cb(f"Installing {name} (conda-forge)…", None)
    prefix = paths.tools_prefix()
    prefix.mkdir(parents=True, exist_ok=True)
    await run_streaming(
        [
            str(mamba),
            "install", "--yes",
            "--prefix", str(prefix),
            "--channel", "conda-forge",
            "--root-prefix", str(paths.data_root() / "mamba-root"),
            name,
        ],
        on_line=on_line,
    )
    installed = find_tool(name)
    if installed is None:
        raise RuntimeError(f"micromamba reported success but {name} was not found")
    return installed


async def ensure_git(status_cb: StatusCallback, on_line: LineCallback | None = None) -> Path:
    return await ensure_conda_tool("git", status_cb, on_line)


async def ensure_ffmpeg(status_cb: StatusCallback, on_line: LineCallback | None = None) -> Path:
    return await ensure_conda_tool("ffmpeg", status_cb, on_line)


async def ensure_colmap(status_cb: StatusCallback, on_line: LineCallback | None = None) -> Path:
    return await ensure_conda_tool("colmap", status_cb, on_line)


def tool_env() -> dict[str, str]:
    """Environment with our managed bin dirs on PATH, for child processes."""
    return subprocess_env(extra_paths=[paths.bin_dir(), *paths.tools_bin_dirs()])


def toolchain_summary() -> dict[str, str | None]:
    """For the status bar: which tools resolve, and to where."""
    return {
        name: (str(p) if (p := find_tool(name)) else None)
        for name in ("uv", "micromamba", "git", "ffmpeg", "colmap")
    }


Ensurer = Callable[[StatusCallback], Awaitable[Path]]
