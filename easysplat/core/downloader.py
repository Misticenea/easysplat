"""Streaming HTTP downloads with byte-level progress, plus archive extraction."""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Callable

import httpx

# progress_cb(downloaded_bytes, total_bytes_or_None)
ProgressCallback = Callable[[int, int | None], None]

_CHUNK = 1 << 16


async def download_file(
    url: str,
    dest: Path,
    progress_cb: ProgressCallback | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` to ``dest`` atomically, reporting byte progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total = (
                int(response.headers["Content-Length"])
                if "Content-Length" in response.headers
                else None
            )
            done = 0
            with tmp.open("wb") as fh:
                async for chunk in response.aiter_bytes(_CHUNK):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress_cb is not None:
                        # decompressed bytes can exceed Content-Length for
                        # gzip-encoded responses; never overshoot the bar
                        progress_cb(min(done, total) if total else done, total)
    tmp.replace(dest)
    return dest


def extract_archive(archive: Path, dest_dir: Path) -> None:
    """Extract .zip / .tar.gz / .tar.bz2 archives (uv and micromamba releases)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest_dir, filter="data")
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")


def find_file(root: Path, filename: str) -> Path | None:
    """Locate ``filename`` anywhere under ``root`` (archives differ in layout)."""
    for candidate in root.rglob(filename):
        if candidate.is_file():
            return candidate
    return None


def install_binary(extracted_root: Path, binary_name: str, dest: Path) -> Path:
    found = find_file(extracted_root, binary_name)
    if found is None:
        raise FileNotFoundError(f"{binary_name} not found in downloaded archive")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, dest)
    dest.chmod(0o755)
    return dest
