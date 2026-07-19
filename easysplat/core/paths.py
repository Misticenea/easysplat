"""Filesystem layout for everything EasySplat writes.

All state lives under ``~/.easysplat`` so the app works on immutable
distributions (Fedora Silverblue, SteamOS, ...) where only the home
directory is writable.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    override = os.environ.get("EASYSPLAT_HOME")
    root = Path(override).expanduser() if override else Path.home() / ".easysplat"
    return root


def bin_dir() -> Path:
    return data_root() / "bin"


def tools_prefix() -> Path:
    """Conda-style prefix where micromamba installs colmap/ffmpeg/git."""
    return data_root() / "tools"


def tools_bin_dirs() -> list[Path]:
    prefix = tools_prefix()
    return [prefix / "bin", prefix / "Library" / "bin", prefix / "Scripts"]


def models_dir() -> Path:
    return data_root() / "models"


def model_dir(model_id: str) -> Path:
    return models_dir() / model_id


def cache_dir() -> Path:
    return data_root() / "cache"


def ensure_dirs() -> None:
    for d in (data_root(), bin_dir(), models_dir(), cache_dir()):
        d.mkdir(parents=True, exist_ok=True)
