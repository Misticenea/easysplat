"""Classify what the user's chosen folder contains.

A dataset folder may hold a single image, many images, or a video (directly
or inside an ``images``/``input`` subfolder). If a COLMAP reconstruction is
already present we skip building one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from easysplat.core.catalog import INPUT_MULTI_IMAGE, INPUT_SINGLE_IMAGE, INPUT_VIDEO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

_IMAGE_SUBDIRS = ("images", "input", "frames")


@dataclass
class ScanResult:
    folder: Path
    kind: str | None  # one of catalog INPUT_* constants, or None if empty
    images: list[Path] = field(default_factory=list)
    video: Path | None = None
    image_dir: Path | None = None
    has_colmap: bool = False

    @property
    def summary(self) -> str:
        if self.kind == INPUT_VIDEO:
            assert self.video is not None
            return f"video: {self.video.name}"
        if self.kind == INPUT_SINGLE_IMAGE:
            return f"1 image: {self.images[0].name}"
        if self.kind == INPUT_MULTI_IMAGE:
            return f"{len(self.images)} images"
        return "no images or videos found"


def effective_image_dir(scan: "ScanResult") -> Path:
    """Where the images actually live.

    Uses the folder the images were found in so COLMAP can read them in
    place — no duplicate ``images/`` copy. For video (no images yet) frames
    are extracted into ``<folder>/images``.
    """
    if scan.image_dir is not None:
        return scan.image_dir
    return scan.folder / "images"


def _list_media(folder: Path) -> tuple[list[Path], list[Path]]:
    images, videos = [], []
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return [], []
    for entry in entries:
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(entry)
        elif ext in VIDEO_EXTS:
            videos.append(entry)
    return images, videos


def has_colmap_data(folder: Path) -> bool:
    for sparse in (folder / "sparse" / "0", folder / "sparse"):
        if any((sparse / f"cameras{ext}").exists() for ext in (".bin", ".txt")):
            return True
    return (folder / "transforms.json").exists()


def scan_folder(folder: Path) -> ScanResult:
    folder = folder.expanduser()
    result = ScanResult(folder=folder, kind=None)
    if not folder.is_dir():
        return result
    result.has_colmap = has_colmap_data(folder)

    images, videos = _list_media(folder)
    image_dir = folder
    if not images and not videos:
        for sub in _IMAGE_SUBDIRS:
            sub_images, sub_videos = _list_media(folder / sub)
            if sub_images or sub_videos:
                images, videos = sub_images, sub_videos
                image_dir = folder / sub
                break

    if videos:
        result.kind = INPUT_VIDEO
        result.video = videos[0]
    elif len(images) == 1:
        result.kind = INPUT_SINGLE_IMAGE
        result.images = images
        result.image_dir = image_dir
    elif len(images) > 1:
        result.kind = INPUT_MULTI_IMAGE
        result.images = images
        result.image_dir = image_dir
    return result
