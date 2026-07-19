"""Automatic COLMAP dataset preparation.

Given a folder with a video or loose images, produce the standard layout the
splat trainers expect::

    dataset/
      images/          extracted or copied frames
      sparse/0/        COLMAP reconstruction (cameras.bin, images.bin, points3D.bin)

Stages: [extract frames] -> feature_extractor -> matcher -> mapper.
Progress is reported per stage; raw tool output streams to the log.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from easysplat.core import toolchain
from easysplat.core.catalog import INPUT_VIDEO
from easysplat.core.gpu import VENDOR_NVIDIA, detect
from easysplat.core.inputs import ScanResult, has_colmap_data
from easysplat.core.proc import LineCallback, run_streaming

# stage_cb(stage_label, stage_index, stage_count)
StageCallback = Callable[[str, int, int], None]


@dataclass
class ColmapPlan:
    dataset: Path
    image_dir: Path
    db_path: Path
    sparse_dir: Path
    sequential: bool  # video frames match sequentially; photo sets exhaustively
    use_gpu: bool


def build_plan(scan: ScanResult) -> ColmapPlan:
    dataset = scan.folder
    return ColmapPlan(
        dataset=dataset,
        image_dir=dataset / "images",
        db_path=dataset / "colmap.db",
        sparse_dir=dataset / "sparse",
        sequential=scan.kind == INPUT_VIDEO,
        # COLMAP's SIFT GPU path is CUDA-only.
        use_gpu=detect().vendor == VENDOR_NVIDIA,
    )


def ffmpeg_command(ffmpeg: Path, video: Path, image_dir: Path, fps: float = 2.0) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner", "-y",
        "-i", str(video),
        "-qscale:v", "2",
        "-vf", f"fps={fps}",
        str(image_dir / "%05d.jpg"),
    ]


def feature_extractor_command(colmap: Path, plan: ColmapPlan) -> list[str]:
    return [
        str(colmap), "feature_extractor",
        "--database_path", str(plan.db_path),
        "--image_path", str(plan.image_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "1" if plan.sequential else "0",
        "--SiftExtraction.use_gpu", "1" if plan.use_gpu else "0",
    ]


def matcher_command(colmap: Path, plan: ColmapPlan) -> list[str]:
    matcher = "sequential_matcher" if plan.sequential else "exhaustive_matcher"
    return [
        str(colmap), matcher,
        "--database_path", str(plan.db_path),
        "--SiftMatching.use_gpu", "1" if plan.use_gpu else "0",
    ]


def mapper_command(colmap: Path, plan: ColmapPlan) -> list[str]:
    return [
        str(colmap), "mapper",
        "--database_path", str(plan.db_path),
        "--image_path", str(plan.image_dir),
        "--output_path", str(plan.sparse_dir),
    ]


async def prepare_dataset(
    scan: ScanResult,
    stage_cb: StageCallback,
    on_line: LineCallback,
    status_cb: toolchain.StatusCallback,
) -> Path:
    """Ensure ``scan.folder`` has images/ + sparse/0; return the dataset path."""
    if scan.has_colmap or has_colmap_data(scan.folder):
        on_line("COLMAP data already present — skipping reconstruction.")
        return scan.folder

    plan = build_plan(scan)
    stages = 4 if plan.sequential else 3
    stage = 0

    plan.image_dir.mkdir(parents=True, exist_ok=True)
    if scan.kind == INPUT_VIDEO:
        assert scan.video is not None
        stage_cb("Extracting frames from video", stage, stages)
        ffmpeg = await toolchain.ensure_ffmpeg(status_cb, on_line)
        await run_streaming(
            ffmpeg_command(ffmpeg, scan.video, plan.image_dir), on_line=on_line
        )
        stage += 1
    elif scan.image_dir is not None and scan.image_dir != plan.image_dir:
        for img in scan.images:
            shutil.copy2(img, plan.image_dir / img.name)
    elif scan.image_dir == scan.folder:
        # loose images in the dataset root: copy into images/ for the trainers
        for img in scan.images:
            target = plan.image_dir / img.name
            if not target.exists():
                shutil.copy2(img, target)

    colmap = await toolchain.ensure_colmap(status_cb, on_line)
    plan.sparse_dir.mkdir(parents=True, exist_ok=True)

    stage_cb("COLMAP: extracting features", stage, stages)
    await run_streaming(feature_extractor_command(colmap, plan), on_line=on_line)
    stage += 1

    stage_cb("COLMAP: matching features", stage, stages)
    await run_streaming(matcher_command(colmap, plan), on_line=on_line)
    stage += 1

    stage_cb("COLMAP: mapping (sparse reconstruction)", stage, stages)
    await run_streaming(mapper_command(colmap, plan), on_line=on_line)

    if not has_colmap_data(scan.folder):
        raise RuntimeError(
            "COLMAP finished but produced no reconstruction — "
            "the images may lack overlap or texture."
        )
    return scan.folder
