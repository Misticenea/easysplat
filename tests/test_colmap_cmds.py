from pathlib import Path

import easysplat.core.colmap as colmap_mod
from easysplat.core.catalog import INPUT_MULTI_IMAGE, INPUT_VIDEO
from easysplat.core.colmap import build_plan, feature_extractor_command, ffmpeg_command, matcher_command, mapper_command
from easysplat.core.gpu import GPUInfo, VENDOR_AMD, VENDOR_NVIDIA
from easysplat.core.inputs import ScanResult


def scan_for(kind: str, folder: Path) -> ScanResult:
    return ScanResult(folder=folder, kind=kind, video=folder / "v.mp4" if kind == INPUT_VIDEO else None)


def test_video_plan_is_sequential(tmp_path, monkeypatch):
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_NVIDIA, ""))
    plan = build_plan(scan_for(INPUT_VIDEO, tmp_path))
    assert plan.sequential
    assert plan.use_gpu
    assert "sequential_matcher" in matcher_command(Path("colmap"), plan)


def test_photo_plan_is_exhaustive_and_no_gpu_on_amd(tmp_path, monkeypatch):
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    plan = build_plan(scan_for(INPUT_MULTI_IMAGE, tmp_path))
    assert not plan.sequential
    assert not plan.use_gpu  # COLMAP SIFT GPU path is CUDA-only
    cmd = feature_extractor_command(Path("colmap"), plan)
    assert cmd[cmd.index("--SiftExtraction.use_gpu") + 1] == "0"
    assert "exhaustive_matcher" in matcher_command(Path("colmap"), plan)


def test_commands_reference_dataset_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_NVIDIA, ""))
    plan = build_plan(scan_for(INPUT_MULTI_IMAGE, tmp_path))
    mapper = mapper_command(Path("colmap"), plan)
    assert str(tmp_path / "images") in mapper
    assert str(tmp_path / "sparse") in mapper


def test_ffmpeg_command_extracts_frames(tmp_path):
    cmd = ffmpeg_command(Path("ffmpeg"), tmp_path / "v.mp4", tmp_path / "images", fps=2.0)
    assert "fps=2.0" in " ".join(cmd)
    assert str(tmp_path / "images" / "%05d.jpg") in cmd
