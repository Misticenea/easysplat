from pathlib import Path

import easysplat.core.colmap as colmap_mod
from easysplat.core.catalog import INPUT_MULTI_IMAGE, INPUT_VIDEO
from easysplat.core.colmap import build_plan, feature_extractor_command, ffmpeg_command, matcher_command, mapper_command
from easysplat.core.gpu import GPUInfo, VENDOR_AMD, VENDOR_NVIDIA
from easysplat.core.inputs import ScanResult


def scan_for(kind: str, folder: Path, n_images: int = 1) -> ScanResult:
    images = [folder / f"{i:05d}.jpg" for i in range(n_images)]
    return ScanResult(
        folder=folder,
        kind=kind,
        images=images if kind != INPUT_VIDEO else [],
        video=folder / "v.mp4" if kind == INPUT_VIDEO else None,
    )


def test_video_plan_is_sequential(tmp_path, monkeypatch):
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_NVIDIA, ""))
    plan = build_plan(scan_for(INPUT_VIDEO, tmp_path))
    assert plan.sequential
    assert plan.use_gpu
    assert "sequential_matcher" in matcher_command(Path("colmap"), plan)


def test_small_photo_set_is_exhaustive_and_no_gpu_on_amd(tmp_path, monkeypatch):
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    plan = build_plan(scan_for(INPUT_MULTI_IMAGE, tmp_path, n_images=30))
    assert not plan.sequential
    assert not plan.use_gpu  # COLMAP SIFT GPU path is CUDA-only
    cmd = feature_extractor_command(Path("colmap"), plan)
    assert cmd[cmd.index("--SiftExtraction.use_gpu") + 1] == "0"
    assert "exhaustive_matcher" in matcher_command(Path("colmap"), plan)


def test_large_photo_set_uses_sequential_matching(tmp_path, monkeypatch):
    # exhaustive matching is O(n^2) — a 2000-image capture must not use it
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    plan = build_plan(scan_for(INPUT_MULTI_IMAGE, tmp_path, n_images=2000))
    assert plan.image_count == 2000
    assert plan.sequential
    assert "sequential_matcher" in matcher_command(Path("colmap"), plan)


def test_loose_images_used_in_place_not_duplicated(tmp_path, monkeypatch):
    # a folder of loose photos must be read where they are, not copied
    # into a new images/ subfolder
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    scan = ScanResult(
        folder=tmp_path,
        kind=INPUT_MULTI_IMAGE,
        images=[tmp_path / f"{i:05d}.jpg" for i in range(300)],
        image_dir=tmp_path,
    )
    plan = build_plan(scan)
    assert plan.image_dir == tmp_path  # not tmp_path / "images"
    cmd = feature_extractor_command(Path("colmap"), plan)
    assert str(tmp_path) in cmd


def test_images_subfolder_used_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    sub = tmp_path / "input"
    scan = ScanResult(
        folder=tmp_path,
        kind=INPUT_MULTI_IMAGE,
        images=[sub / "a.jpg", sub / "b.jpg"],
        image_dir=sub,
    )
    plan = build_plan(scan)
    assert plan.image_dir == sub


def test_video_extracts_into_images_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_NVIDIA, ""))
    plan = build_plan(scan_for(INPUT_VIDEO, tmp_path))
    assert plan.image_dir == tmp_path / "images"  # frames land here


def test_commands_reference_dataset_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(colmap_mod, "detect", lambda: GPUInfo(VENDOR_NVIDIA, ""))
    plan = build_plan(scan_for(INPUT_MULTI_IMAGE, tmp_path, n_images=10))
    mapper = mapper_command(Path("colmap"), plan)
    assert str(tmp_path / "images") in mapper
    assert str(tmp_path / "sparse") in mapper


def test_ffmpeg_command_extracts_frames(tmp_path):
    cmd = ffmpeg_command(Path("ffmpeg"), tmp_path / "v.mp4", tmp_path / "images", fps=2.0)
    assert "fps=2.0" in " ".join(cmd)
    assert str(tmp_path / "images" / "%05d.jpg") in cmd
