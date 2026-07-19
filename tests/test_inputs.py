from pathlib import Path

from easysplat.core.catalog import INPUT_MULTI_IMAGE, INPUT_SINGLE_IMAGE, INPUT_VIDEO
from easysplat.core.inputs import scan_folder


def make(folder: Path, *names: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")
    return folder


def test_single_image(tmp_path):
    scan = scan_folder(make(tmp_path / "d", "photo.jpg"))
    assert scan.kind == INPUT_SINGLE_IMAGE
    assert scan.images[0].name == "photo.jpg"


def test_multi_image(tmp_path):
    scan = scan_folder(make(tmp_path / "d", "a.png", "b.jpg", "c.jpeg"))
    assert scan.kind == INPUT_MULTI_IMAGE
    assert len(scan.images) == 3


def test_video_wins_over_images(tmp_path):
    scan = scan_folder(make(tmp_path / "d", "clip.mp4", "thumb.jpg"))
    assert scan.kind == INPUT_VIDEO
    assert scan.video is not None and scan.video.name == "clip.mp4"


def test_images_subfolder(tmp_path):
    root = tmp_path / "d"
    make(root / "images", "a.jpg", "b.jpg")
    root.mkdir(exist_ok=True)
    scan = scan_folder(root)
    assert scan.kind == INPUT_MULTI_IMAGE
    assert scan.image_dir == root / "images"


def test_empty_folder(tmp_path):
    scan = scan_folder(make(tmp_path / "d"))
    assert scan.kind is None
    assert "no images" in scan.summary


def test_missing_folder(tmp_path):
    scan = scan_folder(tmp_path / "nope")
    assert scan.kind is None


def test_colmap_detection(tmp_path):
    root = make(tmp_path / "d", "a.jpg", "b.jpg")
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(b"x")
    scan = scan_folder(root)
    assert scan.has_colmap


def test_uppercase_extensions(tmp_path):
    root = tmp_path / "d"
    root.mkdir()
    (root / "IMG_001.JPG").write_bytes(b"x")
    scan = scan_folder(root)
    assert scan.kind == INPUT_SINGLE_IMAGE
