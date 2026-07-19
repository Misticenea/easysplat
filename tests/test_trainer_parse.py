from pathlib import Path

from easysplat.core import paths
from easysplat.core.catalog import INPUT_MULTI_IMAGE, INPUT_SINGLE_IMAGE, get_model
from easysplat.core.inputs import ScanResult
from easysplat.core.installer import expand_command, venv_python
from easysplat.core.trainer import TrainRequest, build_train_command


def test_expand_command_pip_token(tmp_path):
    cmd = expand_command(
        ("{pip}", "-r", "{repo}/requirements.txt"),
        {"repo": "/r"},
        uv=Path("/bin/uv"),
        python=Path("/venv/bin/python"),
    )
    assert cmd == ["/bin/uv", "pip", "install", "--python", "/venv/bin/python", "-r", "/r/requirements.txt"]


def test_expand_command_script_token(tmp_path):
    python = tmp_path / "bin" / "python"
    python.parent.mkdir(parents=True)
    script = python.parent / "sharp"
    script.write_text("#!/bin/sh\n")
    cmd = expand_command(("{script:sharp}", "predict"), {}, uv=Path("uv"), python=python)
    assert cmd == [str(script), "predict"]


def test_build_train_command_single_image(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path / "home"))
    spec = get_model("sharp")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image = dataset / "photo.jpg"
    image.write_bytes(b"x")
    scan = ScanResult(folder=dataset, kind=INPUT_SINGLE_IMAGE, images=[image])
    request = TrainRequest(spec=spec, scan=scan)
    cmd = build_train_command(request, uv=Path("uv"))
    assert str(image) in cmd
    assert str(dataset / "output" / "sharp") in cmd
    assert request.output_dir == dataset / "output" / "sharp"


def test_build_train_command_dataset_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path / "home"))
    spec = get_model("gaussian-splatting-inria")
    dataset = tmp_path / "scene"
    dataset.mkdir()
    scan = ScanResult(folder=dataset, kind=INPUT_MULTI_IMAGE)
    cmd = build_train_command(TrainRequest(spec=spec, scan=scan), uv=Path("uv"))
    assert str(dataset) in cmd
    assert cmd[0] == str(venv_python(spec))
    assert paths.data_root() == tmp_path / "home"
