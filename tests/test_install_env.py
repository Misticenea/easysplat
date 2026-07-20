import easysplat.core.installer as installer_mod
from easysplat.core.catalog import get_model
from easysplat.core.gpu import GPUInfo, VENDOR_AMD, VENDOR_APPLE, VENDOR_NVIDIA
from easysplat.core.installer import install_env


def test_amd_gets_rocm_backend_only(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    env = install_env(get_model("sharp"))
    # torch-family packages come from the ROCm index; nothing else is
    # redirected, so no nvidia-* CUDA wheels are ever pulled on AMD
    assert env["UV_TORCH_BACKEND"] == "rocm6.2"
    assert "UV_EXTRA_INDEX_URL" not in env
    assert "UV_INDEX_STRATEGY" not in env
    assert "rocm" in env["PIP_EXTRA_INDEX_URL"]


def test_nvidia_gets_auto_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_NVIDIA, ""))
    env = install_env(get_model("sharp"))
    assert env["UV_TORCH_BACKEND"] == "auto"
    assert "cu12" in env["PIP_EXTRA_INDEX_URL"]


def test_apple_uses_default_index(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_APPLE, "MPS"))
    env = install_env(get_model("sharp"))
    assert "UV_TORCH_BACKEND" not in env
    assert "PIP_EXTRA_INDEX_URL" not in env


def test_no_torch_backend_for_binary_models(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    env = install_env(get_model("brush"))
    assert "UV_TORCH_BACKEND" not in env
    assert "PIP_EXTRA_INDEX_URL" not in env


def test_amd_on_windows_falls_back_to_cpu(monkeypatch, tmp_path):
    import platform

    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    env = install_env(get_model("sharp"))
    assert env["UV_TORCH_BACKEND"] == "cpu"  # no ROCm wheels on Windows
