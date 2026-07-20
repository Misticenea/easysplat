import easysplat.core.installer as installer_mod
from easysplat.core.catalog import get_model
from easysplat.core.gpu import GPUInfo, VENDOR_AMD, VENDOR_APPLE, VENDOR_NVIDIA
from easysplat.core.installer import install_env


def test_torch_index_set_with_best_match_strategy(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    env = install_env(get_model("sharp"))
    assert "rocm" in env["UV_EXTRA_INDEX_URL"]
    # the PyTorch index hosts old certifi/numpy versions; without
    # best-match uv pins to them and resolution fails
    assert env["UV_INDEX_STRATEGY"] == "unsafe-best-match"
    assert env["PIP_EXTRA_INDEX_URL"] == env["UV_EXTRA_INDEX_URL"]


def test_nvidia_gets_cuda_index(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_NVIDIA, ""))
    env = install_env(get_model("sharp"))
    assert "cu12" in env["UV_EXTRA_INDEX_URL"]


def test_apple_uses_default_index(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_APPLE, "MPS"))
    env = install_env(get_model("sharp"))
    assert "UV_EXTRA_INDEX_URL" not in env
    assert "UV_INDEX_STRATEGY" not in env


def test_no_torch_index_for_binary_models(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(installer_mod, "detect", lambda: GPUInfo(VENDOR_AMD, ""))
    env = install_env(get_model("brush"))
    assert "UV_EXTRA_INDEX_URL" not in env
