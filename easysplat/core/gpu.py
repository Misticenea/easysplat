"""Best-effort GPU vendor detection (NVIDIA / AMD / Intel / Apple / CPU).

The result drives two things: which PyTorch wheel index model environments
are installed from, and which catalog models are shown as compatible.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

VENDOR_NVIDIA = "nvidia"
VENDOR_AMD = "amd"
VENDOR_INTEL = "intel"
VENDOR_APPLE = "apple"
VENDOR_CPU = "cpu"

# PyTorch wheel index per vendor (None = default PyPI wheels).
TORCH_INDEX = {
    VENDOR_NVIDIA: "https://download.pytorch.org/whl/cu124",
    VENDOR_AMD: "https://download.pytorch.org/whl/rocm6.4",
    VENDOR_INTEL: "https://download.pytorch.org/whl/xpu",
    VENDOR_APPLE: None,
    VENDOR_CPU: "https://download.pytorch.org/whl/cpu",
}

# uv's torch backend selector (UV_TORCH_BACKEND). Unlike an extra index,
# it redirects ONLY torch-family packages to the vendor index, so an AMD
# machine never downloads nvidia-* CUDA wheels and non-torch packages
# resolve from PyPI alone. "auto" probes the CUDA driver on NVIDIA.
TORCH_BACKEND = {
    VENDOR_NVIDIA: "auto",
    # newest ROCm index — older ones (6.2) stop at torch 2.5.x and can't
    # satisfy current model pins like SHARP's torch==2.8.0
    VENDOR_AMD: "rocm6.4",
    VENDOR_INTEL: "xpu",
    VENDOR_APPLE: None,
    VENDOR_CPU: "cpu",
}


@dataclass(frozen=True)
class GPUInfo:
    vendor: str
    name: str

    @property
    def label(self) -> str:
        pretty = {
            VENDOR_NVIDIA: "NVIDIA",
            VENDOR_AMD: "AMD",
            VENDOR_INTEL: "Intel",
            VENDOR_APPLE: "Apple Silicon",
            VENDOR_CPU: "CPU only",
        }[self.vendor]
        return f"{pretty}" + (f" ({self.name})" if self.name else "")


def _run(cmd: list[str]) -> str:
    from easysplat.core.proc import clean_base_env

    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False,
            env=clean_base_env(),
        )
        return out.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _gpu_names() -> str:
    """Lowercased blob describing the GPUs the OS can see."""
    system = platform.system()
    if system == "Linux":
        if shutil.which("lspci"):
            lines = [
                ln
                for ln in _run(["lspci"]).splitlines()
                if "VGA" in ln or "3D controller" in ln or "Display controller" in ln
            ]
            return " ".join(lines).lower()
        # lspci may be missing on minimal/immutable systems; sysfs vendor ids
        blob = []
        try:
            from pathlib import Path

            for vendor_file in Path("/sys/class/drm").glob("card?/device/vendor"):
                blob.append(vendor_file.read_text().strip())
        except OSError:
            pass
        return " ".join(blob).lower()
    if system == "Windows":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ]
        )
        return out.lower()
    if system == "Darwin":
        return _run(["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"]).lower()
    return ""


@lru_cache(maxsize=1)
def detect() -> GPUInfo:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return GPUInfo(VENDOR_APPLE, "MPS")

    if shutil.which("nvidia-smi"):
        name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip()
        return GPUInfo(VENDOR_NVIDIA, name.splitlines()[0] if name else "")

    names = _gpu_names()
    # PCI vendor ids show up when we fell back to sysfs
    if "nvidia" in names or "0x10de" in names:
        return GPUInfo(VENDOR_NVIDIA, "")
    if "amd" in names or "ati" in names or "radeon" in names or "0x1002" in names:
        return GPUInfo(VENDOR_AMD, "")
    if "intel" in names or "arc" in names or "0x8086" in names:
        return GPUInfo(VENDOR_INTEL, "")
    return GPUInfo(VENDOR_CPU, "")


def torch_index_for(vendor: str) -> str | None:
    import platform as _platform

    index = TORCH_INDEX.get(vendor)
    # ROCm wheels are Linux-only; AMD on Windows falls back to CPU wheels.
    if vendor == VENDOR_AMD and _platform.system() != "Linux":
        return TORCH_INDEX[VENDOR_CPU]
    return index


def torch_backend_for(vendor: str) -> str | None:
    import platform as _platform

    if vendor == VENDOR_AMD and _platform.system() != "Linux":
        return TORCH_BACKEND[VENDOR_CPU]
    return TORCH_BACKEND.get(vendor)
