import platform

import pytest

import easysplat.core.toolchain as tc
from easysplat.core.installer import platform_key


@pytest.mark.parametrize(
    "system,machine,expected_fragment",
    [
        ("Linux", "x86_64", "x86_64-unknown-linux-gnu"),
        ("Linux", "aarch64", "aarch64-unknown-linux-gnu"),
        ("Darwin", "arm64", "aarch64-apple-darwin"),
        ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
    ],
)
def test_uv_url_per_platform(monkeypatch, system, machine, expected_fragment):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert expected_fragment in tc._uv_url()


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        ("Linux", "x86_64", "linux-64"),
        ("Darwin", "arm64", "osx-arm64"),
        ("Windows", "AMD64", "win-64"),
    ],
)
def test_micromamba_url_per_platform(monkeypatch, system, machine, expected):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert expected in tc._micromamba_url()


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        ("Linux", "x86_64", "linux-x86_64"),
        ("Darwin", "arm64", "darwin-arm64"),
        ("Windows", "AMD64", "windows-x86_64"),
    ],
)
def test_platform_key(monkeypatch, system, machine, expected):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert platform_key() == expected


def test_find_tool_prefers_system_path(monkeypatch, tmp_path):
    fake = tmp_path / "sysbin" / "colmap"
    fake.parent.mkdir()
    fake.write_text("")
    monkeypatch.setattr(tc.shutil, "which", lambda name: str(fake))
    assert tc.find_tool("colmap") == fake


def test_find_tool_falls_back_to_managed_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(tc.shutil, "which", lambda name: None)
    managed = tmp_path / "tools" / "bin"
    managed.mkdir(parents=True)
    exe = managed / ("colmap.exe" if platform.system() == "Windows" else "colmap")
    exe.write_text("")
    assert tc.find_tool("colmap") == exe


def test_find_tool_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path))
    monkeypatch.setattr(tc.shutil, "which", lambda name: None)
    assert tc.find_tool("colmap") is None
