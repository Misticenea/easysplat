import os
import sys

from easysplat.core.proc import clean_base_env, subprocess_env


def frozen(monkeypatch, meipass="/tmp/_MEIfake"):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", meipass, raising=False)


def test_not_frozen_env_untouched(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/mylibs")
    assert clean_base_env()["LD_LIBRARY_PATH"] == "/opt/mylibs"


def test_frozen_restores_original_lib_path(monkeypatch):
    frozen(monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIfake")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/opt/mylibs")
    env = clean_base_env()
    assert env["LD_LIBRARY_PATH"] == "/opt/mylibs"
    assert "LD_LIBRARY_PATH_ORIG" not in env


def test_frozen_drops_lib_path_when_no_original(monkeypatch):
    frozen(monkeypatch)
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIfake")
    assert "LD_LIBRARY_PATH" not in clean_base_env()


def test_frozen_strips_only_bundle_dir(monkeypatch):
    frozen(monkeypatch)
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", os.pathsep.join(["/tmp/_MEIfake", "/usr/local/lib"])
    )
    assert clean_base_env()["LD_LIBRARY_PATH"] == "/usr/local/lib"


def test_frozen_empty_orig_unsets_var(monkeypatch):
    frozen(monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIfake")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "")
    assert "LD_LIBRARY_PATH" not in clean_base_env()


def test_subprocess_env_is_clean_when_frozen(monkeypatch, tmp_path):
    frozen(monkeypatch)
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIfake")
    env = subprocess_env(extra_paths=[tmp_path])
    assert "LD_LIBRARY_PATH" not in env
    assert env["PATH"].startswith(str(tmp_path))
