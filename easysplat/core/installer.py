"""Install catalog models on demand.

Each model gets a fully isolated home::

    ~/.easysplat/models/<id>/
      repo/       git clone of the upstream project
      venv/       private virtualenv (uv-managed standalone CPython)
      weights/    downloaded checkpoints, if the model needs any
      installed.json

``uv`` downloads its own CPython build, so installs work on machines with no
system Python at all. The PyTorch wheel index is chosen from the detected GPU
vendor (CUDA / ROCm / XPU / default).
"""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

from easysplat.core import paths, toolchain
from easysplat.core.catalog import KIND_BINARY, ModelSpec
from easysplat.core.downloader import download_file, extract_archive, find_file
from easysplat.core.gpu import detect, torch_backend_for, torch_index_for
from easysplat.core.proc import LineCallback, run_streaming, subprocess_env
from easysplat.core.toolchain import StatusCallback


def platform_key() -> str:
    """Key used to pick a prebuilt binary from a catalog ``binaries`` map."""
    system = {"Linux": "linux", "Darwin": "darwin", "Windows": "windows"}.get(
        platform.system(), platform.system().lower()
    )
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    return f"{system}-{arch}"


def repo_dir(spec: ModelSpec) -> Path:
    return paths.model_dir(spec.id) / "repo"


def venv_dir(spec: ModelSpec) -> Path:
    return paths.model_dir(spec.id) / "venv"


def weights_dir(spec: ModelSpec) -> Path:
    return paths.model_dir(spec.id) / "weights"


def marker_path(spec: ModelSpec) -> Path:
    return paths.model_dir(spec.id) / "installed.json"


def venv_python(spec: ModelSpec) -> Path:
    if platform.system() == "Windows":
        return venv_dir(spec) / "Scripts" / "python.exe"
    return venv_dir(spec) / "bin" / "python"


def bin_dir(spec: ModelSpec) -> Path:
    return paths.model_dir(spec.id) / "bin"


def binary_path(spec: ModelSpec) -> Path | None:
    """The installed executable of a KIND_BINARY model, if present."""
    directory = bin_dir(spec)
    if not directory.is_dir():
        return None
    for name in spec.binary_names:
        for candidate in (directory / name, directory / f"{name}.exe"):
            if candidate.is_file():
                return candidate
    files = [p for p in directory.iterdir() if p.is_file()]
    return files[0] if files else None


def is_installed(spec: ModelSpec) -> bool:
    if not marker_path(spec).exists():
        return False
    if spec.kind == KIND_BINARY:
        return binary_path(spec) is not None
    return venv_python(spec).exists()


def remove_model(spec: ModelSpec) -> None:
    shutil.rmtree(paths.model_dir(spec.id), ignore_errors=True)


def resolve_script(python: Path, name: str) -> Path:
    """Locate a console script (e.g. ``sharp``, ``ns-train``) in a venv."""
    scripts = python.parent
    for candidate in (scripts / name, scripts / f"{name}.exe"):
        if candidate.exists():
            return candidate
    return scripts / name  # let the subprocess fail with a clear path


def expand_command(
    command: tuple[str, ...], mapping: dict[str, str], uv: Path, python: Path
) -> list[str]:
    """Expand catalog command templates.

    - a literal ``{pip}`` token becomes ``uv pip install --python <venv-python>``
    - ``{script:NAME}`` resolves a venv console script cross-platform
    - every other token is ``str.format``-expanded against ``mapping``
    """
    out: list[str] = []
    for token in command:
        if token == "{pip}":
            out += [str(uv), "pip", "install", "--python", str(python)]
        elif token.startswith("{script:") and token.endswith("}"):
            out.append(str(resolve_script(python, token[len("{script:"):-1])))
        else:
            out.append(token.format(**mapping))
    return out


async def _install_prebuilt(
    spec: ModelSpec, status_cb: StatusCallback, on_line: LineCallback
) -> None:
    key = platform_key()
    url = spec.binaries.get(key)
    if url is None:
        raise RuntimeError(
            f"{spec.name} has no prebuilt binary for this platform ({key}). "
            f"Available: {', '.join(sorted(spec.binaries))}"
        )
    on_line(f"Downloading prebuilt binary for {key}: {url}")
    archive = paths.cache_dir() / f"{spec.id}-{Path(url).name}"

    def on_bytes(done: int, total: int | None) -> None:
        status_cb(
            f"Downloading {spec.name}…", (done / total * 0.9) if total else None
        )

    await download_file(url, archive, on_bytes)
    status_cb("Extracting…", 0.9)
    extract_dir = paths.cache_dir() / f"{spec.id}-extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_archive(archive, extract_dir)

    found: Path | None = None
    for name in spec.binary_names:
        found = find_file(extract_dir, name) or find_file(extract_dir, f"{name}.exe")
        if found:
            break
    if found is None:
        # single-binary archives: take the largest file
        files = sorted(extract_dir.rglob("*"), key=lambda p: p.stat().st_size if p.is_file() else -1)
        found = files[-1] if files and files[-1].is_file() else None
    if found is None:
        raise RuntimeError("no executable found in the downloaded archive")

    dest = bin_dir(spec) / found.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, dest)
    dest.chmod(0o755)
    shutil.rmtree(extract_dir, ignore_errors=True)
    archive.unlink(missing_ok=True)
    on_line(f"Installed binary: {dest}")


def install_env(spec: ModelSpec) -> dict[str, str]:
    """Environment for a model's install commands.

    UV_TORCH_BACKEND redirects only torch-family packages to the GPU
    vendor's wheel index (ROCm/CUDA/XPU/CPU); every other package resolves
    from PyPI as usual. This keeps AMD/Intel machines from downloading the
    default CUDA torch build and its nvidia-* dependency wheels, and avoids
    cross-index version clashes for common packages (certifi, numpy, ...).
    """
    env = subprocess_env(extra_paths=[paths.bin_dir(), *paths.tools_bin_dirs()])
    if spec.needs_torch_index:
        vendor = detect().vendor
        backend = torch_backend_for(vendor)
        if backend:
            env["UV_TORCH_BACKEND"] = backend
        index = torch_index_for(vendor)
        if index:
            # pip compatibility for scripts that shell out to pip themselves
            env["PIP_EXTRA_INDEX_URL"] = index
    return env


async def install_model(
    spec: ModelSpec,
    status_cb: StatusCallback,
    on_line: LineCallback,
) -> None:
    """Download + install a model. Progress goes to ``status_cb`` (0..1)."""
    if spec.kind == KIND_BINARY:
        paths.model_dir(spec.id).mkdir(parents=True, exist_ok=True)
        await _install_prebuilt(spec, status_cb, on_line)
        _write_marker(spec)
        status_cb("Installed ✔", 1.0)
        return

    total_steps = 3 + len(spec.install_commands) + len(spec.weights)
    step = 0

    def report(label: str, sub: float | None = None) -> None:
        base = step / total_steps
        frac = base if sub is None else base + sub / total_steps
        status_cb(label, min(frac, 1.0))

    paths.model_dir(spec.id).mkdir(parents=True, exist_ok=True)

    report("Preparing toolchain (git, uv)…")
    git = await toolchain.ensure_git(status_cb, on_line)
    uv = await toolchain.ensure_uv(status_cb)
    step += 1

    report(f"Cloning {spec.source}…")
    repo = repo_dir(spec)
    if not (repo / ".git").exists():
        clone_cmd = [str(git), "clone", "--depth", "1"]
        if spec.git_ref:
            clone_cmd += ["--branch", spec.git_ref]
        if spec.git_submodules:
            clone_cmd += ["--recursive", "--shallow-submodules"]
        clone_cmd += [spec.source, str(repo)]
        await run_streaming(clone_cmd, on_line=on_line, env=toolchain.tool_env())
    else:
        on_line("Repository already cloned — skipping.")
    step += 1

    report(f"Creating Python {spec.python_version} environment…")
    venv = venv_dir(spec)
    if not venv_python(spec).exists():
        await run_streaming(
            [str(uv), "venv", "--python", spec.python_version, str(venv)],
            on_line=on_line,
        )
    step += 1

    python = venv_python(spec)
    mapping = {
        "python": str(python),
        "uv": str(uv),
        "repo": str(repo),
        "model_dir": str(paths.model_dir(spec.id)),
    }
    env = install_env(spec)

    for i, command in enumerate(spec.install_commands, start=1):
        report(f"Installing dependencies ({i}/{len(spec.install_commands)})…")
        await run_streaming(
            expand_command(command, mapping, uv, python),
            cwd=repo,
            env=env,
            on_line=on_line,
        )
        step += 1

    for weight in spec.weights:
        dest = paths.model_dir(spec.id) / weight.dest
        if dest.exists():
            step += 1
            continue

        def on_bytes(done: int, total: int | None) -> None:
            mb = done / (1 << 20)
            if total:
                report(f"Downloading weights {mb:.0f}/{total / (1 << 20):.0f} MB", done / total)
            else:
                report(f"Downloading weights {mb:.0f} MB")

        report(f"Downloading weights: {weight.dest}")
        await download_file(weight.url, dest, on_bytes)
        step += 1

    _write_marker(spec)
    status_cb("Installed ✔", 1.0)


def _write_marker(spec: ModelSpec) -> None:
    marker_path(spec).write_text(
        json.dumps(
            {
                "id": spec.id,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "gpu_vendor": detect().vendor,
                "source": spec.source,
            },
            indent=2,
        )
    )
