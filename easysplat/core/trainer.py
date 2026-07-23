"""Run a model's training command and turn its stdout into progress updates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from easysplat.core import paths, toolchain
from easysplat.core.catalog import KIND_BINARY, ModelSpec
from easysplat.core.inputs import ScanResult
from easysplat.core.installer import (
    binary_path,
    expand_command,
    repo_dir,
    venv_dir,
    venv_python,
)
from easysplat.core.proc import LineCallback, run_streaming, subprocess_env

# progress_cb(step, total) parsed from the trainer's own output
TrainProgress = Callable[[int, int], None]


@dataclass
class TrainRequest:
    spec: ModelSpec
    scan: ScanResult

    @property
    def output_dir(self) -> Path:
        return self.scan.folder / "output" / self.spec.id


def build_train_command(request: TrainRequest, uv: Path) -> list[str]:
    spec = request.spec
    scan = request.scan
    python = venv_python(spec)
    input_image = str(scan.images[0]) if scan.images else ""
    binary = binary_path(spec)
    mapping = {
        "python": str(python),
        "bin": str(binary) if binary else "",
        "repo": str(repo_dir(spec)),
        "model_dir": str(paths.model_dir(spec.id)),
        "dataset": str(scan.folder),
        "images": str(scan.folder / "images"),
        "sparse": str(scan.folder / "sparse"),
        "output": str(request.output_dir),
        "input_image": input_image,
    }
    return expand_command(spec.train_command, mapping, uv, python)


async def train(
    request: TrainRequest,
    progress_cb: TrainProgress,
    on_line: LineCallback,
    status_cb: toolchain.StatusCallback,
) -> Path:
    """Run training; returns the output directory on success."""
    spec = request.spec
    if spec.kind == KIND_BINARY:
        uv = Path("uv")  # not used by binary models' commands
    else:
        uv = await toolchain.ensure_uv(status_cb)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(spec.progress_regex)

    def handle_line(line: str) -> None:
        on_line(line)
        match = pattern.search(line)
        if match:
            try:
                step = int(match.group("step"))
                total = int(match.group("total"))
            except (IndexError, ValueError):
                return
            if total > 0:
                progress_cb(step, total)

    env = subprocess_env(
        extra_paths=[venv_dir(spec) / "bin", venv_dir(spec) / "Scripts",
                     paths.bin_dir(), *paths.tools_bin_dirs()],
        VIRTUAL_ENV=str(venv_dir(spec)),
    )
    # Some training scripts buffer stdout when not attached to a TTY;
    # force unbuffered output so the progress bar moves live.
    env["PYTHONUNBUFFERED"] = "1"
    # Rust binaries (e.g. Brush) are silent by default — without this the
    # log stays empty even while training runs. Show info-level progress.
    if spec.kind == KIND_BINARY:
        env.setdefault("RUST_LOG", "info")

    command = build_train_command(request, uv)
    cwd = repo_dir(spec) if repo_dir(spec).is_dir() else request.scan.folder
    on_line(f"$ {' '.join(command)}")
    await run_streaming(command, cwd=cwd, env=env, on_line=handle_line)
    return request.output_dir
