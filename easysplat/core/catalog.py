"""Model catalog.

The catalog is pure data (``models/catalog.json``): it tells EasySplat where a
model lives on the internet and how to install/run it. Nothing is bundled with
the app — models are fetched on demand from their upstream sites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources

INPUT_SINGLE_IMAGE = "single_image"
INPUT_MULTI_IMAGE = "multi_image"
INPUT_VIDEO = "video"

KIND_PYTHON = "python"  # git clone + private venv + pip install
KIND_BINARY = "binary"  # prebuilt executable downloaded per platform


@dataclass(frozen=True)
class WeightFile:
    url: str
    dest: str  # path relative to the model dir


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    source: str  # git repository URL (upstream project page)
    input_types: tuple[str, ...]
    requires_colmap: bool
    backends: tuple[str, ...]  # subset of {nvidia, amd, intel, apple, cpu}
    install_commands: tuple[tuple[str, ...], ...]
    train_command: tuple[str, ...]
    progress_regex: str  # named groups: step, total
    kind: str = KIND_PYTHON
    git_ref: str | None = None
    git_submodules: bool = False
    python_version: str = "3.10"
    needs_torch_index: bool = False
    weights: tuple[WeightFile, ...] = field(default_factory=tuple)
    # KIND_BINARY only: platform key (e.g. "linux-x86_64") -> archive URL
    binaries: dict[str, str] = field(default_factory=dict)
    binary_names: tuple[str, ...] = ()  # candidate executable names in the archive
    homepage: str | None = None
    approx_size: str | None = None

    def supports_input(self, kind: str) -> bool:
        return kind in self.input_types

    def supports_backend(self, vendor: str) -> bool:
        return vendor in self.backends


def _parse_spec(raw: dict) -> ModelSpec:
    return ModelSpec(
        id=raw["id"],
        name=raw["name"],
        description=raw["description"],
        source=raw["source"],
        input_types=tuple(raw["input_types"]),
        requires_colmap=raw["requires_colmap"],
        backends=tuple(raw["backends"]),
        install_commands=tuple(tuple(cmd) for cmd in raw.get("install_commands", [])),
        train_command=tuple(raw["train_command"]),
        progress_regex=raw["progress_regex"],
        kind=raw.get("kind", KIND_PYTHON),
        git_ref=raw.get("git_ref"),
        git_submodules=raw.get("git_submodules", False),
        python_version=raw.get("python_version", "3.10"),
        needs_torch_index=raw.get("needs_torch_index", False),
        weights=tuple(WeightFile(w["url"], w["dest"]) for w in raw.get("weights", [])),
        binaries=dict(raw.get("binaries", {})),
        binary_names=tuple(raw.get("binary_names", [])),
        homepage=raw.get("homepage"),
        approx_size=raw.get("approx_size"),
    )


def load_catalog() -> list[ModelSpec]:
    text = (
        resources.files("easysplat").joinpath("models/catalog.json").read_text("utf-8")
    )
    data = json.loads(text)
    return [_parse_spec(raw) for raw in data["models"]]


def get_model(model_id: str) -> ModelSpec:
    for spec in load_catalog():
        if spec.id == model_id:
            return spec
    raise KeyError(model_id)
