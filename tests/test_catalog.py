import re

import pytest

from easysplat.core.catalog import (
    INPUT_MULTI_IMAGE,
    INPUT_SINGLE_IMAGE,
    INPUT_VIDEO,
    KIND_BINARY,
    KIND_PYTHON,
    get_model,
    load_catalog,
)

VALID_INPUTS = {INPUT_SINGLE_IMAGE, INPUT_MULTI_IMAGE, INPUT_VIDEO}
VALID_BACKENDS = {"nvidia", "amd", "intel", "apple", "cpu"}


def test_catalog_loads_expected_models():
    ids = {spec.id for spec in load_catalog()}
    assert {"sharp", "brush", "gaussian-splatting-inria", "gsplat", "nerfstudio-splatfacto"} <= ids


def test_every_spec_is_well_formed():
    for spec in load_catalog():
        assert spec.input_types and set(spec.input_types) <= VALID_INPUTS, spec.id
        assert spec.backends and set(spec.backends) <= VALID_BACKENDS, spec.id
        assert spec.source.startswith("https://"), spec.id
        assert spec.train_command, spec.id
        pattern = re.compile(spec.progress_regex)
        assert {"step", "total"} <= set(pattern.groupindex), spec.id
        if spec.kind == KIND_BINARY:
            assert spec.binaries, spec.id
            for url in spec.binaries.values():
                assert url.startswith("https://"), spec.id
        else:
            assert spec.kind == KIND_PYTHON, spec.id
            assert spec.install_commands, spec.id


def test_single_image_model_present_without_colmap():
    sharp = get_model("sharp")
    assert sharp.supports_input(INPUT_SINGLE_IMAGE)
    assert not sharp.requires_colmap


def test_amd_and_intel_have_a_trainer():
    for vendor in ("amd", "intel"):
        assert any(
            spec.supports_backend(vendor) and spec.supports_input(INPUT_MULTI_IMAGE)
            for spec in load_catalog()
        ), f"no multi-image trainer for {vendor}"


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("does-not-exist")


def test_progress_regex_matches_typical_lines():
    lines = {
        "gaussian-splatting-inria": "Training progress:  23%|##3       | 7000/30000",
        "gsplat": "step 500/30000 loss=0.2",
        "brush": "step 100 / 30000",
    }
    for model_id, line in lines.items():
        spec = get_model(model_id)
        match = re.search(spec.progress_regex, line)
        assert match, model_id
        assert int(match.group("total")) == 30000
