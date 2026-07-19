"""End-to-end TUI smoke tests using Textual's Pilot (headless terminal)."""

import pytest
from textual.widgets import ProgressBar, TabbedContent

from easysplat.app import EasySplatApp
from easysplat.ui.models_tab import ModelCard
from easysplat.ui.train_tab import TrainTab


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYSPLAT_HOME", str(tmp_path / ".easysplat"))


async def test_app_boots_with_both_tabs():
    app = EasySplatApp()
    async with app.run_test(size=(120, 40)):
        tabs = app.query_one(TabbedContent)
        assert tabs.active == "tab-models"
        assert app.query(ModelCard)
        assert app.query_one(TrainTab)


async def test_tab_switch_keybindings():
    app = EasySplatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("t")
        assert app.query_one(TabbedContent).active == "tab-train"
        await pilot.press("m")
        assert app.query_one(TabbedContent).active == "tab-models"


async def test_model_cards_show_catalog():
    app = EasySplatApp()
    async with app.run_test(size=(120, 40)):
        ids = {card.spec.id for card in app.query(ModelCard)}
        assert "sharp" in ids and "brush" in ids
        for card in app.query(ModelCard):
            assert not card.query_one(ProgressBar).display  # hidden until installing


async def test_train_tab_scan_updates_summary(tmp_path):
    dataset = tmp_path / "scene"
    dataset.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (dataset / name).write_bytes(b"x")
    app = EasySplatApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("t")
        train = app.query_one(TrainTab)
        train.set_folder(dataset)
        await pilot.pause()
        assert train.scan is not None
        assert train.scan.kind == "multi_image"
        # nothing is installed in the isolated home, so training can't start
        assert app.query_one("#start").disabled
