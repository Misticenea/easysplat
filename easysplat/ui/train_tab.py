"""Training tab: pick a folder in your home directory, pick a model, train.

The COLMAP reconstruction (when the model needs one) and the training run
both report into the same progress bar + log, stage by stage.
"""

from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DirectoryTree, Input, Label, ProgressBar, RichLog, Select, Static

from easysplat.core import colmap, trainer
from easysplat.core.catalog import ModelSpec, load_catalog
from easysplat.core.gpu import detect
from easysplat.core.inputs import ScanResult, scan_folder
from easysplat.core.installer import is_installed


class DatasetTree(DirectoryTree):
    """Home-rooted tree with dotfiles hidden."""

    def filter_paths(self, paths):  # type: ignore[override]
        return [p for p in paths if not p.name.startswith(".")]


class TrainTab(Horizontal):
    def __init__(self) -> None:
        super().__init__()
        self.scan: ScanResult | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="train-left"):
            yield Static("Pick your dataset folder (photos, a video, or one image):", classes="tab-hint")
            yield Input(value=str(Path.home()), placeholder=str(Path.home()), id="path-input")
            yield DatasetTree(Path.home(), id="tree")
        with Vertical(id="train-right"):
            yield Label("No folder selected.", id="scan-summary")
            yield Select([], prompt="Select a model…", id="model-select")
            with Horizontal(id="train-actions"):
                yield Button("Start training", id="start", variant="success", disabled=True)
            yield Label("", id="stage-label")
            yield ProgressBar(total=100, id="train-progress", show_eta=False)
            yield RichLog(id="train-log", highlight=False, markup=False, wrap=True, max_lines=2000)

    def on_mount(self) -> None:
        self.query_one("#train-progress", ProgressBar).display = False

    # --- folder selection -------------------------------------------------

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        event.stop()
        self.query_one("#path-input", Input).value = str(event.path)
        self.set_folder(Path(event.path))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.set_folder(Path(event.value).expanduser())

    def set_folder(self, folder: Path) -> None:
        self.scan = scan_folder(folder)
        summary = self.query_one("#scan-summary", Label)
        if self.scan.kind is None:
            summary.update(f"[red]{folder}[/red] — no images or videos found.")
        else:
            colmap_note = " (COLMAP data present)" if self.scan.has_colmap else ""
            summary.update(f"[b]{folder}[/b]\nFound {self.scan.summary}{colmap_note}")
        self.refresh_model_options()

    # --- model choice -----------------------------------------------------

    def compatible_models(self) -> list[ModelSpec]:
        if self.scan is None or self.scan.kind is None:
            return []
        vendor = detect().vendor
        return [
            spec
            for spec in load_catalog()
            if is_installed(spec)
            and spec.supports_input(self.scan.kind)
            and spec.supports_backend(vendor)
        ]

    def refresh_model_options(self) -> None:
        select = self.query_one("#model-select", Select)
        models = self.compatible_models()
        select.set_options((spec.name, spec.id) for spec in models)
        if models:
            # always keep a concrete selection so Start never sees the
            # blank sentinel (whose identity varies across Textual versions)
            select.value = models[0].id
        start = self.query_one("#start", Button)
        start.disabled = not models
        if self.scan is not None and self.scan.kind is not None and not models:
            self.query_one("#train-log", RichLog).write(
                "No installed model supports this input on your GPU — "
                "install one in the Models tab."
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        event.stop()
        # model ids are strings; any non-str value is the blank sentinel
        self.query_one("#start", Button).disabled = not isinstance(event.value, str)

    # --- training ---------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "start":
            select = self.query_one("#model-select", Select)
            if not isinstance(select.value, str) or self.scan is None:
                return
            from easysplat.core.catalog import get_model

            self.run_training(get_model(select.value))

    @work(exclusive=True, group="train")
    async def run_training(self, spec: ModelSpec) -> None:
        assert self.scan is not None
        log = self.query_one("#train-log", RichLog)
        bar = self.query_one("#train-progress", ProgressBar)
        stage = self.query_one("#stage-label", Label)
        start = self.query_one("#start", Button)
        start.disabled = True
        bar.display = True
        bar.update(total=100, progress=0)

        def on_line(line: str) -> None:
            log.write(line)

        def status_cb(message: str, fraction: float | None) -> None:
            stage.update(f"[yellow]{message}[/yellow]")

        def stage_cb(label: str, index: int, count: int) -> None:
            stage.update(f"[yellow]Step {index + 1}/{count}: {label}[/yellow]")
            bar.update(total=100, progress=index / count * 100)

        def train_progress(step: int, total: int) -> None:
            bar.update(total=100, progress=step / total * 100)
            stage.update(f"[yellow]Training… {step}/{total}[/yellow]")

        try:
            if spec.requires_colmap:
                stage.update("[yellow]Preparing COLMAP data…[/yellow]")
                await colmap.prepare_dataset(self.scan, stage_cb, on_line, status_cb)
                # dataset may now have images/ + sparse/: rescan
                self.scan = scan_folder(self.scan.folder)
            bar.update(total=100, progress=0)
            stage.update("[yellow]Training…[/yellow]")
            request = trainer.TrainRequest(spec=spec, scan=self.scan)
            output = await trainer.train(request, train_progress, on_line, status_cb)
            bar.update(total=100, progress=100)
            stage.update(f"[green]Done — output in {output}[/green]")
            self.app.notify(f"Training finished: {output}", timeout=12)
        except Exception as exc:
            stage.update(f"[red]Failed: {exc}[/red]")
            log.write(f"ERROR: {exc}")
            self.app.notify(f"Training failed: {exc}", severity="error", timeout=12)
        finally:
            start.disabled = False
