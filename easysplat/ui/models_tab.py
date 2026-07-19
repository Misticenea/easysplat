"""Models tab: browse the catalog, install/remove models with live progress."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Label, ProgressBar, Static

from easysplat.core import installer
from easysplat.core.catalog import ModelSpec, load_catalog
from easysplat.core.gpu import detect


class ModelCard(Vertical):
    """One catalog entry: description, status, progress bar, install/remove."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(classes="model-card")
        self.spec = spec

    def compose(self) -> ComposeResult:
        spec = self.spec
        gpu = detect()
        compatible = spec.supports_backend(gpu.vendor)
        inputs = ", ".join(t.replace("_", " ") for t in spec.input_types)
        backends = ", ".join(spec.backends)
        yield Label(f"[b]{spec.name}[/b]", classes="model-title")
        yield Static(spec.description, classes="model-desc")
        meta = (
            f"inputs: {inputs}  •  GPUs: {backends}  •  "
            f"COLMAP: {'auto' if spec.requires_colmap else 'not needed'}"
            + (f"  •  size: {spec.approx_size}" if spec.approx_size else "")
            + f"\nsource: {spec.source}"
        )
        yield Static(meta, classes="model-meta")
        if not compatible:
            yield Static(
                f"⚠ not compatible with this machine's GPU ({gpu.label})",
                classes="model-warn",
            )
        with Horizontal(classes="model-actions"):
            yield Button("Install", id="install", variant="success")
            yield Button("Remove", id="remove", variant="error")
            yield Label("", id="status", classes="model-status")
        yield ProgressBar(total=100, id="progress", show_eta=False)
        yield Label("", id="logline", classes="model-logline")

    def on_mount(self) -> None:
        self.query_one("#progress", ProgressBar).display = False
        self.refresh_status()

    def refresh_status(self) -> None:
        installed = installer.is_installed(self.spec)
        status = self.query_one("#status", Label)
        status.update("[green]✔ installed[/green]" if installed else "[dim]not installed[/dim]")
        self.query_one("#install", Button).disabled = installed
        self.query_one("#remove", Button).disabled = not installed

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "install":
            self.install()
        elif event.button.id == "remove":
            installer.remove_model(self.spec)
            self.app.notify(f"{self.spec.name} removed.")
            self.refresh_status()

    @work(exclusive=True, group="install")
    async def install(self) -> None:
        bar = self.query_one("#progress", ProgressBar)
        logline = self.query_one("#logline", Label)
        status = self.query_one("#status", Label)
        self.query_one("#install", Button).disabled = True
        self.query_one("#remove", Button).disabled = True
        bar.display = True
        bar.update(total=100, progress=0)

        def status_cb(message: str, fraction: float | None) -> None:
            status.update(f"[yellow]{message}[/yellow]")
            if fraction is not None:
                bar.update(total=100, progress=fraction * 100)

        def on_line(line: str) -> None:
            if line.strip():
                logline.update(line.strip()[-120:])

        try:
            await installer.install_model(self.spec, status_cb, on_line)
            self.app.notify(f"{self.spec.name} installed.", severity="information")
        except Exception as exc:  # surface the real error in the UI
            self.app.notify(f"Install failed: {exc}", severity="error", timeout=12)
            status.update("[red]install failed[/red]")
            logline.update(str(exc)[-200:])
        finally:
            bar.display = False
            self.refresh_status()


class ModelsTab(VerticalScroll):
    """Scrollable list of everything in the catalog."""

    def compose(self) -> ComposeResult:
        yield Static(
            "Models are downloaded from their upstream sites on demand — "
            "nothing ships with EasySplat.",
            classes="tab-hint",
        )
        for spec in load_catalog():
            yield ModelCard(spec)
