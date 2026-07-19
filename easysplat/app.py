"""EasySplat — a terminal UI (no graphical GUI) for training Gaussian splats."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from easysplat import __version__
from easysplat.core import paths
from easysplat.core.gpu import detect
from easysplat.core.toolchain import toolchain_summary
from easysplat.ui.models_tab import ModelsTab
from easysplat.ui.train_tab import TrainTab


class EasySplatApp(App):
    TITLE = "EasySplat"
    SUB_TITLE = f"v{__version__} — Gaussian splat training in your terminal"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("m", "show_tab('tab-models')", "Models"),
        ("t", "show_tab('tab-train')", "Train"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="tab-models"):
            with TabPane("📦 Models", id="tab-models"):
                yield ModelsTab()
            with TabPane("🚀 Train", id="tab-train"):
                yield TrainTab()
        yield Static(self._status_line(), id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        paths.ensure_dirs()

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def _status_line(self) -> str:
        gpu = detect()
        tools = toolchain_summary()
        found = [name for name, path in tools.items() if path]
        missing = [name for name, path in tools.items() if not path]
        parts = [f"GPU: [b]{gpu.label}[/b]"]
        if found:
            parts.append(f"tools: {', '.join(found)}")
        if missing:
            parts.append(f"[dim]auto-installed when needed: {', '.join(missing)}[/dim]")
        return "  •  ".join(parts)


def main() -> None:
    EasySplatApp().run()


if __name__ == "__main__":
    main()
