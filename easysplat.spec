# PyInstaller spec: one-file, self-contained EasySplat executable.
# Bundles the Python interpreter, Textual, httpx and the model catalog so the
# app runs on systems with no Python installed (immutable OSes included).

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("easysplat")  # catalog.json + app.tcss
datas += collect_data_files("textual")

# textual lazy-loads widgets via __getattr__ (e.g. textual.widgets._tab_pane),
# which PyInstaller's static analysis cannot see — collect everything.
hiddenimports = collect_submodules("textual")

a = Analysis(
    ["easysplat/__main__.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="easysplat",
    console=True,
    upx=False,
    strip=False,
)
