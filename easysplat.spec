# PyInstaller spec: one-file, self-contained EasySplat executable.
# Bundles the Python interpreter, Textual, httpx and the model catalog so the
# app runs on systems with no Python installed (immutable OSes included).

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("easysplat")  # catalog.json + app.tcss
datas += collect_data_files("textual")

a = Analysis(
    ["easysplat/__main__.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
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
