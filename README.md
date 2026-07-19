# EasySplat

**Train Gaussian splats from your terminal.** A fast, keyboard-driven text UI
(no graphical GUI — maximum performance over SSH, on servers, on anything) that
turns a folder of photos, a video, or even a **single image** into a 3D
Gaussian splat.

```
┌ EasySplat ──────────────────────────── v0.1.0 ┐
│  📦 Models  │  🚀 Train                        │
│ ───────────────────────────────────────────── │
│  Apple SHARP            ✔ installed           │
│  Brush                  ━━━━━━╸ 63% installing│
│  3D Gaussian Splatting  not installed         │
│ ───────────────────────────────────────────── │
│  GPU: NVIDIA (RTX 4070) • tools: git, ffmpeg  │
└───────────────────────────────────────────────┘
```

## What it does

- **Model download tab** — a catalog of splat trainers fetched from their
  upstream sites **on demand** (nothing is bundled with the app):
  - **Apple SHARP** — a splat from a *single image* in under a second
  - **Brush** — trains on **NVIDIA, AMD, Intel and Apple** GPUs (WebGPU)
  - **3D Gaussian Splatting (Inria)** — the original SIGGRAPH reference
  - **gsplat** and **Nerfstudio splatfacto** — the Nerfstudio stack
- **Training tab** — browse your home folder, pick a dataset folder, pick a
  model, hit *Start*. Works with **one image, many images, or a video**.
- **Automatic COLMAP** — if the model needs camera poses and your folder has
  none, EasySplat extracts video frames with ffmpeg and runs the full COLMAP
  pipeline (features → matching → mapping) for you.
- **Progress bars everywhere** — model downloads, COLMAP stages and training
  iterations all report live progress, with the raw tool output in a log pane.
- **Any GPU** — detects NVIDIA / AMD / Intel / Apple Silicon and installs the
  matching PyTorch build (CUDA / ROCm / XPU / MPS) into each model's private
  environment. Models that can't run on your GPU are flagged before you start.

## Install

### Prebuilt binary (recommended — no Python needed)

Grab the executable for your OS from the
[releases page](https://github.com/misticenea/easysplat/releases) and run it:

```sh
chmod +x easysplat-linux-x86_64   # Linux / macOS
./easysplat-linux-x86_64
```

The binary bundles Python and every library the app needs, so it works on
**immutable OSes** (Fedora Silverblue, SteamOS, …) and machines without
Python, compilers or admin rights. Everything EasySplat writes lives in
`~/.easysplat`.

### With pip / pipx

```sh
pipx install easysplat   # or: pip install easysplat
easysplat
```

### From source

```sh
git clone https://github.com/misticenea/easysplat
cd easysplat
pip install -e .
easysplat
```

## Usage

1. **Models tab** (`m`): pick a model, press **Install**. The repo, an
   isolated Python environment and any checkpoints are downloaded with a
   progress bar into `~/.easysplat/models/<id>/`.
2. **Train tab** (`t`): browse to your dataset folder in your home directory
   (or type the path and press Enter). EasySplat tells you what it found —
   one image, N images, or a video — and which installed models can train it.
3. Press **Start training**. If COLMAP data is needed and missing it is built
   automatically; then training runs with a live progress bar. Results land in
   `<your-folder>/output/<model-id>/`.

`q` quits.

## How the self-contained toolchain works

EasySplat never assumes your system has anything installed:

| Need | How it's provided |
|---|---|
| Python for the app | bundled inside the released executable (PyInstaller) |
| Python for models | `uv` downloads standalone CPython builds per model |
| git / ffmpeg / COLMAP | installed from conda-forge via `micromamba` into `~/.easysplat/tools` |
| GPU wheels | PyTorch index chosen from the detected GPU vendor |

System copies of the tools are used when they exist, so nothing is downloaded
twice. Delete `~/.easysplat` to remove everything.

## Adding models to the catalog

The catalog is pure data: [`easysplat/models/catalog.json`](easysplat/models/catalog.json).
An entry declares where the model lives (git URL or prebuilt binaries), what
inputs it accepts, which GPU vendors it supports, its install commands and its
train command template. Add an entry, and the app can download and drive it —
no code changes needed.

## Development

```sh
pip install -e ".[dev]"
pytest          # unit + headless TUI tests
ruff check .
```

Release binaries for Linux/macOS/Windows are built by
[`.github/workflows/release.yml`](.github/workflows/release.yml) on version tags.
