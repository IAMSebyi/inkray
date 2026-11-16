# Inkray Backend

Python + FastAPI inference service for Inkray, providing monocular geometry estimation via **MoGe-3** (`Ruicheng/moge-3-vitl`), intrinsic lighting decomposition via **Marigold IID Lighting v1.1** (`prs-eth/marigold-iid-lighting-v1-1`) through Hugging Face Diffusers, content-addressed disk caching, and binary array transport to the frontend.

## Implemented Functionality

- **MoGe-3 Inference:** Loads `moge.model.v3.MoGeModel` on CUDA when available (or CPU fallback), decodes JPEG/PNG uploads with EXIF orientation applied, and bounds input size while preserving aspect ratio.
- **Marigold IID Lighting Inference:** Lazily loads `diffusers.MarigoldIntrinsicsPipeline` (`prs-eth/marigold-iid-lighting-v1-1`) on first appearance request, predicting `AlbedoMap`, `ShadingMap`, and `ResidualMap` (`(H, W, 3)` linear float32) mapped via `pipeline.target_properties["target_names"]`.
- **Sequential GPU Memory Management:** Acquires a shared inference lock and moves inactive models to CPU (`prepare_model_for_inference`) before running the active model so MoGe-3 and Marigold fit comfortably within an 8 GB VRAM budget.
- **Typed Geometry & Appearance Representations:**
  - `GeometryResult`: `DepthMap`, `NormalMap`, `Points3D`, `CameraIntrinsics`, `ValidMask` using OpenCV camera conventions alongside Three.js conversion helpers.
  - `AppearanceResult` + `app.appearance.color`: `AlbedoMap`, `ShadingMap`, `ResidualMap`, IEC 61966-2-1 sRGB/linear conversions, deterministic per-surface shading scale calibration, and linear-RGB compositing utilities.
- **Content-Addressed Disk Caches:** Stores geometry records under `backend/.cache/geometry` and appearance records under `backend/.cache/appearance` cleanly separate from Hugging Face weight caches (`~/.cache/huggingface`), keyed by SHA-256 of image bytes, model ID, parameters, and format version.
- **API Endpoints:**
  - `POST /api/geometry/infer`, `GET /api/geometry/{geometry_id}`, `GET /api/geometry/{geometry_id}/maps/{kind}` (`depth`, `points`, `normals`, `mask`), `POST /api/geometry/{geometry_id}/surface/select`
  - `POST /api/appearance/infer`, `GET /api/appearance/{appearance_id}`, `GET /api/appearance/{appearance_id}/maps/{kind}` (`albedo`, `shading`, `residual`)

## Setup

Choose either **Anaconda / Miniconda** (recommended) or a standard Python `venv`.

### Option A: Anaconda / Miniconda

1. Create and activate a dedicated Conda environment with Python 3.10:
   ```bash
   conda create -n inkray python=3.10 -y
   conda activate inkray
   ```
2. Install PyTorch (`CUDA` build recommended, or CPU build if no NVIDIA GPU is present):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   *(For CPU-only systems, use `pip install torch torchvision`.)*
3. Install the official MoGe package and Inkray backend dependencies:
   ```bash
   cd backend
   pip install "git+https://github.com/microsoft/MoGe.git"
   pip install -e ".[dev]"
   ```
4. Verify installation and start the API server from the activated `inkray` environment:
   ```bash
   pytest -v
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

### Option B: Python `venv`

Create a virtual environment inside `backend/.venv`, install PyTorch for your CUDA version (or CPU), and install MoGe + backend dependencies (including `diffusers`, `transformers`, `accelerate`, and `safetensors`):

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install "git+https://github.com/microsoft/MoGe.git"
.venv/bin/pip install -e ".[dev]"
```

## Configuration

Optional environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `INKRAY_MOGE_MODEL_ID` | `Ruicheng/moge-3-vitl` | Hugging Face MoGe-3 checkpoint ID |
| `INKRAY_APPEARANCE_MODEL_ID` | `prs-eth/marigold-iid-lighting-v1-1` | Hugging Face Marigold IID Lighting checkpoint ID |
| `INKRAY_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `INKRAY_MAX_IMAGE_SIZE` | `1024` | Maximum image edge sent to MoGe-3 |
| `INKRAY_APPEARANCE_MAX_IMAGE_SIZE` | `1024` | Maximum image edge sent to Marigold |
| `INKRAY_MOGE_RESOLUTION_LEVEL` | `9` | MoGe-3 inference resolution level |
| `INKRAY_MOGE_REFINE_STEPS` | `3` | MoGe-3 iterative refinement steps |
| `INKRAY_MOGE_FP16` | `true` | Run MoGe-3 refinement in half precision on CUDA |
| `INKRAY_APPEARANCE_FP16` | `true` | Load fp16 Marigold variant on CUDA |
| `INKRAY_APPEARANCE_STEPS` | `4` | Marigold LCM denoising steps |
| `INKRAY_APPEARANCE_RES` | `768` | Marigold internal processing resolution |
| `INKRAY_APPEARANCE_ENSEMBLE` | `1` | Marigold prediction ensemble size |
| `INKRAY_APPEARANCE_SEED` | `42` | Deterministic diffusion seed |
| `INKRAY_OFFLOAD_INACTIVE_MODELS` | `true` | Move inactive model weights to CPU before switching on CUDA |
| `INKRAY_CACHE_DIR` | `backend/.cache/geometry` | Disk cache directory for inferred geometry |
| `INKRAY_APPEARANCE_CACHE_DIR` | `backend/.cache/appearance` | Disk cache directory for inferred appearance layers |

## Running and Testing

```bash
cd backend

# Start the API server
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# Run unit tests (synthetic/mocked inference)
.venv/bin/pytest -v

# Run real MoGe-3 CUDA + cache smoke test
.venv/bin/python scripts/smoke_moge.py

# Run real Marigold IID Lighting CUDA + cache + MoGe swap smoke test
.venv/bin/python scripts/smoke_marigold.py --with-moge

# Evaluate appearance calibration & compositing across synthetic lighting scenarios
.venv/bin/python scripts/evaluate_appearance.py
```

## Model Attribution & License Notes

- **MoGe-3 (`Ruicheng/moge-3-vitl` / `microsoft/MoGe`):** Licensed under the **MIT License** (Copyright Microsoft Corporation). Certain bundled or upstream backbone components (such as DINOv2 elements) carry **Apache License 2.0**.
- **Marigold IID Lighting v1.1 (`prs-eth/marigold-iid-lighting-v1-1`):** Run via **Hugging Face Diffusers** (Apache-2.0), while the pretrained model checkpoint itself is released under the **CreativeML OpenRAIL++-M** license, which includes usage and redistribution terms separate from Inkray's MIT license.

Any redistribution or deployment of downloaded pretrained checkpoints or generated derivatives must comply with their respective upstream license terms.

## Planned Extensions

- Interactive surface segmentation using **SAM 2**.
