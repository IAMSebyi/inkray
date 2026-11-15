# Inkray Backend

Python + FastAPI inference service for Inkray, providing monocular geometry estimation via **MoGe-3** (`Ruicheng/moge-3-vitl`), content-addressed disk caching, and binary array transport to the frontend.

## Implemented Functionality

- **MoGe-3 Inference:** Loads `moge.model.v3.MoGeModel` once at server startup on CUDA when available (or CPU fallback), decodes JPEG/PNG uploads with EXIF orientation applied, and bounds input size while preserving aspect ratio.
- **Typed Geometry Representation:** Wraps `DepthMap`, `NormalMap`, `Points3D`, `CameraIntrinsics`, and `ValidMask` inside `GeometryResult` using OpenCV camera conventions (`+X` right, `+Y` down, `+Z` forward, normalized `[0, 1]` intrinsics) alongside conversion helpers to Three.js world space (`+Y` up, `-Z` forward).
- **Content-Addressed Disk Cache:** Keys inferred geometry by SHA-256 of the uploaded image bytes, model ID, inference parameters, and cache schema version. Stores arrays compressed in `geometry.npz` and JSON metadata in `meta.json` under `backend/.cache/geometry` by default (kept separate from Hugging Face weight downloads in `~/.cache/huggingface`).
- **Binary API Endpoints:**
  - `POST /api/geometry/infer` accepts multipart `file` uploads, returns `GeometryRecordMetadata` JSON, and checks for client disconnect before writing incomplete cache records.
  - `GET /api/geometry/{geometry_id}` returns cached metadata.
  - `GET /api/geometry/{geometry_id}/maps/{kind}` serves `depth` (`float32`), `points` (`float32`), `normals` (`float32`), and `mask` (`uint8`) as `application/octet-stream` with `X-Array-Shape` and `X-Array-Dtype` headers.

## Setup

Create a virtual environment inside `backend/.venv`, install PyTorch for your CUDA version (or CPU), and install the official MoGe package plus backend dependencies:

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
| `INKRAY_MOGE_MODEL_ID` | `Ruicheng/moge-3-vitl` | Hugging Face checkpoint ID |
| `INKRAY_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `INKRAY_MAX_IMAGE_SIZE` | `1536` | Maximum image edge sent to inference |
| `INKRAY_RESOLUTION_LEVEL` | `9` | MoGe-3 inference resolution level |
| `INKRAY_REFINE_STEPS` | `3` | MoGe-3 iterative refinement steps |
| `INKRAY_USE_FP16` | `false` | Run MoGe-3 refinement in half precision |
| `INKRAY_CACHE_DIR` | `backend/.cache/geometry` | Disk cache directory for inferred geometry |

## Running and Testing

```bash
cd backend

# Start the API server
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# Run unit tests (synthetic/mocked inference)
.venv/bin/pytest -v

# Run real MoGe-3 CUDA + cache smoke test (downloads weights on first run)
.venv/bin/python scripts/smoke_moge.py
```

## Planned Extensions

- Intrinsic image decomposition using **Marigold IID Lighting** (albedo, shading, and residual layers).
- Interactive surface segmentation using **SAM 2**.
