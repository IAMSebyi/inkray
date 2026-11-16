# Inkray

> **Paint beyond pixels.**

Inkray is an open-source, geometry- and lighting-aware photo editor inspired by Adobe's Project Deep Brush. Instead of editing photographs solely in flat screen coordinates, Inkray reconstructs monocular 3D geometry and intrinsic lighting layers once per photograph, then runs real-time perspective painting, curved-surface artwork wrapping, foreground occlusion, and illumination-aware compositing directly on the GPU.

---

## Contents

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Workflow & Shortcuts](#workflow--shortcuts)
- [Development & Verification](#development--verification)
- [Project Structure](#project-structure)
- [Limitations & Roadmap](#limitations--roadmap)
- [Contributing](#contributing)
- [License & Model Attribution](#license--model-attribution)

---

## Capabilities

- **Automatic 3D Surface-Aware Painting:** Pick **Brush** (`B`) or **Eraser** (`E`) and paint directly on any photograph without manual plane selection. Inkray inspects the inferred geometry beneath each stroke and automatically chooses:
  - *Planar projection* with deterministic local PCA plane fitting and coplanar bridging behind foreground occluders.
  - *Curved / irregular surface projection* using intrinsic first-fundamental-form metrics and bounded shortest-path geodesic distances across the continuous 2.5D mesh without crossing depth jumps or holes.
  - *Clean 2D fallback* when geometry has not been inferred yet or when painting over unmapped regions.
- **2D Artwork Canvas & Curved Surface Wrapping:** Create multi-stroke transparent graphics in a dedicated 2D artwork workspace or import PNG/JPEG images preserving alpha, then place them onto photographs:
  - Automatically selects **planar projection** on flat walls/floors/surfaces, **bounded intrinsic surface wrapping** (parallel-transported tangent parameterizations with local relaxation) on cylindrical, spherical, and smoothly curved objects, or **2D overlay fallback** on invalid/degenerate regions.
  - Supports interactive translation (including seamless surface-to-surface transitions across connected walls, floors, and objects), aspect-preserving scaling, rotation, opacity control, explicit re-anchoring, and live non-destructive edits to shared source artwork assets.
- **Foreground Occlusion:** Strokes and placed artwork depth-test against reconstructed surfaces and occluders so edits naturally sit behind closer foreground elements.
- **Lighting-Aware Compositing:** Decomposes photographs via **Marigold IID Lighting v1.1** into `Albedo`, `Shading`, and `Residual` layers and offers live GPU-accelerated **Flat**, **Shading** (diffuse illumination/shadow preservation), and **Texture** (illuminated micro-texture transfer with configurable **Texture Detail**) paint modes.
- **Inspection, History & High-Resolution Export:** Switch seamlessly between `RGB`, `Depth`, `Normals`, `Mask`, interactive `3D` point cloud, `Mesh` (2.5D triangle mesh), `Albedo`, `Shading`, and `Residual` views, inspect unencumbered results with **Final View** (`P`), undo/redo across brush and placement edits, and export at full source photograph resolution as PNG.

---

## Architecture

Inkray splits heavy monocular ML inference from interactive editing so all painting, surface unwrapping, transforms, and compositing happen locally without per-frame server calls:

```mermaid
flowchart LR
  User["User Viewport<br/>(React + Canvas / WebGL2)"] -->|"POST /api/geometry/infer<br/>(JPEG / PNG)"| API["FastAPI Backend"]
  API -->|"MoGe-3 (CUDA / CPU)"| GeoCache["Disk Geometry Cache<br/>(.npz + JSON metadata)"]
  API -->|"Marigold IID Lighting v1.1"| AppCache["Disk Appearance Cache<br/>(.npz + JSON metadata)"]
  GeoCache -->|"float32 / uint8 maps<br/>(Depth, Points, Normals, Mask)"| User
  AppCache -->|"linear float32 maps<br/>(Albedo, Shading, Residual)"| User
  User -->|"Planar / Geodesic / Intrinsic UV<br/>Compositing Shader"| GPU["Three.js WebGL2 Renderer<br/>& Full-Res PNG Export"]
```

- **Backend (`backend/`):** Python + FastAPI service running **MoGe-3** (`Ruicheng/moge-3-vitl`) and **Marigold IID Lighting v1.1** (`prs-eth/marigold-iid-lighting-v1-1` via Hugging Face Diffusers), backed by SHA-256 content-addressed `.npz` disk caches and sequential VRAM offloading so both models share an 8 GB GPU budget cleanly.
- **Frontend (`frontend/`):** React 19 + strict TypeScript + Vite + Three.js application. Geometry classification, geodesic solvers, and local tangent parameterizations stay decoupled from React UI components inside `src/geometry/` and `src/painting/`.

---

## Quick Start

### Prerequisites

- **Node.js** 20+ (tested on Node 22) and **npm** 10+
- **Python** 3.10+
- **NVIDIA GPU with CUDA** recommended for fast ML inference (CPU fallback is supported for MoGe-3 and Marigold, at higher latency)

> Note: Running **Generate Geometry** or **Generate Appearance** downloads upstream pretrained weights from Hugging Face into your local cache (`~/.cache/huggingface`) on first use.

### 1. Set Up & Start the Backend

Full setup options (**Anaconda / Miniconda** or **`venv`**) and environment variables are documented in [backend/README.md](backend/README.md). Using Conda:

```bash
conda create -n inkray python=3.10 -y
conda activate inkray

# Install CUDA PyTorch (or omit --index-url for CPU-only systems)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

cd backend
pip install "git+https://github.com/microsoft/MoGe.git"
pip install -e ".[dev]"

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

*(Alternatively, if using `backend/.venv`, start the API with `.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000`.)*

### 2. Install & Start the Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the editor (typically at `http://localhost:5173`) and proxies `/api/*` requests to `http://127.0.0.1:8000`.

---

## Workflow & Shortcuts

### Typical Editing Workflow

1. **Import a Photograph:** Click **Import Image** (`O`) or drag and drop a JPEG/PNG into the viewport. You can paint immediately in 2D or run 3D reconstruction.
2. **Reconstruct Scene Layers:**
   - Click **Generate Geometry** to infer depth, normals, 3D point map, validity mask, and camera intrinsics with MoGe-3.
   - Optionally click **Generate Appearance** to estimate intrinsic albedo, diffuse shading, and residual layers with Marigold IID Lighting v1.1.
3. **Paint or Place Artwork:**
   - Use **Brush** (`B`) or **Eraser** (`E`) directly on the photograph — Inkray automatically anchors each stroke to planar perspective, curved mesh geodesics, or 2D space.
   - Switch to **Artwork** (or click **New Canvas** / **Import Art** in the right sidebar) to paint transparent 2D assets or import PNG/JPEG logos, then drag them onto the photo or click **Place on Photo**.
   - With **Pointer** (`V`), drag placements across surfaces, scale via corners, rotate from the top knob, adjust opacity, or click **Re-anchor** / **Edit Art** in the inspector.
4. **Adjust Compositing & Export:** Choose **Flat**, **Shading**, or **Texture** in **Intrinsic Lighting**, toggle **Final View** (`P`) to inspect without guides, and click **Export PNG** to save at the photograph's full original resolution.

### Keyboard Shortcuts

| Shortcut | Action |
| --- | --- |
| `O` or `Ctrl/Cmd + O` | Import or replace source photograph |
| `Space + Left-Drag` / `Middle-Drag` | Pan viewport |
| `Wheel` / `Trackpad Pinch` | Zoom anchored at cursor |
| `V` | Pointer tool (select and transform artwork placements) |
| `H` | Hand tool (pan viewport) |
| `B` | Brush tool |
| `E` | Eraser tool |
| `[` / `]` | Decrease / increase brush size |
| `P` | Toggle **Final View** (hide/show guides & handles) |
| `Delete` / `Backspace` | Remove selected artwork placement |
| `Ctrl/Cmd + Z` | Undo stroke or placement action |
| `Ctrl/Cmd + Shift + Z` or `Ctrl/Cmd + Y` | Redo |
| `F`, `0`, or `Ctrl/Cmd + 0` | Fit image to viewport |
| `1` or `Ctrl/Cmd + 1` | Zoom to 100% (`1 source px = 1 CSS px`) |
| `+` / `-` | Zoom in / out |

---

## Development & Verification

### Frontend Checks (`frontend/`)

```bash
cd frontend
npm run typecheck      # Strict TypeScript check (tsc -b --noEmit)
npm run lint           # ESLint check
npm run format:check   # Prettier style check
npm test               # Vitest unit & synthetic geometry suite
npm run build          # Production typecheck + Vite bundle
```

### Backend Checks (`backend/`)

```bash
cd backend
pytest -v                                  # API, geometry, appearance, & cache tests
python scripts/smoke_moge.py               # Real MoGe-3 CUDA + disk cache smoke test
python scripts/smoke_marigold.py --with-moge  # Marigold CUDA + MoGe memory-swap check
python scripts/evaluate_appearance.py      # Synthetic shading calibration evaluation
```

---

## Project Structure

```text
inkray/
├── frontend/
│   ├── src/
│   │   ├── api/            # Binary array HTTP clients for geometry & appearance endpoints
│   │   ├── components/     # Toolbars, artwork canvas view, inspector sidebar, status bar
│   │   ├── editor/         # Canvas renderer, 3D point/mesh viewer, viewport transforms
│   │   ├── geometry/       # OpenCV geometry math, plane fitting, continuity, geodesic & unwrap solvers
│   │   ├── hooks/          # Resize observer, DPR, image import, keyboard shortcuts
│   │   ├── painting/       # Brush engine, decal/artwork models, lighting calibration, WebGL2 renderer
│   │   ├── state/          # Immutable editor reducer, context provider, cursor store
│   │   ├── styles/         # Theme tokens, global CSS, and editor layout styles
│   │   ├── types/          # Shared TypeScript interfaces
│   │   └── utils/          # File validation and display formatters
│   ├── eslint.config.js
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routers (/api/geometry, /api/appearance)
│   │   ├── appearance/     # Albedo/shading/residual types, sRGB transforms, shading calibration
│   │   ├── cache/          # SHA-256 disk caches for inferred maps
│   │   ├── geometry/       # Depth/normals/points containers, RANSAC plane fitter, surface selector
│   │   ├── inference/      # MoGe-3 & Marigold IID Lighting wrappers + VRAM offload lock
│   │   ├── utils/          # Image decoding and EXIF orientation normalization
│   │   ├── config.py       # Environment configuration (INKRAY_*)
│   │   └── main.py         # FastAPI app entrypoint & model lifespan
│   ├── scripts/            # Standalone CUDA smoke & evaluation utilities
│   ├── tests/              # Pytest suite (mocked inference for fast CI runs)
│   ├── pyproject.toml
│   └── README.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Limitations & Roadmap

### Current Limitations

- Occlusion accuracy and silhouette fidelity depend on MoGe-3 monocular depth and normal quality around fine boundaries.
- Intrinsic curved-surface wrapping parameterizes a bounded connected 2.5D patch; severe topological folds, needle singularities, or heavily disconnected patches fall back to 2D overlay with an inspector notice.
- Lighting-aware compositing uses an additive-diffuse model over linear albedo and calibrated shading (`I_out = I_orig + alpha * S_cal * (P - A)`); it preserves existing diffuse shadows rather than synthesizing directional lights or physical specular relighting.
- When alternating between MoGe-3 and Marigold on 8 GB GPUs (`INKRAY_OFFLOAD_INACTIVE_MODELS=true`), inactive model weights swap between VRAM and CPU RAM.
- Maximum PNG export size is bounded by the active WebGL2 GPU's `MAX_TEXTURE_SIZE` and `MAX_RENDERBUFFER_SIZE` limits.

### Planned Work

- Interactive object & region segmentation with **SAM 2**.
- Named paint/artwork layer stacks and blend controls.
- Physically grounded relighting and specular highlight transfer.

---

## Contributing

Bug reports, feature proposals, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow and pre-PR checks.

---

## License & Model Attribution

- **Inkray source code:** Released under the [MIT License](LICENSE) (`Copyright (c) 2026 Iorga Sebastian-George`).
- **MoGe-3 (`Ruicheng/moge-3-vitl` / `microsoft/MoGe`):** Licensed under the **MIT License** (upstream backbone components carry **Apache License 2.0**). See [backend/README.md](backend/README.md).
- **Marigold IID Lighting v1.1 (`prs-eth/marigold-iid-lighting-v1-1`):** Run through **Hugging Face Diffusers** (Apache-2.0); the pretrained checkpoint is governed by **CreativeML OpenRAIL++-M**. Users deploying or redistributing downloaded weights or model outputs remain responsible for complying with those upstream license terms.
