# Inkray

**Paint beyond pixels.**

Inkray is an open-source, geometry-aware image editor inspired by Adobe's Project Deep Brush. Instead of treating photographs as flat 2D pixel grids, Inkray aims to reconstruct approximate 3D scene geometry and intrinsic illumination from ordinary images so users can paint directly onto inferred surfaces.

---

## Planned Capabilities

- **3D Geometry-Aware Painting:** Reconstruct monocular depth, normals, point maps, and camera intrinsics (**MoGe-3**) so brush strokes follow surface perspective and curvature.
- **Occlusion Awareness:** Paint applied behind foreground subjects respects depth ordering rather than overlapping them.
- **Lighting-Aware Compositing:** Decompose scenes into albedo, shading, and residual layers (**Marigold IID Lighting**) so painted strokes inherit original shadows and illumination.
- **Interactive Surface Selection:** Segment and isolate real-world surfaces (**SAM 2**) for controlled editing.
- **Real-Time GPU Viewport:** Interactive WebGL2 / Three.js rendering and non-destructive stroke and layer editing without repeated ML inference while painting.

---

## Current Status

Inkray currently implements the interactive 2D/3D viewport and monocular geometry inference backend:

- **Implemented:**
  - Frontend workspace (`React` + `TypeScript` + `Vite` + `Three.js`) with strict TypeScript checking, ESLint, Prettier, and Vitest.
  - Dark-themed editor layout (top toolbar, left tool rail, central viewport, right inspector sidebar, bottom status bar).
  - JPEG and PNG image importing via file picker and drag-and-drop, with format/header validation, error handling, non-destructive image replacement, and access to unmodified source pixels.
  - High-DPI Canvas 2D viewport with cursor-anchored wheel/trackpad zoom, middle-mouse and `Space + left-drag` panning, Fit to Screen, 100% zoom, and resize handling.
  - Python + FastAPI backend integrating **MoGe-3** (`Ruicheng/moge-3-vitl`) for monocular depth, surface normals, 3D point maps, valid mask, and normalized camera intrinsics.
  - Content-addressed disk caching of inferred geometry (`.npz` + metadata JSON) with binary array HTTP endpoints (`float32` / `uint8`) and request cancellation.
  - Viewport visualization modes (`RGB`, `Depth`, `Normals`, `Mask`, and interactive `3D` point cloud in Three.js with `OrbitControls`), plus live cursor depth sampling.
- **Not Yet Implemented:**
  - Marigold IID intrinsic lighting decomposition, SAM 2 surface segmentation, 3D brush projection, occlusion-aware painting, lighting-aware compositing, and paint layers.

---

## Project Structure

```text
inkray/
├── frontend/                  # React + TypeScript + Vite editor application
│   ├── src/
│   │   ├── api/               # Binary HTTP client for backend geometry endpoints
│   │   ├── components/        # UI panels, toolbars, and status bar
│   │   ├── editor/            # 2D canvas renderer, 3D point cloud viewer, geometry previews, transforms
│   │   ├── hooks/             # Custom React hooks (resize, shortcuts, drag-and-drop import)
│   │   ├── state/             # Editor reducer, context provider, and cursor coordinate store
│   │   ├── styles/            # Design tokens, global styles, and editor layout CSS
│   │   ├── types/             # Shared TypeScript definitions
│   │   └── utils/             # File validation and display formatting helpers
│   ├── eslint.config.js
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── package.json
├── backend/                   # Python + FastAPI + PyTorch (MoGe-3) geometry service
│   ├── app/
│   │   ├── api/               # FastAPI routes for inference metadata and binary arrays
│   │   ├── cache/             # Content-addressed disk cache for inferred geometry
│   │   ├── geometry/          # Typed geometry containers and coordinate transforms
│   │   ├── inference/         # Model wrapper abstraction and MoGe-3 implementation
│   │   ├── utils/             # Pillow image decoding and EXIF orientation normalization
│   │   ├── config.py          # Environment-driven backend configuration
│   │   └── main.py            # FastAPI app factory, CORS, and model lifespan
│   ├── scripts/               # Standalone model + cache smoke test
│   ├── tests/                 # Fast pytest suite for geometry math, caching, and API routes
│   ├── pyproject.toml
│   └── README.md
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- **Node.js** 20+ (tested on Node 22)
- **npm** 10+
- **Python** 3.10+ (CUDA-capable GPU recommended for MoGe-3 inference; CPU fallback is supported)

### Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Install and Run the Backend

See [backend/README.md](backend/README.md) for full Python environment setup (PyTorch CUDA + official MoGe-3 package). Once `backend/.venv` is configured:

```bash
cd backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Run the Frontend Development Server

```bash
cd frontend
npm run dev
```

Vite proxies `/api/*` requests to `http://127.0.0.1:8000` and prints the local development URL (typically `http://localhost:5173`).

---

## Build and Testing

Run all commands from the `frontend/` directory:

```bash
# Run unit tests (Vitest)
npm test

# Type-check with TypeScript strict mode
npm run typecheck

# Lint with ESLint
npm run lint

# Check formatting with Prettier
npm run format:check

# Build for production (tsc -b && vite build)
npm run build
```

### Keyboard Shortcuts (Viewport)

| Shortcut | Action |
| --- | --- |
| `O` or `Ctrl/Cmd + O` | Import / replace image |
| `Space + Left-Drag` or `Middle-Drag` | Pan viewport |
| `Mouse Wheel` / `Trackpad Pinch` | Smooth zoom around cursor |
| `V` | Pointer / inspect tool |
| `H` | Hand (pan) tool |
| `F` or `0` or `Ctrl/Cmd + 0` | Fit image to screen |
| `1` or `Ctrl/Cmd + 1` | Zoom to 100% (1 source pixel = 1 CSS pixel) |
| `+` / `-` | Zoom in / out |
