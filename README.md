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

Inkray implements monocular 3D geometry inference, 2.5D mesh/point cloud visualization, and geometry-aware planar surface painting:

- **Implemented:**
  - Frontend workspace (`React` + `TypeScript` + `Vite` + `Three.js`) with strict TypeScript checking, ESLint, Prettier, and Vitest.
  - Dark-themed editor layout (top toolbar, left tool rail, central viewport, right inspector sidebar, bottom status bar).
  - JPEG and PNG image importing via file picker and drag-and-drop, with format/header validation, error handling, non-destructive image replacement, and access to unmodified source pixels.
  - High-DPI viewport with cursor-anchored wheel/trackpad zoom, middle-mouse and `Space + left-drag` panning, Fit to Screen, 100% zoom, and resize handling.
  - Python + FastAPI backend integrating **MoGe-3** (`Ruicheng/moge-3-vitl`) for monocular depth, surface normals, 3D point maps, valid mask, and normalized camera intrinsics.
  - Content-addressed disk caching of inferred geometry (`.npz` + metadata JSON) with binary array HTTP endpoints (`float32` / `uint8`) and request cancellation.
  - Viewport visualization modes (`RGB`, `Depth`, `Normals`, `Mask`, and interactive `3D` `Points | 2.5D Mesh` viewer in Three.js with `OrbitControls`), plus live cursor depth sampling.
  - Geometry-aware planar painting: single-request backend NumPy/SciPy RANSAC + connected surface region growing (`POST /api/geometry/{id}/surface/select`), client-side stable plane coordinate frames, perspective GPU decal projection via WebGL2 shaders, depth/normal foreground occlusion handling, anti-accumulation single-stroke brush compositing, cross-surface undo/redo history, and full-resolution PNG export.
- **Not Yet Implemented:**
  - Marigold IID intrinsic lighting decomposition, relighting-aware compositing, SAM 2 interactive segmentation, curved-surface painting, and named paint layers.

---

## Project Structure

```text
inkray/
├── frontend/                  # React + TypeScript + Vite editor application
│   ├── src/
│   │   ├── api/               # Binary HTTP client for backend geometry endpoints
│   │   ├── components/        # UI panels, toolbars, and status bar
│   │   ├── editor/            # 2D canvas renderer, 3D point/mesh viewer, previews, transforms
│   │   ├── geometry/          # OpenCV geometry helpers, plane frame projection, and 2.5D mesh builder
│   │   ├── hooks/             # Custom React hooks (resize, shortcuts, drag-and-drop import)
│   │   ├── painting/          # Brush engine, decal parameterization, visibility math, GPU renderer, document history
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
│   │   ├── api/               # FastAPI routes for inference metadata, binary arrays, and surface selection
│   │   ├── cache/             # Content-addressed disk cache for inferred geometry
│   │   ├── geometry/          # Typed geometry containers, transforms, RANSAC plane fitter, and surface selector
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

### Geometry-Aware Painting Workflow

1. Import a JPEG or PNG photograph (`O`) and click **Generate Geometry**.
2. Optionally switch to `3D` view and toggle between **Points** and **2.5D Mesh** to inspect surface continuity.
3. Activate **Select Planar Surface** (`S`) and click one or more planar regions (walls, floor, tabletops, facades) to add painting surfaces.
4. Open the **Surfaces** tab in the right sidebar to drag-and-drop surface priorities (`#1` wins overlapping intersections), adjust **Footprint Extent** (`50%`-`250%`), or toggle ownership overlays.
5. Switch manually to **Surface Brush** (`B`) or **Surface Eraser** (`E`). Paint or erase continuously across selected planes with a live perspective brush ring, automatic foreground depth occlusion, and atomic cross-surface strokes.
6. Use **Undo** (`Ctrl/Cmd + Z`), **Redo** (`Ctrl/Cmd + Shift + Z`), or **Export PNG** (opens a Save dialog where supported, or downloads `<name>-inkray.png`) while staying in the live RGB painting viewport.

### Keyboard Shortcuts

| Shortcut | Action |
| --- | --- |
| `O` or `Ctrl/Cmd + O` | Import / replace image |
| `Space + Left-Drag` or `Middle-Drag` | Pan viewport |
| `Mouse Wheel` / `Trackpad Pinch` | Smooth zoom around cursor |
| `V` | Pointer / inspect tool |
| `H` | Hand (pan) tool |
| `S` | Select Planar Surface tool (after geometry is ready) |
| `B` | Surface Brush tool (after selecting at least one plane) |
| `E` | Surface Eraser tool (after selecting at least one plane) |
| `[` / `]` | Decrease / increase brush size |
| `Ctrl/Cmd + Z` | Undo painting, surface add/remove, or priority reorder |
| `Ctrl/Cmd + Shift + Z` or `Ctrl + Y` | Redo |
| `F` or `0` or `Ctrl/Cmd + 0` | Fit image to screen |
| `1` or `Ctrl/Cmd + 1` | Zoom to 100% (1 source pixel = 1 CSS pixel) |
| `+` / `-` | Zoom in / out |

### Known Limitations

- Painting operates on local planar approximations; curved surfaces will depart from the fitted plane away from the planar support region.
- Visibility and occlusion quality depend on MoGe-3 depth and normal accuracy near silhouettes.
- Export resolution is bounded by the WebGL `MAX_TEXTURE_SIZE` and `MAX_RENDERBUFFER_SIZE` limits of the active GPU.
- Lighting-aware shading and interactive GPU frame-rate measurements are not included yet.
