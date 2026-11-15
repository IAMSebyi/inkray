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

Inkray currently implements the client-side editor shell and 2D image viewport:

- **Implemented:**
  - Frontend workspace (`React` + `TypeScript` + `Vite`) with strict TypeScript checking, ESLint, Prettier, and Vitest.
  - Dark-themed editor layout (top toolbar, left tool rail, central viewport, right inspector sidebar, bottom status bar).
  - JPEG and PNG image importing via file picker and drag-and-drop, with format/header validation, error handling, non-destructive image replacement, and access to unmodified source pixels.
  - High-DPI Canvas 2D viewport with cursor-anchored wheel/trackpad zoom, middle-mouse and `Space + left-drag` panning, Fit to Screen, 100% zoom, and resize handling.
  - Reusable coordinate transform module (`client <-> viewport <-> image`) with unit tests.
  - Minimal `backend/` scaffold reserved for the future Python/FastAPI service.
- **Not Yet Implemented:**
  - MoGe-3 geometry reconstruction, Marigold IID intrinsic decomposition, SAM 2 segmentation, 3D mesh/Three.js rendering, surface brush tools, occlusion, lighting compositing, and layer editing.

---

## Project Structure

```text
inkray/
├── frontend/                  # React + TypeScript + Vite editor application
│   ├── src/
│   │   ├── components/        # UI panels, toolbars, and status bar
│   │   ├── editor/            # Viewport component, Canvas 2D renderer, image loader, coordinate math
│   │   ├── hooks/             # Custom React hooks (resize, shortcuts, drag-and-drop import)
│   │   ├── state/             # Editor reducer, context provider, and cursor coordinate store
│   │   ├── styles/            # Design tokens, global styles, and editor layout CSS
│   │   ├── types/             # Shared TypeScript definitions
│   │   └── utils/             # File validation and display formatting helpers
│   ├── eslint.config.js
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── package.json
├── backend/                   # Scaffold for future Python/FastAPI + PyTorch service
│   ├── app/
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

### Install Dependencies

```bash
cd frontend
npm install
```

### Run the Development Server

```bash
cd frontend
npm run dev
```

Vite will print the local development URL (typically `http://localhost:5173`).

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
