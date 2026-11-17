import { clamp } from '../editor/viewportTransforms';
import type { PlaneFootprint } from '../geometry/planeFrame';
import type { LoadedImage } from '../types/editor';
import {
  type BrushStroke,
  compositeStrokeMaskOverRgba,
  rasterizeStrokeToMask,
  rebuildSurfaceDecalRgba,
} from './brushEngine';

export interface ArtworkSnapshot {
  width: number;
  height: number;
  strokes: BrushStroke[];
}

export interface DrawnArtwork {
  kind: 'drawn';
  id: string;
  name: string;
  width: number;
  height: number;
  strokes: BrushStroke[];
  undoStack: ArtworkSnapshot[];
  redoStack: ArtworkSnapshot[];
  revision: number;
}

export interface ImportedArtwork {
  kind: 'imported';
  id: string;
  name: string;
  width: number;
  height: number;
  image: LoadedImage;
  revision: number;
}

export type Artwork = DrawnArtwork | ImportedArtwork;

export const MAX_ARTWORK_DIMENSION = 4096;
export const MIN_ARTWORK_DIMENSION = 16;
export const DEFAULT_ARTWORK_DIMENSION = 512;
export const ARTWORK_DRAG_MIME = 'application/x-inkray-artwork';
const MAX_ARTWORK_UNDO_STEPS = 100;

let nextArtworkSeq = 1;
const importedRgbaCache = new WeakMap<LoadedImage, Uint8ClampedArray>();
const drawnRgbaCache = new WeakMap<
  BrushStroke[],
  { width: number; height: number; rgba: Uint8ClampedArray }
>();

export function clampArtworkDimension(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_ARTWORK_DIMENSION;
  }
  return clamp(Math.round(value), MIN_ARTWORK_DIMENSION, MAX_ARTWORK_DIMENSION);
}

/** Pixel-space rectangle matching DecalTextureSpec 1 texel = 1 pixel for BrushEngine reuse. */
export function getArtworkFootprint(width: number, height: number): PlaneFootprint {
  return {
    minU: 0,
    maxU: Math.max(1, width),
    minV: 0,
    maxV: Math.max(1, height),
  };
}

export function rasterizeArtworkRgba(artwork: Artwork): Uint8ClampedArray {
  if (artwork.kind === 'imported') {
    const cached = importedRgbaCache.get(artwork.image);
    if (cached && cached.length === artwork.width * artwork.height * 4) {
      return cached;
    }
    const extracted = artwork.image.getPixelData().data;
    importedRgbaCache.set(artwork.image, extracted);
    return extracted;
  }

  const cached = drawnRgbaCache.get(artwork.strokes);
  if (cached && cached.width === artwork.width && cached.height === artwork.height) {
    return cached.rgba;
  }

  const rgba = rebuildSurfaceDecalRgba(
    artwork.strokes,
    getArtworkFootprint(artwork.width, artwork.height),
    { width: artwork.width, height: artwork.height },
  );
  drawnRgbaCache.set(artwork.strokes, { width: artwork.width, height: artwork.height, rgba });
  return rgba;
}

export function rasterizeArtworkWithLiveStroke(
  artwork: DrawnArtwork,
  liveStroke: BrushStroke | null,
): Uint8ClampedArray {
  const base = rasterizeArtworkRgba(artwork);
  if (!liveStroke || liveStroke.surfaceId !== artwork.id || liveStroke.points.length === 0) {
    return base;
  }
  const out = new Uint8ClampedArray(base);
  const spec = { width: artwork.width, height: artwork.height };
  const mask = rasterizeStrokeToMask(
    liveStroke,
    getArtworkFootprint(artwork.width, artwork.height),
    spec,
  );
  compositeStrokeMaskOverRgba(
    out,
    mask,
    liveStroke.brush,
    spec.width,
    undefined,
    liveStroke.mode ?? 'paint',
  );
  return out;
}

/**
 * Converts straight RGBA texels into premultiplied RGBA bytes for halo-free GPU bilinear sampling.
 */
export function toPremultipliedRgba(straightRgba: Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(straightRgba.length);
  for (let i = 0; i < straightRgba.length; i += 4) {
    const aByte = straightRgba[i + 3] ?? 0;
    if (aByte === 255) {
      out[i] = straightRgba[i] ?? 0;
      out[i + 1] = straightRgba[i + 1] ?? 0;
      out[i + 2] = straightRgba[i + 2] ?? 0;
      out[i + 3] = 255;
    } else if (aByte === 0) {
      // Leave RGB zeroed when alpha is zero so neighboring pixels never pick up dark/light bleed.
      out[i] = 0;
      out[i + 1] = 0;
      out[i + 2] = 0;
      out[i + 3] = 0;
    } else {
      const aNorm = aByte / 255;
      out[i] = Math.round((straightRgba[i] ?? 0) * aNorm);
      out[i + 1] = Math.round((straightRgba[i + 1] ?? 0) * aNorm);
      out[i + 2] = Math.round((straightRgba[i + 2] ?? 0) * aNorm);
      out[i + 3] = aByte;
    }
  }
  return out;
}

function artworkSnapshot(artwork: DrawnArtwork): ArtworkSnapshot {
  return { width: artwork.width, height: artwork.height, strokes: artwork.strokes };
}

export const ArtworkModel = {
  createBlank(options?: { name?: string; width?: number; height?: number }): DrawnArtwork {
    const width = clampArtworkDimension(options?.width ?? DEFAULT_ARTWORK_DIMENSION);
    const height = clampArtworkDimension(options?.height ?? DEFAULT_ARTWORK_DIMENSION);
    const seq = nextArtworkSeq++;
    return {
      kind: 'drawn',
      id: `artwork-${seq}`,
      name: options?.name?.trim() || `Artwork ${seq}`,
      width,
      height,
      strokes: [],
      undoStack: [],
      redoStack: [],
      revision: 1,
    };
  },

  createFromImage(image: LoadedImage): ImportedArtwork {
    const seq = nextArtworkSeq++;
    return {
      kind: 'imported',
      id: `artwork-${seq}`,
      name: image.metadata.name,
      width: image.metadata.width,
      height: image.metadata.height,
      image,
      revision: 1,
    };
  },

  resizeDrawn(artwork: DrawnArtwork, width: number, height: number): DrawnArtwork {
    const nextW = clampArtworkDimension(width);
    const nextH = clampArtworkDimension(height);
    if (nextW === artwork.width && nextH === artwork.height) {
      return artwork;
    }
    return {
      ...artwork,
      width: nextW,
      height: nextH,
      undoStack: [...artwork.undoStack, artworkSnapshot(artwork)].slice(-MAX_ARTWORK_UNDO_STEPS),
      redoStack: [],
      revision: artwork.revision + 1,
    };
  },

  commitDrawnStroke(artwork: DrawnArtwork, stroke: BrushStroke): DrawnArtwork {
    if (stroke.surfaceId !== artwork.id || stroke.points.length === 0) {
      return artwork;
    }
    const nextUndo = [...artwork.undoStack, artworkSnapshot(artwork)];
    const trimmedUndo =
      nextUndo.length > MAX_ARTWORK_UNDO_STEPS
        ? nextUndo.slice(nextUndo.length - MAX_ARTWORK_UNDO_STEPS)
        : nextUndo;
    return {
      ...artwork,
      strokes: [...artwork.strokes, stroke],
      undoStack: trimmedUndo,
      redoStack: [],
      revision: artwork.revision + 1,
    };
  },

  clearDrawn(artwork: DrawnArtwork): DrawnArtwork {
    if (artwork.strokes.length === 0) {
      return artwork;
    }
    const nextUndo = [...artwork.undoStack, artworkSnapshot(artwork)];
    const trimmedUndo =
      nextUndo.length > MAX_ARTWORK_UNDO_STEPS
        ? nextUndo.slice(nextUndo.length - MAX_ARTWORK_UNDO_STEPS)
        : nextUndo;
    return {
      ...artwork,
      strokes: [],
      undoStack: trimmedUndo,
      redoStack: [],
      revision: artwork.revision + 1,
    };
  },

  canUndo(artwork: Artwork | null): boolean {
    return artwork !== null && artwork.kind === 'drawn' && artwork.undoStack.length > 0;
  },

  canRedo(artwork: Artwork | null): boolean {
    return artwork !== null && artwork.kind === 'drawn' && artwork.redoStack.length > 0;
  },

  undoDrawn(artwork: DrawnArtwork): DrawnArtwork {
    if (artwork.undoStack.length === 0) {
      return artwork;
    }
    const prev = artwork.undoStack[artwork.undoStack.length - 1];
    if (!prev) {
      return artwork;
    }
    return {
      ...artwork,
      ...prev,
      undoStack: artwork.undoStack.slice(0, -1),
      redoStack: [...artwork.redoStack, artworkSnapshot(artwork)],
      revision: artwork.revision + 1,
    };
  },

  redoDrawn(artwork: DrawnArtwork): DrawnArtwork {
    if (artwork.redoStack.length === 0) {
      return artwork;
    }
    const next = artwork.redoStack[artwork.redoStack.length - 1];
    if (!next) {
      return artwork;
    }
    return {
      ...artwork,
      ...next,
      undoStack: [...artwork.undoStack, artworkSnapshot(artwork)],
      redoStack: artwork.redoStack.slice(0, -1),
      revision: artwork.revision + 1,
    };
  },
};
