import type { PaintMode } from '../types/editor';
import type { PlaneFootprint, SurfaceSelectionResult } from '../geometry/planeFrame';
import { clamp } from '../editor/viewportTransforms';
import type { BrushSettings, BrushStroke } from './brushEngine';
import { type PlanarDecalSurface, getEffectiveFootprint } from './planarDecal';

export interface BrushUiConfig {
  sizeMultiplier: number;
  color: string;
  opacity: number;
  hardness: number;
}

export interface PaintHistorySnapshot {
  surfaces: PlanarDecalSurface[];
  activeSurfaceId: string | null;
  strokes: BrushStroke[];
}

export interface PaintDocumentState {
  surfaces: PlanarDecalSurface[];
  activeSurfaceId: string | null;
  strokes: BrushStroke[];
  undoStack: PaintHistorySnapshot[];
  redoStack: PaintHistorySnapshot[];
  nextSurfaceNumber: number;
  isSelectingSurface: boolean;
  selectionError: string | null;
  showSupportOverlay: boolean;
  showFootprintPreview: boolean;
  paintMode: PaintMode;
  texturePreservation: number;
  brushConfig: BrushUiConfig;
}

export const DEFAULT_BRUSH_CONFIG: BrushUiConfig = {
  sizeMultiplier: 1.0,
  color: '#ff4d4d',
  opacity: 0.9,
  hardness: 0.75,
};

export const DEFAULT_TEXTURE_PRESERVATION = 0.75;

const MAX_UNDO_HISTORY = 100;
let nextSurfaceSeq = 1;

function createSnapshot(doc: PaintDocumentState): PaintHistorySnapshot {
  return {
    surfaces: doc.surfaces,
    activeSurfaceId: doc.activeSurfaceId,
    strokes: doc.strokes,
  };
}

function pushUndoSnapshot(doc: PaintDocumentState): PaintHistorySnapshot[] {
  const next = [...doc.undoStack, createSnapshot(doc)];
  return next.length > MAX_UNDO_HISTORY ? next.slice(next.length - MAX_UNDO_HISTORY) : next;
}

export const PaintDocument = {
  createInitial(
    previousConfig?: BrushUiConfig,
    previousMode?: PaintMode,
    previousTexturePreservation?: number,
  ): PaintDocumentState {
    return {
      surfaces: [],
      activeSurfaceId: null,
      strokes: [],
      undoStack: [],
      redoStack: [],
      nextSurfaceNumber: 1,
      isSelectingSurface: false,
      selectionError: null,
      showSupportOverlay: true,
      showFootprintPreview: true,
      paintMode: previousMode ?? 'shading',
      texturePreservation:
        previousTexturePreservation !== undefined
          ? clamp(previousTexturePreservation, 0, 1)
          : DEFAULT_TEXTURE_PRESERVATION,
      brushConfig: previousConfig ? { ...previousConfig } : { ...DEFAULT_BRUSH_CONFIG },
    };
  },

  setSelectingSurface(doc: PaintDocumentState, isSelecting: boolean): PaintDocumentState {
    if (doc.isSelectingSurface === isSelecting && (!isSelecting || doc.selectionError === null)) {
      return doc;
    }
    return {
      ...doc,
      isSelectingSurface: isSelecting,
      selectionError: isSelecting ? null : doc.selectionError,
    };
  },

  selectSurface(doc: PaintDocumentState, selection: SurfaceSelectionResult): PaintDocumentState {
    const id = `surface-${nextSurfaceSeq++}`;
    const surfaceNumber = doc.nextSurfaceNumber;
    const newSurface: PlanarDecalSurface = {
      id,
      label: `Surface ${surfaceNumber}`,
      colorIndex: (surfaceNumber - 1) % 6,
      frame: selection.frame,
      baseFootprint: selection.suggestedFootprint,
      extentScale: 1.0,
      supportMask: selection.supportMask,
      inlierCount: selection.inlierCount,
      rmsError: selection.rmsError,
      referenceBrushDiameter: selection.referenceBrushDiameter,
    };

    return {
      ...doc,
      surfaces: [...doc.surfaces, newSurface],
      activeSurfaceId: id,
      nextSurfaceNumber: surfaceNumber + 1,
      undoStack: pushUndoSnapshot(doc),
      redoStack: [],
      isSelectingSurface: false,
      selectionError: null,
    };
  },

  setSelectionFailed(doc: PaintDocumentState, reason: string): PaintDocumentState {
    return {
      ...doc,
      isSelectingSurface: false,
      selectionError: reason,
    };
  },

  clearSelectionError(doc: PaintDocumentState): PaintDocumentState {
    if (doc.selectionError === null) {
      return doc;
    }
    return {
      ...doc,
      selectionError: null,
    };
  },

  setActiveSurface(doc: PaintDocumentState, surfaceId: string): PaintDocumentState {
    if (doc.activeSurfaceId === surfaceId) {
      return doc;
    }
    if (!doc.surfaces.some((s) => s.id === surfaceId)) {
      return doc;
    }
    return {
      ...doc,
      activeSurfaceId: surfaceId,
    };
  },

  removeSurface(doc: PaintDocumentState, surfaceId: string): PaintDocumentState {
    if (!doc.surfaces.some((s) => s.id === surfaceId)) {
      return doc;
    }
    const nextSurfaces = doc.surfaces.filter((s) => s.id !== surfaceId);
    const nextStrokes = doc.strokes.filter((s) => s.surfaceId !== surfaceId);
    const nextActive =
      doc.activeSurfaceId === surfaceId ? (nextSurfaces[0]?.id ?? null) : doc.activeSurfaceId;

    return {
      ...doc,
      surfaces: nextSurfaces,
      activeSurfaceId: nextActive,
      strokes: nextStrokes,
      undoStack: pushUndoSnapshot(doc),
      redoStack: [],
    };
  },

  reorderSurfaces(doc: PaintDocumentState, fromIndex: number, toIndex: number): PaintDocumentState {
    const count = doc.surfaces.length;
    if (
      fromIndex < 0 ||
      fromIndex >= count ||
      toIndex < 0 ||
      toIndex >= count ||
      fromIndex === toIndex
    ) {
      return doc;
    }
    const nextSurfaces = [...doc.surfaces];
    const [moved] = nextSurfaces.splice(fromIndex, 1);
    if (!moved) {
      return doc;
    }
    nextSurfaces.splice(toIndex, 0, moved);

    return {
      ...doc,
      surfaces: nextSurfaces,
      undoStack: pushUndoSnapshot(doc),
      redoStack: [],
    };
  },

  setSurfaceExtent(
    doc: PaintDocumentState,
    surfaceId: string,
    extentScale: number,
  ): PaintDocumentState {
    const clampedScale = clamp(extentScale, 0.5, 2.5);
    const target = doc.surfaces.find((surface) => surface.id === surfaceId);
    if (!target || Math.abs(target.extentScale - clampedScale) < 1e-4) {
      return doc;
    }
    return {
      ...doc,
      surfaces: doc.surfaces.map((surface) =>
        surface.id === surfaceId ? { ...surface, extentScale: clampedScale } : surface,
      ),
    };
  },

  setBrushConfig(doc: PaintDocumentState, patch: Partial<BrushUiConfig>): PaintDocumentState {
    const nextConfig: BrushUiConfig = {
      sizeMultiplier:
        patch.sizeMultiplier !== undefined
          ? clamp(patch.sizeMultiplier, 0.2, 5.0)
          : doc.brushConfig.sizeMultiplier,
      color: patch.color ?? doc.brushConfig.color,
      opacity:
        patch.opacity !== undefined ? clamp(patch.opacity, 0.05, 1.0) : doc.brushConfig.opacity,
      hardness:
        patch.hardness !== undefined ? clamp(patch.hardness, 0.0, 1.0) : doc.brushConfig.hardness,
    };
    return {
      ...doc,
      brushConfig: nextConfig,
    };
  },

  setOverlayOptions(
    doc: PaintDocumentState,
    patch: { showSupportOverlay?: boolean; showFootprintPreview?: boolean },
  ): PaintDocumentState {
    return {
      ...doc,
      showSupportOverlay: patch.showSupportOverlay ?? doc.showSupportOverlay,
      showFootprintPreview: patch.showFootprintPreview ?? doc.showFootprintPreview,
    };
  },

  setPaintMode(doc: PaintDocumentState, paintMode: PaintMode): PaintDocumentState {
    if (doc.paintMode === paintMode) {
      return doc;
    }
    return {
      ...doc,
      paintMode,
    };
  },

  setTexturePreservation(doc: PaintDocumentState, value: number): PaintDocumentState {
    const next = clamp(value, 0, 1);
    if (Math.abs(doc.texturePreservation - next) < 1e-4) {
      return doc;
    }
    return {
      ...doc,
      texturePreservation: next,
    };
  },

  commitStrokes(doc: PaintDocumentState, incoming: readonly BrushStroke[]): PaintDocumentState {
    const surfaceIds = new Set(doc.surfaces.map((s) => s.id));
    const valid = incoming.filter((s) => s.points.length > 0 && surfaceIds.has(s.surfaceId));
    if (valid.length === 0) {
      return doc;
    }
    return {
      ...doc,
      strokes: [...doc.strokes, ...valid],
      undoStack: pushUndoSnapshot(doc),
      redoStack: [],
    };
  },

  commitStroke(doc: PaintDocumentState, stroke: BrushStroke): PaintDocumentState {
    return PaintDocument.commitStrokes(doc, [stroke]);
  },

  undo(doc: PaintDocumentState): PaintDocumentState {
    if (doc.undoStack.length === 0) {
      return doc;
    }
    const prev = doc.undoStack[doc.undoStack.length - 1];
    if (!prev) {
      return doc;
    }
    return {
      ...doc,
      surfaces: prev.surfaces,
      activeSurfaceId: prev.activeSurfaceId,
      strokes: prev.strokes,
      undoStack: doc.undoStack.slice(0, -1),
      redoStack: [...doc.redoStack, createSnapshot(doc)],
    };
  },

  redo(doc: PaintDocumentState): PaintDocumentState {
    if (doc.redoStack.length === 0) {
      return doc;
    }
    const next = doc.redoStack[doc.redoStack.length - 1];
    if (!next) {
      return doc;
    }
    return {
      ...doc,
      surfaces: next.surfaces,
      activeSurfaceId: next.activeSurfaceId,
      strokes: next.strokes,
      undoStack: [...doc.undoStack, createSnapshot(doc)],
      redoStack: doc.redoStack.slice(0, -1),
    };
  },

  canUndo(doc: PaintDocumentState): boolean {
    return doc.undoStack.length > 0;
  },

  canRedo(doc: PaintDocumentState): boolean {
    return doc.redoStack.length > 0;
  },

  getActiveSurface(doc: PaintDocumentState): PlanarDecalSurface | null {
    if (!doc.activeSurfaceId) {
      return null;
    }
    return doc.surfaces.find((s) => s.id === doc.activeSurfaceId) ?? null;
  },

  getActiveFootprint(doc: PaintDocumentState): PlaneFootprint | null {
    const active = PaintDocument.getActiveSurface(doc);
    return active ? getEffectiveFootprint(active) : null;
  },

  getEffectiveBrushSettingsForSurface(
    doc: PaintDocumentState,
    surface: PlanarDecalSurface,
  ): BrushSettings {
    return {
      diameter: surface.referenceBrushDiameter * doc.brushConfig.sizeMultiplier,
      color: doc.brushConfig.color,
      opacity: doc.brushConfig.opacity,
      hardness: doc.brushConfig.hardness,
    };
  },

  getEffectiveBrushSettings(doc: PaintDocumentState): BrushSettings | null {
    const active = PaintDocument.getActiveSurface(doc);
    if (!active) {
      return null;
    }
    return PaintDocument.getEffectiveBrushSettingsForSurface(doc, active);
  },

  getStrokesForSurface(doc: PaintDocumentState, surfaceId: string): BrushStroke[] {
    return doc.strokes.filter((stroke) => stroke.surfaceId === surfaceId);
  },
};
