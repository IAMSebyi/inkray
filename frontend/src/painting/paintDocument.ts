import type { PaintMode } from '../types/editor';
import type { AutoAnchorMode } from '../geometry/autoSurfaceAnchor';
import type { PlaneFootprint, SurfaceSelectionResult } from '../geometry/planeFrame';
import { clamp } from '../editor/viewportTransforms';
import type { Artwork } from './artwork';
import type { ArtworkPlacement } from './artworkPlacement';
import type { BrushSettings, BrushStroke } from './brushEngine';
import type { ImageSpaceStroke } from './imageSpacePaint';
import { type PlanarDecalSurface, getEffectiveFootprint } from './planarDecal';

export interface BrushUiConfig {
  sizeMultiplier: number;
  pixelSize: number;
  color: string;
  opacity: number;
  hardness: number;
}

export interface PaintHistorySnapshot {
  surfaces: PlanarDecalSurface[];
  activeSurfaceId: string | null;
  strokes: BrushStroke[];
  imageStrokes: ImageSpaceStroke[];
  lastAnchorMode: AutoAnchorMode | null;
  artworkPlacements: ArtworkPlacement[];
  activePlacementId: string | null;
}

export interface PaintDocumentState {
  surfaces: PlanarDecalSurface[];
  activeSurfaceId: string | null;
  strokes: BrushStroke[];
  imageStrokes: ImageSpaceStroke[];
  lastAnchorMode: AutoAnchorMode | null;
  artworks: Artwork[];
  activeArtworkId: string | null;
  artworkPlacements: ArtworkPlacement[];
  activePlacementId: string | null;
  nextArtworkNumber: number;
  nextPlacementNumber: number;
  undoStack: PaintHistorySnapshot[];
  redoStack: PaintHistorySnapshot[];
  nextSurfaceNumber: number;
  isSelectingSurface: boolean;
  selectionError: string | null;
  showSupportOverlay: boolean;
  showFootprintPreview: boolean;
  cleanPreview: boolean;
  paintMode: PaintMode;
  texturePreservation: number;
  brushConfig: BrushUiConfig;
}

export const DEFAULT_BRUSH_PIXEL_SIZE = 36;

export const DEFAULT_BRUSH_CONFIG: BrushUiConfig = {
  sizeMultiplier: 1.0,
  pixelSize: DEFAULT_BRUSH_PIXEL_SIZE,
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
    imageStrokes: doc.imageStrokes,
    lastAnchorMode: doc.lastAnchorMode,
    artworkPlacements: doc.artworkPlacements,
    activePlacementId: doc.activePlacementId,
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
    preservedArtworks?: Artwork[],
    preservedActiveArtworkId?: string | null,
    preservedNextArtworkNumber?: number,
  ): PaintDocumentState {
    return {
      surfaces: [],
      activeSurfaceId: null,
      strokes: [],
      imageStrokes: [],
      lastAnchorMode: null,
      artworks: preservedArtworks ? [...preservedArtworks] : [],
      activeArtworkId: preservedActiveArtworkId ?? null,
      artworkPlacements: [],
      activePlacementId: null,
      nextArtworkNumber: preservedNextArtworkNumber ?? 1,
      nextPlacementNumber: 1,
      undoStack: [],
      redoStack: [],
      nextSurfaceNumber: 1,
      isSelectingSurface: false,
      selectionError: null,
      showSupportOverlay: false,
      showFootprintPreview: false,
      cleanPreview: false,
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
    let pixelSize = doc.brushConfig.pixelSize;
    let sizeMultiplier = doc.brushConfig.sizeMultiplier;

    if (patch.pixelSize !== undefined) {
      pixelSize = clamp(Math.round(patch.pixelSize), 4, 240);
      sizeMultiplier = clamp(pixelSize / DEFAULT_BRUSH_PIXEL_SIZE, 0.1, 7.0);
    } else if (patch.sizeMultiplier !== undefined) {
      sizeMultiplier = clamp(patch.sizeMultiplier, 0.2, 5.0);
      pixelSize = clamp(Math.round(sizeMultiplier * DEFAULT_BRUSH_PIXEL_SIZE), 4, 240);
    }

    const nextConfig: BrushUiConfig = {
      sizeMultiplier,
      pixelSize,
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
    patch: { showSupportOverlay?: boolean; showFootprintPreview?: boolean; cleanPreview?: boolean },
  ): PaintDocumentState {
    return {
      ...doc,
      showSupportOverlay: patch.showSupportOverlay ?? doc.showSupportOverlay,
      showFootprintPreview: patch.showFootprintPreview ?? doc.showFootprintPreview,
      cleanPreview: patch.cleanPreview ?? doc.cleanPreview,
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

  commitAutoStroke(
    doc: PaintDocumentState,
    payload: {
      anchorMode: AutoAnchorMode;
      newPlanarSurface?: PlanarDecalSurface | null;
      planarStroke?: BrushStroke | null;
      imageStroke?: ImageSpaceStroke | null;
    },
  ): PaintDocumentState {
    const { anchorMode, newPlanarSurface, planarStroke, imageStroke } = payload;
    const hasNewSurf = Boolean(newPlanarSurface);
    const nextSurfaces = newPlanarSurface ? [...doc.surfaces, newPlanarSurface] : doc.surfaces;
    const nextActive = newPlanarSurface
      ? newPlanarSurface.id
      : planarStroke
        ? planarStroke.surfaceId
        : doc.activeSurfaceId;
    const surfaceIds = new Set(nextSurfaces.map((s) => s.id));
    const validPlanar =
      planarStroke && planarStroke.points.length > 0 && surfaceIds.has(planarStroke.surfaceId)
        ? planarStroke
        : null;
    const validImage = imageStroke ? imageStroke : null;

    if (!validPlanar && !validImage) {
      return doc;
    }

    return {
      ...doc,
      surfaces: nextSurfaces,
      activeSurfaceId: nextActive,
      nextSurfaceNumber: hasNewSurf ? doc.nextSurfaceNumber + 1 : doc.nextSurfaceNumber,
      strokes: validPlanar ? [...doc.strokes, validPlanar] : doc.strokes,
      imageStrokes: validImage ? [...doc.imageStrokes, validImage] : doc.imageStrokes,
      lastAnchorMode: anchorMode,
      undoStack: pushUndoSnapshot(doc),
      redoStack: [],
    };
  },

  addArtwork(doc: PaintDocumentState, artwork: Artwork): PaintDocumentState {
    return {
      ...doc,
      artworks: [...doc.artworks, artwork],
      activeArtworkId: artwork.id,
      nextArtworkNumber: doc.nextArtworkNumber + 1,
    };
  },

  updateArtwork(doc: PaintDocumentState, artwork: Artwork): PaintDocumentState {
    if (!doc.artworks.some((item) => item.id === artwork.id)) {
      return doc;
    }
    return {
      ...doc,
      artworks: doc.artworks.map((item) => (item.id === artwork.id ? artwork : item)),
    };
  },

  setActiveArtwork(doc: PaintDocumentState, artworkId: string | null): PaintDocumentState {
    if (artworkId !== null && !doc.artworks.some((a) => a.id === artworkId)) {
      return doc;
    }
    return doc.activeArtworkId === artworkId ? doc : { ...doc, activeArtworkId: artworkId };
  },

  removeArtwork(doc: PaintDocumentState, artworkId: string): PaintDocumentState {
    if (!doc.artworks.some((a) => a.id === artworkId)) {
      return doc;
    }
    const nextArtworks = doc.artworks.filter((a) => a.id !== artworkId);
    const nextActiveArtwork =
      doc.activeArtworkId === artworkId ? (nextArtworks[0]?.id ?? null) : doc.activeArtworkId;
    const hadPlacements = doc.artworkPlacements.some((p) => p.artworkId === artworkId);
    if (!hadPlacements) {
      return {
        ...doc,
        artworks: nextArtworks,
        activeArtworkId: nextActiveArtwork,
      };
    }
    const nextPlacements = doc.artworkPlacements.filter((p) => p.artworkId !== artworkId);
    const nextActivePlacement =
      doc.activePlacementId && nextPlacements.some((p) => p.id === doc.activePlacementId)
        ? doc.activePlacementId
        : (nextPlacements[nextPlacements.length - 1]?.id ?? null);
    return {
      ...doc,
      artworks: nextArtworks,
      activeArtworkId: nextActiveArtwork,
      artworkPlacements: nextPlacements,
      activePlacementId: nextActivePlacement,
      undoStack: pushUndoSnapshot(doc),
      redoStack: [],
    };
  },

  addPlacement(doc: PaintDocumentState, placement: ArtworkPlacement): PaintDocumentState {
    return {
      ...doc,
      artworkPlacements: [...doc.artworkPlacements, placement],
      activePlacementId: placement.id,
      nextPlacementNumber: doc.nextPlacementNumber + 1,
      undoStack: pushUndoSnapshot(doc),
      redoStack: [],
    };
  },

  updatePlacement(
    doc: PaintDocumentState,
    placement: ArtworkPlacement,
    recordHistory = true,
  ): PaintDocumentState {
    if (!doc.artworkPlacements.some((item) => item.id === placement.id)) {
      return doc;
    }
    return {
      ...doc,
      artworkPlacements: doc.artworkPlacements.map((item) =>
        item.id === placement.id ? placement : item,
      ),
      undoStack: recordHistory ? pushUndoSnapshot(doc) : doc.undoStack,
      redoStack: recordHistory ? [] : doc.redoStack,
    };
  },

  setActivePlacement(doc: PaintDocumentState, placementId: string | null): PaintDocumentState {
    if (placementId !== null && !doc.artworkPlacements.some((p) => p.id === placementId)) {
      return doc;
    }
    return doc.activePlacementId === placementId ? doc : { ...doc, activePlacementId: placementId };
  },

  removePlacement(doc: PaintDocumentState, placementId: string): PaintDocumentState {
    if (!doc.artworkPlacements.some((p) => p.id === placementId)) {
      return doc;
    }
    const nextPlacements = doc.artworkPlacements.filter((p) => p.id !== placementId);
    const nextActive =
      doc.activePlacementId === placementId
        ? (nextPlacements[nextPlacements.length - 1]?.id ?? null)
        : doc.activePlacementId;
    return {
      ...doc,
      artworkPlacements: nextPlacements,
      activePlacementId: nextActive,
      undoStack: pushUndoSnapshot(doc),
      redoStack: [],
    };
  },

  undo(doc: PaintDocumentState): PaintDocumentState {
    if (doc.undoStack.length === 0) {
      return doc;
    }
    const prev = doc.undoStack[doc.undoStack.length - 1];
    if (!prev) {
      return doc;
    }
    const liveArtworkIds = new Set(doc.artworks.map((a) => a.id));
    const validPlacements = prev.artworkPlacements.filter((p) => liveArtworkIds.has(p.artworkId));
    const validActivePlacement =
      prev.activePlacementId && validPlacements.some((p) => p.id === prev.activePlacementId)
        ? prev.activePlacementId
        : (validPlacements[validPlacements.length - 1]?.id ?? null);
    return {
      ...doc,
      surfaces: prev.surfaces,
      activeSurfaceId: prev.activeSurfaceId,
      strokes: prev.strokes,
      imageStrokes: prev.imageStrokes,
      lastAnchorMode: prev.lastAnchorMode,
      artworkPlacements: validPlacements,
      activePlacementId: validActivePlacement,
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
    const liveArtworkIds = new Set(doc.artworks.map((a) => a.id));
    const validPlacements = next.artworkPlacements.filter((p) => liveArtworkIds.has(p.artworkId));
    const validActivePlacement =
      next.activePlacementId && validPlacements.some((p) => p.id === next.activePlacementId)
        ? next.activePlacementId
        : (validPlacements[validPlacements.length - 1]?.id ?? null);
    return {
      ...doc,
      surfaces: next.surfaces,
      activeSurfaceId: next.activeSurfaceId,
      strokes: next.strokes,
      imageStrokes: next.imageStrokes,
      lastAnchorMode: next.lastAnchorMode,
      artworkPlacements: validPlacements,
      activePlacementId: validActivePlacement,
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

  getArtworkById(doc: PaintDocumentState, artworkId: string | null): Artwork | null {
    if (!artworkId) {
      return null;
    }
    return doc.artworks.find((a) => a.id === artworkId) ?? null;
  },

  getActiveArtwork(doc: PaintDocumentState): Artwork | null {
    return PaintDocument.getArtworkById(doc, doc.activeArtworkId);
  },

  getActivePlacement(doc: PaintDocumentState): ArtworkPlacement | null {
    if (!doc.activePlacementId) {
      return null;
    }
    return doc.artworkPlacements.find((p) => p.id === doc.activePlacementId) ?? null;
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
