import { createContext } from 'react';
import type { AutoAnchorMode } from '../geometry/autoSurfaceAnchor';
import type { ArtworkPlacement } from '../painting/artworkPlacement';
import type { BrushEngine, BrushStroke } from '../painting/brushEngine';
import type { ImageSpaceBrushEngine, ImageSpaceStroke } from '../painting/imageSpacePaint';
import type { BrushUiConfig } from '../painting/paintDocument';
import type { PlanarDecalSurface } from '../painting/planarDecal';
import type { PlanarPaintRenderer } from '../painting/planarPaintRenderer';
import type {
  EditorTool,
  PaintMode,
  Point2D,
  Size2D,
  ViewMode,
  ViewportTransform,
  ZoomMode,
} from '../types/editor';
import type { CursorStore } from './cursorStore';
import type { EditorState } from './editorReducer';

export interface EditorContextValue {
  state: EditorState;
  cursorStore: CursorStore;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  artworkFileInputRef: React.RefObject<HTMLInputElement | null>;
  brushEngineRef: React.RefObject<BrushEngine>;
  imageBrushEngineRef: React.RefObject<ImageSpaceBrushEngine>;
  paintRendererRef: React.RefObject<PlanarPaintRenderer | null>;
  openFilePicker: () => void;
  openArtworkFilePicker: () => void;
  importFile: (file: File) => Promise<void>;
  importArtworkFile: (file: File) => Promise<void>;
  clearError: () => void;
  setViewportSize: (size: Size2D) => void;
  setTransform: (transform: ViewportTransform, zoomMode?: ZoomMode) => void;
  fitToScreen: () => void;
  zoomToActualSize: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  setActiveTool: (tool: EditorTool) => void;
  setViewMode: (mode: ViewMode) => void;
  openArtworkEditor: (artworkId?: string) => void;
  createBlankArtwork: (width?: number, height?: number) => void;
  returnToPhoto: () => void;
  resizeActiveArtwork: (width: number, height: number) => void;
  commitArtworkStroke: (stroke: BrushStroke) => void;
  clearActiveArtwork: () => void;
  setActiveArtwork: (artworkId: string | null) => void;
  removeArtwork: (artworkId: string) => void;
  placeArtworkAtImagePoint: (artworkId: string, imagePoint?: Point2D) => void;
  updatePlacement: (placement: ArtworkPlacement, recordHistory?: boolean) => void;
  setActivePlacement: (placementId: string | null) => void;
  togglePlacementVisibility: (placementId: string) => void;
  removePlacement: (placementId: string) => void;
  reanchorPlacement: (placementId: string, imagePoint?: Point2D) => void;
  createBrushObject: (label?: string) => void;
  setActiveBrushObject: (brushObjectId: string | null) => void;
  toggleBrushObjectVisibility: (brushObjectId: string) => void;
  removeBrushObject: (brushObjectId: string) => void;
  reorderSceneObject: (fromIndex: number, toIndex: number) => void;
  generateGeometry: () => Promise<void>;
  cancelGeometry: () => void;
  generateAppearance: () => Promise<void>;
  cancelAppearance: () => void;
  selectSurfaceAtImagePoint: (imagePoint: Point2D) => Promise<void>;
  setActiveSurface: (surfaceId: string) => void;
  removeSurface: (surfaceId: string) => void;
  reorderSurfaces: (fromIndex: number, toIndex: number) => void;
  setFootprintExtent: (surfaceId: string, extentScale: number) => void;
  setBrushConfig: (patch: Partial<BrushUiConfig>) => void;
  setPaintMode: (mode: PaintMode) => void;
  setTexturePreservation: (value: number) => void;
  setPaintOverlayOptions: (patch: {
    showSupportOverlay?: boolean;
    showFootprintPreview?: boolean;
    cleanPreview?: boolean;
  }) => void;
  commitBrushStroke: (stroke: BrushStroke) => void;
  commitBrushStrokeBatch: (strokes: BrushStroke[]) => void;
  commitAutoStroke: (payload: {
    anchorMode: AutoAnchorMode;
    newPlanarSurface?: PlanarDecalSurface | null;
    planarStroke?: BrushStroke | null;
    imageStroke?: ImageSpaceStroke | null;
  }) => void;
  undoPaint: () => void;
  redoPaint: () => void;
  exportPng: () => Promise<void>;
}

export const EditorContext = createContext<EditorContextValue | null>(null);
