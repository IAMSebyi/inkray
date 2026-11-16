import { createContext } from 'react';
import type { BrushEngine, BrushStroke } from '../painting/brushEngine';
import type { BrushUiConfig } from '../painting/paintDocument';
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
  brushEngineRef: React.RefObject<BrushEngine>;
  paintRendererRef: React.RefObject<PlanarPaintRenderer | null>;
  openFilePicker: () => void;
  importFile: (file: File) => Promise<void>;
  clearError: () => void;
  setViewportSize: (size: Size2D) => void;
  setTransform: (transform: ViewportTransform, zoomMode?: ZoomMode) => void;
  fitToScreen: () => void;
  zoomToActualSize: () => void;
  zoomIn: () => void;
  zoomOut: () => void;
  setActiveTool: (tool: EditorTool) => void;
  setViewMode: (mode: ViewMode) => void;
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
  undoPaint: () => void;
  redoPaint: () => void;
  exportPng: () => Promise<void>;
}

export const EditorContext = createContext<EditorContextValue | null>(null);
