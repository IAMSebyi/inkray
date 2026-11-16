import {
  IDENTITY_TRANSFORM,
  ZOOM_STEP_FACTOR,
  computeActualSizeTransform,
  computeFitTransform,
  getEffectiveZoomBounds,
  reconcileTransformOnResize,
  zoomAtPoint,
} from '../editor/viewportTransforms';
import type { SurfaceSelectionResult } from '../geometry/planeFrame';
import type { BrushStroke } from '../painting/brushEngine';
import {
  type BrushUiConfig,
  PaintDocument,
  type PaintDocumentState,
} from '../painting/paintDocument';
import type {
  AppearanceData,
  AppearanceState,
  EditorTool,
  GeometryData,
  GeometryState,
  LoadedImage,
  PaintMode,
  Point2D,
  Size2D,
  ViewMode,
  ViewportTransform,
  ZoomMode,
} from '../types/editor';

export interface EditorState {
  image: LoadedImage | null;
  isLoading: boolean;
  isExporting: boolean;
  error: string | null;
  transform: ViewportTransform;
  zoomMode: ZoomMode;
  viewportSize: Size2D;
  activeTool: EditorTool;
  viewMode: ViewMode;
  geometry: GeometryState;
  appearance: AppearanceState;
  paint: PaintDocumentState;
}

export type EditorAction =
  | { type: 'IMPORT_START' }
  | { type: 'IMPORT_SUCCESS'; image: LoadedImage }
  | { type: 'IMPORT_ERROR'; message: string }
  | { type: 'CLEAR_ERROR' }
  | { type: 'SET_VIEWPORT_SIZE'; size: Size2D }
  | { type: 'SET_TRANSFORM'; transform: ViewportTransform; zoomMode?: ZoomMode }
  | { type: 'FIT_TO_SCREEN' }
  | { type: 'ZOOM_ACTUAL_SIZE' }
  | { type: 'ZOOM_STEP'; direction: 'in' | 'out'; anchor?: Point2D }
  | { type: 'SET_ACTIVE_TOOL'; tool: EditorTool }
  | { type: 'SET_VIEW_MODE'; mode: ViewMode }
  | { type: 'GEOMETRY_START' }
  | { type: 'GEOMETRY_SUCCESS'; data: GeometryData }
  | { type: 'GEOMETRY_ERROR'; imageId: string; message: string }
  | { type: 'GEOMETRY_CANCEL' }
  | { type: 'APPEARANCE_START' }
  | { type: 'APPEARANCE_SUCCESS'; data: AppearanceData }
  | { type: 'APPEARANCE_ERROR'; imageId: string; message: string }
  | { type: 'APPEARANCE_CANCEL' }
  | { type: 'PAINT_SELECT_SURFACE_START' }
  | { type: 'PAINT_SELECT_SURFACE'; selection: SurfaceSelectionResult }
  | { type: 'PAINT_SELECT_SURFACE_ERROR'; reason: string }
  | { type: 'PAINT_SET_ACTIVE_SURFACE'; surfaceId: string }
  | { type: 'PAINT_REMOVE_SURFACE'; surfaceId: string }
  | { type: 'PAINT_REORDER_SURFACES'; fromIndex: number; toIndex: number }
  | { type: 'PAINT_SET_FOOTPRINT_EXTENT'; surfaceId: string; extentScale: number }
  | { type: 'PAINT_SET_BRUSH_CONFIG'; patch: Partial<BrushUiConfig> }
  | { type: 'PAINT_SET_MODE'; mode: PaintMode }
  | { type: 'PAINT_SET_TEXTURE_PRESERVATION'; value: number }
  | {
      type: 'PAINT_SET_OVERLAY_OPTIONS';
      patch: {
        showSupportOverlay?: boolean;
        showFootprintPreview?: boolean;
        cleanPreview?: boolean;
      };
    }
  | { type: 'PAINT_COMMIT_STROKE'; stroke: BrushStroke }
  | { type: 'PAINT_COMMIT_STROKES'; strokes: BrushStroke[] }
  | { type: 'PAINT_UNDO' }
  | { type: 'PAINT_REDO' }
  | { type: 'PAINT_EXPORT_START' }
  | { type: 'PAINT_EXPORT_DONE'; error?: string };

const IDLE_GEOMETRY: GeometryState = {
  status: 'idle',
  data: null,
  error: null,
};

const IDLE_APPEARANCE: AppearanceState = {
  status: 'idle',
  data: null,
  error: null,
};

export const initialEditorState: EditorState = {
  image: null,
  isLoading: false,
  isExporting: false,
  error: null,
  transform: IDENTITY_TRANSFORM,
  zoomMode: 'fit',
  viewportSize: { width: 0, height: 0 },
  activeTool: 'pointer',
  viewMode: 'rgb',
  geometry: IDLE_GEOMETRY,
  appearance: IDLE_APPEARANCE,
  paint: PaintDocument.createInitial(),
};

function getImageSize(image: LoadedImage): Size2D {
  return {
    width: image.metadata.width,
    height: image.metadata.height,
  };
}

function resetPaintToolOnReset(tool: EditorTool): EditorTool {
  if (tool === 'select-surface' || tool === 'brush' || tool === 'eraser') {
    return 'pointer';
  }
  return tool;
}

export function editorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case 'IMPORT_START':
      return {
        ...state,
        isLoading: true,
        error: null,
      };

    case 'IMPORT_SUCCESS': {
      const imageSize = getImageSize(action.image);
      const transform = computeFitTransform(state.viewportSize, imageSize);
      return {
        ...state,
        image: action.image,
        isLoading: false,
        error: null,
        zoomMode: 'fit',
        transform,
        activeTool: resetPaintToolOnReset(state.activeTool),
        viewMode: 'rgb',
        geometry: IDLE_GEOMETRY,
        appearance: IDLE_APPEARANCE,
        paint: PaintDocument.createInitial(
          state.paint.brushConfig,
          state.paint.paintMode,
          state.paint.texturePreservation,
        ),
      };
    }

    case 'IMPORT_ERROR':
      return {
        ...state,
        isLoading: false,
        error: action.message,
      };

    case 'CLEAR_ERROR':
      if (state.error === null) {
        return state;
      }
      return {
        ...state,
        error: null,
      };

    case 'SET_VIEWPORT_SIZE': {
      const { width, height } = action.size;
      if (width === state.viewportSize.width && height === state.viewportSize.height) {
        return state;
      }

      if (!state.image) {
        return {
          ...state,
          viewportSize: action.size,
        };
      }

      const nextTransform = reconcileTransformOnResize(
        state.transform,
        state.viewportSize,
        action.size,
        getImageSize(state.image),
        state.zoomMode,
      );

      return {
        ...state,
        viewportSize: action.size,
        transform: nextTransform,
      };
    }

    case 'SET_TRANSFORM':
      return {
        ...state,
        transform: action.transform,
        zoomMode: action.zoomMode ?? 'custom',
      };

    case 'FIT_TO_SCREEN': {
      if (!state.image) {
        return state;
      }
      return {
        ...state,
        zoomMode: 'fit',
        transform: computeFitTransform(state.viewportSize, getImageSize(state.image)),
      };
    }

    case 'ZOOM_ACTUAL_SIZE': {
      if (!state.image) {
        return state;
      }
      return {
        ...state,
        zoomMode: 'custom',
        transform: computeActualSizeTransform(
          state.viewportSize,
          getImageSize(state.image),
          state.transform,
        ),
      };
    }

    case 'ZOOM_STEP': {
      if (!state.image) {
        return state;
      }
      const imageSize = getImageSize(state.image);
      const bounds = getEffectiveZoomBounds(state.viewportSize, imageSize);
      const targetZoom =
        action.direction === 'in'
          ? state.transform.zoom * ZOOM_STEP_FACTOR
          : state.transform.zoom / ZOOM_STEP_FACTOR;
      const anchor = action.anchor ?? {
        x: state.viewportSize.width / 2,
        y: state.viewportSize.height / 2,
      };
      return {
        ...state,
        zoomMode: 'custom',
        transform: zoomAtPoint(state.transform, anchor, targetZoom, bounds),
      };
    }

    case 'SET_ACTIVE_TOOL': {
      if (state.activeTool === action.tool) {
        return state;
      }
      if (action.tool === 'select-surface' && state.geometry.status !== 'ready') {
        return state;
      }
      if (
        (action.tool === 'brush' || action.tool === 'eraser') &&
        (state.geometry.status !== 'ready' || state.paint.surfaces.length === 0)
      ) {
        return state;
      }
      let nextPaint = state.paint;
      if (
        (action.tool === 'brush' || action.tool === 'eraser') &&
        nextPaint.activeSurfaceId === null &&
        nextPaint.surfaces[0]
      ) {
        nextPaint = PaintDocument.setActiveSurface(nextPaint, nextPaint.surfaces[0].id);
      }
      const nextViewMode =
        action.tool === 'brush' ||
        action.tool === 'eraser' ||
        (action.tool === 'select-surface' &&
          (state.viewMode === 'pointcloud' || state.viewMode === 'mesh'))
          ? 'rgb'
          : state.viewMode;
      return {
        ...state,
        activeTool: action.tool,
        viewMode: nextViewMode,
        paint: nextPaint,
      };
    }

    case 'SET_VIEW_MODE': {
      if (action.mode === 'rgb') {
        return state.viewMode === 'rgb' ? state : { ...state, viewMode: 'rgb' };
      }
      if (action.mode === 'albedo' || action.mode === 'shading' || action.mode === 'residual') {
        const readyAppearance = state.appearance.status === 'ready' ? state.appearance.data : null;
        if (!readyAppearance) {
          return state;
        }
        return {
          ...state,
          viewMode: action.mode,
        };
      }
      const readyData = state.geometry.status === 'ready' ? state.geometry.data : null;
      if (!readyData) {
        return state;
      }
      if (action.mode === 'normals' && !readyData.previews.normals) {
        return state;
      }
      return {
        ...state,
        viewMode: action.mode,
      };
    }

    case 'GEOMETRY_START': {
      if (!state.image) {
        return state;
      }
      return {
        ...state,
        activeTool: resetPaintToolOnReset(state.activeTool),
        viewMode: 'rgb',
        geometry: {
          status: 'loading',
          data: null,
          error: null,
        },
        paint: PaintDocument.createInitial(
          state.paint.brushConfig,
          state.paint.paintMode,
          state.paint.texturePreservation,
        ),
      };
    }

    case 'GEOMETRY_SUCCESS':
      // Ignore stale responses if the user replaced the image before completion.
      if (!state.image || state.image.id !== action.data.imageId) {
        return state;
      }
      return {
        ...state,
        geometry: {
          status: 'ready',
          data: action.data,
          error: null,
        },
      };

    case 'GEOMETRY_ERROR':
      if (!state.image || state.image.id !== action.imageId) {
        return state;
      }
      return {
        ...state,
        geometry: {
          status: 'error',
          data: null,
          error: action.message,
        },
      };

    case 'GEOMETRY_CANCEL': {
      return {
        ...state,
        activeTool: resetPaintToolOnReset(state.activeTool),
        viewMode: 'rgb',
        geometry: IDLE_GEOMETRY,
        paint: PaintDocument.createInitial(
          state.paint.brushConfig,
          state.paint.paintMode,
          state.paint.texturePreservation,
        ),
      };
    }

    case 'APPEARANCE_START': {
      if (!state.image) {
        return state;
      }
      const resetView =
        state.viewMode === 'albedo' ||
        state.viewMode === 'shading' ||
        state.viewMode === 'residual';
      return {
        ...state,
        viewMode: resetView ? 'rgb' : state.viewMode,
        appearance: {
          status: 'loading',
          data: null,
          error: null,
        },
      };
    }

    case 'APPEARANCE_SUCCESS':
      if (!state.image || state.image.id !== action.data.imageId) {
        return state;
      }
      return {
        ...state,
        appearance: {
          status: 'ready',
          data: action.data,
          error: null,
        },
      };

    case 'APPEARANCE_ERROR':
      if (!state.image || state.image.id !== action.imageId) {
        return state;
      }
      return {
        ...state,
        appearance: {
          status: 'error',
          data: null,
          error: action.message,
        },
      };

    case 'APPEARANCE_CANCEL': {
      const resetView =
        state.viewMode === 'albedo' ||
        state.viewMode === 'shading' ||
        state.viewMode === 'residual';
      return {
        ...state,
        viewMode: resetView ? 'rgb' : state.viewMode,
        appearance: IDLE_APPEARANCE,
      };
    }

    case 'PAINT_SELECT_SURFACE_START':
      if (state.geometry.status !== 'ready') {
        return state;
      }
      return {
        ...state,
        paint: PaintDocument.setSelectingSurface(state.paint, true),
      };

    case 'PAINT_SELECT_SURFACE':
      if (state.geometry.status !== 'ready') {
        return state;
      }
      return {
        ...state,
        viewMode:
          state.viewMode === 'pointcloud' || state.viewMode === 'mesh' ? 'rgb' : state.viewMode,
        paint: PaintDocument.selectSurface(state.paint, action.selection),
      };

    case 'PAINT_SELECT_SURFACE_ERROR':
      return {
        ...state,
        paint: PaintDocument.setSelectionFailed(state.paint, action.reason),
      };

    case 'PAINT_SET_ACTIVE_SURFACE':
      return {
        ...state,
        paint: PaintDocument.setActiveSurface(state.paint, action.surfaceId),
      };

    case 'PAINT_REMOVE_SURFACE': {
      const nextPaint = PaintDocument.removeSurface(state.paint, action.surfaceId);
      const nextTool =
        nextPaint.surfaces.length === 0 &&
        (state.activeTool === 'brush' || state.activeTool === 'eraser')
          ? 'select-surface'
          : state.activeTool;
      return {
        ...state,
        activeTool: nextTool,
        paint: nextPaint,
      };
    }

    case 'PAINT_REORDER_SURFACES':
      return {
        ...state,
        paint: PaintDocument.reorderSurfaces(state.paint, action.fromIndex, action.toIndex),
      };

    case 'PAINT_SET_FOOTPRINT_EXTENT':
      return {
        ...state,
        paint: PaintDocument.setSurfaceExtent(state.paint, action.surfaceId, action.extentScale),
      };

    case 'PAINT_SET_BRUSH_CONFIG':
      return {
        ...state,
        paint: PaintDocument.setBrushConfig(state.paint, action.patch),
      };

    case 'PAINT_SET_MODE':
      return {
        ...state,
        paint: PaintDocument.setPaintMode(state.paint, action.mode),
      };

    case 'PAINT_SET_TEXTURE_PRESERVATION':
      return {
        ...state,
        paint: PaintDocument.setTexturePreservation(state.paint, action.value),
      };

    case 'PAINT_SET_OVERLAY_OPTIONS':
      return {
        ...state,
        paint: PaintDocument.setOverlayOptions(state.paint, action.patch),
      };

    case 'PAINT_COMMIT_STROKE':
      return {
        ...state,
        paint: PaintDocument.commitStroke(state.paint, action.stroke),
      };

    case 'PAINT_COMMIT_STROKES':
      return {
        ...state,
        paint: PaintDocument.commitStrokes(state.paint, action.strokes),
      };

    case 'PAINT_UNDO': {
      const nextPaint = PaintDocument.undo(state.paint);
      const nextTool =
        nextPaint.surfaces.length === 0 &&
        (state.activeTool === 'brush' || state.activeTool === 'eraser')
          ? 'select-surface'
          : state.activeTool;
      return {
        ...state,
        activeTool: nextTool,
        paint: nextPaint,
      };
    }

    case 'PAINT_REDO':
      return {
        ...state,
        paint: PaintDocument.redo(state.paint),
      };

    case 'PAINT_EXPORT_START':
      return {
        ...state,
        isExporting: true,
        error: null,
      };

    case 'PAINT_EXPORT_DONE':
      return {
        ...state,
        isExporting: false,
        error: action.error ?? state.error,
      };
  }
}
