import {
  IDENTITY_TRANSFORM,
  ZOOM_STEP_FACTOR,
  computeActualSizeTransform,
  computeFitTransform,
  getEffectiveZoomBounds,
  reconcileTransformOnResize,
  zoomAtPoint,
} from '../editor/viewportTransforms';
import type {
  EditorTool,
  GeometryData,
  GeometryState,
  LoadedImage,
  Point2D,
  Size2D,
  ViewMode,
  ViewportTransform,
  ZoomMode,
} from '../types/editor';

export interface EditorState {
  image: LoadedImage | null;
  isLoading: boolean;
  error: string | null;
  transform: ViewportTransform;
  zoomMode: ZoomMode;
  viewportSize: Size2D;
  activeTool: EditorTool;
  viewMode: ViewMode;
  geometry: GeometryState;
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
  | { type: 'GEOMETRY_CANCEL' };

const IDLE_GEOMETRY: GeometryState = {
  status: 'idle',
  data: null,
  error: null,
};

export const initialEditorState: EditorState = {
  image: null,
  isLoading: false,
  error: null,
  transform: IDENTITY_TRANSFORM,
  zoomMode: 'fit',
  viewportSize: { width: 0, height: 0 },
  activeTool: 'pointer',
  viewMode: 'rgb',
  geometry: IDLE_GEOMETRY,
};

function getImageSize(image: LoadedImage): Size2D {
  return {
    width: image.metadata.width,
    height: image.metadata.height,
  };
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
        viewMode: 'rgb',
        geometry: IDLE_GEOMETRY,
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

    case 'SET_ACTIVE_TOOL':
      if (state.activeTool === action.tool) {
        return state;
      }
      return {
        ...state,
        activeTool: action.tool,
      };

    case 'SET_VIEW_MODE': {
      if (action.mode === 'rgb') {
        return state.viewMode === 'rgb' ? state : { ...state, viewMode: 'rgb' };
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

    case 'GEOMETRY_START':
      if (!state.image) {
        return state;
      }
      return {
        ...state,
        viewMode: 'rgb',
        geometry: {
          status: 'loading',
          data: null,
          error: null,
        },
      };

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

    case 'GEOMETRY_CANCEL':
      return {
        ...state,
        viewMode: 'rgb',
        geometry: IDLE_GEOMETRY,
      };
  }
}
