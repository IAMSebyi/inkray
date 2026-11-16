import { useCallback, useEffect, useMemo, useReducer, useRef, type ReactNode } from 'react';
import { fetchGeometryForImage, selectSurfaceForGeometry } from '../api/geometryClient';
import { ImageLoadError, loadImageFromFile } from '../editor/imageLoader';
import { BrushEngine, type BrushStroke } from '../painting/brushEngine';
import type { BrushUiConfig } from '../painting/paintDocument';
import type { PlanarPaintRenderer } from '../painting/planarPaintRenderer';
import type {
  EditorTool,
  LoadedImage,
  Point2D,
  Size2D,
  ViewMode,
  ViewportTransform,
  ZoomMode,
} from '../types/editor';
import { createCursorStore, type CursorStore } from './cursorStore';
import { EditorContext, type EditorContextValue } from './editorContextValue';
import { editorReducer, initialEditorState } from './editorReducer';

interface SaveFileWritable {
  write: (data: Blob) => Promise<void>;
  close: () => Promise<void>;
}

interface SaveFileHandle {
  createWritable: () => Promise<SaveFileWritable>;
}

interface SaveFilePickerWindow extends Window {
  showSaveFilePicker?: (options?: {
    suggestedName?: string;
    types?: Array<{ description?: string; accept: Record<string, string[]> }>;
  }) => Promise<SaveFileHandle>;
}

function buildExportFilename(sourceName: string): string {
  const stem = sourceName.replace(/\.[^/.]+$/, '').trim() || 'image';
  return `${stem}-inkray.png`;
}

export function EditorProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(editorReducer, initialEditorState);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const cursorStoreRef = useRef<CursorStore | null>(null);
  if (cursorStoreRef.current === null) {
    cursorStoreRef.current = createCursorStore();
  }
  const cursorStore = cursorStoreRef.current;

  const brushEngineRef = useRef<BrushEngine>(new BrushEngine());
  const paintRendererRef = useRef<PlanarPaintRenderer | null>(null);

  const activeAbortRef = useRef<AbortController | null>(null);
  const geometryAbortRef = useRef<AbortController | null>(null);
  const surfaceSelectAbortRef = useRef<AbortController | null>(null);
  const geometryInFlightRef = useRef(false);
  const activeImageRef = useRef<LoadedImage | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    activeImageRef.current = state.image;
  }, [state.image]);

  useEffect(() => {
    return () => {
      activeAbortRef.current?.abort();
      geometryAbortRef.current?.abort();
      surfaceSelectAbortRef.current?.abort();
      activeImageRef.current?.dispose();
    };
  }, []);

  const openFilePicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const importFile = useCallback(
    async (file: File) => {
      activeAbortRef.current?.abort();
      const abortController = new AbortController();
      activeAbortRef.current = abortController;

      dispatch({ type: 'IMPORT_START' });

      try {
        const nextImage = await loadImageFromFile(file, abortController.signal);
        if (abortController.signal.aborted) {
          nextImage.dispose();
          return;
        }

        geometryAbortRef.current?.abort();
        geometryAbortRef.current = null;
        geometryInFlightRef.current = false;
        surfaceSelectAbortRef.current?.abort();
        surfaceSelectAbortRef.current = null;

        const previousImage = activeImageRef.current;
        activeImageRef.current = nextImage;
        brushEngineRef.current.cancelStroke();
        dispatch({ type: 'IMPORT_SUCCESS', image: nextImage });
        cursorStore.setSample(null);

        if (previousImage && previousImage !== nextImage) {
          previousImage.dispose();
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return;
        }
        const message =
          error instanceof ImageLoadError
            ? error.message
            : error instanceof Error
              ? error.message
              : 'An unexpected error occurred while loading the image.';
        dispatch({ type: 'IMPORT_ERROR', message });
      } finally {
        if (activeAbortRef.current === abortController) {
          activeAbortRef.current = null;
        }
      }
    },
    [cursorStore],
  );

  const generateGeometry = useCallback(async () => {
    const targetImage = activeImageRef.current;
    if (!targetImage || geometryInFlightRef.current) {
      return;
    }

    geometryInFlightRef.current = true;
    surfaceSelectAbortRef.current?.abort();
    surfaceSelectAbortRef.current = null;
    const abortController = new AbortController();
    geometryAbortRef.current = abortController;
    brushEngineRef.current.cancelStroke();
    dispatch({ type: 'GEOMETRY_START' });

    try {
      const data = await fetchGeometryForImage(targetImage, abortController.signal);
      if (abortController.signal.aborted || activeImageRef.current?.id !== targetImage.id) {
        return;
      }
      dispatch({ type: 'GEOMETRY_SUCCESS', data });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      if (activeImageRef.current?.id !== targetImage.id) {
        return;
      }
      const message =
        error instanceof Error ? error.message : 'Failed to generate geometry from backend.';
      dispatch({ type: 'GEOMETRY_ERROR', imageId: targetImage.id, message });
    } finally {
      if (geometryAbortRef.current === abortController) {
        geometryAbortRef.current = null;
        geometryInFlightRef.current = false;
      }
    }
  }, []);

  const cancelGeometry = useCallback(() => {
    geometryAbortRef.current?.abort();
    geometryAbortRef.current = null;
    geometryInFlightRef.current = false;
    surfaceSelectAbortRef.current?.abort();
    surfaceSelectAbortRef.current = null;
    brushEngineRef.current.cancelStroke();
    dispatch({ type: 'GEOMETRY_CANCEL' });
  }, []);

  const selectSurfaceAtImagePoint = useCallback(async (imagePoint: Point2D) => {
    const current = stateRef.current;
    const targetImage = current.image;
    const targetGeometry = current.geometry.data;
    if (!targetImage || !targetGeometry || current.geometry.status !== 'ready') {
      return;
    }

    surfaceSelectAbortRef.current?.abort();
    const abortController = new AbortController();
    surfaceSelectAbortRef.current = abortController;

    brushEngineRef.current.cancelStroke();
    dispatch({ type: 'PAINT_SELECT_SURFACE_START' });

    try {
      const outcome = await selectSurfaceForGeometry(
        targetGeometry.metadata.id,
        imagePoint,
        abortController.signal,
      );
      if (
        abortController.signal.aborted ||
        activeImageRef.current?.id !== targetImage.id ||
        stateRef.current.geometry.data?.metadata.id !== targetGeometry.metadata.id
      ) {
        return;
      }

      if (outcome.ok) {
        dispatch({ type: 'PAINT_SELECT_SURFACE', selection: outcome.selection });
      } else {
        dispatch({ type: 'PAINT_SELECT_SURFACE_ERROR', reason: outcome.reason });
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      if (
        abortController.signal.aborted ||
        activeImageRef.current?.id !== targetImage.id ||
        stateRef.current.geometry.data?.metadata.id !== targetGeometry.metadata.id
      ) {
        return;
      }
      const reason =
        error instanceof Error ? error.message : 'Failed to select planar surface on backend.';
      dispatch({ type: 'PAINT_SELECT_SURFACE_ERROR', reason });
    } finally {
      if (surfaceSelectAbortRef.current === abortController) {
        surfaceSelectAbortRef.current = null;
      }
    }
  }, []);

  const setActiveSurface = useCallback((surfaceId: string) => {
    brushEngineRef.current.cancelStroke();
    dispatch({ type: 'PAINT_SET_ACTIVE_SURFACE', surfaceId });
  }, []);

  const removeSurface = useCallback((surfaceId: string) => {
    brushEngineRef.current.cancelStroke();
    dispatch({ type: 'PAINT_REMOVE_SURFACE', surfaceId });
  }, []);

  const reorderSurfaces = useCallback((fromIndex: number, toIndex: number) => {
    brushEngineRef.current.cancelStroke();
    dispatch({ type: 'PAINT_REORDER_SURFACES', fromIndex, toIndex });
  }, []);

  const setFootprintExtent = useCallback((surfaceId: string, extentScale: number) => {
    dispatch({ type: 'PAINT_SET_FOOTPRINT_EXTENT', surfaceId, extentScale });
  }, []);

  const setBrushConfig = useCallback((patch: Partial<BrushUiConfig>) => {
    dispatch({ type: 'PAINT_SET_BRUSH_CONFIG', patch });
  }, []);

  const setPaintOverlayOptions = useCallback(
    (patch: { showSupportOverlay?: boolean; showFootprintPreview?: boolean }) => {
      dispatch({ type: 'PAINT_SET_OVERLAY_OPTIONS', patch });
    },
    [],
  );

  const commitBrushStroke = useCallback((stroke: BrushStroke) => {
    dispatch({ type: 'PAINT_COMMIT_STROKE', stroke });
  }, []);

  const commitBrushStrokeBatch = useCallback((strokes: BrushStroke[]) => {
    dispatch({ type: 'PAINT_COMMIT_STROKES', strokes });
  }, []);

  const undoPaint = useCallback(() => {
    brushEngineRef.current.cancelStroke();
    dispatch({ type: 'PAINT_UNDO' });
  }, []);

  const redoPaint = useCallback(() => {
    brushEngineRef.current.cancelStroke();
    dispatch({ type: 'PAINT_REDO' });
  }, []);

  const exportPng = useCallback(async () => {
    const current = stateRef.current;
    const image = current.image;
    const geometry = current.geometry.data;
    const renderer = paintRendererRef.current;

    if (!image || !geometry || !renderer) {
      dispatch({
        type: 'PAINT_EXPORT_DONE',
        error: 'Load an image and generate geometry before exporting.',
      });
      return;
    }

    const suggestedName = buildExportFilename(image.metadata.name);
    const picker = (window as SaveFilePickerWindow).showSaveFilePicker;
    let fileHandle: SaveFileHandle | null = null;

    if (picker) {
      try {
        fileHandle = await picker({
          suggestedName,
          types: [{ description: 'PNG Image', accept: { 'image/png': ['.png'] } }],
        });
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return;
        }
        const message = error instanceof Error ? error.message : 'Failed to open save file dialog.';
        dispatch({ type: 'PAINT_EXPORT_DONE', error: message });
        return;
      }
    }

    dispatch({ type: 'PAINT_EXPORT_START' });
    try {
      const latest = stateRef.current;
      const exportImage = latest.image ?? image;
      const exportGeometry = latest.geometry.data ?? geometry;
      const blob = await renderer.exportPngBlob({
        image: exportImage,
        geometry: exportGeometry,
        paintDoc: latest.paint,
      });

      if (fileHandle) {
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
      } else {
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = suggestedName;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        window.setTimeout(() => {
          URL.revokeObjectURL(url);
        }, 1000);
      }
      dispatch({ type: 'PAINT_EXPORT_DONE' });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to export PNG image.';
      dispatch({ type: 'PAINT_EXPORT_DONE', error: message });
    }
  }, []);

  const clearError = useCallback(() => {
    dispatch({ type: 'CLEAR_ERROR' });
  }, []);

  const setViewportSize = useCallback((size: Size2D) => {
    dispatch({ type: 'SET_VIEWPORT_SIZE', size });
  }, []);

  const setTransform = useCallback((transform: ViewportTransform, zoomMode?: ZoomMode) => {
    dispatch({ type: 'SET_TRANSFORM', transform, zoomMode });
  }, []);

  const fitToScreen = useCallback(() => {
    dispatch({ type: 'FIT_TO_SCREEN' });
  }, []);

  const zoomToActualSize = useCallback(() => {
    dispatch({ type: 'ZOOM_ACTUAL_SIZE' });
  }, []);

  const zoomIn = useCallback(() => {
    dispatch({ type: 'ZOOM_STEP', direction: 'in' });
  }, []);

  const zoomOut = useCallback(() => {
    dispatch({ type: 'ZOOM_STEP', direction: 'out' });
  }, []);

  const setActiveTool = useCallback((tool: EditorTool) => {
    if (tool !== 'brush' && tool !== 'eraser') {
      brushEngineRef.current.cancelStroke();
    }
    dispatch({ type: 'SET_ACTIVE_TOOL', tool });
  }, []);

  const setViewMode = useCallback((mode: ViewMode) => {
    if (mode !== 'rgb') {
      brushEngineRef.current.cancelStroke();
    }
    dispatch({ type: 'SET_VIEW_MODE', mode });
  }, []);

  const value = useMemo<EditorContextValue>(
    () => ({
      state,
      cursorStore,
      fileInputRef,
      brushEngineRef,
      paintRendererRef,
      openFilePicker,
      importFile,
      clearError,
      setViewportSize,
      setTransform,
      fitToScreen,
      zoomToActualSize,
      zoomIn,
      zoomOut,
      setActiveTool,
      setViewMode,
      generateGeometry,
      cancelGeometry,
      selectSurfaceAtImagePoint,
      setActiveSurface,
      removeSurface,
      reorderSurfaces,
      setFootprintExtent,
      setBrushConfig,
      setPaintOverlayOptions,
      commitBrushStroke,
      commitBrushStrokeBatch,
      undoPaint,
      redoPaint,
      exportPng,
    }),
    [
      state,
      cursorStore,
      openFilePicker,
      importFile,
      clearError,
      setViewportSize,
      setTransform,
      fitToScreen,
      zoomToActualSize,
      zoomIn,
      zoomOut,
      setActiveTool,
      setViewMode,
      generateGeometry,
      cancelGeometry,
      selectSurfaceAtImagePoint,
      setActiveSurface,
      removeSurface,
      reorderSurfaces,
      setFootprintExtent,
      setBrushConfig,
      setPaintOverlayOptions,
      commitBrushStroke,
      commitBrushStrokeBatch,
      undoPaint,
      redoPaint,
      exportPng,
    ],
  );

  return <EditorContext.Provider value={value}>{children}</EditorContext.Provider>;
}
