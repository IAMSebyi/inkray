import { useCallback, useEffect, useMemo, useReducer, useRef, type ReactNode } from 'react';
import { fetchAppearanceForImage } from '../api/appearanceClient';
import { fetchGeometryForImage, selectSurfaceForGeometry } from '../api/geometryClient';
import { ImageLoadError, loadImageFromFile } from '../editor/imageLoader';
import type { AutoAnchorMode } from '../geometry/autoSurfaceAnchor';
import { ArtworkModel } from '../painting/artwork';
import {
  type ArtworkPlacement,
  createAutomaticArtworkPlacement,
  estimatePlacementImageCenter,
  reanchorArtworkPlacement,
} from '../painting/artworkPlacement';
import { BrushEngine, type BrushStroke } from '../painting/brushEngine';
import { ImageSpaceBrushEngine, type ImageSpaceStroke } from '../painting/imageSpacePaint';
import { PaintDocument, type BrushUiConfig } from '../painting/paintDocument';
import type { PlanarDecalSurface } from '../painting/planarDecal';
import type { PlanarPaintRenderer } from '../painting/planarPaintRenderer';
import type {
  EditorTool,
  LoadedImage,
  PaintMode,
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
  const artworkFileInputRef = useRef<HTMLInputElement | null>(null);
  const importedArtImagesRef = useRef<Set<LoadedImage>>(new Set());
  const cursorStoreRef = useRef<CursorStore | null>(null);
  if (cursorStoreRef.current === null) {
    cursorStoreRef.current = createCursorStore();
  }
  const cursorStore = cursorStoreRef.current;

  const brushEngineRef = useRef<BrushEngine>(new BrushEngine());
  const imageBrushEngineRef = useRef<ImageSpaceBrushEngine>(new ImageSpaceBrushEngine());
  const paintRendererRef = useRef<PlanarPaintRenderer | null>(null);

  const cancelActiveStrokes = useCallback(() => {
    brushEngineRef.current.cancelStroke();
    imageBrushEngineRef.current.cancelStroke();
  }, []);

  const activeAbortRef = useRef<AbortController | null>(null);
  const geometryAbortRef = useRef<AbortController | null>(null);
  const appearanceAbortRef = useRef<AbortController | null>(null);
  const surfaceSelectAbortRef = useRef<AbortController | null>(null);
  const geometryInFlightRef = useRef(false);
  const appearanceInFlightRef = useRef(false);
  const activeImageRef = useRef<LoadedImage | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    activeImageRef.current = state.image;
  }, [state.image]);

  useEffect(() => {
    const importedImages = importedArtImagesRef.current;
    return () => {
      activeAbortRef.current?.abort();
      geometryAbortRef.current?.abort();
      appearanceAbortRef.current?.abort();
      surfaceSelectAbortRef.current?.abort();
      activeImageRef.current?.dispose();
      for (const artImg of importedImages) {
        artImg.dispose();
      }
      importedImages.clear();
    };
  }, []);

  const openFilePicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const openArtworkFilePicker = useCallback(() => {
    artworkFileInputRef.current?.click();
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
        appearanceAbortRef.current?.abort();
        appearanceAbortRef.current = null;
        appearanceInFlightRef.current = false;
        surfaceSelectAbortRef.current?.abort();
        surfaceSelectAbortRef.current = null;

        const previousImage = activeImageRef.current;
        activeImageRef.current = nextImage;
        cancelActiveStrokes();
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
    [cursorStore, cancelActiveStrokes],
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
    cancelActiveStrokes();
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
  }, [cancelActiveStrokes]);

  const cancelGeometry = useCallback(() => {
    geometryAbortRef.current?.abort();
    geometryAbortRef.current = null;
    geometryInFlightRef.current = false;
    surfaceSelectAbortRef.current?.abort();
    surfaceSelectAbortRef.current = null;
    cancelActiveStrokes();
    dispatch({ type: 'GEOMETRY_CANCEL' });
  }, [cancelActiveStrokes]);

  const generateAppearance = useCallback(async () => {
    const targetImage = activeImageRef.current;
    if (!targetImage || appearanceInFlightRef.current) {
      return;
    }

    appearanceInFlightRef.current = true;
    const abortController = new AbortController();
    appearanceAbortRef.current = abortController;
    dispatch({ type: 'APPEARANCE_START' });

    try {
      const data = await fetchAppearanceForImage(targetImage, abortController.signal);
      if (abortController.signal.aborted || activeImageRef.current?.id !== targetImage.id) {
        return;
      }
      dispatch({ type: 'APPEARANCE_SUCCESS', data });
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      if (activeImageRef.current?.id !== targetImage.id) {
        return;
      }
      const message =
        error instanceof Error ? error.message : 'Failed to generate appearance from backend.';
      dispatch({ type: 'APPEARANCE_ERROR', imageId: targetImage.id, message });
    } finally {
      if (appearanceAbortRef.current === abortController) {
        appearanceAbortRef.current = null;
        appearanceInFlightRef.current = false;
      }
    }
  }, []);

  const cancelAppearance = useCallback(() => {
    appearanceAbortRef.current?.abort();
    appearanceAbortRef.current = null;
    appearanceInFlightRef.current = false;
    dispatch({ type: 'APPEARANCE_CANCEL' });
  }, []);

  const selectSurfaceAtImagePoint = useCallback(
    async (imagePoint: Point2D) => {
      const current = stateRef.current;
      const targetImage = current.image;
      const targetGeometry = current.geometry.data;
      if (!targetImage || !targetGeometry || current.geometry.status !== 'ready') {
        return;
      }

      surfaceSelectAbortRef.current?.abort();
      const abortController = new AbortController();
      surfaceSelectAbortRef.current = abortController;

      cancelActiveStrokes();
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
    },
    [cancelActiveStrokes],
  );

  const setActiveSurface = useCallback(
    (surfaceId: string) => {
      cancelActiveStrokes();
      dispatch({ type: 'PAINT_SET_ACTIVE_SURFACE', surfaceId });
    },
    [cancelActiveStrokes],
  );

  const removeSurface = useCallback(
    (surfaceId: string) => {
      cancelActiveStrokes();
      dispatch({ type: 'PAINT_REMOVE_SURFACE', surfaceId });
    },
    [cancelActiveStrokes],
  );

  const reorderSurfaces = useCallback(
    (fromIndex: number, toIndex: number) => {
      cancelActiveStrokes();
      dispatch({ type: 'PAINT_REORDER_SURFACES', fromIndex, toIndex });
    },
    [cancelActiveStrokes],
  );

  const setFootprintExtent = useCallback((surfaceId: string, extentScale: number) => {
    dispatch({ type: 'PAINT_SET_FOOTPRINT_EXTENT', surfaceId, extentScale });
  }, []);

  const setBrushConfig = useCallback((patch: Partial<BrushUiConfig>) => {
    dispatch({ type: 'PAINT_SET_BRUSH_CONFIG', patch });
  }, []);

  const setPaintMode = useCallback((mode: PaintMode) => {
    dispatch({ type: 'PAINT_SET_MODE', mode });
  }, []);

  const setTexturePreservation = useCallback((value: number) => {
    dispatch({ type: 'PAINT_SET_TEXTURE_PRESERVATION', value });
  }, []);

  const setPaintOverlayOptions = useCallback(
    (patch: {
      showSupportOverlay?: boolean;
      showFootprintPreview?: boolean;
      cleanPreview?: boolean;
    }) => {
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

  const commitAutoStroke = useCallback(
    (payload: {
      anchorMode: AutoAnchorMode;
      newPlanarSurface?: PlanarDecalSurface | null;
      planarStroke?: BrushStroke | null;
      imageStroke?: ImageSpaceStroke | null;
    }) => {
      dispatch({ type: 'PAINT_COMMIT_AUTO_STROKE', ...payload });
    },
    [],
  );

  const undoPaint = useCallback(() => {
    cancelActiveStrokes();
    dispatch({
      type: stateRef.current.workspaceMode === 'artwork' ? 'ARTWORK_UNDO' : 'PAINT_UNDO',
    });
  }, [cancelActiveStrokes]);

  const redoPaint = useCallback(() => {
    cancelActiveStrokes();
    dispatch({
      type: stateRef.current.workspaceMode === 'artwork' ? 'ARTWORK_REDO' : 'PAINT_REDO',
    });
  }, [cancelActiveStrokes]);

  const createBlankArtwork = useCallback(
    (width?: number, height?: number) => {
      cancelActiveStrokes();
      const seq = stateRef.current.paint.nextArtworkNumber;
      const artwork = ArtworkModel.createBlank({
        name: `Artwork ${seq}`,
        width,
        height,
      });
      dispatch({ type: 'ARTWORK_ADD', artwork, openCanvas: true });
    },
    [cancelActiveStrokes],
  );

  const openArtworkEditor = useCallback(
    (artworkId?: string) => {
      cancelActiveStrokes();
      const targetId = artworkId ?? stateRef.current.paint.activeArtworkId;
      if (targetId && PaintDocument.getArtworkById(stateRef.current.paint, targetId)) {
        dispatch({ type: 'SET_WORKSPACE_MODE', mode: 'artwork', artworkId: targetId });
        return;
      }
      createBlankArtwork();
    },
    [cancelActiveStrokes, createBlankArtwork],
  );

  const returnToPhoto = useCallback(() => {
    cancelActiveStrokes();
    dispatch({ type: 'SET_WORKSPACE_MODE', mode: 'photo' });
  }, [cancelActiveStrokes]);

  const importArtworkFile = useCallback(
    async (file: File) => {
      try {
        const loaded = await loadImageFromFile(file);
        importedArtImagesRef.current.add(loaded);
        const artwork = ArtworkModel.createFromImage(loaded);
        cancelActiveStrokes();
        dispatch({
          type: 'ARTWORK_ADD',
          artwork,
          openCanvas: stateRef.current.workspaceMode === 'artwork',
        });
      } catch (error) {
        const message =
          error instanceof ImageLoadError
            ? error.message
            : error instanceof Error
              ? error.message
              : 'Failed to import artwork image.';
        dispatch({ type: 'IMPORT_ERROR', message });
      }
    },
    [cancelActiveStrokes],
  );

  const resizeActiveArtwork = useCallback(
    (width: number, height: number) => {
      const active = PaintDocument.getActiveArtwork(stateRef.current.paint);
      if (!active || active.kind !== 'drawn') {
        return;
      }
      cancelActiveStrokes();
      const next = ArtworkModel.resizeDrawn(active, width, height);
      if (next !== active) {
        dispatch({ type: 'ARTWORK_UPDATE', artwork: next });
      }
    },
    [cancelActiveStrokes],
  );

  const commitArtworkStroke = useCallback((stroke: BrushStroke) => {
    dispatch({ type: 'ARTWORK_COMMIT_STROKE', stroke });
  }, []);

  const clearActiveArtwork = useCallback(() => {
    cancelActiveStrokes();
    dispatch({ type: 'ARTWORK_CLEAR' });
  }, [cancelActiveStrokes]);

  const setActiveArtwork = useCallback(
    (artworkId: string | null) => {
      cancelActiveStrokes();
      dispatch({ type: 'ARTWORK_SET_ACTIVE', artworkId });
    },
    [cancelActiveStrokes],
  );

  const removeArtwork = useCallback(
    (artworkId: string) => {
      const target = PaintDocument.getArtworkById(stateRef.current.paint, artworkId);
      cancelActiveStrokes();
      dispatch({ type: 'ARTWORK_REMOVE', artworkId });
      if (target && target.kind === 'imported') {
        target.image.dispose();
        importedArtImagesRef.current.delete(target.image);
      }
    },
    [cancelActiveStrokes],
  );

  const placeArtworkAtImagePoint = useCallback((artworkId: string, imagePoint?: Point2D) => {
    const current = stateRef.current;
    const image = current.image;
    const artwork = PaintDocument.getArtworkById(current.paint, artworkId);
    if (!image || !artwork) {
      return;
    }
    const imageSize = { width: image.metadata.width, height: image.metadata.height };
    const anchorPoint = imagePoint ?? {
      x: imageSize.width * 0.5,
      y: imageSize.height * 0.5,
    };
    const seq = current.paint.nextPlacementNumber;
    const placement = createAutomaticArtworkPlacement({
      artwork,
      imagePoint: anchorPoint,
      imageSize,
      geometry: current.geometry.data,
      label: `${artwork.name} #${seq}`,
    });
    dispatch({ type: 'PLACEMENT_ADD', placement });
  }, []);

  const updatePlacement = useCallback((placement: ArtworkPlacement, recordHistory = true) => {
    dispatch({ type: 'PLACEMENT_UPDATE', placement, recordHistory });
  }, []);

  const setActivePlacement = useCallback((placementId: string | null) => {
    dispatch({ type: 'PLACEMENT_SET_ACTIVE', placementId });
  }, []);

  const togglePlacementVisibility = useCallback((placementId: string) => {
    dispatch({ type: 'PLACEMENT_TOGGLE_VISIBILITY', placementId });
  }, []);

  const removePlacement = useCallback((placementId: string) => {
    dispatch({ type: 'PLACEMENT_REMOVE', placementId });
  }, []);

  const createBrushObject = useCallback(
    (label?: string) => {
      cancelActiveStrokes();
      dispatch({ type: 'PAINT_CREATE_BRUSH_OBJECT', label });
    },
    [cancelActiveStrokes],
  );

  const setActiveBrushObject = useCallback(
    (brushObjectId: string | null) => {
      cancelActiveStrokes();
      dispatch({ type: 'PAINT_SET_ACTIVE_BRUSH_OBJECT', brushObjectId });
    },
    [cancelActiveStrokes],
  );

  const toggleBrushObjectVisibility = useCallback(
    (brushObjectId: string) => {
      cancelActiveStrokes();
      dispatch({ type: 'PAINT_TOGGLE_BRUSH_VISIBILITY', brushObjectId });
    },
    [cancelActiveStrokes],
  );

  const removeBrushObject = useCallback(
    (brushObjectId: string) => {
      cancelActiveStrokes();
      dispatch({ type: 'PAINT_REMOVE_BRUSH_OBJECT', brushObjectId });
    },
    [cancelActiveStrokes],
  );

  const reorderSceneObject = useCallback(
    (fromIndex: number, toIndex: number) => {
      cancelActiveStrokes();
      dispatch({ type: 'PAINT_REORDER_SCENE_OBJECT', fromIndex, toIndex });
    },
    [cancelActiveStrokes],
  );

  const reanchorPlacement = useCallback((placementId: string, imagePoint?: Point2D) => {
    const current = stateRef.current;
    const image = current.image;
    const existing = current.paint.artworkPlacements.find((p) => p.id === placementId);
    const artwork = existing
      ? PaintDocument.getArtworkById(current.paint, existing.artworkId)
      : null;
    if (!image || !existing || !artwork) {
      return;
    }
    const imageSize = { width: image.metadata.width, height: image.metadata.height };
    const anchorPoint = imagePoint ??
      estimatePlacementImageCenter(existing, imageSize, current.geometry.data) ?? {
        x: imageSize.width * 0.5,
        y: imageSize.height * 0.5,
      };
    const next = reanchorArtworkPlacement({
      placement: existing,
      artwork,
      imagePoint: anchorPoint,
      imageSize,
      geometry: current.geometry.data,
    });
    dispatch({ type: 'PLACEMENT_UPDATE', placement: next, recordHistory: true });
  }, []);

  const exportPng = useCallback(async () => {
    const current = stateRef.current;
    const image = current.image;
    const geometry = current.geometry.data;
    const renderer = paintRendererRef.current;

    if (!image || !renderer) {
      dispatch({
        type: 'PAINT_EXPORT_DONE',
        error: 'Load an image before exporting.',
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
        // Fallback to direct browser download when save dialog API is unavailable in this context.
        fileHandle = null;
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
        appearance: latest.appearance.data,
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

  const setActiveTool = useCallback(
    (tool: EditorTool) => {
      if (tool !== 'brush' && tool !== 'eraser') {
        cancelActiveStrokes();
      }
      dispatch({ type: 'SET_ACTIVE_TOOL', tool });
    },
    [cancelActiveStrokes],
  );

  const setViewMode = useCallback(
    (mode: ViewMode) => {
      if (mode !== 'rgb') {
        cancelActiveStrokes();
      }
      dispatch({ type: 'SET_VIEW_MODE', mode });
    },
    [cancelActiveStrokes],
  );

  const value = useMemo<EditorContextValue>(
    () => ({
      state,
      cursorStore,
      fileInputRef,
      artworkFileInputRef,
      brushEngineRef,
      imageBrushEngineRef,
      paintRendererRef,
      openFilePicker,
      openArtworkFilePicker,
      importFile,
      importArtworkFile,
      clearError,
      setViewportSize,
      setTransform,
      fitToScreen,
      zoomToActualSize,
      zoomIn,
      zoomOut,
      setActiveTool,
      setViewMode,
      openArtworkEditor,
      createBlankArtwork,
      returnToPhoto,
      resizeActiveArtwork,
      commitArtworkStroke,
      clearActiveArtwork,
      setActiveArtwork,
      removeArtwork,
      placeArtworkAtImagePoint,
      updatePlacement,
      setActivePlacement,
      togglePlacementVisibility,
      removePlacement,
      reanchorPlacement,
      createBrushObject,
      setActiveBrushObject,
      toggleBrushObjectVisibility,
      removeBrushObject,
      reorderSceneObject,
      generateGeometry,
      cancelGeometry,
      generateAppearance,
      cancelAppearance,
      selectSurfaceAtImagePoint,
      setActiveSurface,
      removeSurface,
      reorderSurfaces,
      setFootprintExtent,
      setBrushConfig,
      setPaintMode,
      setTexturePreservation,
      setPaintOverlayOptions,
      commitBrushStroke,
      commitBrushStrokeBatch,
      commitAutoStroke,
      undoPaint,
      redoPaint,
      exportPng,
    }),
    [
      state,
      cursorStore,
      openFilePicker,
      openArtworkFilePicker,
      importFile,
      importArtworkFile,
      clearError,
      setViewportSize,
      setTransform,
      fitToScreen,
      zoomToActualSize,
      zoomIn,
      zoomOut,
      setActiveTool,
      setViewMode,
      openArtworkEditor,
      createBlankArtwork,
      returnToPhoto,
      resizeActiveArtwork,
      commitArtworkStroke,
      clearActiveArtwork,
      setActiveArtwork,
      removeArtwork,
      placeArtworkAtImagePoint,
      updatePlacement,
      setActivePlacement,
      togglePlacementVisibility,
      removePlacement,
      reanchorPlacement,
      createBrushObject,
      setActiveBrushObject,
      toggleBrushObjectVisibility,
      removeBrushObject,
      reorderSceneObject,
      generateGeometry,
      cancelGeometry,
      generateAppearance,
      cancelAppearance,
      selectSurfaceAtImagePoint,
      setActiveSurface,
      removeSurface,
      reorderSurfaces,
      setFootprintExtent,
      setBrushConfig,
      setPaintMode,
      setTexturePreservation,
      setPaintOverlayOptions,
      commitBrushStroke,
      commitBrushStrokeBatch,
      commitAutoStroke,
      undoPaint,
      redoPaint,
      exportPng,
    ],
  );

  return <EditorContext.Provider value={value}>{children}</EditorContext.Provider>;
}
