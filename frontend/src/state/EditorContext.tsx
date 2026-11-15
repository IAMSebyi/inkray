import { useCallback, useEffect, useMemo, useReducer, useRef, type ReactNode } from 'react';
import { ImageLoadError, loadImageFromFile } from '../editor/imageLoader';
import type { EditorTool, LoadedImage, Size2D, ViewportTransform, ZoomMode } from '../types/editor';
import { createCursorStore, type CursorStore } from './cursorStore';
import { EditorContext, type EditorContextValue } from './editorContextValue';
import { editorReducer, initialEditorState } from './editorReducer';

export function EditorProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(editorReducer, initialEditorState);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const cursorStoreRef = useRef<CursorStore | null>(null);
  if (cursorStoreRef.current === null) {
    cursorStoreRef.current = createCursorStore();
  }
  const cursorStore = cursorStoreRef.current;

  const activeAbortRef = useRef<AbortController | null>(null);
  const activeImageRef = useRef<LoadedImage | null>(null);

  useEffect(() => {
    activeImageRef.current = state.image;
  }, [state.image]);

  useEffect(() => {
    return () => {
      activeAbortRef.current?.abort();
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

        const previousImage = activeImageRef.current;
        activeImageRef.current = nextImage;
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
    dispatch({ type: 'SET_ACTIVE_TOOL', tool });
  }, []);

  const value = useMemo<EditorContextValue>(
    () => ({
      state,
      cursorStore,
      fileInputRef,
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
    ],
  );

  return <EditorContext.Provider value={value}>{children}</EditorContext.Provider>;
}
