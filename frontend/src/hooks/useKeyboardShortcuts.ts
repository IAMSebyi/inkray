import { useEffect, useRef, useState } from 'react';
import { useEditor } from './useEditor';

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tagName = target.tagName;
  if (tagName === 'TEXTAREA' || tagName === 'SELECT' || target.isContentEditable) {
    return true;
  }
  if (target instanceof HTMLInputElement) {
    const inputType = target.type.toLowerCase();
    return (
      inputType === 'text' ||
      inputType === 'search' ||
      inputType === 'url' ||
      inputType === 'tel' ||
      inputType === 'email' ||
      inputType === 'password' ||
      inputType === 'number'
    );
  }
  return false;
}

export function useKeyboardShortcuts(): { isSpacePressed: boolean } {
  const {
    state,
    openFilePicker,
    fitToScreen,
    zoomToActualSize,
    zoomIn,
    zoomOut,
    setActiveTool,
    setBrushConfig,
    setPaintOverlayOptions,
    removePlacement,
    undoPaint,
    redoPaint,
  } = useEditor();
  const [isSpacePressed, setIsSpacePressed] = useState(false);

  const hasImage = state.image !== null;
  const isArtworkMode = state.workspaceMode === 'artwork';
  const hasOverlays = state.paint.surfaces.length > 0 || state.paint.artworkPlacements.length > 0;
  const activePlacementId = state.paint.activePlacementId;
  const activeTool = state.activeTool;
  const brushPixelSizeRef = useRef(state.paint.brushConfig.pixelSize);
  const cleanPreviewRef = useRef(state.paint.cleanPreview);

  useEffect(() => {
    brushPixelSizeRef.current = state.paint.brushConfig.pixelSize;
  }, [state.paint.brushConfig.pixelSize]);

  useEffect(() => {
    cleanPreviewRef.current = state.paint.cleanPreview;
  }, [state.paint.cleanPreview]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (isEditableElement(event.target)) {
        return;
      }

      const hasPrimaryMod = event.ctrlKey || event.metaKey;

      if (event.code === 'Space' && !hasPrimaryMod && !event.altKey) {
        event.preventDefault();
        if (!event.repeat) {
          setIsSpacePressed(true);
        }
        return;
      }

      if (hasPrimaryMod && event.key.toLowerCase() === 'o') {
        event.preventDefault();
        openFilePicker();
        return;
      }

      if (hasPrimaryMod && !event.altKey) {
        const lower = event.key.toLowerCase();
        if (lower === 'z' || event.code === 'KeyZ') {
          event.preventDefault();
          if (event.shiftKey) {
            redoPaint();
          } else {
            undoPaint();
          }
          return;
        }
        if ((lower === 'y' || event.code === 'KeyY') && !event.shiftKey) {
          event.preventDefault();
          redoPaint();
          return;
        }
      }

      if (!hasPrimaryMod && !event.altKey && !event.shiftKey) {
        const lower = event.key.toLowerCase();
        if (lower === 'o') {
          event.preventDefault();
          openFilePicker();
          return;
        }
        if (lower === 'v') {
          setActiveTool('pointer');
          return;
        }
        if (lower === 'h') {
          setActiveTool('hand');
          return;
        }
        if (lower === 'b') {
          setActiveTool('brush');
          return;
        }
        if (lower === 'e') {
          setActiveTool('eraser');
          return;
        }
        if (lower === 'p' && !isArtworkMode && hasOverlays) {
          setPaintOverlayOptions({ cleanPreview: !cleanPreviewRef.current });
          return;
        }
        if (event.key === '[') {
          const next = Math.max(4, Math.min(240, Math.round(brushPixelSizeRef.current * 0.85)));
          setBrushConfig({
            pixelSize: next === brushPixelSizeRef.current ? Math.max(4, next - 2) : next,
          });
          return;
        }
        if (event.key === ']') {
          const next = Math.max(4, Math.min(240, Math.round(brushPixelSizeRef.current * 1.18)));
          setBrushConfig({
            pixelSize: next === brushPixelSizeRef.current ? Math.min(240, next + 2) : next,
          });
          return;
        }
        if (
          (event.key === 'Delete' || event.key === 'Backspace') &&
          !isArtworkMode &&
          activeTool === 'pointer' &&
          activePlacementId
        ) {
          event.preventDefault();
          removePlacement(activePlacementId);
          return;
        }
      }

      if (!hasImage || isArtworkMode) {
        return;
      }

      if (
        (hasPrimaryMod && event.key === '0') ||
        (!hasPrimaryMod && !event.altKey && (event.key.toLowerCase() === 'f' || event.key === '0'))
      ) {
        event.preventDefault();
        fitToScreen();
        return;
      }

      if (
        (hasPrimaryMod && event.key === '1') ||
        (!hasPrimaryMod && !event.altKey && event.key === '1')
      ) {
        event.preventDefault();
        zoomToActualSize();
        return;
      }

      if (event.key === '+' || event.key === '=' || event.code === 'NumpadAdd') {
        if (hasPrimaryMod || !event.altKey) {
          event.preventDefault();
          zoomIn();
        }
        return;
      }

      if (event.key === '-' || event.code === 'NumpadSubtract') {
        if (hasPrimaryMod || !event.altKey) {
          event.preventDefault();
          zoomOut();
        }
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space') {
        setIsSpacePressed(false);
      }
    };

    const resetSpace = () => {
      setIsSpacePressed(false);
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', resetSpace);
    document.addEventListener('visibilitychange', resetSpace);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', resetSpace);
      document.removeEventListener('visibilitychange', resetSpace);
    };
  }, [
    hasImage,
    isArtworkMode,
    hasOverlays,
    activeTool,
    activePlacementId,
    openFilePicker,
    fitToScreen,
    zoomToActualSize,
    zoomIn,
    zoomOut,
    setActiveTool,
    setBrushConfig,
    setPaintOverlayOptions,
    removePlacement,
    undoPaint,
    redoPaint,
  ]);

  return { isSpacePressed };
}
