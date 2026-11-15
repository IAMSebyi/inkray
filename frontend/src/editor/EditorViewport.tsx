import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { EmptyViewportState } from '../components/EmptyViewportState';
import { ErrorBanner } from '../components/ErrorBanner';
import { useDevicePixelRatio } from '../hooks/useDevicePixelRatio';
import { useEditor } from '../hooks/useEditor';
import { useElementSize } from '../hooks/useElementSize';
import type { Size2D, ViewportTransform } from '../types/editor';
import { renderViewportCanvas } from './canvasRenderer';
import {
  clampPanToViewport,
  clientToViewport,
  computeWheelZoomTarget,
  getEffectiveZoomBounds,
  panTransform,
  sampleImageAtClientPoint,
  zoomAtPoint,
} from './viewportTransforms';

interface EditorViewportProps {
  isSpacePressed: boolean;
  isDragActive: boolean;
  dropZoneProps: {
    onDragEnter: (event: React.DragEvent<HTMLElement>) => void;
    onDragOver: (event: React.DragEvent<HTMLElement>) => void;
    onDragLeave: (event: React.DragEvent<HTMLElement>) => void;
    onDrop: (event: React.DragEvent<HTMLElement>) => void;
  };
}

interface PanSession {
  active: boolean;
  pointerId: number | null;
  lastClientX: number;
  lastClientY: number;
}

export function EditorViewport({
  isSpacePressed,
  isDragActive,
  dropZoneProps,
}: EditorViewportProps) {
  const { state, cursorStore, setViewportSize, setTransform } = useEditor();
  const dpr = useDevicePixelRatio();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isPanning, setIsPanning] = useState(false);

  const liveTransformRef = useRef<ViewportTransform>(state.transform);
  const lastDispatchedTransformRef = useRef<ViewportTransform | null>(null);
  const renderRafRef = useRef<number | null>(null);
  const syncRafRef = useRef<number | null>(null);
  const panSessionRef = useRef<PanSession>({
    active: false,
    pointerId: null,
    lastClientX: 0,
    lastClientY: 0,
  });

  const latestRenderStateRef = useRef({
    image: state.image,
    viewportSize: state.viewportSize,
    dpr,
  });

  latestRenderStateRef.current = {
    image: state.image,
    viewportSize: state.viewportSize,
    dpr,
  };

  useElementSize(containerRef, setViewportSize);

  const drawCanvasNow = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    const { image, viewportSize, dpr: currentDpr } = latestRenderStateRef.current;
    renderViewportCanvas(ctx, {
      viewportSize,
      dpr: currentDpr,
      image,
      transform: liveTransformRef.current,
    });
  }, []);

  const scheduleCanvasRender = useCallback(() => {
    if (renderRafRef.current !== null) {
      return;
    }
    renderRafRef.current = window.requestAnimationFrame(() => {
      renderRafRef.current = null;
      drawCanvasNow();
    });
  }, [drawCanvasNow]);

  const scheduleTransformSync = useCallback(() => {
    if (syncRafRef.current !== null) {
      return;
    }
    syncRafRef.current = window.requestAnimationFrame(() => {
      syncRafRef.current = null;
      const next = liveTransformRef.current;
      lastDispatchedTransformRef.current = next;
      setTransform(next, 'custom');
    });
  }, [setTransform]);

  // Sync external transform updates (fit, 100%, import, resize) into the live ref.
  useLayoutEffect(() => {
    if (state.transform !== lastDispatchedTransformRef.current) {
      liveTransformRef.current = state.transform;
    }
    drawCanvasNow();
  }, [state.transform, state.image, state.viewportSize, dpr, drawCanvasNow]);

  useEffect(() => {
    return () => {
      if (renderRafRef.current !== null) {
        window.cancelAnimationFrame(renderRafRef.current);
      }
      if (syncRafRef.current !== null) {
        window.cancelAnimationFrame(syncRafRef.current);
      }
    };
  }, []);

  // Non-passive wheel listener blocks browser page zoom during trackpad pinch.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();

      const { image, viewportSize } = latestRenderStateRef.current;
      const canvas = canvasRef.current;
      if (!image || !canvas || viewportSize.width <= 0 || viewportSize.height <= 0) {
        return;
      }

      const imageSize: Size2D = {
        width: image.metadata.width,
        height: image.metadata.height,
      };
      const rect = canvas.getBoundingClientRect();
      const anchor = clientToViewport({ x: event.clientX, y: event.clientY }, rect);
      const bounds = getEffectiveZoomBounds(viewportSize, imageSize);
      const targetZoom = computeWheelZoomTarget(
        liveTransformRef.current.zoom,
        event.deltaY,
        event.deltaMode,
        event.ctrlKey,
        viewportSize.height,
        bounds,
      );

      const nextTransform = zoomAtPoint(liveTransformRef.current, anchor, targetZoom, bounds);

      liveTransformRef.current = nextTransform;
      scheduleCanvasRender();
      scheduleTransformSync();

      cursorStore.setSample(
        sampleImageAtClientPoint(
          { x: event.clientX, y: event.clientY },
          rect,
          nextTransform,
          imageSize,
        ),
      );
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      container.removeEventListener('wheel', handleWheel);
    };
  }, [cursorStore, scheduleCanvasRender, scheduleTransformSync]);

  const updateCursorSample = useCallback(
    (clientX: number, clientY: number) => {
      const image = latestRenderStateRef.current.image;
      const canvas = canvasRef.current;
      if (!image || !canvas) {
        cursorStore.setSample(null);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const sample = sampleImageAtClientPoint(
        { x: clientX, y: clientY },
        rect,
        liveTransformRef.current,
        {
          width: image.metadata.width,
          height: image.metadata.height,
        },
      );
      cursorStore.setSample(sample);
    },
    [cursorStore],
  );

  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    // Disable native middle-click autoscroll.
    if (event.button === 1) {
      event.preventDefault();
    }
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!state.image) {
      return;
    }

    const isMiddleClick = event.button === 1;
    const isLeftPanClick = event.button === 0 && (isSpacePressed || state.activeTool === 'hand');

    if (!isMiddleClick && !isLeftPanClick) {
      return;
    }

    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);

    panSessionRef.current = {
      active: true,
      pointerId: event.pointerId,
      lastClientX: event.clientX,
      lastClientY: event.clientY,
    };
    setIsPanning(true);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const panSession = panSessionRef.current;

    if (panSession.active && panSession.pointerId === event.pointerId) {
      const { image, viewportSize } = latestRenderStateRef.current;
      if (!image) {
        return;
      }

      const deltaX = event.clientX - panSession.lastClientX;
      const deltaY = event.clientY - panSession.lastClientY;
      panSession.lastClientX = event.clientX;
      panSession.lastClientY = event.clientY;

      const imageSize: Size2D = {
        width: image.metadata.width,
        height: image.metadata.height,
      };
      const nextTransform = clampPanToViewport(
        panTransform(liveTransformRef.current, deltaX, deltaY),
        viewportSize,
        imageSize,
      );

      liveTransformRef.current = nextTransform;
      scheduleCanvasRender();
      scheduleTransformSync();
    }

    updateCursorSample(event.clientX, event.clientY);
  };

  const endPanSession = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const panSession = panSessionRef.current;
    if (!panSession.active || panSession.pointerId !== event.pointerId) {
      return;
    }

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    panSessionRef.current = {
      active: false,
      pointerId: null,
      lastClientX: 0,
      lastClientY: 0,
    };
    setIsPanning(false);
  };

  const handlePointerLeave = () => {
    if (!panSessionRef.current.active) {
      cursorStore.setSample(null);
    }
  };

  const cursorModeClass = !state.image
    ? 'cursor-default'
    : isPanning
      ? 'cursor-grabbing'
      : isSpacePressed || state.activeTool === 'hand'
        ? 'cursor-grab'
        : 'cursor-default';

  return (
    <main
      ref={containerRef}
      className={`editor-viewport ${cursorModeClass}`}
      aria-label="Image editor viewport"
      {...dropZoneProps}
    >
      <canvas
        ref={canvasRef}
        className="viewport-canvas"
        onMouseDown={handleMouseDown}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endPanSession}
        onPointerCancel={endPanSession}
        onPointerLeave={handlePointerLeave}
      />

      {!state.image && !state.isLoading && <EmptyViewportState />}

      {state.isLoading && (
        <div className="viewport-loading-overlay" aria-live="polite">
          <div className="loading-pill">Loading image…</div>
        </div>
      )}

      {isDragActive && (
        <div className="drop-overlay" aria-hidden="true">
          <div className="drop-overlay-frame">
            <span className="drop-overlay-label">
              {state.image ? 'Drop JPEG or PNG to replace image' : 'Drop JPEG or PNG to open'}
            </span>
          </div>
        </div>
      )}

      <ErrorBanner />
    </main>
  );
}
