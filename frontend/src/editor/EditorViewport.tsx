import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { EmptyViewportState } from '../components/EmptyViewportState';
import { ErrorBanner } from '../components/ErrorBanner';
import { parseNormalizedIntrinsics } from '../geometry/geometryAccess';
import { useDevicePixelRatio } from '../hooks/useDevicePixelRatio';
import { useEditor } from '../hooks/useEditor';
import { useElementSize } from '../hooks/useElementSize';
import { PaintDocument } from '../painting/paintDocument';
import { getEffectiveFootprint } from '../painting/planarDecal';
import { findSurfaceOwnerAtImagePoint } from '../painting/planeVisibility';
import { PlanarPaintRenderer } from '../painting/planarPaintRenderer';
import type { Point2D, Size2D, ViewportTransform } from '../types/editor';
import { PointCloudViewer } from './PointCloudViewer';
import { buildBrushCursorPreview, renderViewportCanvas } from './canvasRenderer';
import {
  clampPanToViewport,
  clientToImage,
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

interface PaintSession {
  active: boolean;
  pointerId: number | null;
  surfaceId: string | null;
  lastImagePoint: Point2D | null;
}

export function EditorViewport({
  isSpacePressed,
  isDragActive,
  dropZoneProps,
}: EditorViewportProps) {
  const {
    state,
    cursorStore,
    brushEngineRef,
    paintRendererRef,
    setViewportSize,
    setTransform,
    selectSurfaceAtImagePoint,
    commitBrushStrokeBatch,
  } = useEditor();
  const dpr = useDevicePixelRatio();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const paintHostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isPanning, setIsPanning] = useState(false);

  const liveTransformRef = useRef<ViewportTransform>(state.transform);
  const lastDispatchedTransformRef = useRef<ViewportTransform | null>(null);
  const cursorViewportPointRef = useRef<Point2D | null>(null);
  const renderRafRef = useRef<number | null>(null);
  const syncRafRef = useRef<number | null>(null);
  const panSessionRef = useRef<PanSession>({
    active: false,
    pointerId: null,
    lastClientX: 0,
    lastClientY: 0,
  });
  const paintSessionRef = useRef<PaintSession>({
    active: false,
    pointerId: null,
    surfaceId: null,
    lastImagePoint: null,
  });

  const latestRenderStateRef = useRef({
    image: state.image,
    viewportSize: state.viewportSize,
    dpr,
    viewMode: state.viewMode,
    activeTool: state.activeTool,
    geometry: state.geometry.data,
    appearance: state.appearance.data,
    paint: state.paint,
    isPanning,
  });

  latestRenderStateRef.current = {
    image: state.image,
    viewportSize: state.viewportSize,
    dpr,
    viewMode: state.viewMode,
    activeTool: state.activeTool,
    geometry: state.geometry.data,
    appearance: state.appearance.data,
    paint: state.paint,
    isPanning,
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
    const {
      image,
      viewportSize,
      dpr: currentDpr,
      viewMode,
      activeTool,
      geometry,
      appearance,
      paint,
      isPanning: currentIsPanning,
    } = latestRenderStateRef.current;
    if (viewMode === 'pointcloud') {
      return;
    }

    const paintRenderer = paintRendererRef.current;
    const usePaintWebGl =
      viewMode === 'rgb' && image !== null && geometry !== null && paintRenderer !== null;

    if (viewMode === 'rgb' && image !== null && geometry !== null && paintRenderer !== null) {
      paintRenderer.renderViewport({
        viewportSize,
        dpr: currentDpr,
        image,
        transform: liveTransformRef.current,
        geometry,
        appearance,
        paintDoc: paint,
        pendingStrokes: brushEngineRef.current.getPreviewStrokes(),
        activeStroke: brushEngineRef.current.getActiveStroke(),
      });
    }

    const brushCursor =
      image &&
      viewMode === 'rgb' &&
      !currentIsPanning &&
      !isSpacePressed &&
      cursorViewportPointRef.current
        ? buildBrushCursorPreview({
            tool: activeTool,
            viewportPoint: cursorViewportPointRef.current,
            imageSize: { width: image.metadata.width, height: image.metadata.height },
            transform: liveTransformRef.current,
            geometry,
            paintDoc: paint,
          })
        : null;

    renderViewportCanvas(ctx, {
      viewportSize,
      dpr: currentDpr,
      image,
      transform: liveTransformRef.current,
      viewMode,
      geometry,
      appearance,
      borderOnly: usePaintWebGl,
      brushCursor,
    });
  }, [brushEngineRef, isSpacePressed, paintRendererRef]);

  useEffect(() => {
    const host = paintHostRef.current;
    if (!host) {
      return;
    }
    const renderer = new PlanarPaintRenderer(host);
    paintRendererRef.current = renderer;
    drawCanvasNow();
    return () => {
      renderer.dispose();
      paintRendererRef.current = null;
    };
  }, [drawCanvasNow, paintRendererRef]);

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

  // Sync external transform updates (fit, 100%, import, resize, view mode, paint) into the live ref.
  useLayoutEffect(() => {
    if (state.transform !== lastDispatchedTransformRef.current) {
      liveTransformRef.current = state.transform;
    }
    if (state.viewMode === 'pointcloud') {
      cursorStore.setSample(null);
      return;
    }
    drawCanvasNow();
  }, [
    state.transform,
    state.image,
    state.viewportSize,
    state.viewMode,
    state.activeTool,
    state.geometry.data,
    state.appearance.data,
    state.paint,
    state.isExporting,
    isSpacePressed,
    isPanning,
    dpr,
    cursorStore,
    drawCanvasNow,
  ]);

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
      const { image, viewportSize, viewMode } = latestRenderStateRef.current;
      if (viewMode === 'pointcloud') {
        return;
      }

      event.preventDefault();

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
      cursorViewportPointRef.current = anchor;
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

    if (isMiddleClick || isLeftPanClick) {
      if (paintSessionRef.current.active) {
        brushEngineRef.current.cancelStroke();
        paintSessionRef.current = {
          active: false,
          pointerId: null,
          surfaceId: null,
          lastImagePoint: null,
        };
        scheduleCanvasRender();
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
      return;
    }

    if (event.button !== 0 || state.viewMode === 'pointcloud' || !state.geometry.data) {
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    cursorViewportPointRef.current = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    const imagePoint = clientToImage(
      { x: event.clientX, y: event.clientY },
      rect,
      liveTransformRef.current,
    );

    if (state.activeTool === 'select-surface') {
      event.preventDefault();
      void selectSurfaceAtImagePoint(imagePoint);
      return;
    }

    if (
      (state.activeTool === 'brush' || state.activeTool === 'eraser') &&
      state.viewMode === 'rgb'
    ) {
      const imageSize = {
        width: state.image.metadata.width,
        height: state.image.metadata.height,
      };
      const ownerSurface = findSurfaceOwnerAtImagePoint(
        state.paint.surfaces,
        imagePoint,
        imageSize,
        state.geometry.data,
      );
      const targetSurface = ownerSurface ?? PaintDocument.getActiveSurface(state.paint);
      if (!targetSurface) {
        return;
      }

      const intr = parseNormalizedIntrinsics(state.geometry.data.metadata.intrinsics);
      const hit = targetSurface.frame.imagePointToPlane(imagePoint, imageSize, intr);
      if (!hit) {
        return;
      }

      const brush = PaintDocument.getEffectiveBrushSettingsForSurface(state.paint, targetSurface);
      const footprint = getEffectiveFootprint(targetSurface);
      const strokeMode = state.activeTool === 'eraser' ? 'erase' : 'paint';
      const started = brushEngineRef.current.beginStroke(
        targetSurface.id,
        footprint,
        brush,
        { u: hit.u, v: hit.v },
        strokeMode,
      );
      if (!started) {
        return;
      }

      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      paintSessionRef.current = {
        active: true,
        pointerId: event.pointerId,
        surfaceId: targetSurface.id,
        lastImagePoint: imagePoint,
      };
      scheduleCanvasRender();
    }
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    cursorViewportPointRef.current = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };

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
      updateCursorSample(event.clientX, event.clientY);
      return;
    }

    const paintSession = paintSessionRef.current;
    if (paintSession.active && paintSession.pointerId === event.pointerId) {
      const { image, geometry, paint } = latestRenderStateRef.current;
      if (image && geometry && paint.surfaces.length > 0) {
        const imageSize = {
          width: image.metadata.width,
          height: image.metadata.height,
        };
        const imagePoint = clientToImage(
          { x: event.clientX, y: event.clientY },
          rect,
          liveTransformRef.current,
        );
        const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
        const ownerSurface = findSurfaceOwnerAtImagePoint(
          paint.surfaces,
          imagePoint,
          imageSize,
          geometry,
        );
        const prevSurface = paintSession.surfaceId
          ? (paint.surfaces.find((s) => s.id === paintSession.surfaceId) ?? null)
          : null;

        if (ownerSurface && ownerSurface.id !== paintSession.surfaceId) {
          // Refine seam crossing along the pointer segment so cross-surface strokes connect cleanly.
          const prevImg = paintSession.lastImagePoint ?? imagePoint;
          let loImg = prevImg;
          let hiImg = imagePoint;
          for (let step = 0; step < 6; step++) {
            const midImg = { x: 0.5 * (loImg.x + hiImg.x), y: 0.5 * (loImg.y + hiImg.y) };
            const midOwner = findSurfaceOwnerAtImagePoint(
              paint.surfaces,
              midImg,
              imageSize,
              geometry,
            );
            if (midOwner?.id === ownerSurface.id) {
              hiImg = midImg;
            } else {
              loImg = midImg;
            }
          }

          const prevEndHit = prevSurface
            ? prevSurface.frame.imagePointToPlane(loImg, imageSize, intr)
            : null;
          const nextStartHit = ownerSurface.frame.imagePointToPlane(hiImg, imageSize, intr);
          const nextCurrHit = ownerSurface.frame.imagePointToPlane(imagePoint, imageSize, intr);
          const nextStart = nextStartHit ?? nextCurrHit;

          if (nextStart) {
            const nextBrush = PaintDocument.getEffectiveBrushSettingsForSurface(
              paint,
              ownerSurface,
            );
            const nextFootprint = getEffectiveFootprint(ownerSurface);
            const switched = brushEngineRef.current.switchSurfaceStroke(
              ownerSurface.id,
              nextFootprint,
              nextBrush,
              { u: nextStart.u, v: nextStart.v },
              prevEndHit ? { u: prevEndHit.u, v: prevEndHit.v } : null,
            );
            if (switched) {
              if (nextCurrHit && nextCurrHit !== nextStart) {
                brushEngineRef.current.extendStroke({ u: nextCurrHit.u, v: nextCurrHit.v }, true);
              }
              paintSession.surfaceId = ownerSurface.id;
            }
          }
        } else if (prevSurface) {
          const hit = prevSurface.frame.imagePointToPlane(imagePoint, imageSize, intr);
          if (hit) {
            brushEngineRef.current.extendStroke({ u: hit.u, v: hit.v });
          }
        }

        paintSession.lastImagePoint = imagePoint;
        scheduleCanvasRender();
      }
    } else {
      const { activeTool, viewMode } = latestRenderStateRef.current;
      if (viewMode === 'rgb' && (activeTool === 'brush' || activeTool === 'eraser')) {
        scheduleCanvasRender();
      }
    }

    updateCursorSample(event.clientX, event.clientY);
  };

  const endPointerSession = (event: React.PointerEvent<HTMLCanvasElement>, cancelled: boolean) => {
    const panSession = panSessionRef.current;
    if (panSession.active && panSession.pointerId === event.pointerId) {
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
    }

    const paintSession = paintSessionRef.current;
    if (paintSession.active && paintSession.pointerId === event.pointerId) {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      paintSessionRef.current = {
        active: false,
        pointerId: null,
        surfaceId: null,
        lastImagePoint: null,
      };
      if (cancelled) {
        brushEngineRef.current.cancelStroke();
        scheduleCanvasRender();
      } else {
        const batch = brushEngineRef.current.commitBatch();
        if (batch.length > 0) {
          commitBrushStrokeBatch(batch);
        } else {
          scheduleCanvasRender();
        }
      }
    }
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    endPointerSession(event, false);
  };

  const handlePointerCancel = (event: React.PointerEvent<HTMLCanvasElement>) => {
    endPointerSession(event, true);
  };

  const handlePointerLeave = () => {
    cursorViewportPointRef.current = null;
    if (!panSessionRef.current.active && !paintSessionRef.current.active) {
      cursorStore.setSample(null);
      scheduleCanvasRender();
    }
  };

  const isPointCloudMode = state.viewMode === 'pointcloud' && state.geometry.data !== null;
  const isPaintWebGlVisible =
    !isPointCloudMode &&
    state.viewMode === 'rgb' &&
    state.image !== null &&
    state.geometry.data !== null;

  const cursorModeClass =
    !state.image || isPointCloudMode
      ? 'cursor-default'
      : isPanning
        ? 'cursor-grabbing'
        : isSpacePressed || state.activeTool === 'hand'
          ? 'cursor-grab'
          : (state.activeTool === 'brush' || state.activeTool === 'eraser') &&
              state.viewMode === 'rgb' &&
              state.paint.surfaces.length > 0
            ? 'cursor-brush'
            : state.activeTool === 'select-surface'
              ? 'cursor-crosshair'
              : 'cursor-default';

  return (
    <main
      ref={containerRef}
      className={`editor-viewport ${cursorModeClass}`}
      aria-label="Image editor viewport"
      {...dropZoneProps}
    >
      <div
        ref={paintHostRef}
        className="paint-webgl-host"
        style={isPaintWebGlVisible ? undefined : { display: 'none' }}
      />

      <canvas
        ref={canvasRef}
        className="viewport-canvas"
        style={isPointCloudMode ? { display: 'none' } : undefined}
        onMouseDown={handleMouseDown}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerCancel}
        onPointerLeave={handlePointerLeave}
      />

      {isPointCloudMode && state.geometry.data && (
        <PointCloudViewer key={state.geometry.data.metadata.id} geometry={state.geometry.data} />
      )}

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
