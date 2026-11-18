import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { EmptyViewportState } from '../components/EmptyViewportState';
import { ErrorBanner } from '../components/ErrorBanner';
import { classifyAndAnchorSurface } from '../geometry/autoSurfaceAnchor';
import { parseNormalizedIntrinsics } from '../geometry/geometryAccess';
import { useDevicePixelRatio } from '../hooks/useDevicePixelRatio';
import { useEditor } from '../hooks/useEditor';
import { useElementSize } from '../hooks/useElementSize';
import { ARTWORK_DRAG_MIME } from '../painting/artwork';
import {
  type ArtworkPlacement,
  hitTestPlacements,
  maybeExtendCurvedMappingOnCompletedTransform,
  movePlacementByImagePoints,
  rotatePlacementFromImagePoint,
  scalePlacementFromImagePoint,
} from '../painting/artworkPlacement';
import { PaintDocument } from '../painting/paintDocument';
import { getEffectiveFootprint, type PlanarDecalSurface } from '../painting/planarDecal';
import { PlanarPaintRenderer } from '../painting/planarPaintRenderer';
import type { Point2D, Size2D, ViewportTransform } from '../types/editor';
import { PointCloudViewer } from './PointCloudViewer';
import {
  buildBrushCursorPreview,
  buildPlacementGizmoOverlay,
  renderViewportCanvas,
} from './canvasRenderer';
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

type ActivePaintAnchor =
  | {
      mode: 'planar';
      surface: PlanarDecalSurface;
      reusedExisting: boolean;
      surfaceRadius: number;
    }
  | {
      mode: 'curved';
      componentMask: Uint8Array;
      surfaceRadius: number;
    }
  | {
      mode: 'fallback-2d';
      radiusPx: number;
    };

interface PaintSession {
  active: boolean;
  pointerId: number | null;
  anchor: ActivePaintAnchor | null;
  lastImagePoint: Point2D | null;
}

type PlacementDragMode = 'move' | 'scale' | 'rotate';

interface PlacementDragSession {
  active: boolean;
  mode: PlacementDragMode;
  pointerId: number | null;
  initialPlacement: ArtworkPlacement | null;
  previewPlacement: ArtworkPlacement | null;
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
    imageBrushEngineRef,
    paintRendererRef,
    geometryComparison,
    setViewportSize,
    setTransform,
    setViewMode,
    selectSurfaceAtImagePoint,
    commitAutoStroke,
    placeArtworkAtImagePoint,
    setActivePlacement,
    updatePlacement,
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
  const webGlDirtyRef = useRef(true);
  const panSessionRef = useRef<PanSession>({
    active: false,
    pointerId: null,
    lastClientX: 0,
    lastClientY: 0,
  });
  const paintSessionRef = useRef<PaintSession>({
    active: false,
    pointerId: null,
    anchor: null,
    lastImagePoint: null,
  });
  const placementDragRef = useRef<PlacementDragSession>({
    active: false,
    mode: 'move',
    pointerId: null,
    initialPlacement: null,
    previewPlacement: null,
    lastImagePoint: null,
  });

  const latestRenderStateRef = useRef({
    image: state.image,
    workspaceMode: state.workspaceMode,
    viewportSize: state.viewportSize,
    dpr,
    viewMode: state.viewMode,
    activeTool: state.activeTool,
    geometry: state.geometry.data,
    depthDifference: geometryComparison?.preview,
    appearance: state.appearance.data,
    paint: state.paint,
    isPanning,
    isSpacePressed,
  });

  latestRenderStateRef.current = {
    image: state.image,
    workspaceMode: state.workspaceMode,
    viewportSize: state.viewportSize,
    dpr,
    viewMode: state.viewMode,
    activeTool: state.activeTool,
    geometry: state.geometry.data,
    depthDifference: geometryComparison?.preview,
    appearance: state.appearance.data,
    paint: state.paint,
    isPanning,
    isSpacePressed,
  };

  const handleViewportResize = useCallback(
    (size: Size2D) => {
      if (size.width > 0 && size.height > 0) setViewportSize(size);
    },
    [setViewportSize],
  );
  useElementSize(containerRef, handleViewportResize);

  const drawCanvasNow = useCallback(() => {
    if (latestRenderStateRef.current.workspaceMode !== 'photo') return;
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
      depthDifference,
      appearance,
      paint,
      isPanning: currentIsPanning,
      isSpacePressed: currentSpacePressed,
    } = latestRenderStateRef.current;
    if (viewMode === 'pointcloud' || viewMode === 'mesh') {
      return;
    }

    const paintRenderer = paintRendererRef.current;
    const usePaintWebGl = viewMode === 'rgb' && image !== null && paintRenderer !== null;
    const activeAnchor = paintSessionRef.current.active ? paintSessionRef.current.anchor : null;
    const previewSurfaces =
      activeAnchor && activeAnchor.mode === 'planar' && !activeAnchor.reusedExisting
        ? [activeAnchor.surface]
        : undefined;
    const livePreviewPlacement = placementDragRef.current.active
      ? placementDragRef.current.previewPlacement
      : null;

    if (viewMode === 'rgb' && image !== null && paintRenderer !== null && webGlDirtyRef.current) {
      webGlDirtyRef.current = false;
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
        previewSurfaces,
        activeImagePreview: imageBrushEngineRef.current.getActivePreview(),
        previewPlacement: livePreviewPlacement,
      });
    }

    const activePlacement = livePreviewPlacement
      ? livePreviewPlacement
      : paint.activePlacementId
        ? (paint.artworkPlacements.find((p) => p.id === paint.activePlacementId) ?? null)
        : null;
    const placementGizmo =
      image &&
      viewMode === 'rgb' &&
      activeTool === 'pointer' &&
      !paint.cleanPreview &&
      activePlacement
        ? buildPlacementGizmoOverlay({
            placement: activePlacement,
            imageSize: { width: image.metadata.width, height: image.metadata.height },
            transform: liveTransformRef.current,
            geometry,
          })
        : null;

    const brushCursor =
      image &&
      viewMode === 'rgb' &&
      !currentIsPanning &&
      !currentSpacePressed &&
      !paint.cleanPreview &&
      cursorViewportPointRef.current
        ? buildBrushCursorPreview({
            tool: activeTool,
            viewportPoint: cursorViewportPointRef.current,
            imageSize: { width: image.metadata.width, height: image.metadata.height },
            transform: liveTransformRef.current,
            geometry,
            paintDoc: paint,
            activeAnchorMode: activeAnchor?.mode ?? null,
            activePlanarSurface: activeAnchor?.mode === 'planar' ? activeAnchor.surface : null,
            activeSurfaceRadius:
              activeAnchor && activeAnchor.mode !== 'fallback-2d'
                ? activeAnchor.surfaceRadius
                : null,
            activeComponentMask:
              activeAnchor?.mode === 'curved' ? activeAnchor.componentMask : null,
          })
        : null;

    renderViewportCanvas(ctx, {
      viewportSize,
      dpr: currentDpr,
      image,
      transform: liveTransformRef.current,
      viewMode,
      geometry,
      depthDifference,
      appearance,
      borderOnly: usePaintWebGl,
      brushCursor,
      placementGizmo,
    });
  }, [brushEngineRef, imageBrushEngineRef, paintRendererRef]);

  useEffect(() => {
    const host = paintHostRef.current;
    if (!host) {
      return;
    }
    const renderer = new PlanarPaintRenderer(host);
    paintRendererRef.current = renderer;
    webGlDirtyRef.current = true;
    drawCanvasNow();
    return () => {
      renderer.dispose();
      paintRendererRef.current = null;
    };
  }, [drawCanvasNow, paintRendererRef]);

  const scheduleOverlayRender = useCallback(() => {
    if (renderRafRef.current !== null) {
      return;
    }
    renderRafRef.current = window.requestAnimationFrame(() => {
      renderRafRef.current = null;
      drawCanvasNow();
    });
  }, [drawCanvasNow]);

  const scheduleCanvasRender = useCallback(() => {
    webGlDirtyRef.current = true;
    scheduleOverlayRender();
  }, [scheduleOverlayRender]);

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
    if (state.viewMode === 'pointcloud' || state.viewMode === 'mesh') {
      cursorStore.setSample(null);
      return;
    }
    webGlDirtyRef.current = true;
    drawCanvasNow();
  }, [
    state.transform,
    state.workspaceMode,
    state.image,
    state.viewportSize,
    state.viewMode,
    state.activeTool,
    state.geometry.data,
    geometryComparison,
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
      if (viewMode === 'pointcloud' || viewMode === 'mesh') {
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
        imageBrushEngineRef.current.cancelStroke();
        paintSessionRef.current = {
          active: false,
          pointerId: null,
          anchor: null,
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

    if (event.button !== 0 || state.viewMode === 'pointcloud' || state.viewMode === 'mesh') {
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    const viewportPoint = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    cursorViewportPointRef.current = viewportPoint;
    const imagePoint = clientToImage(
      { x: event.clientX, y: event.clientY },
      rect,
      liveTransformRef.current,
    );

    if (state.activeTool === 'pointer' && state.viewMode === 'rgb') {
      const imageSize = {
        width: state.image.metadata.width,
        height: state.image.metadata.height,
      };
      const activePlacement = state.paint.activePlacementId
        ? (state.paint.artworkPlacements.find((p) => p.id === state.paint.activePlacementId) ??
          null)
        : null;

      if (activePlacement && !state.paint.cleanPreview) {
        const gizmo = buildPlacementGizmoOverlay({
          placement: activePlacement,
          imageSize,
          transform: liveTransformRef.current,
          geometry: state.geometry.data,
        });
        if (gizmo) {
          let dragMode: PlacementDragMode | null = null;
          if (
            Math.hypot(
              viewportPoint.x - gizmo.rotationHandle.x,
              viewportPoint.y - gizmo.rotationHandle.y,
            ) <= 13
          ) {
            dragMode = 'rotate';
          } else {
            for (let i = 0; i < gizmo.corners.length; i++) {
              const corner = gizmo.corners[i];
              if (
                corner &&
                Math.hypot(viewportPoint.x - corner.x, viewportPoint.y - corner.y) <= 11
              ) {
                dragMode = 'scale';
                break;
              }
            }
          }
          if (!dragMode) {
            const directHit = hitTestPlacements(
              [activePlacement],
              imagePoint,
              imageSize,
              state.geometry.data,
              state.paint.artworks,
            );
            if (directHit) {
              dragMode = 'move';
            }
          }
          if (dragMode) {
            event.preventDefault();
            event.currentTarget.setPointerCapture(event.pointerId);
            placementDragRef.current = {
              active: true,
              mode: dragMode,
              pointerId: event.pointerId,
              initialPlacement: activePlacement,
              previewPlacement: activePlacement,
              lastImagePoint: imagePoint,
            };
            scheduleCanvasRender();
            return;
          }
        }
      }

      const picked = hitTestPlacements(
        PaintDocument.getOrderedPlacementsBottomToTop(state.paint),
        imagePoint,
        imageSize,
        state.geometry.data,
        state.paint.artworks,
      );
      if (picked) {
        setActivePlacement(picked.id);
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        placementDragRef.current = {
          active: true,
          mode: 'move',
          pointerId: event.pointerId,
          initialPlacement: picked,
          previewPlacement: picked,
          lastImagePoint: imagePoint,
        };
        scheduleCanvasRender();
        return;
      }

      if (state.paint.activePlacementId !== null) {
        setActivePlacement(null);
      }
      return;
    }

    if (state.activeTool === 'select-surface') {
      if (state.geometry.data) {
        event.preventDefault();
        void selectSurfaceAtImagePoint(imagePoint);
      }
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
      const strokeMode = state.activeTool === 'eraser' ? 'erase' : 'paint';
      const anchorResult = classifyAndAnchorSurface({
        geometry: state.geometry.data,
        imagePoint,
        imageSize,
        brushPixelDiameter: state.paint.brushConfig.pixelSize,
        existingSurfaces: state.paint.surfaces,
        nextSurfaceNumber: state.paint.nextSurfaceNumber,
      });

      const brushConfig = state.paint.brushConfig;
      let activeAnchor: ActivePaintAnchor | null = null;
      let started = false;

      if (anchorResult.mode === 'planar' && state.geometry.data) {
        const intr = parseNormalizedIntrinsics(state.geometry.data.metadata.intrinsics);
        const planeHit = anchorResult.surface.frame.imagePointToPlane(imagePoint, imageSize, intr);
        if (planeHit) {
          const footprint = getEffectiveFootprint(anchorResult.surface);
          const startedStroke = brushEngineRef.current.beginStroke(
            anchorResult.surface.id,
            footprint,
            {
              diameter: anchorResult.surfaceRadius * 2,
              color: brushConfig.color,
              opacity: brushConfig.opacity,
              hardness: brushConfig.hardness,
            },
            { u: planeHit.u, v: planeHit.v },
            strokeMode,
          );
          started = startedStroke !== null;
          if (started) {
            activeAnchor = {
              mode: 'planar',
              surface: anchorResult.surface,
              reusedExisting: anchorResult.reusedExisting,
              surfaceRadius: anchorResult.surfaceRadius,
            };
          }
        }
      } else if (anchorResult.mode === 'curved' && state.geometry.data) {
        started = imageBrushEngineRef.current.beginCurvedStroke({
          imagePoint,
          imageSize,
          geometry: state.geometry.data,
          componentMask: anchorResult.componentMask,
          surfaceRadius: anchorResult.surfaceRadius,
          brush: {
            diameter: brushConfig.pixelSize,
            color: brushConfig.color,
            opacity: brushConfig.opacity,
            hardness: brushConfig.hardness,
          },
          mode: strokeMode,
        });
        if (started) {
          activeAnchor = {
            mode: 'curved',
            componentMask: anchorResult.componentMask,
            surfaceRadius: anchorResult.surfaceRadius,
          };
        }
      } else {
        const radiusPx =
          anchorResult.mode === 'fallback-2d' ? anchorResult.radiusPx : brushConfig.pixelSize * 0.5;
        started = imageBrushEngineRef.current.beginFallback2dStroke({
          imagePoint,
          imageSize,
          pixelDiameter: radiusPx * 2,
          brush: {
            diameter: brushConfig.pixelSize,
            color: brushConfig.color,
            opacity: brushConfig.opacity,
            hardness: brushConfig.hardness,
          },
          mode: strokeMode,
        });
        if (started) {
          activeAnchor = {
            mode: 'fallback-2d',
            radiusPx,
          };
        }
      }

      if (!started || !activeAnchor) {
        return;
      }

      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      paintSessionRef.current = {
        active: true,
        pointerId: event.pointerId,
        anchor: activeAnchor,
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

    const placementDrag = placementDragRef.current;
    if (placementDrag.active && placementDrag.pointerId === event.pointerId) {
      const { image, geometry } = latestRenderStateRef.current;
      const basePlacement = placementDrag.previewPlacement ?? placementDrag.initialPlacement;
      if (image && basePlacement) {
        const imageSize = {
          width: image.metadata.width,
          height: image.metadata.height,
        };
        const imagePoint = clientToImage(
          { x: event.clientX, y: event.clientY },
          rect,
          liveTransformRef.current,
        );
        const prevPt = placementDrag.lastImagePoint ?? imagePoint;
        let nextPlacement = basePlacement;
        if (placementDrag.mode === 'move') {
          nextPlacement = movePlacementByImagePoints(
            basePlacement,
            prevPt,
            imagePoint,
            imageSize,
            geometry,
          );
        } else if (placementDrag.mode === 'scale') {
          nextPlacement = scalePlacementFromImagePoint(
            basePlacement,
            imagePoint,
            imageSize,
            geometry,
          );
        } else {
          nextPlacement = rotatePlacementFromImagePoint(
            basePlacement,
            imagePoint,
            imageSize,
            geometry,
          );
        }
        placementDrag.previewPlacement = nextPlacement;
        placementDrag.lastImagePoint = imagePoint;
        scheduleCanvasRender();
      }
      updateCursorSample(event.clientX, event.clientY);
      return;
    }

    const paintSession = paintSessionRef.current;
    if (paintSession.active && paintSession.pointerId === event.pointerId) {
      const { image, geometry } = latestRenderStateRef.current;
      const anchor = paintSession.anchor;
      if (image && anchor) {
        const imageSize = {
          width: image.metadata.width,
          height: image.metadata.height,
        };
        const imagePoint = clientToImage(
          { x: event.clientX, y: event.clientY },
          rect,
          liveTransformRef.current,
        );

        if (anchor.mode === 'planar' && geometry) {
          const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
          const hit = anchor.surface.frame.imagePointToPlane(imagePoint, imageSize, intr);
          if (hit) {
            brushEngineRef.current.extendStroke({ u: hit.u, v: hit.v });
          }
        } else if (anchor.mode === 'curved' || anchor.mode === 'fallback-2d') {
          imageBrushEngineRef.current.extendStroke(imagePoint);
        }

        paintSession.lastImagePoint = imagePoint;
        scheduleCanvasRender();
      }
    } else {
      const { activeTool, viewMode } = latestRenderStateRef.current;
      if (viewMode === 'rgb' && (activeTool === 'brush' || activeTool === 'eraser')) {
        scheduleOverlayRender();
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

    const placementDrag = placementDragRef.current;
    if (placementDrag.active && placementDrag.pointerId === event.pointerId) {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      const { initialPlacement, previewPlacement, mode: dragMode } = placementDrag;
      placementDragRef.current = {
        active: false,
        mode: 'move',
        pointerId: null,
        initialPlacement: null,
        previewPlacement: null,
        lastImagePoint: null,
      };
      if (!cancelled && previewPlacement && previewPlacement !== initialPlacement) {
        const { image, geometry } = latestRenderStateRef.current;
        const finalPlacement =
          image && (dragMode === 'move' || dragMode === 'scale')
            ? maybeExtendCurvedMappingOnCompletedTransform(
                previewPlacement,
                { width: image.metadata.width, height: image.metadata.height },
                geometry,
              )
            : previewPlacement;
        updatePlacement(finalPlacement, true);
      } else {
        scheduleCanvasRender();
      }
    }

    const paintSession = paintSessionRef.current;
    if (paintSession.active && paintSession.pointerId === event.pointerId) {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      const finishedAnchor = paintSession.anchor;
      paintSessionRef.current = {
        active: false,
        pointerId: null,
        anchor: null,
        lastImagePoint: null,
      };
      if (cancelled || !finishedAnchor) {
        brushEngineRef.current.cancelStroke();
        imageBrushEngineRef.current.cancelStroke();
        scheduleCanvasRender();
      } else if (finishedAnchor.mode === 'planar') {
        const batch = brushEngineRef.current.commitBatch();
        const planarStroke = batch[0] ?? null;
        if (planarStroke) {
          commitAutoStroke({
            anchorMode: 'planar',
            newPlanarSurface: finishedAnchor.reusedExisting ? null : finishedAnchor.surface,
            planarStroke,
          });
        } else {
          scheduleCanvasRender();
        }
      } else {
        const imageStroke = imageBrushEngineRef.current.commitStroke();
        if (imageStroke) {
          commitAutoStroke({
            anchorMode: finishedAnchor.mode,
            imageStroke,
          });
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
      scheduleOverlayRender();
    }
  };

  const isPointCloudMode =
    (state.viewMode === 'pointcloud' || state.viewMode === 'mesh') && state.geometry.data !== null;
  const isPaintWebGlVisible = !isPointCloudMode && state.viewMode === 'rgb' && state.image !== null;

  const cursorModeClass =
    !state.image || isPointCloudMode
      ? 'cursor-default'
      : isPanning
        ? 'cursor-grabbing'
        : isSpacePressed || state.activeTool === 'hand'
          ? 'cursor-grab'
          : (state.activeTool === 'brush' || state.activeTool === 'eraser') &&
              state.viewMode === 'rgb'
            ? 'cursor-brush'
            : state.activeTool === 'select-surface'
              ? 'cursor-crosshair'
              : 'cursor-default';

  const handleMainDragOver = (event: React.DragEvent<HTMLElement>) => {
    if (state.image && Array.from(event.dataTransfer.types).includes(ARTWORK_DRAG_MIME)) {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'copy';
      return;
    }
    dropZoneProps.onDragOver(event);
  };

  const handleMainDrop = (event: React.DragEvent<HTMLElement>) => {
    if (Array.from(event.dataTransfer.types).includes(ARTWORK_DRAG_MIME)) {
      event.preventDefault();
      event.stopPropagation();
      const artworkId = event.dataTransfer.getData(ARTWORK_DRAG_MIME);
      const canvas = canvasRef.current;
      if (artworkId && canvas && state.image) {
        const rect = canvas.getBoundingClientRect();
        const imagePoint = clientToImage(
          { x: event.clientX, y: event.clientY },
          rect,
          liveTransformRef.current,
        );
        placeArtworkAtImagePoint(artworkId, imagePoint);
      }
      return;
    }
    dropZoneProps.onDrop(event);
  };

  return (
    <main
      ref={containerRef}
      style={state.workspaceMode === 'artwork' ? { display: 'none' } : undefined}
      className={`editor-viewport ${cursorModeClass}`}
      aria-label="Image editor viewport"
      onDragEnter={dropZoneProps.onDragEnter}
      onDragOver={handleMainDragOver}
      onDragLeave={dropZoneProps.onDragLeave}
      onDrop={handleMainDrop}
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
        <PointCloudViewer
          key={state.geometry.data.metadata.id}
          geometry={state.geometry.data}
          mode={state.viewMode === 'mesh' ? 'mesh' : 'points'}
          onModeChange={(nextMode) => setViewMode(nextMode === 'mesh' ? 'mesh' : 'pointcloud')}
        />
      )}

      {!state.image && !state.isLoading && <EmptyViewportState />}

      {state.isLoading && (
        <div className="viewport-loading-overlay" aria-live="polite">
          <div className="loading-pill">Loading image...</div>
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
