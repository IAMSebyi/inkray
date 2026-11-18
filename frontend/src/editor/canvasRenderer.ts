import type { AutoAnchorMode } from '../geometry/autoSurfaceAnchor';
import {
  type ParsedIntrinsics,
  parseNormalizedIntrinsics,
  projectOpenCvPointToImage,
} from '../geometry/geometryAccess';
import {
  buildMetricEllipseImagePolygon,
  computeInterpolatedSurfaceMetric,
  computeSurfaceBrushRadius,
} from '../geometry/localSurfaceMetric';
import {
  type SurfaceGridIndex,
  getSurfaceGridIndex,
  pickSurfaceHit,
} from '../geometry/surfacePicking';
import { type ArtworkPlacement, computePlacementImageQuad } from '../painting/artworkPlacement';
import { PaintDocument, type PaintDocumentState } from '../painting/paintDocument';
import type { PlanarDecalSurface } from '../painting/planarDecal';
import { clamp } from './viewportTransforms';
import { findSurfaceOwnerAtImagePoint } from '../painting/planeVisibility';
import type {
  AppearanceData,
  EditorTool,
  GeometryData,
  LoadedImage,
  Point2D,
  Size2D,
  ViewMode,
  ViewportTransform,
} from '../types/editor';
import { clientToImage, getImageBoundsInViewport } from './viewportTransforms';

export interface BrushCursorPreview {
  mode: 'paint' | 'erase';
  viewportPoint: Point2D;
  screenPolygon: Point2D[] | null;
  fallbackRadiusPx: number;
  onSurface: boolean;
}

function mapImagePolygonToViewport(
  polygon: readonly Point2D[],
  transform: ViewportTransform,
  viewportPoint: Point2D,
): Point2D[] | null {
  if (polygon.length < 3) {
    return null;
  }
  const out: Point2D[] = [];
  for (let i = 0; i < polygon.length; i++) {
    const pt = polygon[i];
    if (!pt) {
      return null;
    }
    const sx = pt.x * transform.zoom + transform.panX;
    const sy = pt.y * transform.zoom + transform.panY;
    if (!Number.isFinite(sx) || !Number.isFinite(sy)) {
      return null;
    }
    if (Math.hypot(sx - viewportPoint.x, sy - viewportPoint.y) > 600) {
      return null;
    }
    out.push({ x: sx, y: sy });
  }
  return out;
}

function projectPlanarRingToViewport(
  surface: PlanarDecalSurface,
  centerU: number,
  centerV: number,
  radiusWorld: number,
  intr: ParsedIntrinsics,
  imageSize: Size2D,
  transform: ViewportTransform,
  viewportPoint: Point2D,
): Point2D[] | null {
  const segments = 28;
  const polygon: Point2D[] = [];
  for (let i = 0; i < segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    const worldPt = surface.frame.localToWorld(
      centerU + Math.cos(theta) * radiusWorld,
      centerV + Math.sin(theta) * radiusWorld,
    );
    if (worldPt[2] <= 0.05) {
      return null;
    }
    const imgProj = projectOpenCvPointToImage(worldPt, intr, imageSize);
    if (!imgProj) {
      return null;
    }
    const sx = imgProj.x * transform.zoom + transform.panX;
    const sy = imgProj.y * transform.zoom + transform.panY;
    if (Math.hypot(sx - viewportPoint.x, sy - viewportPoint.y) > 600) {
      return null;
    }
    polygon.push({ x: sx, y: sy });
  }
  return polygon;
}

function projectMetricRingToViewport(
  gridIndex: SurfaceGridIndex,
  geometry: GeometryData,
  imagePoint: Point2D,
  radiusWorld: number,
  imageSize: Size2D,
  transform: ViewportTransform,
  viewportPoint: Point2D,
): Point2D[] | null {
  const metric = computeInterpolatedSurfaceMetric(gridIndex, geometry, imagePoint, imageSize);
  if (!metric.valid) {
    return null;
  }
  const imagePolygon = buildMetricEllipseImagePolygon(metric, imagePoint, radiusWorld, 28);
  if (!imagePolygon) {
    return null;
  }
  return mapImagePolygonToViewport(imagePolygon, transform, viewportPoint);
}

export function buildBrushCursorPreview(params: {
  tool: EditorTool;
  viewportPoint: Point2D;
  imageSize: Size2D;
  transform: ViewportTransform;
  geometry: GeometryData | null;
  paintDoc: PaintDocumentState;
  activeAnchorMode?: AutoAnchorMode | null;
  activePlanarSurface?: PlanarDecalSurface | null;
  activeSurfaceRadius?: number | null;
  activeComponentMask?: Uint8Array | null;
}): BrushCursorPreview | null {
  const {
    tool,
    viewportPoint,
    imageSize,
    transform,
    geometry,
    paintDoc,
    activeAnchorMode = null,
    activePlanarSurface = null,
    activeSurfaceRadius = null,
    activeComponentMask = null,
  } = params;

  if ((tool !== 'brush' && tool !== 'eraser') || imageSize.width <= 0 || imageSize.height <= 0) {
    return null;
  }

  const mode: 'paint' | 'erase' = tool === 'eraser' ? 'erase' : 'paint';
  const pixelRadiusPx = clamp(paintDoc.brushConfig.pixelSize * 0.5 * transform.zoom, 3, 240);

  if (!geometry || activeAnchorMode === 'fallback-2d') {
    return {
      mode,
      viewportPoint,
      screenPolygon: null,
      fallbackRadiusPx: pixelRadiusPx,
      onSurface: activeAnchorMode === 'fallback-2d',
    };
  }

  const fakeRect = { left: 0, top: 0, width: 1, height: 1 } as DOMRect;
  const imagePoint = clientToImage(viewportPoint, fakeRect, transform);
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);

  if (activeAnchorMode === 'planar' && activePlanarSurface) {
    const planeHit = activePlanarSurface.frame.imagePointToPlane(imagePoint, imageSize, intr);
    const radiusWorld =
      activeSurfaceRadius ??
      PaintDocument.getEffectiveBrushSettingsForSurface(paintDoc, activePlanarSurface).diameter *
        0.5;
    const approxDepth = planeHit
      ? planeHit.depth
      : Math.max(0.1, Math.abs(activePlanarSurface.frame.origin[2]));
    const fallbackRadiusPx = clamp(
      (radiusWorld / approxDepth) * intr.fx * imageSize.width * transform.zoom,
      3,
      240,
    );
    const owner = findSurfaceOwnerAtImagePoint(
      [activePlanarSurface],
      imagePoint,
      imageSize,
      geometry,
    );
    if (!planeHit || !owner) {
      return {
        mode,
        viewportPoint,
        screenPolygon: null,
        fallbackRadiusPx,
        onSurface: false,
      };
    }
    const poly = projectPlanarRingToViewport(
      activePlanarSurface,
      planeHit.u,
      planeHit.v,
      radiusWorld,
      intr,
      imageSize,
      transform,
      viewportPoint,
    );
    return {
      mode,
      viewportPoint,
      screenPolygon: poly,
      fallbackRadiusPx,
      onSurface: true,
    };
  }

  const surfaceHit = pickSurfaceHit(geometry, imagePoint, imageSize);
  if (!surfaceHit.valid || !surfaceHit.pixel || surfaceHit.depth === null) {
    return {
      mode,
      viewportPoint,
      screenPolygon: null,
      fallbackRadiusPx: pixelRadiusPx,
      onSurface: false,
    };
  }

  const gridIndex = getSurfaceGridIndex(geometry);
  const onActiveComponent =
    !activeComponentMask ||
    activeComponentMask[surfaceHit.pixel.y * gridIndex.width + surfaceHit.pixel.x] === 1;
  const radiusWorld =
    activeSurfaceRadius ??
    computeSurfaceBrushRadius(surfaceHit.depth, paintDoc.brushConfig.pixelSize, intr, imageSize);
  const fallbackRadiusPx = clamp(
    (radiusWorld / Math.max(0.1, surfaceHit.depth)) * intr.fx * imageSize.width * transform.zoom,
    3,
    240,
  );
  const screenPolygon = projectMetricRingToViewport(
    gridIndex,
    geometry,
    imagePoint,
    radiusWorld,
    imageSize,
    transform,
    viewportPoint,
  );

  return {
    mode,
    viewportPoint,
    screenPolygon,
    fallbackRadiusPx,
    onSurface: onActiveComponent,
  };
}

let cachedCheckerPatternCanvas: HTMLCanvasElement | null = null;

export function getCheckerboardTileCanvas(): HTMLCanvasElement {
  if (cachedCheckerPatternCanvas) {
    return cachedCheckerPatternCanvas;
  }

  const size = 16;
  const tile = document.createElement('canvas');
  tile.width = size;
  tile.height = size;
  const tileCtx = tile.getContext('2d');

  if (tileCtx) {
    const half = size / 2;
    tileCtx.fillStyle = '#1b1d23';
    tileCtx.fillRect(0, 0, size, size);
    tileCtx.fillStyle = '#242831';
    tileCtx.fillRect(0, 0, half, half);
    tileCtx.fillRect(half, half, half, half);
  }

  cachedCheckerPatternCanvas = tile;
  return tile;
}

export interface PlacementGizmoOverlay {
  corners: [Point2D, Point2D, Point2D, Point2D];
  outlinePolygon?: Point2D[];
  center: Point2D;
  topMid: Point2D;
  rotationHandle: Point2D;
  isPlanar: boolean;
  isCurved: boolean;
  isApproximatelyPlanar: boolean;
}

export function buildPlacementGizmoOverlay(params: {
  placement: ArtworkPlacement;
  imageSize: Size2D;
  transform: ViewportTransform;
  geometry: GeometryData | null;
}): PlacementGizmoOverlay | null {
  const { placement, imageSize, transform, geometry } = params;
  if (!placement.visible) {
    return null;
  }
  const quad = computePlacementImageQuad(placement, imageSize, geometry);
  if (!quad) {
    return null;
  }
  const toScreen = (pt: Point2D): Point2D => ({
    x: pt.x * transform.zoom + transform.panX,
    y: pt.y * transform.zoom + transform.panY,
  });
  const corners: [Point2D, Point2D, Point2D, Point2D] = [
    toScreen(quad.corners[0]),
    toScreen(quad.corners[1]),
    toScreen(quad.corners[2]),
    toScreen(quad.corners[3]),
  ];
  const outlinePolygon = quad.outlinePolygon?.map(toScreen);
  const center = toScreen(quad.center);
  const topMid = toScreen(quad.topMid);
  const dx = topMid.x - center.x;
  const dy = topMid.y - center.y;
  const len = Math.hypot(dx, dy);
  const ux = len > 1e-3 ? dx / len : 0;
  const uy = len > 1e-3 ? dy / len : -1;
  const rotationHandle: Point2D = {
    x: topMid.x + ux * 24,
    y: topMid.y + uy * 24,
  };
  return {
    corners,
    outlinePolygon,
    center,
    topMid,
    rotationHandle,
    isPlanar: placement.mapping.type === 'planar',
    isCurved: placement.mapping.type === 'curved',
    isApproximatelyPlanar:
      placement.mapping.type === 'planar' && placement.mapping.isApproximatelyPlanar,
  };
}

export interface CanvasRenderParams {
  viewportSize: Size2D;
  dpr: number;
  image: LoadedImage | null;
  transform: ViewportTransform;
  viewMode?: Exclude<ViewMode, 'pointcloud' | 'mesh'>;
  geometry?: GeometryData | null;
  depthDifference?: HTMLCanvasElement | null;
  appearance?: AppearanceData | null;
  borderOnly?: boolean;
  brushCursor?: BrushCursorPreview | null;
  placementGizmo?: PlacementGizmoOverlay | null;
}

export function syncCanvasBackingSize(
  canvas: HTMLCanvasElement,
  viewportSize: Size2D,
  dpr: number,
): Size2D {
  const safeDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  const targetWidth = Math.max(1, Math.round(viewportSize.width * safeDpr));
  const targetHeight = Math.max(1, Math.round(viewportSize.height * safeDpr));

  // Avoid reallocating the canvas backing buffer when dimensions have not changed.
  if (canvas.width !== targetWidth) {
    canvas.width = targetWidth;
  }
  if (canvas.height !== targetHeight) {
    canvas.height = targetHeight;
  }

  return { width: targetWidth, height: targetHeight };
}

export function renderViewportCanvas(
  ctx: CanvasRenderingContext2D,
  params: CanvasRenderParams,
): void {
  const {
    viewportSize,
    dpr,
    image,
    transform,
    viewMode = 'rgb',
    geometry = null,
    depthDifference = null,
    appearance = null,
    borderOnly = false,
    brushCursor = null,
    placementGizmo = null,
  } = params;
  const safeDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  const backingSize = syncCanvasBackingSize(ctx.canvas, viewportSize, safeDpr);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, backingSize.width, backingSize.height);

  if (!image || viewportSize.width <= 0 || viewportSize.height <= 0) {
    return;
  }

  let drawSource: HTMLImageElement | HTMLCanvasElement = image.element;
  let showCheckerboard = image.metadata.formatLabel === 'PNG';

  if (viewMode === 'depth-difference' && depthDifference) {
    drawSource = depthDifference;
    showCheckerboard = true;
  }

  if (geometry) {
    if (viewMode === 'depth') {
      drawSource = geometry.previews.depth;
      showCheckerboard = true;
    } else if (viewMode === 'normals' && geometry.previews.normals) {
      drawSource = geometry.previews.normals;
      showCheckerboard = true;
    } else if (viewMode === 'mask') {
      drawSource = geometry.previews.mask;
      showCheckerboard = false;
    }
  }

  if (appearance) {
    if (viewMode === 'albedo') {
      drawSource = appearance.previews.albedo;
      showCheckerboard = false;
    } else if (viewMode === 'shading') {
      drawSource = appearance.previews.shading ?? drawSource;
      showCheckerboard = false;
    } else if (viewMode === 'residual') {
      drawSource = appearance.previews.residual ?? drawSource;
      showCheckerboard = false;
    }
  }

  const imageSize: Size2D = {
    width: image.metadata.width,
    height: image.metadata.height,
  };
  const bounds = getImageBoundsInViewport(imageSize, transform);

  const isVisible =
    bounds.x + bounds.width >= -2 &&
    bounds.y + bounds.height >= -2 &&
    bounds.x <= viewportSize.width + 2 &&
    bounds.y <= viewportSize.height + 2;

  if (!isVisible) {
    return;
  }

  if (!borderOnly) {
    // Screen-space checkerboard behind transparent pixels so tiles stay crisp across zoom levels.
    if (showCheckerboard) {
      ctx.save();
      ctx.setTransform(safeDpr, 0, 0, safeDpr, 0, 0);
      const pattern = ctx.createPattern(getCheckerboardTileCanvas(), 'repeat');
      if (pattern) {
        ctx.fillStyle = pattern;
        ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
      }
      ctx.restore();
    }

    ctx.save();
    ctx.setTransform(
      safeDpr * transform.zoom,
      0,
      0,
      safeDpr * transform.zoom,
      safeDpr * transform.panX,
      safeDpr * transform.panY,
    );

    // Switch to nearest-neighbor when zoomed in far enough to inspect individual pixels.
    if (viewMode === 'depth-difference' || transform.zoom >= 3) {
      ctx.imageSmoothingEnabled = false;
    } else {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
    }

    ctx.drawImage(drawSource, 0, 0, image.metadata.width, image.metadata.height);
    ctx.restore();
  }

  ctx.save();
  ctx.setTransform(safeDpr, 0, 0, safeDpr, 0, 0);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.lineWidth = 1 / safeDpr;
  ctx.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);

  if (placementGizmo) {
    const accentColor = placementGizmo.isCurved
      ? '#38bdf8'
      : placementGizmo.isApproximatelyPlanar
        ? '#f59e0b'
        : placementGizmo.isPlanar
          ? '#4c8dff'
          : '#34d399';
    ctx.save();
    if (!placementGizmo.isPlanar && !placementGizmo.isCurved) {
      ctx.setLineDash([5, 4]);
    }
    ctx.beginPath();
    const outline =
      placementGizmo.outlinePolygon && placementGizmo.outlinePolygon.length >= 4
        ? placementGizmo.outlinePolygon
        : placementGizmo.corners;
    const firstPt = outline[0];
    if (firstPt) {
      ctx.moveTo(firstPt.x, firstPt.y);
      for (let i = 1; i < outline.length; i++) {
        const pt = outline[i];
        if (pt) ctx.lineTo(pt.x, pt.y);
      }
      ctx.closePath();
    }
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.72)';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);

    // Rotation stem and knob outside the projected top edge.
    ctx.beginPath();
    ctx.moveTo(placementGizmo.topMid.x, placementGizmo.topMid.y);
    ctx.lineTo(placementGizmo.rotationHandle.x, placementGizmo.rotationHandle.y);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.72)';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1.25;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(placementGizmo.rotationHandle.x, placementGizmo.rotationHandle.y, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = accentColor;
    ctx.fill();
    ctx.strokeStyle = '#0d1016';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    for (let i = 0; i < placementGizmo.corners.length; i++) {
      const corner = placementGizmo.corners[i];
      if (!corner) continue;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(corner.x - 4, corner.y - 4, 8, 8);
      ctx.strokeStyle = accentColor;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(corner.x - 4, corner.y - 4, 8, 8);
    }

    ctx.beginPath();
    ctx.arc(placementGizmo.center.x, placementGizmo.center.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1.25;
    ctx.stroke();
    ctx.restore();
  }

  if (brushCursor) {
    const buildCursorPath = () => {
      ctx.beginPath();
      const poly = brushCursor.screenPolygon;
      if (poly && poly.length >= 3) {
        const first = poly[0];
        if (first) {
          ctx.moveTo(first.x, first.y);
          for (let i = 1; i < poly.length; i++) {
            const pt = poly[i];
            if (pt) ctx.lineTo(pt.x, pt.y);
          }
          ctx.closePath();
        }
      } else {
        ctx.arc(
          brushCursor.viewportPoint.x,
          brushCursor.viewportPoint.y,
          brushCursor.fallbackRadiusPx,
          0,
          Math.PI * 2,
        );
      }
    };

    if (!brushCursor.onSurface) {
      ctx.setLineDash([4, 4]);
    }
    buildCursorPath();
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.72)';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    buildCursorPath();
    ctx.strokeStyle =
      brushCursor.mode === 'erase' ? 'rgba(255, 196, 140, 0.96)' : 'rgba(255, 255, 255, 0.96)';
    ctx.lineWidth = 1.25;
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(brushCursor.viewportPoint.x, brushCursor.viewportPoint.y, 1.5, 0, Math.PI * 2);
    ctx.fillStyle =
      brushCursor.mode === 'erase' ? 'rgba(255, 196, 140, 0.96)' : 'rgba(255, 255, 255, 0.96)';
    ctx.fill();
  }

  ctx.restore();
}
