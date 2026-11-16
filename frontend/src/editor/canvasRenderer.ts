import { parseNormalizedIntrinsics, projectOpenCvPointToImage } from '../geometry/geometryAccess';
import { PaintDocument, type PaintDocumentState } from '../painting/paintDocument';
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

export function buildBrushCursorPreview(params: {
  tool: EditorTool;
  viewportPoint: Point2D;
  imageSize: Size2D;
  transform: ViewportTransform;
  geometry: GeometryData | null;
  paintDoc: PaintDocumentState;
}): BrushCursorPreview | null {
  const { tool, viewportPoint, imageSize, transform, geometry, paintDoc } = params;
  if ((tool !== 'brush' && tool !== 'eraser') || !geometry || paintDoc.surfaces.length === 0) {
    return null;
  }

  const fakeRect = { left: 0, top: 0, width: 1, height: 1 } as DOMRect;
  const imagePoint = clientToImage(viewportPoint, fakeRect, transform);
  const owner = findSurfaceOwnerAtImagePoint(paintDoc.surfaces, imagePoint, imageSize, geometry);
  const refSurface =
    owner ?? PaintDocument.getActiveSurface(paintDoc) ?? paintDoc.surfaces[0] ?? null;
  if (!refSurface) {
    return null;
  }

  const mode: 'paint' | 'erase' = tool === 'eraser' ? 'erase' : 'paint';
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const brush = PaintDocument.getEffectiveBrushSettingsForSurface(paintDoc, refSurface);
  const radius = brush.diameter * 0.5;
  const hit = refSurface.frame.imagePointToPlane(imagePoint, imageSize, intr);
  const approxDepth = hit ? hit.depth : Math.max(0.1, Math.abs(refSurface.frame.origin[2]));
  const fallbackRadiusPx = clamp(
    (radius / approxDepth) * intr.fx * imageSize.width * transform.zoom,
    3,
    240,
  );

  if (!hit || !owner) {
    return {
      mode,
      viewportPoint,
      screenPolygon: null,
      fallbackRadiusPx,
      onSurface: owner !== null,
    };
  }

  const segments = 28;
  const polygon: Point2D[] = [];
  for (let i = 0; i < segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    const worldPt = refSurface.frame.localToWorld(
      hit.u + Math.cos(theta) * radius,
      hit.v + Math.sin(theta) * radius,
    );
    if (worldPt[2] <= 0.05) {
      return {
        mode,
        viewportPoint,
        screenPolygon: null,
        fallbackRadiusPx,
        onSurface: true,
      };
    }
    const imgProj = projectOpenCvPointToImage(worldPt, intr, imageSize);
    if (!imgProj) {
      return {
        mode,
        viewportPoint,
        screenPolygon: null,
        fallbackRadiusPx,
        onSurface: true,
      };
    }
    const sx = imgProj.x * transform.zoom + transform.panX;
    const sy = imgProj.y * transform.zoom + transform.panY;
    if (Math.hypot(sx - viewportPoint.x, sy - viewportPoint.y) > 600) {
      return {
        mode,
        viewportPoint,
        screenPolygon: null,
        fallbackRadiusPx,
        onSurface: true,
      };
    }
    polygon.push({ x: sx, y: sy });
  }

  return {
    mode,
    viewportPoint,
    screenPolygon: polygon,
    fallbackRadiusPx,
    onSurface: true,
  };
}

let cachedCheckerPatternCanvas: HTMLCanvasElement | null = null;

function getCheckerboardTileCanvas(): HTMLCanvasElement {
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

export interface CanvasRenderParams {
  viewportSize: Size2D;
  dpr: number;
  image: LoadedImage | null;
  transform: ViewportTransform;
  viewMode?: Exclude<ViewMode, 'pointcloud'>;
  geometry?: GeometryData | null;
  appearance?: AppearanceData | null;
  borderOnly?: boolean;
  brushCursor?: BrushCursorPreview | null;
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
    appearance = null,
    borderOnly = false,
    brushCursor = null,
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
      drawSource = appearance.previews.shading;
      showCheckerboard = false;
    } else if (viewMode === 'residual') {
      drawSource = appearance.previews.residual;
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
    if (transform.zoom >= 3) {
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
