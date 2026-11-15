import type {
  GeometryData,
  LoadedImage,
  Size2D,
  ViewMode,
  ViewportTransform,
} from '../types/editor';
import { getImageBoundsInViewport } from './viewportTransforms';

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
  const { viewportSize, dpr, image, transform, viewMode = 'rgb', geometry = null } = params;
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

  ctx.save();
  ctx.setTransform(safeDpr, 0, 0, safeDpr, 0, 0);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.lineWidth = 1 / safeDpr;
  ctx.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);
  ctx.restore();
}
