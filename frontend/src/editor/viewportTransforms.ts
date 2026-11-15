import type {
  ImageCoordinateSample,
  Point2D,
  Size2D,
  ViewportRect,
  ViewportTransform,
  ZoomMode,
} from '../types/editor';

export const MIN_ZOOM = 0.02;
export const MAX_ZOOM = 32;
export const DEFAULT_FIT_PADDING = 32;
export const ZOOM_STEP_FACTOR = 1.25;

export const IDENTITY_TRANSFORM: ViewportTransform = {
  zoom: 1,
  panX: 0,
  panY: 0,
};

export interface ZoomBounds {
  minZoom?: number;
  maxZoom?: number;
}

export interface FitTransformOptions extends ZoomBounds {
  padding?: number;
}

export function clamp(value: number, min: number, max: number): number {
  if (min > max) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function assertPositiveZoom(zoom: number): void {
  if (!Number.isFinite(zoom) || zoom <= 0) {
    throw new RangeError(`Viewport zoom must be a finite positive number, received ${zoom}.`);
  }
}

export function getEffectiveZoomBounds(
  viewportSize?: Size2D,
  imageSize?: Size2D,
): { minZoom: number; maxZoom: number } {
  if (
    viewportSize &&
    imageSize &&
    viewportSize.width > 0 &&
    viewportSize.height > 0 &&
    imageSize.width > 0 &&
    imageSize.height > 0
  ) {
    const rawFitZoom = Math.min(
      viewportSize.width / imageSize.width,
      viewportSize.height / imageSize.height,
    );
    // Allow zooming below MIN_ZOOM for ultra-high-res images so Fit can still frame them.
    const adaptiveMin = Math.max(0.002, Math.min(MIN_ZOOM, rawFitZoom * 0.5));
    return {
      minZoom: adaptiveMin,
      maxZoom: MAX_ZOOM,
    };
  }

  return {
    minZoom: MIN_ZOOM,
    maxZoom: MAX_ZOOM,
  };
}

export function clampZoom(zoom: number, bounds?: ZoomBounds): number {
  const minZoom = bounds?.minZoom ?? MIN_ZOOM;
  const maxZoom = bounds?.maxZoom ?? MAX_ZOOM;
  if (!Number.isFinite(zoom) || zoom <= 0) {
    return clamp(1, minZoom, maxZoom);
  }
  return clamp(zoom, minZoom, maxZoom);
}

export function clientToViewport(
  clientPoint: Point2D,
  viewportRect: Pick<ViewportRect, 'left' | 'top'>,
): Point2D {
  return {
    x: clientPoint.x - viewportRect.left,
    y: clientPoint.y - viewportRect.top,
  };
}

export function viewportToClient(
  viewportPoint: Point2D,
  viewportRect: Pick<ViewportRect, 'left' | 'top'>,
): Point2D {
  return {
    x: viewportPoint.x + viewportRect.left,
    y: viewportPoint.y + viewportRect.top,
  };
}

export function imageToViewport(imagePoint: Point2D, transform: ViewportTransform): Point2D {
  assertPositiveZoom(transform.zoom);
  return {
    x: imagePoint.x * transform.zoom + transform.panX,
    y: imagePoint.y * transform.zoom + transform.panY,
  };
}

export function viewportToImage(viewportPoint: Point2D, transform: ViewportTransform): Point2D {
  assertPositiveZoom(transform.zoom);
  return {
    x: (viewportPoint.x - transform.panX) / transform.zoom,
    y: (viewportPoint.y - transform.panY) / transform.zoom,
  };
}

export function clientToImage(
  clientPoint: Point2D,
  viewportRect: Pick<ViewportRect, 'left' | 'top'>,
  transform: ViewportTransform,
): Point2D {
  return viewportToImage(clientToViewport(clientPoint, viewportRect), transform);
}

export function imageToClient(
  imagePoint: Point2D,
  viewportRect: Pick<ViewportRect, 'left' | 'top'>,
  transform: ViewportTransform,
): Point2D {
  return viewportToClient(imageToViewport(imagePoint, transform), viewportRect);
}

export function isPointInsideImage(imagePoint: Point2D, imageSize: Size2D): boolean {
  if (imageSize.width <= 0 || imageSize.height <= 0) {
    return false;
  }
  return (
    imagePoint.x >= 0 &&
    imagePoint.x < imageSize.width &&
    imagePoint.y >= 0 &&
    imagePoint.y < imageSize.height
  );
}

export function imagePointToPixel(imagePoint: Point2D, imageSize: Size2D): Point2D | null {
  if (!isPointInsideImage(imagePoint, imageSize)) {
    return null;
  }
  return {
    x: Math.floor(imagePoint.x),
    y: Math.floor(imagePoint.y),
  };
}

export function sampleImageAtViewportPoint(
  viewportPoint: Point2D,
  transform: ViewportTransform,
  imageSize: Size2D,
): ImageCoordinateSample {
  const imagePoint = viewportToImage(viewportPoint, transform);
  const pixel = imagePointToPixel(imagePoint, imageSize);
  return {
    imagePoint,
    pixel,
    inBounds: pixel !== null,
  };
}

export function sampleImageAtClientPoint(
  clientPoint: Point2D,
  viewportRect: Pick<ViewportRect, 'left' | 'top'>,
  transform: ViewportTransform,
  imageSize: Size2D,
): ImageCoordinateSample {
  return sampleImageAtViewportPoint(
    clientToViewport(clientPoint, viewportRect),
    transform,
    imageSize,
  );
}

export function computeCenteredTransform(
  viewportSize: Size2D,
  imageSize: Size2D,
  zoom: number,
): ViewportTransform {
  assertPositiveZoom(zoom);
  return {
    zoom,
    panX: (viewportSize.width - imageSize.width * zoom) / 2,
    panY: (viewportSize.height - imageSize.height * zoom) / 2,
  };
}

export function computeFitTransform(
  viewportSize: Size2D,
  imageSize: Size2D,
  options?: FitTransformOptions,
): ViewportTransform {
  if (
    viewportSize.width <= 0 ||
    viewportSize.height <= 0 ||
    imageSize.width <= 0 ||
    imageSize.height <= 0
  ) {
    return IDENTITY_TRANSFORM;
  }

  const requestedPadding = Math.max(0, options?.padding ?? DEFAULT_FIT_PADDING);
  const padX = viewportSize.width > requestedPadding * 2.5 ? requestedPadding : 0;
  const padY = viewportSize.height > requestedPadding * 2.5 ? requestedPadding : 0;

  const availableWidth = Math.max(1, viewportSize.width - padX * 2);
  const availableHeight = Math.max(1, viewportSize.height - padY * 2);

  const rawZoom = Math.min(availableWidth / imageSize.width, availableHeight / imageSize.height);

  const effectiveBounds = getEffectiveZoomBounds(viewportSize, imageSize);
  const zoom = clampZoom(rawZoom, {
    minZoom: options?.minZoom ?? effectiveBounds.minZoom,
    maxZoom: options?.maxZoom ?? effectiveBounds.maxZoom,
  });

  return computeCenteredTransform(viewportSize, imageSize, zoom);
}

/**
 * Scales to targetZoom while keeping the image point under anchorViewportPoint fixed.
 */
export function zoomAtPoint(
  transform: ViewportTransform,
  anchorViewportPoint: Point2D,
  targetZoom: number,
  bounds?: ZoomBounds,
): ViewportTransform {
  assertPositiveZoom(transform.zoom);
  const nextZoom = clampZoom(targetZoom, bounds);

  if (nextZoom === transform.zoom) {
    return transform;
  }

  const imageAnchor = viewportToImage(anchorViewportPoint, transform);
  return {
    zoom: nextZoom,
    panX: anchorViewportPoint.x - imageAnchor.x * nextZoom,
    panY: anchorViewportPoint.y - imageAnchor.y * nextZoom,
  };
}

export function computeActualSizeTransform(
  viewportSize: Size2D,
  imageSize: Size2D,
  currentTransform?: ViewportTransform,
): ViewportTransform {
  if (
    viewportSize.width <= 0 ||
    viewportSize.height <= 0 ||
    imageSize.width <= 0 ||
    imageSize.height <= 0
  ) {
    return IDENTITY_TRANSFORM;
  }

  if (!currentTransform) {
    return computeCenteredTransform(viewportSize, imageSize, 1);
  }

  const viewportCenter: Point2D = {
    x: viewportSize.width / 2,
    y: viewportSize.height / 2,
  };
  const zoomed = zoomAtPoint(currentTransform, viewportCenter, 1);
  return clampPanToViewport(zoomed, viewportSize, imageSize);
}

export function panTransform(
  transform: ViewportTransform,
  deltaX: number,
  deltaY: number,
): ViewportTransform {
  return {
    zoom: transform.zoom,
    panX: transform.panX + deltaX,
    panY: transform.panY + deltaY,
  };
}

/**
 * Keeps at least minVisibleMargin pixels of the image inside the viewport so panning
 * cannot lose the canvas off-screen.
 */
export function clampPanToViewport(
  transform: ViewportTransform,
  viewportSize: Size2D,
  imageSize: Size2D,
  minVisibleMargin = 40,
): ViewportTransform {
  if (
    viewportSize.width <= 0 ||
    viewportSize.height <= 0 ||
    imageSize.width <= 0 ||
    imageSize.height <= 0
  ) {
    return transform;
  }

  const screenWidth = imageSize.width * transform.zoom;
  const screenHeight = imageSize.height * transform.zoom;

  const marginX = Math.min(minVisibleMargin, screenWidth / 2, viewportSize.width / 2);
  const marginY = Math.min(minVisibleMargin, screenHeight / 2, viewportSize.height / 2);

  const minPanX = marginX - screenWidth;
  const maxPanX = viewportSize.width - marginX;
  const minPanY = marginY - screenHeight;
  const maxPanY = viewportSize.height - marginY;

  return {
    zoom: transform.zoom,
    panX: clamp(transform.panX, minPanX, maxPanX),
    panY: clamp(transform.panY, minPanY, maxPanY),
  };
}

export function reconcileTransformOnResize(
  transform: ViewportTransform,
  prevViewportSize: Size2D,
  nextViewportSize: Size2D,
  imageSize: Size2D,
  zoomMode: ZoomMode,
): ViewportTransform {
  if (nextViewportSize.width <= 0 || nextViewportSize.height <= 0) {
    return transform;
  }

  if (zoomMode === 'fit' || prevViewportSize.width <= 0 || prevViewportSize.height <= 0) {
    return computeFitTransform(nextViewportSize, imageSize);
  }

  // Keep the image point at the viewport center anchored when resizing in custom zoom mode.
  return {
    zoom: transform.zoom,
    panX: transform.panX + (nextViewportSize.width - prevViewportSize.width) / 2,
    panY: transform.panY + (nextViewportSize.height - prevViewportSize.height) / 2,
  };
}

export function computeWheelZoomTarget(
  currentZoom: number,
  deltaY: number,
  deltaMode: number,
  isPinchGesture: boolean,
  viewportHeight: number,
  bounds?: ZoomBounds,
): number {
  let pixelDelta = deltaY;
  if (deltaMode === 1) {
    pixelDelta *= 16;
  } else if (deltaMode === 2) {
    pixelDelta *= Math.max(100, viewportHeight);
  }

  // Bound wheel momentum spikes so zoom steps stay controllable.
  const clampedDelta = clamp(pixelDelta, -250, 250);
  const sensitivity = isPinchGesture ? 0.01 : 0.0018;
  const factor = Math.exp(-clampedDelta * sensitivity);

  return clampZoom(currentZoom * factor, bounds);
}

export function getImageBoundsInViewport(
  imageSize: Size2D,
  transform: ViewportTransform,
): { x: number; y: number; width: number; height: number } {
  const topLeft = imageToViewport({ x: 0, y: 0 }, transform);
  return {
    x: topLeft.x,
    y: topLeft.y,
    width: imageSize.width * transform.zoom,
    height: imageSize.height * transform.zoom,
  };
}
