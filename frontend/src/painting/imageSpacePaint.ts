import type { Point2D, Size2D } from '../types/editor';
import { clamp } from '../editor/viewportTransforms';
import { blendSrgbBytes } from './colorSpace';
import type { GeometryInput } from '../geometry/geometryAccess';
import {
  CurvedBrushWorkspace,
  type ImageTexelSpec,
  stampCurvedBrushDab,
} from '../geometry/curvedBrushSolver';
import { type SurfaceGridIndex, getSurfaceGridIndex } from '../geometry/surfacePicking';
import {
  type BrushSettings,
  type BrushStrokeMode,
  type DirtyTexelBounds,
  compositeStrokeMaskOverRgba,
  computeBrushFalloff,
  parseHexColorRgb,
  pointToSegmentDistanceSq,
} from './brushEngine';

export type ImagePaintLayerKind = 'curved' | 'fallback-2d';

export interface ImageSpaceStroke {
  id: string;
  layer: ImagePaintLayerKind;
  brushObjectId?: string;
  order?: number;
  mode: BrushStrokeMode;
  brush: BrushSettings;
  layerWidth: number;
  layerHeight: number;
  bounds: DirtyTexelBounds;
  alphaMask: Uint8Array;
}

export interface ActiveImageStrokePreview {
  id: string;
  layer: ImagePaintLayerKind;
  mode: BrushStrokeMode;
  brush: BrushSettings;
  layerWidth: number;
  layerHeight: number;
  alphaMask: Uint8Array;
  revision: number;
}

export function computeImagePaintSpec(imageSize: Size2D, maxEdge = 1536): ImageTexelSpec {
  const w = Math.max(1, Math.round(imageSize.width));
  const h = Math.max(1, Math.round(imageSize.height));
  const longest = Math.max(w, h);
  if (longest <= maxEdge) {
    return { width: w, height: h };
  }
  const scale = maxEdge / longest;
  return {
    width: Math.max(1, Math.round(w * scale)),
    height: Math.max(1, Math.round(h * scale)),
  };
}

export function mergeDirtyBounds(
  a: DirtyTexelBounds | null,
  b: DirtyTexelBounds | null,
): DirtyTexelBounds | null {
  if (!a) return b ? { ...b } : null;
  if (!b) return { ...a };
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  };
}

export function stamp2dSegmentIntoMask(
  strokeMask: Uint8Array,
  spec: ImageTexelSpec,
  imageSize: Size2D,
  aImg: Point2D,
  bImg: Point2D,
  diameterPx: number,
  hardness: number,
): DirtyTexelBounds | null {
  if (
    imageSize.width <= 0 ||
    imageSize.height <= 0 ||
    spec.width <= 0 ||
    spec.height <= 0 ||
    diameterPx <= 0
  ) {
    return null;
  }

  const radiusPx = Math.max(0.5, diameterPx * 0.5);
  const ax = (aImg.x / imageSize.width) * spec.width;
  const ay = (aImg.y / imageSize.height) * spec.height;
  const bx = (bImg.x / imageSize.width) * spec.width;
  const by = (bImg.y / imageSize.height) * spec.height;
  const radiusTex = radiusPx * (spec.width / imageSize.width);
  const radiusSq = radiusTex * radiusTex;

  const minX = clamp(Math.floor(Math.min(ax, bx) - radiusTex) - 1, 0, spec.width - 1);
  const maxX = clamp(Math.ceil(Math.max(ax, bx) + radiusTex) + 1, 0, spec.width - 1);
  const minY = clamp(Math.floor(Math.min(ay, by) - radiusTex) - 1, 0, spec.height - 1);
  const maxY = clamp(Math.ceil(Math.max(ay, by) + radiusTex) + 1, 0, spec.height - 1);

  let dirtyMinX = spec.width;
  let dirtyMaxX = -1;
  let dirtyMinY = spec.height;
  let dirtyMaxY = -1;

  for (let py = minY; py <= maxY; py++) {
    const ty = py + 0.5;
    const rowOffset = py * spec.width;
    for (let px = minX; px <= maxX; px++) {
      const tx = px + 0.5;
      const distSq = pointToSegmentDistanceSq(tx, ty, ax, ay, bx, by);
      if (distSq >= radiusSq) {
        continue;
      }
      const cov = computeBrushFalloff(Math.sqrt(distSq), radiusTex, hardness);
      const nextAlpha = Math.round(clamp(cov, 0, 1) * 255);
      const idx = rowOffset + px;
      const prevAlpha = strokeMask[idx] ?? 0;
      if (nextAlpha > prevAlpha) {
        strokeMask[idx] = nextAlpha;
        if (px < dirtyMinX) dirtyMinX = px;
        if (px > dirtyMaxX) dirtyMaxX = px;
        if (py < dirtyMinY) dirtyMinY = py;
        if (py > dirtyMaxY) dirtyMaxY = py;
      }
    }
  }

  return dirtyMaxX >= dirtyMinX && dirtyMaxY >= dirtyMinY
    ? { minX: dirtyMinX, minY: dirtyMinY, maxX: dirtyMaxX, maxY: dirtyMaxY }
    : null;
}

export function cropMaskToBounds(
  mask: Uint8Array,
  fullWidth: number,
  bounds: DirtyTexelBounds,
): Uint8Array {
  const bw = bounds.maxX - bounds.minX + 1;
  const bh = bounds.maxY - bounds.minY + 1;
  const cropped = new Uint8Array(bw * bh);
  for (let y = 0; y < bh; y++) {
    const srcStart = (bounds.minY + y) * fullWidth + bounds.minX;
    cropped.set(mask.subarray(srcStart, srcStart + bw), y * bw);
  }
  return cropped;
}

export function compositeImageStrokeOverRgba(
  dstRgba: Uint8ClampedArray | Uint8Array,
  layerWidth: number,
  layerHeight: number,
  stroke: ImageSpaceStroke,
): void {
  const [sr, sg, sb] = parseHexColorRgb(stroke.brush.color);
  const opacity = stroke.mode === 'erase' ? 1 : clamp(stroke.brush.opacity, 0, 1);
  if (opacity <= 0) {
    return;
  }

  const bw = stroke.bounds.maxX - stroke.bounds.minX + 1;
  const bh = stroke.bounds.maxY - stroke.bounds.minY + 1;

  // Fast direct blit when layer dimensions match the recorded stroke spec.
  if (stroke.layerWidth === layerWidth && stroke.layerHeight === layerHeight) {
    for (let ly = 0; ly < bh; ly++) {
      const py = stroke.bounds.minY + ly;
      if (py < 0 || py >= layerHeight) continue;
      const srcRow = ly * bw;
      const dstRow = py * layerWidth;
      for (let lx = 0; lx < bw; lx++) {
        const px = stroke.bounds.minX + lx;
        if (px < 0 || px >= layerWidth) continue;
        const maskVal = stroke.alphaMask[srcRow + lx] ?? 0;
        if (maskVal === 0) continue;

        const srcA = (maskVal / 255) * opacity;
        const outBase = (dstRow + px) * 4;
        const dstA = (dstRgba[outBase + 3] ?? 0) / 255;

        if (stroke.mode === 'erase') {
          if (dstA <= 0) continue;
          const outA = dstA * (1 - srcA);
          if (outA <= 1e-6) {
            dstRgba[outBase] = 0;
            dstRgba[outBase + 1] = 0;
            dstRgba[outBase + 2] = 0;
            dstRgba[outBase + 3] = 0;
          } else {
            dstRgba[outBase + 3] = Math.round(clamp(outA, 0, 1) * 255);
          }
          continue;
        }

        const outA = srcA + dstA * (1 - srcA);
        if (outA <= 1e-6) continue;
        const dstR = dstRgba[outBase] ?? 0;
        const dstG = dstRgba[outBase + 1] ?? 0;
        const dstB = dstRgba[outBase + 2] ?? 0;
        const inv = 1 / outA;
        dstRgba[outBase] = blendSrgbBytes(sr, dstR, srcA * inv, dstA * (1 - srcA) * inv);
        dstRgba[outBase + 1] = blendSrgbBytes(sg, dstG, srcA * inv, dstA * (1 - srcA) * inv);
        dstRgba[outBase + 2] = blendSrgbBytes(sb, dstB, srcA * inv, dstA * (1 - srcA) * inv);
        dstRgba[outBase + 3] = Math.round(clamp(outA, 0, 1) * 255);
      }
    }
    return;
  }

  // General nearest remap if container dimensions differ.
  const sx = stroke.layerWidth / Math.max(1, layerWidth);
  const sy = stroke.layerHeight / Math.max(1, layerHeight);
  const fullMask = new Uint8Array(layerWidth * layerHeight);
  for (let py = 0; py < layerHeight; py++) {
    const syIdx = Math.floor((py + 0.5) * sy) - stroke.bounds.minY;
    if (syIdx < 0 || syIdx >= bh) continue;
    for (let px = 0; px < layerWidth; px++) {
      const sxIdx = Math.floor((px + 0.5) * sx) - stroke.bounds.minX;
      if (sxIdx < 0 || sxIdx >= bw) continue;
      fullMask[py * layerWidth + px] = stroke.alphaMask[syIdx * bw + sxIdx] ?? 0;
    }
  }
  compositeStrokeMaskOverRgba(dstRgba, fullMask, stroke.brush, layerWidth, undefined, stroke.mode);
}

export function rebuildImageLayerRgba(
  strokes: readonly ImageSpaceStroke[],
  spec: ImageTexelSpec,
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(spec.width * spec.height * 4);
  for (let i = 0; i < strokes.length; i++) {
    const stroke = strokes[i];
    if (!stroke) continue;
    compositeImageStrokeOverRgba(rgba, spec.width, spec.height, stroke);
  }
  return rgba;
}

let nextImageStrokeSeq = 1;

interface ActiveSessionState {
  id: string;
  layer: ImagePaintLayerKind;
  mode: BrushStrokeMode;
  brush: BrushSettings;
  spec: ImageTexelSpec;
  imageSize: Size2D;
  alphaMask: Uint8Array;
  bounds: DirtyTexelBounds | null;
  revision: number;
  lastOnSurfacePoint: Point2D | null;
  // Curved-only anchor parameters:
  surfaceRadius: number;
  componentMask: Uint8Array | null;
  geometry: GeometryInput | null;
  gridIndex: SurfaceGridIndex | null;
  // 2D-only step spacing:
  pixelDiameter: number;
}

export class ImageSpaceBrushEngine {
  private readonly workspace = new CurvedBrushWorkspace();
  private session: ActiveSessionState | null = null;

  isActive(): boolean {
    return this.session !== null;
  }

  getActivePreview(): ActiveImageStrokePreview | null {
    const s = this.session;
    if (!s || !s.bounds) {
      return null;
    }
    return {
      id: s.id,
      layer: s.layer,
      mode: s.mode,
      brush: s.brush,
      layerWidth: s.spec.width,
      layerHeight: s.spec.height,
      alphaMask: s.alphaMask,
      revision: s.revision,
    };
  }

  beginCurvedStroke(params: {
    imagePoint: Point2D;
    imageSize: Size2D;
    geometry: GeometryInput;
    componentMask: Uint8Array;
    surfaceRadius: number;
    brush: BrushSettings;
    mode: BrushStrokeMode;
  }): boolean {
    const { imagePoint, imageSize, geometry, componentMask, surfaceRadius, brush, mode } = params;
    const spec = computeImagePaintSpec(imageSize);
    const gridIndex = getSurfaceGridIndex(geometry);
    const hardness = mode === 'erase' ? 1 : clamp(brush.hardness, 0, 1);
    const effectiveBrush: BrushSettings = {
      diameter: Math.max(1, brush.diameter),
      color: brush.color,
      opacity: mode === 'erase' ? 1 : clamp(brush.opacity, 0.01, 1),
      hardness,
    };
    const alphaMask = new Uint8Array(spec.width * spec.height);

    const { bounds } = stampCurvedBrushDab({
      strokeMask: alphaMask,
      texSpec: spec,
      imageSize,
      centerImagePt: imagePoint,
      surfaceRadius,
      hardness,
      geometry,
      gridIndex,
      componentMask,
      workspace: this.workspace,
    });

    this.session = {
      id: `img-stroke-${nextImageStrokeSeq++}`,
      layer: 'curved',
      mode,
      brush: effectiveBrush,
      spec,
      imageSize,
      alphaMask,
      bounds,
      revision: 1,
      lastOnSurfacePoint: { x: imagePoint.x, y: imagePoint.y },
      surfaceRadius,
      componentMask,
      geometry,
      gridIndex,
      pixelDiameter: Math.max(1, brush.diameter),
    };
    return true;
  }

  beginFallback2dStroke(params: {
    imagePoint: Point2D;
    imageSize: Size2D;
    pixelDiameter: number;
    brush: BrushSettings;
    mode: BrushStrokeMode;
  }): boolean {
    const { imagePoint, imageSize, pixelDiameter, brush, mode } = params;
    const spec = computeImagePaintSpec(imageSize);
    const hardness = mode === 'erase' ? 1 : clamp(brush.hardness, 0, 1);
    const effectiveBrush: BrushSettings = {
      diameter: Math.max(1, pixelDiameter),
      color: brush.color,
      opacity: mode === 'erase' ? 1 : clamp(brush.opacity, 0.01, 1),
      hardness,
    };
    const alphaMask = new Uint8Array(spec.width * spec.height);
    const bounds = stamp2dSegmentIntoMask(
      alphaMask,
      spec,
      imageSize,
      imagePoint,
      imagePoint,
      pixelDiameter,
      hardness,
    );

    this.session = {
      id: `img-stroke-${nextImageStrokeSeq++}`,
      layer: 'fallback-2d',
      mode,
      brush: effectiveBrush,
      spec,
      imageSize,
      alphaMask,
      bounds,
      revision: 1,
      lastOnSurfacePoint: { x: imagePoint.x, y: imagePoint.y },
      surfaceRadius: 0,
      componentMask: null,
      geometry: null,
      gridIndex: null,
      pixelDiameter: Math.max(1, pixelDiameter),
    };
    return true;
  }

  private isOnAnchoredComponent(s: ActiveSessionState, imgPt: Point2D): boolean {
    if (!s.componentMask || !s.gridIndex) {
      return false;
    }
    if (
      imgPt.x < 0 ||
      imgPt.x >= s.imageSize.width ||
      imgPt.y < 0 ||
      imgPt.y >= s.imageSize.height
    ) {
      return false;
    }
    const col = clamp(
      Math.floor((imgPt.x * s.gridIndex.width) / s.imageSize.width),
      0,
      s.gridIndex.width - 1,
    );
    const row = clamp(
      Math.floor((imgPt.y * s.gridIndex.height) / s.imageSize.height),
      0,
      s.gridIndex.height - 1,
    );
    return (s.componentMask[row * s.gridIndex.width + col] ?? 0) === 1;
  }

  extendStroke(imagePoint: Point2D): boolean {
    const s = this.session;
    if (!s) {
      return false;
    }

    if (s.layer === 'fallback-2d') {
      const prev = s.lastOnSurfacePoint ?? imagePoint;
      const dist = Math.hypot(imagePoint.x - prev.x, imagePoint.y - prev.y);
      const minStep = Math.max(1.5, s.pixelDiameter * 0.12);
      if (dist < minStep) {
        return false;
      }
      const stepBounds = stamp2dSegmentIntoMask(
        s.alphaMask,
        s.spec,
        s.imageSize,
        prev,
        imagePoint,
        s.pixelDiameter,
        s.brush.hardness,
      );
      s.lastOnSurfacePoint = { x: imagePoint.x, y: imagePoint.y };
      if (stepBounds) {
        s.bounds = mergeDirtyBounds(s.bounds, stepBounds);
        s.revision++;
        return true;
      }
      return false;
    }

    // Curved anchored stroke: validate anchor component at current cursor without jumping gaps.
    if (!s.geometry || !s.gridIndex || !s.componentMask) {
      return false;
    }
    const currValid = this.isOnAnchoredComponent(s, imagePoint);
    if (!currValid) {
      s.lastOnSurfacePoint = null;
      return false;
    }

    const prev = s.lastOnSurfacePoint;
    if (!prev) {
      const { bounds } = stampCurvedBrushDab({
        strokeMask: s.alphaMask,
        texSpec: s.spec,
        imageSize: s.imageSize,
        centerImagePt: imagePoint,
        surfaceRadius: s.surfaceRadius,
        hardness: s.brush.hardness,
        geometry: s.geometry,
        gridIndex: s.gridIndex,
        componentMask: s.componentMask,
        workspace: this.workspace,
      });
      s.lastOnSurfacePoint = { x: imagePoint.x, y: imagePoint.y };
      if (bounds) {
        s.bounds = mergeDirtyBounds(s.bounds, bounds);
        s.revision++;
        return true;
      }
      return false;
    }

    const dist = Math.hypot(imagePoint.x - prev.x, imagePoint.y - prev.y);
    const stepSpacing = Math.max(2, s.pixelDiameter * 0.18);
    if (dist < stepSpacing) {
      return false;
    }

    // Sub-step smoothly only while the segment stays on the anchored component.
    const steps = Math.min(32, Math.max(1, Math.ceil(dist / stepSpacing)));
    let dirty: DirtyTexelBounds | null = null;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const samplePt: Point2D = {
        x: prev.x + (imagePoint.x - prev.x) * t,
        y: prev.y + (imagePoint.y - prev.y) * t,
      };
      if (!this.isOnAnchoredComponent(s, samplePt)) {
        // Suspend stamping cleanly across holes/occluders so we never bridge across unrelated objects.
        s.lastOnSurfacePoint = null;
        if (dirty) {
          s.bounds = mergeDirtyBounds(s.bounds, dirty);
          s.revision++;
        }
        return dirty !== null;
      }
      const { bounds } = stampCurvedBrushDab({
        strokeMask: s.alphaMask,
        texSpec: s.spec,
        imageSize: s.imageSize,
        centerImagePt: samplePt,
        surfaceRadius: s.surfaceRadius,
        hardness: s.brush.hardness,
        geometry: s.geometry,
        gridIndex: s.gridIndex,
        componentMask: s.componentMask,
        workspace: this.workspace,
      });
      dirty = mergeDirtyBounds(dirty, bounds);
    }

    s.lastOnSurfacePoint = { x: imagePoint.x, y: imagePoint.y };
    if (dirty) {
      s.bounds = mergeDirtyBounds(s.bounds, dirty);
      s.revision++;
      return true;
    }
    return false;
  }

  commitStroke(): ImageSpaceStroke | null {
    const s = this.session;
    this.session = null;
    if (!s || !s.bounds) {
      return null;
    }
    return {
      id: s.id,
      layer: s.layer,
      mode: s.mode,
      brush: s.brush,
      layerWidth: s.spec.width,
      layerHeight: s.spec.height,
      bounds: s.bounds,
      alphaMask: cropMaskToBounds(s.alphaMask, s.spec.width, s.bounds),
    };
  }

  cancelStroke(): void {
    this.session = null;
  }
}
