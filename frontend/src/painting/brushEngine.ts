import type { PlaneFootprint } from '../geometry/planeFrame';
import { clamp } from '../editor/viewportTransforms';
import { type DecalTextureSpec, decalTexelToPlaneCoords, isInsideFootprint } from './planarDecal';

export interface BrushSettings {
  /** Brush diameter expressed as a multiplier of the active surface reference diameter, or raw plane units when used directly in BrushStroke. */
  diameter: number;
  color: string;
  opacity: number;
  hardness: number;
}

export type BrushStrokeMode = 'paint' | 'erase';

export interface BrushPoint {
  u: number;
  v: number;
}

export interface BrushStroke {
  id: string;
  surfaceId: string;
  brushObjectId?: string;
  mode?: BrushStrokeMode;
  brush: BrushSettings;
  points: BrushPoint[];
}

export interface DirtyTexelBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function parseHexColorRgb(hex: string): [number, number, number] {
  const cleaned = hex.trim().replace(/^#/, '');
  if (cleaned.length === 3) {
    const r = parseInt((cleaned[0] ?? 'f') + (cleaned[0] ?? 'f'), 16);
    const g = parseInt((cleaned[1] ?? 'f') + (cleaned[1] ?? 'f'), 16);
    const b = parseInt((cleaned[2] ?? 'f') + (cleaned[2] ?? 'f'), 16);
    if (Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b)) {
      return [r, g, b];
    }
  }
  if (cleaned.length === 6) {
    const r = parseInt(cleaned.slice(0, 2), 16);
    const g = parseInt(cleaned.slice(2, 4), 16);
    const b = parseInt(cleaned.slice(4, 6), 16);
    if (Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b)) {
      return [r, g, b];
    }
  }
  return [255, 77, 77];
}

export function pointToSegmentDistanceSq(
  pu: number,
  pv: number,
  au: number,
  av: number,
  bu: number,
  bv: number,
): number {
  const abU = bu - au;
  const abV = bv - av;
  const apU = pu - au;
  const apV = pv - av;
  const abLenSq = abU * abU + abV * abV;
  if (abLenSq <= 1e-18) {
    return apU * apU + apV * apV;
  }
  const t = clamp((apU * abU + apV * abV) / abLenSq, 0, 1);
  const closestU = au + t * abU;
  const closestV = av + t * abV;
  const du = pu - closestU;
  const dv = pv - closestV;
  return du * du + dv * dv;
}

export function computeBrushFalloff(dist: number, radius: number, hardness: number): number {
  if (dist >= radius || radius <= 0) {
    return 0;
  }
  const norm = dist / radius;
  const h = clamp(hardness, 0, 1);
  if (norm <= h) {
    return 1;
  }
  const edgeSpan = Math.max(1e-4, 1 - h);
  const t = clamp((norm - h) / edgeSpan, 0, 1);
  return 1 - t * t * (3 - 2 * t);
}

/**
 * Stamps a single plane-space capsule segment into a per-stroke alpha coverage mask (0..255).
 * Using max() per texel ensures overlapping stamps within one stroke never over-darken opacity.
 */
export function stampSegmentIntoStrokeMask(
  strokeMask: Uint8Array,
  footprint: PlaneFootprint,
  spec: DecalTextureSpec,
  a: BrushPoint,
  b: BrushPoint,
  diameter: number,
  hardness: number,
): DirtyTexelBounds | null {
  const radius = Math.max(1e-6, diameter * 0.5);
  const spanU = Math.max(1e-6, footprint.maxU - footprint.minU);
  const spanV = Math.max(1e-6, footprint.maxV - footprint.minV);

  const minU = Math.min(a.u, b.u) - radius;
  const maxU = Math.max(a.u, b.u) + radius;
  const minV = Math.min(a.v, b.v) - radius;
  const maxV = Math.max(a.v, b.v) + radius;

  if (
    maxU < footprint.minU ||
    minU > footprint.maxU ||
    maxV < footprint.minV ||
    minV > footprint.maxV
  ) {
    return null;
  }

  const minX = clamp(
    Math.floor(((minU - footprint.minU) / spanU) * spec.width) - 1,
    0,
    spec.width - 1,
  );
  const maxX = clamp(
    Math.ceil(((maxU - footprint.minU) / spanU) * spec.width) + 1,
    0,
    spec.width - 1,
  );
  const minY = clamp(
    Math.floor(((minV - footprint.minV) / spanV) * spec.height) - 1,
    0,
    spec.height - 1,
  );
  const maxY = clamp(
    Math.ceil(((maxV - footprint.minV) / spanV) * spec.height) + 1,
    0,
    spec.height - 1,
  );

  const radiusSq = radius * radius;
  let changed = false;

  for (let py = minY; py <= maxY; py++) {
    const rowOffset = py * spec.width;
    for (let px = minX; px <= maxX; px++) {
      const [pu, pv] = decalTexelToPlaneCoords(px, py, footprint, spec);
      const distSq = pointToSegmentDistanceSq(pu, pv, a.u, a.v, b.u, b.v);
      if (distSq >= radiusSq) {
        continue;
      }
      const coverage = computeBrushFalloff(Math.sqrt(distSq), radius, hardness);
      const nextAlpha = Math.round(clamp(coverage, 0, 1) * 255);
      const idx = rowOffset + px;
      const prevAlpha = strokeMask[idx] ?? 0;
      if (nextAlpha > prevAlpha) {
        strokeMask[idx] = nextAlpha;
        changed = true;
      }
    }
  }

  return changed ? { minX, minY, maxX, maxY } : null;
}

export function rasterizeStrokeToMask(
  stroke: BrushStroke,
  footprint: PlaneFootprint,
  spec: DecalTextureSpec,
): Uint8Array {
  const mask = new Uint8Array(spec.width * spec.height);
  const pts = stroke.points;
  if (pts.length === 0) {
    return mask;
  }

  const isErase = stroke.mode === 'erase';
  const strokeHardness = isErase ? 1 : stroke.brush.hardness;

  if (pts.length === 1) {
    const first = pts[0];
    if (first) {
      stampSegmentIntoStrokeMask(
        mask,
        footprint,
        spec,
        first,
        first,
        stroke.brush.diameter,
        strokeHardness,
      );
    }
    return mask;
  }

  for (let i = 1; i < pts.length; i++) {
    const prev = pts[i - 1];
    const curr = pts[i];
    if (!prev || !curr) continue;
    stampSegmentIntoStrokeMask(
      mask,
      footprint,
      spec,
      prev,
      curr,
      stroke.brush.diameter,
      strokeHardness,
    );
  }
  return mask;
}

/** Composites a single-stroke alpha mask with uniform brush opacity into an RGBA buffer. */
export function compositeStrokeMaskOverRgba(
  dstRgba: Uint8ClampedArray | Uint8Array,
  strokeMask: Uint8Array,
  brush: BrushSettings,
  width: number,
  bounds?: DirtyTexelBounds,
  mode: BrushStrokeMode = 'paint',
): void {
  const [sr, sg, sb] = parseHexColorRgb(brush.color);
  const opacity = mode === 'erase' ? 1 : clamp(brush.opacity, 0, 1);
  if (opacity <= 0) {
    return;
  }

  const height = Math.floor(strokeMask.length / Math.max(1, width));
  const minX = bounds ? clamp(bounds.minX, 0, width - 1) : 0;
  const maxX = bounds ? clamp(bounds.maxX, 0, width - 1) : width - 1;
  const minY = bounds ? clamp(bounds.minY, 0, height - 1) : 0;
  const maxY = bounds ? clamp(bounds.maxY, 0, height - 1) : height - 1;

  for (let py = minY; py <= maxY; py++) {
    const rowOffset = py * width;
    for (let px = minX; px <= maxX; px++) {
      const idx = rowOffset + px;
      const maskVal = strokeMask[idx] ?? 0;
      if (maskVal === 0) {
        continue;
      }

      const srcA = (maskVal / 255) * opacity;
      const outBase = idx * 4;
      const dstA = (dstRgba[outBase + 3] ?? 0) / 255;

      if (mode === 'erase') {
        if (dstA <= 0) {
          continue;
        }
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
      if (outA <= 1e-6) {
        continue;
      }

      const dstR = dstRgba[outBase] ?? 0;
      const dstG = dstRgba[outBase + 1] ?? 0;
      const dstB = dstRgba[outBase + 2] ?? 0;

      const invOutA = 1 / outA;
      dstRgba[outBase] = Math.round((sr * srcA + dstR * dstA * (1 - srcA)) * invOutA);
      dstRgba[outBase + 1] = Math.round((sg * srcA + dstG * dstA * (1 - srcA)) * invOutA);
      dstRgba[outBase + 2] = Math.round((sb * srcA + dstB * dstA * (1 - srcA)) * invOutA);
      dstRgba[outBase + 3] = Math.round(clamp(outA, 0, 1) * 255);
    }
  }
}

export function rebuildSurfaceDecalRgba(
  strokes: readonly BrushStroke[],
  footprint: PlaneFootprint,
  spec: DecalTextureSpec,
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(spec.width * spec.height * 4);
  for (let i = 0; i < strokes.length; i++) {
    const stroke = strokes[i];
    if (!stroke) continue;
    const strokeMask = rasterizeStrokeToMask(stroke, footprint, spec);
    compositeStrokeMaskOverRgba(
      rgba,
      strokeMask,
      stroke.brush,
      spec.width,
      undefined,
      stroke.mode ?? 'paint',
    );
  }
  return rgba;
}

let nextStrokeSeq = 1;

export class BrushEngine {
  private pendingStrokes: BrushStroke[] = [];
  private activeStroke: BrushStroke | null = null;
  private activeFootprint: PlaneFootprint | null = null;
  private activeMode: BrushStrokeMode = 'paint';
  private lastSample: BrushPoint | null = null;
  private stepSpacing = 0.005;

  getActiveStroke(): BrushStroke | null {
    return this.activeStroke;
  }

  getPreviewStrokes(): BrushStroke[] {
    if (this.pendingStrokes.length === 0) {
      return this.activeStroke ? [this.activeStroke] : [];
    }
    return this.activeStroke
      ? [...this.pendingStrokes, this.activeStroke]
      : [...this.pendingStrokes];
  }

  isStrokeActive(): boolean {
    return this.activeStroke !== null || this.pendingStrokes.length > 0;
  }

  private initActiveStroke(
    surfaceId: string,
    footprint: PlaneFootprint,
    brush: BrushSettings,
    point: BrushPoint,
    mode: BrushStrokeMode,
  ): BrushStroke | null {
    // Reject strokes initiated outside the target plane footprint.
    if (!isInsideFootprint(point.u, point.v, footprint)) {
      return null;
    }

    const stroke: BrushStroke = {
      id: `stroke-${nextStrokeSeq++}`,
      surfaceId,
      mode,
      brush: {
        diameter: Math.max(1e-5, brush.diameter),
        color: brush.color,
        opacity: mode === 'erase' ? 1 : clamp(brush.opacity, 0.01, 1),
        hardness: mode === 'erase' ? 1 : clamp(brush.hardness, 0, 1),
      },
      points: [{ u: point.u, v: point.v }],
    };

    const footprintSpan = Math.max(
      footprint.maxU - footprint.minU,
      footprint.maxV - footprint.minV,
    );
    this.stepSpacing = Math.max(brush.diameter * 0.15, footprintSpan * 0.001);
    this.activeStroke = stroke;
    this.activeFootprint = footprint;
    this.activeMode = mode;
    this.lastSample = { u: point.u, v: point.v };
    return stroke;
  }

  beginStroke(
    surfaceId: string,
    footprint: PlaneFootprint,
    brush: BrushSettings,
    point: BrushPoint,
    mode: BrushStrokeMode = 'paint',
  ): BrushStroke | null {
    this.pendingStrokes = [];
    this.activeStroke = null;
    return this.initActiveStroke(surfaceId, footprint, brush, point, mode);
  }

  switchSurfaceStroke(
    nextSurfaceId: string,
    nextFootprint: PlaneFootprint,
    nextBrush: BrushSettings,
    nextStartPoint: BrushPoint,
    prevEndPoint?: BrushPoint | null,
  ): BrushStroke | null {
    if (this.activeStroke) {
      if (prevEndPoint) {
        this.extendStroke(prevEndPoint, true);
      }
      if (this.activeStroke.points.length > 0) {
        this.pendingStrokes.push({
          ...this.activeStroke,
          points: [...this.activeStroke.points],
        });
      }
      this.activeStroke = null;
      this.activeFootprint = null;
      this.lastSample = null;
    }
    return this.initActiveStroke(
      nextSurfaceId,
      nextFootprint,
      nextBrush,
      nextStartPoint,
      this.activeMode,
    );
  }

  extendStroke(point: BrushPoint, forceAppend = false): BrushStroke | null {
    if (!this.activeStroke || !this.lastSample || !this.activeFootprint) {
      return null;
    }

    const du = point.u - this.lastSample.u;
    const dv = point.v - this.lastSample.v;
    const dist = Math.hypot(du, dv);
    if (!forceAppend && dist < this.stepSpacing) {
      return null;
    }
    if (dist <= 1e-12) {
      return this.activeStroke;
    }

    const stepCount = Math.max(1, Math.floor(dist / this.stepSpacing));
    const startU = this.lastSample.u;
    const startV = this.lastSample.v;

    for (let step = 1; step <= stepCount; step++) {
      const t = step / stepCount;
      this.activeStroke.points.push({
        u: startU + du * t,
        v: startV + dv * t,
      });
    }

    this.lastSample = { u: point.u, v: point.v };
    return this.activeStroke;
  }

  commitStroke(): BrushStroke | null {
    const batch = this.commitBatch();
    return batch[0] ?? null;
  }

  commitBatch(): BrushStroke[] {
    const out = [...this.pendingStrokes];
    if (this.activeStroke && this.activeStroke.points.length > 0) {
      out.push({
        ...this.activeStroke,
        points: [...this.activeStroke.points],
      });
    }
    this.pendingStrokes = [];
    this.activeStroke = null;
    this.activeFootprint = null;
    this.lastSample = null;
    return out;
  }

  cancelStroke(): void {
    this.pendingStrokes = [];
    this.activeStroke = null;
    this.activeFootprint = null;
    this.lastSample = null;
  }
}
