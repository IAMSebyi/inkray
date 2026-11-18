import type { Point2D, Size2D } from '../types/editor';
import {
  type GeometryInput,
  type ParsedIntrinsics,
  type Vec3Tuple,
  imagePointToInferencePixel,
  imagePointToNormalizedUv,
  normalizedUvToRayOpenCV,
  parseNormalizedIntrinsics,
  readNormalOpenCV,
  vecDot,
  vecSub,
} from '../geometry/geometryAccess';
import type { PlaneFootprint, PlaneFrame } from '../geometry/planeFrame';
import { clamp } from '../editor/viewportTransforms';
import { type PlanarDecalSurface, getEffectiveFootprint, isInsideFootprint } from './planarDecal';

export type VisibilityReason =
  | 'visible'
  | 'invalid-geometry'
  | 'invalid-projection'
  | 'behind-camera'
  | 'outside-footprint'
  | 'occluded'
  | 'background'
  | 'incompatible-surface';

export interface PixelVisibilityResult {
  visible: boolean;
  alpha: number;
  reason: VisibilityReason;
  u: number;
  v: number;
  zPlane: number;
  zObserved: number;
}

export interface VisibilityOptions {
  depthToleranceRatio?: number;
  fitRms?: number;
  minNormalDot?: number;
  featherWidthRatio?: number;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  if (edge0 === edge1) {
    return x >= edge1 ? 1 : 0;
  }
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

export function computeScaleAwareDepthTolerance(
  zPlane: number,
  depthToleranceRatio = 0.035,
  fitRms = 0,
): number {
  const baseTol = Math.max(depthToleranceRatio * zPlane, 2.5 * fitRms);
  return clamp(baseTol, 0.015 * zPlane, 0.14 * zPlane);
}

export const PlaneVisibility = {
  /**
   * Reference CPU evaluation matching the GLSL visibility logic in PlanarPaintRenderer.
   */
  evaluatePixel(
    uvImg: Point2D,
    intrinsics: ParsedIntrinsics,
    frame: PlaneFrame,
    footprint: PlaneFootprint,
    observedValid: boolean,
    observedDepth: number,
    observedNormal: Vec3Tuple | null,
    options?: VisibilityOptions,
  ): PixelVisibilityResult {
    if (!observedValid || !Number.isFinite(observedDepth) || observedDepth <= 0) {
      return {
        visible: false,
        alpha: 0,
        reason: 'invalid-geometry',
        u: 0,
        v: 0,
        zPlane: 0,
        zObserved: observedDepth,
      };
    }

    const rayDir = normalizedUvToRayOpenCV(uvImg, intrinsics, false);
    const denom = vecDot(frame.normal, rayDir);
    if (Math.abs(denom) < 1e-6) {
      return {
        visible: false,
        alpha: 0,
        reason: 'invalid-projection',
        u: 0,
        v: 0,
        zPlane: 0,
        zObserved: observedDepth,
      };
    }

    const t = vecDot(frame.normal, frame.origin) / denom;
    if (!Number.isFinite(t) || t <= 1e-4) {
      return {
        visible: false,
        alpha: 0,
        reason: 'behind-camera',
        u: 0,
        v: 0,
        zPlane: t,
        zObserved: observedDepth,
      };
    }

    const hitPoint: Vec3Tuple = [rayDir[0] * t, rayDir[1] * t, t];
    const [u, v] = frame.worldToLocal(hitPoint);

    if (!isInsideFootprint(u, v, footprint)) {
      return {
        visible: false,
        alpha: 0,
        reason: 'outside-footprint',
        u,
        v,
        zPlane: t,
        zObserved: observedDepth,
      };
    }

    const tol = computeScaleAwareDepthTolerance(
      t,
      options?.depthToleranceRatio ?? 0.035,
      options?.fitRms ?? 0,
    );
    const diff = observedDepth - t;
    const absDiff = Math.abs(diff);

    if (diff < -tol) {
      return {
        visible: false,
        alpha: 0,
        reason: 'occluded',
        u,
        v,
        zPlane: t,
        zObserved: observedDepth,
      };
    }

    if (diff > tol) {
      return {
        visible: false,
        alpha: 0,
        reason: 'background',
        u,
        v,
        zPlane: t,
        zObserved: observedDepth,
      };
    }

    const minNormalDot = options?.minNormalDot ?? 0.45;
    if (observedNormal !== null) {
      const normalAlign = vecDot(observedNormal, frame.normal);
      if (normalAlign < minNormalDot) {
        return {
          visible: false,
          alpha: 0,
          reason: 'incompatible-surface',
          u,
          v,
          zPlane: t,
          zObserved: observedDepth,
        };
      }
    }

    const feather = clamp(options?.featherWidthRatio ?? 0.15, 0, 0.45);
    const innerTol = tol * (1 - feather);
    const outerTol = tol * (1 + feather * 0.35);
    const alpha = feather > 0 ? smoothstep(outerTol, innerTol, absDiff) : 1;

    return {
      visible: alpha >= 0.5,
      alpha,
      reason: 'visible',
      u,
      v,
      zPlane: t,
      zObserved: observedDepth,
    };
  },

  /** Computes binary pixel visibility mask across an inference grid for testing and verification. */
  computeGridMask(
    geometry: GeometryInput,
    intrinsics: ParsedIntrinsics,
    frame: PlaneFrame,
    footprint: PlaneFootprint,
    options?: VisibilityOptions,
  ): Uint8Array {
    const w = geometry.metadata.inference_width;
    const h = geometry.metadata.inference_height;
    const out = new Uint8Array(w * h);

    for (let row = 0; row < h; row++) {
      const vImg = (row + 0.5) / h;
      const rowOffset = row * w;
      for (let col = 0; col < w; col++) {
        const idx = rowOffset + col;
        const uImg = (col + 0.5) / w;
        const maskVal = geometry.mask[idx] ?? 0;
        const depthVal = geometry.depth[idx] ?? 0;
        const valid = maskVal !== 0 && Number.isFinite(depthVal) && depthVal > 0;
        const normalVal = geometry.normals ? readNormalOpenCV(geometry, col, row) : null;
        const res = PlaneVisibility.evaluatePixel(
          { x: uImg, y: vImg },
          intrinsics,
          frame,
          footprint,
          valid,
          depthVal,
          normalVal,
          options,
        );
        out[idx] = res.visible ? 1 : 0;
      }
    }
    return out;
  },
};

export function computeMaskIoU(predicted: Uint8Array, truth: Uint8Array): number {
  const len = Math.min(predicted.length, truth.length);
  let intersection = 0;
  let union = 0;
  for (let i = 0; i < len; i++) {
    const p = (predicted[i] ?? 0) !== 0;
    const t = (truth[i] ?? 0) !== 0;
    if (p && t) intersection += 1;
    if (p || t) union += 1;
  }
  return union === 0 ? 1 : intersection / union;
}

// Keep unused vecSub helper referenced cleanly.
export function distancePointToPlane(point: Vec3Tuple, frame: PlaneFrame): number {
  return Math.abs(vecDot(frame.normal, vecSub(point, frame.origin)));
}

/**
 * Resolves the highest-priority surface claiming an image point using the exact
 * GPU visibility test on observed depth, mask, and surface normals.
 */
export function findSurfaceOwnerAtImagePoint(
  surfaces: readonly PlanarDecalSurface[],
  imagePoint: Point2D,
  originalSize: Size2D,
  geometry: GeometryInput,
): PlanarDecalSurface | null {
  if (surfaces.length === 0) {
    return null;
  }
  const infW = geometry.metadata.inference_width;
  const infH = geometry.metadata.inference_height;
  const pixel = imagePointToInferencePixel(imagePoint, originalSize, infW, infH);
  if (!pixel) {
    return null;
  }

  const idx = pixel.y * infW + pixel.x;
  const maskVal = geometry.mask[idx] ?? 0;
  const depthVal = geometry.depth[idx] ?? 0;
  const valid = maskVal !== 0 && Number.isFinite(depthVal) && depthVal > 0;
  if (!valid) {
    return null;
  }

  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const uvImg = imagePointToNormalizedUv(imagePoint, originalSize);
  const obsNormal = geometry.normals ? readNormalOpenCV(geometry, pixel.x, pixel.y) : null;

  for (let i = 0; i < surfaces.length; i++) {
    const surface = surfaces[i];
    if (!surface) continue;
    const res = PlaneVisibility.evaluatePixel(
      uvImg,
      intr,
      surface.frame,
      getEffectiveFootprint(surface),
      true,
      depthVal,
      obsNormal,
      { fitRms: surface.rmsError },
    );
    if (res.visible) {
      return surface;
    }
  }
  return null;
}

/**
 * Assigns each inference grid pixel to at most one surface (1-based priority index,
 * 0 when unowned) so intersecting planes resolve deterministically by user priority.
 */
export function computeSurfaceOwnerGrid(
  surfaces: readonly PlanarDecalSurface[],
  geometry: GeometryInput,
): Uint8Array {
  const w = geometry.metadata.inference_width;
  const h = geometry.metadata.inference_height;
  const out = new Uint8Array(w * h);
  if (surfaces.length === 0 || w <= 0 || h <= 0) {
    return out;
  }

  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const surfaceLimit = Math.min(surfaces.length, 255);
  const prepared = new Array<{
    ox: number;
    oy: number;
    oz: number;
    nx: number;
    ny: number;
    nz: number;
    ux: number;
    uy: number;
    uz: number;
    vx: number;
    vy: number;
    vz: number;
    planeD: number;
    minU: number;
    maxU: number;
    minV: number;
    maxV: number;
    fitRms: number;
  }>(surfaceLimit);

  for (let sIdx = 0; sIdx < surfaceLimit; sIdx++) {
    const surface = surfaces[sIdx];
    if (!surface) continue;
    const fp = getEffectiveFootprint(surface);
    const { origin, normal, uAxis, vAxis } = surface.frame;
    prepared[sIdx] = {
      ox: origin[0],
      oy: origin[1],
      oz: origin[2],
      nx: normal[0],
      ny: normal[1],
      nz: normal[2],
      ux: uAxis[0],
      uy: uAxis[1],
      uz: uAxis[2],
      vx: vAxis[0],
      vy: vAxis[1],
      vz: vAxis[2],
      planeD: normal[0] * origin[0] + normal[1] * origin[1] + normal[2] * origin[2],
      minU: fp.minU,
      maxU: fp.maxU,
      minV: fp.minV,
      maxV: fp.maxV,
      fitRms: surface.rmsError,
    };
  }

  const normals = geometry.normals;
  const invFx = 1 / intr.fx;
  const invFy = 1 / intr.fy;
  const invW = 1 / w;
  const invH = 1 / h;

  for (let row = 0; row < h; row++) {
    const vImg = (row + 0.5) * invH;
    const yNorm = (vImg - intr.cy) * invFy;
    const rowOffset = row * w;
    for (let col = 0; col < w; col++) {
      const idx = rowOffset + col;
      const maskVal = geometry.mask[idx] ?? 0;
      const depthVal = geometry.depth[idx] ?? 0;
      if (maskVal === 0 || !Number.isFinite(depthVal) || depthVal <= 0) {
        continue;
      }

      const uImg = (col + 0.5) * invW;
      const xNorm = (uImg - intr.cx - intr.skew * yNorm) * invFx;

      let hasNormal = false;
      let onx = 0;
      let ony = 0;
      let onz = 0;
      if (normals) {
        const nBase = idx * 3;
        const rx = normals[nBase] ?? 0;
        const ry = normals[nBase + 1] ?? 0;
        const rz = normals[nBase + 2] ?? 0;
        if (Number.isFinite(rx) && Number.isFinite(ry) && Number.isFinite(rz)) {
          const len = Math.hypot(rx, ry, rz);
          if (len > 1e-6) {
            hasNormal = true;
            onx = rx / len;
            ony = ry / len;
            onz = rz / len;
          }
        }
      }

      for (let sIdx = 0; sIdx < surfaceLimit; sIdx++) {
        const s = prepared[sIdx];
        if (!s) continue;

        const denom = s.nx * xNorm + s.ny * yNorm + s.nz;
        if (Math.abs(denom) < 1e-6) continue;

        const t = s.planeD / denom;
        if (!Number.isFinite(t) || t <= 1e-4) continue;

        const dx = xNorm * t - s.ox;
        const dy = yNorm * t - s.oy;
        const dz = t - s.oz;
        const u = dx * s.ux + dy * s.uy + dz * s.uz;
        if (u < s.minU || u > s.maxU) continue;
        const v = dx * s.vx + dy * s.vy + dz * s.vz;
        if (v < s.minV || v > s.maxV) continue;

        const tol = clamp(Math.max(0.035 * t, 2.5 * s.fitRms), 0.015 * t, 0.14 * t);
        const absDiff = Math.abs(depthVal - t);
        if (absDiff > tol) continue;

        if (hasNormal && onx * s.nx + ony * s.ny + onz * s.nz < 0.45) {
          continue;
        }

        const innerTol = tol * 0.85;
        const outerTol = tol * 1.0525;
        const st = clamp((absDiff - outerTol) / (innerTol - outerTol), 0, 1);
        const alpha = st * st * (3 - 2 * st);
        if (alpha >= 0.5) {
          out[idx] = sIdx + 1;
          break;
        }
      }
    }
  }
  return out;
}
