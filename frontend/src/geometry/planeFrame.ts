import type { Point2D, Size2D } from '../types/editor';
import {
  type ParsedIntrinsics,
  type Vec3Tuple,
  imagePointToNormalizedUv,
  normalizedUvToRayOpenCV,
  vecAdd,
  vecCross,
  vecDot,
  vecNormalize,
  vecScale,
  vecSub,
} from './geometryAccess';

export interface PlaneModel {
  origin: Vec3Tuple;
  normal: Vec3Tuple;
}

export interface PlaneFootprint {
  minU: number;
  maxU: number;
  minV: number;
  maxV: number;
}

export interface SurfaceSelectionResult {
  seedImagePoint: Point2D;
  seedInferencePixel: Point2D;
  seedPoint3D: Vec3Tuple;
  plane: PlaneModel;
  frame: PlaneFrame;
  supportMask: Uint8Array;
  inlierCount: number;
  rmsError: number;
  suggestedFootprint: PlaneFootprint;
  referenceBrushDiameter: number;
}

export type SurfaceSelectionOutcome =
  { ok: true; selection: SurfaceSelectionResult } | { ok: false; reason: string };

/** Orients a plane normal to point toward the camera at the OpenCV origin. */
export function orientNormalTowardCamera(
  normal: readonly [number, number, number],
  origin: readonly [number, number, number],
): Vec3Tuple {
  const unit = vecNormalize(normal) ?? [0, 0, -1];
  const toCamera: Vec3Tuple = [-origin[0], -origin[1], -origin[2]];
  if (vecDot(unit, toCamera) < 0) {
    return [-unit[0], -unit[1], -unit[2]];
  }
  return unit;
}

export interface RayPlaneHit {
  point: Vec3Tuple;
  u: number;
  v: number;
  depth: number;
  t: number;
}

/**
 * Orthogonal right-handed local coordinate frame attached to a 3D plane in OpenCV camera space.
 * +u aligns approximately with screen-right (+X) and +v with screen-down (+Y).
 */
export class PlaneFrame {
  readonly origin: Vec3Tuple;
  readonly normal: Vec3Tuple;
  readonly uAxis: Vec3Tuple;
  readonly vAxis: Vec3Tuple;

  constructor(origin: Vec3Tuple, normal: Vec3Tuple, uAxis: Vec3Tuple, vAxis: Vec3Tuple) {
    this.origin = origin;
    this.normal = normal;
    this.uAxis = uAxis;
    this.vAxis = vAxis;
  }

  static fromPlane(plane: PlaneModel, anchorPoint?: Vec3Tuple): PlaneFrame {
    const unitNormal = orientNormalTowardCamera(plane.normal, plane.origin);

    let frameOrigin = plane.origin;
    if (anchorPoint) {
      const signedDist = vecDot(unitNormal, vecSub(anchorPoint, plane.origin));
      frameOrigin = vecSub(anchorPoint, vecScale(unitNormal, signedDist));
    }

    // Project camera screen-right (+X) onto the plane; fall back when nearly parallel to normal.
    const camRight: Vec3Tuple = [1, 0, 0];
    let projRight = vecSub(camRight, vecScale(unitNormal, vecDot(camRight, unitNormal)));
    let uAxis = vecNormalize(projRight);

    if (!uAxis) {
      const camForward: Vec3Tuple = [0, 0, 1];
      projRight = vecSub(camForward, vecScale(unitNormal, vecDot(camForward, unitNormal)));
      uAxis = vecNormalize(projRight) ?? [0, 1, 0];
    }

    // In OpenCV (+X right, +Y down, +Z forward) with camera-facing normal (nz < 0),
    // uAxis x normal points along screen-down (+Y).
    let vAxis = vecNormalize(vecCross(uAxis, unitNormal)) ?? [0, 1, 0];
    if (vAxis[1] < 0 && Math.abs(unitNormal[1]) < 0.999) {
      // Ensure screen-down preference when fallback axes changed handedness.
      vAxis = vecScale(vAxis, -1);
      uAxis = vecNormalize(vecCross(unitNormal, vAxis)) ?? uAxis;
    }

    return new PlaneFrame(frameOrigin, unitNormal, uAxis, vAxis);
  }

  worldToLocal(point: readonly [number, number, number]): Vec3Tuple {
    const delta = vecSub(point, this.origin);
    return [vecDot(delta, this.uAxis), vecDot(delta, this.vAxis), vecDot(delta, this.normal)];
  }

  localToWorld(u: number, v: number, normalOffset = 0): Vec3Tuple {
    return vecAdd(
      vecAdd(this.origin, vecScale(this.uAxis, u)),
      vecAdd(vecScale(this.vAxis, v), vecScale(this.normal, normalOffset)),
    );
  }

  intersectRay(
    rayOrigin: readonly [number, number, number],
    rayDir: readonly [number, number, number],
  ): RayPlaneHit | null {
    const denom = vecDot(this.normal, rayDir);
    if (Math.abs(denom) < 1e-6) {
      return null;
    }
    const t = vecDot(this.normal, vecSub(this.origin, rayOrigin)) / denom;
    if (!Number.isFinite(t) || t <= 1e-4) {
      return null;
    }
    const hitPoint = vecAdd(rayOrigin, vecScale(rayDir, t));
    if (hitPoint[2] <= 1e-4) {
      return null;
    }
    const [u, v] = this.worldToLocal(hitPoint);
    return {
      point: hitPoint,
      u,
      v,
      depth: hitPoint[2],
      t,
    };
  }

  imagePointToPlane(
    imagePoint: Point2D,
    imageSize: Size2D,
    intrinsics: ParsedIntrinsics,
  ): RayPlaneHit | null {
    const uv = imagePointToNormalizedUv(imagePoint, imageSize);
    const rayDir = normalizedUvToRayOpenCV(uv, intrinsics, false);
    return this.intersectRay([0, 0, 0], rayDir);
  }
}
