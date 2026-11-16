import type { Point2D, Size2D } from '../types/editor';

export type Vec3Tuple = [number, number, number];

export interface GeometryInput {
  metadata: {
    inference_width: number;
    inference_height: number;
    intrinsics: number[][];
  };
  depth: Float32Array;
  points: Float32Array;
  mask: Uint8Array;
  normals: Float32Array | null;
}

export interface ParsedIntrinsics {
  fx: number;
  fy: number;
  skew: number;
  cx: number;
  cy: number;
}

export function parseNormalizedIntrinsics(intrinsics: number[][]): ParsedIntrinsics {
  const row0 = intrinsics[0];
  const row1 = intrinsics[1];
  const fx = row0?.[0] ?? 1;
  const skew = row0?.[1] ?? 0;
  const cx = row0?.[2] ?? 0.5;
  const fy = row1?.[1] ?? 1;
  const cy = row1?.[2] ?? 0.5;

  if (!Number.isFinite(fx) || fx <= 0 || !Number.isFinite(fy) || fy <= 0) {
    throw new Error('Camera intrinsics must have positive finite focal lengths.');
  }

  return { fx, fy, skew: Number.isFinite(skew) ? skew : 0, cx, cy };
}

export function computePixelAngularSpan(
  intrinsics: ParsedIntrinsics,
  width: number,
  height: number,
): number {
  const stepX = width > 0 ? 1 / (intrinsics.fx * width) : 0.01;
  const stepY = height > 0 ? 1 / (intrinsics.fy * height) : 0.01;
  return Math.max(stepX, stepY);
}

export function isValidGridPixel(geom: GeometryInput, idx: number): boolean {
  if (geom.mask[idx] === 0) {
    return false;
  }
  const z = geom.depth[idx];
  if (z === undefined || !Number.isFinite(z) || z <= 0) {
    return false;
  }
  const base = idx * 3;
  const px = geom.points[base];
  const py = geom.points[base + 1];
  const pz = geom.points[base + 2];
  return (
    px !== undefined &&
    py !== undefined &&
    pz !== undefined &&
    Number.isFinite(px) &&
    Number.isFinite(py) &&
    Number.isFinite(pz) &&
    pz > 0
  );
}

export function readPointOpenCV(geom: GeometryInput, col: number, row: number): Vec3Tuple | null {
  const w = geom.metadata.inference_width;
  const h = geom.metadata.inference_height;
  if (col < 0 || col >= w || row < 0 || row >= h) {
    return null;
  }
  const idx = row * w + col;
  if (!isValidGridPixel(geom, idx)) {
    return null;
  }
  const base = idx * 3;
  const x = geom.points[base] ?? 0;
  const y = geom.points[base + 1] ?? 0;
  const z = geom.points[base + 2] ?? 0;
  return [x, y, z];
}

export function readNormalOpenCV(geom: GeometryInput, col: number, row: number): Vec3Tuple | null {
  const normals = geom.normals;
  if (!normals) {
    return null;
  }
  const w = geom.metadata.inference_width;
  const h = geom.metadata.inference_height;
  if (col < 0 || col >= w || row < 0 || row >= h) {
    return null;
  }
  const idx = row * w + col;
  if (!isValidGridPixel(geom, idx)) {
    return null;
  }
  const base = idx * 3;
  const nx = normals[base];
  const ny = normals[base + 1];
  const nz = normals[base + 2];
  if (
    nx === undefined ||
    ny === undefined ||
    nz === undefined ||
    !Number.isFinite(nx) ||
    !Number.isFinite(ny) ||
    !Number.isFinite(nz)
  ) {
    return null;
  }
  const len = Math.hypot(nx, ny, nz);
  if (len <= 1e-6) {
    return null;
  }
  return [nx / len, ny / len, nz / len];
}

export function imagePointToNormalizedUv(imagePoint: Point2D, imageSize: Size2D): Point2D {
  return {
    x: imageSize.width > 0 ? imagePoint.x / imageSize.width : 0,
    y: imageSize.height > 0 ? imagePoint.y / imageSize.height : 0,
  };
}

export function normalizedUvToRayOpenCV(
  uv: Point2D,
  intrinsics: ParsedIntrinsics,
  unitLength = true,
): Vec3Tuple {
  const yNorm = (uv.y - intrinsics.cy) / intrinsics.fy;
  const xNorm = (uv.x - intrinsics.cx - intrinsics.skew * yNorm) / intrinsics.fx;
  if (!unitLength) {
    return [xNorm, yNorm, 1];
  }
  const len = Math.hypot(xNorm, yNorm, 1);
  return [xNorm / len, yNorm / len, 1 / len];
}

export function projectOpenCvPointToImage(
  point: readonly [number, number, number],
  intrinsics: ParsedIntrinsics,
  imageSize: Size2D,
): Point2D | null {
  const z = point[2];
  if (!Number.isFinite(z) || z <= 1e-6) {
    return null;
  }
  const x = point[0] / z;
  const y = point[1] / z;
  const u = intrinsics.fx * x + intrinsics.skew * y + intrinsics.cx;
  const v = intrinsics.fy * y + intrinsics.cy;
  return {
    x: u * imageSize.width,
    y: v * imageSize.height,
  };
}

export function imagePointToInferencePixel(
  imagePoint: Point2D,
  originalSize: Size2D,
  inferenceWidth: number,
  inferenceHeight: number,
): Point2D | null {
  if (
    originalSize.width <= 0 ||
    originalSize.height <= 0 ||
    inferenceWidth <= 0 ||
    inferenceHeight <= 0 ||
    imagePoint.x < 0 ||
    imagePoint.x >= originalSize.width ||
    imagePoint.y < 0 ||
    imagePoint.y >= originalSize.height
  ) {
    return null;
  }
  const col = Math.min(
    inferenceWidth - 1,
    Math.max(0, Math.floor((imagePoint.x * inferenceWidth) / originalSize.width)),
  );
  const row = Math.min(
    inferenceHeight - 1,
    Math.max(0, Math.floor((imagePoint.y * inferenceHeight) / originalSize.height)),
  );
  return { x: col, y: row };
}

export function vecDot(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vecSub(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): Vec3Tuple {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vecAdd(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): Vec3Tuple {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vecScale(a: readonly [number, number, number], s: number): Vec3Tuple {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function vecCross(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): Vec3Tuple {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function vecLength(a: readonly [number, number, number]): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function vecNormalize(a: readonly [number, number, number]): Vec3Tuple | null {
  const len = Math.hypot(a[0], a[1], a[2]);
  if (!Number.isFinite(len) || len <= 1e-10) {
    return null;
  }
  return [a[0] / len, a[1] / len, a[2] / len];
}
