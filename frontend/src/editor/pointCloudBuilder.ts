export const DEFAULT_POINT_BUDGET = 300_000;

export interface PointCloudAttributes {
  positions: Float32Array;
  colors: Float32Array;
  pointCount: number;
  stride: number;
  center: [number, number, number];
  boundingRadius: number;
}

export interface PointCloudBuildParams {
  points: Float32Array;
  colors: Uint8ClampedArray;
  mask: Uint8Array;
  width: number;
  height: number;
  stride?: number;
  maxPoints?: number;
}

export interface InitialCameraPose {
  position: [number, number, number];
  target: [number, number, number];
  fovDeg: number;
}

export function opencvPointToThreejs(x: number, y: number, z: number): [number, number, number] {
  return [x, -y, -z];
}

export function computePointCloudStride(
  width: number,
  height: number,
  maxPoints = DEFAULT_POINT_BUDGET,
): number {
  if (width <= 0 || height <= 0 || maxPoints <= 0) {
    return 1;
  }
  const total = width * height;
  if (total <= maxPoints) {
    return 1;
  }
  return Math.max(1, Math.ceil(Math.sqrt(total / maxPoints)));
}

export function buildPointCloudAttributes(params: PointCloudBuildParams): PointCloudAttributes {
  const { points, colors, mask, width, height } = params;
  const stride =
    params.stride !== undefined
      ? Math.max(1, Math.round(params.stride))
      : computePointCloudStride(width, height, params.maxPoints ?? DEFAULT_POINT_BUDGET);

  const maxCandidates = Math.ceil(width / stride) * Math.ceil(height / stride);
  const posScratch = new Float32Array(maxCandidates * 3);
  const colScratch = new Float32Array(maxCandidates * 3);

  let count = 0;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (let row = 0; row < height; row += stride) {
    for (let col = 0; col < width; col += stride) {
      const idx = row * width + col;
      if (mask[idx] === 0) {
        continue;
      }

      const pBase = idx * 3;
      const ox = points[pBase];
      const oy = points[pBase + 1];
      const oz = points[pBase + 2];

      if (
        ox === undefined ||
        oy === undefined ||
        oz === undefined ||
        !Number.isFinite(ox) ||
        !Number.isFinite(oy) ||
        !Number.isFinite(oz) ||
        oz <= 0
      ) {
        continue;
      }

      const [tx, ty, tz] = opencvPointToThreejs(ox, oy, oz);
      const outBase = count * 3;
      posScratch[outBase] = tx;
      posScratch[outBase + 1] = ty;
      posScratch[outBase + 2] = tz;

      const cBase = idx * 4;
      colScratch[outBase] = (colors[cBase] ?? 0) / 255;
      colScratch[outBase + 1] = (colors[cBase + 1] ?? 0) / 255;
      colScratch[outBase + 2] = (colors[cBase + 2] ?? 0) / 255;

      if (tx < minX) minX = tx;
      if (ty < minY) minY = ty;
      if (tz < minZ) minZ = tz;
      if (tx > maxX) maxX = tx;
      if (ty > maxY) maxY = ty;
      if (tz > maxZ) maxZ = tz;

      count += 1;
    }
  }

  if (count === 0) {
    return {
      positions: new Float32Array(0),
      colors: new Float32Array(0),
      pointCount: 0,
      stride,
      center: [0, 0, -2],
      boundingRadius: 1,
    };
  }

  const center: [number, number, number] = [
    (minX + maxX) * 0.5,
    (minY + maxY) * 0.5,
    (minZ + maxZ) * 0.5,
  ];
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  const boundingRadius = Math.max(0.1, 0.5 * Math.hypot(dx, dy, dz));

  return {
    positions: posScratch.subarray(0, count * 3),
    colors: colScratch.subarray(0, count * 3),
    pointCount: count,
    stride,
    center,
    boundingRadius,
  };
}

export function computeFovYDegrees(intrinsics: number[][] | undefined): number {
  const fyNorm = intrinsics?.[1]?.[1];
  if (fyNorm === undefined || !Number.isFinite(fyNorm) || fyNorm <= 0) {
    return 55;
  }
  const rawDeg = (2 * Math.atan(0.5 / fyNorm) * 180) / Math.PI;
  return Math.min(100, Math.max(25, rawDeg));
}

export function computeInitialCameraPose(
  center: [number, number, number],
  boundingRadius: number,
  intrinsics?: number[][],
): InitialCameraPose {
  const fovDeg = computeFovYDegrees(intrinsics);
  const distFromOrigin = Math.hypot(center[0], center[1], center[2]);

  // Camera sits at origin looking down -Z when scene points are in front of it.
  if (center[2] < -0.05 && distFromOrigin >= 0.05) {
    return {
      position: [0, 0, 0],
      target: center,
      fovDeg,
    };
  }

  const offset = Math.max(1.5, boundingRadius * 2.2);
  return {
    position: [center[0], center[1], center[2] + offset],
    target: center,
    fovDeg,
  };
}
