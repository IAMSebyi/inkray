import type { Size2D, Point2D } from '../types/editor';
import { clamp } from '../editor/viewportTransforms';
import type { PlanarDecalSurface } from '../painting/planarDecal';
import {
  type GeometryInput,
  type Vec3Tuple,
  computePixelAngularSpan,
  parseNormalizedIntrinsics,
  vecDot,
  vecNormalize,
  vecSub,
} from './geometryAccess';
import {
  PlaneFrame,
  type PlaneFootprint,
  type PlaneModel,
  orientNormalTowardCamera,
} from './planeFrame';
import {
  EDGE_DOWN,
  EDGE_DOWN_LEFT,
  EDGE_DOWN_RIGHT,
  EDGE_LEFT,
  EDGE_RIGHT,
  EDGE_UP,
  EDGE_UP_LEFT,
  EDGE_UP_RIGHT,
  type SurfaceGridIndex,
  type SurfaceHit,
  findConnectedMeshComponent,
  getSurfaceGridIndex,
  pickSurfaceHit,
} from './surfacePicking';
import { computeInterpolatedSurfaceMetric, computeSurfaceBrushRadius } from './localSurfaceMetric';

let nextAutoPlaneSeq = 1;

export type AutoAnchorMode = 'planar' | 'curved' | 'fallback-2d';

export type SurfaceAnchor =
  | {
      mode: 'planar';
      surface: PlanarDecalSurface;
      reusedExisting: boolean;
      surfaceRadius: number;
      seedHit: SurfaceHit;
    }
  | {
      mode: 'curved';
      componentMask: Uint8Array;
      surfaceRadius: number;
      seedHit: SurfaceHit;
    }
  | {
      mode: 'fallback-2d';
      radiusPx: number;
      reason: string;
    };

export interface PlanarFitSummary {
  plane: PlaneModel;
  rmsError: number;
  patchSpan: number;
  normalDotP08: number;
  sampleCount: number;
}

const CARDINAL_STEPS: ReadonlyArray<readonly [number, number, number]> = [
  [EDGE_RIGHT, 1, 0],
  [EDGE_LEFT, -1, 0],
  [EDGE_DOWN, 0, 1],
  [EDGE_UP, 0, -1],
];

const ALL_STEPS: ReadonlyArray<readonly [number, number, number]> = [
  ...CARDINAL_STEPS,
  [EDGE_DOWN_RIGHT, 1, 1],
  [EDGE_DOWN_LEFT, -1, 1],
  [EDGE_UP_RIGHT, 1, -1],
  [EDGE_UP_LEFT, -1, -1],
];

/** Analytic symmetric 3x3 smallest eigenvector via Jacobi rotations (deterministic, zero-alloc). */
function smallestEigenvector3x3(
  xx: number,
  xy: number,
  xz: number,
  yy: number,
  yz: number,
  zz: number,
): Vec3Tuple | null {
  const a = [xx, xy, xz, xy, yy, yz, xz, yz, zz];
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];

  for (let sweep = 0; sweep < 20; sweep++) {
    let p = 0;
    let q = 1;
    let maxVal = Math.abs(a[1] ?? 0);
    if (Math.abs(a[2] ?? 0) > maxVal) {
      maxVal = Math.abs(a[2] ?? 0);
      p = 0;
      q = 2;
    }
    if (Math.abs(a[5] ?? 0) > maxVal) {
      maxVal = Math.abs(a[5] ?? 0);
      p = 1;
      q = 2;
    }
    if (maxVal <= 1e-18) {
      break;
    }

    const app = a[p * 3 + p] ?? 0;
    const aqq = a[q * 3 + q] ?? 0;
    const apq = a[p * 3 + q] ?? 0;
    const tau = (aqq - app) / (2 * apq);
    const t =
      tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    a[p * 3 + p] = app - t * apq;
    a[q * 3 + q] = aqq + t * apq;
    a[p * 3 + q] = 0;
    a[q * 3 + p] = 0;

    for (let r = 0; r < 3; r++) {
      if (r === p || r === q) continue;
      const arp = a[r * 3 + p] ?? 0;
      const arq = a[r * 3 + q] ?? 0;
      const nextRp = c * arp - s * arq;
      const nextRq = s * arp + c * arq;
      a[r * 3 + p] = nextRp;
      a[p * 3 + r] = nextRp;
      a[r * 3 + q] = nextRq;
      a[q * 3 + r] = nextRq;
    }

    for (let r = 0; r < 3; r++) {
      const vrp = v[r * 3 + p] ?? 0;
      const vrq = v[r * 3 + q] ?? 0;
      v[r * 3 + p] = c * vrp - s * vrq;
      v[r * 3 + q] = s * vrp + c * vrq;
    }
  }

  const eigs = [a[0] ?? 0, a[4] ?? 0, a[8] ?? 0];
  let minCol = 0;
  if ((eigs[1] ?? Infinity) < (eigs[minCol] ?? Infinity)) minCol = 1;
  if ((eigs[2] ?? Infinity) < (eigs[minCol] ?? Infinity)) minCol = 2;

  const sorted = [...eigs].sort((x, y) => x - y);
  if ((sorted[1] ?? 0) <= 1e-14) {
    return null;
  }

  return vecNormalize([v[minCol] ?? 0, v[3 + minCol] ?? 0, v[6 + minCol] ?? -1]);
}

export function fitPlaneOnPointList(
  points: Float32Array,
  indices: readonly number[],
  normals: Float32Array | null,
  seedPoint: Vec3Tuple,
): PlanarFitSummary | null {
  if (indices.length < 3) {
    return null;
  }

  const runCovariance = (
    subset: readonly number[],
  ): { origin: Vec3Tuple; normal: Vec3Tuple } | null => {
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let i = 0; i < subset.length; i++) {
      const base = (subset[i] ?? 0) * 3;
      sx += points[base] ?? 0;
      sy += points[base + 1] ?? 0;
      sz += points[base + 2] ?? 0;
    }
    const invN = 1 / subset.length;
    const cx = sx * invN;
    const cy = sy * invN;
    const cz = sz * invN;

    let cxx = 0;
    let cxy = 0;
    let cxz = 0;
    let cyy = 0;
    let cyz = 0;
    let czz = 0;
    for (let i = 0; i < subset.length; i++) {
      const base = (subset[i] ?? 0) * 3;
      const dx = (points[base] ?? 0) - cx;
      const dy = (points[base + 1] ?? 0) - cy;
      const dz = (points[base + 2] ?? 0) - cz;
      cxx += dx * dx;
      cxy += dx * dy;
      cxz += dx * dz;
      cyy += dy * dy;
      cyz += dy * dz;
      czz += dz * dz;
    }
    const eigVec = smallestEigenvector3x3(cxx, cxy, cxz, cyy, cyz, czz);
    if (!eigVec) {
      return null;
    }
    const origin: Vec3Tuple = [cx, cy, cz];
    const normal = orientNormalTowardCamera(eigVec, origin);
    return { origin, normal };
  };

  let fit = runCovariance(indices);
  if (!fit) {
    return null;
  }

  // Single IRLS trimming pass rejects hard discontinuous outliers while preserving smooth curvature.
  if (indices.length >= 12) {
    const rawDists: number[] = [];
    for (let i = 0; i < indices.length; i++) {
      const b = (indices[i] ?? 0) * 3;
      const d =
        ((points[b] ?? 0) - fit.origin[0]) * fit.normal[0] +
        ((points[b + 1] ?? 0) - fit.origin[1]) * fit.normal[1] +
        ((points[b + 2] ?? 0) - fit.origin[2]) * fit.normal[2];
      rawDists.push(Math.abs(d));
    }
    const tol = 0.06 * Math.max(0.1, seedPoint[2]);

    const trimmed: number[] = [];
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i] ?? 0;
      if ((rawDists[i] ?? Infinity) > tol) {
        continue;
      }
      if (normals) {
        const nb = idx * 3;
        const unitN = vecNormalize([normals[nb] ?? 0, normals[nb + 1] ?? 0, normals[nb + 2] ?? 0]);
        if (unitN && vecDot(unitN, fit.normal) < 0.7) {
          continue;
        }
      }
      trimmed.push(idx);
    }
    if (trimmed.length >= 6 && trimmed.length < indices.length) {
      const refined = runCovariance(trimmed);
      if (refined) {
        fit = refined;
      }
    }
  }

  let sumSq = 0;
  let maxSpan = 0;
  const normalDots: number[] = [];
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i] ?? 0;
    const b = idx * 3;
    const px = points[b] ?? 0;
    const py = points[b + 1] ?? 0;
    const pz = points[b + 2] ?? 0;
    const dist =
      (px - fit.origin[0]) * fit.normal[0] +
      (py - fit.origin[1]) * fit.normal[1] +
      (pz - fit.origin[2]) * fit.normal[2];
    sumSq += dist * dist;
    const r = Math.hypot(px - seedPoint[0], py - seedPoint[1], pz - seedPoint[2]);
    if (r > maxSpan) maxSpan = r;

    if (normals) {
      const unitN = vecNormalize([normals[b] ?? 0, normals[b + 1] ?? 0, normals[b + 2] ?? 0]);
      if (unitN) {
        normalDots.push(vecDot(unitN, fit.normal));
      }
    }
  }

  const rmsError = Math.sqrt(sumSq / indices.length);
  let normalDotP08 = 1.0;
  if (normalDots.length >= 3) {
    normalDots.sort((a, b) => a - b);
    normalDotP08 = normalDots[Math.floor((normalDots.length - 1) * 0.08)] ?? 1.0;
  }

  return {
    plane: fit,
    rmsError,
    patchSpan: maxSpan,
    normalDotP08,
    sampleCount: indices.length,
  };
}

function stepToNeighborBit(dc: number, dr: number): number {
  if (dc === 1 && dr === 0) return EDGE_RIGHT;
  if (dc === -1 && dr === 0) return EDGE_LEFT;
  if (dc === 0 && dr === 1) return EDGE_DOWN;
  if (dc === 0 && dr === -1) return EDGE_UP;
  if (dc === 1 && dr === 1) return EDGE_DOWN_RIGHT;
  if (dc === -1 && dr === 1) return EDGE_DOWN_LEFT;
  if (dc === 1 && dr === -1) return EDGE_UP_RIGHT;
  if (dc === -1 && dr === -1) return EDGE_UP_LEFT;
  return 0;
}

export function collectLocalContinuousPatch(
  gridIndex: SurfaceGridIndex,
  seedCol: number,
  seedRow: number,
  windowRadius: number,
): number[] {
  const { width, height, validMask, neighborFlags } = gridIndex;
  const seedIdx = seedRow * width + seedCol;
  if ((validMask[seedIdx] ?? 0) === 0) {
    return [];
  }

  const denseRadius = Math.min(2, windowRadius);
  const minC = Math.max(0, seedCol - denseRadius);
  const maxC = Math.min(width - 1, seedCol + denseRadius);
  const minR = Math.max(0, seedRow - denseRadius);
  const maxR = Math.min(height - 1, seedRow + denseRadius);
  const boxW = maxC - minC + 1;
  const boxH = maxR - minR + 1;
  const visited = new Uint8Array(boxW * boxH);
  const queue = new Int32Array(boxW * boxH);
  const out: number[] = [];

  let head = 0;
  let tail = 0;
  visited[(seedRow - minR) * boxW + (seedCol - minC)] = 1;
  queue[tail++] = seedIdx;

  while (head < tail) {
    const cur = queue[head++] ?? 0;
    out.push(cur);
    const flags = neighborFlags[cur] ?? 0;
    if (flags === 0) continue;

    const r = Math.floor(cur / width);
    const c = cur - r * width;
    for (let k = 0; k < ALL_STEPS.length; k++) {
      const step = ALL_STEPS[k];
      if (!step) continue;
      const [bit, dc, dr] = step;
      if ((flags & bit) === 0) continue;
      const nc = c + dc;
      const nr = r + dr;
      if (nc < minC || nc > maxC || nr < minR || nr > maxR) continue;
      const localIdx = (nr - minR) * boxW + (nc - minC);
      if ((visited[localIdx] ?? 0) === 1) continue;
      const nIdx = nr * width + nc;
      if ((validMask[nIdx] ?? 0) === 0) continue;
      visited[localIdx] = 1;
      queue[tail++] = nIdx;
    }
  }

  if (windowRadius > denseRadius) {
    const rayDirs: ReadonlyArray<readonly [number, number]> = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
      [1, 1],
      [-1, 1],
      [1, -1],
      [-1, -1],
      [2, 1],
      [2, -1],
      [-2, 1],
      [-2, -1],
      [1, 2],
      [-1, 2],
      [1, -2],
      [-1, -2],
    ];
    const sampleStride = Math.max(1, Math.floor(windowRadius / 10));
    for (let d = 0; d < rayDirs.length; d++) {
      const dir = rayDirs[d];
      if (!dir) continue;
      let curCol = seedCol;
      let curRow = seedRow;
      let prevPx = seedCol;
      let prevPy = seedRow;
      const norm = Math.hypot(dir[0], dir[1]);
      const maxSteps = Math.round(
        windowRadius / (norm / Math.max(Math.abs(dir[0]), Math.abs(dir[1]))),
      );
      for (let s = 1; s <= maxSteps; s++) {
        const t = s / Math.max(Math.abs(dir[0]), Math.abs(dir[1]));
        const nextCol = seedCol + Math.round(dir[0] * t);
        const nextRow = seedRow + Math.round(dir[1] * t);
        const dc = Math.sign(nextCol - prevPx);
        const dr = Math.sign(nextRow - prevPy);
        if (dc === 0 && dr === 0) continue;
        if (nextCol < 0 || nextCol >= width || nextRow < 0 || nextRow >= height) break;
        const curIdx = curRow * width + curCol;
        const bit = stepToNeighborBit(dc, dr);
        if (((neighborFlags[curIdx] ?? 0) & bit) === 0) break;
        const nIdx = nextRow * width + nextCol;
        if ((validMask[nIdx] ?? 0) === 0) break;
        curCol = nextCol;
        curRow = nextRow;
        prevPx = nextCol;
        prevPy = nextRow;
        if (s > denseRadius && s % sampleStride === 0) {
          out.push(nIdx);
        }
      }
    }
  }

  return out;
}

function buildAutoPlanarSurface(
  geometry: GeometryInput,
  gridIndex: SurfaceGridIndex,
  fit: PlanarFitSummary,
  seedPoint: Vec3Tuple,
  imageSize: Size2D,
  nextColorIndex: number,
  nextSurfaceNumber: number,
): PlanarDecalSurface {
  const { width, height, validMask } = gridIndex;
  const total = width * height;
  const { plane, rmsError } = fit;
  const tol = Math.max(0.028 * seedPoint[2], 3.0 * rmsError);
  const minNormalDot = 0.78;
  const frame = PlaneFrame.fromPlane(plane, seedPoint);
  const ox = frame.origin[0];
  const oy = frame.origin[1];
  const oz = frame.origin[2];
  const nx = plane.normal[0];
  const ny = plane.normal[1];
  const nz = plane.normal[2];
  const ux = frame.uAxis[0];
  const uy = frame.uAxis[1];
  const uz = frame.uAxis[2];
  const vx = frame.vAxis[0];
  const vy = frame.vAxis[1];
  const vz = frame.vAxis[2];

  const points = geometry.points;
  const normals = geometry.normals;
  const supportMask = new Uint8Array(total);
  const intrinsics = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const pixelAngle = computePixelAngularSpan(intrinsics, width, height);

  const sampleCap = 4096;
  const sampleStep = Math.max(1, Math.floor(total / sampleCap));
  const uSample = new Float32Array(sampleCap);
  const vSample = new Float32Array(sampleCap);
  let sampleCount = 0;
  let inlierCount = 0;

  for (let i = 0; i < total; i++) {
    if ((validMask[i] ?? 0) === 0) continue;
    const b = i * 3;
    const dx = (points[b] ?? 0) - ox;
    const dy = (points[b + 1] ?? 0) - oy;
    const dz = (points[b + 2] ?? 0) - oz;
    const dist = Math.abs(dx * nx + dy * ny + dz * nz);
    if (dist > tol) continue;

    if (normals) {
      const gnx = normals[b] ?? 0;
      const gny = normals[b + 1] ?? 0;
      const gnz = normals[b + 2] ?? 0;
      const nLen = Math.hypot(gnx, gny, gnz);
      if (nLen > 1e-6 && (gnx * nx + gny * ny + gnz * nz) / nLen < minNormalDot) {
        continue;
      }
    }

    supportMask[i] = 1;
    inlierCount++;
    if (i % sampleStep === 0 && sampleCount < sampleCap) {
      uSample[sampleCount] = dx * ux + dy * uy + dz * uz;
      vSample[sampleCount] = dx * vx + dy * vy + dz * vz;
      sampleCount++;
    }
  }

  const uView = uSample.subarray(0, sampleCount).sort();
  const vView = vSample.subarray(0, sampleCount).sort();
  const trimLo = sampleCount >= 40 ? Math.floor((sampleCount - 1) * 0.01) : 0;
  const trimHi =
    sampleCount >= 40 ? Math.ceil((sampleCount - 1) * 0.99) : Math.max(0, sampleCount - 1);

  const rawMinU = Math.min(0, uView[trimLo] ?? 0);
  const rawMaxU = Math.max(0, uView[trimHi] ?? 0);
  const rawMinV = Math.min(0, vView[trimLo] ?? 0);
  const rawMaxV = Math.max(0, vView[trimHi] ?? 0);

  const minHalf = Math.max(1e-4, 4 * seedPoint[2] * pixelAngle);
  const spanU = Math.max(minHalf * 2, rawMaxU - rawMinU);
  const spanV = Math.max(minHalf * 2, rawMaxV - rawMinV);
  const padU = Math.max(minHalf, spanU * 0.1);
  const padV = Math.max(minHalf, spanV * 0.1);

  const footprint: PlaneFootprint = {
    minU: rawMinU - padU,
    maxU: rawMaxU + padU,
    minV: rawMinV - padV,
    maxV: rawMaxV + padV,
  };

  const avgFocal = 0.5 * (intrinsics.fx * imageSize.width + intrinsics.fy * imageSize.height);
  const refDiameter = Math.max(1e-4, (36 * seedPoint[2]) / Math.max(1, avgFocal));

  return {
    id: `auto-plane-${nextAutoPlaneSeq++}`,
    label: `Auto Plane ${nextSurfaceNumber}`,
    colorIndex: nextColorIndex % 6,
    frame,
    baseFootprint: footprint,
    extentScale: 1.0,
    supportMask,
    inlierCount,
    rmsError,
    referenceBrushDiameter: refDiameter,
  };
}

function findMatchingExistingPlanarSurface(
  surfaces: readonly PlanarDecalSurface[],
  hit: SurfaceHit,
  gridWidth: number,
): PlanarDecalSurface | null {
  if (!hit.valid || !hit.pixel || !hit.point3D) {
    return null;
  }
  const idx = hit.pixel.y * gridWidth + hit.pixel.x;
  let bestSurface: PlanarDecalSurface | null = null;
  let bestDist = Infinity;

  for (let i = 0; i < surfaces.length; i++) {
    const s = surfaces[i];
    if (!s) continue;
    const planeDist = Math.abs(vecDot(s.frame.normal, vecSub(hit.point3D, s.frame.origin)));
    const tol = Math.max(0.03 * hit.point3D[2], 3 * s.rmsError, 0.01);
    if (planeDist > tol) {
      continue;
    }
    if (hit.normal && vecDot(hit.normal, s.frame.normal) < 0.88) {
      continue;
    }
    if ((s.supportMask[idx] ?? 0) !== 0) {
      return s;
    }
    if (planeDist < bestDist) {
      bestDist = planeDist;
      bestSurface = s;
    }
  }
  return bestSurface;
}

export function isPlanarPatch(
  fit: PlanarFitSummary,
  seedDepth: number,
  hasNormals: boolean,
): boolean {
  if (fit.sampleCount < 6 || fit.patchSpan <= 1e-6) {
    return false;
  }
  const relRms = fit.rmsError / fit.patchSpan;
  if (relRms > 0.02 || fit.rmsError > 0.018 * seedDepth) {
    return false;
  }
  if (hasNormals && fit.normalDotP08 < 0.978) {
    return false;
  }
  return true;
}

export function classifyAndAnchorSurface(params: {
  geometry: GeometryInput | null;
  imagePoint: Point2D;
  imageSize: Size2D;
  brushPixelDiameter: number;
  existingSurfaces: readonly PlanarDecalSurface[];
  nextSurfaceNumber: number;
}): SurfaceAnchor {
  const {
    geometry,
    imagePoint,
    imageSize,
    brushPixelDiameter,
    existingSurfaces,
    nextSurfaceNumber,
  } = params;

  const fallbackRadiusPx = Math.max(1, brushPixelDiameter * 0.5);
  if (!geometry) {
    return {
      mode: 'fallback-2d',
      radiusPx: fallbackRadiusPx,
      reason: 'Scene geometry has not been generated yet.',
    };
  }

  const hit = pickSurfaceHit(geometry, imagePoint, imageSize);
  if (!hit.valid || !hit.pixel || !hit.point3D || !hit.depth) {
    return {
      mode: 'fallback-2d',
      radiusPx: fallbackRadiusPx,
      reason: 'Invalid or missing depth geometry at stroke start.',
    };
  }

  const gridIndex = getSurfaceGridIndex(geometry);
  const { width: infW, height: infH } = gridIndex;
  const intrinsics = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const surfaceRadius = computeSurfaceBrushRadius(
    hit.depth,
    brushPixelDiameter,
    intrinsics,
    imageSize,
  );

  // Collect connected neighborhood first so thin curved edges never get grabbed by an adjacent plane.
  const brushInfRadius = Math.ceil(
    (brushPixelDiameter * 0.5 * infW) / Math.max(1, imageSize.width),
  );
  const windowRadius = clamp(
    Math.max(12, Math.round(Math.max(infW, infH) * 0.14), Math.round(brushInfRadius * 2.5)),
    8,
    72,
  );
  const patchIndices = collectLocalContinuousPatch(
    gridIndex,
    hit.pixel.x,
    hit.pixel.y,
    windowRadius,
  );

  if (patchIndices.length < 6) {
    return {
      mode: 'fallback-2d',
      radiusPx: fallbackRadiusPx,
      reason: 'Isolated or sparse geometry region at stroke start.',
    };
  }

  const fit = fitPlaneOnPointList(geometry.points, patchIndices, geometry.normals, hit.point3D);
  const hasNormals = geometry.normals !== null;
  if (fit && isPlanarPatch(fit, hit.depth, hasNormals)) {
    const matched = findMatchingExistingPlanarSurface(existingSurfaces, hit, infW);
    if (matched) {
      return {
        mode: 'planar',
        surface: matched,
        reusedExisting: true,
        surfaceRadius,
        seedHit: hit,
      };
    }

    const newSurface = buildAutoPlanarSurface(
      geometry,
      gridIndex,
      fit,
      hit.point3D,
      imageSize,
      Math.max(0, nextSurfaceNumber - 1),
      nextSurfaceNumber,
    );
    return {
      mode: 'planar',
      surface: newSurface,
      reusedExisting: false,
      surfaceRadius,
      seedHit: hit,
    };
  }

  const metric = computeInterpolatedSurfaceMetric(gridIndex, geometry, imagePoint, imageSize);
  if (!metric.valid) {
    return {
      mode: 'fallback-2d',
      radiusPx: fallbackRadiusPx,
      reason: 'Degenerate local surface metric; using 2D fallback.',
    };
  }

  const componentMask = findConnectedMeshComponent(gridIndex, hit.pixel.x, hit.pixel.y);
  return {
    mode: 'curved',
    componentMask,
    surfaceRadius,
    seedHit: hit,
  };
}
