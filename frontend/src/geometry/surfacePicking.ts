import type { Point2D, Size2D } from '../types/editor';
import {
  type GeometryInput,
  type Vec3Tuple,
  imagePointToInferencePixel,
  isValidGridPixel,
  readNormalOpenCV,
  readPointOpenCV,
} from './geometryAccess';
import {
  type MeshContinuityParams,
  buildQuadTriangles,
  computeDefaultContinuityParams,
  isEdgeContinuous,
  isTriangleContinuous,
  readPointTupleAt,
} from './meshContinuity';

export const EDGE_RIGHT = 1;
export const EDGE_LEFT = 2;
export const EDGE_DOWN = 4;
export const EDGE_UP = 8;
export const EDGE_DOWN_RIGHT = 16;
export const EDGE_DOWN_LEFT = 32;
export const EDGE_UP_RIGHT = 64;
export const EDGE_UP_LEFT = 128;

export interface SurfaceHit {
  imagePoint: Point2D;
  pixel: Point2D | null;
  point3D: Vec3Tuple | null;
  normal: Vec3Tuple | null;
  depth: number | null;
  triangleVertices: [number, number, number] | null;
  barycentric: [number, number, number] | null;
  valid: boolean;
}

export interface SurfaceGridIndex {
  width: number;
  height: number;
  continuity: MeshContinuityParams;
  validMask: Uint8Array;
  neighborFlags: Uint8Array;
}

const gridIndexCache = new WeakMap<Float32Array, SurfaceGridIndex>();

function markSymmetricEdge(flags: Uint8Array, idxA: number, idxB: number, width: number): void {
  const rowA = Math.floor(idxA / width);
  const colA = idxA - rowA * width;
  const rowB = Math.floor(idxB / width);
  const colB = idxB - rowB * width;
  const dc = colB - colA;
  const dr = rowB - rowA;

  if (dc === 1 && dr === 0) {
    flags[idxA] = (flags[idxA] ?? 0) | EDGE_RIGHT;
    flags[idxB] = (flags[idxB] ?? 0) | EDGE_LEFT;
  } else if (dc === -1 && dr === 0) {
    flags[idxA] = (flags[idxA] ?? 0) | EDGE_LEFT;
    flags[idxB] = (flags[idxB] ?? 0) | EDGE_RIGHT;
  } else if (dc === 0 && dr === 1) {
    flags[idxA] = (flags[idxA] ?? 0) | EDGE_DOWN;
    flags[idxB] = (flags[idxB] ?? 0) | EDGE_UP;
  } else if (dc === 0 && dr === -1) {
    flags[idxA] = (flags[idxA] ?? 0) | EDGE_UP;
    flags[idxB] = (flags[idxB] ?? 0) | EDGE_DOWN;
  } else if (dc === 1 && dr === 1) {
    flags[idxA] = (flags[idxA] ?? 0) | EDGE_DOWN_RIGHT;
    flags[idxB] = (flags[idxB] ?? 0) | EDGE_UP_LEFT;
  } else if (dc === -1 && dr === -1) {
    flags[idxA] = (flags[idxA] ?? 0) | EDGE_UP_LEFT;
    flags[idxB] = (flags[idxB] ?? 0) | EDGE_DOWN_RIGHT;
  } else if (dc === -1 && dr === 1) {
    flags[idxA] = (flags[idxA] ?? 0) | EDGE_DOWN_LEFT;
    flags[idxB] = (flags[idxB] ?? 0) | EDGE_UP_RIGHT;
  } else if (dc === 1 && dr === -1) {
    flags[idxA] = (flags[idxA] ?? 0) | EDGE_UP_RIGHT;
    flags[idxB] = (flags[idxB] ?? 0) | EDGE_DOWN_LEFT;
  }
}

export function buildSurfaceGridIndex(geometry: GeometryInput): SurfaceGridIndex {
  const width = geometry.metadata.inference_width;
  const height = geometry.metadata.inference_height;
  const total = width * height;
  const continuity = computeDefaultContinuityParams(geometry);
  const validMask = new Uint8Array(total);
  const neighborFlags = new Uint8Array(total);

  for (let i = 0; i < total; i++) {
    validMask[i] = isValidGridPixel(geometry, i) ? 1 : 0;
  }

  for (let row = 0; row < height - 1; row++) {
    for (let col = 0; col < width - 1; col++) {
      const candidates = buildQuadTriangles(geometry, validMask, row, col);
      let continuousCount = 0;
      for (let k = 0; k < candidates.length; k++) {
        const tri = candidates[k];
        if (!tri || !isTriangleContinuous(tri.i0, tri.i1, tri.i2, geometry, continuity)) {
          continue;
        }
        continuousCount += 1;
        markSymmetricEdge(neighborFlags, tri.i0, tri.i1, width);
        markSymmetricEdge(neighborFlags, tri.i1, tri.i2, width);
        markSymmetricEdge(neighborFlags, tri.i2, tri.i0, width);
      }
      // On fully continuous quads, allow symmetric traversal along the cross-diagonal as well
      // so geodesic distances do not exhibit arbitrary split-orientation bias.
      if (continuousCount === 2) {
        const tl = row * width + col;
        const tr = tl + 1;
        const bl = (row + 1) * width + col;
        const br = bl + 1;
        const pTl = readPointTupleAt(geometry.points, tl);
        const pBr = readPointTupleAt(geometry.points, br);
        if (isEdgeContinuous(tl, br, width, pTl, pBr, continuity)) {
          markSymmetricEdge(neighborFlags, tl, br, width);
        }
        const pTr = readPointTupleAt(geometry.points, tr);
        const pBl = readPointTupleAt(geometry.points, bl);
        if (isEdgeContinuous(tr, bl, width, pTr, pBl, continuity)) {
          markSymmetricEdge(neighborFlags, tr, bl, width);
        }
      }
    }
  }

  return { width, height, continuity, validMask, neighborFlags };
}

export function getSurfaceGridIndex(geometry: GeometryInput): SurfaceGridIndex {
  const cached = gridIndexCache.get(geometry.points);
  if (
    cached &&
    cached.width === geometry.metadata.inference_width &&
    cached.height === geometry.metadata.inference_height
  ) {
    return cached;
  }
  const built = buildSurfaceGridIndex(geometry);
  gridIndexCache.set(geometry.points, built);
  return built;
}

function pointBarycentricIn2D(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
): [number, number, number] | null {
  const v0x = bx - ax;
  const v0y = by - ay;
  const v1x = cx - ax;
  const v1y = cy - ay;
  const v2x = px - ax;
  const v2y = py - ay;
  const denom = v0x * v1y - v1x * v0y;
  if (Math.abs(denom) <= 1e-10) {
    return null;
  }
  const inv = 1 / denom;
  const w1 = (v2x * v1y - v1x * v2y) * inv;
  const w2 = (v0x * v2y - v2x * v0y) * inv;
  const w0 = 1 - w1 - w2;
  if (w0 < -1e-5 || w1 < -1e-5 || w2 < -1e-5) {
    return null;
  }
  const cw0 = Math.max(0, w0);
  const cw1 = Math.max(0, w1);
  const cw2 = Math.max(0, w2);
  const sum = cw0 + cw1 + cw2;
  return sum > 0 ? [cw0 / sum, cw1 / sum, cw2 / sum] : null;
}

export function findContinuousTriangleAtGridPoint(
  gridIndex: SurfaceGridIndex,
  geometry: GeometryInput,
  gx: number,
  gy: number,
): { vertices: [number, number, number]; barycentric: [number, number, number] } | null {
  const { width, height, validMask, continuity } = gridIndex;
  if (width < 2 || height < 2) {
    return null;
  }
  // Grid vertex (col, row) has center at (col + 0.5, row + 0.5).
  const vx = gx - 0.5;
  const vy = gy - 0.5;
  const col = Math.min(width - 2, Math.max(0, Math.floor(vx)));
  const row = Math.min(height - 2, Math.max(0, Math.floor(vy)));

  const candidates = buildQuadTriangles(geometry, validMask, row, col);
  for (let i = 0; i < candidates.length; i++) {
    const tri = candidates[i];
    if (!tri || !isTriangleContinuous(tri.i0, tri.i1, tri.i2, geometry, continuity)) {
      continue;
    }
    const r0 = Math.floor(tri.i0 / width);
    const c0 = tri.i0 - r0 * width;
    const r1 = Math.floor(tri.i1 / width);
    const c1 = tri.i1 - r1 * width;
    const r2 = Math.floor(tri.i2 / width);
    const c2 = tri.i2 - r2 * width;
    const bary = pointBarycentricIn2D(vx, vy, c0, r0, c1, r1, c2, r2);
    if (bary) {
      return { vertices: [tri.i0, tri.i1, tri.i2], barycentric: bary };
    }
  }
  return null;
}

export function pickSurfaceHit(
  geometry: GeometryInput | null,
  imagePoint: Point2D,
  imageSize: Size2D,
): SurfaceHit {
  if (!geometry || imageSize.width <= 0 || imageSize.height <= 0) {
    return {
      imagePoint,
      pixel: null,
      point3D: null,
      normal: null,
      depth: null,
      triangleVertices: null,
      barycentric: null,
      valid: false,
    };
  }

  const infW = geometry.metadata.inference_width;
  const infH = geometry.metadata.inference_height;
  const pixel = imagePointToInferencePixel(imagePoint, imageSize, infW, infH);
  if (!pixel) {
    return {
      imagePoint,
      pixel: null,
      point3D: null,
      normal: null,
      depth: null,
      triangleVertices: null,
      barycentric: null,
      valid: false,
    };
  }

  const point3D = readPointOpenCV(geometry, pixel.x, pixel.y);
  if (!point3D) {
    return {
      imagePoint,
      pixel,
      point3D: null,
      normal: null,
      depth: null,
      triangleVertices: null,
      barycentric: null,
      valid: false,
    };
  }

  const gridIndex = getSurfaceGridIndex(geometry);
  const gx = (imagePoint.x * infW) / imageSize.width;
  const gy = (imagePoint.y * infH) / imageSize.height;
  const triMatch = findContinuousTriangleAtGridPoint(gridIndex, geometry, gx, gy);

  const normal = readNormalOpenCV(geometry, pixel.x, pixel.y);
  let hitPoint = point3D;
  if (triMatch) {
    const [ia, ib, ic] = triMatch.vertices;
    const [w0, w1, w2] = triMatch.barycentric;
    const pa = geometry.points;
    const aBase = ia * 3;
    const bBase = ib * 3;
    const cBase = ic * 3;
    const interpZ =
      w0 * (pa[aBase + 2] ?? 0) + w1 * (pa[bBase + 2] ?? 0) + w2 * (pa[cBase + 2] ?? 0);
    if (Number.isFinite(interpZ) && interpZ > 1e-5) {
      hitPoint = [
        w0 * (pa[aBase] ?? 0) + w1 * (pa[bBase] ?? 0) + w2 * (pa[cBase] ?? 0),
        w0 * (pa[aBase + 1] ?? 0) + w1 * (pa[bBase + 1] ?? 0) + w2 * (pa[cBase + 1] ?? 0),
        interpZ,
      ];
    }
  }

  return {
    imagePoint,
    pixel,
    point3D: hitPoint,
    normal,
    depth: hitPoint[2],
    triangleVertices: triMatch?.vertices ?? null,
    barycentric: triMatch?.barycentric ?? null,
    valid: true,
  };
}

const NEIGHBOR_OFFSETS: ReadonlyArray<readonly [number, number, number]> = [
  [EDGE_RIGHT, 1, 0],
  [EDGE_LEFT, -1, 0],
  [EDGE_DOWN, 0, 1],
  [EDGE_UP, 0, -1],
  [EDGE_DOWN_RIGHT, 1, 1],
  [EDGE_DOWN_LEFT, -1, 1],
  [EDGE_UP_RIGHT, 1, -1],
  [EDGE_UP_LEFT, -1, -1],
];

export function findConnectedMeshComponent(
  gridIndex: SurfaceGridIndex,
  seedCol: number,
  seedRow: number,
  maxNodes = gridIndex.width * gridIndex.height,
): Uint8Array {
  const { width, height, validMask, neighborFlags } = gridIndex;
  const component = new Uint8Array(width * height);
  if (seedCol < 0 || seedCol >= width || seedRow < 0 || seedRow >= height) {
    return component;
  }
  const seedIdx = seedRow * width + seedCol;
  if ((validMask[seedIdx] ?? 0) === 0) {
    return component;
  }

  const queue = new Int32Array(Math.min(maxNodes, width * height));
  let head = 0;
  let tail = 0;
  component[seedIdx] = 1;
  queue[tail++] = seedIdx;

  while (head < tail && tail < queue.length) {
    const cur = queue[head++] ?? 0;
    const flags = neighborFlags[cur] ?? 0;
    if (flags === 0) {
      continue;
    }
    const r = Math.floor(cur / width);
    const c = cur - r * width;
    for (let k = 0; k < NEIGHBOR_OFFSETS.length; k++) {
      const entry = NEIGHBOR_OFFSETS[k];
      if (!entry) continue;
      const [bit, dc, dr] = entry;
      if ((flags & bit) === 0) {
        continue;
      }
      const nc = c + dc;
      const nr = r + dr;
      if (nc < 0 || nc >= width || nr < 0 || nr >= height) {
        continue;
      }
      const nIdx = nr * width + nc;
      if ((component[nIdx] ?? 0) === 1 || (validMask[nIdx] ?? 0) === 0) {
        continue;
      }
      component[nIdx] = 1;
      if (tail < queue.length) {
        queue[tail++] = nIdx;
      }
    }
  }

  return component;
}
