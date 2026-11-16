import {
  type GeometryInput,
  type Vec3Tuple,
  computePixelAngularSpan,
  parseNormalizedIntrinsics,
} from './geometryAccess';

export interface MeshContinuityParams {
  pixelAngle: number;
  maxSlopeFactor: number;
  maxRelDepthDiff: number;
  minNormalDot: number;
}

export interface QuadTriangleCandidate {
  i0: number;
  i1: number;
  i2: number;
}

export function computeDefaultContinuityParams(
  geometry: GeometryInput,
  options?: {
    maxSlopeFactor?: number;
    maxRelativeDepthDiff?: number;
    minNormalDot?: number;
  },
): MeshContinuityParams {
  const width = geometry.metadata.inference_width;
  const height = geometry.metadata.inference_height;
  const intrinsics = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const pixelAngle = computePixelAngularSpan(intrinsics, width, height);
  return {
    pixelAngle,
    maxSlopeFactor: options?.maxSlopeFactor ?? 8.0,
    maxRelDepthDiff:
      options?.maxRelativeDepthDiff ?? Math.max(0.06, Math.min(0.35, 6 * pixelAngle)),
    minNormalDot: options?.minNormalDot ?? 0.2,
  };
}

export function readPointTupleAt(points: Float32Array, idx: number): Vec3Tuple {
  const base = idx * 3;
  return [points[base] ?? 0, points[base + 1] ?? 0, points[base + 2] ?? 0];
}

export function readUnitNormalAt(normals: Float32Array, idx: number): Vec3Tuple | null {
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

export function distSq3(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

export function isEdgeContinuous(
  idxA: number,
  idxB: number,
  gridWidth: number,
  pA: readonly [number, number, number],
  pB: readonly [number, number, number],
  params: MeshContinuityParams,
): boolean {
  const rowA = Math.floor(idxA / gridWidth);
  const colA = idxA - rowA * gridWidth;
  const rowB = Math.floor(idxB / gridWidth);
  const colB = idxB - rowB * gridWidth;
  const gridDist = Math.abs(rowA - rowB) + Math.abs(colA - colB);
  const edgeWeight = gridDist > 1 ? 1.45 : 1.0;

  const minZ = Math.min(pA[2], pB[2]);
  if (minZ <= 0) {
    return false;
  }
  const relDepthDiff = Math.abs(pA[2] - pB[2]) / minZ;
  if (relDepthDiff > params.maxRelDepthDiff * edgeWeight) {
    return false;
  }

  const maxEdgeLen = minZ * params.pixelAngle * params.maxSlopeFactor * edgeWeight;
  return distSq3(pA, pB) <= maxEdgeLen * maxEdgeLen;
}

export function isTriangleContinuous(
  ia: number,
  ib: number,
  ic: number,
  geometry: GeometryInput,
  params: MeshContinuityParams,
): boolean {
  const width = geometry.metadata.inference_width;
  const pa = readPointTupleAt(geometry.points, ia);
  const pb = readPointTupleAt(geometry.points, ib);
  const pc = readPointTupleAt(geometry.points, ic);

  if (
    !isEdgeContinuous(ia, ib, width, pa, pb, params) ||
    !isEdgeContinuous(ib, ic, width, pb, pc, params) ||
    !isEdgeContinuous(ic, ia, width, pc, pa, params)
  ) {
    return false;
  }

  const abx = pb[0] - pa[0];
  const aby = pb[1] - pa[1];
  const abz = pb[2] - pa[2];
  const acx = pc[0] - pa[0];
  const acy = pc[1] - pa[1];
  const acz = pc[2] - pa[2];
  const abSq = abx * abx + aby * aby + abz * abz;
  const acSq = acx * acx + acy * acy + acz * acz;
  const bcSq = distSq3(pb, pc);
  if (abSq <= 1e-16 || acSq <= 1e-16 || bcSq <= 1e-16) {
    return false;
  }

  const cx = aby * acz - abz * acy;
  const cy = abz * acx - abx * acz;
  const cz = abx * acy - aby * acx;
  const crossSq = cx * cx + cy * cy + cz * cz;
  if (crossSq <= 1e-12 * abSq * acSq) {
    return false;
  }

  if (geometry.normals) {
    const na = readUnitNormalAt(geometry.normals, ia);
    const nb = readUnitNormalAt(geometry.normals, ib);
    const nc = readUnitNormalAt(geometry.normals, ic);
    if (na && nb && nc) {
      const dab = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
      const dbc = nb[0] * nc[0] + nb[1] * nc[1] + nb[2] * nc[2];
      const dca = nc[0] * na[0] + nc[1] * na[1] + nc[2] * na[2];
      if (dab < params.minNormalDot || dbc < params.minNormalDot || dca < params.minNormalDot) {
        return false;
      }
    }
  }

  return true;
}

/** Enumerates candidate triangles for a 2x2 grid quad using the shorter 3D diagonal split. */
export function buildQuadTriangles(
  geometry: GeometryInput,
  vertexValid: Uint8Array,
  row: number,
  col: number,
): QuadTriangleCandidate[] {
  const width = geometry.metadata.inference_width;
  const i00 = row * width + col;
  const i01 = i00 + 1;
  const i10 = (row + 1) * width + col;
  const i11 = i10 + 1;

  const v00 = vertexValid[i00] === 1;
  const v01 = vertexValid[i01] === 1;
  const v10 = vertexValid[i10] === 1;
  const v11 = vertexValid[i11] === 1;
  const validCount = (v00 ? 1 : 0) + (v01 ? 1 : 0) + (v10 ? 1 : 0) + (v11 ? 1 : 0);
  if (validCount < 3) {
    return [];
  }

  if (validCount === 4) {
    const p00 = readPointTupleAt(geometry.points, i00);
    const p01 = readPointTupleAt(geometry.points, i01);
    const p10 = readPointTupleAt(geometry.points, i10);
    const p11 = readPointTupleAt(geometry.points, i11);
    if (distSq3(p00, p11) <= distSq3(p01, p10)) {
      return [
        { i0: i00, i1: i10, i2: i11 },
        { i0: i00, i1: i11, i2: i01 },
      ];
    }
    return [
      { i0: i00, i1: i10, i2: i01 },
      { i0: i01, i1: i10, i2: i11 },
    ];
  }

  if (!v00) return [{ i0: i01, i1: i10, i2: i11 }];
  if (!v01) return [{ i0: i00, i1: i10, i2: i11 }];
  if (!v10) return [{ i0: i00, i1: i11, i2: i01 }];
  return [{ i0: i00, i1: i10, i2: i01 }];
}
