import type { Point2D, Size2D } from '../types/editor';
import { CurvedBrushWorkspace } from './curvedBrushSolver';
import {
  type GeometryInput,
  type ParsedIntrinsics,
  type Vec3Tuple,
  parseNormalizedIntrinsics,
  projectOpenCvPointToImage,
  readNormalOpenCV,
  vecAdd,
  vecCross,
  vecDot,
  vecLength,
  vecNormalize,
  vecScale,
  vecSub,
} from './geometryAccess';
import {
  type MeshContinuityParams,
  buildQuadTriangles,
  computeDefaultContinuityParams,
  isEdgeContinuous,
  isTriangleContinuous,
  readPointTupleAt,
} from './meshContinuity';
import { PlaneFrame, orientNormalTowardCamera } from './planeFrame';
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
  getSurfaceGridIndex,
  pickSurfaceHit,
} from './surfacePicking';

export interface SurfaceMappingDiagnostics {
  visitedVertexCount: number;
  acceptedVertexCount: number;
  candidateTriangleCount: number;
  acceptedTriangleCount: number;
  flippedTriangleCount: number;
  excessiveStretchTriangleCount: number;
  maxStretchRatio: number;
  meanStretchError: number;
  anchorStretchError: number;
  usedEstimatedNormals: boolean;
  rejectionReason: string | null;
}

export interface SurfaceMappingSample {
  valid: boolean;
  u: number;
  v: number;
  depth: number;
  point3D: Vec3Tuple;
  tangentU: Vec3Tuple;
  tangentV: Vec3Tuple;
  imagePoint: Point2D;
}

interface UvLookupGrid {
  minU: number;
  minV: number;
  cellW: number;
  cellH: number;
  cols: number;
  rows: number;
  buckets: number[][];
}

export interface SurfaceMapping {
  readonly id: string;
  readonly geometryId: string;
  readonly anchorImagePoint: Point2D;
  readonly anchorPoint3D: Vec3Tuple;
  readonly anchorNormal: Vec3Tuple;
  readonly anchorFrame: PlaneFrame;
  readonly patchRadius: number;
  readonly inferenceWidth: number;
  readonly inferenceHeight: number;
  readonly gridStride: number;
  readonly gridOffsetX: number;
  readonly gridOffsetY: number;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly vertexGridIndices: Int32Array;
  readonly vertexPoints3D: Float32Array;
  readonly vertexPhotoUvs: Float32Array;
  readonly vertexSurfaceUvs: Float32Array;
  readonly vertexDepths: Float32Array;
  readonly triangleIndices: Uint32Array;
  readonly minU: number;
  readonly maxU: number;
  readonly minV: number;
  readonly maxV: number;
  readonly diagnostics: SurfaceMappingDiagnostics;
  readonly quadTriangleMap: Map<number, number[]>;
  readonly uvGrid: UvLookupGrid;
}

export interface LocalSurfaceParameterizationOutcome {
  valid: boolean;
  mapping: SurfaceMapping | null;
  diagnostics: SurfaceMappingDiagnostics;
}

export interface LocalSurfaceParameterizerOptions {
  geometry: GeometryInput;
  geometryId?: string;
  gridIndex?: SurfaceGridIndex;
  imagePoint: Point2D;
  imageSize: Size2D;
  maxRadius: number;
  maxVisitedNodes?: number;
  gridStride?: number;
  maxStretch?: number;
  maxAnisotropy?: number;
  relaxationPasses?: number;
  id?: string;
}

const NEIGHBOR_MASKS = [
  EDGE_RIGHT,
  EDGE_LEFT,
  EDGE_DOWN,
  EDGE_UP,
  EDGE_DOWN_RIGHT,
  EDGE_DOWN_LEFT,
  EDGE_UP_RIGHT,
  EDGE_UP_LEFT,
] as const;

const NEIGHBOR_DC = [1, -1, 0, 0, 1, -1, 1, -1] as const;
const NEIGHBOR_DR = [0, 0, 1, -1, 1, 1, -1, -1] as const;

let mappingCounter = 0;

function nextMappingId(): string {
  mappingCounter += 1;
  return `surf-map-${mappingCounter}-${Date.now().toString(36)}`;
}

function makeRejectedOutcome(
  rejectionReason: string,
  partial?: Partial<SurfaceMappingDiagnostics>,
): LocalSurfaceParameterizationOutcome {
  const diagnostics: SurfaceMappingDiagnostics = {
    visitedVertexCount: partial?.visitedVertexCount ?? 0,
    acceptedVertexCount: partial?.acceptedVertexCount ?? 0,
    candidateTriangleCount: partial?.candidateTriangleCount ?? 0,
    acceptedTriangleCount: partial?.acceptedTriangleCount ?? 0,
    flippedTriangleCount: partial?.flippedTriangleCount ?? 0,
    excessiveStretchTriangleCount: partial?.excessiveStretchTriangleCount ?? 0,
    maxStretchRatio: partial?.maxStretchRatio ?? Infinity,
    meanStretchError: partial?.meanStretchError ?? Infinity,
    anchorStretchError: partial?.anchorStretchError ?? Infinity,
    usedEstimatedNormals: partial?.usedEstimatedNormals ?? false,
    rejectionReason,
  };
  return { valid: false, mapping: null, diagnostics };
}

/**
 * Computes minimum SO(3) rotation taking unit normal `nPrev` to unit normal `nNext`
 * and applies it to `vec` without trigonometric calls.
 */
function rotateVectorBetweenUnitNormals(
  vec: Vec3Tuple,
  nPrev: Vec3Tuple,
  nNext: Vec3Tuple,
): Vec3Tuple {
  const c = vecDot(nPrev, nNext);
  if (c > 0.999999) {
    return vec;
  }
  if (c < -0.9999) {
    // 180-degree antiparallel fallback around an orthogonal vector.
    const ref: Vec3Tuple = Math.abs(nPrev[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0];
    const rotAxis = vecNormalize(vecCross(nPrev, ref)) ?? [0, 0, 1];
    const dot2 = 2 * vecDot(rotAxis, vec);
    return [rotAxis[0] * dot2 - vec[0], rotAxis[1] * dot2 - vec[1], rotAxis[2] * dot2 - vec[2]];
  }

  const k = vecCross(nPrev, nNext);
  const kCrossV = vecCross(k, vec);
  const kDotV = vecDot(k, vec) / (1 + c);
  return [
    vec[0] * c + kCrossV[0] + k[0] * kDotV,
    vec[1] * c + kCrossV[1] + k[1] * kDotV,
    vec[2] * c + kCrossV[2] + k[2] * kDotV,
  ];
}

function orthonormalizeTangentPair(
  rawU: Vec3Tuple,
  rawV: Vec3Tuple,
  normal: Vec3Tuple,
): { u: Vec3Tuple; v: Vec3Tuple } {
  const uProj = vecSub(rawU, vecScale(normal, vecDot(rawU, normal)));
  let u = vecNormalize(uProj);
  if (!u) {
    const ref: Vec3Tuple = Math.abs(normal[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0];
    u = vecNormalize(vecCross(normal, ref)) ?? [1, 0, 0];
  }
  // Keep handedness consistent with PlaneFrame: u x v points along -normal.
  let v = vecNormalize(vecCross(u, normal));
  if (!v) {
    v = vecNormalize(vecSub(rawV, vecScale(normal, vecDot(rawV, normal)))) ?? [0, 1, 0];
  }
  if (vecDot(v, rawV) < 0) {
    v = vecScale(v, -1);
    u = vecNormalize(vecCross(normal, v)) ?? u;
  }
  return { u, v };
}

function computeMidpointBasisPair(
  uA: Vec3Tuple,
  vA: Vec3Tuple,
  uB: Vec3Tuple,
  vB: Vec3Tuple,
): { mU: Vec3Tuple; mV: Vec3Tuple } {
  const sumU = vecAdd(uA, uB);
  const sumV = vecAdd(vA, vB);
  const mU = vecNormalize(sumU) ?? uA;
  const mV = vecNormalize(sumV) ?? vA;
  return { mU, mV };
}

function estimateVertexNormalFromPatch(
  gridIdx: number,
  geometry: GeometryInput,
  gridIndex: SurfaceGridIndex,
): Vec3Tuple | null {
  const { width, height, validMask, continuity } = gridIndex;
  const row = Math.floor(gridIdx / width);
  const col = gridIdx - row * width;
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let count = 0;

  for (let dr = -1; dr <= 0; dr++) {
    const r = row + dr;
    if (r < 0 || r >= height - 1) continue;
    for (let dc = -1; dc <= 0; dc++) {
      const c = col + dc;
      if (c < 0 || c >= width - 1) continue;
      const tris = buildQuadTriangles(geometry, validMask, r, c);
      for (let t = 0; t < tris.length; t++) {
        const tri = tris[t];
        if (!tri || (tri.i0 !== gridIdx && tri.i1 !== gridIdx && tri.i2 !== gridIdx)) {
          continue;
        }
        if (!isTriangleContinuous(tri.i0, tri.i1, tri.i2, geometry, continuity)) {
          continue;
        }
        const p0 = readPointTupleAt(geometry.points, tri.i0);
        const p1 = readPointTupleAt(geometry.points, tri.i1);
        const p2 = readPointTupleAt(geometry.points, tri.i2);
        const n = vecCross(vecSub(p1, p0), vecSub(p2, p0));
        const len = vecLength(n);
        if (len > 1e-10) {
          sx += n[0];
          sy += n[1];
          sz += n[2];
          count += 1;
        }
      }
    }
  }

  if (count === 0) {
    return null;
  }
  const p = readPointTupleAt(geometry.points, gridIdx);
  return orientNormalTowardCamera([sx, sy, sz], p);
}

interface TriangleMetricEvaluation {
  oriented: boolean;
  sMin: number;
  sMax: number;
  anisotropy: number;
  stretchError: number;
}

function evaluateTriangleParameterization(
  p0: Vec3Tuple,
  p1: Vec3Tuple,
  p2: Vec3Tuple,
  u0: number,
  v0: number,
  u1: number,
  v1: number,
  u2: number,
  v2: number,
  localFrameCross: Vec3Tuple,
): TriangleMetricEvaluation {
  const edge1 = vecSub(p1, p0);
  const edge2 = vecSub(p2, p0);
  const len1 = vecLength(edge1);
  if (!Number.isFinite(len1) || len1 <= 1e-8) {
    return {
      oriented: false,
      sMin: 0,
      sMax: Infinity,
      anisotropy: Infinity,
      stretchError: Infinity,
    };
  }
  const n3 = vecCross(edge1, edge2);
  const area2 = vecLength(n3);
  if (!Number.isFinite(area2) || area2 <= 1e-10) {
    return {
      oriented: false,
      sMin: 0,
      sMax: Infinity,
      anisotropy: Infinity,
      stretchError: Infinity,
    };
  }

  const n3Unit: Vec3Tuple = [n3[0] / area2, n3[1] / area2, n3[2] / area2];
  const t1: Vec3Tuple = [edge1[0] / len1, edge1[1] / len1, edge1[2] / len1];
  const t2 = vecCross(n3Unit, t1);

  const x1 = len1;
  const x2 = vecDot(edge2, t1);
  const y2 = vecDot(edge2, t2);
  if (Math.abs(y2) <= 1e-8) {
    return {
      oriented: false,
      sMin: 0,
      sMax: Infinity,
      anisotropy: Infinity,
      stretchError: Infinity,
    };
  }

  const du1 = u1 - u0;
  const dv1 = v1 - v0;
  const du2 = u2 - u0;
  const dv2 = v2 - v0;
  const areaUv2 = du1 * dv2 - dv1 * du2;
  const orient3D = vecDot(n3Unit, localFrameCross);
  // localFrameCross = uAxis x vAxis. Expected sign matches areaUv2 * orient3D > 0.
  if (areaUv2 * orient3D <= 1e-12) {
    return {
      oriented: false,
      sMin: 0,
      sMax: Infinity,
      anisotropy: Infinity,
      stretchError: Infinity,
    };
  }

  const invX1 = 1 / x1;
  const invY2 = 1 / y2;
  const x2OverX1 = x2 * invX1;

  const j00 = du1 * invX1;
  const j01 = (du2 - du1 * x2OverX1) * invY2;
  const j10 = dv1 * invX1;
  const j11 = (dv2 - dv1 * x2OverX1) * invY2;

  const p = j00 * j00 + j10 * j10;
  const q = j01 * j01 + j11 * j11;
  const r = j00 * j01 + j10 * j11;
  const traceHalf = 0.5 * (p + q);
  const diffHalf = 0.5 * (p - q);
  const rad = Math.hypot(diffHalf, r);
  const sMax = Math.sqrt(Math.max(0, traceHalf + rad));
  const sMin = Math.sqrt(Math.max(0, traceHalf - rad));
  const anisotropy = sMin > 1e-7 ? sMax / sMin : Infinity;
  const stretchError = Math.max(Math.abs(sMax - 1), Math.abs(sMin - 1));

  return { oriented: true, sMin, sMax, anisotropy, stretchError };
}

function readBufVec3(arr: Float32Array, vertexIndex: number): Vec3Tuple {
  const base = vertexIndex * 3;
  return [arr[base] ?? 0, arr[base + 1] ?? 0, arr[base + 2] ?? 0];
}

function hasGridEdge(flags: Uint8Array, width: number, idxA: number, idxB: number): boolean {
  const rowA = Math.floor(idxA / width);
  const colA = idxA - rowA * width;
  const rowB = Math.floor(idxB / width);
  const colB = idxB - rowB * width;
  const dc = colB - colA;
  const dr = rowB - rowA;
  const f = flags[idxA] ?? 0;
  if (dc === 1 && dr === 0) return (f & EDGE_RIGHT) !== 0;
  if (dc === -1 && dr === 0) return (f & EDGE_LEFT) !== 0;
  if (dc === 0 && dr === 1) return (f & EDGE_DOWN) !== 0;
  if (dc === 0 && dr === -1) return (f & EDGE_UP) !== 0;
  if (dc === 1 && dr === 1) return (f & EDGE_DOWN_RIGHT) !== 0;
  if (dc === -1 && dr === 1) return (f & EDGE_DOWN_LEFT) !== 0;
  if (dc === 1 && dr === -1) return (f & EDGE_UP_RIGHT) !== 0;
  if (dc === -1 && dr === -1) return (f & EDGE_UP_LEFT) !== 0;
  return false;
}

function buildUvLookupGrid(
  vertexSurfaceUvs: Float32Array,
  triangleIndices: Uint32Array,
  minU: number,
  maxU: number,
  minV: number,
  maxV: number,
): UvLookupGrid {
  const cols = 24;
  const rows = 24;
  const spanU = Math.max(1e-5, maxU - minU);
  const spanV = Math.max(1e-5, maxV - minV);
  const cellW = spanU / cols;
  const cellH = spanV / rows;
  const buckets: number[][] = Array.from({ length: cols * rows }, () => []);
  const triCount = triangleIndices.length / 3;

  for (let t = 0; t < triCount; t++) {
    const i0 = triangleIndices[t * 3] ?? 0;
    const i1 = triangleIndices[t * 3 + 1] ?? 0;
    const i2 = triangleIndices[t * 3 + 2] ?? 0;
    const u0 = vertexSurfaceUvs[i0 * 2] ?? 0;
    const v0 = vertexSurfaceUvs[i0 * 2 + 1] ?? 0;
    const u1 = vertexSurfaceUvs[i1 * 2] ?? 0;
    const v1 = vertexSurfaceUvs[i1 * 2 + 1] ?? 0;
    const u2 = vertexSurfaceUvs[i2 * 2] ?? 0;
    const v2 = vertexSurfaceUvs[i2 * 2 + 1] ?? 0;

    const tMinU = Math.min(u0, u1, u2);
    const tMaxU = Math.max(u0, u1, u2);
    const tMinV = Math.min(v0, v1, v2);
    const tMaxV = Math.max(v0, v1, v2);

    const c0 = Math.max(0, Math.min(cols - 1, Math.floor((tMinU - minU) / cellW)));
    const c1 = Math.max(0, Math.min(cols - 1, Math.floor((tMaxU - minU) / cellW)));
    const r0 = Math.max(0, Math.min(rows - 1, Math.floor((tMinV - minV) / cellH)));
    const r1 = Math.max(0, Math.min(rows - 1, Math.floor((tMaxV - minV) / cellH)));

    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        buckets[r * cols + c]?.push(t);
      }
    }
  }

  return { minU, minV, cellW, cellH, cols, rows, buckets };
}

function pointBarycentric2D(
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
  if (Math.abs(denom) <= 1e-12) {
    return null;
  }
  const inv = 1 / denom;
  const w1 = (v2x * v1y - v1x * v2y) * inv;
  const w2 = (v0x * v2y - v2x * v0y) * inv;
  const w0 = 1 - w1 - w2;
  if (w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4) {
    return null;
  }
  const c0 = Math.max(0, w0);
  const c1 = Math.max(0, w1);
  const c2 = Math.max(0, w2);
  const sum = c0 + c1 + c2;
  return sum > 0 ? [c0 / sum, c1 / sum, c2 / sum] : null;
}

function computeTriangleTangents(
  mapping: SurfaceMapping,
  i0: number,
  i1: number,
  i2: number,
): { tangentU: Vec3Tuple; tangentV: Vec3Tuple } {
  const pts = mapping.vertexPoints3D;
  const uvs = mapping.vertexSurfaceUvs;
  const p0 = readBufVec3(pts, i0);
  const p1 = readBufVec3(pts, i1);
  const p2 = readBufVec3(pts, i2);
  const u0 = uvs[i0 * 2] ?? 0;
  const v0 = uvs[i0 * 2 + 1] ?? 0;
  const du1 = (uvs[i1 * 2] ?? 0) - u0;
  const dv1 = (uvs[i1 * 2 + 1] ?? 0) - v0;
  const du2 = (uvs[i2 * 2] ?? 0) - u0;
  const dv2 = (uvs[i2 * 2 + 1] ?? 0) - v0;
  const det = du1 * dv2 - dv1 * du2;
  if (Math.abs(det) <= 1e-12) {
    return { tangentU: mapping.anchorFrame.uAxis, tangentV: mapping.anchorFrame.vAxis };
  }
  const e1 = vecSub(p1, p0);
  const e2 = vecSub(p2, p0);
  const rawU: Vec3Tuple = [
    (e1[0] * dv2 - e2[0] * dv1) / det,
    (e1[1] * dv2 - e2[1] * dv1) / det,
    (e1[2] * dv2 - e2[2] * dv1) / det,
  ];
  const rawV: Vec3Tuple = [
    (-e1[0] * du2 + e2[0] * du1) / det,
    (-e1[1] * du2 + e2[1] * du1) / det,
    (-e1[2] * du2 + e2[2] * du1) / det,
  ];
  return {
    tangentU: vecNormalize(rawU) ?? mapping.anchorFrame.uAxis,
    tangentV: vecNormalize(rawV) ?? mapping.anchorFrame.vAxis,
  };
}

/**
 * Samples the curved `SurfaceMapping` at an image coordinate by locating the containing
 * mapped triangle in O(1) time via the quad lookup table.
 */
export function sampleSurfaceMappingAtImagePoint(
  mapping: SurfaceMapping,
  imagePoint: Point2D,
  imageSize: Size2D,
): SurfaceMappingSample | null {
  if (imageSize.width <= 0 || imageSize.height <= 0) {
    return null;
  }
  const infW = mapping.inferenceWidth;
  const infH = mapping.inferenceHeight;
  const vx = (imagePoint.x * infW) / imageSize.width - 0.5;
  const vy = (imagePoint.y * infH) / imageSize.height - 0.5;
  const cvx = (vx - mapping.gridOffsetX) / Math.max(1, mapping.gridStride);
  const cvy = (vy - mapping.gridOffsetY) / Math.max(1, mapping.gridStride);
  const cW = mapping.gridCols;
  const cH = mapping.gridRows;
  const col = Math.min(cW - 2, Math.max(0, Math.floor(cvx)));
  const row = Math.min(cH - 2, Math.max(0, Math.floor(cvy)));
  const quadKey = row * (cW - 1) + col;
  const triList = mapping.quadTriangleMap.get(quadKey);
  if (!triList || triList.length === 0) {
    return null;
  }

  const pts = mapping.vertexPoints3D;
  const uvs = mapping.vertexSurfaceUvs;
  const gridIndices = mapping.vertexGridIndices;
  const depths = mapping.vertexDepths;

  for (let k = 0; k < triList.length; k++) {
    const tIdx = triList[k] ?? 0;
    const i0 = mapping.triangleIndices[tIdx * 3] ?? 0;
    const i1 = mapping.triangleIndices[tIdx * 3 + 1] ?? 0;
    const i2 = mapping.triangleIndices[tIdx * 3 + 2] ?? 0;
    const g0 = gridIndices[i0] ?? 0;
    const g1 = gridIndices[i1] ?? 0;
    const g2 = gridIndices[i2] ?? 0;
    const r0 = Math.floor(g0 / infW);
    const c0 = g0 - r0 * infW;
    const r1 = Math.floor(g1 / infW);
    const c1 = g1 - r1 * infW;
    const r2 = Math.floor(g2 / infW);
    const c2 = g2 - r2 * infW;

    const bary = pointBarycentric2D(vx, vy, c0, r0, c1, r1, c2, r2);
    if (!bary) {
      continue;
    }

    const [w0, w1, w2] = bary;
    const u = w0 * (uvs[i0 * 2] ?? 0) + w1 * (uvs[i1 * 2] ?? 0) + w2 * (uvs[i2 * 2] ?? 0);
    const v =
      w0 * (uvs[i0 * 2 + 1] ?? 0) + w1 * (uvs[i1 * 2 + 1] ?? 0) + w2 * (uvs[i2 * 2 + 1] ?? 0);
    const depth = w0 * (depths[i0] ?? 0) + w1 * (depths[i1] ?? 0) + w2 * (depths[i2] ?? 0);
    const point3D: Vec3Tuple = [
      w0 * (pts[i0 * 3] ?? 0) + w1 * (pts[i1 * 3] ?? 0) + w2 * (pts[i2 * 3] ?? 0),
      w0 * (pts[i0 * 3 + 1] ?? 0) + w1 * (pts[i1 * 3 + 1] ?? 0) + w2 * (pts[i2 * 3 + 1] ?? 0),
      w0 * (pts[i0 * 3 + 2] ?? 0) + w1 * (pts[i1 * 3 + 2] ?? 0) + w2 * (pts[i2 * 3 + 2] ?? 0),
    ];
    const { tangentU, tangentV } = computeTriangleTangents(mapping, i0, i1, i2);
    return {
      valid: true,
      u,
      v,
      depth,
      point3D,
      tangentU,
      tangentV,
      imagePoint,
    };
  }

  return null;
}

/**
 * Maps an intrinsic local surface coordinate `(u, v)` back to image pixels.
 * Uses the UV triangle lookup grid when inside the mapped patch, and extrapolates smoothly
 * via the local boundary triangle tangent frame or intrinsics fallback when slightly outside.
 */
export function mapSurfaceUvToImagePoint(
  mapping: SurfaceMapping,
  u: number,
  v: number,
  imageSize: Size2D,
  intrinsics?: ParsedIntrinsics | null,
): Point2D | null {
  if (imageSize.width <= 0 || imageSize.height <= 0) {
    return null;
  }

  const { uvGrid, vertexSurfaceUvs, vertexPhotoUvs, vertexPoints3D, triangleIndices } = mapping;
  const col = Math.max(0, Math.min(uvGrid.cols - 1, Math.floor((u - uvGrid.minU) / uvGrid.cellW)));
  const row = Math.max(0, Math.min(uvGrid.rows - 1, Math.floor((v - uvGrid.minV) / uvGrid.cellH)));
  const candidates = uvGrid.buckets[row * uvGrid.cols + col];

  if (candidates && candidates.length > 0) {
    for (let k = 0; k < candidates.length; k++) {
      const tIdx = candidates[k] ?? 0;
      const i0 = triangleIndices[tIdx * 3] ?? 0;
      const i1 = triangleIndices[tIdx * 3 + 1] ?? 0;
      const i2 = triangleIndices[tIdx * 3 + 2] ?? 0;
      const bary = pointBarycentric2D(
        u,
        v,
        vertexSurfaceUvs[i0 * 2] ?? 0,
        vertexSurfaceUvs[i0 * 2 + 1] ?? 0,
        vertexSurfaceUvs[i1 * 2] ?? 0,
        vertexSurfaceUvs[i1 * 2 + 1] ?? 0,
        vertexSurfaceUvs[i2 * 2] ?? 0,
        vertexSurfaceUvs[i2 * 2 + 1] ?? 0,
      );
      if (!bary) {
        continue;
      }
      const [w0, w1, w2] = bary;
      const uImg =
        w0 * (vertexPhotoUvs[i0 * 2] ?? 0) +
        w1 * (vertexPhotoUvs[i1 * 2] ?? 0) +
        w2 * (vertexPhotoUvs[i2 * 2] ?? 0);
      const vImg =
        w0 * (vertexPhotoUvs[i0 * 2 + 1] ?? 0) +
        w1 * (vertexPhotoUvs[i1 * 2 + 1] ?? 0) +
        w2 * (vertexPhotoUvs[i2 * 2 + 1] ?? 0);
      return {
        x: uImg * imageSize.width,
        y: vImg * imageSize.height,
      };
    }
  }

  // Smooth extrapolation from nearest mapped vertex tangent when (u, v) extends just beyond the patch.
  if (intrinsics) {
    const vCount = vertexSurfaceUvs.length / 2;
    if (vCount > 0) {
      let bestVert = 0;
      let bestDistSq = Infinity;
      const step = vCount > 1200 ? Math.ceil(vCount / 600) : 1;
      for (let i = 0; i < vCount; i += step) {
        const du = u - (vertexSurfaceUvs[i * 2] ?? 0);
        const dv = v - (vertexSurfaceUvs[i * 2 + 1] ?? 0);
        const d2 = du * du + dv * dv;
        if (d2 < bestDistSq) {
          bestDistSq = d2;
          bestVert = i;
        }
      }
      const pNearest = readBufVec3(vertexPoints3D, bestVert);
      const du = u - (vertexSurfaceUvs[bestVert * 2] ?? 0);
      const dv = v - (vertexSurfaceUvs[bestVert * 2 + 1] ?? 0);
      const approxPoint = vecAdd(
        pNearest,
        vecAdd(vecScale(mapping.anchorFrame.uAxis, du), vecScale(mapping.anchorFrame.vAxis, dv)),
      );
      if (approxPoint[2] > 1e-4) {
        return projectOpenCvPointToImage(approxPoint, intrinsics, imageSize);
      }
    }
    const anchorApprox = mapping.anchorFrame.localToWorld(u, v);
    if (anchorApprox[2] > 1e-4) {
      return projectOpenCvPointToImage(anchorApprox, intrinsics, imageSize);
    }
  }

  return null;
}

interface StridedSurfaceView {
  geometry: GeometryInput;
  gridIndex: SurfaceGridIndex;
  hit: SurfaceHit;
  stride: number;
  offsetX: number;
  offsetY: number;
  cols: number;
  rows: number;
}

function isFullStepChainContinuous(
  fullFlags: Uint8Array,
  fullWidth: number,
  startCol: number,
  startRow: number,
  dc: number,
  dr: number,
  stride: number,
  dirBit: number,
): boolean {
  let c = startCol;
  let r = startRow;
  for (let s = 0; s < stride; s++) {
    const idx = r * fullWidth + c;
    if (((fullFlags[idx] ?? 0) & dirBit) === 0) {
      return false;
    }
    c += dc;
    r += dr;
  }
  return true;
}

function buildStridedSurfaceView(
  geometry: GeometryInput,
  fullGridIndex: SurfaceGridIndex,
  hit: SurfaceHit,
  stride: number,
): StridedSurfaceView {
  const fullW = fullGridIndex.width;
  const fullH = fullGridIndex.height;
  const seedCol = hit.pixel?.x ?? 0;
  const seedRow = hit.pixel?.y ?? 0;

  if (stride <= 1) {
    return {
      geometry,
      gridIndex: fullGridIndex,
      hit,
      stride: 1,
      offsetX: 0,
      offsetY: 0,
      cols: fullW,
      rows: fullH,
    };
  }

  const offsetX = seedCol % stride;
  const offsetY = seedRow % stride;
  const cW = Math.floor((fullW - 1 - offsetX) / stride) + 1;
  const cH = Math.floor((fullH - 1 - offsetY) / stride) + 1;
  const cTotal = cW * cH;

  const cDepth = new Float32Array(cTotal);
  const cPoints = new Float32Array(cTotal * 3);
  const cMask = new Uint8Array(cTotal);
  const cNormals = geometry.normals ? new Float32Array(cTotal * 3) : null;
  const cValidMask = new Uint8Array(cTotal);
  const cFlags = new Uint8Array(cTotal);

  const fullFlags = fullGridIndex.neighborFlags;
  const fullValid = fullGridIndex.validMask;

  for (let rr = 0; rr < cH; rr++) {
    const fRow = offsetY + rr * stride;
    for (let cc = 0; cc < cW; cc++) {
      const fCol = offsetX + cc * stride;
      const fIdx = fRow * fullW + fCol;
      const cIdx = rr * cW + cc;
      if ((fullValid[fIdx] ?? 0) === 0) {
        continue;
      }
      cMask[cIdx] = 1;
      cValidMask[cIdx] = 1;
      cDepth[cIdx] = geometry.depth[fIdx] ?? 0;
      cPoints[cIdx * 3] = geometry.points[fIdx * 3] ?? 0;
      cPoints[cIdx * 3 + 1] = geometry.points[fIdx * 3 + 1] ?? 0;
      cPoints[cIdx * 3 + 2] = geometry.points[fIdx * 3 + 2] ?? 0;
      if (cNormals && geometry.normals) {
        cNormals[cIdx * 3] = geometry.normals[fIdx * 3] ?? 0;
        cNormals[cIdx * 3 + 1] = geometry.normals[fIdx * 3 + 1] ?? 0;
        cNormals[cIdx * 3 + 2] = geometry.normals[fIdx * 3 + 2] ?? 0;
      }
    }
  }

  for (let rr = 0; rr < cH; rr++) {
    const fRow = offsetY + rr * stride;
    for (let cc = 0; cc < cW; cc++) {
      const cIdx = rr * cW + cc;
      if ((cValidMask[cIdx] ?? 0) === 0) continue;
      const fCol = offsetX + cc * stride;

      if (
        cc + 1 < cW &&
        (cValidMask[cIdx + 1] ?? 0) === 1 &&
        isFullStepChainContinuous(fullFlags, fullW, fCol, fRow, 1, 0, stride, EDGE_RIGHT)
      ) {
        cFlags[cIdx] = (cFlags[cIdx] ?? 0) | EDGE_RIGHT;
        cFlags[cIdx + 1] = (cFlags[cIdx + 1] ?? 0) | EDGE_LEFT;
      }
      if (
        rr + 1 < cH &&
        (cValidMask[cIdx + cW] ?? 0) === 1 &&
        isFullStepChainContinuous(fullFlags, fullW, fCol, fRow, 0, 1, stride, EDGE_DOWN)
      ) {
        cFlags[cIdx] = (cFlags[cIdx] ?? 0) | EDGE_DOWN;
        cFlags[cIdx + cW] = (cFlags[cIdx + cW] ?? 0) | EDGE_UP;
      }
      if (
        cc + 1 < cW &&
        rr + 1 < cH &&
        (cValidMask[cIdx + cW + 1] ?? 0) === 1 &&
        isFullStepChainContinuous(fullFlags, fullW, fCol, fRow, 1, 1, stride, EDGE_DOWN_RIGHT)
      ) {
        cFlags[cIdx] = (cFlags[cIdx] ?? 0) | EDGE_DOWN_RIGHT;
        cFlags[cIdx + cW + 1] = (cFlags[cIdx + cW + 1] ?? 0) | EDGE_UP_LEFT;
      }
      if (
        cc - 1 >= 0 &&
        rr + 1 < cH &&
        (cValidMask[cIdx + cW - 1] ?? 0) === 1 &&
        isFullStepChainContinuous(fullFlags, fullW, fCol, fRow, -1, 1, stride, EDGE_DOWN_LEFT)
      ) {
        cFlags[cIdx] = (cFlags[cIdx] ?? 0) | EDGE_DOWN_LEFT;
        cFlags[cIdx + cW - 1] = (cFlags[cIdx + cW - 1] ?? 0) | EDGE_UP_RIGHT;
      }
    }
  }

  const cGeom: GeometryInput = {
    metadata: {
      inference_width: cW,
      inference_height: cH,
      intrinsics: geometry.metadata.intrinsics,
    },
    depth: cDepth,
    points: cPoints,
    mask: cMask,
    normals: cNormals,
  };
  const baseContinuity = computeDefaultContinuityParams(cGeom);
  const cContinuity: MeshContinuityParams = {
    ...baseContinuity,
    maxRelDepthDiff: Math.max(
      fullGridIndex.continuity.maxRelDepthDiff * stride,
      baseContinuity.maxRelDepthDiff,
    ),
    minNormalDot: Math.min(fullGridIndex.continuity.minNormalDot, baseContinuity.minNormalDot),
  };

  const cHit: SurfaceHit = {
    ...hit,
    pixel: {
      x: Math.floor((seedCol - offsetX) / stride),
      y: Math.floor((seedRow - offsetY) / stride),
    },
    triangleVertices: null,
    barycentric: null,
  };

  return {
    geometry: cGeom,
    gridIndex: {
      width: cW,
      height: cH,
      continuity: cContinuity,
      validMask: cValidMask,
      neighborFlags: cFlags,
    },
    hit: cHit,
    stride,
    offsetX,
    offsetY,
    cols: cW,
    rows: cH,
  };
}

function chooseAdaptiveGridStride(
  geometry: GeometryInput,
  fullGridIndex: SurfaceGridIndex,
  hit: SurfaceHit,
  maxRadius: number,
  explicitStride?: number,
): number {
  if (explicitStride && explicitStride >= 1) {
    return Math.min(4, Math.floor(explicitStride));
  }
  const fullW = fullGridIndex.width;
  const fullH = fullGridIndex.height;
  if (fullW < 120 || fullH < 120) {
    return 1;
  }
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const hitDepth = hit.depth ?? hit.point3D?.[2] ?? 1.0;
  const worldPerPx = hitDepth / Math.max(1, intr.fx * fullW);
  const estRadiusPx = maxRadius / Math.max(1e-6, worldPerPx);

  let validCount = 0;
  const vm = fullGridIndex.validMask;
  for (let i = 0; i < vm.length; i += 4) {
    if ((vm[i] ?? 0) === 1) validCount += 4;
  }
  const estAreaPx = Math.min(validCount, Math.PI * estRadiusPx * estRadiusPx);
  if (estAreaPx <= 32_000) return 1;
  if (estAreaPx <= 128_000) return 2;
  if (estAreaPx <= 288_000) return 3;
  return 4;
}

/**
 * Computes a bounded intrinsic 2D surface parameterization `(u, v)` around an image hit point
 * over the locally connected 2.5D mesh patch without crossing depth or normal discontinuities.
 */
export class LocalSurfaceParameterizer {
  private readonly workspace = new CurvedBrushWorkspace();
  private globalToLocal = new Int32Array(0);

  private ensureLookupCapacity(totalPixels: number): Int32Array {
    if (this.globalToLocal.length < totalPixels) {
      this.globalToLocal = new Int32Array(totalPixels);
      this.globalToLocal.fill(-1);
    }
    return this.globalToLocal;
  }

  build(options: LocalSurfaceParameterizerOptions): LocalSurfaceParameterizationOutcome {
    const {
      geometry: rawGeometry,
      geometryId = 'active-geometry',
      imagePoint,
      imageSize,
      maxRadius,
      maxVisitedNodes = 55_000,
      gridStride: explicitStride,
      maxStretch = 2.6,
      maxAnisotropy = 3.2,
      relaxationPasses = 4,
      id,
    } = options;

    if (imageSize.width <= 0 || imageSize.height <= 0 || maxRadius <= 0) {
      return makeRejectedOutcome('Invalid geometry or image bounds for surface parameterization.');
    }

    const rawHit: SurfaceHit = pickSurfaceHit(rawGeometry, imagePoint, imageSize);
    if (!rawHit.valid || !rawHit.point3D || !rawHit.pixel) {
      return makeRejectedOutcome('Anchor does not lie on valid 3D surface geometry.');
    }

    const fullGridIndex = options.gridIndex ?? getSurfaceGridIndex(rawGeometry);
    const fullW = fullGridIndex.width;
    const fullH = fullGridIndex.height;
    const rawSeedGridIdx = rawHit.pixel.y * fullW + rawHit.pixel.x;

    let usedEstimatedNormals = false;
    let anchorNormal = rawHit.normal
      ? orientNormalTowardCamera(rawHit.normal, rawHit.point3D)
      : null;
    if (!anchorNormal) {
      anchorNormal = estimateVertexNormalFromPatch(rawSeedGridIdx, rawGeometry, fullGridIndex);
      usedEstimatedNormals = true;
    }
    if (!anchorNormal) {
      return makeRejectedOutcome('Surface normal is unavailable at placement anchor.');
    }

    const anchorFrame = PlaneFrame.fromPlane(
      { origin: rawHit.point3D, normal: anchorNormal },
      rawHit.point3D,
    );

    const chosenStride = chooseAdaptiveGridStride(
      rawGeometry,
      fullGridIndex,
      rawHit,
      maxRadius,
      explicitStride,
    );
    const view = buildStridedSurfaceView(rawGeometry, fullGridIndex, rawHit, chosenStride);
    const geometry = view.geometry;
    const gridIndex = view.gridIndex;
    const hit = view.hit;
    const { width, height, validMask, neighborFlags, continuity } = gridIndex;
    const seedGridIdx = (hit.pixel?.y ?? 0) * width + (hit.pixel?.x ?? 0);

    const geodesic = this.workspace.runBoundedGeodesic({
      gridIndex,
      geometry,
      hit,
      maxRadius,
      maxVisitedNodes,
    });

    const visitedCount = geodesic.visitedCount;
    if (visitedCount < 4) {
      return makeRejectedOutcome('Connected surface patch is too small for curved wrapping.', {
        visitedVertexCount: visitedCount,
        usedEstimatedNormals,
      });
    }

    // Map global inference grid index -> dense local index [0..visitedCount - 1] via reusable O(1) LUT.
    const visitedIndices = geodesic.visitedIndices;
    const globalToLocal = this.ensureLookupCapacity(width * height);
    let minRow = height;
    let maxRow = -1;
    let minCol = width;
    let maxCol = -1;

    try {
      for (let li = 0; li < visitedCount; li++) {
        const gIdx = visitedIndices[li] ?? 0;
        globalToLocal[gIdx] = li;
        const row = Math.floor(gIdx / width);
        const col = gIdx - row * width;
        if (row < minRow) minRow = row;
        if (row > maxRow) maxRow = row;
        if (col < minCol) minCol = col;
        if (col > maxCol) maxCol = col;
      }

      // Gather positions and smoothly oriented vertex normals across visited vertices.
      const pts = new Float32Array(visitedCount * 3);
      const norms = new Float32Array(visitedCount * 3);
      const tangU = new Float32Array(visitedCount * 3);
      const tangV = new Float32Array(visitedCount * 3);
      const surfU = new Float32Array(visitedCount);
      const surfV = new Float32Array(visitedCount);
      const processed = new Uint8Array(visitedCount);

      for (let li = 0; li < visitedCount; li++) {
        const gIdx = visitedIndices[li] ?? 0;
        const pBase = gIdx * 3;
        const px = geometry.points[pBase] ?? 0;
        const py = geometry.points[pBase + 1] ?? 0;
        const pz = geometry.points[pBase + 2] ?? 0;
        pts[li * 3] = px;
        pts[li * 3 + 1] = py;
        pts[li * 3 + 2] = pz;

        const col = gIdx % width;
        const row = Math.floor(gIdx / width);
        let nVec = readNormalOpenCV(geometry, col, row);
        if (!nVec) {
          nVec = estimateVertexNormalFromPatch(gIdx, geometry, gridIndex) ?? anchorFrame.normal;
          usedEstimatedNormals = true;
        } else {
          nVec = orientNormalTowardCamera(nVec, [px, py, pz]);
        }
        norms[li * 3] = nVec[0];
        norms[li * 3 + 1] = nVec[1];
        norms[li * 3 + 2] = nVec[2];
      }

      // Propagate tangent bases via discrete parallel transport along Dijkstra wavefront order
      // with upwind continuous neighbor blending to avoid tree branch-cut seams.
      let minSeedLocal = -1;
      let minSeedDist = Infinity;

      for (let li = 0; li < visitedCount; li++) {
        const gIdx = visitedIndices[li] ?? 0;
        const parentG = geodesic.parentArr[gIdx] ?? -1;
        const nCur = readBufVec3(norms, li);
        const pCur = readBufVec3(pts, li);

        const pLocalLookup = parentG >= 0 ? (globalToLocal[parentG] ?? -1) : -1;
        if (pLocalLookup < 0) {
          const rotU = rotateVectorBetweenUnitNormals(anchorFrame.uAxis, anchorFrame.normal, nCur);
          const rotV = rotateVectorBetweenUnitNormals(anchorFrame.vAxis, anchorFrame.normal, nCur);
          const basis = orthonormalizeTangentPair(rotU, rotV, nCur);
          tangU[li * 3] = basis.u[0];
          tangU[li * 3 + 1] = basis.u[1];
          tangU[li * 3 + 2] = basis.u[2];
          tangV[li * 3] = basis.v[0];
          tangV[li * 3 + 1] = basis.v[1];
          tangV[li * 3 + 2] = basis.v[2];

          const mid = computeMidpointBasisPair(
            anchorFrame.uAxis,
            anchorFrame.vAxis,
            basis.u,
            basis.v,
          );
          const dp = vecSub(pCur, anchorFrame.origin);
          surfU[li] = vecDot(dp, mid.mU);
          surfV[li] = vecDot(dp, mid.mV);

          const sd = vecLength(dp);
          if (sd < minSeedDist) {
            minSeedDist = sd;
            minSeedLocal = li;
          }
          processed[li] = 1;
          continue;
        }

        const pLocal = pLocalLookup;
        const nPar = readBufVec3(norms, pLocal);
        const uPar = readBufVec3(tangU, pLocal);
        const vPar = readBufVec3(tangV, pLocal);

        const rotU = rotateVectorBetweenUnitNormals(uPar, nPar, nCur);
        const rotV = rotateVectorBetweenUnitNormals(vPar, nPar, nCur);
        const basis = orthonormalizeTangentPair(rotU, rotV, nCur);
        tangU[li * 3] = basis.u[0];
        tangU[li * 3 + 1] = basis.u[1];
        tangU[li * 3 + 2] = basis.u[2];
        tangV[li * 3] = basis.v[0];
        tangV[li * 3 + 1] = basis.v[1];
        tangV[li * 3 + 2] = basis.v[2];

        const flags = neighborFlags[gIdx] ?? 0;
        const row = Math.floor(gIdx / width);
        const col = gIdx - row * width;

        let accU = 0;
        let accV = 0;
        let accW = 0;

        for (let d = 0; d < 8; d++) {
          const nMask = NEIGHBOR_MASKS[d] ?? 0;
          if ((flags & nMask) === 0) continue;
          const nRow = row + (NEIGHBOR_DR[d] ?? 0);
          const nCol = col + (NEIGHBOR_DC[d] ?? 0);
          if (nRow < 0 || nRow >= height || nCol < 0 || nCol >= width) continue;
          const nbG = nRow * width + nCol;
          const nbLocal = globalToLocal[nbG] ?? -1;
          if (nbLocal < 0 || processed[nbLocal] !== 1) continue;

          const pNb = readBufVec3(pts, nbLocal);
          const edge = vecSub(pCur, pNb);
          const edgeLen = vecLength(edge);
          if (edgeLen <= 1e-8) continue;

          const uNb = readBufVec3(tangU, nbLocal);
          const vNb = readBufVec3(tangV, nbLocal);
          const mid = computeMidpointBasisPair(uNb, vNb, basis.u, basis.v);
          const du = vecDot(edge, mid.mU);
          const dv = vecDot(edge, mid.mV);

          const baseW = 1 / (edgeLen + 1e-4);
          const w = nbLocal === pLocal ? baseW * 1.5 : baseW;
          accU += w * ((surfU[nbLocal] ?? 0) + du);
          accV += w * ((surfV[nbLocal] ?? 0) + dv);
          accW += w;
        }

        if (accW > 0) {
          surfU[li] = accU / accW;
          surfV[li] = accV / accW;
        } else {
          const pPar = readBufVec3(pts, pLocal);
          const mid = computeMidpointBasisPair(uPar, vPar, basis.u, basis.v);
          const edge = vecSub(pCur, pPar);
          surfU[li] = (surfU[pLocal] ?? 0) + vecDot(edge, mid.mU);
          surfV[li] = (surfV[pLocal] ?? 0) + vecDot(edge, mid.mV);
        }

        processed[li] = 1;
      }

      // Build precomputed neighbor target displacements for fast local tangent relaxation.
      if (relaxationPasses > 0 && visitedCount > 4) {
        const edgeStarts = new Int32Array(visitedCount + 1);
        const maxEdges = visitedCount * 8;
        const edgeTarget = new Int32Array(maxEdges);
        const edgeDeltaU = new Float32Array(maxEdges);
        const edgeDeltaV = new Float32Array(maxEdges);
        const edgeWeight = new Float32Array(maxEdges);
        let edgeCursor = 0;

        for (let li = 0; li < visitedCount; li++) {
          edgeStarts[li] = edgeCursor;
          const gIdx = visitedIndices[li] ?? 0;
          const flags = neighborFlags[gIdx] ?? 0;
          const row = Math.floor(gIdx / width);
          const col = gIdx - row * width;
          const pA = readBufVec3(pts, li);
          const uA = readBufVec3(tangU, li);
          const vA = readBufVec3(tangV, li);

          for (let d = 0; d < 8; d++) {
            const nMask = NEIGHBOR_MASKS[d] ?? 0;
            if ((flags & nMask) === 0) continue;
            const nRow = row + (NEIGHBOR_DR[d] ?? 0);
            const nCol = col + (NEIGHBOR_DC[d] ?? 0);
            if (nRow < 0 || nRow >= height || nCol < 0 || nCol >= width) continue;
            const nbG = nRow * width + nCol;
            const nbLocal = globalToLocal[nbG] ?? -1;
            if (nbLocal < 0) continue;

            const pB = readBufVec3(pts, nbLocal);
            const edge = vecSub(pB, pA);
            const edgeLen = vecLength(edge);
            if (edgeLen <= 1e-8) continue;

            const uB = readBufVec3(tangU, nbLocal);
            const vB = readBufVec3(tangV, nbLocal);
            const mid = computeMidpointBasisPair(uA, vA, uB, vB);

            edgeTarget[edgeCursor] = nbLocal;
            edgeDeltaU[edgeCursor] = vecDot(edge, mid.mU);
            edgeDeltaV[edgeCursor] = vecDot(edge, mid.mV);
            edgeWeight[edgeCursor] = 1 / (edgeLen + 1e-4);
            edgeCursor += 1;
          }
        }
        edgeStarts[visitedCount] = edgeCursor;

        // Preserve the anchor UV value on the closest seed vertex during relaxation.
        const anchorPinU = minSeedLocal >= 0 ? (surfU[minSeedLocal] ?? 0) : 0;
        const anchorPinV = minSeedLocal >= 0 ? (surfV[minSeedLocal] ?? 0) : 0;

        for (let pass = 0; pass < relaxationPasses; pass++) {
          for (let li = 0; li < visitedCount; li++) {
            if (li === minSeedLocal) {
              surfU[li] = anchorPinU;
              surfV[li] = anchorPinV;
              continue;
            }
            const start = edgeStarts[li] ?? 0;
            const end = edgeStarts[li + 1] ?? 0;
            if (end <= start) continue;
            let sumU = 0;
            let sumV = 0;
            let sumW = 0;
            for (let e = start; e < end; e++) {
              const nb = edgeTarget[e] ?? 0;
              const w = edgeWeight[e] ?? 0;
              sumU += w * ((surfU[nb] ?? 0) - (edgeDeltaU[e] ?? 0));
              sumV += w * ((surfV[nb] ?? 0) - (edgeDeltaV[e] ?? 0));
              sumW += w;
            }
            if (sumW > 0) {
              surfU[li] = sumU / sumW;
              surfV[li] = sumV / sumW;
            }
          }
        }
      }

      // Re-center exact subpixel hit point to (0, 0) using hit triangle barycentric coords if available.
      let centerShiftU = 0;
      let centerShiftV = 0;
      if (hit.triangleVertices && hit.barycentric) {
        const l0 = globalToLocal[hit.triangleVertices[0]] ?? -1;
        const l1 = globalToLocal[hit.triangleVertices[1]] ?? -1;
        const l2 = globalToLocal[hit.triangleVertices[2]] ?? -1;
        if (l0 >= 0 && l1 >= 0 && l2 >= 0) {
          const [w0, w1, w2] = hit.barycentric;
          centerShiftU = w0 * (surfU[l0] ?? 0) + w1 * (surfU[l1] ?? 0) + w2 * (surfU[l2] ?? 0);
          centerShiftV = w0 * (surfV[l0] ?? 0) + w1 * (surfV[l1] ?? 0) + w2 * (surfV[l2] ?? 0);
        }
      } else if (minSeedLocal >= 0) {
        const dpSeed = vecSub(readBufVec3(pts, minSeedLocal), anchorFrame.origin);
        centerShiftU = (surfU[minSeedLocal] ?? 0) - vecDot(dpSeed, anchorFrame.uAxis);
        centerShiftV = (surfV[minSeedLocal] ?? 0) - vecDot(dpSeed, anchorFrame.vAxis);
      }
      if (Number.isFinite(centerShiftU) && Number.isFinite(centerShiftV)) {
        for (let li = 0; li < visitedCount; li++) {
          surfU[li] = (surfU[li] ?? 0) - centerShiftU;
          surfV[li] = (surfV[li] ?? 0) - centerShiftV;
        }
      }

      // Assemble continuous quad triangles, check orientation/stretch quality, and retain
      // only the connected triangle component around the seed.
      const candA: number[] = [];
      const candB: number[] = [];
      const candC: number[] = [];
      const candQuadKey: number[] = [];
      const candStretchErr: number[] = [];
      const candStretchRatio: number[] = [];

      let candidateTriangleCount = 0;
      let flippedTriangleCount = 0;
      let excessiveStretchTriangleCount = 0;
      const minScale = 1 / maxStretch;

      const rEnd = Math.min(height - 2, maxRow);
      const cEnd = Math.min(width - 2, maxCol);
      const rStart = Math.max(0, minRow);
      const cStart = Math.max(0, minCol);

      for (let row = rStart; row <= rEnd; row++) {
        for (let col = cStart; col <= cEnd; col++) {
          const i00 = row * width + col;
          const i01 = i00 + 1;
          const i10 = (row + 1) * width + col;
          const i11 = i10 + 1;
          const validCornerCount =
            ((globalToLocal[i00] ?? -1) >= 0 ? 1 : 0) +
            ((globalToLocal[i01] ?? -1) >= 0 ? 1 : 0) +
            ((globalToLocal[i10] ?? -1) >= 0 ? 1 : 0) +
            ((globalToLocal[i11] ?? -1) >= 0 ? 1 : 0);
          if (validCornerCount < 3) {
            continue;
          }
          const tris = buildQuadTriangles(geometry, validMask, row, col);
          const quadKey = row * (width - 1) + col;
          for (let t = 0; t < tris.length; t++) {
            const tri = tris[t];
            if (!tri) continue;
            const l0 = globalToLocal[tri.i0] ?? -1;
            const l1 = globalToLocal[tri.i1] ?? -1;
            const l2 = globalToLocal[tri.i2] ?? -1;
            if (l0 < 0 || l1 < 0 || l2 < 0) {
              continue;
            }
            if (!isTriangleContinuous(tri.i0, tri.i1, tri.i2, geometry, continuity)) {
              continue;
            }
            if (view.stride > 1) {
              const e01 = hasGridEdge(neighborFlags, width, tri.i0, tri.i1);
              const e12 = hasGridEdge(neighborFlags, width, tri.i1, tri.i2);
              const e20 = hasGridEdge(neighborFlags, width, tri.i2, tri.i0);
              if (!e01 || !e12 || !e20) {
                continue;
              }
            }

            candidateTriangleCount += 1;
            const p0 = readBufVec3(pts, l0);
            const p1 = readBufVec3(pts, l1);
            const p2 = readBufVec3(pts, l2);
            const fCross = vecCross(readBufVec3(tangU, l0), readBufVec3(tangV, l0));

            const metrics = evaluateTriangleParameterization(
              p0,
              p1,
              p2,
              surfU[l0] ?? 0,
              surfV[l0] ?? 0,
              surfU[l1] ?? 0,
              surfV[l1] ?? 0,
              surfU[l2] ?? 0,
              surfV[l2] ?? 0,
              fCross,
            );

            if (!metrics.oriented) {
              flippedTriangleCount += 1;
              continue;
            }
            if (
              !Number.isFinite(metrics.sMax) ||
              !Number.isFinite(metrics.sMin) ||
              metrics.sMax > maxStretch ||
              metrics.sMin < minScale ||
              metrics.anisotropy > maxAnisotropy
            ) {
              excessiveStretchTriangleCount += 1;
              continue;
            }

            candA.push(l0);
            candB.push(l1);
            candC.push(l2);
            candQuadKey.push(quadKey);
            candStretchErr.push(metrics.stretchError);
            candStretchRatio.push(Math.max(metrics.sMax, 1 / Math.max(metrics.sMin, 1e-6)));
          }
        }
      }

      const totalAcceptedRaw = candA.length;
      if (totalAcceptedRaw < 2) {
        return makeRejectedOutcome(
          'Curved surface could not form valid oriented parameterization triangles.',
          {
            visitedVertexCount: visitedCount,
            candidateTriangleCount,
            flippedTriangleCount,
            excessiveStretchTriangleCount,
            usedEstimatedNormals,
          },
        );
      }

      // Build vertex -> accepted triangle adjacency to extract the seed connected triangle component.
      const vertToTris = new Map<number, number[]>();
      for (let t = 0; t < totalAcceptedRaw; t++) {
        const a = candA[t] ?? 0;
        const b = candB[t] ?? 0;
        const c = candC[t] ?? 0;
        let listA = vertToTris.get(a);
        if (!listA) {
          listA = [];
          vertToTris.set(a, listA);
        }
        listA.push(t);
        let listB = vertToTris.get(b);
        if (!listB) {
          listB = [];
          vertToTris.set(b, listB);
        }
        listB.push(t);
        let listC = vertToTris.get(c);
        if (!listC) {
          listC = [];
          vertToTris.set(c, listC);
        }
        listC.push(t);
      }

      // Choose seed triangle at or nearest to the clicked anchor pixel.
      let startTri = -1;
      const seedLocal = globalToLocal[seedGridIdx] ?? -1;
      const seedQuadX = Math.min(
        width - 2,
        Math.max(
          0,
          Math.floor(((imagePoint.x * fullW) / imageSize.width - 0.5 - view.offsetX) / view.stride),
        ),
      );
      const seedQuadY = Math.min(
        height - 2,
        Math.max(
          0,
          Math.floor(
            ((imagePoint.y * fullH) / imageSize.height - 0.5 - view.offsetY) / view.stride,
          ),
        ),
      );
      const seedQuadKey = seedQuadY * (width - 1) + seedQuadX;

      for (let t = 0; t < totalAcceptedRaw; t++) {
        if (candQuadKey[t] === seedQuadKey) {
          startTri = t;
          break;
        }
      }
      if (startTri < 0 && seedLocal >= 0) {
        const incident = vertToTris.get(seedLocal);
        if (incident && incident.length > 0) {
          startTri = incident[0] ?? -1;
        }
      }
      if (startTri < 0 && minSeedLocal >= 0) {
        const incident = vertToTris.get(minSeedLocal);
        if (incident && incident.length > 0) {
          startTri = incident[0] ?? -1;
        }
      }

      if (startTri < 0) {
        return makeRejectedOutcome('Anchor neighborhood could not be parameterized cleanly.', {
          visitedVertexCount: visitedCount,
          candidateTriangleCount,
          flippedTriangleCount,
          excessiveStretchTriangleCount,
          usedEstimatedNormals,
        });
      }

      const triKeep = new Uint8Array(totalAcceptedRaw);
      const triQueue: number[] = [startTri];
      triKeep[startTri] = 1;
      let qHead = 0;

      // Require shared continuous edge (2 shared vertices) when propagating connected triangles.
      while (qHead < triQueue.length) {
        const t = triQueue[qHead++] ?? 0;
        const ta = candA[t] ?? 0;
        const tb = candB[t] ?? 0;
        const tc = candC[t] ?? 0;
        const edges: [number, number][] = [
          [ta, tb],
          [tb, tc],
          [tc, ta],
        ];
        for (let e = 0; e < 3; e++) {
          const edgePair = edges[e];
          if (!edgePair) continue;
          const [u, v] = edgePair;
          const listU = vertToTris.get(u);
          if (!listU) continue;
          for (let k = 0; k < listU.length; k++) {
            const nbT = listU[k] ?? 0;
            if (triKeep[nbT] === 1) continue;
            const na = candA[nbT] ?? 0;
            const nb = candB[nbT] ?? 0;
            const nc = candC[nbT] ?? 0;
            if (na === v || nb === v || nc === v) {
              // Verify shared edge itself satisfies mesh continuity.
              const gU = visitedIndices[u] ?? 0;
              const gV = visitedIndices[v] ?? 0;
              if (
                isEdgeContinuous(
                  gU,
                  gV,
                  width,
                  readPointTupleAt(geometry.points, gU),
                  readPointTupleAt(geometry.points, gV),
                  continuity,
                )
              ) {
                triKeep[nbT] = 1;
                triQueue.push(nbT);
              }
            }
          }
        }
      }

      // Compact only kept vertices & triangles.
      const keptTris = triQueue;
      const acceptedTriangleCount = keptTris.length;
      if (acceptedTriangleCount < 4) {
        return makeRejectedOutcome('Connected parameterized patch has too few valid triangles.', {
          visitedVertexCount: visitedCount,
          candidateTriangleCount,
          acceptedTriangleCount,
          flippedTriangleCount,
          excessiveStretchTriangleCount,
          usedEstimatedNormals,
        });
      }

      let maxObservedStretch = 1;
      let sumStretchErr = 0;
      for (let k = 0; k < acceptedTriangleCount; k++) {
        const t = keptTris[k] ?? 0;
        const r = candStretchRatio[t] ?? 1;
        if (r > maxObservedStretch) maxObservedStretch = r;
        sumStretchErr += candStretchErr[t] ?? 0;
      }
      const meanStretchError = sumStretchErr / acceptedTriangleCount;
      const anchorStretchError = candStretchErr[startTri] ?? 0;

      const rejectedTriCount = flippedTriangleCount + excessiveStretchTriangleCount;
      if (
        anchorStretchError > 0.65 ||
        rejectedTriCount > candidateTriangleCount * 0.6 ||
        (rejectedTriCount > acceptedTriangleCount * 0.6 && acceptedTriangleCount < 24)
      ) {
        return makeRejectedOutcome(
          'Surface curvature variation exceeds safe unwrap distortion limit.',
          {
            visitedVertexCount: visitedCount,
            candidateTriangleCount,
            acceptedTriangleCount,
            flippedTriangleCount,
            excessiveStretchTriangleCount,
            maxStretchRatio: maxObservedStretch,
            meanStretchError,
            anchorStretchError,
            usedEstimatedNormals,
          },
        );
      }

      const oldToCompact = new Int32Array(visitedCount).fill(-1);
      const compactVerts: number[] = [];
      for (let k = 0; k < acceptedTriangleCount; k++) {
        const t = keptTris[k] ?? 0;
        const aIdx = candA[t] ?? 0;
        const bIdx = candB[t] ?? 0;
        const cIdx = candC[t] ?? 0;
        if (oldToCompact[aIdx] === -1) {
          oldToCompact[aIdx] = compactVerts.length;
          compactVerts.push(aIdx);
        }
        if (oldToCompact[bIdx] === -1) {
          oldToCompact[bIdx] = compactVerts.length;
          compactVerts.push(bIdx);
        }
        if (oldToCompact[cIdx] === -1) {
          oldToCompact[cIdx] = compactVerts.length;
          compactVerts.push(cIdx);
        }
      }

      const acceptedVertexCount = compactVerts.length;
      const vertexGridIndices = new Int32Array(acceptedVertexCount);
      const vertexPoints3D = new Float32Array(acceptedVertexCount * 3);
      const vertexPhotoUvs = new Float32Array(acceptedVertexCount * 2);
      const vertexSurfaceUvs = new Float32Array(acceptedVertexCount * 2);
      const vertexDepths = new Float32Array(acceptedVertexCount);

      let minU = Infinity;
      let maxU = -Infinity;
      let minV = Infinity;
      let maxV = -Infinity;

      for (let ci = 0; ci < acceptedVertexCount; ci++) {
        const oldIdx = compactVerts[ci] ?? 0;
        const cIdx = visitedIndices[oldIdx] ?? 0;
        const cRow = Math.floor(cIdx / width);
        const cCol = cIdx - cRow * width;
        const fRow = view.offsetY + cRow * view.stride;
        const fCol = view.offsetX + cCol * view.stride;
        const fullIdx = fRow * fullW + fCol;

        vertexGridIndices[ci] = fullIdx;
        vertexPoints3D[ci * 3] = pts[oldIdx * 3] ?? 0;
        vertexPoints3D[ci * 3 + 1] = pts[oldIdx * 3 + 1] ?? 0;
        vertexPoints3D[ci * 3 + 2] = pts[oldIdx * 3 + 2] ?? 0;
        vertexDepths[ci] = pts[oldIdx * 3 + 2] ?? 0;
        vertexPhotoUvs[ci * 2] = (fCol + 0.5) / fullW;
        vertexPhotoUvs[ci * 2 + 1] = (fRow + 0.5) / fullH;

        const u = surfU[oldIdx] ?? 0;
        const v = surfV[oldIdx] ?? 0;
        vertexSurfaceUvs[ci * 2] = u;
        vertexSurfaceUvs[ci * 2 + 1] = v;
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }

      const triangleIndices = new Uint32Array(acceptedTriangleCount * 3);
      const quadTriangleMap = new Map<number, number[]>();
      for (let k = 0; k < acceptedTriangleCount; k++) {
        const t = keptTris[k] ?? 0;
        triangleIndices[k * 3] = oldToCompact[candA[t] ?? 0] ?? 0;
        triangleIndices[k * 3 + 1] = oldToCompact[candB[t] ?? 0] ?? 0;
        triangleIndices[k * 3 + 2] = oldToCompact[candC[t] ?? 0] ?? 0;
        const qKey = candQuadKey[t] ?? 0;
        let bucket = quadTriangleMap.get(qKey);
        if (!bucket) {
          bucket = [];
          quadTriangleMap.set(qKey, bucket);
        }
        bucket.push(k);
      }

      const uvGrid = buildUvLookupGrid(vertexSurfaceUvs, triangleIndices, minU, maxU, minV, maxV);

      const diagnostics: SurfaceMappingDiagnostics = {
        visitedVertexCount: visitedCount,
        acceptedVertexCount,
        candidateTriangleCount,
        acceptedTriangleCount,
        flippedTriangleCount,
        excessiveStretchTriangleCount,
        maxStretchRatio: maxObservedStretch,
        meanStretchError,
        anchorStretchError,
        usedEstimatedNormals,
        rejectionReason: null,
      };

      const mapping: SurfaceMapping = {
        id: id ?? nextMappingId(),
        geometryId,
        anchorImagePoint: { ...imagePoint },
        anchorPoint3D: rawHit.point3D,
        anchorNormal,
        anchorFrame,
        patchRadius: maxRadius,
        inferenceWidth: fullW,
        inferenceHeight: fullH,
        gridStride: view.stride,
        gridOffsetX: view.offsetX,
        gridOffsetY: view.offsetY,
        gridCols: view.cols,
        gridRows: view.rows,
        vertexGridIndices,
        vertexPoints3D,
        vertexPhotoUvs,
        vertexSurfaceUvs,
        vertexDepths,
        triangleIndices,
        minU,
        maxU,
        minV,
        maxV,
        diagnostics,
        quadTriangleMap,
        uvGrid,
      };

      return { valid: true, mapping, diagnostics };
    } finally {
      for (let li = 0; li < visitedCount; li++) {
        globalToLocal[visitedIndices[li] ?? 0] = -1;
      }
    }
  }
}
