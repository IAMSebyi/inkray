import type { Point2D, Size2D } from '../types/editor';
import { clamp } from '../editor/viewportTransforms';
import type { DirtyTexelBounds } from '../painting/brushEngine';
import { type GeometryInput, type Vec3Tuple, vecDot, vecLength, vecSub } from './geometryAccess';
import {
  type LocalSurfaceMetric,
  computeMetricEllipseSemiAxes,
  computeInterpolatedSurfaceMetric,
  evaluateMetricDistanceSq,
} from './localSurfaceMetric';
import { type MeshContinuityParams, readUnitNormalAt } from './meshContinuity';
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
  findContinuousTriangleAtGridPoint,
} from './surfacePicking';

const GRAPH_MOVES: ReadonlyArray<readonly [number, number, number]> = [
  [EDGE_RIGHT, 1, 0],
  [EDGE_LEFT, -1, 0],
  [EDGE_DOWN, 0, 1],
  [EDGE_UP, 0, -1],
  [EDGE_DOWN_RIGHT, 1, 1],
  [EDGE_DOWN_LEFT, -1, 1],
  [EDGE_UP_RIGHT, 1, -1],
  [EDGE_UP_LEFT, -1, -1],
];

const CARDINAL_MOVES: ReadonlyArray<readonly [number, number, number]> = [
  [EDGE_RIGHT, 1, 0],
  [EDGE_LEFT, -1, 0],
  [EDGE_DOWN, 0, 1],
  [EDGE_UP, 0, -1],
];

export interface BoundedGeodesicResult {
  generation: number;
  visitedIndices: Int32Array;
  visitedCount: number;
  distArr: Float32Array;
  seenGen: Int32Array;
  parentArr: Int32Array;
}

export interface ImageTexelSpec {
  width: number;
  height: number;
}

function readPt(points: Float32Array, idx: number): Vec3Tuple {
  const b = idx * 3;
  return [points[b] ?? 0, points[b + 1] ?? 0, points[b + 2] ?? 0];
}

function computeHardnessFalloff(dist: number, radius: number, hardness: number): number {
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
 * Reusable zero-allocation scratch workspace for bounded shortest-path queries
 * over the continuous 2.5D mesh edge graph.
 */
export class CurvedBrushWorkspace {
  private capacity = 0;
  private generation = 0;
  private seenGen = new Int32Array(0);
  private finalizedGen = new Int32Array(0);
  private distArr = new Float32Array(0);
  private parentArr = new Int32Array(0);
  private visitedList = new Int32Array(0);
  private heapIdx = new Int32Array(0);
  private heapDist = new Float32Array(0);
  private heapLen = 0;

  private ensureCapacity(total: number, maxHeap: number): void {
    if (total > this.capacity) {
      this.seenGen = new Int32Array(total);
      this.finalizedGen = new Int32Array(total);
      this.distArr = new Float32Array(total);
      this.parentArr = new Int32Array(total);
      this.visitedList = new Int32Array(total);
      this.capacity = total;
      this.generation = 0;
    }
    if (maxHeap > this.heapIdx.length) {
      this.heapIdx = new Int32Array(maxHeap);
      this.heapDist = new Float32Array(maxHeap);
    }
  }

  private pushHeap(nodeIdx: number, dist: number): void {
    if (this.heapLen >= this.heapIdx.length) {
      const nextCap = Math.max(1024, this.heapIdx.length * 2);
      const nextIdx = new Int32Array(nextCap);
      const nextDist = new Float32Array(nextCap);
      nextIdx.set(this.heapIdx);
      nextDist.set(this.heapDist);
      this.heapIdx = nextIdx;
      this.heapDist = nextDist;
    }
    let pos = this.heapLen++;
    this.heapIdx[pos] = nodeIdx;
    this.heapDist[pos] = dist;
    while (pos > 0) {
      const parent = (pos - 1) >> 1;
      const pDist = this.heapDist[parent] ?? 0;
      if (dist >= pDist) {
        break;
      }
      this.heapIdx[pos] = this.heapIdx[parent] ?? 0;
      this.heapDist[pos] = pDist;
      this.heapIdx[parent] = nodeIdx;
      this.heapDist[parent] = dist;
      pos = parent;
    }
  }

  private popHeap(): { idx: number; dist: number } {
    const outIdx = this.heapIdx[0] ?? 0;
    const outDist = this.heapDist[0] ?? 0;
    const last = --this.heapLen;
    if (last > 0) {
      const tailIdx = this.heapIdx[last] ?? 0;
      const tailDist = this.heapDist[last] ?? 0;
      this.heapIdx[0] = tailIdx;
      this.heapDist[0] = tailDist;
      let pos = 0;
      let left = pos * 2 + 1;
      while (left < last) {
        const right = left + 1;
        let smallest = left;
        if (right < last && (this.heapDist[right] ?? 0) < (this.heapDist[left] ?? 0)) {
          smallest = right;
        }
        const sDist = this.heapDist[smallest] ?? 0;
        if (tailDist <= sDist) {
          break;
        }
        this.heapIdx[pos] = this.heapIdx[smallest] ?? 0;
        this.heapDist[pos] = sDist;
        this.heapIdx[smallest] = tailIdx;
        this.heapDist[smallest] = tailDist;
        pos = smallest;
        left = pos * 2 + 1;
      }
    }
    return { idx: outIdx, dist: outDist };
  }

  runBoundedGeodesic(params: {
    gridIndex: SurfaceGridIndex;
    geometry: GeometryInput;
    hit: SurfaceHit;
    maxRadius: number;
    componentMask?: Uint8Array | null;
    maxVisitedNodes?: number;
  }): BoundedGeodesicResult {
    const {
      gridIndex,
      geometry,
      hit,
      maxRadius,
      componentMask = null,
      maxVisitedNodes = 24_000,
    } = params;
    const { width, height, validMask, neighborFlags } = gridIndex;
    const total = width * height;
    this.ensureCapacity(total, Math.min(total, maxVisitedNodes * 2 + 64));

    if (this.generation >= 2_000_000_000) {
      this.seenGen.fill(0);
      this.finalizedGen.fill(0);
      this.generation = 0;
    }
    const gen = ++this.generation;
    this.heapLen = 0;
    let visitedCount = 0;

    if (!hit.valid || !hit.point3D || !hit.pixel || maxRadius <= 0) {
      return {
        generation: gen,
        visitedIndices: this.visitedList,
        visitedCount: 0,
        distArr: this.distArr,
        seenGen: this.seenGen,
        parentArr: this.parentArr,
      };
    }

    const seedCenterPt = hit.point3D;
    const seedVertex = hit.pixel.y * width + hit.pixel.x;

    const trySeedVertex = (vIdx: number, initDist: number) => {
      if ((validMask[vIdx] ?? 0) === 0 || initDist > maxRadius) {
        return;
      }
      if (componentMask && (componentMask[vIdx] ?? 0) === 0) {
        return;
      }
      if ((this.seenGen[vIdx] ?? 0) !== gen || initDist < (this.distArr[vIdx] ?? Infinity)) {
        this.seenGen[vIdx] = gen;
        this.distArr[vIdx] = initDist;
        this.parentArr[vIdx] = -1;
        this.pushHeap(vIdx, initDist);
      }
    };

    if (hit.triangleVertices) {
      const seededSet = new Set<number>();
      for (let i = 0; i < 3; i++) {
        const vIdx = hit.triangleVertices[i] ?? -1;
        if (vIdx >= 0 && vIdx < total) {
          seededSet.add(vIdx);
          const pv = readPt(geometry.points, vIdx);
          trySeedVertex(vIdx, vecLength(vecSub(pv, seedCenterPt)));
        }
      }
      // If the anchor triangle sits in a fully continuous quad, seed its 4th quad corner too
      // when connected to a triangle vertex by an existing continuous edge.
      if (seededSet.size === 3) {
        const rows = hit.triangleVertices.map((idx) => Math.floor(idx / width));
        const cols = hit.triangleVertices.map((idx) => idx - Math.floor(idx / width) * width);
        const minR = Math.min(...rows);
        const minC = Math.min(...cols);
        if (minR >= 0 && minR + 1 < height && minC >= 0 && minC + 1 < width) {
          const quadCorners = [
            minR * width + minC,
            minR * width + minC + 1,
            (minR + 1) * width + minC,
            (minR + 1) * width + minC + 1,
          ];
          const anchorV = hit.triangleVertices[0];
          for (let q = 0; q < quadCorners.length; q++) {
            const cIdx = quadCorners[q] ?? -1;
            if (cIdx < 0 || seededSet.has(cIdx)) {
              continue;
            }
            const cRow = Math.floor(cIdx / width);
            const cCol = cIdx - cRow * width;
            if (isCornerEdgeConnected(gridIndex, anchorV, cCol, cRow)) {
              const pq = readPt(geometry.points, cIdx);
              trySeedVertex(cIdx, vecLength(vecSub(pq, seedCenterPt)));
            }
          }
        }
      }
    }
    if (this.heapLen === 0) {
      trySeedVertex(seedVertex, 0);
    }

    while (this.heapLen > 0) {
      const { idx: u, dist: du } = this.popHeap();
      if ((this.finalizedGen[u] ?? 0) === gen) {
        continue;
      }
      if (du > maxRadius) {
        break;
      }
      this.finalizedGen[u] = gen;
      this.visitedList[visitedCount++] = u;
      if (visitedCount >= maxVisitedNodes) {
        break;
      }

      const flags = neighborFlags[u] ?? 0;
      if (flags === 0) {
        continue;
      }
      const r = Math.floor(u / width);
      const c = u - r * width;
      const pu = readPt(geometry.points, u);

      for (let k = 0; k < GRAPH_MOVES.length; k++) {
        const move = GRAPH_MOVES[k];
        if (!move) continue;
        const [bit, dc, dr] = move;
        if ((flags & bit) === 0) {
          continue;
        }
        const nc = c + dc;
        const nr = r + dr;
        if (nc < 0 || nc >= width || nr < 0 || nr >= height) {
          continue;
        }
        const v = nr * width + nc;
        if ((validMask[v] ?? 0) === 0 || (this.finalizedGen[v] ?? 0) === gen) {
          continue;
        }
        if (componentMask && (componentMask[v] ?? 0) === 0) {
          continue;
        }
        const pv = readPt(geometry.points, v);
        const edgeLen = vecLength(vecSub(pv, pu));
        const nd = du + edgeLen;
        if (nd > maxRadius) {
          continue;
        }
        if ((this.seenGen[v] ?? 0) !== gen || nd < (this.distArr[v] ?? Infinity)) {
          this.seenGen[v] = gen;
          this.distArr[v] = nd;
          this.parentArr[v] = u;
          this.pushHeap(v, nd);
        }
      }
    }

    return {
      generation: gen,
      visitedIndices: this.visitedList,
      visitedCount,
      distArr: this.distArr,
      seenGen: this.seenGen,
      parentArr: this.parentArr,
    };
  }
}

function shouldUseBoundedGeodesic(
  gridIndex: SurfaceGridIndex,
  geometry: GeometryInput,
  componentMask: Uint8Array,
  centerPixel: Point2D,
  boxRadiusInf: number,
  centerNormal: Vec3Tuple,
): boolean {
  if (boxRadiusInf > 3.5) {
    return true;
  }
  const { width, height, validMask, neighborFlags } = gridIndex;
  const r0 = Math.max(0, Math.floor(centerPixel.y - boxRadiusInf - 1));
  const r1 = Math.min(height - 1, Math.ceil(centerPixel.y + boxRadiusInf + 1));
  const c0 = Math.max(0, Math.floor(centerPixel.x - boxRadiusInf - 1));
  const c1 = Math.min(width - 1, Math.ceil(centerPixel.x + boxRadiusInf + 1));
  const fullMask = EDGE_RIGHT | EDGE_LEFT | EDGE_DOWN | EDGE_UP;

  for (let row = r0; row <= r1; row++) {
    const rowOffset = row * width;
    for (let col = c0; col <= c1; col++) {
      const idx = rowOffset + col;
      if ((validMask[idx] ?? 0) === 0 || (componentMask[idx] ?? 0) === 0) {
        return true;
      }
      if (row > r0 && row < r1 && col > c0 && col < c1) {
        if (((neighborFlags[idx] ?? 0) & fullMask) !== fullMask) {
          return true;
        }
      }
      if (geometry.normals) {
        const n = readUnitNormalAt(geometry.normals, idx);
        if (n && vecDot(n, centerNormal) < 0.96) {
          return true;
        }
      }
    }
  }
  return false;
}

function isCornerEdgeConnected(
  gridIndex: SurfaceGridIndex,
  anchorIdx: number,
  cornerCol: number,
  cornerRow: number,
): boolean {
  const { width, neighborFlags } = gridIndex;
  const anchorRow = Math.floor(anchorIdx / width);
  const anchorCol = anchorIdx - anchorRow * width;
  if (cornerCol === anchorCol && cornerRow === anchorRow) {
    return true;
  }
  const dc = cornerCol - anchorCol;
  const dr = cornerRow - anchorRow;
  const flags = neighborFlags[anchorIdx] ?? 0;
  for (let k = 0; k < GRAPH_MOVES.length; k++) {
    const move = GRAPH_MOVES[k];
    if (!move) continue;
    if (move[1] === dc && move[2] === dr) {
      return (flags & move[0]) !== 0;
    }
  }
  return false;
}

export function stampCurvedBrushDab(params: {
  strokeMask: Uint8Array;
  texSpec: ImageTexelSpec;
  imageSize: Size2D;
  centerImagePt: Point2D;
  surfaceRadius: number;
  hardness: number;
  geometry: GeometryInput;
  gridIndex: SurfaceGridIndex;
  componentMask: Uint8Array;
  workspace: CurvedBrushWorkspace;
}): { bounds: DirtyTexelBounds | null; usedGeodesic: boolean } {
  const {
    strokeMask,
    texSpec,
    imageSize,
    centerImagePt,
    surfaceRadius,
    hardness,
    geometry,
    gridIndex,
    componentMask,
    workspace,
  } = params;

  const { width: infW, height: infH, validMask } = gridIndex;
  if (
    imageSize.width <= 0 ||
    imageSize.height <= 0 ||
    texSpec.width <= 0 ||
    texSpec.height <= 0 ||
    surfaceRadius <= 0
  ) {
    return { bounds: null, usedGeodesic: false };
  }

  const centerCol = clamp(Math.floor((centerImagePt.x * infW) / imageSize.width), 0, infW - 1);
  const centerRow = clamp(Math.floor((centerImagePt.y * infH) / imageSize.height), 0, infH - 1);
  const centerIdx = centerRow * infW + centerCol;
  if ((validMask[centerIdx] ?? 0) === 0 || (componentMask[centerIdx] ?? 0) === 0) {
    return { bounds: null, usedGeodesic: false };
  }

  const metric: LocalSurfaceMetric = computeInterpolatedSurfaceMetric(
    gridIndex,
    geometry,
    centerImagePt,
    imageSize,
  );
  if (!metric.valid) {
    return { bounds: null, usedGeodesic: false };
  }

  const gxCenter = (centerImagePt.x * infW) / imageSize.width;
  const gyCenter = (centerImagePt.y * infH) / imageSize.height;
  const triMatch = findContinuousTriangleAtGridPoint(gridIndex, geometry, gxCenter, gyCenter);

  const centerHit: SurfaceHit = {
    imagePoint: centerImagePt,
    pixel: { x: centerCol, y: centerRow },
    point3D: metric.point,
    normal: metric.normal,
    depth: metric.point[2],
    triangleVertices: triMatch?.vertices ?? null,
    barycentric: triMatch?.barycentric ?? null,
    valid: true,
  };

  const axes = computeMetricEllipseSemiAxes(metric, surfaceRadius, 400);
  const maxImgRadiusPx = axes ? clamp(Math.max(axes.axis1Px, axes.axis2Px) * 1.35, 3, 420) : 64;
  const boxRadiusInf = maxImgRadiusPx * (infW / imageSize.width);
  const useGeodesic = shouldUseBoundedGeodesic(
    gridIndex,
    geometry,
    componentMask,
    { x: centerCol, y: centerRow },
    boxRadiusInf,
    metric.normal,
  );

  let geo: BoundedGeodesicResult | null = null;
  let reachInfBounds: DirtyTexelBounds | null = null;

  if (useGeodesic) {
    const searchRadius = surfaceRadius * 1.15;
    geo = workspace.runBoundedGeodesic({
      gridIndex,
      geometry,
      hit: centerHit,
      maxRadius: searchRadius,
      componentMask,
    });
    if (geo.visitedCount === 0) {
      return { bounds: null, usedGeodesic: true };
    }
    let minC = infW - 1;
    let maxC = 0;
    let minR = infH - 1;
    let maxR = 0;
    for (let i = 0; i < geo.visitedCount; i++) {
      const idx = geo.visitedIndices[i] ?? 0;
      const r = Math.floor(idx / infW);
      const c = idx - r * infW;
      if (c < minC) minC = c;
      if (c > maxC) maxC = c;
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
    }
    reachInfBounds = {
      minX: clamp(Math.floor((minC / infW) * texSpec.width) - 2, 0, texSpec.width - 1),
      maxX: clamp(Math.ceil(((maxC + 1) / infW) * texSpec.width) + 2, 0, texSpec.width - 1),
      minY: clamp(Math.floor((minR / infH) * texSpec.height) - 2, 0, texSpec.height - 1),
      maxY: clamp(Math.ceil(((maxR + 1) / infH) * texSpec.height) + 2, 0, texSpec.height - 1),
    };
  }

  const metricBounds: DirtyTexelBounds = {
    minX: clamp(
      Math.floor(((centerImagePt.x - maxImgRadiusPx) / imageSize.width) * texSpec.width) - 1,
      0,
      texSpec.width - 1,
    ),
    maxX: clamp(
      Math.ceil(((centerImagePt.x + maxImgRadiusPx) / imageSize.width) * texSpec.width) + 1,
      0,
      texSpec.width - 1,
    ),
    minY: clamp(
      Math.floor(((centerImagePt.y - maxImgRadiusPx) / imageSize.height) * texSpec.height) - 1,
      0,
      texSpec.height - 1,
    ),
    maxY: clamp(
      Math.ceil(((centerImagePt.y + maxImgRadiusPx) / imageSize.height) * texSpec.height) + 1,
      0,
      texSpec.height - 1,
    ),
  };

  const bounds = reachInfBounds ?? metricBounds;
  const radiusSq = surfaceRadius * surfaceRadius;
  let dirtyMinX = texSpec.width;
  let dirtyMaxX = -1;
  let dirtyMinY = texSpec.height;
  let dirtyMaxY = -1;

  for (let py = bounds.minY; py <= bounds.maxY; py++) {
    const imgY = ((py + 0.5) / texSpec.height) * imageSize.height;
    const gy = ((py + 0.5) / texSpec.height) * infH;
    const infRow = clamp(Math.floor(gy), 0, infH - 1);
    const vQuad = clamp(gy - 0.5, 0, infH - 1);
    const r0 = Math.floor(vQuad);
    const r1 = Math.min(infH - 1, r0 + 1);
    const wy = vQuad - r0;
    const rowOffset = py * texSpec.width;

    for (let px = bounds.minX; px <= bounds.maxX; px++) {
      const imgX = ((px + 0.5) / texSpec.width) * imageSize.width;
      const gx = ((px + 0.5) / texSpec.width) * infW;
      const infCol = clamp(Math.floor(gx), 0, infW - 1);
      const anchorIdx = infRow * infW + infCol;

      if ((validMask[anchorIdx] ?? 0) === 0 || (componentMask[anchorIdx] ?? 0) === 0) {
        continue;
      }

      let surfDist: number;
      if (geo) {
        if ((geo.seenGen[anchorIdx] ?? 0) !== geo.generation) {
          continue;
        }
        const uQuad = clamp(gx - 0.5, 0, infW - 1);
        const c0 = Math.floor(uQuad);
        const c1 = Math.min(infW - 1, c0 + 1);
        const wx = uQuad - c0;

        let sumDist = 0;
        let sumW = 0;
        const corners: ReadonlyArray<readonly [number, number, number]> = [
          [c0, r0, (1 - wx) * (1 - wy)],
          [c1, r0, wx * (1 - wy)],
          [c0, r1, (1 - wx) * wy],
          [c1, r1, wx * wy],
        ];
        for (let k = 0; k < 4; k++) {
          const corner = corners[k];
          if (!corner) continue;
          const [cc, rr, w] = corner;
          if (w <= 0) continue;
          const cIdx = rr * infW + cc;
          if (
            (componentMask[cIdx] ?? 0) === 0 ||
            (geo.seenGen[cIdx] ?? 0) !== geo.generation ||
            !isCornerEdgeConnected(gridIndex, anchorIdx, cc, rr)
          ) {
            continue;
          }
          sumDist += (geo.distArr[cIdx] ?? 0) * w;
          sumW += w;
        }
        if (sumW <= 0) {
          surfDist = geo.distArr[anchorIdx] ?? Infinity;
        } else {
          surfDist = sumDist / sumW;
        }
      } else {
        const dSq = evaluateMetricDistanceSq(metric, centerImagePt, { x: imgX, y: imgY });
        if (dSq >= radiusSq) {
          continue;
        }
        surfDist = Math.sqrt(dSq);
      }

      if (surfDist >= surfaceRadius) {
        continue;
      }

      const cov = computeHardnessFalloff(surfDist, surfaceRadius, hardness);
      const nextAlpha = Math.round(clamp(cov, 0, 1) * 255);
      if (nextAlpha <= 0) {
        continue;
      }

      const tIdx = rowOffset + px;
      const prevAlpha = strokeMask[tIdx] ?? 0;
      if (nextAlpha > prevAlpha) {
        strokeMask[tIdx] = nextAlpha;
        if (px < dirtyMinX) dirtyMinX = px;
        if (px > dirtyMaxX) dirtyMaxX = px;
        if (py < dirtyMinY) dirtyMinY = py;
        if (py > dirtyMaxY) dirtyMaxY = py;
      }
    }
  }

  return {
    bounds:
      dirtyMaxX >= dirtyMinX && dirtyMaxY >= dirtyMinY
        ? { minX: dirtyMinX, minY: dirtyMinY, maxX: dirtyMaxX, maxY: dirtyMaxY }
        : null,
    usedGeodesic: useGeodesic,
  };
}

export type { MeshContinuityParams };
export { CARDINAL_MOVES };
