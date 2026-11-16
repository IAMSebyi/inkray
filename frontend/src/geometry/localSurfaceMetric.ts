import type { Point2D, Size2D } from '../types/editor';
import { clamp } from '../editor/viewportTransforms';
import {
  type GeometryInput,
  type ParsedIntrinsics,
  type Vec3Tuple,
  vecCross,
  vecDot,
  vecLength,
  vecNormalize,
  vecSub,
} from './geometryAccess';
import { orientNormalTowardCamera } from './planeFrame';
import {
  EDGE_DOWN,
  EDGE_DOWN_RIGHT,
  EDGE_LEFT,
  EDGE_RIGHT,
  EDGE_UP,
  EDGE_UP_LEFT,
  type SurfaceGridIndex,
  findContinuousTriangleAtGridPoint,
} from './surfacePicking';

export interface LocalSurfaceMetric {
  valid: boolean;
  point: Vec3Tuple;
  normal: Vec3Tuple;
  pxInf: Vec3Tuple;
  pyInf: Vec3Tuple;
  gxx: number;
  gxy: number;
  gyy: number;
  det: number;
}

function invalidMetric(point: Vec3Tuple = [0, 0, 0]): LocalSurfaceMetric {
  return {
    valid: false,
    point,
    normal: [0, 0, -1],
    pxInf: [0, 0, 0],
    pyInf: [0, 0, 0],
    gxx: 0,
    gxy: 0,
    gyy: 0,
    det: 0,
  };
}

function readPt(points: Float32Array, idx: number): Vec3Tuple {
  const b = idx * 3;
  return [points[b] ?? 0, points[b + 1] ?? 0, points[b + 2] ?? 0];
}

function readNrm(normals: Float32Array | null, idx: number): Vec3Tuple | null {
  if (!normals) return null;
  const b = idx * 3;
  const nx = normals[b] ?? 0;
  const ny = normals[b + 1] ?? 0;
  const nz = normals[b + 2] ?? 0;
  if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz)) return null;
  return vecNormalize([nx, ny, nz]);
}

function computeOneSidedOrCentral(
  idx: number,
  col: number,
  row: number,
  width: number,
  height: number,
  flags: number,
  points: Float32Array,
  posBit: number,
  negBit: number,
  dc: number,
  dr: number,
): Vec3Tuple | null {
  const hasPos =
    (flags & posBit) !== 0 &&
    col + dc >= 0 &&
    col + dc < width &&
    row + dr >= 0 &&
    row + dr < height;
  const hasNeg =
    (flags & negBit) !== 0 &&
    col - dc >= 0 &&
    col - dc < width &&
    row - dr >= 0 &&
    row - dr < height;

  const pCenter = readPt(points, idx);
  if (hasPos && hasNeg) {
    const pPos = readPt(points, (row + dr) * width + (col + dc));
    const pNeg = readPt(points, (row - dr) * width + (col - dc));
    return [0.5 * (pPos[0] - pNeg[0]), 0.5 * (pPos[1] - pNeg[1]), 0.5 * (pPos[2] - pNeg[2])];
  }
  if (hasPos) {
    const pPos = readPt(points, (row + dr) * width + (col + dc));
    return vecSub(pPos, pCenter);
  }
  if (hasNeg) {
    const pNeg = readPt(points, (row - dr) * width + (col - dc));
    return vecSub(pCenter, pNeg);
  }
  return null;
}

function buildMetricFromGridDerivatives(
  point: Vec3Tuple,
  normalInput: Vec3Tuple | null,
  pxInf: Vec3Tuple,
  pyInf: Vec3Tuple,
  infWidth: number,
  infHeight: number,
  imageSize: Size2D,
): LocalSurfaceMetric {
  if (imageSize.width <= 0 || imageSize.height <= 0 || infWidth <= 0 || infHeight <= 0) {
    return invalidMetric(point);
  }
  const scaleX = infWidth / imageSize.width;
  const scaleY = infHeight / imageSize.height;
  const pxImg: Vec3Tuple = [pxInf[0] * scaleX, pxInf[1] * scaleX, pxInf[2] * scaleX];
  const pyImg: Vec3Tuple = [pyInf[0] * scaleY, pyInf[1] * scaleY, pyInf[2] * scaleY];

  const gxx = vecDot(pxImg, pxImg);
  const gxy = vecDot(pxImg, pyImg);
  const gyy = vecDot(pyImg, pyImg);
  const det = gxx * gyy - gxy * gxy;
  const trace = gxx + gyy;

  if (
    !Number.isFinite(det) ||
    !Number.isFinite(trace) ||
    trace <= 1e-16 ||
    det <= 1e-4 * trace * trace
  ) {
    return invalidMetric(point);
  }

  const crossNrm = vecNormalize(vecCross(pxInf, pyInf));
  const resolvedNormal = orientNormalTowardCamera(normalInput ?? crossNrm ?? [0, 0, -1], point);

  return {
    valid: true,
    point,
    normal: resolvedNormal,
    pxInf,
    pyInf,
    gxx,
    gxy,
    gyy,
    det,
  };
}

export function computeGridSurfaceMetric(
  gridIndex: SurfaceGridIndex,
  geometry: GeometryInput,
  col: number,
  row: number,
  imageSize: Size2D,
): LocalSurfaceMetric {
  const { width, height, validMask, neighborFlags } = gridIndex;
  if (col < 0 || col >= width || row < 0 || row >= height) {
    return invalidMetric();
  }
  const idx = row * width + col;
  if ((validMask[idx] ?? 0) === 0) {
    return invalidMetric();
  }

  const pCenter = readPt(geometry.points, idx);
  const flags = neighborFlags[idx] ?? 0;
  let pxInf = computeOneSidedOrCentral(
    idx,
    col,
    row,
    width,
    height,
    flags,
    geometry.points,
    EDGE_RIGHT,
    EDGE_LEFT,
    1,
    0,
  );
  let pyInf = computeOneSidedOrCentral(
    idx,
    col,
    row,
    width,
    height,
    flags,
    geometry.points,
    EDGE_DOWN,
    EDGE_UP,
    0,
    1,
  );

  // Recover the missing cardinal derivative along valid boundary diagonal edges if available.
  if (!pxInf && pyInf) {
    if ((flags & EDGE_DOWN_RIGHT) !== 0) {
      const diag = vecSub(readPt(geometry.points, (row + 1) * width + (col + 1)), pCenter);
      pxInf = vecSub(diag, pyInf);
    } else if ((flags & EDGE_UP_LEFT) !== 0) {
      const diag = vecSub(pCenter, readPt(geometry.points, (row - 1) * width + (col - 1)));
      pxInf = vecSub(diag, pyInf);
    }
  }
  if (!pyInf && pxInf) {
    if ((flags & EDGE_DOWN_RIGHT) !== 0) {
      const diag = vecSub(readPt(geometry.points, (row + 1) * width + (col + 1)), pCenter);
      pyInf = vecSub(diag, pxInf);
    } else if ((flags & EDGE_UP_LEFT) !== 0) {
      const diag = vecSub(pCenter, readPt(geometry.points, (row - 1) * width + (col - 1)));
      pyInf = vecSub(diag, pxInf);
    }
  }

  if (!pxInf || !pyInf) {
    return invalidMetric(pCenter);
  }

  const normal = readNrm(geometry.normals, idx);
  return buildMetricFromGridDerivatives(pCenter, normal, pxInf, pyInf, width, height, imageSize);
}

export function computeInterpolatedSurfaceMetric(
  gridIndex: SurfaceGridIndex,
  geometry: GeometryInput,
  imagePoint: Point2D,
  imageSize: Size2D,
): LocalSurfaceMetric {
  if (imageSize.width <= 0 || imageSize.height <= 0) {
    return invalidMetric();
  }
  const { width, height } = gridIndex;
  const gx = (imagePoint.x * width) / imageSize.width;
  const gy = (imagePoint.y * height) / imageSize.height;
  const tri = findContinuousTriangleAtGridPoint(gridIndex, geometry, gx, gy);

  if (!tri) {
    const col = clamp(Math.floor(gx), 0, width - 1);
    const row = clamp(Math.floor(gy), 0, height - 1);
    return computeGridSurfaceMetric(gridIndex, geometry, col, row, imageSize);
  }

  const [ia, ib, ic] = tri.vertices;
  const [w0, w1, w2] = tri.barycentric;
  const pa = readPt(geometry.points, ia);
  const pb = readPt(geometry.points, ib);
  const pc = readPt(geometry.points, ic);

  const point: Vec3Tuple = [
    w0 * pa[0] + w1 * pb[0] + w2 * pc[0],
    w0 * pa[1] + w1 * pb[1] + w2 * pc[1],
    w0 * pa[2] + w1 * pb[2] + w2 * pc[2],
  ];

  const r0 = Math.floor(ia / width);
  const c0 = ia - r0 * width;
  const r1 = Math.floor(ib / width);
  const c1 = ib - r1 * width;
  const r2 = Math.floor(ic / width);
  const c2 = ic - r2 * width;

  const dc1 = c1 - c0;
  const dr1 = r1 - r0;
  const dc2 = c2 - c0;
  const dr2 = r2 - r0;
  const det2 = dc1 * dr2 - dc2 * dr1;
  if (Math.abs(det2) <= 1e-8) {
    const col = clamp(Math.floor(gx), 0, width - 1);
    const row = clamp(Math.floor(gy), 0, height - 1);
    return computeGridSurfaceMetric(gridIndex, geometry, col, row, imageSize);
  }

  const dP1 = vecSub(pb, pa);
  const dP2 = vecSub(pc, pa);
  const invDet2 = 1 / det2;
  const pxInf: Vec3Tuple = [
    (dP1[0] * dr2 - dP2[0] * dr1) * invDet2,
    (dP1[1] * dr2 - dP2[1] * dr1) * invDet2,
    (dP1[2] * dr2 - dP2[2] * dr1) * invDet2,
  ];
  const pyInf: Vec3Tuple = [
    (dP2[0] * dc1 - dP1[0] * dc2) * invDet2,
    (dP2[1] * dc1 - dP1[1] * dc2) * invDet2,
    (dP2[2] * dc1 - dP1[2] * dc2) * invDet2,
  ];

  let interpNormal: Vec3Tuple | null = null;
  if (geometry.normals) {
    const na = readNrm(geometry.normals, ia);
    const nb = readNrm(geometry.normals, ib);
    const nc = readNrm(geometry.normals, ic);
    if (na && nb && nc) {
      interpNormal = vecNormalize([
        w0 * na[0] + w1 * nb[0] + w2 * nc[0],
        w0 * na[1] + w1 * nb[1] + w2 * nc[1],
        w0 * na[2] + w1 * nb[2] + w2 * nc[2],
      ]);
    }
  }

  return buildMetricFromGridDerivatives(
    point,
    interpNormal,
    pxInf,
    pyInf,
    width,
    height,
    imageSize,
  );
}

/**
 * Converts an image-pixel-equivalent brush diameter at the initial anchor depth
 * into a fixed 3D tangent-space surface radius preserved throughout the stroke.
 */
export function computeSurfaceBrushRadius(
  anchorDepth: number,
  pixelDiameter: number,
  intrinsics: ParsedIntrinsics,
  imageSize: Size2D,
): number {
  const avgFocalPx = 0.5 * (intrinsics.fx * imageSize.width + intrinsics.fy * imageSize.height);
  const safeDepth = Math.max(1e-4, anchorDepth);
  return Math.max(1e-5, (Math.max(1, pixelDiameter) * 0.5 * safeDepth) / Math.max(1, avgFocalPx));
}

export function evaluateMetricDistanceSq(
  metric: LocalSurfaceMetric,
  centerImagePt: Point2D,
  sampleImagePt: Point2D,
): number {
  const dx = sampleImagePt.x - centerImagePt.x;
  const dy = sampleImagePt.y - centerImagePt.y;
  return Math.max(0, metric.gxx * dx * dx + 2 * metric.gxy * dx * dy + metric.gyy * dy * dy);
}

export function computeMetricEllipseSemiAxes(
  metric: LocalSurfaceMetric,
  surfaceRadius: number,
  maxAxisPx = 600,
): {
  e1: Point2D;
  e2: Point2D;
  axis1Px: number;
  axis2Px: number;
} | null {
  if (!metric.valid || surfaceRadius <= 0) {
    return null;
  }
  const { gxx, gxy, gyy } = metric;
  const halfTrace = 0.5 * (gxx + gyy);
  const diffHalf = 0.5 * (gxx - gyy);
  const disc = Math.sqrt(Math.max(0, diffHalf * diffHalf + gxy * gxy));
  const lam1 = halfTrace + disc;
  const lam2 = halfTrace - disc;
  if (lam1 <= 1e-18 || lam2 <= 1e-18) {
    return null;
  }

  let e1x: number;
  let e1y: number;
  if (Math.abs(gxy) > 1e-18) {
    const vx = lam1 - gyy;
    const vy = gxy;
    const len = Math.hypot(vx, vy);
    e1x = len > 0 ? vx / len : 1;
    e1y = len > 0 ? vy / len : 0;
  } else if (gxx >= gyy) {
    e1x = 1;
    e1y = 0;
  } else {
    e1x = 0;
    e1y = 1;
  }
  const e2x = -e1y;
  const e2y = e1x;

  const axis1 = surfaceRadius / Math.sqrt(lam1);
  const axis2 = surfaceRadius / Math.sqrt(lam2);
  if (
    !Number.isFinite(axis1) ||
    !Number.isFinite(axis2) ||
    axis1 > maxAxisPx ||
    axis2 > maxAxisPx
  ) {
    return null;
  }

  return {
    e1: { x: e1x, y: e1y },
    e2: { x: e2x, y: e2y },
    axis1Px: axis1,
    axis2Px: axis2,
  };
}

export function buildMetricEllipseImagePolygon(
  metric: LocalSurfaceMetric,
  centerImagePt: Point2D,
  surfaceRadius: number,
  segments = 28,
): Point2D[] | null {
  const axes = computeMetricEllipseSemiAxes(metric, surfaceRadius);
  if (!axes) {
    return null;
  }
  const poly: Point2D[] = [];
  for (let i = 0; i < segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    const ct = Math.cos(theta) * axes.axis1Px;
    const st = Math.sin(theta) * axes.axis2Px;
    poly.push({
      x: centerImagePt.x + axes.e1.x * ct + axes.e2.x * st,
      y: centerImagePt.y + axes.e1.y * ct + axes.e2.y * st,
    });
  }
  return poly;
}

export function computeImageScaleFromPoints(a: Vec3Tuple, b: Vec3Tuple): number {
  return vecLength(vecSub(b, a));
}
