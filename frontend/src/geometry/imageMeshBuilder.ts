import * as THREE from 'three';
import { type GeometryInput, isValidGridPixel } from './geometryAccess';
import { computeDefaultContinuityParams, type MeshContinuityParams } from './meshContinuity';

export interface ImageMeshBuildOptions {
  coordinateSystem?: 'threejs' | 'opencv';
  colors?: Uint8ClampedArray;
  stride?: number;
  maxSlopeFactor?: number;
  maxRelativeDepthDiff?: number;
  minNormalDot?: number;
}

function isFlatEdgeContinuous(
  points: Float32Array,
  ia: number,
  ib: number,
  edgeScale: number,
  params: MeshContinuityParams,
): boolean {
  const aBase = ia * 3;
  const bBase = ib * 3;
  const ax = points[aBase] ?? 0;
  const ay = points[aBase + 1] ?? 0;
  const az = points[aBase + 2] ?? 0;
  const bx = points[bBase] ?? 0;
  const by = points[bBase + 1] ?? 0;
  const bz = points[bBase + 2] ?? 0;

  const minZ = az < bz ? az : bz;
  if (minZ <= 0) {
    return false;
  }

  const relLimit =
    edgeScale <= 1.5
      ? params.maxRelDepthDiff * edgeScale
      : Math.max(params.maxRelDepthDiff, Math.min(0.45, params.maxRelDepthDiff * edgeScale));
  if (Math.abs(az - bz) > relLimit * minZ) {
    return false;
  }

  const maxEdgeLen = minZ * params.pixelAngle * params.maxSlopeFactor * edgeScale;
  const dx = ax - bx;
  const dy = ay - by;
  const dz = az - bz;
  return dx * dx + dy * dy + dz * dz <= maxEdgeLen * maxEdgeLen;
}

function isFlatTriangleContinuous(
  points: Float32Array,
  normals: Float32Array | null,
  ia: number,
  ib: number,
  ic: number,
  abScale: number,
  bcScale: number,
  caScale: number,
  params: MeshContinuityParams,
): boolean {
  if (
    !isFlatEdgeContinuous(points, ia, ib, abScale, params) ||
    !isFlatEdgeContinuous(points, ib, ic, bcScale, params) ||
    !isFlatEdgeContinuous(points, ic, ia, caScale, params)
  ) {
    return false;
  }

  const aBase = ia * 3;
  const bBase = ib * 3;
  const cBase = ic * 3;
  const ax = points[aBase] ?? 0;
  const ay = points[aBase + 1] ?? 0;
  const az = points[aBase + 2] ?? 0;
  const bx = points[bBase] ?? 0;
  const by = points[bBase + 1] ?? 0;
  const bz = points[bBase + 2] ?? 0;
  const cx = points[cBase] ?? 0;
  const cy = points[cBase + 1] ?? 0;
  const cz = points[cBase + 2] ?? 0;

  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;
  const bcx = cx - bx;
  const bcy = cy - by;
  const bcz = cz - bz;

  const abSq = abx * abx + aby * aby + abz * abz;
  const acSq = acx * acx + acy * acy + acz * acz;
  const bcSq = bcx * bcx + bcy * bcy + bcz * bcz;
  if (abSq <= 1e-16 || acSq <= 1e-16 || bcSq <= 1e-16) {
    return false;
  }

  const crx = aby * acz - abz * acy;
  const cry = abz * acx - abx * acz;
  const crz = abx * acy - aby * acx;
  if (crx * crx + cry * cry + crz * crz <= 1e-12 * abSq * acSq) {
    return false;
  }

  if (normals) {
    const nax = normals[aBase] ?? 0;
    const nay = normals[aBase + 1] ?? 0;
    const naz = normals[aBase + 2] ?? 0;
    const nbx = normals[bBase] ?? 0;
    const nby = normals[bBase + 1] ?? 0;
    const nbz = normals[bBase + 2] ?? 0;
    const ncx = normals[cBase] ?? 0;
    const ncy = normals[cBase + 1] ?? 0;
    const ncz = normals[cBase + 2] ?? 0;

    if (
      Number.isFinite(nax) &&
      Number.isFinite(nay) &&
      Number.isFinite(naz) &&
      Number.isFinite(nbx) &&
      Number.isFinite(nby) &&
      Number.isFinite(nbz) &&
      Number.isFinite(ncx) &&
      Number.isFinite(ncy) &&
      Number.isFinite(ncz)
    ) {
      const la = Math.hypot(nax, nay, naz);
      const lb = Math.hypot(nbx, nby, nbz);
      const lc = Math.hypot(ncx, ncy, ncz);
      if (la > 1e-6 && lb > 1e-6 && lc > 1e-6) {
        const dab = (nax * nbx + nay * nby + naz * nbz) / (la * lb);
        const dbc = (nbx * ncx + nby * ncy + nbz * ncz) / (lb * lc);
        const dca = (ncx * nax + ncy * nay + ncz * naz) / (lc * la);
        if (dab < params.minNormalDot || dbc < params.minNormalDot || dca < params.minNormalDot) {
          return false;
        }
      }
    }
  }

  return true;
}

export const ImageMeshBuilder = {
  build(geometry: GeometryInput, options?: ImageMeshBuildOptions): THREE.BufferGeometry {
    const width = geometry.metadata.inference_width;
    const height = geometry.metadata.inference_height;
    const totalPixels = width * height;
    const bufferGeometry = new THREE.BufferGeometry();

    if (width < 2 || height < 2 || totalPixels === 0) {
      bufferGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      return bufferGeometry;
    }

    const coordinateSystem = options?.coordinateSystem ?? 'threejs';
    const colors = options?.colors;
    const stride = Math.max(1, Math.round(options?.stride ?? 1));
    const params = computeDefaultContinuityParams(geometry, options);

    const rowStarts: number[] = [];
    for (let r = 0; r < height - 1; r += stride) {
      rowStarts.push(r);
    }
    const colStarts: number[] = [];
    for (let c = 0; c < width - 1; c += stride) {
      colStarts.push(c);
    }

    const vertexValid = new Uint8Array(totalPixels);
    for (let ri = 0; ri < rowStarts.length; ri++) {
      const r0 = rowStarts[ri] ?? 0;
      const r1 = Math.min(height - 1, r0 + stride);
      for (let ci = 0; ci < colStarts.length; ci++) {
        const c0 = colStarts[ci] ?? 0;
        const c1 = Math.min(width - 1, c0 + stride);
        const i00 = r0 * width + c0;
        const i01 = r0 * width + c1;
        const i10 = r1 * width + c0;
        const i11 = r1 * width + c1;
        if (vertexValid[i00] === 0 && isValidGridPixel(geometry, i00)) vertexValid[i00] = 1;
        if (vertexValid[i01] === 0 && isValidGridPixel(geometry, i01)) vertexValid[i01] = 1;
        if (vertexValid[i10] === 0 && isValidGridPixel(geometry, i10)) vertexValid[i10] = 1;
        if (vertexValid[i11] === 0 && isValidGridPixel(geometry, i11)) vertexValid[i11] = 1;
      }
    }

    const points = geometry.points;
    const normals = geometry.normals;
    const rawTriangles = new Uint32Array(rowStarts.length * colStarts.length * 6);
    let rawCount = 0;

    for (let ri = 0; ri < rowStarts.length; ri++) {
      const row = rowStarts[ri] ?? 0;
      const nextRow = Math.min(height - 1, row + stride);
      const stepR = nextRow - row;
      for (let ci = 0; ci < colStarts.length; ci++) {
        const col = colStarts[ci] ?? 0;
        const nextCol = Math.min(width - 1, col + stride);
        const stepC = nextCol - col;

        const i00 = row * width + col;
        const i01 = row * width + nextCol;
        const i10 = nextRow * width + col;
        const i11 = nextRow * width + nextCol;

        const v00 = vertexValid[i00] === 1;
        const v01 = vertexValid[i01] === 1;
        const v10 = vertexValid[i10] === 1;
        const v11 = vertexValid[i11] === 1;
        const validCount = (v00 ? 1 : 0) + (v01 ? 1 : 0) + (v10 ? 1 : 0) + (v11 ? 1 : 0);
        if (validCount < 3) {
          continue;
        }

        const diagScale = Math.hypot(stepC, stepR) * (1.45 / Math.SQRT2);
        if (validCount === 4) {
          const b00 = i00 * 3;
          const b01 = i01 * 3;
          const b10 = i10 * 3;
          const b11 = i11 * 3;
          const dxA = (points[b00] ?? 0) - (points[b11] ?? 0);
          const dyA = (points[b00 + 1] ?? 0) - (points[b11 + 1] ?? 0);
          const dzA = (points[b00 + 2] ?? 0) - (points[b11 + 2] ?? 0);
          const dxB = (points[b01] ?? 0) - (points[b10] ?? 0);
          const dyB = (points[b01 + 1] ?? 0) - (points[b10 + 1] ?? 0);
          const dzB = (points[b01 + 2] ?? 0) - (points[b10 + 2] ?? 0);

          if (dxA * dxA + dyA * dyA + dzA * dzA <= dxB * dxB + dyB * dyB + dzB * dzB) {
            if (
              isFlatTriangleContinuous(
                points,
                normals,
                i00,
                i10,
                i11,
                stepR,
                stepC,
                diagScale,
                params,
              )
            ) {
              rawTriangles[rawCount++] = i00;
              rawTriangles[rawCount++] = i10;
              rawTriangles[rawCount++] = i11;
            }
            if (
              isFlatTriangleContinuous(
                points,
                normals,
                i00,
                i11,
                i01,
                diagScale,
                stepR,
                stepC,
                params,
              )
            ) {
              rawTriangles[rawCount++] = i00;
              rawTriangles[rawCount++] = i11;
              rawTriangles[rawCount++] = i01;
            }
          } else {
            if (
              isFlatTriangleContinuous(
                points,
                normals,
                i00,
                i10,
                i01,
                stepR,
                diagScale,
                stepC,
                params,
              )
            ) {
              rawTriangles[rawCount++] = i00;
              rawTriangles[rawCount++] = i10;
              rawTriangles[rawCount++] = i01;
            }
            if (
              isFlatTriangleContinuous(
                points,
                normals,
                i01,
                i10,
                i11,
                diagScale,
                stepC,
                stepR,
                params,
              )
            ) {
              rawTriangles[rawCount++] = i01;
              rawTriangles[rawCount++] = i10;
              rawTriangles[rawCount++] = i11;
            }
          }
        } else if (!v00) {
          if (
            isFlatTriangleContinuous(
              points,
              normals,
              i01,
              i10,
              i11,
              diagScale,
              stepC,
              stepR,
              params,
            )
          ) {
            rawTriangles[rawCount++] = i01;
            rawTriangles[rawCount++] = i10;
            rawTriangles[rawCount++] = i11;
          }
        } else if (!v01) {
          if (
            isFlatTriangleContinuous(
              points,
              normals,
              i00,
              i10,
              i11,
              stepR,
              stepC,
              diagScale,
              params,
            )
          ) {
            rawTriangles[rawCount++] = i00;
            rawTriangles[rawCount++] = i10;
            rawTriangles[rawCount++] = i11;
          }
        } else if (!v10) {
          if (
            isFlatTriangleContinuous(
              points,
              normals,
              i00,
              i11,
              i01,
              diagScale,
              stepR,
              stepC,
              params,
            )
          ) {
            rawTriangles[rawCount++] = i00;
            rawTriangles[rawCount++] = i11;
            rawTriangles[rawCount++] = i01;
          }
        } else if (
          isFlatTriangleContinuous(points, normals, i00, i10, i01, stepR, diagScale, stepC, params)
        ) {
          rawTriangles[rawCount++] = i00;
          rawTriangles[rawCount++] = i10;
          rawTriangles[rawCount++] = i01;
        }
      }
    }

    if (rawCount === 0) {
      bufferGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      return bufferGeometry;
    }

    // Compact referenced grid vertices into contiguous vertex buffers.
    const vertexRemap = new Int32Array(totalPixels).fill(-1);
    const usedGridIndices: number[] = [];
    for (let i = 0; i < rawCount; i++) {
      const gridIdx = rawTriangles[i] ?? 0;
      let mapped = vertexRemap[gridIdx] ?? -1;
      if (mapped === -1) {
        mapped = usedGridIndices.length;
        vertexRemap[gridIdx] = mapped;
        usedGridIndices.push(gridIdx);
      }
      rawTriangles[i] = mapped;
    }

    const vertexCount = usedGridIndices.length;
    const positions = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const outNormals = geometry.normals ? new Float32Array(vertexCount * 3) : null;
    const outColors = colors ? new Float32Array(vertexCount * 3) : null;
    const flipSign = coordinateSystem === 'threejs' ? -1 : 1;

    for (let vi = 0; vi < vertexCount; vi++) {
      const gridIdx = usedGridIndices[vi] ?? 0;
      const row = Math.floor(gridIdx / width);
      const col = gridIdx - row * width;

      const pBase = gridIdx * 3;
      const outBase = vi * 3;
      positions[outBase] = geometry.points[pBase] ?? 0;
      positions[outBase + 1] = flipSign * (geometry.points[pBase + 1] ?? 0);
      positions[outBase + 2] = flipSign * (geometry.points[pBase + 2] ?? 0);

      const uvBase = vi * 2;
      uvs[uvBase] = (col + 0.5) / width;
      uvs[uvBase + 1] =
        coordinateSystem === 'threejs' ? 1 - (row + 0.5) / height : (row + 0.5) / height;

      if (outNormals && geometry.normals) {
        outNormals[outBase] = geometry.normals[pBase] ?? 0;
        outNormals[outBase + 1] = flipSign * (geometry.normals[pBase + 1] ?? 0);
        outNormals[outBase + 2] = flipSign * (geometry.normals[pBase + 2] ?? -1);
      }

      if (outColors && colors) {
        const cBase = gridIdx * 4;
        outColors[outBase] = (colors[cBase] ?? 0) / 255;
        outColors[outBase + 1] = (colors[cBase + 1] ?? 0) / 255;
        outColors[outBase + 2] = (colors[cBase + 2] ?? 0) / 255;
      }
    }

    bufferGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    bufferGeometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    if (outNormals) {
      bufferGeometry.setAttribute('normal', new THREE.BufferAttribute(outNormals, 3));
    }
    if (outColors) {
      bufferGeometry.setAttribute('color', new THREE.BufferAttribute(outColors, 3));
    }

    const indexArray =
      vertexCount > 65535
        ? rawTriangles.slice(0, rawCount)
        : new Uint16Array(rawTriangles.subarray(0, rawCount));
    bufferGeometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
    bufferGeometry.computeBoundingSphere();
    return bufferGeometry;
  },
};
