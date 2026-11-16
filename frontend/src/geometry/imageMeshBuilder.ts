import * as THREE from 'three';
import {
  type GeometryInput,
  computePixelAngularSpan,
  isValidGridPixel,
  parseNormalizedIntrinsics,
} from './geometryAccess';

export interface ImageMeshBuildOptions {
  coordinateSystem?: 'threejs' | 'opencv';
  colors?: Uint8ClampedArray;
  maxSlopeFactor?: number;
  maxRelativeDepthDiff?: number;
  minNormalDot?: number;
}

interface CandidateTriangle {
  i0: number;
  i1: number;
  i2: number;
}

function readGridCoords(geom: GeometryInput, idx: number): [number, number, number] {
  const base = idx * 3;
  return [geom.points[base] ?? 0, geom.points[base + 1] ?? 0, geom.points[base + 2] ?? 0];
}

function readGridNormal(normals: Float32Array, idx: number): [number, number, number] | null {
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

function distSq3(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

function isEdgeContinuous(
  idxA: number,
  idxB: number,
  gridWidth: number,
  pA: readonly [number, number, number],
  pB: readonly [number, number, number],
  pixelAngle: number,
  maxSlopeFactor: number,
  maxRelDepthDiff: number,
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
  if (relDepthDiff > maxRelDepthDiff * edgeWeight) {
    return false;
  }

  const maxEdgeLen = minZ * pixelAngle * maxSlopeFactor * edgeWeight;
  return distSq3(pA, pB) <= maxEdgeLen * maxEdgeLen;
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
    const intrinsics = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
    const pixelAngle = computePixelAngularSpan(intrinsics, width, height);
    const maxSlopeFactor = options?.maxSlopeFactor ?? 8.0;
    const maxRelDepthDiff =
      options?.maxRelativeDepthDiff ?? Math.max(0.06, Math.min(0.35, 6 * pixelAngle));
    const minNormalDot = options?.minNormalDot ?? 0.2;

    const vertexValid = new Uint8Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      vertexValid[i] = isValidGridPixel(geometry, i) ? 1 : 0;
    }

    const rawTriangles: number[] = [];

    const tryAcceptTriangle = (ia: number, ib: number, ic: number) => {
      const pa = readGridCoords(geometry, ia);
      const pb = readGridCoords(geometry, ib);
      const pc = readGridCoords(geometry, ic);

      if (
        !isEdgeContinuous(ia, ib, width, pa, pb, pixelAngle, maxSlopeFactor, maxRelDepthDiff) ||
        !isEdgeContinuous(ib, ic, width, pb, pc, pixelAngle, maxSlopeFactor, maxRelDepthDiff) ||
        !isEdgeContinuous(ic, ia, width, pc, pa, pixelAngle, maxSlopeFactor, maxRelDepthDiff)
      ) {
        return;
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
        return;
      }

      const cx = aby * acz - abz * acy;
      const cy = abz * acx - abx * acz;
      const cz = abx * acy - aby * acx;
      const crossSq = cx * cx + cy * cy + cz * cz;
      // Reject collinear or near-zero-area triangles relative to edge scale.
      if (crossSq <= 1e-12 * abSq * acSq) {
        return;
      }

      if (geometry.normals) {
        const na = readGridNormal(geometry.normals, ia);
        const nb = readGridNormal(geometry.normals, ib);
        const nc = readGridNormal(geometry.normals, ic);
        if (na && nb && nc) {
          const dab = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
          const dbc = nb[0] * nc[0] + nb[1] * nc[1] + nb[2] * nc[2];
          const dca = nc[0] * na[0] + nc[1] * na[1] + nc[2] * na[2];
          if (dab < minNormalDot || dbc < minNormalDot || dca < minNormalDot) {
            return;
          }
        }
      }

      rawTriangles.push(ia, ib, ic);
    };

    for (let row = 0; row < height - 1; row++) {
      const rowOffset = row * width;
      const nextRowOffset = (row + 1) * width;
      for (let col = 0; col < width - 1; col++) {
        const i00 = rowOffset + col;
        const i01 = i00 + 1;
        const i10 = nextRowOffset + col;
        const i11 = i10 + 1;

        const v00 = vertexValid[i00] === 1;
        const v01 = vertexValid[i01] === 1;
        const v10 = vertexValid[i10] === 1;
        const v11 = vertexValid[i11] === 1;

        const validCount = (v00 ? 1 : 0) + (v01 ? 1 : 0) + (v10 ? 1 : 0) + (v11 ? 1 : 0);
        if (validCount < 3) {
          continue;
        }

        if (validCount === 4) {
          const p00 = readGridCoords(geometry, i00);
          const p01 = readGridCoords(geometry, i01);
          const p10 = readGridCoords(geometry, i10);
          const p11 = readGridCoords(geometry, i11);

          // Split along the shorter 3D diagonal to minimize stretch on slanted surfaces.
          const candidates: [CandidateTriangle, CandidateTriangle] =
            distSq3(p00, p11) <= distSq3(p01, p10)
              ? [
                  { i0: i00, i1: i10, i2: i11 },
                  { i0: i00, i1: i11, i2: i01 },
                ]
              : [
                  { i0: i00, i1: i10, i2: i01 },
                  { i0: i01, i1: i10, i2: i11 },
                ];
          tryAcceptTriangle(candidates[0].i0, candidates[0].i1, candidates[0].i2);
          tryAcceptTriangle(candidates[1].i0, candidates[1].i1, candidates[1].i2);
        } else if (!v00) {
          tryAcceptTriangle(i01, i10, i11);
        } else if (!v01) {
          tryAcceptTriangle(i00, i10, i11);
        } else if (!v10) {
          tryAcceptTriangle(i00, i11, i01);
        } else {
          tryAcceptTriangle(i00, i10, i01);
        }
      }
    }

    if (rawTriangles.length === 0) {
      bufferGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      return bufferGeometry;
    }

    // Compact referenced grid vertices into contiguous vertex buffers.
    const vertexRemap = new Int32Array(totalPixels).fill(-1);
    const usedGridIndices: number[] = [];
    for (let i = 0; i < rawTriangles.length; i++) {
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
      vertexCount > 65535 ? new Uint32Array(rawTriangles) : new Uint16Array(rawTriangles);
    bufferGeometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
    bufferGeometry.computeBoundingSphere();
    return bufferGeometry;
  },
};
