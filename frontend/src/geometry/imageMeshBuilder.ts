import * as THREE from 'three';
import { type GeometryInput, isValidGridPixel } from './geometryAccess';
import {
  buildQuadTriangles,
  computeDefaultContinuityParams,
  isTriangleContinuous,
} from './meshContinuity';

export interface ImageMeshBuildOptions {
  coordinateSystem?: 'threejs' | 'opencv';
  colors?: Uint8ClampedArray;
  maxSlopeFactor?: number;
  maxRelativeDepthDiff?: number;
  minNormalDot?: number;
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
    const params = computeDefaultContinuityParams(geometry, options);

    const vertexValid = new Uint8Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
      vertexValid[i] = isValidGridPixel(geometry, i) ? 1 : 0;
    }

    const rawTriangles: number[] = [];

    for (let row = 0; row < height - 1; row++) {
      for (let col = 0; col < width - 1; col++) {
        const candidates = buildQuadTriangles(geometry, vertexValid, row, col);
        for (let k = 0; k < candidates.length; k++) {
          const tri = candidates[k];
          if (tri && isTriangleContinuous(tri.i0, tri.i1, tri.i2, geometry, params)) {
            rawTriangles.push(tri.i0, tri.i1, tri.i2);
          }
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
