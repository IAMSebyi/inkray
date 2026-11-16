import type { GeometryData } from '../types/editor';
import {
  type GeometryInput,
  type Vec3Tuple,
  computePixelAngularSpan,
  normalizedUvToRayOpenCV,
  parseNormalizedIntrinsics,
  readPointOpenCV,
  vecDot,
  vecNormalize,
} from './geometryAccess';
import { PlaneFrame, type SurfaceSelectionResult } from './planeFrame';

export interface RectRegion {
  minCol: number;
  maxCol: number;
  minRow: number;
  maxRow: number;
}

export interface DepthOverrideRegion extends RectRegion {
  depth: number;
  normal?: Vec3Tuple;
}

export interface InvalidRegion extends RectRegion {
  depthValue?: number;
}

export interface SyntheticSceneOptions {
  width: number;
  height: number;
  intrinsics?: number[][];
  planeOrigin: Vec3Tuple;
  planeNormal: Vec3Tuple;
  occluders?: DepthOverrideRegion[];
  backgrounds?: DepthOverrideRegion[];
  invalidRegions?: InvalidRegion[];
  noiseAmplitude?: number;
}

export interface SyntheticScene {
  geometry: GeometryInput;
  colors: Uint8ClampedArray;
  truthPlaneMask: Uint8Array;
  planeOrigin: Vec3Tuple;
  planeNormal: Vec3Tuple;
}

export function makeSyntheticIntrinsics(fx = 1.0, fy = 1.0, cx = 0.5, cy = 0.5): number[][] {
  return [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
  ];
}

function insideRect(col: number, row: number, rect: RectRegion): boolean {
  return col >= rect.minCol && col <= rect.maxCol && row >= rect.minRow && row <= rect.maxRow;
}

/** Deterministic zero-mean scalar noise in [-1, 1] per pixel index. */
function pixelHashNoise(idx: number): number {
  let x = (idx + 1) * 0x27d4eb2d;
  x = Math.imul(x ^ (x >>> 15), 0x85ebca6b);
  x = x ^ (x >>> 13);
  return ((x >>> 0) / 4294967295) * 2 - 1;
}

export function createSyntheticPlanarScene(options: SyntheticSceneOptions): SyntheticScene {
  const { width, height } = options;
  const rawIntrinsics = options.intrinsics ?? makeSyntheticIntrinsics();
  const intr = parseNormalizedIntrinsics(rawIntrinsics);
  const unitNormal = vecNormalize(options.planeNormal) ?? [0, 0, -1];
  const origin = options.planeOrigin;

  const total = width * height;
  const depth = new Float32Array(total);
  const points = new Float32Array(total * 3);
  const mask = new Uint8Array(total);
  const normals = new Float32Array(total * 3);
  const colors = new Uint8ClampedArray(total * 4);
  const truthPlaneMask = new Uint8Array(total);

  const occluders = options.occluders ?? [];
  const backgrounds = options.backgrounds ?? [];
  const invalidRegions = options.invalidRegions ?? [];
  const noiseAmplitude = options.noiseAmplitude ?? 0;

  for (let row = 0; row < height; row++) {
    const vImg = (row + 0.5) / height;
    for (let col = 0; col < width; col++) {
      const idx = row * width + col;
      const pBase = idx * 3;
      const cBase = idx * 4;

      colors[cBase] = Math.round((col / Math.max(1, width - 1)) * 200) + 30;
      colors[cBase + 1] = Math.round((row / Math.max(1, height - 1)) * 180) + 40;
      colors[cBase + 2] = 140;
      colors[cBase + 3] = 255;

      const invalidMatch = invalidRegions.find((r) => insideRect(col, row, r));
      if (invalidMatch) {
        mask[idx] = 0;
        depth[idx] = invalidMatch.depthValue ?? Number.NaN;
        points[pBase] = 0;
        points[pBase + 1] = 0;
        points[pBase + 2] = 0;
        normals[pBase] = 0;
        normals[pBase + 1] = 0;
        normals[pBase + 2] = -1;
        truthPlaneMask[idx] = 0;
        continue;
      }

      const rayDir = normalizedUvToRayOpenCV({ x: (col + 0.5) / width, y: vImg }, intr, false);
      const denom = vecDot(unitNormal, rayDir);
      const tPlane = Math.abs(denom) > 1e-6 ? vecDot(unitNormal, origin) / denom : -1;

      let chosenDepth = tPlane;
      let chosenNormal: Vec3Tuple = unitNormal;
      let isOnTargetPlane = tPlane > 0;

      for (let i = 0; i < occluders.length; i++) {
        const occ = occluders[i];
        if (occ && insideRect(col, row, occ) && occ.depth > 0) {
          if (tPlane <= 0 || occ.depth < tPlane) {
            chosenDepth = occ.depth;
            chosenNormal = vecNormalize(occ.normal ?? [0, 0, -1]) ?? [0, 0, -1];
            isOnTargetPlane = false;
          }
        }
      }

      if (isOnTargetPlane) {
        for (let i = 0; i < backgrounds.length; i++) {
          const bg = backgrounds[i];
          if (bg && insideRect(col, row, bg) && bg.depth > 0) {
            chosenDepth = bg.depth;
            chosenNormal = vecNormalize(bg.normal ?? [0, 0, -1]) ?? [0, 0, -1];
            isOnTargetPlane = false;
          }
        }
      }

      if (chosenDepth <= 0 || !Number.isFinite(chosenDepth)) {
        mask[idx] = 0;
        depth[idx] = 0;
        truthPlaneMask[idx] = 0;
        continue;
      }

      const jitter = noiseAmplitude > 0 ? 1 + noiseAmplitude * pixelHashNoise(idx) : 1;
      const noisyDepth = Math.max(1e-3, chosenDepth * jitter);

      mask[idx] = 1;
      depth[idx] = noisyDepth;
      points[pBase] = rayDir[0] * noisyDepth;
      points[pBase + 1] = rayDir[1] * noisyDepth;
      points[pBase + 2] = noisyDepth;
      normals[pBase] = chosenNormal[0];
      normals[pBase + 1] = chosenNormal[1];
      normals[pBase + 2] = chosenNormal[2];
      truthPlaneMask[idx] = isOnTargetPlane ? 1 : 0;
    }
  }

  return {
    geometry: {
      metadata: {
        inference_width: width,
        inference_height: height,
        intrinsics: rawIntrinsics,
      },
      depth,
      points,
      mask,
      normals,
    },
    colors,
    truthPlaneMask,
    planeOrigin: origin,
    planeNormal: unitNormal,
  };
}

export function sceneToGeometryData(scene: SyntheticScene, imageId = 'stub-img'): GeometryData {
  let validCount = 0;
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < scene.geometry.mask.length; i++) {
    if (scene.geometry.mask[i] !== 0) {
      const z = scene.geometry.depth[i] ?? 0;
      if (Number.isFinite(z) && z > 0) {
        validCount += 1;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
      }
    }
  }

  return {
    imageId,
    metadata: {
      id: `geom-${imageId}`,
      cache_version: 1,
      image_sha256: 'synthetic-sha256',
      model_id: 'Ruicheng/moge-3-vitl',
      resolution_level: 9,
      refine_steps: 3,
      use_fp16: true,
      max_image_size: 1024,
      original_width: scene.geometry.metadata.inference_width,
      original_height: scene.geometry.metadata.inference_height,
      inference_width: scene.geometry.metadata.inference_width,
      inference_height: scene.geometry.metadata.inference_height,
      coordinate_system: 'opencv',
      intrinsics_convention: 'normalized',
      intrinsics: scene.geometry.metadata.intrinsics,
      depth_range: validCount > 0 ? [minZ, maxZ] : null,
      valid_pixel_count: validCount,
      has_normals: scene.geometry.normals !== null,
      cached: false,
      arrays: {},
    },
    depth: scene.geometry.depth,
    points: scene.geometry.points,
    mask: scene.geometry.mask,
    normals: scene.geometry.normals,
    colors: scene.colors,
    previews: {
      depth: {} as HTMLCanvasElement,
      normals: {} as HTMLCanvasElement,
      mask: {} as HTMLCanvasElement,
    },
  };
}

export function buildSyntheticSurfaceSelection(
  scene: SyntheticScene,
  seedCol: number,
  seedRow: number,
): SurfaceSelectionResult {
  const w = scene.geometry.metadata.inference_width;
  const h = scene.geometry.metadata.inference_height;
  const seedPoint = readPointOpenCV(scene.geometry, seedCol, seedRow) ?? scene.planeOrigin;
  const plane = { origin: scene.planeOrigin, normal: scene.planeNormal };
  const frame = PlaneFrame.fromPlane(plane, seedPoint);
  const intr = parseNormalizedIntrinsics(scene.geometry.metadata.intrinsics);
  const pixelAngle = computePixelAngularSpan(intr, w, h);

  let minU = 0;
  let maxU = 0;
  let minV = 0;
  let maxV = 0;
  let inlierCount = 0;

  for (let row = 0; row < h; row++) {
    for (let col = 0; col < w; col++) {
      const idx = row * w + col;
      if (scene.truthPlaneMask[idx] === 0) {
        continue;
      }
      const pt = readPointOpenCV(scene.geometry, col, row);
      if (!pt) continue;
      const [u, v] = frame.worldToLocal(pt);
      if (inlierCount === 0) {
        minU = Math.min(0, u);
        maxU = Math.max(0, u);
        minV = Math.min(0, v);
        maxV = Math.max(0, v);
      } else {
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
      inlierCount += 1;
    }
  }

  const spanU = Math.max(1e-3, maxU - minU);
  const spanV = Math.max(1e-3, maxV - minV);
  const padU = Math.max(4 * seedPoint[2] * pixelAngle, spanU * 0.05);
  const padV = Math.max(4 * seedPoint[2] * pixelAngle, spanV * 0.05);
  const footprintSpan = Math.min(spanU + padU * 2, spanV + padV * 2);

  return {
    seedImagePoint: { x: seedCol, y: seedRow },
    seedInferencePixel: { x: seedCol, y: seedRow },
    seedPoint3D: seedPoint,
    plane,
    frame,
    supportMask: scene.truthPlaneMask.slice(),
    inlierCount,
    rmsError: 0.002,
    suggestedFootprint: {
      minU: minU - padU,
      maxU: maxU + padU,
      minV: minV - padV,
      maxV: maxV + padV,
    },
    referenceBrushDiameter: Math.max(
      seedPoint[2] * pixelAngle * 2,
      Math.min(seedPoint[2] * 0.08, footprintSpan * 0.12),
    ),
  };
}
