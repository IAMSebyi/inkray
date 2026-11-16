import { describe, expect, it } from 'vitest';
import { type GeometryInput, parseNormalizedIntrinsics } from './geometryAccess';
import {
  LocalSurfaceParameterizer,
  mapSurfaceUvToImagePoint,
  sampleSurfaceMappingAtImagePoint,
} from './localSurfaceParameterizer';
import {
  createSyntheticCylinderScene,
  createSyntheticPlanarScene,
  createSyntheticSphereScene,
  makeSyntheticIntrinsics,
} from './testFixtures';

function buildWavyIrregularScene(width = 80, height = 80): GeometryInput {
  const intrRaw = makeSyntheticIntrinsics(1.0, 1.0, 0.5, 0.5);
  const total = width * height;
  const depth = new Float32Array(total);
  const points = new Float32Array(total * 3);
  const mask = new Uint8Array(total);

  for (let row = 0; row < height; row++) {
    const vImg = (row + 0.5) / height;
    const ny = vImg - 0.5;
    for (let col = 0; col < width; col++) {
      const uImg = (col + 0.5) / width;
      const nx = uImg - 0.5;
      const idx = row * width + col;
      // Smooth bumpy surface: z(nx, ny) = 2.4 + 0.22 * sin(nx * 5) * cos(ny * 4)
      const z = 2.4 + 0.22 * Math.sin(nx * 5.0) * Math.cos(ny * 4.0);
      depth[idx] = z;
      points[idx * 3] = nx * z;
      points[idx * 3 + 1] = ny * z;
      points[idx * 3 + 2] = z;
      mask[idx] = 1;
    }
  }

  return {
    metadata: {
      inference_width: width,
      inference_height: height,
      intrinsics: intrRaw,
    },
    depth,
    points,
    mask,
    normals: null,
  };
}

function buildSharpCreaseFragmentedScene(width = 48, height = 48): GeometryInput {
  const intrRaw = makeSyntheticIntrinsics(1.0, 1.0, 0.5, 0.5);
  const total = width * height;
  const depth = new Float32Array(total);
  const points = new Float32Array(total * 3);
  const mask = new Uint8Array(total);
  const normals = new Float32Array(total * 3);

  for (let row = 0; row < height; row++) {
    const ny = (row + 0.5) / height - 0.5;
    for (let col = 0; col < width; col++) {
      const nx = (col + 0.5) / width - 0.5;
      const idx = row * width + col;
      // Alternating sharp sawtooth crease right across the anchor breaks valid triangle formation.
      const z = 1.8 + ((row + col) % 2 === 0 ? 0.0 : 0.28);
      points[idx * 3] = nx * z;
      points[idx * 3 + 1] = ny * z;
      points[idx * 3 + 2] = z;
      depth[idx] = z;
      mask[idx] = 1;
      normals[idx * 3] = 0;
      normals[idx * 3 + 1] = 0;
      normals[idx * 3 + 2] = -1;
    }
  }

  return {
    metadata: {
      inference_width: width,
      inference_height: height,
      intrinsics: intrRaw,
    },
    depth,
    points,
    mask,
    normals,
  };
}

describe('LocalSurfaceParameterizer', () => {
  it('matches analytic arc-length (R*theta, y) on a curved cylinder with zero flipped triangles', () => {
    const cylRadius = 0.8;
    const cylAxisZ = 2.2;
    const cyl = createSyntheticCylinderScene({
      width: 96,
      height: 96,
      axisCenter: [0, 0, cylAxisZ],
      radius: cylRadius,
    });
    const imageSize = { width: 960, height: 960 };
    const intr = parseNormalizedIntrinsics(cyl.geometry.metadata.intrinsics);
    const parameterizer = new LocalSurfaceParameterizer();

    const outcome = parameterizer.build({
      geometry: cyl.geometry,
      imagePoint: { x: 480, y: 480 },
      imageSize,
      maxRadius: 1.0,
    });

    expect(outcome.valid).toBe(true);
    expect(outcome.mapping).not.toBeNull();
    const mapping = outcome.mapping;
    if (!mapping) throw new Error('Expected valid cylinder SurfaceMapping');
    expect(mapping.diagnostics.flippedTriangleCount).toBe(0);
    expect(mapping.diagnostics.acceptedTriangleCount).toBeGreaterThan(300);
    expect(mapping.diagnostics.meanStretchError).toBeLessThan(0.05);

    // Compare parameterized (u, v) against exact analytic cylinder coordinates (R*theta, y).
    const vCount = mapping.vertexGridIndices.length;
    let sqErrSum = 0;
    let checked = 0;
    let maxThetaRad = 0;

    for (let i = 0; i < vCount; i++) {
      const px = mapping.vertexPoints3D[i * 3] ?? 0;
      const py = mapping.vertexPoints3D[i * 3 + 1] ?? 0;
      const pz = mapping.vertexPoints3D[i * 3 + 2] ?? 0;
      const theta = Math.atan2(px, -(pz - cylAxisZ));
      maxThetaRad = Math.max(maxThetaRad, Math.abs(theta));
      const analyticU = cylRadius * theta;
      const analyticV = py;

      const u = mapping.vertexSurfaceUvs[i * 2] ?? 0;
      const v = mapping.vertexSurfaceUvs[i * 2 + 1] ?? 0;
      const err = Math.hypot(u - analyticU, v - analyticV);
      sqErrSum += err * err;
      checked += 1;
    }

    const rmse = Math.sqrt(sqErrSum / Math.max(1, checked));
    expect(maxThetaRad).toBeGreaterThan(0.5);
    expect(rmse).toBeLessThan(0.03 * cylRadius);

    // Curved projection at fixed +u must curve inward in screen X compared to a flat front tangent plane.
    const testArcU = 0.55 * cylRadius;
    const flatPlaneScreenX =
      imageSize.width * 0.5 + (testArcU / (cylAxisZ - cylRadius)) * intr.fx * imageSize.width;
    const curvedScreenPt = mapSurfaceUvToImagePoint(mapping, testArcU, 0, imageSize, intr);
    expect(curvedScreenPt).not.toBeNull();
    if (curvedScreenPt) {
      expect(curvedScreenPt.x).toBeGreaterThan(imageSize.width * 0.52);
      expect(curvedScreenPt.x).toBeLessThan(flatPlaneScreenX - 15);
    }
  });

  it('produces continuous seam-free local UVs on a sphere cap and round-trips lookup queries', () => {
    const sphere = createSyntheticSphereScene({
      width: 90,
      height: 90,
      center: [0, 0, 2.1],
      radius: 0.9,
    });
    const imageSize = { width: 900, height: 900 };
    const intr = parseNormalizedIntrinsics(sphere.geometry.metadata.intrinsics);
    const parameterizer = new LocalSurfaceParameterizer();

    const outcome = parameterizer.build({
      geometry: sphere.geometry,
      imagePoint: { x: 450, y: 450 },
      imageSize,
      maxRadius: 0.72,
    });

    expect(outcome.valid).toBe(true);
    const mapping = outcome.mapping;
    if (!mapping) throw new Error('Expected valid sphere SurfaceMapping');
    expect(mapping.diagnostics.flippedTriangleCount).toBe(0);
    expect(mapping.diagnostics.acceptedTriangleCount).toBeGreaterThan(200);
    expect(mapping.diagnostics.maxStretchRatio).toBeLessThan(1.45);

    // Verify image <-> surface UV round-trip consistency off center.
    const probeImg = { x: 540, y: 390 };
    const sample = sampleSurfaceMappingAtImagePoint(mapping, probeImg, imageSize);
    expect(sample).not.toBeNull();
    if (!sample) throw new Error('Expected valid sphere UV sample');
    expect(sample.u).toBeGreaterThan(0.05);
    expect(sample.v).toBeLessThan(-0.02);

    const roundTrip = mapSurfaceUvToImagePoint(mapping, sample.u, sample.v, imageSize, intr);
    expect(roundTrip).not.toBeNull();
    if (roundTrip) {
      expect(Math.hypot(roundTrip.x - probeImg.x, roundTrip.y - probeImg.y)).toBeLessThan(1.5);
    }
  });

  it('handles oblique planes isometrically and isolates across holes and disconnected objects', () => {
    const tiltRad = Math.PI / 5;
    const cyl = createSyntheticCylinderScene({
      width: 80,
      height: 80,
      axisCenter: [0, 0, 2.4],
      radius: 0.85,
      occluders: [{ minCol: 52, maxCol: 66, minRow: 10, maxRow: 70, depth: 0.9 }],
      invalidRegions: [{ minCol: 14, maxCol: 24, minRow: 28, maxRow: 46 }],
      backgrounds: [{ minCol: 0, maxCol: 8, minRow: 0, maxRow: 79, depth: 6.5 }],
    });
    const imageSize = { width: 800, height: 800 };
    const parameterizer = new LocalSurfaceParameterizer();

    const cylOutcome = parameterizer.build({
      geometry: cyl.geometry,
      imagePoint: { x: 380, y: 400 },
      imageSize,
      maxRadius: 1.5,
    });
    expect(cylOutcome.valid).toBe(true);
    const cylMap = cylOutcome.mapping;
    if (!cylMap) throw new Error('Expected valid cylinder SurfaceMapping');

    // Must never cross into foreground occluder, hole, or background slab.
    for (let i = 0; i < cylMap.vertexGridIndices.length; i++) {
      const gIdx = cylMap.vertexGridIndices[i] ?? 0;
      expect(cyl.truthPlaneMask[gIdx]).toBe(1);
    }

    // Check oblique plane as well for exact metric accuracy.
    const wall = createSyntheticPlanarScene({
      width: 64,
      height: 64,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [Math.sin(tiltRad), 0, -Math.cos(tiltRad)],
    });
    const wallOutcome = parameterizer.build({
      geometry: wall.geometry,
      imagePoint: { x: 320, y: 320 },
      imageSize: { width: 640, height: 640 },
      maxRadius: 1.1,
    });
    expect(wallOutcome.valid).toBe(true);
    expect(wallOutcome.mapping?.diagnostics.meanStretchError ?? Infinity).toBeLessThan(0.01);
  });

  it('estimates normals cleanly on noisy wavy surfaces without precomputed normals', () => {
    const noisyCyl = createSyntheticCylinderScene({
      width: 84,
      height: 84,
      axisCenter: [0, 0, 2.3],
      radius: 0.85,
      noiseAmplitude: 0.005,
    });
    const geomNoNormals: GeometryInput = {
      ...noisyCyl.geometry,
      normals: null,
    };

    const parameterizer = new LocalSurfaceParameterizer();
    const noisyOutcome = parameterizer.build({
      geometry: geomNoNormals,
      imagePoint: { x: 420, y: 420 },
      imageSize: { width: 840, height: 840 },
      maxRadius: 0.85,
    });

    expect(noisyOutcome.valid).toBe(true);
    expect(noisyOutcome.diagnostics.usedEstimatedNormals).toBe(true);
    expect(noisyOutcome.diagnostics.flippedTriangleCount).toBe(0);

    const wavy = buildWavyIrregularScene(80, 80);
    const wavyOutcome = parameterizer.build({
      geometry: wavy,
      imagePoint: { x: 400, y: 400 },
      imageSize: { width: 800, height: 800 },
      maxRadius: 0.75,
    });
    expect(wavyOutcome.valid).toBe(true);
    expect(wavyOutcome.diagnostics.acceptedTriangleCount).toBeGreaterThan(250);
  });

  it('rejects folded or degenerately distorted surface regions cleanly with diagnostics', () => {
    const parameterizer = new LocalSurfaceParameterizer();

    // 1. Tight Gaussian curvature beyond strict stretch limit prunes excessive-stretch outer ring and rejects when >60% fail.
    const tightSphere = createSyntheticSphereScene({
      width: 64,
      height: 64,
      center: [0, 0, 1.5],
      radius: 0.22,
    });
    const stretchOutcome = parameterizer.build({
      geometry: tightSphere.geometry,
      imagePoint: { x: 320, y: 320 },
      imageSize: { width: 640, height: 640 },
      maxRadius: 0.34,
      maxStretch: 1.01,
    });
    expect(stretchOutcome.valid).toBe(false);
    expect(stretchOutcome.mapping).toBeNull();
    expect(stretchOutcome.diagnostics.rejectionReason).toBeTruthy();
    expect(stretchOutcome.diagnostics.excessiveStretchTriangleCount).toBeGreaterThan(0);

    // 2. Heavily fragmented / sawtooth discontinuous region cannot form a valid mapped patch.
    const fragmented = buildSharpCreaseFragmentedScene(48, 48);
    const fragOutcome = parameterizer.build({
      geometry: fragmented,
      imagePoint: { x: 240, y: 240 },
      imageSize: { width: 480, height: 480 },
      maxRadius: 0.35,
    });
    expect(fragOutcome.valid).toBe(false);
    expect(fragOutcome.mapping).toBeNull();
    expect(fragOutcome.diagnostics.rejectionReason).toBeTruthy();
  });
});
