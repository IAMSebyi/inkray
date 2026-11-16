import { describe, expect, it } from 'vitest';
import { parseNormalizedIntrinsics } from './geometryAccess';
import {
  buildMetricEllipseImagePolygon,
  computeGridSurfaceMetric,
  computeInterpolatedSurfaceMetric,
  computeMetricEllipseSemiAxes,
  computeSurfaceBrushRadius,
  evaluateMetricDistanceSq,
} from './localSurfaceMetric';
import {
  createSyntheticCylinderScene,
  createSyntheticPlanarScene,
  createSyntheticSphereScene,
} from './testFixtures';
import { getSurfaceGridIndex } from './surfacePicking';

describe('localSurfaceMetric', () => {
  it('computes symmetric first fundamental form on front-facing planes and foreshortening on oblique planes', () => {
    const front = createSyntheticPlanarScene({
      width: 48,
      height: 48,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
    });
    const frontGrid = getSurfaceGridIndex(front.geometry);
    const frontMetric = computeGridSurfaceMetric(frontGrid, front.geometry, 24, 24, {
      width: 480,
      height: 480,
    });
    expect(frontMetric.valid).toBe(true);
    expect(frontMetric.gxx).toBeGreaterThan(0);
    expect(frontMetric.gyy).toBeGreaterThan(0);
    expect(Math.abs(frontMetric.gxy)).toBeLessThan(frontMetric.gxx * 0.05);
    expect(frontMetric.gxx / frontMetric.gyy).toBeCloseTo(1.0, 2);

    const tiltRad = Math.PI / 3; // 60 degrees tilt toward +X -> 1 / cos(60) = 2x stretch along X.
    const oblique = createSyntheticPlanarScene({
      width: 64,
      height: 64,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [Math.sin(tiltRad), 0, -Math.cos(tiltRad)],
    });
    const oblGrid = getSurfaceGridIndex(oblique.geometry);
    const oblMetric = computeGridSurfaceMetric(oblGrid, oblique.geometry, 32, 32, {
      width: 640,
      height: 640,
    });
    expect(oblMetric.valid).toBe(true);
    const stepRatioXvsY = Math.sqrt(oblMetric.gxx / oblMetric.gyy);
    expect(stepRatioXvsY).toBeCloseTo(1 / Math.cos(tiltRad), 1);

    // Ellipse image semi-axes must match surface radius foreshortening along oblique slant.
    const surfaceRadius = 0.2;
    const axes = computeMetricEllipseSemiAxes(oblMetric, surfaceRadius);
    expect(axes).not.toBeNull();
    if (axes) {
      const major = Math.max(axes.axis1Px, axes.axis2Px);
      const minor = Math.min(axes.axis1Px, axes.axis2Px);
      expect(major / minor).toBeCloseTo(1 / Math.cos(tiltRad), 1);
    }

    // Distance squared along Y vs X reflects exact oblique metric.
    const dSqY = evaluateMetricDistanceSq(oblMetric, { x: 0, y: 0 }, { x: 0, y: 10 });
    const dSqX = evaluateMetricDistanceSq(oblMetric, { x: 0, y: 0 }, { x: 10, y: 0 });
    expect(Math.sqrt(dSqX / dSqY)).toBeCloseTo(1 / Math.cos(tiltRad), 1);
  });

  it('tracks smooth anisotropy on curved cylinders and spheres while staying continuous under interpolation', () => {
    const cyl = createSyntheticCylinderScene({
      width: 64,
      height: 64,
      axisCenter: [0, 0, 2.5],
      radius: 1.0,
    });
    const cylGrid = getSurfaceGridIndex(cyl.geometry);
    const centerMetric = computeInterpolatedSurfaceMetric(
      cylGrid,
      cyl.geometry,
      { x: 320, y: 320 },
      { width: 640, height: 640 },
    );
    const limbMetric = computeInterpolatedSurfaceMetric(
      cylGrid,
      cyl.geometry,
      { x: 460, y: 320 },
      { width: 640, height: 640 },
    );
    expect(centerMetric.valid).toBe(true);
    expect(limbMetric.valid).toBe(true);
    // Horizontal surface stretch gxx increases dramatically near the curved cylinder limb vs center.
    expect(limbMetric.gxx).toBeGreaterThan(centerMetric.gxx * 1.4);

    const sphere = createSyntheticSphereScene({
      width: 64,
      height: 64,
      center: [0, 0, 2.5],
      radius: 1.1,
    });
    const sphGrid = getSurfaceGridIndex(sphere.geometry);
    const pole = computeInterpolatedSurfaceMetric(
      sphGrid,
      sphere.geometry,
      { x: 320, y: 320 },
      { width: 640, height: 640 },
    );
    expect(pole.valid).toBe(true);
    expect(pole.gxx / pole.gyy).toBeCloseTo(1, 1);

    const poly = buildMetricEllipseImagePolygon(pole, { x: 320, y: 320 }, 0.15, 24);
    expect(poly).not.toBeNull();
    expect(poly?.length).toBe(24);
  });

  it('isolates finite-difference derivatives at occluder boundaries and scales surface radius across depth', () => {
    const scene = createSyntheticPlanarScene({
      width: 48,
      height: 48,
      planeOrigin: [0, 0, 4.0],
      planeNormal: [0, 0, -1],
      occluders: [{ minCol: 24, maxCol: 47, minRow: 0, maxRow: 47, depth: 1.2 }],
      invalidRegions: [{ minCol: 0, maxCol: 6, minRow: 0, maxRow: 47 }],
    });
    const grid = getSurfaceGridIndex(scene.geometry);
    // Col 23 sits right next to a massive depth jump at Col 24: one-sided difference avoids spike.
    const edgeMetric = computeGridSurfaceMetric(grid, scene.geometry, 23, 24, {
      width: 480,
      height: 480,
    });
    const interiorMetric = computeGridSurfaceMetric(grid, scene.geometry, 15, 24, {
      width: 480,
      height: 480,
    });
    expect(edgeMetric.valid).toBe(true);
    expect(interiorMetric.valid).toBe(true);
    expect(Math.sqrt(edgeMetric.gxx)).toBeCloseTo(Math.sqrt(interiorMetric.gxx), 2);

    const invalidMetric = computeGridSurfaceMetric(grid, scene.geometry, 3, 24, {
      width: 480,
      height: 480,
    });
    expect(invalidMetric.valid).toBe(false);

    const intr = parseNormalizedIntrinsics(scene.geometry.metadata.intrinsics);
    const rNear = computeSurfaceBrushRadius(2.0, 40, intr, { width: 480, height: 480 });
    const rFar = computeSurfaceBrushRadius(4.0, 40, intr, { width: 480, height: 480 });
    expect(rFar).toBeCloseTo(rNear * 2, 5);
  });
});
