import { describe, expect, it } from 'vitest';
import { classifyAndAnchorSurface } from './autoSurfaceAnchor';
import { pickSurfaceHit } from './surfacePicking';
import {
  createSyntheticCylinderScene,
  createSyntheticPlanarScene,
  createSyntheticSphereScene,
} from './testFixtures';

describe('autoSurfaceAnchor', () => {
  it('maps image coordinates accurately to the geometry grid and falls back cleanly on invalid or missing regions', () => {
    const plane = createSyntheticPlanarScene({
      width: 64,
      height: 48,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
      invalidRegions: [{ minCol: 0, maxCol: 15, minRow: 0, maxRow: 15 }],
    });
    const imageSize = { width: 1280, height: 960 };

    const validHit = pickSurfaceHit(plane.geometry, { x: 640, y: 480 }, imageSize);
    expect(validHit.valid).toBe(true);
    expect(validHit.pixel?.x).toBe(32);
    expect(validHit.pixel?.y).toBe(24);
    expect(validHit.depth).toBeCloseTo(3.0, 2);

    const invalidHit = pickSurfaceHit(plane.geometry, { x: 100, y: 100 }, imageSize);
    expect(invalidHit.valid).toBe(false);

    const anchorInvalid = classifyAndAnchorSurface({
      geometry: plane.geometry,
      imagePoint: { x: 100, y: 100 },
      imageSize,
      brushPixelDiameter: 40,
      existingSurfaces: [],
      nextSurfaceNumber: 1,
    });
    expect(anchorInvalid.mode).toBe('fallback-2d');

    const anchorMissingGeom = classifyAndAnchorSurface({
      geometry: null,
      imagePoint: { x: 640, y: 480 },
      imageSize,
      brushPixelDiameter: 40,
      existingSurfaces: [],
      nextSurfaceNumber: 1,
    });
    expect(anchorMissingGeom.mode).toBe('fallback-2d');
  });

  it('classifies flat and oblique planes as planar while bridging coplanar regions across occluders', () => {
    const wallWithPerson = createSyntheticPlanarScene({
      width: 72,
      height: 72,
      planeOrigin: [0, 0, 4.0],
      planeNormal: [Math.sin(Math.PI / 6), 0, -Math.cos(Math.PI / 6)],
      occluders: [{ minCol: 30, maxCol: 42, minRow: 10, maxRow: 71, depth: 1.5 }],
    });
    const imageSize = { width: 720, height: 720 };

    // Press left of foreground occluder -> fits single planar wall bridging both left and right sides.
    const anchor = classifyAndAnchorSurface({
      geometry: wallWithPerson.geometry,
      imagePoint: { x: 160, y: 360 },
      imageSize,
      brushPixelDiameter: 36,
      existingSurfaces: [],
      nextSurfaceNumber: 1,
    });
    expect(anchor.mode).toBe('planar');
    if (anchor.mode !== 'planar') return;

    expect(anchor.reusedExisting).toBe(false);
    // Support mask excludes foreground occluder (col 36) while bridging coplanar right side (col 56)!
    expect(anchor.surface.supportMask[36 * 72 + 16]).toBe(1);
    expect(anchor.surface.supportMask[36 * 72 + 36]).toBe(0);
    expect(anchor.surface.supportMask[36 * 72 + 56]).toBe(1);

    // Subsequent stroke on same plane (e.g. right side of occluder) reuses existing plane surface cleanly.
    const nextAnchor = classifyAndAnchorSurface({
      geometry: wallWithPerson.geometry,
      imagePoint: { x: 560, y: 360 },
      imageSize,
      brushPixelDiameter: 36,
      existingSurfaces: [anchor.surface],
      nextSurfaceNumber: 2,
    });
    expect(nextAnchor.mode).toBe('planar');
    if (nextAnchor.mode === 'planar') {
      expect(nextAnchor.reusedExisting).toBe(true);
      expect(nextAnchor.surface.id).toBe(anchor.surface.id);
    }
  });

  it('classifies cylindrical and spherical regions cleanly as curved without false planar flattening', () => {
    const cyl = createSyntheticCylinderScene({
      width: 72,
      height: 72,
      axisCenter: [0, 0, 2.2],
      radius: 0.9,
      noiseAmplitude: 0.008,
    });
    const cylAnchor = classifyAndAnchorSurface({
      geometry: cyl.geometry,
      imagePoint: { x: 360, y: 360 },
      imageSize: { width: 720, height: 720 },
      brushPixelDiameter: 36,
      existingSurfaces: [],
      nextSurfaceNumber: 1,
    });
    expect(cylAnchor.mode).toBe('curved');

    const sphere = createSyntheticSphereScene({
      width: 72,
      height: 72,
      center: [0, 0, 2.2],
      radius: 0.95,
    });
    const sphAnchor = classifyAndAnchorSurface({
      geometry: sphere.geometry,
      imagePoint: { x: 360, y: 360 },
      imageSize: { width: 720, height: 720 },
      brushPixelDiameter: 36,
      existingSurfaces: [],
      nextSurfaceNumber: 1,
    });
    expect(sphAnchor.mode).toBe('curved');
  });
});
