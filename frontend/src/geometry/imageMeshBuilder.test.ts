import { describe, expect, it } from 'vitest';
import { ImageMeshBuilder } from './imageMeshBuilder';
import { createSyntheticPlanarScene } from './testFixtures';

describe('ImageMeshBuilder', () => {
  it('builds a valid compacted 2.5D grid triangle mesh for a continuous planar surface', () => {
    const w = 16;
    const h = 12;
    const scene = createSyntheticPlanarScene({
      width: w,
      height: h,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
    });

    const meshGeom = ImageMeshBuilder.build(scene.geometry, {
      coordinateSystem: 'threejs',
      colors: scene.colors,
    });

    const posAttr = meshGeom.getAttribute('position');
    const uvAttr = meshGeom.getAttribute('uv');
    const normAttr = meshGeom.getAttribute('normal');
    const colorAttr = meshGeom.getAttribute('color');
    const index = meshGeom.getIndex();

    expect(posAttr.count).toBe(w * h);
    expect(uvAttr.count).toBe(w * h);
    expect(normAttr.count).toBe(w * h);
    expect(colorAttr.count).toBe(w * h);
    if (!index) {
      throw new Error('Expected non-null triangle index buffer');
    }
    expect(index.count).toBe((w - 1) * (h - 1) * 6);

    // Three.js coordinate flip negates Y and Z compared to OpenCV camera space.
    for (let i = 0; i < posAttr.count; i++) {
      expect(Number.isFinite(posAttr.getX(i))).toBe(true);
      expect(Number.isFinite(posAttr.getY(i))).toBe(true);
      expect(posAttr.getZ(i)).toBeCloseTo(-3.0, 5);
    }

    meshGeom.dispose();
  });

  it('rejects discontinuity triangles across foreground-to-background depth jumps and excludes invalid vertices', () => {
    const w = 24;
    const h = 24;
    const occluder = { minCol: 8, maxCol: 14, minRow: 8, maxRow: 14, depth: 1.5 };
    const invalidRect = { minCol: 2, maxCol: 4, minRow: 2, maxRow: 4 };
    const scene = createSyntheticPlanarScene({
      width: w,
      height: h,
      planeOrigin: [0, 0, 4.0],
      planeNormal: [0, 0, -1],
      occluders: [occluder],
      invalidRegions: [invalidRect],
    });

    const meshGeom = ImageMeshBuilder.build(scene.geometry, { coordinateSystem: 'opencv' });
    const posAttr = meshGeom.getAttribute('position');
    const index = meshGeom.getIndex();
    if (!index) {
      throw new Error('Expected non-null triangle index buffer');
    }

    // Compact vertex count should exclude the 3x3 = 9 invalid pixels.
    expect(posAttr.count).toBe(w * h - 9);

    // No accepted triangle may bridge between foreground depth (~1.5) and background (~4.0).
    const idxArr = index.array;
    for (let i = 0; i < idxArr.length; i += 3) {
      const a = idxArr[i] ?? 0;
      const b = idxArr[i + 1] ?? 0;
      const c = idxArr[i + 2] ?? 0;
      const za = posAttr.getZ(a);
      const zb = posAttr.getZ(b);
      const zc = posAttr.getZ(c);
      const maxSpan = Math.max(za, zb, zc) - Math.min(za, zb, zc);
      expect(maxSpan).toBeLessThan(0.5);
    }

    meshGeom.dispose();
  });

  it('rejects triangles whose surface normals diverge across sharp creases', () => {
    const w = 10;
    const h = 10;
    const scene = createSyntheticPlanarScene({
      width: w,
      height: h,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
    });

    // Inject perpendicular normals along one half without jumping depth.
    if (scene.geometry.normals) {
      for (let row = 0; row < h; row++) {
        for (let col = 5; col < w; col++) {
          const base = (row * w + col) * 3;
          scene.geometry.normals[base] = 1;
          scene.geometry.normals[base + 1] = 0;
          scene.geometry.normals[base + 2] = 0;
        }
      }
    }

    const meshGeom = ImageMeshBuilder.build(scene.geometry, { minNormalDot: 0.2 });
    const index = meshGeom.getIndex();
    if (!index) {
      throw new Error('Expected non-null triangle index buffer');
    }

    // Boundary quads along col=4..5 should be rejected cleanly by normal check.
    const fullTriangles = (w - 1) * (h - 1) * 2;
    const expectedReduced = fullTriangles - (h - 1) * 2;
    expect(index.count / 3).toBe(expectedReduced);
    meshGeom.dispose();
  });

  it('returns an empty position buffer for grids smaller than 2x2', () => {
    const scene = createSyntheticPlanarScene({
      width: 1,
      height: 1,
      planeOrigin: [0, 0, 2.0],
      planeNormal: [0, 0, -1],
    });
    const geom = ImageMeshBuilder.build(scene.geometry);
    expect(geom.getAttribute('position').count).toBe(0);
    geom.dispose();
  });
});
