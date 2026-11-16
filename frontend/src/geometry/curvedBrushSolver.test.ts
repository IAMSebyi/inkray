import { describe, expect, it } from 'vitest';
import { ImageSpaceBrushEngine } from '../painting/imageSpacePaint';
import { CurvedBrushWorkspace, stampCurvedBrushDab } from './curvedBrushSolver';
import { parseNormalizedIntrinsics } from './geometryAccess';
import { computeSurfaceBrushRadius } from './localSurfaceMetric';
import { findConnectedMeshComponent, getSurfaceGridIndex, pickSurfaceHit } from './surfacePicking';
import { createSyntheticCylinderScene, createSyntheticPlanarScene } from './testFixtures';

describe('curvedBrushSolver & ImageSpaceBrushEngine', () => {
  it('agrees accurately with analytical planar & cylindrical distances within bounded geodesic radius', () => {
    const cyl = createSyntheticCylinderScene({
      width: 64,
      height: 64,
      axisCenter: [0, 0, 2.5],
      radius: 1.0,
    });
    const grid = getSurfaceGridIndex(cyl.geometry);
    const imageSize = { width: 640, height: 640 };
    const intr = parseNormalizedIntrinsics(cyl.geometry.metadata.intrinsics);
    const centerImg = { x: 320, y: 320 };
    const hit = pickSurfaceHit(cyl.geometry, centerImg, imageSize);
    expect(hit.valid).toBe(true);
    if (!hit.pixel || hit.depth === null) {
      throw new Error('Expected valid cylinder center hit');
    }

    const componentMask = findConnectedMeshComponent(grid, hit.pixel.x, hit.pixel.y);
    const surfaceRadius = computeSurfaceBrushRadius(hit.depth, 180, intr, imageSize);
    const workspace = new CurvedBrushWorkspace();
    const res = workspace.runBoundedGeodesic({
      gridIndex: grid,
      geometry: cyl.geometry,
      hit,
      maxRadius: surfaceRadius * 1.15,
      componentMask,
    });

    expect(res.visitedCount).toBeGreaterThan(20);
    // Corners outside maxRadius must never be reached.
    expect(res.seenGen[0]).not.toBe(res.generation);

    // Every visited cylinder pixel geodesic distance closely tracks analytical cylinder surface arc distance.
    const centerPt = hit.point3D;
    if (!centerPt) {
      throw new Error('Expected hit center point3D');
    }
    const cx = 0;
    const cz = 2.5;
    // Measure azimuth relative to camera-facing -Z so the angle stays branch-cut free around centerPt.
    const theta0 = Math.atan2(centerPt[0] - cx, cz - centerPt[2]);
    const radius = 1.0;

    let maxErrRatio = 0;
    for (let i = 0; i < res.visitedCount; i++) {
      const idx = res.visitedIndices[i] ?? 0;
      const px = cyl.geometry.points[idx * 3] ?? 0;
      const py = cyl.geometry.points[idx * 3 + 1] ?? 0;
      const pz = cyl.geometry.points[idx * 3 + 2] ?? 0;
      const theta = Math.atan2(px - cx, cz - pz);
      const dArc = radius * Math.abs(theta - theta0);
      const dy = py - centerPt[1];
      const trueCylDist = Math.hypot(dArc, dy);
      const geoDist = res.distArr[idx] ?? Number.POSITIVE_INFINITY;
      if (geoDist <= surfaceRadius && trueCylDist > 0.04) {
        const errRatio = Math.abs(geoDist - trueCylDist) / trueCylDist;
        if (errRatio > maxErrRatio) {
          maxErrRatio = errRatio;
        }
      }
    }
    expect(maxErrRatio).toBeLessThan(0.12);

    // Also verify bounded geodesic agrees with exact Euclidean distance on an oblique 3D plane.
    const obl = createSyntheticPlanarScene({
      width: 64,
      height: 64,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [Math.sin(Math.PI / 4), 0, -Math.cos(Math.PI / 4)],
    });
    const oblGrid = getSurfaceGridIndex(obl.geometry);
    const oblHit = pickSurfaceHit(obl.geometry, centerImg, imageSize);
    expect(oblHit.valid).toBe(true);
    if (!oblHit.point3D || oblHit.depth === null) {
      throw new Error('Expected valid oblique hit');
    }
    const oblRadius = computeSurfaceBrushRadius(oblHit.depth, 140, intr, imageSize);
    const oblRes = workspace.runBoundedGeodesic({
      gridIndex: oblGrid,
      geometry: obl.geometry,
      hit: oblHit,
      maxRadius: oblRadius * 1.1,
    });
    let maxPlaneErrRatio = 0;
    for (let i = 0; i < oblRes.visitedCount; i++) {
      const idx = oblRes.visitedIndices[i] ?? 0;
      const px = obl.geometry.points[idx * 3] ?? 0;
      const py = obl.geometry.points[idx * 3 + 1] ?? 0;
      const pz = obl.geometry.points[idx * 3 + 2] ?? 0;
      const euclid = Math.hypot(
        px - oblHit.point3D[0],
        py - oblHit.point3D[1],
        pz - oblHit.point3D[2],
      );
      const geoDist = oblRes.distArr[idx] ?? Number.POSITIVE_INFINITY;
      if (geoDist <= oblRadius && euclid > 0.06) {
        const ratio = Math.abs(geoDist - euclid) / euclid;
        if (ratio > maxPlaneErrRatio) maxPlaneErrRatio = ratio;
      }
    }
    // Bounded 8-connected graph Dijkstra has known ~13% knight-move zig-zag upper bound on 45deg foreshortened grids.
    expect(maxPlaneErrRatio).toBeLessThan(0.15);
  });

  it('never leaks geodesic distance across depth discontinuities, invalid holes, or disconnected components', () => {
    // Disjoint wall strips separated by a closer occluding pillar down the center.
    const scene = createSyntheticPlanarScene({
      width: 64,
      height: 64,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
      occluders: [{ minCol: 28, maxCol: 36, minRow: 0, maxRow: 63, depth: 1.0 }],
      invalidRegions: [{ minCol: 18, maxCol: 21, minRow: 0, maxRow: 63 }],
    });
    const grid = getSurfaceGridIndex(scene.geometry);
    const imageSize = { width: 640, height: 640 };

    // Anchor on left background strip (col 12, row 32).
    const compLeft = findConnectedMeshComponent(grid, 12, 32);
    expect(compLeft[32 * 64 + 12]).toBe(1);
    // Occluder pillar (col 32) and invalid gap (col 20) and isolated middle-left strip (col 24) are NOT connected.
    expect(compLeft[32 * 64 + 32]).toBe(0);
    expect(compLeft[32 * 64 + 20]).toBe(0);
    expect(compLeft[32 * 64 + 24]).toBe(0);

    const workspace = new CurvedBrushWorkspace();
    const mask = new Uint8Array(imageSize.width * imageSize.height);
    const { bounds } = stampCurvedBrushDab({
      strokeMask: mask,
      texSpec: { width: imageSize.width, height: imageSize.height },
      imageSize,
      centerImagePt: { x: 140, y: 320 },
      surfaceRadius: 0.45,
      hardness: 0.8,
      geometry: scene.geometry,
      gridIndex: grid,
      componentMask: compLeft,
      workspace,
    });
    expect(bounds).not.toBeNull();
    // Zero alpha leakage into invalid hole or across hole into isolated strip or occluder!
    expect(mask[320 * 640 + 200]).toBe(0);
    expect(mask[320 * 640 + 240]).toBe(0);
    expect(mask[320 * 640 + 320]).toBe(0);
    // Valid interior around col 14 receives paint cleanly.
    expect(mask[320 * 640 + 140]).toBeGreaterThan(240);
  });

  it('suspends curved stroke across holes/occluders and resumes on same component without straight-line bridging', () => {
    const cyl = createSyntheticCylinderScene({
      width: 64,
      height: 64,
      axisCenter: [0, 0, 2.5],
      radius: 1.0,
      occluders: [{ minCol: 29, maxCol: 34, minRow: 20, maxRow: 44, depth: 0.8 }],
    });
    const grid = getSurfaceGridIndex(cyl.geometry);
    const imageSize = { width: 640, height: 640 };
    const intr = parseNormalizedIntrinsics(cyl.geometry.metadata.intrinsics);
    const hitLeft = pickSurfaceHit(cyl.geometry, { x: 220, y: 320 }, imageSize);
    expect(hitLeft.valid).toBe(true);
    if (!hitLeft.pixel || hitLeft.depth === null) {
      throw new Error('Expected left cylinder hit');
    }
    const compMask = findConnectedMeshComponent(grid, hitLeft.pixel.x, hitLeft.pixel.y);
    // Notice cylinder top/bottom rows connect left & right sides around the occluder box!
    expect(compMask[32 * 64 + 42]).toBe(1);
    expect(compMask[32 * 64 + 32]).toBe(0);

    const engine = new ImageSpaceBrushEngine();
    const ok = engine.beginCurvedStroke({
      imagePoint: { x: 220, y: 320 },
      imageSize,
      geometry: cyl.geometry,
      componentMask: compMask,
      surfaceRadius: computeSurfaceBrushRadius(hitLeft.depth, 28, intr, imageSize),
      brush: { diameter: 28, color: '#ff3366', opacity: 0.9, hardness: 0.8 },
      mode: 'paint',
    });
    expect(ok).toBe(true);

    // Drag smoothly straight across the foreground occluder box (x = 315) to x = 420.
    engine.extendStroke({ x: 315, y: 320 });
    engine.extendStroke({ x: 420, y: 320 });

    const committed = engine.commitStroke();
    expect(committed).not.toBeNull();
    if (!committed) return;

    const sampleCommitted = (ix: number, iy: number): number => {
      if (
        ix < committed.bounds.minX ||
        ix > committed.bounds.maxX ||
        iy < committed.bounds.minY ||
        iy > committed.bounds.maxY
      ) {
        return 0;
      }
      const w = committed.bounds.maxX - committed.bounds.minX + 1;
      return (
        committed.alphaMask[(iy - committed.bounds.minY) * w + (ix - committed.bounds.minX)] ?? 0
      );
    };

    // Painted on left side and resumed on right side of cylinder, with ZERO paint on occluder box center!
    expect(sampleCommitted(220, 320)).toBeGreaterThan(200);
    expect(sampleCommitted(420, 320)).toBeGreaterThan(200);
    expect(sampleCommitted(315, 320)).toBe(0);
  });

  it('supports stable noisy curved meshes and ordinary 2D fallback strokes without geometry', () => {
    const noisyCyl = createSyntheticCylinderScene({
      width: 64,
      height: 64,
      axisCenter: [0, 0, 2.5],
      radius: 1.0,
      noiseAmplitude: 0.004,
    });
    const grid = getSurfaceGridIndex(noisyCyl.geometry);
    const imageSize = { width: 320, height: 320 };
    const intr = parseNormalizedIntrinsics(noisyCyl.geometry.metadata.intrinsics);
    const hit = pickSurfaceHit(noisyCyl.geometry, { x: 160, y: 160 }, imageSize);
    expect(hit.valid).toBe(true);
    if (!hit.pixel || hit.depth === null) {
      throw new Error('Expected hit on noisy cylinder');
    }
    const comp = findConnectedMeshComponent(grid, hit.pixel.x, hit.pixel.y);
    const engine = new ImageSpaceBrushEngine();
    expect(
      engine.beginCurvedStroke({
        imagePoint: { x: 160, y: 160 },
        imageSize,
        geometry: noisyCyl.geometry,
        componentMask: comp,
        surfaceRadius: computeSurfaceBrushRadius(hit.depth, 36, intr, imageSize),
        brush: { diameter: 36, color: '#22cc88', opacity: 0.8, hardness: 0.7 },
        mode: 'paint',
      }),
    ).toBe(true);
    engine.extendStroke({ x: 185, y: 160 });
    expect(engine.commitStroke()).not.toBeNull();

    // Ordinary 2D fallback stroke when no geometry is available.
    expect(
      engine.beginFallback2dStroke({
        imagePoint: { x: 50, y: 50 },
        imageSize,
        pixelDiameter: 24,
        brush: { diameter: 24, color: '#3388ff', opacity: 1.0, hardness: 0.9 },
        mode: 'paint',
      }),
    ).toBe(true);
    engine.extendStroke({ x: 100, y: 50 });
    const stroke2d = engine.commitStroke();
    expect(stroke2d?.layer).toBe('fallback-2d');
    expect(stroke2d?.bounds.minX).toBeLessThanOrEqual(40);
    expect(stroke2d?.bounds.maxX).toBeGreaterThanOrEqual(110);
  });
});
