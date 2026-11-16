import { describe, expect, it } from 'vitest';
import { parseNormalizedIntrinsics, vecNormalize } from '../geometry/geometryAccess';
import { PlaneFrame } from '../geometry/planeFrame';
import {
  buildSyntheticSurfaceSelection,
  createSyntheticPlanarScene,
  makeSyntheticIntrinsics,
} from '../geometry/testFixtures';
import {
  PlaneVisibility,
  computeMaskIoU,
  computeSurfaceOwnerGrid,
  findSurfaceOwnerAtImagePoint,
} from './planeVisibility';
import { PaintDocument } from './paintDocument';

describe('PlaneVisibility', () => {
  it('classifies visible, occluded, background, invalid-geometry, outside-footprint, and opposing normals', () => {
    const intr = parseNormalizedIntrinsics(makeSyntheticIntrinsics());
    const frame = PlaneFrame.fromPlane({ origin: [0, 0, 4.0], normal: [0, 0, -1] });
    const footprint = { minU: -1.0, maxU: 1.0, minV: -1.0, maxV: 1.0 };

    // Case A: visible point matching plane depth and normal.
    const vis = PlaneVisibility.evaluatePixel(
      { x: 0.5, y: 0.5 },
      intr,
      frame,
      footprint,
      true,
      4.0,
      [0, 0, -1],
    );
    expect(vis.visible).toBe(true);
    expect(vis.reason).toBe('visible');

    // Case B: occluded by closer foreground surface.
    const occ = PlaneVisibility.evaluatePixel(
      { x: 0.5, y: 0.5 },
      intr,
      frame,
      footprint,
      true,
      2.2,
      [0, 0, -1],
    );
    expect(occ.visible).toBe(false);
    expect(occ.reason).toBe('occluded');

    // Case C: deeper background behind the plane.
    const bg = PlaneVisibility.evaluatePixel(
      { x: 0.5, y: 0.5 },
      intr,
      frame,
      footprint,
      true,
      6.5,
      [0, 0, -1],
    );
    expect(bg.visible).toBe(false);
    expect(bg.reason).toBe('background');

    // Case D: invalid geometry / mask=0 or non-finite depth.
    const inv = PlaneVisibility.evaluatePixel(
      { x: 0.5, y: 0.5 },
      intr,
      frame,
      footprint,
      false,
      Number.NaN,
      null,
    );
    expect(inv.visible).toBe(false);
    expect(inv.reason).toBe('invalid-geometry');

    // Case E: outside bounded decal footprint.
    // Ray at x = 0.85 -> xNorm = 0.35 -> u = 0.35 * 4.0 = 1.4 > maxU (1.0).
    const out = PlaneVisibility.evaluatePixel(
      { x: 0.85, y: 0.5 },
      intr,
      frame,
      footprint,
      true,
      4.0,
      [0, 0, -1],
    );
    expect(out.visible).toBe(false);
    expect(out.reason).toBe('outside-footprint');

    // Normal incompatibility at matched depth (e.g. grazing perpendicular strip).
    const incompat = PlaneVisibility.evaluatePixel(
      { x: 0.5, y: 0.5 },
      intr,
      frame,
      footprint,
      true,
      4.0,
      [1, 0, 0],
    );
    expect(incompat.visible).toBe(false);
    expect(incompat.reason).toBe('incompatible-surface');
  });

  it('achieves >= 0.98 visibility IoU on synthetic front-facing, oblique, and occluded scenes', () => {
    const w = 48;
    const h = 48;

    // 1) Front-facing wall scene with subtle depth jitter.
    const frontScene = createSyntheticPlanarScene({
      width: w,
      height: h,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
      noiseAmplitude: 0.004,
    });
    const frontSelection = buildSyntheticSurfaceSelection(frontScene, 24, 24);
    const frontMask = PlaneVisibility.computeGridMask(
      frontScene.geometry,
      parseNormalizedIntrinsics(frontScene.geometry.metadata.intrinsics),
      frontSelection.frame,
      frontSelection.suggestedFootprint,
      { fitRms: frontSelection.rmsError },
    );
    expect(computeMaskIoU(frontMask, frontScene.truthPlaneMask)).toBeGreaterThanOrEqual(0.98);

    // 2) Oblique slanted wall scene.
    const obliqueNormal = vecNormalize([0.55, -0.25, -0.8]) ?? [0, 0, -1];
    const obliqueScene = createSyntheticPlanarScene({
      width: w,
      height: h,
      planeOrigin: [0, 0, 3.8],
      planeNormal: obliqueNormal,
      noiseAmplitude: 0.004,
    });
    const obliqueSelection = buildSyntheticSurfaceSelection(obliqueScene, 24, 24);
    const obliqueMask = PlaneVisibility.computeGridMask(
      obliqueScene.geometry,
      parseNormalizedIntrinsics(obliqueScene.geometry.metadata.intrinsics),
      obliqueSelection.frame,
      obliqueSelection.suggestedFootprint,
      { fitRms: obliqueSelection.rmsError },
    );
    expect(computeMaskIoU(obliqueMask, obliqueScene.truthPlaneMask)).toBeGreaterThanOrEqual(0.98);

    // 3) Occluded slanted scene with foreground box, deeper background region, and invalid hole.
    const occludedScene = createSyntheticPlanarScene({
      width: w,
      height: h,
      planeOrigin: [0, 0, 4.0],
      planeNormal: vecNormalize([0.4, 0.15, -0.9]) ?? [0, 0, -1],
      occluders: [{ minCol: 18, maxCol: 28, minRow: 14, maxRow: 34, depth: 1.8 }],
      backgrounds: [{ minCol: 38, maxCol: 47, minRow: 0, maxRow: 47, depth: 9.0 }],
      invalidRegions: [{ minCol: 2, maxCol: 6, minRow: 2, maxRow: 6 }],
    });
    const occludedSelection = buildSyntheticSurfaceSelection(occludedScene, 10, 24);
    const occludedMask = PlaneVisibility.computeGridMask(
      occludedScene.geometry,
      parseNormalizedIntrinsics(occludedScene.geometry.metadata.intrinsics),
      occludedSelection.frame,
      occludedSelection.suggestedFootprint,
      { fitRms: occludedSelection.rmsError },
    );
    expect(computeMaskIoU(occludedMask, occludedScene.truthPlaneMask)).toBeGreaterThanOrEqual(0.98);
  });

  it('resolves intersecting overlapping surfaces strictly according to priority order', () => {
    const w = 24;
    const h = 24;
    const scene = createSyntheticPlanarScene({
      width: w,
      height: h,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
    });
    const selA = buildSyntheticSurfaceSelection(scene, 10, 12);
    const selB = buildSyntheticSurfaceSelection(scene, 14, 12);

    let doc = PaintDocument.createInitial();
    doc = PaintDocument.selectSurface(doc, selA);
    const surfA = doc.surfaces[0];
    doc = PaintDocument.selectSurface(doc, selB);
    const surfB = doc.surfaces[1];
    if (!surfA || !surfB) throw new Error('Expected two surfaces');

    // Center pixel belongs to both planes; index 0 (surfA) wins priority #1.
    const owner1 = findSurfaceOwnerAtImagePoint(
      doc.surfaces,
      { x: 12, y: 12 },
      { width: w, height: h },
      scene.geometry,
    );
    expect(owner1?.id).toBe(surfA.id);
    const grid1 = computeSurfaceOwnerGrid(doc.surfaces, scene.geometry);
    expect(grid1[12 * w + 12]).toBe(1);

    // Move surfB above surfA; ownership immediately switches to surfB.
    doc = PaintDocument.reorderSurfaces(doc, 1, 0);
    const owner2 = findSurfaceOwnerAtImagePoint(
      doc.surfaces,
      { x: 12, y: 12 },
      { width: w, height: h },
      scene.geometry,
    );
    expect(owner2?.id).toBe(surfB.id);
    const grid2 = computeSurfaceOwnerGrid(doc.surfaces, scene.geometry);
    expect(grid2[12 * w + 12]).toBe(1);
  });
});
