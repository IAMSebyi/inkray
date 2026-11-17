import { describe, expect, it } from 'vitest';
import { type GeometryInput, parseNormalizedIntrinsics } from '../geometry/geometryAccess';
import {
  createSyntheticCylinderScene,
  createSyntheticPlanarScene,
  createSyntheticSphereScene,
  makeSyntheticIntrinsics,
} from '../geometry/testFixtures';
import { ArtworkModel, rasterizeArtworkRgba } from './artwork';
import {
  computePlacementImageQuad,
  createAutomaticArtworkPlacement,
  hitTestPlacements,
  maybeExtendCurvedMappingOnCompletedTransform,
  movePlacementByImagePoints,
  reanchorArtworkPlacement,
  rotatePlacementFromImagePoint,
  samplePlacementTextureCoord,
  scalePlacementFromImagePoint,
} from './artworkPlacement';
import { sampleSurfaceMappingAtImagePoint } from '../geometry/localSurfaceParameterizer';
import { BrushEngine } from './brushEngine';
import { compositeLightingAwarePixel } from './colorSpace';
import { PlaneVisibility } from './planeVisibility';

function createDrawnLogo(width = 64, height = 64) {
  let art = ArtworkModel.createBlank({ name: 'Sticker', width, height });
  const engine = new BrushEngine();
  const stroke = engine.beginStroke(
    art.id,
    { minU: 0, maxU: width, minV: 0, maxV: height },
    { diameter: 20, color: '#22cc88', opacity: 1, hardness: 1 },
    { u: width * 0.5, v: height * 0.5 },
    'paint',
  );
  if (stroke) {
    art = ArtworkModel.commitDrawnStroke(art, engine.commitStroke() ?? stroke);
  }
  return art;
}

describe('ArtworkPlacement & automatic planar projection', () => {
  it('places artwork automatically on oblique planes, falls back cleanly on invalid/missing geometry, and reuses one source asset', () => {
    const art = createDrawnLogo(120, 60);
    const tiltRad = Math.PI / 4; // 45 deg wall
    const wall = createSyntheticPlanarScene({
      width: 64,
      height: 64,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [Math.sin(tiltRad), 0, -Math.cos(tiltRad)],
      invalidRegions: [{ minCol: 0, maxCol: 10, minRow: 0, maxRow: 10 }],
    });
    const imageSize = { width: 640, height: 640 };

    // Valid oblique wall drop -> automatic planar placement preserving 2:1 artwork aspect ratio.
    const wallPlace = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 320, y: 320 },
      imageSize,
      geometry: wall.geometry,
    });
    expect(wallPlace.mapping.type).toBe('planar');
    if (wallPlace.mapping.type === 'planar') {
      expect(wallPlace.mapping.isApproximatelyPlanar).toBe(false);
      expect(wallPlace.mapping.baseWidthWorld / wallPlace.mapping.baseHeightWorld).toBeCloseTo(
        2,
        4,
      );

      // With normal [+sin45, 0, -cos45], the plane equation is z = x + 3: left (-X) is closer
      // to the camera so the left projected edge appears taller than the right (+X) edge.
      const quad = computePlacementImageQuad(wallPlace, imageSize, wall.geometry);
      expect(quad).not.toBeNull();
      if (quad) {
        const leftEdgePxHeight = Math.hypot(
          quad.corners[3].x - quad.corners[0].x,
          quad.corners[3].y - quad.corners[0].y,
        );
        const rightEdgePxHeight = Math.hypot(
          quad.corners[2].x - quad.corners[1].x,
          quad.corners[2].y - quad.corners[1].y,
        );
        expect(leftEdgePxHeight).toBeGreaterThan(rightEdgePxHeight * 1.1);
      }
    }

    // Invalid hole drop and missing geometry both trigger deterministic 2D fallback placement.
    const holePlace = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 50, y: 50 },
      imageSize,
      geometry: wall.geometry,
    });
    expect(holePlace.mapping.type).toBe('image-2d');

    const noGeomPlace = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 320, y: 320 },
      imageSize,
      geometry: null,
    });
    expect(noGeomPlace.mapping.type).toBe('image-2d');

    // Both placements share one source asset ID cleanly without mutating the source artwork.
    expect(wallPlace.artworkId).toBe(art.id);
    expect(holePlace.artworkId).toBe(art.id);
    expect(art.strokes.length).toBe(1);
  });

  it('wraps artwork onto curved cylindrical and spherical surfaces instead of staying flat planar', () => {
    const art = createDrawnLogo(120, 80);
    const cyl = createSyntheticCylinderScene({
      width: 96,
      height: 96,
      axisCenter: [0, 0, 2.2],
      radius: 0.85,
    });
    const sph = createSyntheticSphereScene({
      width: 90,
      height: 90,
      center: [0, 0, 2.1],
      radius: 0.9,
    });
    const imageSize = { width: 960, height: 960 };

    const cylPlace = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 480, y: 480 },
      imageSize,
      geometry: cyl.geometry,
      scale: 1.5,
    });
    expect(cylPlace.mapping.type).toBe('curved');
    if (cylPlace.mapping.type === 'curved') {
      const { surfaceMapping, baseWidthWorld, baseHeightWorld } = cylPlace.mapping;
      expect(baseWidthWorld / baseHeightWorld).toBeCloseTo(120 / 80, 4);

      const quad = computePlacementImageQuad(cylPlace, imageSize, cyl.geometry);
      expect(quad).not.toBeNull();
      expect(quad?.outlinePolygon?.length).toBeGreaterThanOrEqual(16);

      // Verify right-side-up / left-to-right orientation without mirroring on curved UV sampling.
      const centerUv = samplePlacementTextureCoord(
        cylPlace,
        { x: 480, y: 480 },
        imageSize,
        cyl.geometry,
      );
      const rightUv = samplePlacementTextureCoord(
        cylPlace,
        { x: 560, y: 480 },
        imageSize,
        cyl.geometry,
      );
      const downUv = samplePlacementTextureCoord(
        cylPlace,
        { x: 480, y: 540 },
        imageSize,
        cyl.geometry,
      );
      expect(centerUv?.x).toBeCloseTo(0.5, 2);
      expect(centerUv?.y).toBeCloseTo(0.5, 2);
      expect(rightUv).not.toBeNull();
      expect(downUv).not.toBeNull();
      if (centerUv && rightUv && downUv) {
        expect(rightUv.x).toBeGreaterThan(centerUv.x);
        expect(downUv.y).toBeGreaterThan(centerUv.y);
      }

      // Curved outline must span multiple continuous surface triangles cleanly.
      expect(surfaceMapping.triangleIndices.length / 3).toBeGreaterThan(200);
    }

    const sphPlace = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 450, y: 450 },
      imageSize: { width: 900, height: 900 },
      geometry: sph.geometry,
    });
    expect(sphPlace.mapping.type).toBe('curved');
  });

  it('transforms move/scale/rotate within a stable anchor without switching surfaces and respects foreground occlusion & lighting', () => {
    const art = createDrawnLogo(64, 64);
    const wallWithPerson = createSyntheticPlanarScene({
      width: 64,
      height: 64,
      planeOrigin: [0, 0, 3.5],
      planeNormal: [0, 0, -1],
      occluders: [{ minCol: 28, maxCol: 36, minRow: 10, maxRow: 63, depth: 1.4 }],
    });
    const imageSize = { width: 640, height: 640 };
    const intr = parseNormalizedIntrinsics(wallWithPerson.geometry.metadata.intrinsics);

    const initial = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 180, y: 320 },
      imageSize,
      geometry: wallWithPerson.geometry,
    });
    expect(initial.mapping.type).toBe('planar');
    if (initial.mapping.type !== 'planar') return;
    const lockedNormal = initial.mapping.frame.normal;

    // Move placement directly across foreground occluder onto x=320: anchor stays locked on the wall!
    const moved = movePlacementByImagePoints(
      initial,
      { x: 180, y: 320 },
      { x: 320, y: 320 },
      imageSize,
      wallWithPerson.geometry,
    );
    expect(moved.mapping.type).toBe('planar');
    if (moved.mapping.type === 'planar') {
      expect(moved.mapping.frame.normal).toEqual(lockedNormal);
      expect(moved.mapping.frame.origin).toEqual(initial.mapping.frame.origin);
      expect(moved.mapping.centerU).toBeGreaterThan(initial.mapping.centerU);
    }

    // Scale and rotate in plane coordinates.
    const quadBefore = computePlacementImageQuad(moved, imageSize, wallWithPerson.geometry);
    if (!quadBefore) throw new Error('Expected placement quad');
    const scaled = scalePlacementFromImagePoint(
      moved,
      { x: quadBefore.corners[1].x + 30, y: quadBefore.corners[1].y - 30 },
      imageSize,
      wallWithPerson.geometry,
    );
    expect(scaled.scale).toBeGreaterThan(moved.scale);

    const rotated = rotatePlacementFromImagePoint(
      scaled,
      { x: quadBefore.center.x + 120, y: quadBefore.center.y },
      imageSize,
      wallWithPerson.geometry,
    );
    expect(rotated.rotationRad).toBeCloseTo(Math.PI * 0.5, 1);

    // Explicit re-anchor at x=320 (occluder pixel) now adopts the foreground depth cleanly.
    const reanchored = reanchorArtworkPlacement({
      placement: rotated,
      artwork: art,
      imagePoint: { x: 320, y: 320 },
      imageSize,
      geometry: wallWithPerson.geometry,
    });
    expect(reanchored.mapping.type).toBe('planar');
    if (reanchored.mapping.type === 'planar') {
      expect(reanchored.mapping.anchorPoint3D[2]).toBeCloseTo(1.4, 2);
    }

    // Check UV mapping & PlaneVisibility foreground occlusion over the occluder pixel on 'moved'.
    if (moved.mapping.type === 'planar') {
      const uvCenter = samplePlacementTextureCoord(
        moved,
        { x: 320, y: 320 },
        imageSize,
        wallWithPerson.geometry,
      );
      expect(uvCenter).not.toBeNull();
      expect(uvCenter?.x).toBeCloseTo(0.5, 2);
      expect(uvCenter?.y).toBeCloseTo(0.5, 2);

      // Finite footprint rejects coordinates far outside artwork bounds.
      expect(
        samplePlacementTextureCoord(moved, { x: 10, y: 10 }, imageSize, wallWithPerson.geometry),
      ).toBeNull();

      // Occluder pixel at col 32 has depth 1.4 < wall 3.5 -> PlaneVisibility marks occluded!
      const visOcc = PlaneVisibility.evaluatePixel(
        { x: 0.5, y: 0.5 },
        intr,
        moved.mapping.frame,
        { minU: -10, maxU: 10, minV: -10, maxV: 10 },
        true,
        1.4,
        [0, 0, -1],
      );
      expect(visOcc.visible).toBe(false);
    }

    // Hit testing respects placement draw order (last placement wins overlapping hit).
    const secondOnSameWall = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 320, y: 320 },
      imageSize,
      geometry: wallWithPerson.geometry,
    });
    const hit = hitTestPlacements(
      [moved, secondOnSameWall],
      { x: 320, y: 320 },
      imageSize,
      wallWithPerson.geometry,
    );
    expect(hit?.id).toBe(secondOnSameWall.id);

    // Check lighting-aware compositing preserves shadow gradients on artwork RGB pixels
    // and preserves background untouched where artwork alpha == 0.
    const artRgba = rasterizeArtworkRgba(art);
    const centerA = (artRgba[(32 * 64 + 32) * 4 + 3] ?? 0) / 255;
    const cornerA = (artRgba[3] ?? 0) / 255;
    expect(centerA).toBe(1);
    expect(cornerA).toBe(0);

    const litOut = compositeLightingAwarePixel({
      origLinear: [0.4, 0.4, 0.4],
      paintLinear: [0.1, 0.7, 0.4],
      alpha: centerA,
      albedo: [0.5, 0.5, 0.5],
      shadingRaw: [0.4, 0.4, 0.4],
      shadingScale: 2.0,
      mode: 'shading',
    });
    const shadowOut = compositeLightingAwarePixel({
      origLinear: [0.1, 0.1, 0.1],
      paintLinear: [0.1, 0.7, 0.4],
      alpha: centerA,
      albedo: [0.5, 0.5, 0.5],
      shadingRaw: [0.1, 0.1, 0.1],
      shadingScale: 2.0,
      mode: 'shading',
    });
    expect(litOut[1]).toBeGreaterThan(shadowOut[1]);
  });

  it('supports curved move, rotate, scale, completed boundary extension, occlusion, and informative 2D fallback', () => {
    const art = createDrawnLogo(100, 100);
    const cyl = createSyntheticCylinderScene({
      width: 96,
      height: 96,
      axisCenter: [0, 0, 2.3],
      radius: 0.9,
      occluders: [{ minCol: 54, maxCol: 64, minRow: 20, maxRow: 76, depth: 0.85 }],
      backgrounds: [{ minCol: 0, maxCol: 8, minRow: 0, maxRow: 95, depth: 5.5 }],
    });
    const imageSize = { width: 960, height: 960 };

    const curved = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 420, y: 480 },
      imageSize,
      geometry: cyl.geometry,
    });
    expect(curved.mapping.type).toBe('curved');
    if (curved.mapping.type !== 'curved') return;
    const initialMappingId = curved.mapping.surfaceMapping.id;

    // Interactive drag along curved surface keeps same SurfaceMapping ID during active drag!
    const moved = movePlacementByImagePoints(
      curved,
      { x: 420, y: 480 },
      { x: 490, y: 460 },
      imageSize,
      cyl.geometry,
    );
    expect(moved.mapping.type).toBe('curved');
    if (moved.mapping.type === 'curved') {
      expect(moved.mapping.surfaceMapping.id).toBe(initialMappingId);
      expect(moved.mapping.centerU).toBeGreaterThan(curved.mapping.centerU);
      expect(moved.mapping.centerV).toBeLessThan(curved.mapping.centerV);
    }

    // Scale and rotate in local intrinsic surface coordinates cleanly.
    const scaled = scalePlacementFromImagePoint(moved, { x: 640, y: 310 }, imageSize, cyl.geometry);
    expect(scaled.scale).toBeGreaterThan(moved.scale);

    const rotated = rotatePlacementFromImagePoint(
      scaled,
      { x: 620, y: 460 },
      imageSize,
      cyl.geometry,
    );
    expect(Math.abs(rotated.rotationRad)).toBeGreaterThan(0.2);

    // Completed scale/move extending toward patch boundary extends the parameterized patch safely.
    const pushedFar =
      rotated.mapping.type === 'curved'
        ? {
            ...rotated,
            scale: rotated.scale * 2.2,
            mapping: {
              ...rotated.mapping,
              centerU: rotated.mapping.surfaceMapping.maxU * 0.65,
            },
          }
        : rotated;
    const extended = maybeExtendCurvedMappingOnCompletedTransform(
      pushedFar,
      imageSize,
      cyl.geometry,
    );
    expect(extended.mapping.type).toBe('curved');
    if (extended.mapping.type === 'curved') {
      expect(extended.mapping.surfaceMapping.id).not.toBe(initialMappingId);
      expect(extended.mapping.centerU).toBeCloseTo(0, 4);
    }

    // Never leaks onto foreground occluder or background slab!
    expect(
      sampleSurfaceMappingAtImagePoint(
        curved.mapping.surfaceMapping,
        { x: 590, y: 480 },
        imageSize,
      ),
    ).toBeNull();
    expect(
      sampleSurfaceMappingAtImagePoint(curved.mapping.surfaceMapping, { x: 40, y: 480 }, imageSize),
    ).toBeNull();

    // Hit testing matches curved wrapped bounds accurately.
    const hit = hitTestPlacements([curved], { x: 420, y: 480 }, imageSize, cyl.geometry);
    expect(hit?.id).toBe(curved.id);

    // Unsuitable/missing geometry returns image-2d with clear non-blocking fallbackReason.
    const fallback = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 20, y: 20 },
      imageSize,
      geometry: null,
    });
    expect(fallback.mapping.type).toBe('image-2d');
    if (fallback.mapping.type === 'image-2d') {
      expect(fallback.mapping.fallbackReason).toBeTruthy();
    }
  });

  it('wraps wide banners across full high-res curved cylinders and transitions smoothly across wall-to-floor corners', () => {
    // 1. Wide banner on a 200x200 cylinder (40k pixels -> triggers adaptive stride & full limb coverage).
    const wideBanner = createDrawnLogo(900, 300);
    const largeCyl = createSyntheticCylinderScene({
      width: 200,
      height: 200,
      intrinsics: makeSyntheticIntrinsics(2.4, 2.4, 0.5, 0.5),
      axisCenter: [0, 0, 8.8],
      radius: 0.72,
    });
    const canImgSize = { width: 1000, height: 1000 };
    const placedBanner = createAutomaticArtworkPlacement({
      artwork: wideBanner,
      imagePoint: { x: 500, y: 500 },
      imageSize: canImgSize,
      geometry: largeCyl.geometry,
      scale: 1.6,
    });
    expect(placedBanner.mapping.type).toBe('curved');
    if (placedBanner.mapping.type === 'curved') {
      const sm = placedBanner.mapping.surfaceMapping;
      // Adaptive stride ensures wide patch covers limbs cleanly rather than truncating near center.
      expect(sm.minU).toBeLessThan(-0.6);
      expect(sm.maxU).toBeGreaterThan(0.6);
      const extended = maybeExtendCurvedMappingOnCompletedTransform(
        placedBanner,
        canImgSize,
        largeCyl.geometry,
      );
      expect(extended.mapping.type).toBe('curved');
      expect(
        samplePlacementTextureCoord(extended, { x: 350, y: 500 }, canImgSize, largeCyl.geometry),
      ).not.toBeNull();
      expect(
        samplePlacementTextureCoord(extended, { x: 650, y: 500 }, canImgSize, largeCyl.geometry),
      ).not.toBeNull();
    }

    // 2. Wall-to-floor scene: vertical wall on upper rows (0..55), horizontal floor on lower rows (56..79).
    const w = 80;
    const h = 80;
    const total = w * h;
    const intrRaw = makeSyntheticIntrinsics(1.0, 1.0, 0.5, 0.5);
    const depth = new Float32Array(total);
    const points = new Float32Array(total * 3);
    const mask = new Uint8Array(total).fill(1);
    const normals = new Float32Array(total * 3);
    const zWall = 4.0;
    const yFloor = zWall * ((56 + 0.5) / h - 0.5);

    for (let row = 0; row < h; row++) {
      const ny = (row + 0.5) / h - 0.5;
      const isFloor = row >= 56;
      const zVal = isFloor ? Math.max(0.8, yFloor / Math.max(0.05, ny)) : zWall;
      for (let col = 0; col < w; col++) {
        const nx = (col + 0.5) / w - 0.5;
        const idx = row * w + col;
        depth[idx] = zVal;
        points[idx * 3] = nx * zVal;
        points[idx * 3 + 1] = ny * zVal;
        points[idx * 3 + 2] = zVal;
        normals[idx * 3] = 0;
        normals[idx * 3 + 1] = isFloor ? -1 : 0;
        normals[idx * 3 + 2] = isFloor ? 0 : -1;
      }
    }

    const wfGeom: GeometryInput = {
      metadata: { inference_width: w, inference_height: h, intrinsics: intrRaw },
      depth,
      points,
      mask,
      normals,
    };
    const wfImgSize = { width: 800, height: 800 };
    let wfPlace = createAutomaticArtworkPlacement({
      artwork: createDrawnLogo(120, 60),
      imagePoint: { x: 400, y: 240 },
      imageSize: wfImgSize,
      geometry: wfGeom,
    });
    expect(wfPlace.mapping.type).toBe('planar');

    // Drag down onto the floor: must transition onto floor surface instead of disappearing behind it.
    wfPlace = movePlacementByImagePoints(
      wfPlace,
      { x: 400, y: 240 },
      { x: 400, y: 690 },
      wfImgSize,
      wfGeom,
    );
    expect(
      samplePlacementTextureCoord(wfPlace, { x: 400, y: 690 }, wfImgSize, wfGeom),
    ).not.toBeNull();
    if (wfPlace.mapping.type === 'planar') {
      expect(wfPlace.mapping.frame.normal[1]).toBeLessThan(-0.7);
    }

    // Drag back up onto the wall: must transition back cleanly instead of freezing at floor horizon.
    wfPlace = movePlacementByImagePoints(
      wfPlace,
      { x: 400, y: 690 },
      { x: 400, y: 260 },
      wfImgSize,
      wfGeom,
    );
    expect(
      samplePlacementTextureCoord(wfPlace, { x: 400, y: 260 }, wfImgSize, wfGeom),
    ).not.toBeNull();
  });
});

it('does not select occluded, transparent or zero-opacity artwork', () => {
  const scene = createSyntheticPlanarScene({
    width: 64,
    height: 64,
    planeOrigin: [0, 0, 3],
    planeNormal: [0, 0, -1],
  });
  const imageSize = { width: 640, height: 640 };
  const artwork = createDrawnLogo();
  const placement = createAutomaticArtworkPlacement({
    artwork,
    imagePoint: { x: 320, y: 320 },
    imageSize,
    geometry: scene.geometry,
  });
  expect(
    hitTestPlacements([placement], { x: 320, y: 320 }, imageSize, scene.geometry, [artwork])?.id,
  ).toBe(placement.id);
  expect(
    hitTestPlacements(
      [{ ...placement, opacity: 0 }],
      { x: 320, y: 320 },
      imageSize,
      scene.geometry,
      [artwork],
    ),
  ).toBeNull();
  const transparent = ArtworkModel.createBlank();
  const empty = { ...placement, artworkId: transparent.id };
  expect(
    hitTestPlacements([empty], { x: 320, y: 320 }, imageSize, scene.geometry, [transparent]),
  ).toBeNull();
  const occluded = createSyntheticPlanarScene({
    width: 64,
    height: 64,
    planeOrigin: [0, 0, 3],
    planeNormal: [0, 0, -1],
    occluders: [{ minCol: 28, maxCol: 36, minRow: 28, maxRow: 36, depth: 1 }],
  });
  expect(
    hitTestPlacements([placement], { x: 320, y: 320 }, imageSize, occluded.geometry, [artwork]),
  ).toBeNull();
});
