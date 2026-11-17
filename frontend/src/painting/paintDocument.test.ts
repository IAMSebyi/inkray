import { describe, expect, it } from 'vitest';
import {
  buildSyntheticSurfaceSelection,
  createSyntheticCylinderScene,
  createSyntheticPlanarScene,
} from '../geometry/testFixtures';
import { ArtworkModel } from './artwork';
import { createAutomaticArtworkPlacement } from './artworkPlacement';
import type { BrushStroke } from './brushEngine';
import { rebuildImageLayerRgba, type ImageSpaceStroke } from './imageSpacePaint';
import { PaintDocument } from './paintDocument';
import type { PlanarDecalSurface } from './planarDecal';
import {
  computeDecalTextureSpec,
  decalTexelToPlaneCoords,
  planeCoordsToDecalUv,
  scaleFootprint,
} from './planarDecal';
import { computeViewportQuadTransform, flipRgbaRowsVertically } from './planarPaintRenderer';

function createTestSelection() {
  const scene = createSyntheticPlanarScene({
    width: 24,
    height: 24,
    planeOrigin: [0, 0, 3.0],
    planeNormal: [0, 0, -1],
  });
  return buildSyntheticSurfaceSelection(scene, 12, 12);
}

describe('PaintDocument & PlanarDecal', () => {
  it('computes bounded decal specs and maps plane coordinates to texels consistently', () => {
    const base = { minU: -2, maxU: 2, minV: -1, maxV: 1 };
    const scaled = scaleFootprint(base, 1.5);
    expect(scaled.minU).toBeCloseTo(-3, 5);
    expect(scaled.maxU).toBeCloseTo(3, 5);
    expect(scaled.minV).toBeCloseTo(-1.5, 5);
    expect(scaled.maxV).toBeCloseTo(1.5, 5);

    const spec = computeDecalTextureSpec(base, 1024, 256);
    expect(spec.width).toBe(1024);
    expect(spec.height).toBe(512);

    const [uvX, uvY] = planeCoordsToDecalUv(0, 0, base);
    expect(uvX).toBeCloseTo(0.5, 6);
    expect(uvY).toBeCloseTo(0.5, 6);

    const [pu, pv] = decalTexelToPlaneCoords(511.5, 255.5, base, spec);
    expect(pu).toBeCloseTo(0, 2);
    expect(pv).toBeCloseTo(0, 2);
  });

  it('retains multiple surfaces, tracks priority reorder and deletion, and undoes/redoes atomically across history', () => {
    let doc = PaintDocument.createInitial();
    const sel1 = createTestSelection();
    doc = PaintDocument.selectSurface(doc, sel1);
    const surf1Id = doc.activeSurfaceId;
    if (!surf1Id) throw new Error('Expected first surface');

    doc = PaintDocument.setSurfaceExtent(doc, surf1Id, 4.0);
    expect(PaintDocument.getActiveSurface(doc)?.extentScale).toBe(2.5);

    const sel2 = createTestSelection();
    doc = PaintDocument.selectSurface(doc, sel2);
    const surf2Id = doc.activeSurfaceId;
    if (!surf2Id) throw new Error('Expected second surface');

    // Selecting surf2 retains surf1 cleanly without pruning.
    expect(doc.surfaces.length).toBe(2);
    expect(doc.surfaces[0]?.id).toBe(surf1Id);
    expect(doc.surfaces[1]?.id).toBe(surf2Id);

    // Reorder surf2 above surf1 as priority #1, then undo/redo priority change.
    doc = PaintDocument.reorderSurfaces(doc, 1, 0);
    expect(doc.surfaces[0]?.id).toBe(surf2Id);
    doc = PaintDocument.undo(doc);
    expect(doc.surfaces[0]?.id).toBe(surf1Id);
    doc = PaintDocument.redo(doc);
    expect(doc.surfaces[0]?.id).toBe(surf2Id);

    // Cross-surface stroke batch commits as one atomic undo step.
    const s1: BrushStroke = {
      id: 'stroke-a',
      surfaceId: surf1Id,
      brush: { diameter: 0.2, color: '#ff0000', opacity: 0.9, hardness: 0.8 },
      points: [{ u: 0, v: 0 }],
    };
    const s2: BrushStroke = {
      id: 'stroke-b',
      surfaceId: surf2Id,
      brush: { diameter: 0.2, color: '#ff0000', opacity: 0.9, hardness: 0.8 },
      points: [{ u: 0.1, v: 0 }],
    };
    doc = PaintDocument.commitStrokes(doc, [s1, s2]);
    expect(doc.strokes.length).toBe(2);

    // One undo removes the whole cross-surface drag gesture.
    doc = PaintDocument.undo(doc);
    expect(doc.strokes.length).toBe(0);
    expect(PaintDocument.canRedo(doc)).toBe(true);

    doc = PaintDocument.redo(doc);
    expect(PaintDocument.getStrokesForSurface(doc, surf1Id).length).toBe(1);
    expect(PaintDocument.getStrokesForSurface(doc, surf2Id).length).toBe(1);

    // Removing surf1 cleans up its strokes while remaining undoable.
    doc = PaintDocument.removeSurface(doc, surf1Id);
    expect(doc.surfaces.some((s) => s.id === surf1Id)).toBe(false);
    expect(PaintDocument.getStrokesForSurface(doc, surf1Id).length).toBe(0);
    doc = PaintDocument.undo(doc);
    expect(doc.surfaces.some((s) => s.id === surf1Id)).toBe(true);
    expect(PaintDocument.getStrokesForSurface(doc, surf1Id).length).toBe(1);
  });

  it('computes full-resolution PNG export vertical flip, unpremultiplication, and viewport NDC quad alignment accurately', () => {
    const bottomUp = new Uint8Array([
      10,
      20,
      30,
      255,
      64,
      0,
      0,
      128, // bottom row (GL row 0, right pixel premultiplied 50% red)
      200,
      100,
      50,
      255,
      210,
      110,
      60,
      255, // top row (GL row 1)
    ]);
    const topDown = flipRgbaRowsVertically(bottomUp, 2, 2, true);
    expect(Array.from(topDown.subarray(0, 4))).toEqual([200, 100, 50, 255]);
    expect(Array.from(topDown.subarray(8, 12))).toEqual([10, 20, 30, 255]);
    // Premultiplied (64, 0, 0, 128) unpremultiplies cleanly back to ~128 straight red.
    expect(topDown[12]).toBeGreaterThanOrEqual(127);
    expect(topDown[15]).toBe(128);

    const ndc = computeViewportQuadTransform(
      { width: 800, height: 600 },
      { width: 400, height: 300 },
      { zoom: 1, panX: 200, panY: 150 },
    );
    // Image fills [200..600] x [150..450] -> half width 0.5, half height 0.5 centered at (0, 0).
    expect(ndc[0]).toBeCloseTo(0.5, 6);
    expect(ndc[1]).toBeCloseTo(0.5, 6);
    expect(ndc[2]).toBeCloseTo(0, 6);
    expect(ndc[3]).toBeCloseTo(0, 6);
  });

  it('commits mixed planar, curved, and 2D strokes atomically, syncs brush pixelSize, and layers image paint + erase cleanly', () => {
    let doc = PaintDocument.createInitial();
    doc = PaintDocument.setBrushConfig(doc, { pixelSize: 72 });
    expect(doc.brushConfig.pixelSize).toBe(72);
    expect(doc.brushConfig.sizeMultiplier).toBeCloseTo(2.0, 4);

    // 1. 2D fallback stroke before any surface exists.
    const mask2d = new Uint8Array(100).fill(255);
    const stroke2d: ImageSpaceStroke = {
      id: 'img-2d-1',
      layer: 'fallback-2d',
      mode: 'paint',
      brush: { diameter: 20, color: '#ff0000', opacity: 1, hardness: 1 },
      layerWidth: 20,
      layerHeight: 20,
      bounds: { minX: 0, minY: 0, maxX: 9, maxY: 9 },
      alphaMask: mask2d,
    };
    doc = PaintDocument.commitAutoStroke(doc, {
      anchorMode: 'fallback-2d',
      imageStroke: stroke2d,
    });
    expect(doc.lastAnchorMode).toBe('fallback-2d');
    expect(doc.imageStrokes.length).toBe(1);

    // 2. Curved paint stroke + curved erase stroke.
    const curvedPaint: ImageSpaceStroke = {
      id: 'img-curved-1',
      layer: 'curved',
      mode: 'paint',
      brush: { diameter: 20, color: '#00ff88', opacity: 0.8, hardness: 1 },
      layerWidth: 20,
      layerHeight: 20,
      bounds: { minX: 0, minY: 0, maxX: 9, maxY: 9 },
      alphaMask: new Uint8Array(100).fill(255),
    };
    const curvedErase: ImageSpaceStroke = {
      id: 'img-curved-2',
      layer: 'curved',
      mode: 'erase',
      brush: { diameter: 10, color: '#00ff88', opacity: 1, hardness: 1 },
      layerWidth: 20,
      layerHeight: 20,
      bounds: { minX: 0, minY: 0, maxX: 3, maxY: 3 },
      alphaMask: new Uint8Array(16).fill(255),
    };
    doc = PaintDocument.commitAutoStroke(doc, { anchorMode: 'curved', imageStroke: curvedPaint });
    doc = PaintDocument.commitAutoStroke(doc, { anchorMode: 'curved', imageStroke: curvedErase });
    expect(doc.imageStrokes.length).toBe(3);

    const curvedRgba = rebuildImageLayerRgba(
      doc.imageStrokes.filter((s) => s.layer === 'curved'),
      { width: 20, height: 20 },
    );
    // Pixel (1, 1) was erased cleanly while (7, 7) retains curved paint!
    expect(curvedRgba[(1 * 20 + 1) * 4 + 3]).toBe(0);
    expect(curvedRgba[(7 * 20 + 7) * 4 + 3]).toBeGreaterThan(180);

    // 3. Auto planar surface + stroke committed together in one atomic step.
    const sel = createTestSelection();
    const autoSurface: PlanarDecalSurface = {
      id: 'auto-surf-1',
      label: 'Surface 1',
      colorIndex: 0,
      frame: sel.frame,
      baseFootprint: sel.suggestedFootprint,
      extentScale: 1,
      supportMask: sel.supportMask,
      inlierCount: sel.inlierCount,
      rmsError: sel.rmsError,
      referenceBrushDiameter: sel.referenceBrushDiameter,
    };
    doc = PaintDocument.commitAutoStroke(doc, {
      anchorMode: 'planar',
      newPlanarSurface: autoSurface,
      planarStroke: {
        id: 'planar-1',
        surfaceId: autoSurface.id,
        mode: 'paint',
        brush: { diameter: 0.2, color: '#3388ff', opacity: 0.9, hardness: 0.8 },
        points: [{ u: 0, v: 0 }],
      },
    });
    expect(doc.surfaces.length).toBe(1);
    expect(doc.strokes.length).toBe(1);
    expect(doc.lastAnchorMode).toBe('planar');

    // Undo cleanly reverts planar stroke + auto surface together, then curved erase.
    doc = PaintDocument.undo(doc);
    expect(doc.surfaces.length).toBe(0);
    expect(doc.strokes.length).toBe(0);
    expect(doc.lastAnchorMode).toBe('curved');

    doc = PaintDocument.redo(doc);
    expect(doc.surfaces.length).toBe(1);
    expect(doc.strokes.length).toBe(1);
    expect(doc.lastAnchorMode).toBe('planar');
  });

  it('keeps independent artwork assets reusable across placements and edits artwork without disturbing placement transforms or existing brush strokes', () => {
    const scene = createSyntheticPlanarScene({
      width: 32,
      height: 32,
      planeOrigin: [0, 0, 3.0],
      planeNormal: [0, 0, -1],
    });
    let doc = PaintDocument.createInitial();

    // Commit an existing fallback-2d brush stroke first.
    doc = PaintDocument.commitAutoStroke(doc, {
      anchorMode: 'fallback-2d',
      imageStroke: {
        id: 'existing-brush',
        layer: 'fallback-2d',
        mode: 'paint',
        brush: { diameter: 20, color: '#ff0000', opacity: 1, hardness: 1 },
        layerWidth: 64,
        layerHeight: 64,
        bounds: { minX: 0, minY: 0, maxX: 5, maxY: 5 },
        alphaMask: new Uint8Array(36).fill(255),
      },
    });

    // Create blank artwork, draw a stroke, and place it twice cleanly.
    let art = ArtworkModel.createBlank({ name: 'Logo', width: 64, height: 64 });
    art = ArtworkModel.commitDrawnStroke(art, {
      id: 'art-s1',
      surfaceId: art.id,
      mode: 'paint',
      brush: { diameter: 12, color: '#00aaff', opacity: 1, hardness: 1 },
      points: [{ u: 32, v: 32 }],
    });
    doc = PaintDocument.addArtwork(doc, art);

    const p1 = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 16, y: 16 },
      imageSize: { width: 320, height: 320 },
      geometry: scene.geometry,
      label: 'Logo #1',
    });
    const p2 = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 240, y: 240 },
      imageSize: { width: 320, height: 320 },
      geometry: scene.geometry,
      label: 'Logo #2',
    });
    doc = PaintDocument.addPlacement(doc, p1);
    doc = PaintDocument.addPlacement(doc, p2);
    expect(doc.artworkPlacements.length).toBe(2);
    expect(doc.activePlacementId).toBe(p2.id);

    // Transform placement #2 and verify undo/redo restores it without disturbing brush strokes.
    doc = PaintDocument.updatePlacement(doc, { ...p2, scale: 2.0, rotationRad: 0.7 }, true);
    expect(PaintDocument.getActivePlacement(doc)?.scale).toBe(2.0);
    doc = PaintDocument.undo(doc);
    expect(PaintDocument.getActivePlacement(doc)?.scale).toBe(p2.scale);
    expect(doc.imageStrokes.length).toBe(1);
    doc = PaintDocument.redo(doc);
    expect(PaintDocument.getActivePlacement(doc)?.scale).toBe(2.0);

    // Editing the source artwork increments revision while keeping both placements' transforms intact.
    const editedArt = ArtworkModel.commitDrawnStroke(art, {
      id: 'art-s2',
      surfaceId: art.id,
      mode: 'paint',
      brush: { diameter: 10, color: '#ffffff', opacity: 1, hardness: 1 },
      points: [{ u: 20, v: 20 }],
    });
    doc = PaintDocument.updateArtwork(doc, editedArt);
    expect(PaintDocument.getArtworkById(doc, art.id)?.revision).toBe(art.revision + 1);
    expect(doc.artworkPlacements[0]?.artworkId).toBe(art.id);
    expect(doc.artworkPlacements[1]?.scale).toBe(2.0);
    expect(doc.imageStrokes.length).toBe(1);

    // Removing placement #2 leaves placement #1 and brush strokes clean, and undoing restores placement #2.
    doc = PaintDocument.removePlacement(doc, p2.id);
    expect(doc.artworkPlacements.length).toBe(1);
    doc = PaintDocument.undo(doc);
    expect(doc.artworkPlacements.length).toBe(2);

    // Curved placement on a cylinder shares undo/redo and source artwork reactivity cleanly.
    const cylScene = createSyntheticCylinderScene({
      width: 64,
      height: 64,
      axisCenter: [0, 0, 2.2],
      radius: 0.85,
    });
    const curvedPlacement = createAutomaticArtworkPlacement({
      artwork: editedArt,
      imagePoint: { x: 320, y: 320 },
      imageSize: { width: 640, height: 640 },
      geometry: cylScene.geometry,
      label: 'Logo Curved',
    });
    expect(curvedPlacement.mapping.type).toBe('curved');
    doc = PaintDocument.addPlacement(doc, curvedPlacement);
    expect(doc.artworkPlacements.length).toBe(3);
    doc = PaintDocument.undo(doc);
    expect(doc.artworkPlacements.length).toBe(2);
    doc = PaintDocument.redo(doc);
    expect(doc.artworkPlacements.length).toBe(3);
    expect(doc.artworkPlacements[2]?.mapping.type).toBe('curved');
  });

  it('tracks separate brush objects and artwork placements in unified priority order with visibility, deletion, and undo/redo', () => {
    const sel = createTestSelection();
    const surface: PlanarDecalSurface = {
      id: 'shared-wall',
      label: 'Wall',
      colorIndex: 0,
      frame: sel.frame,
      baseFootprint: sel.suggestedFootprint,
      extentScale: 1,
      supportMask: sel.supportMask,
      inlierCount: sel.inlierCount,
      rmsError: sel.rmsError,
      referenceBrushDiameter: sel.referenceBrushDiameter,
    };
    const art = ArtworkModel.createBlank({ name: 'Banner', width: 64, height: 64 });
    let doc = PaintDocument.createInitial();
    doc = PaintDocument.addArtwork(doc, art);

    // 1. First stroke automatically creates Brush 1 at sceneOrder[0].
    doc = PaintDocument.commitAutoStroke(doc, {
      anchorMode: 'planar',
      newPlanarSurface: surface,
      planarStroke: {
        id: 'b1-s1',
        surfaceId: surface.id,
        mode: 'paint',
        brush: { diameter: 0.2, color: '#ff0000', opacity: 1, hardness: 0.9 },
        points: [{ u: 0, v: 0 }],
      },
    });
    expect(doc.brushObjects.length).toBe(1);
    const brush1Id = doc.brushObjects[0]?.id;
    if (!brush1Id) throw new Error('Expected auto-created Brush 1');
    expect(doc.activeBrushObjectId).toBe(brush1Id);
    expect(doc.strokes[0]?.brushObjectId).toBe(brush1Id);
    expect(doc.sceneOrder).toEqual([{ kind: 'brush', id: brush1Id }]);

    // 2. Place two artworks onto the photo; newest sits at sceneOrder[0].
    const p1 = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 10, y: 10 },
      imageSize: { width: 24, height: 24 },
      geometry: null,
      label: 'Banner #1',
    });
    const p2 = createAutomaticArtworkPlacement({
      artwork: art,
      imagePoint: { x: 14, y: 14 },
      imageSize: { width: 24, height: 24 },
      geometry: null,
      label: 'Banner #2',
    });
    doc = PaintDocument.addPlacement(doc, p1);
    doc = PaintDocument.addPlacement(doc, p2);

    // 3. User creates a new Brush Object (#2) and paints onto the SAME wall surface on top.
    doc = PaintDocument.createBrushObject(doc);
    expect(doc.brushObjects.length).toBe(2);
    const brush2Id = doc.activeBrushObjectId;
    if (!brush2Id || brush2Id === brush1Id) throw new Error('Expected new Brush 2');

    doc = PaintDocument.commitAutoStroke(doc, {
      anchorMode: 'planar',
      planarStroke: {
        id: 'b2-s1',
        surfaceId: surface.id,
        mode: 'paint',
        brush: { diameter: 0.2, color: '#00aaff', opacity: 1, hardness: 0.9 },
        points: [{ u: 0.1, v: 0 }],
      },
    });

    // Display priority stack top-to-bottom: [Brush 2, p2, p1, Brush 1].
    expect(doc.sceneOrder).toEqual([
      { kind: 'brush', id: brush2Id },
      { kind: 'artwork', id: p2.id },
      { kind: 'artwork', id: p1.id },
      { kind: 'brush', id: brush1Id },
    ]);
    expect(PaintDocument.getStrokesForSurface(doc, surface.id, brush1Id).length).toBe(1);
    expect(PaintDocument.getStrokesForSurface(doc, surface.id, brush2Id).length).toBe(1);

    // 4. Drag Brush 2 behind p2 (index 0 -> index 1) and reorder p1 above p2 (index 2 -> index 0).
    doc = PaintDocument.reorderSceneObject(doc, 0, 1);
    expect(doc.sceneOrder).toEqual([
      { kind: 'artwork', id: p2.id },
      { kind: 'brush', id: brush2Id },
      { kind: 'artwork', id: p1.id },
      { kind: 'brush', id: brush1Id },
    ]);
    doc = PaintDocument.reorderSceneObject(doc, 2, 0);
    expect(PaintDocument.getOrderedPlacementsBottomToTop(doc).map((p) => p.id)).toEqual([
      p2.id,
      p1.id,
    ]);

    // Undo restores placement hit-test order, redo puts p1 back on top.
    doc = PaintDocument.undo(doc);
    expect(PaintDocument.getOrderedPlacementsBottomToTop(doc).map((p) => p.id)).toEqual([
      p1.id,
      p2.id,
    ]);
    doc = PaintDocument.redo(doc);
    expect(PaintDocument.getOrderedPlacementsBottomToTop(doc).map((p) => p.id)).toEqual([
      p2.id,
      p1.id,
    ]);

    // 5. Toggle visibility and remove Brush 2 cleanly without affecting Brush 1's stroke.
    doc = PaintDocument.toggleBrushObjectVisibility(doc, brush1Id);
    doc = PaintDocument.togglePlacementVisibility(doc, p1.id);
    expect(doc.brushObjects.find((b) => b.id === brush1Id)?.visible).toBe(false);
    expect(doc.artworkPlacements.find((p) => p.id === p1.id)?.visible).toBe(false);

    doc = PaintDocument.removeBrushObject(doc, brush2Id);
    expect(doc.brushObjects.some((b) => b.id === brush2Id)).toBe(false);
    expect(PaintDocument.getStrokesForSurface(doc, surface.id, brush1Id).length).toBe(1);

    doc = PaintDocument.undo(doc);
    expect(PaintDocument.getStrokesForSurface(doc, surface.id, brush2Id).length).toBe(1);
  });
});

it('restores deleted artwork and placements through undo and redo', () => {
  const artwork = ArtworkModel.createBlank();
  let doc = PaintDocument.addArtwork(PaintDocument.createInitial(), artwork);
  const placement = createAutomaticArtworkPlacement({
    artwork,
    imagePoint: { x: 50, y: 50 },
    imageSize: { width: 100, height: 100 },
    geometry: null,
  });
  doc = PaintDocument.addPlacement(doc, placement);
  doc = PaintDocument.removeArtwork(doc, artwork.id);
  expect(doc.artworks).toHaveLength(0);
  doc = PaintDocument.undo(doc);
  expect(doc.artworks).toEqual([artwork]);
  expect(doc.artworkPlacements).toEqual([placement]);
  doc = PaintDocument.redo(doc);
  expect(doc.artworks).toHaveLength(0);
  expect(doc.artworkPlacements).toHaveLength(0);
});

it('can undo deletion of an artwork without placements', () => {
  const artwork = ArtworkModel.createBlank();
  const doc = PaintDocument.removeArtwork(
    PaintDocument.addArtwork(PaintDocument.createInitial(), artwork),
    artwork.id,
  );
  expect(PaintDocument.undo(doc).artworks).toEqual([artwork]);
});

it('retains source artwork edits when undoing a photo operation', () => {
  const artwork = ArtworkModel.createBlank();
  let doc = PaintDocument.addArtwork(PaintDocument.createInitial(), artwork);
  doc = PaintDocument.createBrushObject(doc);
  const resized = ArtworkModel.resizeDrawn(artwork, 64, 32);
  doc = PaintDocument.updateArtwork(doc, resized);
  doc = PaintDocument.undo(doc);
  expect(doc.artworks).toEqual([resized]);
});
