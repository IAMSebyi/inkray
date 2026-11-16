import { describe, expect, it } from 'vitest';
import {
  buildSyntheticSurfaceSelection,
  createSyntheticPlanarScene,
} from '../geometry/testFixtures';
import type { BrushStroke } from './brushEngine';
import { PaintDocument } from './paintDocument';
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
});
