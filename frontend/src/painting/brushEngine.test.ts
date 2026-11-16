import { describe, expect, it } from 'vitest';
import type { PlaneFootprint } from '../geometry/planeFrame';
import {
  BrushEngine,
  type BrushSettings,
  type BrushStroke,
  compositeStrokeMaskOverRgba,
  rasterizeStrokeToMask,
  rebuildSurfaceDecalRgba,
} from './brushEngine';
import type { DecalTextureSpec } from './planarDecal';

const FOOTPRINT: PlaneFootprint = {
  minU: -1,
  maxU: 1,
  minV: -1,
  maxV: 1,
};

const SPEC: DecalTextureSpec = {
  width: 200,
  height: 200,
};

describe('BrushEngine & stroke rasterization', () => {
  it('rasterizes single-click dots with smooth hardness falloff centered at click point', () => {
    const brush: BrushSettings = {
      diameter: 0.4,
      color: '#ff0000',
      opacity: 0.8,
      hardness: 0.5,
    };
    const stroke: BrushStroke = {
      id: 's1',
      surfaceId: 'surf1',
      brush,
      points: [{ u: 0, v: 0 }],
    };

    const mask = rasterizeStrokeToMask(stroke, FOOTPRINT, SPEC);
    const rgba = new Uint8ClampedArray(SPEC.width * SPEC.height * 4);
    compositeStrokeMaskOverRgba(rgba, mask, brush, SPEC.width);

    // Center texel (~99, 99) has distance ~0.005 < core radius (0.2 * 0.5 = 0.1).
    const centerIdx = 100 * SPEC.width + 100;
    const expectedAlpha = Math.round(0.8 * 255);
    expect(mask[centerIdx]).toBe(255);
    expect(rgba[centerIdx * 4]).toBe(255);
    expect(rgba[centerIdx * 4 + 1]).toBe(0);
    expect(rgba[centerIdx * 4 + 2]).toBe(0);
    expect(rgba[centerIdx * 4 + 3]).toBeCloseTo(expectedAlpha, -1);

    // Far corner (0, 0) lies well outside brush radius (0.2).
    expect(rgba[3]).toBe(0);

    // Edge texel inside falloff ring (~0.15 distance) has intermediate non-zero alpha.
    // u = +0.15 -> col = floor(((1.15 / 2) * 200)) = 115, row = 100.
    const edgeIdx = 100 * SPEC.width + 115;
    const edgeAlpha = rgba[edgeIdx * 4 + 3] ?? 0;
    expect(edgeAlpha).toBeGreaterThan(0);
    expect(edgeAlpha).toBeLessThan(expectedAlpha);
  });

  it('prevents within-stroke opacity accumulation while allowing separate overlapping strokes to stack', () => {
    const brush: BrushSettings = {
      diameter: 0.5,
      color: '#3366ff',
      opacity: 0.4,
      hardness: 1.0,
    };

    // One continuous stroke passing back and forth over the same region multiple times.
    const denseStroke: BrushStroke = {
      id: 'dense',
      surfaceId: 'surf1',
      brush,
      points: [
        { u: -0.2, v: 0 },
        { u: 0, v: 0 },
        { u: 0.2, v: 0 },
        { u: 0, v: 0 },
        { u: -0.1, v: 0 },
        { u: 0.1, v: 0 },
      ],
    };

    const singleStrokeRgba = rebuildSurfaceDecalRgba([denseStroke], FOOTPRINT, SPEC);
    const expectedSingleAlpha = Math.round(0.4 * 255);
    const centerBase = (100 * SPEC.width + 100) * 4;
    expect(singleStrokeRgba[centerBase + 3]).toBe(expectedSingleAlpha);

    // Two distinct strokes overlapping at (0, 0) must composite cleanly on top of each other.
    const strokeA: BrushStroke = {
      id: 'a',
      surfaceId: 'surf1',
      brush,
      points: [{ u: 0, v: 0 }],
    };
    const strokeB: BrushStroke = {
      id: 'b',
      surfaceId: 'surf1',
      brush,
      points: [{ u: 0, v: 0 }],
    };

    const twoStrokeRgba = rebuildSurfaceDecalRgba([strokeA, strokeB], FOOTPRINT, SPEC);
    const expectedTwoAlpha = Math.round((0.4 + 0.4 * (1 - 0.4)) * 255);
    expect(Math.abs((twoStrokeRgba[centerBase + 3] ?? 0) - expectedTwoAlpha)).toBeLessThanOrEqual(
      2,
    );
  });

  it('tracks live stroke sessions, rejects outside starts, and interpolates smooth drag samples', () => {
    const engine = new BrushEngine();
    const brush: BrushSettings = {
      diameter: 0.2,
      color: '#00cc88',
      opacity: 1,
      hardness: 0.8,
    };

    expect(engine.beginStroke('surf1', FOOTPRINT, brush, { u: 1.5, v: 0 })).toBeNull();
    expect(engine.isStrokeActive()).toBe(false);

    const started = engine.beginStroke('surf1', FOOTPRINT, brush, { u: -0.5, v: 0 });
    if (!started) {
      throw new Error('Expected stroke inside footprint to start');
    }
    expect(engine.isStrokeActive()).toBe(true);

    // Long drag should produce densely interpolated points along the segment.
    const extended = engine.extendStroke({ u: 0.5, v: 0 });
    if (!extended) {
      throw new Error('Expected stroke to extend');
    }
    expect(extended.points.length).toBeGreaterThan(10);

    const committed = engine.commitStroke();
    if (!committed) {
      throw new Error('Expected committed stroke');
    }
    expect(committed.points[committed.points.length - 1]?.u).toBeCloseTo(0.5, 4);
    expect(engine.isStrokeActive()).toBe(false);

    // Verify gapless rasterization across a horizontal stroke strip.
    const rgba = rebuildSurfaceDecalRgba([committed], FOOTPRINT, SPEC);
    for (let col = 55; col <= 145; col++) {
      const a = rgba[(100 * SPEC.width + col) * 4 + 3] ?? 0;
      expect(a).toBe(255);
    }
  });

  it('erases paint completely inside the eraser stroke regardless of brush opacity/hardness and batches cross-surface strokes', () => {
    const paintStroke: BrushStroke = {
      id: 'p1',
      surfaceId: 'surf1',
      mode: 'paint',
      brush: { diameter: 0.6, color: '#00aaee', opacity: 1.0, hardness: 1.0 },
      points: [{ u: 0, v: 0 }],
    };
    const eraseStroke: BrushStroke = {
      id: 'e1',
      surfaceId: 'surf1',
      mode: 'erase',
      brush: { diameter: 0.3, color: '#000000', opacity: 0.25, hardness: 0.2 },
      points: [
        { u: -0.05, v: 0 },
        { u: 0.05, v: 0 },
      ],
    };

    const rgbaAfterErase = rebuildSurfaceDecalRgba([paintStroke, eraseStroke], FOOTPRINT, SPEC);
    const erasedBase = (100 * SPEC.width + 100) * 4;
    // Inside eraser radius -> completely cleared back to transparent black.
    expect(rgbaAfterErase[erasedBase]).toBe(0);
    expect(rgbaAfterErase[erasedBase + 1]).toBe(0);
    expect(rgbaAfterErase[erasedBase + 2]).toBe(0);
    expect(rgbaAfterErase[erasedBase + 3]).toBe(0);

    // Outside eraser radius but inside painted disk (~u=0.22 -> col=122) -> paint remains intact.
    const untouchedAlpha = rgbaAfterErase[(100 * SPEC.width + 122) * 4 + 3] ?? 0;
    expect(untouchedAlpha).toBe(255);

    const engine = new BrushEngine();
    const s1 = engine.beginStroke(
      'surf1',
      FOOTPRINT,
      { diameter: 0.2, color: '#ffffff', opacity: 0.3, hardness: 0.4 },
      { u: -0.2, v: 0 },
      'erase',
    );
    if (!s1) throw new Error('Expected erase session to start');
    expect(s1.brush.opacity).toBe(1);
    expect(s1.brush.hardness).toBe(1);

    const s2 = engine.switchSurfaceStroke(
      'surf2',
      FOOTPRINT,
      { diameter: 0.2, color: '#ffffff', opacity: 1, hardness: 1 },
      { u: 0.1, v: 0 },
      { u: 0.2, v: 0 },
    );
    if (!s2) throw new Error('Expected cross-surface stroke transition');
    expect(s2.mode).toBe('erase');

    const batch = engine.commitBatch();
    expect(batch.length).toBe(2);
    expect(batch[0]?.surfaceId).toBe('surf1');
    expect(batch[0]?.points[batch[0].points.length - 1]?.u).toBeCloseTo(0.2, 4);
    expect(batch[1]?.surfaceId).toBe('surf2');
    expect(batch[1]?.mode).toBe('erase');
  });
});
