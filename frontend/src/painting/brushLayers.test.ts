import { describe, expect, it } from 'vitest';
import { buildBrushLayers, nextStrokeOrder } from './brushLayers';
import type { BrushStroke } from './brushEngine';
import type { ImageSpaceStroke } from './imageSpacePaint';

const brush = { diameter: 10, color: '#ff0000', opacity: 1, hardness: 1 };

function planar(id: string, order: number, mode: 'paint' | 'erase' = 'paint'): BrushStroke {
  return { id, order, mode, brush, surfaceId: 'wall', points: [{ u: 0, v: 0 }] };
}

function image(id: string, order: number): ImageSpaceStroke {
  return {
    id,
    order,
    mode: 'paint',
    brush,
    layer: 'fallback-2d',
    layerWidth: 1,
    layerHeight: 1,
    bounds: { minX: 0, minY: 0, maxX: 0, maxY: 0 },
    alphaMask: new Uint8Array([255]),
  };
}

describe('ordered brush layers', () => {
  it('keeps image strokes, planar strokes and erasing in gesture order', () => {
    const layers = buildBrushLayers(
      [planar('red', 1), planar('erase', 2, 'erase')],
      [image('blue', 0)],
    );
    expect(layers.map((layer) => [layer.kind, layer.key, layer.erase])).toEqual([
      ['image', 'blue', false],
      ['planar', 'red', false],
      ['planar', 'erase', true],
    ]);
  });

  it('batches only consecutive compatible strokes', () => {
    const layers = buildBrushLayers(
      [planar('a', 0), planar('b', 1), planar('c', 3)],
      [image('image', 2)],
    );
    expect(layers.map((layer) => layer.strokes.map((stroke) => stroke.id))).toEqual([
      ['a', 'b'],
      ['image'],
      ['c'],
    ]);
    expect(buildBrushLayers([], [])).toEqual([]);
  });

  it('retains increasing order after a high-order stroke survives deletion', () => {
    expect(nextStrokeOrder([planar('retained', 40)], [])).toBe(41);
  });
});
