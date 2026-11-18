import { expect, it } from 'vitest';
import type { GeometryData } from '../types/editor';
import { compareDepth } from './depthComparison';

function geometry(depth: number[], mask: number[]): GeometryData {
  return {
    imageId: 'photo',
    metadata: { image_sha256: 'photo', inference_width: depth.length, inference_height: 1 },
    depth: new Float32Array(depth),
    mask: new Uint8Array(mask),
  } as GeometryData;
}

it('aligns different grids by the median log ratio on shared valid pixels', () => {
  const reference = geometry([2, 4, 6, 8], [1, 1, 1, 0]);
  const candidate = geometry([1, 1, 2, 2, 6, 6, 70, 70], [1, 1, 1, 1, 1, 1, 1, 1]);
  const result = compareDepth(reference, candidate);
  expect(result?.scale).toBeCloseTo(2);
  expect(result?.overlap).toBe(0.75);
  expect(result?.differingFraction).toBeCloseTo(1 / 3);
  expect(result?.logDifference[0]).toBeCloseTo(0);
  expect(result?.logDifference[2]).toBeCloseTo(Math.log(2));
  expect(result?.logDifference[3]).toBeNaN();
});

it('returns no comparison with disjoint masks or different images', () => {
  const reference = geometry([1, 2], [1, 0]);
  expect(compareDepth(reference, geometry([1, 2], [0, 1]))).toBeNull();
  expect(compareDepth(reference, { ...reference, imageId: 'other' })).toBeNull();
  expect(
    compareDepth(reference, {
      ...reference,
      metadata: { ...reference.metadata, image_sha256: 'other' },
    }),
  ).toBeNull();
});

it('excludes invalid depths and reports disagreement symmetrically', () => {
  const mask = [1, 1, 1, 1, 1, 1, 1, 1];
  const reference = geometry([4, 4, 4, 4, 4, 4, 4, 4], mask);
  const candidate = geometry([1, 2, 2, 2.4, 4, NaN, 0, -2], mask);
  const comparison = compareDepth(reference, candidate);
  const reverse = compareDepth(candidate, reference);
  expect(comparison?.scale).toBeCloseTo(2);
  expect(reverse?.scale).toBeCloseTo(0.5);
  expect(comparison?.overlap).toBe(5 / 8);
  expect(comparison?.differingFraction).toBe(2 / 5);
  expect(reverse?.differingFraction).toBe(2 / 5);
  expect(comparison?.logDifference[0]).toBeCloseTo(-Math.log(2));
  expect(comparison?.logDifference[4]).toBeCloseTo(Math.log(2));
  expect(Array.from(comparison?.logDifference.slice(5) ?? [])).toEqual([NaN, NaN, NaN]);
});
