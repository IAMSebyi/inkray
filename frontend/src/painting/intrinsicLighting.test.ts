import { expect, it } from 'vitest';
import { compositeLightingAwarePixel } from './colorSpace';
import { deriveLighting } from './intrinsicLighting';

it('derives colored diffuse shading and leaves clipped highlights in the residual', () => {
  const albedo = new Float32Array([0.4, 0.5, 0.2, 0.1, 0.1, 0]);
  const image = new Float32Array([0.2, 0.5, 0.3, 1, 0.08, 0]);
  const { shading, residual } = deriveLighting(image, albedo, 2, 1);
  expect(shading[0]).toBeCloseTo(0.5);
  expect(shading[1]).toBeCloseTo(1);
  expect(shading[2]).toBeCloseTo(1.5);
  expect(shading[3]).toBeCloseTo(0.8);
  expect(residual[3]).toBeCloseTo(0.92);
  expect([...shading, ...residual].every(Number.isFinite)).toBe(true);
  for (const [i, value] of image.entries()) {
    expect((albedo[i] ?? 0) * (shading[i] ?? 0) + (residual[i] ?? 0)).toBeCloseTo(value);
  }
});
it('keeps a small unsaturated highlight on the surface during recoloring', () => {
  const width = 5;
  const albedo = new Float32Array(width * width * 3).fill(0.5);
  const image = new Float32Array(albedo.length).fill(0.3);
  const center = 3 * (2 * width + 2);
  image.fill(0.6, center, center + 3);
  const { shading, residual } = deriveLighting(image, albedo, width, width);
  expect(shading[center]).toBeCloseTo(0.6);
  expect(residual[center]).toBeCloseTo(0.3);
  const recolored = compositeLightingAwarePixel({
    origLinear: [0.6, 0.6, 0.6],
    paintLinear: [0.2, 0.3, 0.4],
    alpha: 1,
    albedo: [0.5, 0.5, 0.5],
    shadingRaw: [shading[center] ?? 0, shading[center + 1] ?? 0, shading[center + 2] ?? 0],
    shadingScale: 1,
    mode: 'shading',
  });
  expect(recolored[0]).toBeCloseTo(0.42);
  expect(recolored[1]).toBeCloseTo(0.48);
  expect(recolored[2]).toBeCloseTo(0.54);
});

it('keeps a hard shadow edge and material changes', () => {
  const albedo = new Float32Array(5 * 5 * 3);
  const image = new Float32Array(albedo.length);
  for (let i = 0; i < 25; i++) {
    const column = i % 5;
    const color = column === 3 ? 0.15 : 0.5;
    const shade = column < 2 ? 0.3 : 1.2;
    albedo.fill(color, i * 3, i * 3 + 3);
    image.fill(color * shade, i * 3, i * 3 + 3);
  }
  const { shading } = deriveLighting(image, albedo, 5, 5);
  expect(shading[3 * (2 * 5 + 1)]).toBeCloseTo(0.3);
  expect(shading[3 * (2 * 5 + 2)]).toBeCloseTo(1.2);
  expect(shading[3 * (2 * 5 + 3)]).toBeCloseTo(1.2);
});

it('retains signed residuals when a dark color channel needs regularization', () => {
  const albedo = new Float32Array([0.5, 0.5, 0.02]);
  const image = new Float32Array([0.3, 0.3, 0]);
  const { shading, residual } = deriveLighting(image, albedo, 1, 1);
  expect(shading[2]).toBeCloseTo(0.36, 3);
  expect(residual[2]).toBeCloseTo(-0.0072, 4);
  expect((shading[2] ?? 0) * (albedo[2] ?? 0) + (residual[2] ?? 0)).toBeCloseTo(0, 6);
});

it('rejects mismatched and non-finite input maps', () => {
  const pixel = new Float32Array([0.3, 0.4, 0.5]);
  expect(() => deriveLighting(pixel, pixel, 2, 1)).toThrow('same RGB grid');
  expect(() => deriveLighting(pixel, new Float32Array([0, NaN, 0]), 1, 1)).toThrow('finite');
});
