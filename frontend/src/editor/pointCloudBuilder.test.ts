import { describe, expect, it } from 'vitest';
import {
  buildPointCloudAttributes,
  computeInitialCameraPose,
  computePointCloudStride,
  opencvPointToThreejs,
} from './pointCloudBuilder';

describe('pointCloudBuilder', () => {
  it('converts OpenCV camera coordinates to Three.js coordinates', () => {
    expect(opencvPointToThreejs(1.5, 2.0, 4.0)).toEqual([1.5, -2.0, -4.0]);
    expect(opencvPointToThreejs(-0.5, -1.0, 2.5)).toEqual([-0.5, 1.0, -2.5]);
  });

  it('excludes invalid masked and non-finite points', () => {
    const width = 2;
    const height = 2;
    const points = new Float32Array([
      1,
      2,
      3,
      Number.POSITIVE_INFINITY,
      0,
      Number.POSITIVE_INFINITY,
      -1,
      -2,
      5,
      0,
      0,
      -1, // behind camera (oz <= 0)
    ]);
    const colors = new Uint8ClampedArray([
      255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ]);
    const mask = new Uint8Array([1, 0, 1, 1]);

    const attrs = buildPointCloudAttributes({
      points,
      colors,
      mask,
      width,
      height,
      stride: 1,
    });

    expect(attrs.pointCount).toBe(2);
    expect(Array.from(attrs.positions)).toEqual([1, -2, -3, -1, 2, -5]);
    expect(attrs.colors[0]).toBeCloseTo(1, 5);
    expect(attrs.colors[5]).toBeCloseTo(1, 5);
    expect(attrs.center).toEqual([0, 0, -4]);
  });

  it('computes stride from point budget and derives camera pose', () => {
    expect(computePointCloudStride(1000, 1000, 300_000)).toBe(2);
    expect(computePointCloudStride(400, 300, 300_000)).toBe(1);

    const pose = computeInitialCameraPose([0, 0, -4], 2, [
      [0.8, 0, 0.5],
      [0, 1.0, 0.5],
      [0, 0, 1],
    ]);
    expect(pose.position).toEqual([0, 0, 0]);
    expect(pose.target).toEqual([0, 0, -4]);
    expect(pose.fovDeg).toBeCloseTo((2 * Math.atan(0.5) * 180) / Math.PI, 4);
  });
});
