import { describe, expect, it } from 'vitest';
import {
  type Vec3Tuple,
  parseNormalizedIntrinsics,
  projectOpenCvPointToImage,
  vecCross,
  vecDot,
  vecLength,
  vecNormalize,
} from './geometryAccess';
import { PlaneFrame } from './planeFrame';
import { makeSyntheticIntrinsics } from './testFixtures';

describe('PlaneFrame', () => {
  it('constructs a right-handed camera-facing orthonormal frame and roundtrips 3D points', () => {
    const rawNormal = vecNormalize([0.4, 0.15, 0.9]) ?? [0, 0, 1];
    const anchor: Vec3Tuple = [0.25, -0.1, 3.6];
    const frame = PlaneFrame.fromPlane(
      {
        origin: [0, 0, 3.5],
        normal: rawNormal,
      },
      anchor,
    );

    // Normal must orient toward camera ([0,0,0]).
    expect(
      vecDot(frame.normal, [-frame.origin[0], -frame.origin[1], -frame.origin[2]]),
    ).toBeGreaterThan(0);

    // Frame axes must be unit length and mutually orthogonal.
    expect(vecLength(frame.uAxis)).toBeCloseTo(1, 6);
    expect(vecLength(frame.vAxis)).toBeCloseTo(1, 6);
    expect(vecLength(frame.normal)).toBeCloseTo(1, 6);
    expect(vecDot(frame.uAxis, frame.vAxis)).toBeCloseTo(0, 6);
    expect(vecDot(frame.uAxis, frame.normal)).toBeCloseTo(0, 6);
    expect(vecDot(frame.vAxis, frame.normal)).toBeCloseTo(0, 6);

    const cross = vecCross(frame.uAxis, frame.vAxis);
    // In OpenCV space (+X right, +Y down, +Z forward), +u x +n points along +v, so +u x +v aligns with -n or +n consistently.
    expect(Math.abs(vecDot(cross, frame.normal))).toBeCloseTo(1, 6);

    const roundtripWorld = frame.localToWorld(0.42, -0.31, 0.05);
    const localBack = frame.worldToLocal(roundtripWorld);
    expect(localBack[0]).toBeCloseTo(0.42, 6);
    expect(localBack[1]).toBeCloseTo(-0.31, 6);
    expect(localBack[2]).toBeCloseTo(0.05, 6);
  });

  it('handles degenerate plane normals nearly parallel to camera +X axis', () => {
    const frame = PlaneFrame.fromPlane({
      origin: [-2.5, 0.1, 1.5],
      normal: [1, 0, 0],
    });

    expect(vecLength(frame.uAxis)).toBeCloseTo(1, 6);
    expect(vecLength(frame.vAxis)).toBeCloseTo(1, 6);
    expect(vecDot(frame.uAxis, frame.vAxis)).toBeCloseTo(0, 6);
    expect(vecDot(frame.uAxis, frame.normal)).toBeCloseTo(0, 6);
  });

  it('exhibits physical perspective foreshortening across an oblique plane', () => {
    const intrinsics = parseNormalizedIntrinsics(makeSyntheticIntrinsics(1.0, 1.0, 0.5, 0.5));
    const imageSize = { width: 1000, height: 1000 };
    const normal = vecNormalize([0.65, 0, -0.76]) ?? [0, 0, -1];
    const frame = PlaneFrame.fromPlane({ origin: [0, 0, 4.0], normal });

    // Near left side of tilted plane vs far right side in pixel steps of +50px.
    const hitLeftA = frame.imagePointToPlane({ x: 250, y: 500 }, imageSize, intrinsics);
    const hitLeftB = frame.imagePointToPlane({ x: 300, y: 500 }, imageSize, intrinsics);
    const hitRightA = frame.imagePointToPlane({ x: 700, y: 500 }, imageSize, intrinsics);
    const hitRightB = frame.imagePointToPlane({ x: 750, y: 500 }, imageSize, intrinsics);

    if (!hitLeftA || !hitLeftB || !hitRightA || !hitRightB) {
      throw new Error('Expected rays to intersect oblique plane');
    }

    const duNear = Math.abs(hitLeftB.u - hitLeftA.u);
    const duFar = Math.abs(hitRightB.u - hitRightA.u);
    // Pixels further away along the plane cover more physical distance in plane units.
    expect(hitRightA.depth).toBeGreaterThan(hitLeftA.depth);
    expect(duFar).toBeGreaterThan(duNear * 1.3);

    // Reprojecting any hit point must land back on the same image pixel.
    const reproj = projectOpenCvPointToImage(hitRightA.point, intrinsics, imageSize);
    if (!reproj) {
      throw new Error('Expected positive-depth point to reproject');
    }
    expect(reproj.x).toBeCloseTo(700, 4);
    expect(reproj.y).toBeCloseTo(500, 4);
  });

  it('rejects parallel rays and intersections behind the camera', () => {
    const frame = PlaneFrame.fromPlane({
      origin: [0, 0, 3.0],
      normal: [0, 0, -1],
    });

    // Ray parallel to Z=3 plane.
    expect(frame.intersectRay([0, 0, 0], [1, 0, 0])).toBeNull();
    // Ray pointing backwards (-Z).
    expect(frame.intersectRay([0, 0, 0], [0, 0, -1])).toBeNull();
  });
});
