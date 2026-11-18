import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { pickPointInObject3D, pickPointInPointCloud } from './pointCloudPicking';

function makeCamera(): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  return camera;
}

describe('pointCloudPicking', () => {
  it('picks the frontmost visible point within screen-pixel tolerance', () => {
    const camera = makeCamera();
    const positions = new Float32Array([
      0,
      0,
      0, // center front point
      0,
      0,
      -3, // center rear point behind front point
      5,
      5,
      0, // far off-center point
      0,
      0,
      10, // behind camera
    ]);

    const hit = pickPointInPointCloud({
      positions,
      camera,
      ndcX: 0,
      ndcY: 0,
      viewportWidth: 400,
      viewportHeight: 400,
      pixelRadius: 8,
    });
    expect(hit?.toArray()).toEqual([0, 0, 0]);

    const miss = pickPointInPointCloud({
      positions,
      camera,
      ndcX: 0.9,
      ndcY: -0.9,
      viewportWidth: 400,
      viewportHeight: 400,
      pixelRadius: 6,
    });
    expect(miss).toBeNull();
  });

  it('picks exact surface intersections on both THREE.Points and THREE.Mesh', () => {
    const camera = makeCamera();

    const cloudGeom = new THREE.BufferGeometry();
    cloudGeom.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([0.2, -0.1, -1.5, 2, 2, 0], 3),
    );
    const cloud = new THREE.Points(cloudGeom, new THREE.PointsMaterial({ size: 2 }));
    const projected = new THREE.Vector3(0.2, -0.1, -1.5).project(camera);
    const cloudHit = pickPointInObject3D({
      object: cloud,
      camera,
      ndcX: projected.x,
      ndcY: projected.y,
      viewportWidth: 800,
      viewportHeight: 800,
      pointSize: 2,
    });
    expect(cloudHit).not.toBeNull();
    expect(cloudHit?.x).toBeCloseTo(0.2, 5);
    expect(cloudHit?.y).toBeCloseTo(-0.1, 5);
    expect(cloudHit?.z).toBeCloseTo(-1.5, 5);
    cloudGeom.dispose();
    (cloud.material as THREE.Material).dispose();

    const meshGeom = new THREE.BufferGeometry();
    meshGeom.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-2, -2, -2, 2, -2, -2, 0, 2, -2], 3),
    );
    meshGeom.setIndex([0, 1, 2]);
    const mesh = new THREE.Mesh(meshGeom, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
    const meshHit = pickPointInObject3D({
      object: mesh,
      camera,
      ndcX: 0,
      ndcY: 0,
      viewportWidth: 800,
      viewportHeight: 800,
      pointSize: 2,
    });
    expect(meshHit).not.toBeNull();
    expect(meshHit?.x).toBeCloseTo(0, 4);
    expect(meshHit?.y).toBeCloseTo(0, 4);
    expect(meshHit?.z).toBeCloseTo(-2, 4);
    meshGeom.dispose();
    (mesh.material as THREE.Material).dispose();
  });
});
