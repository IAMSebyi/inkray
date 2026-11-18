import * as THREE from 'three';

const _vpMatrix = new THREE.Matrix4();
const _mvpMatrix = new THREE.Matrix4();
const _ndcPointer = new THREE.Vector2();
const _raycaster = new THREE.Raycaster();

export interface PointCloudPickParams {
  positions: Float32Array;
  camera: THREE.Camera;
  ndcX: number;
  ndcY: number;
  viewportWidth: number;
  viewportHeight: number;
  pixelRadius: number;
  modelMatrix?: THREE.Matrix4;
}

/**
 * Pick the nearest point to the camera within `pixelRadius` screen pixels of `(ndcX, ndcY)`.
 */
export function pickPointInPointCloud({
  positions,
  camera,
  ndcX,
  ndcY,
  viewportWidth,
  viewportHeight,
  pixelRadius,
  modelMatrix,
}: PointCloudPickParams): THREE.Vector3 | null {
  if (positions.length < 3 || viewportWidth <= 0 || viewportHeight <= 0 || pixelRadius <= 0) {
    return null;
  }

  camera.updateMatrixWorld();
  _vpMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  if (modelMatrix) {
    _mvpMatrix.multiplyMatrices(_vpMatrix, modelMatrix);
  } else {
    _mvpMatrix.copy(_vpMatrix);
  }

  const m = _mvpMatrix.elements;
  const m0 = m[0];
  const m1 = m[1];
  const m2 = m[2];
  const m3 = m[3];
  const m4 = m[4];
  const m5 = m[5];
  const m6 = m[6];
  const m7 = m[7];
  const m8 = m[8];
  const m9 = m[9];
  const m10 = m[10];
  const m11 = m[11];
  const m12 = m[12];
  const m13 = m[13];
  const m14 = m[14];
  const m15 = m[15];

  const halfW = viewportWidth * 0.5;
  const halfH = viewportHeight * 0.5;
  const maxDistSq = pixelRadius * pixelRadius;
  const tolNdcX = pixelRadius / halfW;
  const tolNdcY = pixelRadius / halfH;

  let bestIndex = -1;
  let bestVz = Number.POSITIVE_INFINITY;
  let bestDistSq = Number.POSITIVE_INFINITY;

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i] ?? 0;
    const y = positions[i + 1] ?? 0;
    const z = positions[i + 2] ?? 0;

    const clipW = m3 * x + m7 * y + m11 * z + m15;
    if (clipW <= 0) continue;

    const invW = 1 / clipW;
    const vx = (m0 * x + m4 * y + m8 * z + m12) * invW;
    if (Math.abs(vx - ndcX) > tolNdcX) continue;

    const vy = (m1 * x + m5 * y + m9 * z + m13) * invW;
    if (Math.abs(vy - ndcY) > tolNdcY) continue;

    const vz = (m2 * x + m6 * y + m10 * z + m14) * invW;
    if (vz < -1 || vz > 1) continue;

    const dx = (vx - ndcX) * halfW;
    const dy = (vy - ndcY) * halfH;
    const distSq = dx * dx + dy * dy;
    if (distSq > maxDistSq) continue;

    if (vz < bestVz - 1e-4 || (Math.abs(vz - bestVz) <= 1e-4 && distSq < bestDistSq)) {
      bestIndex = i;
      bestVz = vz;
      bestDistSq = distSq;
    }
  }

  if (bestIndex < 0) {
    return null;
  }

  const hit = new THREE.Vector3(
    positions[bestIndex] ?? 0,
    positions[bestIndex + 1] ?? 0,
    positions[bestIndex + 2] ?? 0,
  );
  if (modelMatrix) {
    hit.applyMatrix4(modelMatrix);
  }
  return hit;
}

export interface Object3DPickParams {
  object: THREE.Object3D;
  camera: THREE.Camera;
  ndcX: number;
  ndcY: number;
  viewportWidth: number;
  viewportHeight: number;
  pointSize: number;
}

/**
 * Pick a world-space surface point on either a `THREE.Points` cloud or a `THREE.Mesh`.
 */
export function pickPointInObject3D({
  object,
  camera,
  ndcX,
  ndcY,
  viewportWidth,
  viewportHeight,
  pointSize,
}: Object3DPickParams): THREE.Vector3 | null {
  object.updateWorldMatrix(true, false);

  if (object instanceof THREE.Points) {
    const geom = object.geometry as THREE.BufferGeometry;
    const positionAttr = geom.getAttribute('position');
    if (!(positionAttr instanceof THREE.BufferAttribute)) {
      return null;
    }
    const array = positionAttr.array;
    if (!(array instanceof Float32Array)) {
      return null;
    }
    return pickPointInPointCloud({
      positions: array,
      camera,
      ndcX,
      ndcY,
      viewportWidth,
      viewportHeight,
      pixelRadius: Math.max(8, pointSize + 6),
      modelMatrix: object.matrixWorld,
    });
  }

  if (object instanceof THREE.Mesh) {
    camera.updateMatrixWorld();
    _ndcPointer.set(ndcX, ndcY);
    _raycaster.setFromCamera(_ndcPointer, camera);
    const hits = _raycaster.intersectObject(object, false);
    return hits[0]?.point.clone() ?? null;
  }

  return null;
}
