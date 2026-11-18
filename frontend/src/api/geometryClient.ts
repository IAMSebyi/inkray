import type {
  GeometryData,
  GeometryMetadata,
  GeometryModel,
  LoadedImage,
  NormalModel,
  Point2D,
} from '../types/editor';
import {
  createDepthPreviewCanvas,
  createMaskPreviewCanvas,
  createNormalsPreviewCanvas,
  extractImageColorsAtSize,
} from '../editor/geometryVisualizations';
import type { Vec3Tuple } from '../geometry/geometryAccess';
import { PlaneFrame, type SurfaceSelectionOutcome } from '../geometry/planeFrame';
import { API_BASE, parseErrorMessage } from './backend';

async function fetchBinaryMap(
  geometryId: string,
  kind: 'depth' | 'normals' | 'points' | 'mask',
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const response = await fetch(
    `${API_BASE}/api/geometry/${encodeURIComponent(geometryId)}/maps/${kind}`,
    { signal },
  );
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return response.arrayBuffer();
}

export interface GeometryRequestOptions {
  normalModel?: NormalModel | null;
  smoothIterations?: number;
}

export async function inferGeometryMetadata(
  file: File,
  model: GeometryModel | undefined,
  signal?: AbortSignal,
  options?: GeometryRequestOptions,
): Promise<GeometryMetadata> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  if (model) params.set('model', model);
  if (options?.normalModel) {
    params.set('refine_normals', 'true');
    params.set('normal_model', options.normalModel);
  }
  if (options?.smoothIterations && options.smoothIterations > 0) {
    params.set('smooth_iterations', String(options.smoothIterations));
  }
  const query = params.size > 0 ? `?${params}` : '';
  const response = await fetch(`${API_BASE}/api/geometry/infer${query}`, {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return (await response.json()) as GeometryMetadata;
}

export async function fetchGeometryForImage(
  image: LoadedImage,
  model: GeometryModel | undefined,
  signal?: AbortSignal,
  options?: GeometryRequestOptions,
): Promise<GeometryData> {
  const metadata = await inferGeometryMetadata(image.sourceFile, model, signal, options);
  const { inference_width: width, inference_height: height, id } = metadata;
  const pixelCount = width * height;

  const [depthBuf, pointsBuf, maskBuf, normalsBuf] = await Promise.all([
    fetchBinaryMap(id, 'depth', signal),
    fetchBinaryMap(id, 'points', signal),
    fetchBinaryMap(id, 'mask', signal),
    metadata.has_normals ? fetchBinaryMap(id, 'normals', signal) : Promise.resolve(null),
  ]);

  const depth = new Float32Array(depthBuf);
  const points = new Float32Array(pointsBuf);
  const mask = new Uint8Array(maskBuf);
  const normals = normalsBuf ? new Float32Array(normalsBuf) : null;

  if (depth.length !== pixelCount) {
    throw new Error(`Expected ${pixelCount} depth samples, received ${depth.length}.`);
  }
  if (points.length !== pixelCount * 3) {
    throw new Error(`Expected ${pixelCount * 3} point values, received ${points.length}.`);
  }
  if (mask.length !== pixelCount) {
    throw new Error(`Expected ${pixelCount} mask samples, received ${mask.length}.`);
  }
  if (normals && normals.length !== pixelCount * 3) {
    throw new Error(`Expected ${pixelCount * 3} normal values, received ${normals.length}.`);
  }

  const colors = extractImageColorsAtSize(image.element, width, height);
  const previews = {
    depth: createDepthPreviewCanvas(depth, mask, width, height),
    normals: normals ? createNormalsPreviewCanvas(normals, mask, width, height) : null,
    mask: createMaskPreviewCanvas(mask, width, height),
  };

  return {
    imageId: image.id,
    metadata: { ...metadata, model: metadata.model ?? model },
    depth,
    points,
    mask,
    normals,
    colors,
    previews,
  };
}

interface BackendVec2 {
  x: number;
  y: number;
}

interface BackendSurfaceSelectionPayload {
  ok: boolean;
  reason?: string;
  selection?: {
    seed_image_point: BackendVec2;
    seed_inference_pixel: BackendVec2;
    seed_point_3d: Vec3Tuple;
    plane: {
      origin: Vec3Tuple;
      normal: Vec3Tuple;
    };
    frame: {
      origin: Vec3Tuple;
      normal: Vec3Tuple;
      u_axis: Vec3Tuple;
      v_axis: Vec3Tuple;
    };
    support_mask_b64: string;
    support_width: number;
    support_height: number;
    inlier_count: number;
    rms_error: number;
    suggested_footprint: {
      min_u: number;
      max_u: number;
      min_v: number;
      max_v: number;
    };
    reference_brush_diameter: number;
  };
}

function decodeBase64Uint8(encoded: string): Uint8Array {
  const binary = window.atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export async function selectSurfaceForGeometry(
  geometryId: string,
  imagePoint: Point2D,
  signal?: AbortSignal,
): Promise<SurfaceSelectionOutcome> {
  const response = await fetch(
    `${API_BASE}/api/geometry/${encodeURIComponent(geometryId)}/surface/select`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_x: imagePoint.x, image_y: imagePoint.y }),
      signal,
    },
  );

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  const payload = (await response.json()) as BackendSurfaceSelectionPayload;
  if (!payload.ok || !payload.selection) {
    return {
      ok: false,
      reason: payload.reason ?? 'Could not select a planar surface at that location.',
    };
  }

  const s = payload.selection;
  const supportMask = decodeBase64Uint8(s.support_mask_b64);
  const expectedLen = s.support_width * s.support_height;
  if (supportMask.length !== expectedLen) {
    throw new Error(
      `Expected ${expectedLen} surface support mask bytes, received ${supportMask.length}.`,
    );
  }

  const frame = new PlaneFrame(s.frame.origin, s.frame.normal, s.frame.u_axis, s.frame.v_axis);

  return {
    ok: true,
    selection: {
      seedImagePoint: s.seed_image_point,
      seedInferencePixel: s.seed_inference_pixel,
      seedPoint3D: s.seed_point_3d,
      plane: s.plane,
      frame,
      supportMask,
      inlierCount: s.inlier_count,
      rmsError: s.rms_error,
      suggestedFootprint: {
        minU: s.suggested_footprint.min_u,
        maxU: s.suggested_footprint.max_u,
        minV: s.suggested_footprint.min_v,
        maxV: s.suggested_footprint.max_v,
      },
      referenceBrushDiameter: s.reference_brush_diameter,
    },
  };
}
