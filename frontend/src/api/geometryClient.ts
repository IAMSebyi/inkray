import type { GeometryData, GeometryMetadata, LoadedImage } from '../types/editor';
import {
  createDepthPreviewCanvas,
  createMaskPreviewCanvas,
  createNormalsPreviewCanvas,
  extractImageColorsAtSize,
} from '../editor/geometryVisualizations';

const API_BASE = ((import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '').replace(
  /\/+$/,
  '',
);

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === 'string' && body.detail.trim() !== '') {
      return body.detail;
    }
  } catch {
    // Ignore non-JSON error bodies.
  }
  return `Request failed with HTTP ${response.status}.`;
}

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

export async function inferGeometryMetadata(
  file: File,
  signal?: AbortSignal,
): Promise<GeometryMetadata> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/api/geometry/infer`, {
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
  signal?: AbortSignal,
): Promise<GeometryData> {
  const metadata = await inferGeometryMetadata(image.sourceFile, signal);
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
    metadata,
    depth,
    points,
    mask,
    normals,
    colors,
    previews,
  };
}
