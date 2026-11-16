import {
  createAlbedoPreviewCanvas,
  createResidualPreviewCanvas,
  createShadingPreviewCanvas,
} from '../editor/appearanceVisualizations';
import { extractImageColorsAtSize } from '../editor/geometryVisualizations';
import { rgbaUint8ToLinearRgb } from '../painting/colorSpace';
import type { AppearanceData, AppearanceMetadata, LoadedImage } from '../types/editor';

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

async function fetchAppearanceMap(
  appearanceId: string,
  kind: 'albedo' | 'shading' | 'residual',
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const response = await fetch(
    `${API_BASE}/api/appearance/${encodeURIComponent(appearanceId)}/maps/${kind}`,
    { signal },
  );
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return response.arrayBuffer();
}

export async function inferAppearanceMetadata(
  file: File,
  signal?: AbortSignal,
): Promise<AppearanceMetadata> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/api/appearance/infer`, {
    method: 'POST',
    body: formData,
    signal,
  });
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return (await response.json()) as AppearanceMetadata;
}

export async function fetchAppearanceForImage(
  image: LoadedImage,
  signal?: AbortSignal,
): Promise<AppearanceData> {
  const metadata = await inferAppearanceMetadata(image.sourceFile, signal);
  const { appearance_width: width, appearance_height: height, id } = metadata;
  const expectedValues = width * height * 3;

  const [albedoBuf, shadingBuf, residualBuf] = await Promise.all([
    fetchAppearanceMap(id, 'albedo', signal),
    fetchAppearanceMap(id, 'shading', signal),
    fetchAppearanceMap(id, 'residual', signal),
  ]);

  const albedo = new Float32Array(albedoBuf);
  const shading = new Float32Array(shadingBuf);
  const residual = new Float32Array(residualBuf);

  if (albedo.length !== expectedValues) {
    throw new Error(`Expected ${expectedValues} albedo values, received ${albedo.length}.`);
  }
  if (shading.length !== expectedValues) {
    throw new Error(`Expected ${expectedValues} shading values, received ${shading.length}.`);
  }
  if (residual.length !== expectedValues) {
    throw new Error(`Expected ${expectedValues} residual values, received ${residual.length}.`);
  }

  const refSrgbRgba = extractImageColorsAtSize(image.element, width, height);
  const linearReference = rgbaUint8ToLinearRgb(refSrgbRgba);

  const previews = {
    albedo: createAlbedoPreviewCanvas(albedo, width, height),
    shading: createShadingPreviewCanvas(shading, width, height),
    residual: createResidualPreviewCanvas(residual, width, height),
  };

  return {
    imageId: image.id,
    metadata,
    albedo,
    shading,
    residual,
    linearReference,
    previews,
  };
}
