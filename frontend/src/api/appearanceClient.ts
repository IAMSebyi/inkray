import {
  createAlbedoPreviewCanvas,
  createResidualPreviewCanvas,
  createShadingPreviewCanvas,
} from '../editor/appearanceVisualizations';
import { extractImageColorsAtSize } from '../editor/geometryVisualizations';
import { rgbaUint8ToLinearRgb } from '../painting/colorSpace';
import { deriveLighting } from '../painting/intrinsicLighting';
import type {
  AppearanceData,
  AppearanceMetadata,
  AppearanceModel,
  LoadedImage,
} from '../types/editor';
import { API_BASE, parseErrorMessage } from './backend';

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
  model: AppearanceModel,
  signal?: AbortSignal,
): Promise<AppearanceMetadata> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${API_BASE}/api/appearance/infer?model=${encodeURIComponent(model)}`,
    {
      method: 'POST',
      body: formData,
      signal,
    },
  );
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
  return (await response.json()) as AppearanceMetadata;
}

export async function fetchAppearanceForImage(
  image: LoadedImage,
  model: AppearanceModel,
  signal?: AbortSignal,
): Promise<AppearanceData> {
  const metadata = await inferAppearanceMetadata(image.sourceFile, model, signal);
  const { appearance_width: width, appearance_height: height, id } = metadata;
  const expectedValues = width * height * 3;

  const [albedoBuf, shadingBuf, residualBuf] = await Promise.all([
    fetchAppearanceMap(id, 'albedo', signal),
    metadata.arrays.shading ? fetchAppearanceMap(id, 'shading', signal) : null,
    metadata.arrays.residual ? fetchAppearanceMap(id, 'residual', signal) : null,
  ]);

  const albedo = new Float32Array(albedoBuf);
  let shading = shadingBuf ? new Float32Array(shadingBuf) : null;
  let residual = residualBuf ? new Float32Array(residualBuf) : null;

  if (albedo.length !== expectedValues) {
    throw new Error(`Expected ${expectedValues} albedo values, received ${albedo.length}.`);
  }
  if (shading && shading.length !== expectedValues) {
    throw new Error(`Expected ${expectedValues} shading values, received ${shading.length}.`);
  }
  if (residual && residual.length !== expectedValues) {
    throw new Error(`Expected ${expectedValues} residual values, received ${residual.length}.`);
  }

  const refSrgbRgba = extractImageColorsAtSize(image.element, width, height);
  const linearReference = rgbaUint8ToLinearRgb(refSrgbRgba);
  const derivedShading = !shading && !residual && metadata.prediction_space === 'linear';
  if (derivedShading) {
    ({ shading, residual } = deriveLighting(linearReference, albedo, width, height));
  }

  const previews = {
    albedo: createAlbedoPreviewCanvas(albedo, width, height),
    shading: shading ? createShadingPreviewCanvas(shading, width, height) : null,
    residual: residual ? createResidualPreviewCanvas(residual, width, height) : null,
  };

  return {
    imageId: image.id,
    metadata,
    albedo,
    shading,
    residual,
    linearReference,
    derivedShading,
    previews,
  };
}
