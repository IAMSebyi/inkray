import { clamp } from '../editor/viewportTransforms';
import { linearChannelToSrgb } from '../painting/colorSpace';

function createRgbPreviewCanvas(
  rgb: Float32Array,
  width: number,
  height: number,
  normalizePercentile: number | null,
  minScale = 1e-4,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return canvas;
  }

  let scale = 1.0;
  if (normalizePercentile !== null) {
    const pixelCount = width * height;
    const stride = pixelCount > 80_000 ? 4 : 1;
    const mags: number[] = [];
    for (let i = 0; i < pixelCount; i += stride) {
      const base = i * 3;
      const luma =
        0.2126 * Math.max(0, rgb[base] ?? 0) +
        0.7152 * Math.max(0, rgb[base + 1] ?? 0) +
        0.0722 * Math.max(0, rgb[base + 2] ?? 0);
      if (Number.isFinite(luma)) {
        mags.push(luma);
      }
    }
    if (mags.length > 0) {
      mags.sort((a, b) => a - b);
      const idx = Math.floor((clamp(normalizePercentile, 0, 100) / 100) * (mags.length - 1));
      const refVal = Math.max(minScale, mags[idx] ?? minScale);
      scale = 1.0 / refVal;
    }
  }

  const img = ctx.createImageData(width, height);
  const pixelCount = width * height;
  for (let i = 0; i < pixelCount; i++) {
    const src = i * 3;
    const dst = i * 4;
    const rLin = Math.max(0, rgb[src] ?? 0) * scale;
    const gLin = Math.max(0, rgb[src + 1] ?? 0) * scale;
    const bLin = Math.max(0, rgb[src + 2] ?? 0) * scale;
    img.data[dst] = Math.round(linearChannelToSrgb(rLin) * 255);
    img.data[dst + 1] = Math.round(linearChannelToSrgb(gLin) * 255);
    img.data[dst + 2] = Math.round(linearChannelToSrgb(bLin) * 255);
    img.data[dst + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

export function createAlbedoPreviewCanvas(
  albedo: Float32Array,
  width: number,
  height: number,
): HTMLCanvasElement {
  return createRgbPreviewCanvas(albedo, width, height, null);
}

export function createShadingPreviewCanvas(
  shading: Float32Array,
  width: number,
  height: number,
): HTMLCanvasElement {
  return createRgbPreviewCanvas(shading, width, height, 98, 1e-4);
}

export function createResidualPreviewCanvas(
  residual: Float32Array,
  width: number,
  height: number,
): HTMLCanvasElement {
  return createRgbPreviewCanvas(residual, width, height, 99, 0.25);
}
