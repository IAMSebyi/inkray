import type { DepthComparison } from '../geometry/depthComparison';
import type { GeometryData, Point2D, Size2D } from '../types/editor';

const TURBO_STOPS: readonly [number, number, number][] = [
  [122, 4, 3],
  [210, 49, 5],
  [251, 128, 34],
  [246, 207, 57],
  [162, 252, 60],
  [44, 217, 165],
  [38, 152, 239],
  [65, 72, 189],
  [48, 18, 59],
];

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function sampleTurboPalette(t: number): [number, number, number] {
  const scaled = clamp01(t) * (TURBO_STOPS.length - 1);
  const idx = Math.min(TURBO_STOPS.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const c0 = TURBO_STOPS[idx] ?? [122, 4, 3];
  const c1 = TURBO_STOPS[idx + 1] ?? [48, 18, 59];
  return [
    Math.round(c0[0] + (c1[0] - c0[0]) * frac),
    Math.round(c0[1] + (c1[1] - c0[1]) * frac),
    Math.round(c0[2] + (c1[2] - c0[2]) * frac),
  ];
}

export function computeValidDepthBounds(
  depth: Float32Array,
  mask: Uint8Array,
): [number, number] | null {
  const total = depth.length;
  const step = Math.max(1, Math.floor(total / 50_000));
  const samples: number[] = [];

  for (let i = 0; i < total; i += step) {
    if (mask[i] === 0) {
      continue;
    }
    const d = depth[i];
    if (d !== undefined && Number.isFinite(d) && d > 0) {
      samples.push(d);
    }
  }

  if (samples.length === 0) {
    for (let i = 0; i < total; i++) {
      if (mask[i] === 0) {
        continue;
      }
      const d = depth[i];
      if (d !== undefined && Number.isFinite(d) && d > 0) {
        samples.push(d);
      }
    }
  }

  if (samples.length === 0) {
    return null;
  }

  samples.sort((a, b) => a - b);
  const lowIdx = Math.floor((samples.length - 1) * 0.02);
  const highIdx = Math.floor((samples.length - 1) * 0.98);
  const low = samples[lowIdx] ?? samples[0] ?? 0;
  const high = samples[highIdx] ?? samples[samples.length - 1] ?? low;

  if (high <= low) {
    return [samples[0] ?? low, samples[samples.length - 1] ?? high];
  }
  return [low, high];
}

function createCanvasFromRgba(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to create 2D canvas context for geometry preview.');
  }
  const imageData = new ImageData(new Uint8ClampedArray(rgba), width, height);
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

export function createDepthDifferenceCanvas(comparison: DepthComparison): HTMLCanvasElement {
  const rgba = new Uint8ClampedArray(comparison.logDifference.length * 4);
  for (const [i, value] of comparison.logDifference.entries()) {
    if (!Number.isFinite(value)) continue;
    const intensity = clamp01(Math.abs(value) / Math.log(2));
    const color = value < 0 ? ([70, 138, 235] as const) : ([255, 148, 77] as const);
    const offset = i * 4;
    rgba[offset] = 31 + (color[0] - 31) * intensity;
    rgba[offset + 1] = 35 + (color[1] - 35) * intensity;
    rgba[offset + 2] = 44 + (color[2] - 44) * intensity;
    rgba[offset + 3] = 255;
  }
  return createCanvasFromRgba(rgba, comparison.width, comparison.height);
}

export function createDepthPreviewCanvas(
  depth: Float32Array,
  mask: Uint8Array,
  width: number,
  height: number,
): HTMLCanvasElement {
  const pixelCount = width * height;
  const rgba = new Uint8ClampedArray(pixelCount * 4);
  const bounds = computeValidDepthBounds(depth, mask);

  const nearInv = bounds ? 1 / Math.max(bounds[0], 1e-4) : 1;
  const farInv = bounds ? 1 / Math.max(bounds[1], 1e-4) : 0;
  const invSpan = Math.max(nearInv - farInv, 1e-6);

  for (let i = 0; i < pixelCount; i++) {
    const out = i * 4;
    const d = depth[i];
    if (mask[i] === 0 || d === undefined || !Number.isFinite(d) || d <= 0 || !bounds) {
      // Leave invalid pixels partially transparent so the backdrop grid stays visible.
      rgba[out] = 18;
      rgba[out + 1] = 20;
      rgba[out + 2] = 26;
      rgba[out + 3] = 96;
      continue;
    }

    const disparity = 1 / Math.max(d, 1e-4);
    const t = 1 - clamp01((disparity - farInv) / invSpan);
    const [r, g, b] = sampleTurboPalette(t);
    rgba[out] = r;
    rgba[out + 1] = g;
    rgba[out + 2] = b;
    rgba[out + 3] = 255;
  }

  return createCanvasFromRgba(rgba, width, height);
}

export function createNormalsPreviewCanvas(
  normals: Float32Array,
  mask: Uint8Array,
  width: number,
  height: number,
): HTMLCanvasElement {
  const pixelCount = width * height;
  const rgba = new Uint8ClampedArray(pixelCount * 4);

  for (let i = 0; i < pixelCount; i++) {
    const inIdx = i * 3;
    const out = i * 4;
    const nx = normals[inIdx];
    const ny = normals[inIdx + 1];
    const nz = normals[inIdx + 2];

    if (
      mask[i] === 0 ||
      nx === undefined ||
      ny === undefined ||
      nz === undefined ||
      !Number.isFinite(nx) ||
      !Number.isFinite(ny) ||
      !Number.isFinite(nz)
    ) {
      rgba[out] = 18;
      rgba[out + 1] = 20;
      rgba[out + 2] = 26;
      rgba[out + 3] = 96;
      continue;
    }

    // Flip Y and Z from OpenCV frame so camera-facing normals appear standard blue-violet.
    rgba[out] = Math.round(clamp01(nx * 0.5 + 0.5) * 255);
    rgba[out + 1] = Math.round(clamp01(-ny * 0.5 + 0.5) * 255);
    rgba[out + 2] = Math.round(clamp01(-nz * 0.5 + 0.5) * 255);
    rgba[out + 3] = 255;
  }

  return createCanvasFromRgba(rgba, width, height);
}

export function createMaskPreviewCanvas(
  mask: Uint8Array,
  width: number,
  height: number,
): HTMLCanvasElement {
  const pixelCount = width * height;
  const rgba = new Uint8ClampedArray(pixelCount * 4);

  for (let i = 0; i < pixelCount; i++) {
    const out = i * 4;
    if (mask[i] !== 0) {
      rgba[out] = 230;
      rgba[out + 1] = 235;
      rgba[out + 2] = 242;
      rgba[out + 3] = 255;
    } else {
      rgba[out] = 24;
      rgba[out + 1] = 26;
      rgba[out + 2] = 32;
      rgba[out + 3] = 255;
    }
  }

  return createCanvasFromRgba(rgba, width, height);
}

export function extractImageColorsAtSize(
  image: HTMLImageElement,
  width: number,
  height: number,
): Uint8ClampedArray {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Failed to create 2D canvas context for color extraction.');
  }
  ctx.drawImage(image, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height).data;
}

export function sampleGeometryDepthAtImagePoint(
  geometry: GeometryData,
  imagePoint: Point2D,
  originalSize: Size2D,
): { depth: number | null; valid: boolean } | null {
  if (
    originalSize.width <= 0 ||
    originalSize.height <= 0 ||
    imagePoint.x < 0 ||
    imagePoint.x >= originalSize.width ||
    imagePoint.y < 0 ||
    imagePoint.y >= originalSize.height
  ) {
    return null;
  }

  const infW = geometry.metadata.inference_width;
  const infH = geometry.metadata.inference_height;
  const col = Math.min(
    infW - 1,
    Math.max(0, Math.floor((imagePoint.x * infW) / originalSize.width)),
  );
  const row = Math.min(
    infH - 1,
    Math.max(0, Math.floor((imagePoint.y * infH) / originalSize.height)),
  );
  const idx = row * infW + col;
  const valid = geometry.mask[idx] !== 0;
  const depthVal = geometry.depth[idx];

  if (!valid || depthVal === undefined || !Number.isFinite(depthVal)) {
    return { depth: null, valid: false };
  }
  return { depth: depthVal, valid: true };
}
