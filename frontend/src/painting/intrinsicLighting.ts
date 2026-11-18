import { clamp } from '../editor/viewportTransforms';
import type { AppearanceData, Size2D } from '../types/editor';
import {
  calibrateShadingScale,
  DEFAULT_MAX_CALIBRATED_SHADING,
  type ShadingCalibrationResult,
} from './colorSpace';

const MIN_ALBEDO = 0.05;

function quotientWeight(albedo: number, image: number): number {
  return clamp(albedo / MIN_ALBEDO, 0, 1) * clamp((1 - image) / 0.08, 0, 1);
}

export function deriveLighting(
  image: Float32Array,
  albedo: Float32Array,
  width: number,
  height: number,
): { shading: Float32Array; residual: Float32Array } {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    image.length !== width * height * 3 ||
    image.length !== albedo.length
  ) {
    throw new Error('The image and albedo must share the same RGB grid.');
  }

  const quotient = new Float32Array(image.length);
  const strength = new Float32Array(width * height);
  for (let i = 0; i < image.length; i += 3) {
    let product = 0;
    let energy = 0;
    for (let c = i; c < i + 3; c++) {
      const value = image[c] ?? 0;
      const reflectance = albedo[c] ?? 0;
      if (!Number.isFinite(value) || !Number.isFinite(reflectance)) {
        throw new Error('Lighting estimation requires finite RGB values.');
      }
      const w = quotientWeight(reflectance, value);
      product += w * reflectance * value;
      energy += w * reflectance * reflectance;
    }

    const gray = energy >= MIN_ALBEDO ** 2 ? product / energy : NaN;
    strength[i / 3] = gray;
    const fallback = Number.isFinite(gray) ? gray : 1;
    for (let c = i; c < i + 3; c++) {
      const value = image[c] ?? 0;
      const reflectance = albedo[c] ?? 0;
      const weight = quotientWeight(reflectance, value);
      const ratio = value / Math.max(reflectance, MIN_ALBEDO);
      const estimate = fallback + weight * (ratio - fallback);
      quotient[c] = clamp(estimate, 0, DEFAULT_MAX_CALIBRATED_SHADING);
    }
  }

  const shading = quotient.slice();
  const residual = new Float32Array(image.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = y * width + x;
      const i = pixel * 3;
      const brightness = strength[pixel] ?? NaN;
      let count = 0;
      let brightestNeighbor = 0;
      let sumR = 0;
      let sumG = 0;
      let sumB = 0;

      for (let row = Math.max(0, y - 1); row <= Math.min(height - 1, y + 1); row++) {
        for (let col = Math.max(0, x - 1); col <= Math.min(width - 1, x + 1); col++) {
          const other = row * width + col;
          const neighbor = strength[other] ?? NaN;
          if (other === pixel || !Number.isFinite(neighbor)) continue;
          const j = other * 3;
          const dr = (albedo[i] ?? 0) - (albedo[j] ?? 0);
          const dg = (albedo[i + 1] ?? 0) - (albedo[j + 1] ?? 0);
          const db = (albedo[i + 2] ?? 0) - (albedo[j + 2] ?? 0);
          if (dr * dr + dg * dg + db * db > 0.04) continue;
          count++;
          brightestNeighbor = Math.max(brightestNeighbor, neighbor);
          sumR += quotient[j] ?? 0;
          sumG += quotient[j + 1] ?? 0;
          sumB += quotient[j + 2] ?? 0;
        }
      }

      // A bright edge has bright neighbors. Only replace isolated peaks on similar albedo.
      if (
        count >= 5 &&
        (!Number.isFinite(brightness) || brightness > brightestNeighbor * 1.35 + 0.08)
      ) {
        shading[i] = sumR / count;
        shading[i + 1] = sumG / count;
        shading[i + 2] = sumB / count;
      }
      for (let c = i; c < i + 3; c++) {
        residual[c] = (image[c] ?? 0) - (albedo[c] ?? 0) * (shading[c] ?? 0);
      }
    }
  }
  return { shading, residual };
}

export function getAppearanceShadingCalibration(
  appearance: AppearanceData,
  support?: { mask: Uint8Array; size: Size2D },
): ShadingCalibrationResult | null {
  if (!appearance.shading) return null;
  if (appearance.derivedShading) {
    return { scale: 1, isReliable: true, validSampleCount: 0, medianResidual: 0 };
  }
  return calibrateShadingScale({
    linearReference: appearance.linearReference,
    albedo: appearance.albedo,
    shading: appearance.shading,
    residual: appearance.residual,
    appearanceWidth: appearance.metadata.appearance_width,
    appearanceHeight: appearance.metadata.appearance_height,
    supportMask: support?.mask,
    supportWidth: support?.size.width,
    supportHeight: support?.size.height,
  });
}
