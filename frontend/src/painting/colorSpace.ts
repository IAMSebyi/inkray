import { clamp } from '../editor/viewportTransforms';
import type { PaintMode } from '../types/editor';

export const DEFAULT_MAX_CALIBRATED_SHADING = 2.5;

export interface ShadingCalibrationResult {
  scale: number;
  isReliable: boolean;
  validSampleCount: number;
  medianResidual: number;
}

export function srgbChannelToLinear(c: number): number {
  const x = clamp(c, 0, 1);
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

export function linearChannelToSrgb(c: number): number {
  const x = clamp(c, 0, 1);
  const out = x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  return clamp(out, 0, 1);
}

export function rgbaUint8ToLinearRgb(rgba: Uint8ClampedArray): Float32Array {
  const pixelCount = Math.floor(rgba.length / 4);
  const linear = new Float32Array(pixelCount * 3);
  for (let i = 0; i < pixelCount; i++) {
    const src = i * 4;
    const dst = i * 3;
    linear[dst] = srgbChannelToLinear((rgba[src] ?? 0) / 255);
    linear[dst + 1] = srgbChannelToLinear((rgba[src + 1] ?? 0) / 255);
    linear[dst + 2] = srgbChannelToLinear((rgba[src + 2] ?? 0) / 255);
  }
  return linear;
}

function percentileSorted(sorted: readonly number[], pct: number): number {
  if (sorted.length === 0) {
    return 0;
  }
  const idx = (clamp(pct, 0, 100) / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const vLo = sorted[lo] ?? 0;
  const vHi = sorted[hi] ?? vLo;
  return vLo + (vHi - vLo) * (idx - lo);
}

export function calibrateShadingScale(params: {
  linearReference: Float32Array;
  albedo: Float32Array;
  shading: Float32Array;
  residual?: Float32Array | null;
  appearanceWidth: number;
  appearanceHeight: number;
  supportMask?: Uint8Array | null;
  supportWidth?: number;
  supportHeight?: number;
  minAlbedo?: number;
  maxReference?: number;
  minScale?: number;
  maxScale?: number;
}): ShadingCalibrationResult {
  const {
    linearReference,
    albedo,
    shading,
    residual = null,
    appearanceWidth,
    appearanceHeight,
    supportMask = null,
    supportWidth = appearanceWidth,
    supportHeight = appearanceHeight,
    minAlbedo = 0.05,
    maxReference = 0.96,
    minScale = 0.15,
    maxScale = 8.0,
  } = params;

  const collectRatios = (
    strict: boolean,
  ): { ratios: number[]; prods: number[]; targets: number[] } => {
    const ratios: number[] = [];
    const prods: number[] = [];
    const targets: number[] = [];
    const albThresh = strict ? minAlbedo : 0.02;
    const shdThresh = strict ? 1e-3 : 1e-4;
    const refThresh = strict ? 0.02 : 0.01;
    const stride = appearanceWidth * appearanceHeight > 120_000 ? 2 : 1;

    for (let y = 0; y < appearanceHeight; y += stride) {
      const supY = supportMask
        ? Math.min(
            supportHeight - 1,
            Math.floor((y / Math.max(1, appearanceHeight)) * supportHeight),
          )
        : 0;
      for (let x = 0; x < appearanceWidth; x += stride) {
        if (supportMask) {
          const supX = Math.min(
            supportWidth - 1,
            Math.floor((x / Math.max(1, appearanceWidth)) * supportWidth),
          );
          if ((supportMask[supY * supportWidth + supX] ?? 0) === 0) {
            continue;
          }
        }

        const base = (y * appearanceWidth + x) * 3;
        const rRef = linearReference[base] ?? 0;
        const gRef = linearReference[base + 1] ?? 0;
        const bRef = linearReference[base + 2] ?? 0;
        const rAlb = albedo[base] ?? 0;
        const gAlb = albedo[base + 1] ?? 0;
        const bAlb = albedo[base + 2] ?? 0;
        const rShd = shading[base] ?? 0;
        const gShd = shading[base + 1] ?? 0;
        const bShd = shading[base + 2] ?? 0;

        if (
          !Number.isFinite(rRef) ||
          !Number.isFinite(gRef) ||
          !Number.isFinite(bRef) ||
          !Number.isFinite(rAlb) ||
          !Number.isFinite(gAlb) ||
          !Number.isFinite(bAlb) ||
          !Number.isFinite(rShd) ||
          !Number.isFinite(gShd) ||
          !Number.isFinite(bShd)
        ) {
          continue;
        }

        const albMean = (rAlb + gAlb + bAlb) / 3;
        const shdMean = (rShd + gShd + bShd) / 3;
        const refMean = (rRef + gRef + bRef) / 3;
        const refMax = Math.max(rRef, gRef, bRef);
        if (albMean < albThresh || shdMean < shdThresh || refMean < refThresh) {
          continue;
        }
        if (strict && refMax > maxReference) {
          continue;
        }

        const rRes = residual ? clamp(residual[base] ?? 0, 0, 1) : 0;
        const gRes = residual ? clamp(residual[base + 1] ?? 0, 0, 1) : 0;
        const bRes = residual ? clamp(residual[base + 2] ?? 0, 0, 1) : 0;
        const tr = Math.max(0, rRef - rRes);
        const tg = Math.max(0, gRef - gRes);
        const tb = Math.max(0, bRef - bRes);

        const pr = rAlb * rShd;
        const pg = gAlb * gShd;
        const pb = bAlb * bShd;
        const den = pr * pr + pg * pg + pb * pb;
        if (den <= 1e-6) {
          continue;
        }
        const num = tr * pr + tg * pg + tb * pb;
        const ratio = num / den;
        if (!Number.isFinite(ratio) || ratio <= 0) {
          continue;
        }
        ratios.push(ratio);
        prods.push(pr, pg, pb);
        targets.push(tr, tg, tb);
      }
    }
    return { ratios, prods, targets };
  };

  let collected = collectRatios(true);
  if (collected.ratios.length < 16) {
    collected = collectRatios(false);
  }

  const count = collected.ratios.length;
  if (count === 0) {
    return { scale: 1.0, isReliable: false, validSampleCount: 0, medianResidual: 0 };
  }

  const sorted = [...collected.ratios].sort((a, b) => a - b);
  if (count < 8) {
    const rawScale = percentileSorted(sorted, 50);
    const scale = clamp(rawScale, minScale, maxScale);
    return { scale, isReliable: false, validSampleCount: count, medianResidual: 0 };
  }

  const lo = percentileSorted(sorted, 10);
  const hi = percentileSorted(sorted, 90);
  const trimmed = sorted.filter((v) => v >= lo && v <= hi);
  const rawScale = percentileSorted(trimmed.length > 0 ? trimmed : sorted, 50);
  const scale = clamp(rawScale, minScale, maxScale);

  const absErrors: number[] = [];
  for (let i = 0; i < collected.targets.length; i++) {
    const tVal = collected.targets[i] ?? 0;
    const pVal = collected.prods[i] ?? 0;
    absErrors.push(Math.abs(tVal - scale * pVal));
  }
  absErrors.sort((a, b) => a - b);
  return {
    scale,
    isReliable: true,
    validSampleCount: count,
    medianResidual: percentileSorted(absErrors, 50),
  };
}

export function compositeLightingAwarePixel(params: {
  origLinear: [number, number, number];
  paintLinear: [number, number, number];
  alpha: number;
  albedo: [number, number, number];
  albedoSmooth?: [number, number, number];
  shadingRaw: [number, number, number];
  shadingScale: number;
  mode: PaintMode;
  texturePreservation?: number;
  maxCalibratedShading?: number;
}): [number, number, number] {
  const {
    origLinear,
    paintLinear,
    alpha,
    albedo,
    albedoSmooth = albedo,
    shadingRaw,
    shadingScale,
    mode,
    texturePreservation = 1.0,
    maxCalibratedShading = DEFAULT_MAX_CALIBRATED_SHADING,
  } = params;

  const a = clamp(alpha, 0, 1);
  if (a <= 0) {
    return [origLinear[0], origLinear[1], origLinear[2]];
  }
  if (mode === 'flat') {
    return [
      clamp(origLinear[0] * (1 - a) + paintLinear[0] * a, 0, 1),
      clamp(origLinear[1] * (1 - a) + paintLinear[1] * a, 0, 1),
      clamp(origLinear[2] * (1 - a) + paintLinear[2] * a, 0, 1),
    ];
  }

  const tp = mode === 'texture' ? clamp(texturePreservation, 0, 1) : 0;
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const orig = origLinear[c] ?? 0;
    const paint = paintLinear[c] ?? 0;
    const alb = clamp(albedo[c] ?? 0, 0, 1);
    const albSm = Math.max(0.05, clamp(albedoSmooth[c] ?? alb, 0, 1));
    const detail = clamp(alb / albSm, 0.5, 1.5);
    const targetAlb = clamp(paint * (1 - tp + tp * detail), 0, 1);
    const sCal = Math.min(Math.max(0, shadingRaw[c] ?? 0) * shadingScale, maxCalibratedShading);
    out[c] = clamp(orig + a * sCal * (targetAlb - alb), 0, 1);
  }
  return out;
}
