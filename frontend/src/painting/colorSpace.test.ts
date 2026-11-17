import { describe, expect, it } from 'vitest';
import {
  blendSrgbBytes,
  calibrateShadingScale,
  compositeLightingAwarePixel,
  linearChannelToSrgb,
  rgbaUint8ToLinearRgb,
  srgbChannelToLinear,
} from './colorSpace';

describe('colorSpace & lighting-aware compositing', () => {
  it('round-trips sRGB <-> linear RGB cleanly including piecewise toe', () => {
    const samples = [0, 0.01, 0.04045, 0.2, 0.5, 0.8, 1];
    for (const s of samples) {
      const rec = linearChannelToSrgb(srgbChannelToLinear(s));
      expect(rec).toBeCloseTo(s, 5);
    }

    const rgba = new Uint8ClampedArray([0, 128, 255, 255]);
    const lin = rgbaUint8ToLinearRgb(rgba);
    expect(lin[0]).toBeCloseTo(0, 6);
    expect(lin[1]).toBeCloseTo(srgbChannelToLinear(128 / 255), 6);
    expect(lin[2]).toBeCloseTo(1, 6);
  });

  it('calibrates deterministic up-to-scale shading across normal and colored lighting with outlier robustness', () => {
    const width = 20;
    const height = 20;
    const pixelCount = width * height;
    const albedo = new Float32Array(pixelCount * 3);
    const shading = new Float32Array(pixelCount * 3);
    const linearRef = new Float32Array(pixelCount * 3);
    const support = new Uint8Array(pixelCount).fill(1);

    const trueScale = 2.4;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        const base = i * 3;
        const aR = 0.3 + 0.3 * (x / width);
        const aG = 0.35 + 0.2 * (y / height);
        const aB = 0.4;
        // Warm gradient illumination on GPU.
        const sTrueR = (0.4 + 0.7 * (x / width)) * 1.15;
        const sTrueG = (0.4 + 0.7 * (x / width)) * 1.0;
        const sTrueB = (0.4 + 0.7 * (x / width)) * 0.78;

        albedo[base] = aR;
        albedo[base + 1] = aG;
        albedo[base + 2] = aB;
        shading[base] = sTrueR / trueScale;
        shading[base + 1] = sTrueG / trueScale;
        shading[base + 2] = sTrueB / trueScale;
        linearRef[base] = Math.min(1, aR * sTrueR);
        linearRef[base + 1] = Math.min(1, aG * sTrueG);
        linearRef[base + 2] = Math.min(1, aB * sTrueB);
      }
    }

    // Inject clipped highlight corner and dark near-zero albedo block.
    for (let y = 0; y < 3; y++) {
      for (let x = 0; x < 3; x++) {
        const base = (y * width + x) * 3;
        linearRef[base] = 0.99;
        linearRef[base + 1] = 0.99;
        linearRef[base + 2] = 0.99;
      }
    }
    for (let y = height - 3; y < height; y++) {
      for (let x = width - 3; x < width; x++) {
        const base = (y * width + x) * 3;
        albedo[base] = 0.005;
        albedo[base + 1] = 0.005;
        albedo[base + 2] = 0.005;
        linearRef[base] = 0.002;
        linearRef[base + 1] = 0.002;
        linearRef[base + 2] = 0.002;
      }
    }

    const calib = calibrateShadingScale({
      linearReference: linearRef,
      albedo,
      shading,
      appearanceWidth: width,
      appearanceHeight: height,
      supportMask: support,
    });
    expect(calib.isReliable).toBe(true);
    expect(calib.scale).toBeCloseTo(trueScale, 1);

    // Calling again is bit-deterministic and independent of stroke history.
    const calibRepeat = calibrateShadingScale({
      linearReference: linearRef,
      albedo,
      shading,
      appearanceWidth: width,
      appearanceHeight: height,
      supportMask: support,
    });
    expect(calibRepeat.scale).toBe(calib.scale);

    // Empty mask safely marks calibration unreliable with fallback scale 1.0.
    const emptyMask = new Uint8Array(pixelCount);
    const fallback = calibrateShadingScale({
      linearReference: linearRef,
      albedo,
      shading,
      appearanceWidth: width,
      appearanceHeight: height,
      supportMask: emptyMask,
    });
    expect(fallback.isReliable).toBe(false);
    expect(fallback.scale).toBe(1.0);
  });

  it('composites flat, shading-aware, and texture-preserving paint cleanly while keeping unpainted pixels intact', () => {
    const alb: [number, number, number] = [0.42, 0.36, 0.3];
    const albSmooth: [number, number, number] = [0.35, 0.35, 0.35];
    const origLit: [number, number, number] = [0.42, 0.36, 0.3];
    const origShadow: [number, number, number] = [0.105, 0.09, 0.075];
    const paintColor: [number, number, number] = [0.6, 0.25, 0.25];

    // Zero alpha leaves original pixel unmodified.
    const untouched = compositeLightingAwarePixel({
      origLinear: origLit,
      paintLinear: paintColor,
      alpha: 0,
      albedo: alb,
      albedoSmooth: albSmooth,
      shadingRaw: [0.5, 0.5, 0.5],
      shadingScale: 2.0,
      mode: 'shading',
    });
    expect(untouched).toEqual(origLit);

    // Shadowed vs sunlit response under shading mode.
    const outLit = compositeLightingAwarePixel({
      origLinear: origLit,
      paintLinear: paintColor,
      alpha: 1,
      albedo: alb,
      shadingRaw: [0.5, 0.5, 0.5],
      shadingScale: 2.0,
      mode: 'shading',
    });
    const outShadow = compositeLightingAwarePixel({
      origLinear: origShadow,
      paintLinear: paintColor,
      alpha: 1,
      albedo: alb,
      shadingRaw: [0.125, 0.125, 0.125],
      shadingScale: 2.0,
      mode: 'shading',
    });
    expect(outLit[0]).toBeGreaterThan(outShadow[0] * 3);

    // Colored blue illumination shifts red paint darker relative to blue channel.
    const outBlueLight = compositeLightingAwarePixel({
      origLinear: [alb[0] * 0.4, alb[1] * 0.8, alb[2] * 1.4],
      paintLinear: [0.5, 0.5, 0.5],
      alpha: 1,
      albedo: alb,
      shadingRaw: [0.2, 0.4, 0.7],
      shadingScale: 2.0,
      mode: 'shading',
    });
    expect(outBlueLight[2]).toBeGreaterThan(outBlueLight[0] * 2);

    // Texture mode increases output brightness on brighter micro-texture grains.
    const brightGrain = compositeLightingAwarePixel({
      origLinear: [0.45, 0.45, 0.45],
      paintLinear: [0.5, 0.5, 0.5],
      alpha: 1,
      albedo: [0.45, 0.45, 0.45],
      albedoSmooth: [0.35, 0.35, 0.35],
      shadingRaw: [0.5, 0.5, 0.5],
      shadingScale: 2.0,
      mode: 'texture',
      texturePreservation: 1.0,
    });
    const darkGrain = compositeLightingAwarePixel({
      origLinear: [0.25, 0.25, 0.25],
      paintLinear: [0.5, 0.5, 0.5],
      alpha: 1,
      albedo: [0.25, 0.25, 0.25],
      albedoSmooth: [0.35, 0.35, 0.35],
      shadingRaw: [0.5, 0.5, 0.5],
      shadingScale: 2.0,
      mode: 'texture',
      texturePreservation: 1.0,
    });
    expect(brightGrain[0]).toBeGreaterThan(darkGrain[0]);

    // Flat mode ignores shading differences completely.
    const flatLit = compositeLightingAwarePixel({
      origLinear: origLit,
      paintLinear: paintColor,
      alpha: 1,
      albedo: alb,
      shadingRaw: [0.5, 0.5, 0.5],
      shadingScale: 2.0,
      mode: 'flat',
    });
    const flatShadow = compositeLightingAwarePixel({
      origLinear: origShadow,
      paintLinear: paintColor,
      alpha: 1,
      albedo: alb,
      shadingRaw: [0.1, 0.1, 0.1],
      shadingScale: 2.0,
      mode: 'flat',
    });
    expect(flatLit).toEqual(flatShadow);
  });
});

it('blends partial coverage in linear light before encoding to sRGB', () => {
  expect(blendSrgbBytes(255, 0, 0.5, 0.5)).toBe(188);
});
