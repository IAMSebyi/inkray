import type { PlaneFootprint, PlaneFrame } from '../geometry/planeFrame';
import { clamp } from '../editor/viewportTransforms';

export interface DecalTextureSpec {
  width: number;
  height: number;
}

export function scaleFootprint(base: PlaneFootprint, extentScale: number): PlaneFootprint {
  const safeScale = clamp(extentScale, 0.5, 2.5);
  const centerU = 0.5 * (base.minU + base.maxU);
  const centerV = 0.5 * (base.minV + base.maxV);
  const halfU = 0.5 * Math.max(1e-4, base.maxU - base.minU) * safeScale;
  const halfV = 0.5 * Math.max(1e-4, base.maxV - base.minV) * safeScale;
  return {
    minU: centerU - halfU,
    maxU: centerU + halfU,
    minV: centerV - halfV,
    maxV: centerV + halfV,
  };
}

export function isInsideFootprint(u: number, v: number, footprint: PlaneFootprint): boolean {
  return u >= footprint.minU && u <= footprint.maxU && v >= footprint.minV && v <= footprint.maxV;
}

export function computeDecalTextureSpec(
  footprint: PlaneFootprint,
  maxEdge = 1024,
  minEdge = 256,
): DecalTextureSpec {
  const spanU = Math.max(1e-5, footprint.maxU - footprint.minU);
  const spanV = Math.max(1e-5, footprint.maxV - footprint.minV);
  if (spanU >= spanV) {
    return {
      width: maxEdge,
      height: clamp(Math.round(maxEdge * (spanV / spanU)), minEdge, maxEdge),
    };
  }
  return {
    width: clamp(Math.round(maxEdge * (spanU / spanV)), minEdge, maxEdge),
    height: maxEdge,
  };
}

export function planeCoordsToDecalUv(
  u: number,
  v: number,
  footprint: PlaneFootprint,
): [number, number] {
  const spanU = Math.max(1e-6, footprint.maxU - footprint.minU);
  const spanV = Math.max(1e-6, footprint.maxV - footprint.minV);
  return [(u - footprint.minU) / spanU, (v - footprint.minV) / spanV];
}

export function decalTexelToPlaneCoords(
  px: number,
  py: number,
  footprint: PlaneFootprint,
  spec: DecalTextureSpec,
): [number, number] {
  const normU = (px + 0.5) / spec.width;
  const normV = (py + 0.5) / spec.height;
  return [
    footprint.minU + normU * (footprint.maxU - footprint.minU),
    footprint.minV + normV * (footprint.maxV - footprint.minV),
  ];
}

export interface SurfaceAccentColor {
  hex: string;
  rgb: [number, number, number];
}

export const SURFACE_PALETTE: readonly SurfaceAccentColor[] = [
  { hex: '#4c8dff', rgb: [76, 141, 255] },
  { hex: '#34d399', rgb: [52, 211, 153] },
  { hex: '#f59e0b', rgb: [245, 158, 11] },
  { hex: '#a78bfa', rgb: [167, 139, 250] },
  { hex: '#f472b6', rgb: [244, 114, 182] },
  { hex: '#22d3ee', rgb: [34, 211, 238] },
];

export function getSurfaceAccentColor(colorIndex: number): SurfaceAccentColor {
  const safeIndex = Math.abs(Math.floor(colorIndex)) % SURFACE_PALETTE.length;
  return (
    SURFACE_PALETTE[safeIndex] ?? SURFACE_PALETTE[0] ?? { hex: '#4c8dff', rgb: [76, 141, 255] }
  );
}

export interface PlanarDecalSurface {
  id: string;
  label: string;
  colorIndex: number;
  frame: PlaneFrame;
  baseFootprint: PlaneFootprint;
  extentScale: number;
  supportMask: Uint8Array;
  inlierCount: number;
  rmsError: number;
  referenceBrushDiameter: number;
}

export function getEffectiveFootprint(surface: PlanarDecalSurface): PlaneFootprint {
  return scaleFootprint(surface.baseFootprint, surface.extentScale);
}
