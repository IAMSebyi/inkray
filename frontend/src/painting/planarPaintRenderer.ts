import * as THREE from 'three';
import { getImageBoundsInViewport } from '../editor/viewportTransforms';
import { parseNormalizedIntrinsics } from '../geometry/geometryAccess';
import type { SurfaceMapping } from '../geometry/localSurfaceParameterizer';
import type { PlaneFootprint } from '../geometry/planeFrame';
import type {
  AppearanceData,
  GeometryData,
  LoadedImage,
  PaintMode,
  Size2D,
  ViewportTransform,
} from '../types/editor';
import {
  DEFAULT_MAX_CALIBRATED_SHADING,
  type ShadingCalibrationResult,
  calibrateShadingScale,
} from './colorSpace';
import { type Artwork, rasterizeArtworkRgba, toPremultipliedRgba } from './artwork';
import type { ArtworkPlacement } from './artworkPlacement';
import {
  type BrushStroke,
  type DirtyTexelBounds,
  compositeStrokeMaskOverRgba,
  rebuildSurfaceDecalRgba,
  stampSegmentIntoStrokeMask,
} from './brushEngine';
import {
  type ActiveImageStrokePreview,
  type ImagePaintLayerKind,
  type ImageSpaceStroke,
  compositeImageStrokeOverRgba,
  computeImagePaintSpec,
  rebuildImageLayerRgba,
} from './imageSpacePaint';
import type { PaintDocumentState } from './paintDocument';
import {
  type DecalTextureSpec,
  type PlanarDecalSurface,
  computeDecalTextureSpec,
  getEffectiveFootprint,
  getSurfaceAccentColor,
} from './planarDecal';
import { computeSurfaceOwnerGrid } from './planeVisibility';

const QUAD_VERTEX_SHADER = `
uniform vec4 uQuadTransform;
varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(
    position.x * uQuadTransform.x + uQuadTransform.z,
    position.y * uQuadTransform.y + uQuadTransform.w,
    0.0,
    1.0
  );
}
`;

const CURVED_ARTWORK_VERTEX_SHADER = `
uniform vec4 uQuadTransform;
attribute vec2 aSurfaceUv;
attribute float aSurfaceDepth;
varying vec2 vUv;
varying vec2 vSurfaceUv;
varying float vSurfaceDepth;

void main() {
  vUv = uv;
  vSurfaceUv = aSurfaceUv;
  vSurfaceDepth = aSurfaceDepth;
  gl_Position = vec4(
    position.x * uQuadTransform.x + uQuadTransform.z,
    position.y * uQuadTransform.y + uQuadTransform.w,
    0.0,
    1.0
  );
}
`;

const BASE_FRAGMENT_SHADER = `
uniform sampler2D uRgbTex;
uniform sampler2D uSupportTex;
uniform float uIsPng;
uniform float uCompositeChecker;
uniform float uShowSupport;
uniform vec2 uImageCssSize;
varying vec2 vUv;

void main() {
  vec4 base = texture2D(uRgbTex, vUv);
  vec4 outColor = base;

  if (uCompositeChecker > 0.5 && uIsPng > 0.5 && base.a < 0.999) {
    vec2 px = vUv * uImageCssSize;
    float cx = floor(px.x / 8.0);
    float cy = floor(px.y / 8.0);
    float tile = mod(cx + cy, 2.0);
    vec3 c0 = vec3(0.106, 0.114, 0.137);
    vec3 c1 = vec3(0.141, 0.157, 0.192);
    vec3 checker = mix(c0, c1, tile);
    outColor = vec4(mix(checker, base.rgb, base.a), 1.0);
  }

  if (uShowSupport > 0.5) {
    vec2 uvImg = vec2(vUv.x, 1.0 - vUv.y);
    vec4 sup = texture2D(uSupportTex, uvImg);
    if (sup.a > 0.01) {
      outColor.rgb = mix(outColor.rgb, sup.rgb, sup.a * 0.28);
    }
  }

  gl_FragColor = vec4(outColor.rgb * outColor.a, outColor.a);
}
`;

const SHARED_PAINT_FRAGMENT_PREFIX = `
uniform sampler2D uRgbTex;
uniform sampler2D uSupportTex;
uniform sampler2D uDepthTex;
uniform sampler2D uMaskTex;
uniform sampler2D uNormalsTex;
uniform sampler2D uOwnerTex;
uniform sampler2D uDecalTex;
uniform sampler2D uAlbedoTex;
uniform sampler2D uShadingTex;
uniform float uIsPng;
uniform float uCompositeChecker;
uniform float uShowSupport;
uniform vec2 uImageCssSize;
uniform vec2 uAppearanceSize;
uniform int uEffectivePaintMode;
uniform float uShadingScale;
uniform float uMaxCalibratedShading;
uniform float uTexturePreservation;
uniform float uHasNormals;
uniform float uShowFootprint;
uniform float uUseOwnerMask;
uniform float uSurfaceIndex;
uniform vec3 uSurfaceAccent;
uniform vec4 uIntrinsics0;
uniform float uCy;
uniform vec3 uPlaneOrigin;
uniform vec3 uPlaneNormal;
uniform vec3 uPlaneU;
uniform vec3 uPlaneV;
uniform vec4 uFootprint;
uniform float uDepthTolRatio;
uniform float uFitRms;
uniform float uMinNormalDot;
uniform float uFeatherRatio;
uniform float uHasDecalPaint;
uniform int uLayerKind;
uniform float uIsArtwork;
uniform vec2 uImagePixelSize;
uniform vec2 uArtCenter;
uniform vec2 uArtSize;
uniform vec2 uArtRot;
uniform float uArtworkOpacity;
uniform float uPremultipliedPaint;
varying vec2 vUv;

float computeScaleAwareTol(float zPlane) {
  float baseTol = max(uDepthTolRatio * zPlane, 2.5 * uFitRms);
  return clamp(baseTol, 0.015 * zPlane, 0.14 * zPlane);
}

vec3 srgbToLinearVec3(vec3 c) {
  vec3 x = clamp(c, 0.0, 1.0);
  vec3 low = x / 12.92;
  vec3 high = pow((x + vec3(0.055)) / 1.055, vec3(2.4));
  return vec3(
    x.r <= 0.04045 ? low.r : high.r,
    x.g <= 0.04045 ? low.g : high.g,
    x.b <= 0.04045 ? low.b : high.b
  );
}

vec3 linearToSrgbVec3(vec3 c) {
  vec3 x = clamp(c, 0.0, 1.0);
  vec3 low = x * 12.92;
  vec3 high = vec3(1.055) * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - vec3(0.055);
  return clamp(vec3(
    x.r <= 0.0031308 ? low.r : high.r,
    x.g <= 0.0031308 ? low.g : high.g,
    x.b <= 0.0031308 ? low.b : high.b
  ), 0.0, 1.0);
}

vec3 sampleBilinearFloat3(sampler2D tex, vec2 uvImg, vec2 texSize) {
  vec2 clampedUv = clamp(uvImg, vec2(0.0), vec2(1.0));
  vec2 pos = clampedUv * texSize - vec2(0.5);
  vec2 baseIdx = floor(pos);
  vec2 f = clamp(pos - baseIdx, vec2(0.0), vec2(1.0));
  vec2 maxIdx = max(texSize - vec2(1.0), vec2(0.0));
  vec2 idx0 = clamp(baseIdx, vec2(0.0), maxIdx);
  vec2 idx1 = clamp(baseIdx + vec2(1.0), vec2(0.0), maxIdx);
  vec2 invSize = 1.0 / max(texSize, vec2(1.0));

  vec3 c00 = texture2D(tex, (vec2(idx0.x, idx0.y) + 0.5) * invSize).rgb;
  vec3 c10 = texture2D(tex, (vec2(idx1.x, idx0.y) + 0.5) * invSize).rgb;
  vec3 c01 = texture2D(tex, (vec2(idx0.x, idx1.y) + 0.5) * invSize).rgb;
  vec3 c11 = texture2D(tex, (vec2(idx1.x, idx1.y) + 0.5) * invSize).rgb;
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

vec3 sampleAlbedoSmooth5x5(sampler2D tex, vec2 uvImg, vec2 texSize, vec3 centerAlb) {
  float stepPx = max(1.0, floor(min(texSize.x, texSize.y) / 40.0));
  vec2 invSize = stepPx / max(texSize, vec2(1.0));
  vec3 sumAlb = vec3(0.0);
  float sumW = 0.0;
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      vec2 offUv = clamp(uvImg + vec2(float(dx), float(dy)) * invSize, vec2(0.0), vec2(1.0));
      vec3 nbr = clamp(sampleBilinearFloat3(tex, offUv, texSize), 0.0, 1.0);
      vec3 d = nbr - centerAlb;
      float w = exp(-dot(d, d) / 0.25);
      sumAlb += nbr * w;
      sumW += w;
    }
  }
  return sumAlb / max(sumW, 1e-4);
}

vec4 computeBasePixel() {
  vec4 base = texture2D(uRgbTex, vUv);
  vec4 outColor = base;
  if (uCompositeChecker > 0.5 && uIsPng > 0.5 && base.a < 0.999) {
    vec2 px = vUv * uImageCssSize;
    float cx = floor(px.x / 8.0);
    float cy = floor(px.y / 8.0);
    float tile = mod(cx + cy, 2.0);
    vec3 c0 = vec3(0.106, 0.114, 0.137);
    vec3 c1 = vec3(0.141, 0.157, 0.192);
    vec3 checker = mix(c0, c1, tile);
    outColor = vec4(mix(checker, base.rgb, base.a), 1.0);
  }
  return outColor;
}

vec3 computeLitTargetRgb(vec2 uvImg, vec3 paintRgb) {
  if (uEffectivePaintMode == 0) {
    return paintRgb;
  }
  vec4 origColor = computeBasePixel();
  if (origColor.a <= 0.001) {
    return paintRgb;
  }
  vec3 origLin = srgbToLinearVec3(origColor.rgb);
  vec3 paintLin = srgbToLinearVec3(paintRgb);
  vec3 alb = clamp(sampleBilinearFloat3(uAlbedoTex, uvImg, uAppearanceSize), 0.0, 1.0);
  vec3 shdRaw = max(sampleBilinearFloat3(uShadingTex, uvImg, uAppearanceSize), vec3(0.0));
  vec3 sCal = min(shdRaw * uShadingScale, vec3(uMaxCalibratedShading));
  vec3 targetAlb = paintLin;
  if (uEffectivePaintMode == 2 && uTexturePreservation > 0.001) {
    vec3 albSmooth = sampleAlbedoSmooth5x5(uAlbedoTex, uvImg, uAppearanceSize, alb);
    vec3 detail = clamp(alb / max(albSmooth, vec3(0.05)), 0.5, 1.5);
    targetAlb = clamp(paintLin * mix(vec3(1.0), detail, clamp(uTexturePreservation, 0.0, 1.0)), 0.0, 1.0);
  }
  vec3 litFullLin = clamp(origLin + sCal * (targetAlb - alb), 0.0, 1.0);
  return linearToSrgbVec3(litFullLin);
}

vec3 sampleStraightPaintRgb(vec4 rawSample) {
  if (uPremultipliedPaint > 0.5) {
    return rawSample.a > 1e-4 ? clamp(rawSample.rgb / rawSample.a, 0.0, 1.0) : vec3(0.0);
  }
  return rawSample.rgb;
}
`;

const DECAL_FRAGMENT_SHADER = `${SHARED_PAINT_FRAGMENT_PREFIX}
void main() {
  vec2 uvImg = vec2(vUv.x, 1.0 - vUv.y);
  if (uLayerKind == 1 || uLayerKind == 2) {
    vec4 paint = texture2D(uDecalTex, uvImg);
    if (paint.a <= 0.001) {
      discard;
    }
    vec3 paintRgb = sampleStraightPaintRgb(paint);
    vec3 outRgb = uLayerKind == 1 ? computeLitTargetRgb(uvImg, paintRgb) : paintRgb;
    gl_FragColor = vec4(outRgb * paint.a, paint.a);
    return;
  }

  if (uLayerKind == 3) {
    vec2 imgPx = uvImg * uImagePixelSize;
    vec2 dImg = imgPx - uArtCenter;
    float rx = dImg.x * uArtRot.x + dImg.y * uArtRot.y;
    float ry = -dImg.x * uArtRot.y + dImg.y * uArtRot.x;
    vec2 artUv = vec2(rx / max(1e-6, uArtSize.x) + 0.5, ry / max(1e-6, uArtSize.y) + 0.5);
    if (artUv.x < 0.0 || artUv.x > 1.0 || artUv.y < 0.0 || artUv.y > 1.0) {
      discard;
    }
    vec4 raw = texture2D(uDecalTex, artUv);
    float paintAlpha = clamp(raw.a * uArtworkOpacity, 0.0, 1.0);
    if (paintAlpha <= 0.001) {
      discard;
    }
    vec3 paintRgb = sampleStraightPaintRgb(raw);
    vec3 outRgb = computeLitTargetRgb(uvImg, paintRgb);
    gl_FragColor = vec4(outRgb * paintAlpha, paintAlpha);
    return;
  }

  if (uUseOwnerMask > 0.5 && uIsArtwork < 0.5) {
    float ownerId = floor(texture2D(uOwnerTex, uvImg).r * 255.0 + 0.5);
    if (ownerId > 0.5 && abs(ownerId - uSurfaceIndex) > 0.5) {
      discard;
    }
  }

  float yNorm = (uvImg.y - uCy) / uIntrinsics0.w;
  float xNorm = (uvImg.x - uIntrinsics0.z - uIntrinsics0.y * yNorm) / uIntrinsics0.x;
  vec3 rayDir = vec3(xNorm, yNorm, 1.0);

  float denom = dot(uPlaneNormal, rayDir);
  if (abs(denom) < 1e-6) {
    discard;
  }

  float t = dot(uPlaneNormal, uPlaneOrigin) / denom;
  if (t <= 1e-4) {
    discard;
  }

  vec3 hitPoint = rayDir * t;
  vec3 delta = hitPoint - uPlaneOrigin;
  float uLocal = dot(delta, uPlaneU);
  float vLocal = dot(delta, uPlaneV);

  vec2 decalCoord = vec2(0.0);
  float spanU = 1.0;
  float spanV = 1.0;

  if (uIsArtwork > 0.5) {
    vec2 dLocal = vec2(uLocal, vLocal) - uArtCenter;
    float rx = dLocal.x * uArtRot.x + dLocal.y * uArtRot.y;
    float ry = -dLocal.x * uArtRot.y + dLocal.y * uArtRot.x;
    decalCoord = vec2(rx / max(1e-6, uArtSize.x) + 0.5, ry / max(1e-6, uArtSize.y) + 0.5);
    if (decalCoord.x < 0.0 || decalCoord.x > 1.0 || decalCoord.y < 0.0 || decalCoord.y > 1.0) {
      discard;
    }
  } else {
    if (
      uLocal < uFootprint.x ||
      uLocal > uFootprint.y ||
      vLocal < uFootprint.z ||
      vLocal > uFootprint.w
    ) {
      discard;
    }
    spanU = max(1e-6, uFootprint.y - uFootprint.x);
    spanV = max(1e-6, uFootprint.w - uFootprint.z);
    decalCoord = vec2(
      (uLocal - uFootprint.x) / spanU,
      (vLocal - uFootprint.z) / spanV
    );
  }

  // Evaluate visibility identically to PlaneVisibility.evaluatePixel.
  float obsValid = texture2D(uMaskTex, uvImg).r;
  float obsDepth = texture2D(uDepthTex, uvImg).r;
  float visibility = 0.0;

  if (obsValid > 0.5 && obsDepth > 0.0) {
    float tol = computeScaleAwareTol(t);
    float diff = obsDepth - t;
    float absDiff = abs(diff);
    if (diff >= -tol && diff <= tol) {
      bool normalOk = true;
      if (uHasNormals > 0.5) {
        vec3 obsNormal = texture2D(uNormalsTex, uvImg).xyz;
        float nLen = length(obsNormal);
        if (nLen > 1e-4 && dot(obsNormal / nLen, uPlaneNormal) < uMinNormalDot) {
          normalOk = false;
        }
      }
      if (normalOk) {
        float innerTol = tol * (1.0 - uFeatherRatio);
        float outerTol = tol * (1.0 + uFeatherRatio * 0.35);
        visibility = uFeatherRatio > 0.0
          ? 1.0 - smoothstep(innerTol, outerTol, absDiff)
          : 1.0;
      }
    }
  }

  if (uIsArtwork > 0.5) {
    if (visibility <= 0.001) {
      discard;
    }
    vec4 raw = texture2D(uDecalTex, decalCoord);
    float paintAlpha = clamp(raw.a * uArtworkOpacity, 0.0, 1.0) * visibility;
    if (paintAlpha <= 0.001) {
      discard;
    }
    vec3 paintRgb = sampleStraightPaintRgb(raw);
    vec3 outRgb = computeLitTargetRgb(uvImg, paintRgb);
    gl_FragColor = vec4(outRgb * paintAlpha, paintAlpha);
    return;
  }

  float paintAlpha = 0.0;
  vec3 paintRgb = vec3(0.0);
  if (uHasDecalPaint > 0.5 && visibility > 0.001) {
    // Decal DataTexture uses flipY = false so row 0 (minV) maps directly to decalCoord.y = 0.
    vec4 paint = texture2D(uDecalTex, decalCoord);
    paintAlpha = paint.a * visibility;
    paintRgb = sampleStraightPaintRgb(paint);
  }

  bool showBorder = false;
  bool showTint = false;
  if (uShowFootprint > 0.5) {
    float minSpan = min(spanU, spanV);
    float borderU = min(decalCoord.x, 1.0 - decalCoord.x) * (spanU / minSpan);
    float borderV = min(decalCoord.y, 1.0 - decalCoord.y) * (spanV / minSpan);
    if (min(borderU, borderV) < 0.014) {
      showBorder = true;
    } else if (paintAlpha <= 0.001 && visibility > 0.5) {
      showTint = true;
    }
  }

  if (paintAlpha <= 0.001 && !showBorder && !showTint) {
    discard;
  }

  vec3 outRgb = paintAlpha > 0.001 ? computeLitTargetRgb(uvImg, paintRgb) : vec3(0.0);
  float outAlpha = paintAlpha;

  if (showBorder) {
    float lineAlpha = visibility > 0.5 ? 0.85 : 0.25;
    vec3 lineCol = visibility > 0.5 ? uSurfaceAccent : vec3(0.85, 0.65, 0.25);
    outRgb = paintAlpha > 0.001 ? mix(outRgb, lineCol, lineAlpha) : lineCol;
    outAlpha = max(outAlpha, lineAlpha);
  } else if (showTint) {
    outRgb = uSurfaceAccent;
    outAlpha = 0.06;
  }

  gl_FragColor = vec4(outRgb * outAlpha, outAlpha);
}
`;

const CURVED_ARTWORK_FRAGMENT_SHADER = `${SHARED_PAINT_FRAGMENT_PREFIX}
varying vec2 vSurfaceUv;
varying float vSurfaceDepth;

void main() {
  vec2 uvImg = vec2(vUv.x, 1.0 - vUv.y);
  vec2 dLocal = vSurfaceUv - uArtCenter;
  float rx = dLocal.x * uArtRot.x + dLocal.y * uArtRot.y;
  float ry = -dLocal.x * uArtRot.y + dLocal.y * uArtRot.x;
  vec2 artUv = vec2(rx / max(1e-6, uArtSize.x) + 0.5, ry / max(1e-6, uArtSize.y) + 0.5);
  if (artUv.x < 0.0 || artUv.x > 1.0 || artUv.y < 0.0 || artUv.y > 1.0) {
    discard;
  }

  float obsValid = texture2D(uMaskTex, uvImg).r;
  float obsDepth = texture2D(uDepthTex, uvImg).r;
  if (obsValid <= 0.5 || obsDepth <= 0.0) {
    discard;
  }

  float tol = max(0.015, vSurfaceDepth * 0.055);
  float diff = obsDepth - vSurfaceDepth;
  if (diff < -tol || diff > tol * 2.0) {
    discard;
  }

  vec4 raw = texture2D(uDecalTex, artUv);
  float paintAlpha = clamp(raw.a * uArtworkOpacity, 0.0, 1.0);
  if (paintAlpha <= 0.001) {
    discard;
  }
  vec3 paintRgb = sampleStraightPaintRgb(raw);
  vec3 outRgb = computeLitTargetRgb(uvImg, paintRgb);
  gl_FragColor = vec4(outRgb * paintAlpha, paintAlpha);
}
`;

interface DecalGpuCacheEntry {
  spec: DecalTextureSpec;
  footprint: PlaneFootprint;
  footprintKey: string;
  committedSignature: string;
  committedRgba: Uint8ClampedArray;
  displayRgba: Uint8Array;
  texture: THREE.DataTexture;
  hasCommittedPaint: boolean;
  activeStrokeId: string | null;
  activeStrokePointCount: number;
  activeMask: Uint8Array | null;
}

interface ImageLayerGpuCacheEntry {
  width: number;
  height: number;
  committedSignature: string;
  committedRgba: Uint8ClampedArray;
  displayRgba: Uint8Array;
  texture: THREE.DataTexture;
  hasCommittedPaint: boolean;
  activePreviewId: string | null;
  activePreviewRevision: number;
}

interface ArtworkGpuCacheEntry {
  revision: number;
  width: number;
  height: number;
  texture: THREE.DataTexture;
}

function createEmptyTexture1x1(): THREE.DataTexture {
  const tex = new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1, THREE.RGBAFormat);
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function buildFootprintKey(fp: PlaneFootprint): string {
  return `${fp.minU.toFixed(6)}:${fp.maxU.toFixed(6)}:${fp.minV.toFixed(6)}:${fp.maxV.toFixed(6)}`;
}

function buildCommittedSignature(strokes: readonly BrushStroke[]): string {
  return strokes.map((s) => `${s.id}:${s.points.length}`).join('|');
}

export function computeViewportQuadTransform(
  viewportSize: Size2D,
  imageSize: Size2D,
  transform: ViewportTransform,
): [number, number, number, number] {
  const bounds = getImageBoundsInViewport(imageSize, transform);
  const ndcLeft = (bounds.x / viewportSize.width) * 2 - 1;
  const ndcRight = ((bounds.x + bounds.width) / viewportSize.width) * 2 - 1;
  const ndcTop = 1 - (bounds.y / viewportSize.height) * 2;
  const ndcBottom = 1 - ((bounds.y + bounds.height) / viewportSize.height) * 2;

  return [
    0.5 * (ndcRight - ndcLeft),
    0.5 * (ndcTop - ndcBottom),
    0.5 * (ndcLeft + ndcRight),
    0.5 * (ndcTop + ndcBottom),
  ];
}

export function flipRgbaRowsVertically(
  bottomUp: Uint8Array,
  width: number,
  height: number,
  unpremultiply = false,
): Uint8ClampedArray {
  const topDown = new Uint8ClampedArray(width * height * 4);
  const rowStride = width * 4;
  for (let y = 0; y < height; y++) {
    const srcRow = (height - 1 - y) * rowStride;
    const dstRow = y * rowStride;
    if (!unpremultiply) {
      topDown.set(bottomUp.subarray(srcRow, srcRow + rowStride), dstRow);
      continue;
    }
    for (let x = 0; x < width; x++) {
      const s = srcRow + x * 4;
      const d = dstRow + x * 4;
      const a = bottomUp[s + 3] ?? 0;
      if (a === 255 || a === 0) {
        topDown[d] = bottomUp[s] ?? 0;
        topDown[d + 1] = bottomUp[s + 1] ?? 0;
        topDown[d + 2] = bottomUp[s + 2] ?? 0;
        topDown[d + 3] = a;
      } else {
        const inv = 255 / a;
        topDown[d] = Math.min(255, Math.round((bottomUp[s] ?? 0) * inv));
        topDown[d + 1] = Math.min(255, Math.round((bottomUp[s + 1] ?? 0) * inv));
        topDown[d + 2] = Math.min(255, Math.round((bottomUp[s + 2] ?? 0) * inv));
        topDown[d + 3] = a;
      }
    }
  }
  return topDown;
}

export class PlanarPaintRenderer {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.OrthographicCamera;
  private readonly quadGeometry: THREE.PlaneGeometry;
  private readonly baseMaterial: THREE.ShaderMaterial;
  private readonly overlayMaterial: THREE.ShaderMaterial;
  private readonly curvedArtworkMaterial: THREE.ShaderMaterial;
  private readonly mesh: THREE.Mesh;
  private readonly fallbackTex: THREE.DataTexture = createEmptyTexture1x1();

  private readonly uBaseQuadTransform = new THREE.Uniform(new THREE.Vector4(1, 1, 0, 0));
  private readonly uBaseRgbTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uBaseSupportTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uBaseIsPng = new THREE.Uniform(0);
  private readonly uBaseCompositeChecker = new THREE.Uniform(1);
  private readonly uBaseShowSupport = new THREE.Uniform(0);
  private readonly uBaseImageCssSize = new THREE.Uniform(new THREE.Vector2(1, 1));

  private readonly uDecalQuadTransform = new THREE.Uniform(new THREE.Vector4(1, 1, 0, 0));
  private readonly uDecalRgbTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalSupportTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalDepthTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalMaskTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalNormalsTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalOwnerTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalPaintTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalAlbedoTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalShadingTex = new THREE.Uniform<THREE.Texture>(this.fallbackTex);
  private readonly uDecalIsPng = new THREE.Uniform(0);
  private readonly uDecalCompositeChecker = new THREE.Uniform(1);
  private readonly uDecalShowSupport = new THREE.Uniform(0);
  private readonly uDecalImageCssSize = new THREE.Uniform(new THREE.Vector2(1, 1));
  private readonly uDecalAppearanceSize = new THREE.Uniform(new THREE.Vector2(1, 1));
  private readonly uDecalEffectivePaintMode = new THREE.Uniform(0);
  private readonly uDecalShadingScale = new THREE.Uniform(1.0);
  private readonly uDecalMaxCalibratedShading = new THREE.Uniform(DEFAULT_MAX_CALIBRATED_SHADING);
  private readonly uDecalTexturePreservation = new THREE.Uniform(0.75);
  private readonly uDecalHasNormals = new THREE.Uniform(0);
  private readonly uDecalShowFootprint = new THREE.Uniform(0);
  private readonly uDecalUseOwnerMask = new THREE.Uniform(0);
  private readonly uDecalSurfaceIndex = new THREE.Uniform(1);
  private readonly uDecalSurfaceAccent = new THREE.Uniform(new THREE.Vector3(0.298, 0.553, 1.0));
  private readonly uDecalIntrinsics0 = new THREE.Uniform(new THREE.Vector4(1, 0, 0.5, 1));
  private readonly uDecalCy = new THREE.Uniform(0.5);
  private readonly uDecalPlaneOrigin = new THREE.Uniform(new THREE.Vector3());
  private readonly uDecalPlaneNormal = new THREE.Uniform(new THREE.Vector3(0, 0, -1));
  private readonly uDecalPlaneU = new THREE.Uniform(new THREE.Vector3(1, 0, 0));
  private readonly uDecalPlaneV = new THREE.Uniform(new THREE.Vector3(0, 1, 0));
  private readonly uDecalFootprint = new THREE.Uniform(new THREE.Vector4(-1, 1, -1, 1));
  private readonly uDecalDepthTolRatio = new THREE.Uniform(0.035);
  private readonly uDecalFitRms = new THREE.Uniform(0);
  private readonly uDecalMinNormalDot = new THREE.Uniform(0.45);
  private readonly uDecalHasPaint = new THREE.Uniform(0);
  private readonly uDecalLayerKind = new THREE.Uniform(0);
  private readonly uDecalIsArtwork = new THREE.Uniform(0);
  private readonly uDecalImagePixelSize = new THREE.Uniform(new THREE.Vector2(1, 1));
  private readonly uDecalArtCenter = new THREE.Uniform(new THREE.Vector2(0, 0));
  private readonly uDecalArtSize = new THREE.Uniform(new THREE.Vector2(1, 1));
  private readonly uDecalArtRot = new THREE.Uniform(new THREE.Vector2(1, 0));
  private readonly uDecalArtworkOpacity = new THREE.Uniform(1.0);
  private readonly uDecalPremultipliedPaint = new THREE.Uniform(0);

  private uploadedImageId: string | null = null;
  private rgbTex: THREE.Texture | null = null;
  private rgbIsNearest = false;
  private rgbIsPng = false;

  private uploadedGeometryId: string | null = null;
  private depthTex: THREE.DataTexture | null = null;
  private maskTex: THREE.DataTexture | null = null;
  private normalsTex: THREE.DataTexture | null = null;
  private geometryWidth = 0;
  private geometryHeight = 0;

  private uploadedAppearanceId: string | null = null;
  private albedoTex: THREE.DataTexture | null = null;
  private shadingTex: THREE.DataTexture | null = null;
  private appearanceWidth = 0;
  private appearanceHeight = 0;
  private readonly surfaceCalibrations = new Map<string, ShadingCalibrationResult>();
  private readonly sceneCalibrations = new Map<string, ShadingCalibrationResult>();

  private uploadedOwnerKey: string | null = null;
  private cachedOwnerGrid: Uint8Array | null = null;
  private ownerTex: THREE.DataTexture | null = null;

  private uploadedSupportKey: string | null = null;
  private supportTex: THREE.DataTexture | null = null;

  private readonly decalEntries = new Map<string, DecalGpuCacheEntry>();
  private readonly artworkEntries = new Map<string, ArtworkGpuCacheEntry>();
  private readonly curvedMappingGeometries = new Map<string, THREE.BufferGeometry>();
  private readonly imageLayerEntries = new Map<string, ImageLayerGpuCacheEntry>();
  private disposed = false;

  constructor(hostContainer: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    this.renderer.autoClear = false;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.domElement.className = 'paint-webgl-canvas';
    hostContainer.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quadGeometry = new THREE.PlaneGeometry(2, 2);

    this.baseMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERTEX_SHADER,
      fragmentShader: BASE_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
      transparent: false,
      blending: THREE.NoBlending,
      uniforms: {
        uQuadTransform: this.uBaseQuadTransform,
        uRgbTex: this.uBaseRgbTex,
        uSupportTex: this.uBaseSupportTex,
        uIsPng: this.uBaseIsPng,
        uCompositeChecker: this.uBaseCompositeChecker,
        uShowSupport: this.uBaseShowSupport,
        uImageCssSize: this.uBaseImageCssSize,
      },
    });

    const decalUniforms = {
      uQuadTransform: this.uDecalQuadTransform,
      uRgbTex: this.uDecalRgbTex,
      uSupportTex: this.uDecalSupportTex,
      uDepthTex: this.uDecalDepthTex,
      uMaskTex: this.uDecalMaskTex,
      uNormalsTex: this.uDecalNormalsTex,
      uOwnerTex: this.uDecalOwnerTex,
      uDecalTex: this.uDecalPaintTex,
      uAlbedoTex: this.uDecalAlbedoTex,
      uShadingTex: this.uDecalShadingTex,
      uIsPng: this.uDecalIsPng,
      uCompositeChecker: this.uDecalCompositeChecker,
      uShowSupport: this.uDecalShowSupport,
      uImageCssSize: this.uDecalImageCssSize,
      uAppearanceSize: this.uDecalAppearanceSize,
      uEffectivePaintMode: this.uDecalEffectivePaintMode,
      uShadingScale: this.uDecalShadingScale,
      uMaxCalibratedShading: this.uDecalMaxCalibratedShading,
      uTexturePreservation: this.uDecalTexturePreservation,
      uHasNormals: this.uDecalHasNormals,
      uShowFootprint: this.uDecalShowFootprint,
      uUseOwnerMask: this.uDecalUseOwnerMask,
      uSurfaceIndex: this.uDecalSurfaceIndex,
      uSurfaceAccent: this.uDecalSurfaceAccent,
      uIntrinsics0: this.uDecalIntrinsics0,
      uCy: this.uDecalCy,
      uPlaneOrigin: this.uDecalPlaneOrigin,
      uPlaneNormal: this.uDecalPlaneNormal,
      uPlaneU: this.uDecalPlaneU,
      uPlaneV: this.uDecalPlaneV,
      uFootprint: this.uDecalFootprint,
      uDepthTolRatio: this.uDecalDepthTolRatio,
      uFitRms: this.uDecalFitRms,
      uMinNormalDot: this.uDecalMinNormalDot,
      uFeatherRatio: new THREE.Uniform(0.15),
      uHasDecalPaint: this.uDecalHasPaint,
      uLayerKind: this.uDecalLayerKind,
      uIsArtwork: this.uDecalIsArtwork,
      uImagePixelSize: this.uDecalImagePixelSize,
      uArtCenter: this.uDecalArtCenter,
      uArtSize: this.uDecalArtSize,
      uArtRot: this.uDecalArtRot,
      uArtworkOpacity: this.uDecalArtworkOpacity,
      uPremultipliedPaint: this.uDecalPremultipliedPaint,
    };

    this.overlayMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERTEX_SHADER,
      fragmentShader: DECAL_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendEquationAlpha: THREE.AddEquation,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      uniforms: decalUniforms,
    });

    this.curvedArtworkMaterial = new THREE.ShaderMaterial({
      vertexShader: CURVED_ARTWORK_VERTEX_SHADER,
      fragmentShader: CURVED_ARTWORK_FRAGMENT_SHADER,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendEquationAlpha: THREE.AddEquation,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
      uniforms: decalUniforms,
    });

    this.mesh = new THREE.Mesh(this.quadGeometry, this.baseMaterial);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
  }

  private ensureImageUploaded(image: LoadedImage): void {
    if (this.uploadedImageId === image.id && this.rgbTex) {
      return;
    }
    this.rgbTex?.dispose();
    const tex = new THREE.Texture(image.element);
    tex.flipY = true;
    tex.colorSpace = THREE.NoColorSpace;
    tex.generateMipmaps = false;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;

    this.rgbTex = tex;
    this.rgbIsNearest = false;
    this.rgbIsPng = image.metadata.formatLabel === 'PNG';
    this.uploadedImageId = image.id;
  }

  private ensureGeometryUploaded(geometry: GeometryData): void {
    if (this.uploadedGeometryId === geometry.metadata.id && this.depthTex && this.maskTex) {
      return;
    }
    this.depthTex?.dispose();
    this.maskTex?.dispose();
    this.normalsTex?.dispose();

    const w = geometry.metadata.inference_width;
    const h = geometry.metadata.inference_height;
    const pixelCount = w * h;

    // Sanitize non-finite MoGe depths to 0.0 so GLSL never encounters Inf or NaN.
    const cleanDepth = new Float32Array(pixelCount);
    const cleanMask = new Uint8Array(pixelCount);
    for (let i = 0; i < pixelCount; i++) {
      const m = (geometry.mask[i] ?? 0) !== 0;
      const d = geometry.depth[i] ?? 0;
      if (m && Number.isFinite(d) && d > 0) {
        cleanMask[i] = 255;
        cleanDepth[i] = d;
      }
    }

    const depthTex = new THREE.DataTexture(cleanDepth, w, h, THREE.RedFormat, THREE.FloatType);
    depthTex.colorSpace = THREE.NoColorSpace;
    depthTex.flipY = false;
    depthTex.unpackAlignment = 1;
    depthTex.minFilter = THREE.NearestFilter;
    depthTex.magFilter = THREE.NearestFilter;
    depthTex.wrapS = THREE.ClampToEdgeWrapping;
    depthTex.wrapT = THREE.ClampToEdgeWrapping;
    depthTex.needsUpdate = true;

    const maskTex = new THREE.DataTexture(cleanMask, w, h, THREE.RedFormat, THREE.UnsignedByteType);
    maskTex.colorSpace = THREE.NoColorSpace;
    maskTex.flipY = false;
    maskTex.unpackAlignment = 1;
    maskTex.minFilter = THREE.NearestFilter;
    maskTex.magFilter = THREE.NearestFilter;
    maskTex.wrapS = THREE.ClampToEdgeWrapping;
    maskTex.wrapT = THREE.ClampToEdgeWrapping;
    maskTex.needsUpdate = true;

    let normalsTex: THREE.DataTexture | null = null;
    if (geometry.normals) {
      const rgbaNormals = new Float32Array(pixelCount * 4);
      for (let i = 0; i < pixelCount; i++) {
        const src = i * 3;
        const dst = i * 4;
        const nx = geometry.normals[src] ?? 0;
        const ny = geometry.normals[src + 1] ?? 0;
        const nz = geometry.normals[src + 2] ?? -1;
        if (Number.isFinite(nx) && Number.isFinite(ny) && Number.isFinite(nz)) {
          rgbaNormals[dst] = nx;
          rgbaNormals[dst + 1] = ny;
          rgbaNormals[dst + 2] = nz;
        }
        rgbaNormals[dst + 3] = 1;
      }
      normalsTex = new THREE.DataTexture(rgbaNormals, w, h, THREE.RGBAFormat, THREE.FloatType);
      normalsTex.colorSpace = THREE.NoColorSpace;
      normalsTex.flipY = false;
      normalsTex.unpackAlignment = 4;
      normalsTex.minFilter = THREE.NearestFilter;
      normalsTex.magFilter = THREE.NearestFilter;
      normalsTex.wrapS = THREE.ClampToEdgeWrapping;
      normalsTex.wrapT = THREE.ClampToEdgeWrapping;
      normalsTex.needsUpdate = true;
    }

    this.depthTex = depthTex;
    this.maskTex = maskTex;
    this.normalsTex = normalsTex;
    this.geometryWidth = w;
    this.geometryHeight = h;
    this.uploadedGeometryId = geometry.metadata.id;
    this.uploadedOwnerKey = null;
    this.cachedOwnerGrid = null;
    this.uploadedSupportKey = null;
    this.surfaceCalibrations.clear();
  }

  private createFloatRgbTexture(
    rgb: Float32Array,
    width: number,
    height: number,
    clampUnit: boolean,
  ): THREE.DataTexture {
    const pixelCount = width * height;
    const rgba = new Float32Array(pixelCount * 4);
    for (let i = 0; i < pixelCount; i++) {
      const src = i * 3;
      const dst = i * 4;
      const rRaw = rgb[src] ?? 0;
      const gRaw = rgb[src + 1] ?? 0;
      const bRaw = rgb[src + 2] ?? 0;
      const r = Number.isFinite(rRaw) ? Math.max(0, rRaw) : 0;
      const g = Number.isFinite(gRaw) ? Math.max(0, gRaw) : 0;
      const b = Number.isFinite(bRaw) ? Math.max(0, bRaw) : 0;
      rgba[dst] = clampUnit ? Math.min(1, r) : r;
      rgba[dst + 1] = clampUnit ? Math.min(1, g) : g;
      rgba[dst + 2] = clampUnit ? Math.min(1, b) : b;
      rgba[dst + 3] = 1;
    }

    const tex = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat, THREE.FloatType);
    tex.colorSpace = THREE.NoColorSpace;
    tex.flipY = false;
    tex.unpackAlignment = 4;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }

  private ensureAppearanceUploaded(appearance: AppearanceData | null): void {
    const nextId = appearance?.metadata.id ?? null;
    if (this.uploadedAppearanceId === nextId) {
      return;
    }

    this.albedoTex?.dispose();
    this.shadingTex?.dispose();
    this.albedoTex = null;
    this.shadingTex = null;
    this.appearanceWidth = 0;
    this.appearanceHeight = 0;
    this.surfaceCalibrations.clear();
    this.uploadedAppearanceId = nextId;

    if (!appearance) {
      return;
    }

    const w = appearance.metadata.appearance_width;
    const h = appearance.metadata.appearance_height;
    this.albedoTex = this.createFloatRgbTexture(appearance.albedo, w, h, true);
    this.shadingTex = this.createFloatRgbTexture(appearance.shading, w, h, false);
    this.appearanceWidth = w;
    this.appearanceHeight = h;
  }

  private getSurfaceCalibration(
    surface: PlanarDecalSurface,
    geometry: GeometryData,
    appearance: AppearanceData,
  ): ShadingCalibrationResult {
    const key = `${appearance.metadata.id}|${surface.id}`;
    const cached = this.surfaceCalibrations.get(key);
    if (cached) {
      return cached;
    }

    const calibration = calibrateShadingScale({
      linearReference: appearance.linearReference,
      albedo: appearance.albedo,
      shading: appearance.shading,
      residual: appearance.residual,
      appearanceWidth: appearance.metadata.appearance_width,
      appearanceHeight: appearance.metadata.appearance_height,
      supportMask: surface.supportMask,
      supportWidth: geometry.metadata.inference_width,
      supportHeight: geometry.metadata.inference_height,
    });
    this.surfaceCalibrations.set(key, calibration);
    return calibration;
  }

  private getSceneCalibration(
    geometry: GeometryData | null,
    appearance: AppearanceData,
  ): ShadingCalibrationResult {
    const key = `${appearance.metadata.id}|${geometry?.metadata.id ?? 'none'}`;
    const cached = this.sceneCalibrations.get(key);
    if (cached) {
      return cached;
    }

    const calibration = calibrateShadingScale({
      linearReference: appearance.linearReference,
      albedo: appearance.albedo,
      shading: appearance.shading,
      residual: appearance.residual,
      appearanceWidth: appearance.metadata.appearance_width,
      appearanceHeight: appearance.metadata.appearance_height,
      supportMask: geometry?.mask ?? null,
      supportWidth: geometry?.metadata.inference_width,
      supportHeight: geometry?.metadata.inference_height,
    });
    this.sceneCalibrations.set(key, calibration);
    return calibration;
  }

  private getOrSyncArtworkTexture(artwork: Artwork): THREE.DataTexture {
    const existing = this.artworkEntries.get(artwork.id);
    if (
      existing &&
      existing.revision === artwork.revision &&
      existing.width === artwork.width &&
      existing.height === artwork.height
    ) {
      return existing.texture;
    }

    existing?.texture.dispose();
    const straight = rasterizeArtworkRgba(artwork);
    const premultiplied = toPremultipliedRgba(straight);
    const texture = new THREE.DataTexture(
      premultiplied,
      artwork.width,
      artwork.height,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    texture.flipY = false;
    texture.unpackAlignment = 4;
    texture.colorSpace = THREE.NoColorSpace;
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;

    this.artworkEntries.set(artwork.id, {
      revision: artwork.revision,
      width: artwork.width,
      height: artwork.height,
      texture,
    });
    return texture;
  }

  private getOrSyncImageLayerEntry(
    layer: ImagePaintLayerKind,
    imageSize: Size2D,
    allStrokes: readonly ImageSpaceStroke[],
    brushObjectId: string | null,
    activePreview: ActiveImageStrokePreview | null,
  ): { entry: ImageLayerGpuCacheEntry; hasPaint: boolean } {
    const layerStrokes =
      brushObjectId === null
        ? []
        : allStrokes.filter(
            (s) =>
              s.layer === layer &&
              (brushObjectId === '__untagged__'
                ? s.brushObjectId === undefined
                : s.brushObjectId === brushObjectId),
          );
    const livePreview = activePreview && activePreview.layer === layer ? activePreview : null;
    const spec = computeImagePaintSpec(imageSize);
    const sig = layerStrokes.map((s) => s.id).join('|');
    const cacheKey = `${brushObjectId ?? '__live__'}:${layer}`;

    let entry = this.imageLayerEntries.get(cacheKey) ?? null;
    if (!entry || entry.width !== spec.width || entry.height !== spec.height) {
      entry?.texture.dispose();
      const byteLength = spec.width * spec.height * 4;
      const committedRgba = new Uint8ClampedArray(byteLength);
      const displayRgba = new Uint8Array(byteLength);
      const texture = new THREE.DataTexture(
        displayRgba,
        spec.width,
        spec.height,
        THREE.RGBAFormat,
        THREE.UnsignedByteType,
      );
      texture.flipY = false;
      texture.unpackAlignment = 4;
      texture.colorSpace = THREE.NoColorSpace;
      texture.generateMipmaps = false;
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.needsUpdate = true;

      entry = {
        width: spec.width,
        height: spec.height,
        committedSignature: '__dirty__',
        committedRgba,
        displayRgba,
        texture,
        hasCommittedPaint: false,
        activePreviewId: null,
        activePreviewRevision: 0,
      };
      this.imageLayerEntries.set(cacheKey, entry);
    }

    let needsUpload = false;
    if (entry.committedSignature !== sig) {
      entry.committedRgba = rebuildImageLayerRgba(layerStrokes, spec);
      entry.committedSignature = sig;
      entry.hasCommittedPaint = layerStrokes.length > 0;
      entry.displayRgba.set(entry.committedRgba);
      entry.activePreviewId = null;
      entry.activePreviewRevision = 0;
      needsUpload = true;
    }

    if (livePreview) {
      if (
        entry.activePreviewId !== livePreview.id ||
        entry.activePreviewRevision !== livePreview.revision
      ) {
        entry.displayRgba.set(entry.committedRgba);
        if (livePreview.layerWidth === spec.width && livePreview.layerHeight === spec.height) {
          compositeStrokeMaskOverRgba(
            entry.displayRgba,
            livePreview.alphaMask,
            livePreview.brush,
            spec.width,
            undefined,
            livePreview.mode,
          );
        } else {
          compositeImageStrokeOverRgba(entry.displayRgba, spec.width, spec.height, {
            id: livePreview.id,
            layer: livePreview.layer,
            mode: livePreview.mode,
            brush: livePreview.brush,
            layerWidth: livePreview.layerWidth,
            layerHeight: livePreview.layerHeight,
            bounds: {
              minX: 0,
              minY: 0,
              maxX: livePreview.layerWidth - 1,
              maxY: livePreview.layerHeight - 1,
            },
            alphaMask: livePreview.alphaMask,
          });
        }
        entry.activePreviewId = livePreview.id;
        entry.activePreviewRevision = livePreview.revision;
        needsUpload = true;
      }
    } else if (entry.activePreviewId !== null) {
      entry.activePreviewId = null;
      entry.activePreviewRevision = 0;
      entry.displayRgba.set(entry.committedRgba);
      needsUpload = true;
    }

    if (needsUpload) {
      entry.texture.needsUpdate = true;
    }

    return { entry, hasPaint: entry.hasCommittedPaint || livePreview !== null };
  }

  private ensureOwnerUploaded(
    geometry: GeometryData,
    surfaces: readonly PlanarDecalSurface[],
  ): void {
    const ownerKey =
      `${geometry.metadata.id}|` +
      surfaces.map((s) => `${s.id}:${s.extentScale.toFixed(4)}`).join(',');
    if (this.uploadedOwnerKey === ownerKey && this.ownerTex && this.cachedOwnerGrid) {
      return;
    }

    const ownerGrid = computeSurfaceOwnerGrid(surfaces, geometry);
    this.cachedOwnerGrid = ownerGrid;
    this.ownerTex?.dispose();

    const tex = new THREE.DataTexture(
      ownerGrid,
      this.geometryWidth,
      this.geometryHeight,
      THREE.RedFormat,
      THREE.UnsignedByteType,
    );
    tex.colorSpace = THREE.NoColorSpace;
    tex.flipY = false;
    tex.unpackAlignment = 1;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;

    this.ownerTex = tex;
    this.uploadedOwnerKey = ownerKey;
    this.uploadedSupportKey = null;
  }

  private ensureSupportUploaded(
    doc: PaintDocumentState,
    surfaces: readonly PlanarDecalSurface[],
  ): void {
    if (surfaces.length === 0 || !this.cachedOwnerGrid) {
      return;
    }
    const supportKey = `${this.uploadedOwnerKey ?? ''}|${doc.activeSurfaceId ?? ''}`;
    if (this.uploadedSupportKey === supportKey && this.supportTex) {
      return;
    }

    this.supportTex?.dispose();
    const pixelCount = this.geometryWidth * this.geometryHeight;
    const rgba = new Uint8Array(pixelCount * 4);
    const ownerGrid = this.cachedOwnerGrid;

    for (let i = 0; i < pixelCount; i++) {
      const owner1 = ownerGrid[i] ?? 0;
      if (owner1 === 0) {
        continue;
      }
      const surf = surfaces[owner1 - 1];
      if (!surf) continue;
      const accent = getSurfaceAccentColor(surf.colorIndex);
      const isActive = surf.id === doc.activeSurfaceId;
      const inSupport = (surf.supportMask[i] ?? 0) !== 0;
      const alpha = inSupport ? (isActive ? 225 : 130) : isActive ? 75 : 30;
      const base = i * 4;
      rgba[base] = accent.rgb[0];
      rgba[base + 1] = accent.rgb[1];
      rgba[base + 2] = accent.rgb[2];
      rgba[base + 3] = alpha;
    }

    const tex = new THREE.DataTexture(
      rgba,
      this.geometryWidth,
      this.geometryHeight,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    tex.colorSpace = THREE.NoColorSpace;
    tex.flipY = false;
    tex.unpackAlignment = 4;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;

    this.supportTex = tex;
    this.uploadedSupportKey = supportKey;
  }

  private getOrSyncDecalEntry(
    surface: PlanarDecalSurface,
    doc: PaintDocumentState,
    brushObjectId: string | null,
    pendingStrokes: readonly BrushStroke[],
    activeStroke: BrushStroke | null,
  ): { entry: DecalGpuCacheEntry; hasPaint: boolean } {
    const footprint = getEffectiveFootprint(surface);
    const footprintKey = buildFootprintKey(footprint);
    const committedStrokes =
      brushObjectId === null
        ? []
        : doc.strokes.filter(
            (stroke) =>
              stroke.surfaceId === surface.id &&
              (brushObjectId === '__untagged__'
                ? stroke.brushObjectId === undefined
                : stroke.brushObjectId === brushObjectId),
          );
    const liveStroke = activeStroke && activeStroke.surfaceId === surface.id ? activeStroke : null;
    const frozenPending = pendingStrokes.filter(
      (s) => s.surfaceId === surface.id && s.id !== liveStroke?.id,
    );
    const baseStrokes =
      frozenPending.length > 0 ? [...committedStrokes, ...frozenPending] : committedStrokes;
    const committedSig = buildCommittedSignature(baseStrokes);
    const cacheKey = `${brushObjectId ?? '__live__'}:${surface.id}`;

    let entry = this.decalEntries.get(cacheKey);
    const spec =
      entry?.footprintKey === footprintKey ? entry.spec : computeDecalTextureSpec(footprint);

    if (!entry || entry.footprintKey !== footprintKey) {
      entry?.texture.dispose();
      const byteLength = spec.width * spec.height * 4;
      const committedRgba = new Uint8ClampedArray(byteLength);
      const displayRgba = new Uint8Array(byteLength);
      const texture = new THREE.DataTexture(
        displayRgba,
        spec.width,
        spec.height,
        THREE.RGBAFormat,
        THREE.UnsignedByteType,
      );
      texture.flipY = false;
      texture.unpackAlignment = 4;
      texture.colorSpace = THREE.NoColorSpace;
      texture.generateMipmaps = false;
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.needsUpdate = true;

      entry = {
        spec,
        footprint,
        footprintKey,
        committedSignature: '__dirty__',
        committedRgba,
        displayRgba,
        texture,
        hasCommittedPaint: false,
        activeStrokeId: null,
        activeStrokePointCount: 0,
        activeMask: null,
      };
      this.decalEntries.set(cacheKey, entry);
    }

    let needsTextureUpload = false;

    if (entry.committedSignature !== committedSig) {
      entry.committedRgba = rebuildSurfaceDecalRgba(baseStrokes, footprint, spec);
      entry.committedSignature = committedSig;
      entry.hasCommittedPaint = baseStrokes.length > 0;
      entry.displayRgba.set(entry.committedRgba);
      entry.activeStrokeId = null;
      entry.activeStrokePointCount = 0;
      needsTextureUpload = true;
    }

    if (liveStroke && liveStroke.points.length > 0) {
      const pts = liveStroke.points;
      const isNewStroke = entry.activeStrokeId !== liveStroke.id || !entry.activeMask;
      if (isNewStroke || pts.length < entry.activeStrokePointCount) {
        entry.activeStrokeId = liveStroke.id;
        entry.activeStrokePointCount = 0;
        entry.activeMask = new Uint8Array(spec.width * spec.height);
        entry.displayRgba.set(entry.committedRgba);
        needsTextureUpload = true;
      }

      const mask = entry.activeMask;
      if (mask) {
        const startIdx = entry.activeStrokePointCount;
        let combinedBounds: DirtyTexelBounds | null = null;

        if (startIdx === 0 && pts.length === 1) {
          const p0 = pts[0];
          if (p0) {
            combinedBounds = stampSegmentIntoStrokeMask(
              mask,
              footprint,
              spec,
              p0,
              p0,
              liveStroke.brush.diameter,
              liveStroke.brush.hardness,
            );
          }
        } else {
          const begin = Math.max(1, startIdx);
          for (let i = begin; i < pts.length; i++) {
            const prev = pts[i - 1];
            const curr = pts[i];
            if (!prev || !curr) continue;
            const segBounds = stampSegmentIntoStrokeMask(
              mask,
              footprint,
              spec,
              prev,
              curr,
              liveStroke.brush.diameter,
              liveStroke.brush.hardness,
            );
            if (segBounds) {
              if (!combinedBounds) {
                combinedBounds = { ...segBounds };
              } else {
                combinedBounds.minX = Math.min(combinedBounds.minX, segBounds.minX);
                combinedBounds.minY = Math.min(combinedBounds.minY, segBounds.minY);
                combinedBounds.maxX = Math.max(combinedBounds.maxX, segBounds.maxX);
                combinedBounds.maxY = Math.max(combinedBounds.maxY, segBounds.maxY);
              }
            }
          }
        }

        if (combinedBounds) {
          for (let py = combinedBounds.minY; py <= combinedBounds.maxY; py++) {
            const rowStart = (py * spec.width + combinedBounds.minX) * 4;
            const rowEnd = (py * spec.width + combinedBounds.maxX + 1) * 4;
            entry.displayRgba.set(entry.committedRgba.subarray(rowStart, rowEnd), rowStart);
          }
          compositeStrokeMaskOverRgba(
            entry.displayRgba,
            mask,
            liveStroke.brush,
            spec.width,
            combinedBounds,
            liveStroke.mode ?? 'paint',
          );
          needsTextureUpload = true;
        }
        entry.activeStrokePointCount = pts.length;
      }
    } else if (entry.activeStrokeId !== null) {
      entry.activeStrokeId = null;
      entry.activeStrokePointCount = 0;
      entry.activeMask = null;
      entry.displayRgba.set(entry.committedRgba);
      needsTextureUpload = true;
    }

    if (needsTextureUpload) {
      entry.texture.needsUpdate = true;
    }

    const hasPaint = entry.hasCommittedPaint || liveStroke !== null;
    return { entry, hasPaint };
  }

  private resolveEffectiveModeCode(
    paintMode: PaintMode,
    appearance: AppearanceData | null,
    calibration: ShadingCalibrationResult | null,
  ): number {
    if (
      paintMode === 'flat' ||
      !appearance ||
      !this.albedoTex ||
      !this.shadingTex ||
      !calibration?.isReliable
    ) {
      return 0;
    }
    return paintMode === 'texture' ? 2 : 1;
  }

  private getOrBuildCurvedMappingGeometry(mapping: SurfaceMapping): THREE.BufferGeometry {
    const existing = this.curvedMappingGeometries.get(mapping.id);
    if (existing) {
      return existing;
    }

    const vertCount = mapping.vertexGridIndices.length;
    const positions = new Float32Array(vertCount * 3);
    const uvs = new Float32Array(vertCount * 2);

    for (let i = 0; i < vertCount; i++) {
      const uImg = mapping.vertexPhotoUvs[i * 2] ?? 0;
      const vImg = mapping.vertexPhotoUvs[i * 2 + 1] ?? 0;
      positions[i * 3] = uImg * 2.0 - 1.0;
      positions[i * 3 + 1] = 1.0 - 2.0 * vImg;
      positions[i * 3 + 2] = 0;
      uvs[i * 2] = uImg;
      uvs[i * 2 + 1] = 1.0 - vImg;
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geom.setAttribute('aSurfaceUv', new THREE.BufferAttribute(mapping.vertexSurfaceUvs, 2));
    geom.setAttribute('aSurfaceDepth', new THREE.BufferAttribute(mapping.vertexDepths, 1));
    geom.setIndex(new THREE.BufferAttribute(mapping.triangleIndices, 1));
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 4);

    this.curvedMappingGeometries.set(mapping.id, geom);
    return geom;
  }

  private renderBrushObjectPass(params: {
    brushObjectId: string | null;
    doc: PaintDocumentState;
    imageSize: Size2D;
    geometry: GeometryData | null;
    appearance: AppearanceData | null;
    allSurfaces: readonly PlanarDecalSurface[];
    pendingStrokes: readonly BrushStroke[];
    activeStroke: BrushStroke | null;
    activeImagePreview: ActiveImageStrokePreview | null;
  }): void {
    const {
      brushObjectId,
      doc,
      imageSize,
      geometry,
      appearance,
      allSurfaces,
      pendingStrokes,
      activeStroke,
      activeImagePreview,
    } = params;

    this.mesh.material = this.overlayMaterial;
    this.uDecalIsArtwork.value = 0;
    this.uDecalPremultipliedPaint.value = 0;
    this.uDecalArtworkOpacity.value = 1;
    this.uDecalShowFootprint.value = 0;
    this.uDecalDepthTolRatio.value = 0.035;
    this.uDecalMinNormalDot.value = 0.45;
    // Brush objects composite cleanly in user display priority without clipping across older surface ownership.
    this.uDecalUseOwnerMask.value = 0;

    if (geometry && allSurfaces.length > 0) {
      this.uDecalLayerKind.value = 0;
      for (let i = 0; i < allSurfaces.length; i++) {
        const surface = allSurfaces[i];
        if (!surface) continue;

        const { entry, hasPaint } = this.getOrSyncDecalEntry(
          surface,
          doc,
          brushObjectId,
          pendingStrokes,
          activeStroke,
        );
        if (!hasPaint) {
          continue;
        }

        const calibration =
          appearance && doc.paintMode !== 'flat'
            ? this.getSurfaceCalibration(surface, geometry, appearance)
            : null;
        this.uDecalEffectivePaintMode.value = this.resolveEffectiveModeCode(
          doc.paintMode,
          appearance,
          calibration,
        );
        this.uDecalShadingScale.value = calibration?.scale ?? 1.0;

        const accent = getSurfaceAccentColor(surface.colorIndex);
        this.uDecalPaintTex.value = entry.texture;
        this.uDecalHasPaint.value = 1;
        this.uDecalSurfaceIndex.value = i + 1;
        this.uDecalSurfaceAccent.value.set(
          accent.rgb[0] / 255,
          accent.rgb[1] / 255,
          accent.rgb[2] / 255,
        );
        this.uDecalPlaneOrigin.value.set(...surface.frame.origin);
        this.uDecalPlaneNormal.value.set(...surface.frame.normal);
        this.uDecalPlaneU.value.set(...surface.frame.uAxis);
        this.uDecalPlaneV.value.set(...surface.frame.vAxis);
        this.uDecalFootprint.value.set(
          entry.footprint.minU,
          entry.footprint.maxU,
          entry.footprint.minV,
          entry.footprint.maxV,
        );
        this.uDecalFitRms.value = surface.rmsError;

        this.renderer.render(this.scene, this.camera);
      }
    }

    const { entry: curvedEntry, hasPaint: hasCurvedPaint } = this.getOrSyncImageLayerEntry(
      'curved',
      imageSize,
      doc.imageStrokes,
      brushObjectId,
      activeImagePreview,
    );
    if (geometry && hasCurvedPaint) {
      const sceneCalib =
        appearance && doc.paintMode !== 'flat'
          ? this.getSceneCalibration(geometry, appearance)
          : null;
      this.uDecalLayerKind.value = 1;
      this.uDecalPaintTex.value = curvedEntry.texture;
      this.uDecalEffectivePaintMode.value = this.resolveEffectiveModeCode(
        doc.paintMode,
        appearance,
        sceneCalib,
      );
      this.uDecalShadingScale.value = sceneCalib?.scale ?? 1.0;
      this.renderer.render(this.scene, this.camera);
    }

    const { entry: fallbackEntry, hasPaint: hasFallbackPaint } = this.getOrSyncImageLayerEntry(
      'fallback-2d',
      imageSize,
      doc.imageStrokes,
      brushObjectId,
      activeImagePreview,
    );
    if (hasFallbackPaint) {
      this.uDecalLayerKind.value = 2;
      this.uDecalPaintTex.value = fallbackEntry.texture;
      this.uDecalEffectivePaintMode.value = 0;
      this.renderer.render(this.scene, this.camera);
    }
  }

  private renderArtworkPlacementPass(params: {
    placement: ArtworkPlacement;
    doc: PaintDocumentState;
    geometry: GeometryData | null;
    effectiveMode: number;
    shadingScale: number;
  }): void {
    const { placement, doc, geometry, effectiveMode, shadingScale } = params;
    if (!placement.visible || placement.opacity <= 0.001) {
      return;
    }
    const artwork = doc.artworks.find((a) => a.id === placement.artworkId);
    if (!artwork) {
      return;
    }

    this.mesh.material = this.overlayMaterial;
    this.uDecalIsArtwork.value = 1;
    this.uDecalPremultipliedPaint.value = 1;
    this.uDecalShowFootprint.value = 0;
    this.uDecalHasPaint.value = 1;
    this.uDecalUseOwnerMask.value = 0;
    this.uDecalEffectivePaintMode.value = effectiveMode;
    this.uDecalShadingScale.value = shadingScale;

    const artTex = this.getOrSyncArtworkTexture(artwork);
    this.uDecalPaintTex.value = artTex;
    this.uDecalArtworkOpacity.value = placement.opacity;
    this.uDecalArtRot.value.set(Math.cos(placement.rotationRad), Math.sin(placement.rotationRad));

    if (placement.mapping.type === 'planar' && geometry) {
      this.uDecalLayerKind.value = 0;
      this.uDecalPlaneOrigin.value.set(...placement.mapping.frame.origin);
      this.uDecalPlaneNormal.value.set(...placement.mapping.frame.normal);
      this.uDecalPlaneU.value.set(...placement.mapping.frame.uAxis);
      this.uDecalPlaneV.value.set(...placement.mapping.frame.vAxis);
      this.uDecalArtCenter.value.set(placement.mapping.centerU, placement.mapping.centerV);
      this.uDecalArtSize.value.set(
        placement.mapping.baseWidthWorld * placement.scale,
        placement.mapping.baseHeightWorld * placement.scale,
      );
      this.uDecalDepthTolRatio.value = placement.mapping.depthToleranceRatio;
      this.uDecalFitRms.value = placement.mapping.rmsError;
      this.uDecalMinNormalDot.value = placement.mapping.isApproximatelyPlanar ? 0.35 : 0.45;
      this.renderer.render(this.scene, this.camera);
    } else if (placement.mapping.type === 'curved' && geometry) {
      const curvedGeom = this.getOrBuildCurvedMappingGeometry(placement.mapping.surfaceMapping);
      this.uDecalArtCenter.value.set(placement.mapping.centerU, placement.mapping.centerV);
      this.uDecalArtSize.value.set(
        placement.mapping.baseWidthWorld * placement.scale,
        placement.mapping.baseHeightWorld * placement.scale,
      );
      this.mesh.geometry = curvedGeom;
      this.mesh.material = this.curvedArtworkMaterial;
      try {
        this.renderer.render(this.scene, this.camera);
      } finally {
        this.mesh.geometry = this.quadGeometry;
        this.mesh.material = this.overlayMaterial;
      }
    } else if (placement.mapping.type === 'image-2d') {
      this.uDecalLayerKind.value = 3;
      this.uDecalArtCenter.value.set(placement.mapping.centerX, placement.mapping.centerY);
      this.uDecalArtSize.value.set(
        placement.mapping.baseWidthPx * placement.scale,
        placement.mapping.baseHeightPx * placement.scale,
      );
      this.renderer.render(this.scene, this.camera);
    }

    this.uDecalIsArtwork.value = 0;
    this.uDecalPremultipliedPaint.value = 0;
    this.uDecalArtworkOpacity.value = 1;
    this.uDecalDepthTolRatio.value = 0.035;
    this.uDecalMinNormalDot.value = 0.45;
  }

  private renderScenePasses(
    image: LoadedImage,
    geometry: GeometryData | null,
    appearance: AppearanceData | null,
    doc: PaintDocumentState,
    pendingStrokes: readonly BrushStroke[],
    activeStroke: BrushStroke | null,
    previewSurfaces: readonly PlanarDecalSurface[],
    activeImagePreview: ActiveImageStrokePreview | null,
    previewPlacement: ArtworkPlacement | null,
    quadTransform: [number, number, number, number],
    imageCssSize: [number, number],
    isExport: boolean,
  ): void {
    this.ensureImageUploaded(image);
    this.ensureAppearanceUploaded(appearance);

    const allSurfaces =
      previewSurfaces.length > 0 ? [...doc.surfaces, ...previewSurfaces] : doc.surfaces;
    if (geometry) {
      this.ensureGeometryUploaded(geometry);
      this.ensureOwnerUploaded(geometry, allSurfaces);
    }

    const hideOverlays = isExport || doc.cleanPreview;
    const showSupport =
      geometry !== null && !hideOverlays && doc.showSupportOverlay && allSurfaces.length > 0;
    if (showSupport) {
      this.ensureSupportUploaded(doc, allSurfaces);
    }

    this.uBaseQuadTransform.value.set(...quadTransform);
    this.uBaseRgbTex.value = this.rgbTex ?? this.fallbackTex;
    this.uBaseSupportTex.value =
      showSupport && this.supportTex ? this.supportTex : this.fallbackTex;
    this.uBaseIsPng.value = this.rgbIsPng ? 1 : 0;
    this.uBaseCompositeChecker.value = isExport ? 0 : 1;
    this.uBaseShowSupport.value = showSupport ? 1 : 0;
    this.uBaseImageCssSize.value.set(...imageCssSize);

    this.mesh.material = this.baseMaterial;
    this.renderer.clear(true, false, false);
    this.renderer.render(this.scene, this.camera);

    this.uDecalQuadTransform.value.set(...quadTransform);
    this.uDecalRgbTex.value = this.rgbTex ?? this.fallbackTex;
    this.uDecalSupportTex.value =
      showSupport && this.supportTex ? this.supportTex : this.fallbackTex;
    this.uDecalAlbedoTex.value = this.albedoTex ?? this.fallbackTex;
    this.uDecalShadingTex.value = this.shadingTex ?? this.fallbackTex;
    this.uDecalIsPng.value = this.rgbIsPng ? 1 : 0;
    this.uDecalCompositeChecker.value = isExport ? 0 : 1;
    this.uDecalShowSupport.value = showSupport ? 1 : 0;
    this.uDecalImageCssSize.value.set(...imageCssSize);
    this.uDecalAppearanceSize.value.set(
      Math.max(1, this.appearanceWidth),
      Math.max(1, this.appearanceHeight),
    );
    this.uDecalTexturePreservation.value = doc.texturePreservation;
    this.uDecalImagePixelSize.value.set(image.metadata.width, image.metadata.height);
    this.uDecalIsArtwork.value = 0;
    this.uDecalPremultipliedPaint.value = 0;
    this.uDecalArtworkOpacity.value = 1;
    this.uDecalDepthTolRatio.value = 0.035;
    this.uDecalMinNormalDot.value = 0.45;

    const hasLiveBrushInput =
      activeStroke !== null || pendingStrokes.length > 0 || activeImagePreview !== null;
    const hasUntaggedStrokes =
      doc.strokes.some((s) => !s.brushObjectId) || doc.imageStrokes.some((s) => !s.brushObjectId);

    const liveSurfaceIds = new Set(allSurfaces.map((s) => s.id));
    const liveBrushKeyIds = new Set<string>(doc.brushObjects.map((b) => b.id));
    if (hasLiveBrushInput) {
      liveBrushKeyIds.add('__live__');
    }
    if (hasUntaggedStrokes) {
      liveBrushKeyIds.add('__untagged__');
    }

    for (const [key, entry] of this.decalEntries.entries()) {
      const sep = key.lastIndexOf(':');
      const brushKey = sep >= 0 ? key.slice(0, sep) : '';
      const surfaceId = sep >= 0 ? key.slice(sep + 1) : '';
      if (!liveSurfaceIds.has(surfaceId) || !liveBrushKeyIds.has(brushKey)) {
        entry.texture.dispose();
        this.decalEntries.delete(key);
      }
    }
    for (const [key, entry] of this.imageLayerEntries.entries()) {
      const sep = key.lastIndexOf(':');
      const brushKey = sep >= 0 ? key.slice(0, sep) : '';
      if (!liveBrushKeyIds.has(brushKey)) {
        entry.texture.dispose();
        this.imageLayerEntries.delete(key);
      }
    }

    const liveArtworkIds = new Set(doc.artworks.map((a) => a.id));
    for (const [id, entry] of this.artworkEntries.entries()) {
      if (!liveArtworkIds.has(id)) {
        entry.texture.dispose();
        this.artworkEntries.delete(id);
      }
    }

    const liveCurvedMappingIds = new Set<string>();
    for (let i = 0; i < doc.artworkPlacements.length; i++) {
      const pl = doc.artworkPlacements[i];
      if (pl?.mapping.type === 'curved') {
        liveCurvedMappingIds.add(pl.mapping.surfaceMapping.id);
      }
    }
    if (previewPlacement?.mapping.type === 'curved') {
      liveCurvedMappingIds.add(previewPlacement.mapping.surfaceMapping.id);
    }
    for (const [mappingId, cachedGeom] of this.curvedMappingGeometries.entries()) {
      if (!liveCurvedMappingIds.has(mappingId)) {
        cachedGeom.dispose();
        this.curvedMappingGeometries.delete(mappingId);
      }
    }

    if (geometry) {
      const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
      this.uDecalDepthTex.value = this.depthTex ?? this.fallbackTex;
      this.uDecalMaskTex.value = this.maskTex ?? this.fallbackTex;
      this.uDecalNormalsTex.value = this.normalsTex ?? this.fallbackTex;
      this.uDecalOwnerTex.value = this.ownerTex ?? this.fallbackTex;
      this.uDecalHasNormals.value = this.normalsTex ? 1 : 0;
      this.uDecalUseOwnerMask.value = this.ownerTex ? 1 : 0;
      this.uDecalIntrinsics0.value.set(intr.fx, intr.skew, intr.cx, intr.fy);
      this.uDecalCy.value = intr.cy;
    }

    const imageSize = { width: image.metadata.width, height: image.metadata.height };
    const sceneCalib =
      appearance && doc.paintMode !== 'flat'
        ? this.getSceneCalibration(geometry, appearance)
        : null;
    const effectiveArtMode = this.resolveEffectiveModeCode(doc.paintMode, appearance, sceneCalib);
    const artShadingScale = sceneCalib?.scale ?? 1.0;

    if (hasUntaggedStrokes) {
      this.renderBrushObjectPass({
        brushObjectId: '__untagged__',
        doc,
        imageSize,
        geometry,
        appearance,
        allSurfaces,
        pendingStrokes: [],
        activeStroke: null,
        activeImagePreview: null,
      });
    }

    const targetLiveBrushId = doc.activeBrushObjectId;
    let renderedLiveBrush = false;
    const renderedBrushIds = new Set<string>();
    const renderedPlacementIds = new Set<string>();

    // Walk sceneOrder back-to-front (index 0 renders last / frontmost).
    for (let i = doc.sceneOrder.length - 1; i >= 0; i--) {
      const ref = doc.sceneOrder[i];
      if (!ref) continue;

      if (ref.kind === 'brush') {
        const brushObj = doc.brushObjects.find((b) => b.id === ref.id);
        if (!brushObj) continue;
        renderedBrushIds.add(brushObj.id);
        const isLiveTarget = brushObj.id === targetLiveBrushId && hasLiveBrushInput;
        if (!brushObj.visible && !isLiveTarget) {
          continue;
        }
        if (isLiveTarget) {
          renderedLiveBrush = true;
        }
        this.renderBrushObjectPass({
          brushObjectId: brushObj.id,
          doc,
          imageSize,
          geometry,
          appearance,
          allSurfaces,
          pendingStrokes: isLiveTarget ? pendingStrokes : [],
          activeStroke: isLiveTarget ? activeStroke : null,
          activeImagePreview: isLiveTarget ? activeImagePreview : null,
        });
      } else {
        const stored = doc.artworkPlacements.find((p) => p.id === ref.id);
        if (!stored) continue;
        renderedPlacementIds.add(stored.id);
        const placement =
          previewPlacement && previewPlacement.id === stored.id ? previewPlacement : stored;
        this.renderArtworkPlacementPass({
          placement,
          doc,
          geometry,
          effectiveMode: effectiveArtMode,
          shadingScale: artShadingScale,
        });
      }
    }

    // Defensive fallback if any state entries were created outside sceneOrder.
    for (let i = 0; i < doc.brushObjects.length; i++) {
      const brushObj = doc.brushObjects[i];
      if (!brushObj || renderedBrushIds.has(brushObj.id)) continue;
      const isLiveTarget = brushObj.id === targetLiveBrushId && hasLiveBrushInput;
      if (!brushObj.visible && !isLiveTarget) continue;
      if (isLiveTarget) renderedLiveBrush = true;
      this.renderBrushObjectPass({
        brushObjectId: brushObj.id,
        doc,
        imageSize,
        geometry,
        appearance,
        allSurfaces,
        pendingStrokes: isLiveTarget ? pendingStrokes : [],
        activeStroke: isLiveTarget ? activeStroke : null,
        activeImagePreview: isLiveTarget ? activeImagePreview : null,
      });
    }
    for (let i = 0; i < doc.artworkPlacements.length; i++) {
      const stored = doc.artworkPlacements[i];
      if (!stored || renderedPlacementIds.has(stored.id)) continue;
      const placement =
        previewPlacement && previewPlacement.id === stored.id ? previewPlacement : stored;
      this.renderArtworkPlacementPass({
        placement,
        doc,
        geometry,
        effectiveMode: effectiveArtMode,
        shadingScale: artShadingScale,
      });
    }

    // Render live preview when the user begins their very first stroke before any BrushObject exists.
    if (hasLiveBrushInput && !renderedLiveBrush) {
      this.renderBrushObjectPass({
        brushObjectId: null,
        doc,
        imageSize,
        geometry,
        appearance,
        allSurfaces,
        pendingStrokes,
        activeStroke,
        activeImagePreview,
      });
    }

    if (!hideOverlays && doc.showFootprintPreview && geometry && allSurfaces.length > 0) {
      this.mesh.material = this.overlayMaterial;
      this.uDecalLayerKind.value = 0;
      this.uDecalIsArtwork.value = 0;
      this.uDecalHasPaint.value = 0;
      this.uDecalShowFootprint.value = 1;
      this.uDecalUseOwnerMask.value = this.ownerTex ? 1 : 0;
      for (let i = 0; i < allSurfaces.length; i++) {
        const surface = allSurfaces[i];
        if (!surface || surface.id !== doc.activeSurfaceId) continue;
        const footprint = getEffectiveFootprint(surface);
        const accent = getSurfaceAccentColor(surface.colorIndex);
        this.uDecalSurfaceIndex.value = i + 1;
        this.uDecalSurfaceAccent.value.set(
          accent.rgb[0] / 255,
          accent.rgb[1] / 255,
          accent.rgb[2] / 255,
        );
        this.uDecalPlaneOrigin.value.set(...surface.frame.origin);
        this.uDecalPlaneNormal.value.set(...surface.frame.normal);
        this.uDecalPlaneU.value.set(...surface.frame.uAxis);
        this.uDecalPlaneV.value.set(...surface.frame.vAxis);
        this.uDecalFootprint.value.set(
          footprint.minU,
          footprint.maxU,
          footprint.minV,
          footprint.maxV,
        );
        this.uDecalFitRms.value = surface.rmsError;
        this.renderer.render(this.scene, this.camera);
      }
      this.uDecalShowFootprint.value = 0;
    }
  }

  renderViewport(params: {
    viewportSize: Size2D;
    dpr: number;
    image: LoadedImage;
    transform: ViewportTransform;
    geometry: GeometryData | null;
    appearance?: AppearanceData | null;
    paintDoc: PaintDocumentState;
    pendingStrokes?: readonly BrushStroke[];
    activeStroke: BrushStroke | null;
    previewSurfaces?: readonly PlanarDecalSurface[];
    activeImagePreview?: ActiveImageStrokePreview | null;
    previewPlacement?: ArtworkPlacement | null;
  }): void {
    if (this.disposed) {
      return;
    }
    const {
      viewportSize,
      dpr,
      image,
      transform,
      geometry,
      appearance = null,
      paintDoc,
      pendingStrokes = [],
      activeStroke,
      previewSurfaces = [],
      activeImagePreview = null,
      previewPlacement = null,
    } = params;
    if (viewportSize.width <= 0 || viewportSize.height <= 0) {
      return;
    }

    const safeDpr = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
    this.renderer.setRenderTarget(null);
    this.renderer.setPixelRatio(safeDpr);
    this.renderer.setSize(viewportSize.width, viewportSize.height, false);
    this.renderer.setViewport(0, 0, viewportSize.width, viewportSize.height);
    this.renderer.setClearColor(0x000000, 0);

    this.ensureImageUploaded(image);
    const wantNearest = transform.zoom >= 3;
    if (this.rgbTex && this.rgbIsNearest !== wantNearest) {
      this.rgbTex.magFilter = wantNearest ? THREE.NearestFilter : THREE.LinearFilter;
      this.rgbTex.needsUpdate = true;
      this.rgbIsNearest = wantNearest;
    }

    const bounds = getImageBoundsInViewport(
      { width: image.metadata.width, height: image.metadata.height },
      transform,
    );
    const quadTransform = computeViewportQuadTransform(
      viewportSize,
      { width: image.metadata.width, height: image.metadata.height },
      transform,
    );

    this.renderScenePasses(
      image,
      geometry,
      appearance,
      paintDoc,
      pendingStrokes,
      activeStroke,
      previewSurfaces,
      activeImagePreview,
      previewPlacement,
      quadTransform,
      [bounds.width, bounds.height],
      false,
    );
  }

  async exportPngBlob(params: {
    image: LoadedImage;
    geometry: GeometryData | null;
    appearance?: AppearanceData | null;
    paintDoc: PaintDocumentState;
  }): Promise<Blob> {
    if (this.disposed) {
      throw new Error('Renderer has been disposed.');
    }

    const { image, geometry, appearance = null, paintDoc } = params;
    const width = image.metadata.width;
    const height = image.metadata.height;
    const gl = this.renderer.getContext();
    const maxTex = this.renderer.capabilities.maxTextureSize;
    const rawRb: unknown = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE);
    const maxRb = typeof rawRb === 'number' && rawRb > 0 ? rawRb : maxTex;
    const maxLimit = Math.min(maxTex, maxRb);

    if (width > maxLimit || height > maxLimit) {
      throw new Error(
        `Image dimensions (${width}x${height}) exceed the GPU render target limit (${maxLimit}px).`,
      );
    }

    this.ensureImageUploaded(image);
    if (this.rgbTex) {
      this.rgbTex.magFilter = THREE.NearestFilter;
      this.rgbTex.needsUpdate = true;
      this.rgbIsNearest = true;
    }

    const renderTarget = new THREE.WebGLRenderTarget(width, height, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      colorSpace: THREE.NoColorSpace,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });

    const prevPixelRatio = this.renderer.getPixelRatio();
    try {
      this.renderer.setPixelRatio(1);
      this.renderer.setRenderTarget(renderTarget);
      this.renderer.setViewport(0, 0, width, height);
      this.renderer.setClearColor(0x000000, 0);

      this.renderScenePasses(
        image,
        geometry,
        appearance,
        paintDoc,
        [],
        null,
        [],
        null,
        null,
        [1, 1, 0, 0],
        [width, height],
        true,
      );

      const bottomUp = new Uint8Array(width * height * 4);
      this.renderer.readRenderTargetPixels(renderTarget, 0, 0, width, height, bottomUp);

      // Flip rows vertically and convert premultiplied framebuffer output back to straight PNG RGBA.
      const topDown = flipRgbaRowsVertically(bottomUp, width, height, true);

      const outCanvas = document.createElement('canvas');
      outCanvas.width = width;
      outCanvas.height = height;
      const ctx = outCanvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to acquire 2D canvas context for PNG export.');
      }
      ctx.putImageData(new ImageData(topDown, width, height), 0, 0);

      return await new Promise<Blob>((resolve, reject) => {
        outCanvas.toBlob((blob) => {
          if (blob) {
            resolve(blob);
          } else {
            reject(new Error('Browser failed to encode PNG blob.'));
          }
        }, 'image/png');
      });
    } finally {
      this.renderer.setRenderTarget(null);
      this.renderer.setPixelRatio(prevPixelRatio);
      renderTarget.dispose();
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    for (const entry of this.decalEntries.values()) {
      entry.texture.dispose();
    }
    this.decalEntries.clear();
    for (const entry of this.artworkEntries.values()) {
      entry.texture.dispose();
    }
    this.artworkEntries.clear();
    for (const cachedGeom of this.curvedMappingGeometries.values()) {
      cachedGeom.dispose();
    }
    this.curvedMappingGeometries.clear();
    for (const entry of this.imageLayerEntries.values()) {
      entry.texture.dispose();
    }
    this.imageLayerEntries.clear();

    this.rgbTex?.dispose();
    this.depthTex?.dispose();
    this.maskTex?.dispose();
    this.normalsTex?.dispose();
    this.albedoTex?.dispose();
    this.shadingTex?.dispose();
    this.surfaceCalibrations.clear();
    this.sceneCalibrations.clear();
    this.ownerTex?.dispose();
    this.supportTex?.dispose();
    this.fallbackTex.dispose();

    this.quadGeometry.dispose();
    this.baseMaterial.dispose();
    this.overlayMaterial.dispose();
    this.curvedArtworkMaterial.dispose();
    this.renderer.dispose();

    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}
