import type { Point2D, Size2D } from '../types/editor';
import {
  type GeometryInput,
  type Vec3Tuple,
  imagePointToInferencePixel,
  isValidGridPixel,
  parseNormalizedIntrinsics,
  projectOpenCvPointToImage,
  readNormalOpenCV,
  vecCross,
  vecDot,
  vecNormalize,
} from '../geometry/geometryAccess';
import {
  LocalSurfaceParameterizer,
  type SurfaceMapping,
  mapSurfaceUvToImagePoint,
  sampleSurfaceMappingAtImagePoint,
} from '../geometry/localSurfaceParameterizer';
import {
  collectLocalContinuousPatch,
  fitPlaneOnPointList,
  isPlanarPatch,
} from '../geometry/autoSurfaceAnchor';
import { PlaneFrame } from '../geometry/planeFrame';
import { getSurfaceGridIndex, pickSurfaceHit } from '../geometry/surfacePicking';
import type { Artwork } from './artwork';

export type ArtworkMapping =
  | {
      type: 'planar';
      frame: PlaneFrame;
      anchorPoint3D: Vec3Tuple;
      rmsError: number;
      depthToleranceRatio: number;
      centerU: number;
      centerV: number;
      baseWidthWorld: number;
      baseHeightWorld: number;
      isApproximatelyPlanar: boolean;
    }
  | {
      type: 'curved';
      surfaceMapping: SurfaceMapping;
      centerU: number;
      centerV: number;
      baseWidthWorld: number;
      baseHeightWorld: number;
    }
  | {
      type: 'image-2d';
      centerX: number;
      centerY: number;
      baseWidthPx: number;
      baseHeightPx: number;
      fallbackReason?: string;
    };

export interface ArtworkPlacement {
  id: string;
  label: string;
  artworkId: string;
  visible: boolean;
  scale: number;
  rotationRad: number;
  opacity: number;
  mapping: ArtworkMapping;
}

export interface PlacementImageQuad {
  corners: [Point2D, Point2D, Point2D, Point2D];
  center: Point2D;
  topMid: Point2D;
  outlinePolygon?: Point2D[];
}

export const MIN_PLACEMENT_SCALE = 0.05;
export const MAX_PLACEMENT_SCALE = 15.0;

const sharedParameterizer = new LocalSurfaceParameterizer();
let nextPlacementSeq = 1;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function normalizeAngleRad(angleRad: number): number {
  let a = angleRad;
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}

function computeDefaultPlacementImageWidth(artwork: Artwork, imageSize: Size2D): number {
  const aspect = artwork.width / Math.max(1, artwork.height);
  const maxByHeight = imageSize.height * 0.5 * aspect;
  const minW = Math.min(Math.max(40, imageSize.width * 0.08), imageSize.width * 0.65);
  const maxW = Math.min(
    imageSize.width * 0.55,
    maxByHeight > 0 ? maxByHeight : imageSize.width * 0.55,
  );
  const desired = Math.min(imageSize.width * 0.28, Math.max(minW, artwork.width));
  return clamp(desired, Math.min(minW, maxW), Math.max(minW, maxW));
}

function buildCurvedTangentFrameAtCenter(
  mapping: Extract<ArtworkMapping, { type: 'curved' }>,
  imageSize: Size2D,
  geometry: GeometryInput,
): PlaneFrame {
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const imgPt = mapSurfaceUvToImagePoint(
    mapping.surfaceMapping,
    mapping.centerU,
    mapping.centerV,
    imageSize,
    intr,
  );
  if (imgPt) {
    const sample = sampleSurfaceMappingAtImagePoint(mapping.surfaceMapping, imgPt, imageSize);
    if (sample && sample.valid) {
      const normal = vecNormalize(vecCross(sample.tangentV, sample.tangentU));
      if (normal) {
        return new PlaneFrame(sample.point3D, normal, sample.tangentU, sample.tangentV);
      }
    }
  }
  return new PlaneFrame(
    mapping.surfaceMapping.anchorFrame.localToWorld(mapping.centerU, mapping.centerV),
    mapping.surfaceMapping.anchorFrame.normal,
    mapping.surfaceMapping.anchorFrame.uAxis,
    mapping.surfaceMapping.anchorFrame.vAxis,
  );
}

function buildMappingFromAspectAndWidth(params: {
  aspectHeightOverWidth: number;
  defaultWidthPx: number;
  desiredWidthPx: number;
  imagePoint: Point2D;
  imageSize: Size2D;
  geometry: GeometryInput | null;
}): { mapping: ArtworkMapping; initialScale: number } {
  const { aspectHeightOverWidth, defaultWidthPx, desiredWidthPx, imagePoint, imageSize, geometry } =
    params;

  const fallback2d = (reason?: string): { mapping: ArtworkMapping; initialScale: number } => ({
    mapping: {
      type: 'image-2d',
      centerX: clamp(imagePoint.x, 0, Math.max(1, imageSize.width)),
      centerY: clamp(imagePoint.y, 0, Math.max(1, imageSize.height)),
      baseWidthPx: defaultWidthPx,
      baseHeightPx: defaultWidthPx * aspectHeightOverWidth,
      ...(reason ? { fallbackReason: reason } : {}),
    },
    initialScale: clamp(
      desiredWidthPx / Math.max(1, defaultWidthPx),
      MIN_PLACEMENT_SCALE,
      MAX_PLACEMENT_SCALE,
    ),
  });

  if (!geometry) {
    return fallback2d('Geometry is unavailable; used 2D overlay.');
  }

  const hit = pickSurfaceHit(geometry, imagePoint, imageSize);
  if (!hit.valid || !hit.pixel || !hit.point3D || !hit.depth) {
    return fallback2d('No valid 3D surface geometry at anchor; used 2D overlay.');
  }

  const gridIndex = getSurfaceGridIndex(geometry);
  const { width: infW, height: infH } = gridIndex;
  const artHalfSpanInf = (desiredWidthPx / Math.max(1, imageSize.width)) * infW * 0.55;
  const windowRadius = clamp(
    Math.round(Math.max(Math.min(infW, infH) * 0.14, artHalfSpanInf)),
    8,
    96,
  );
  const patchIndices = collectLocalContinuousPatch(
    gridIndex,
    hit.pixel.x,
    hit.pixel.y,
    windowRadius,
  );
  if (patchIndices.length < 4) {
    return fallback2d('Surface patch is too fragmented for 3D projection; used 2D overlay.');
  }

  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const baseWidthWorld = Math.max(
    1e-4,
    (defaultWidthPx * hit.depth) / Math.max(1, intr.fx * imageSize.width),
  );
  const baseHeightWorld = baseWidthWorld * aspectHeightOverWidth;
  const initialScale = clamp(
    desiredWidthPx / Math.max(1, defaultWidthPx),
    MIN_PLACEMENT_SCALE,
    MAX_PLACEMENT_SCALE,
  );

  const fit = fitPlaneOnPointList(geometry.points, patchIndices, geometry.normals, hit.point3D);
  const hasNormals = geometry.normals !== null;
  const strictlyPlanar =
    fit !== null &&
    fit.sampleCount >= 6 &&
    fit.patchSpan > 1e-6 &&
    isPlanarPatch(fit, hit.depth, hasNormals) &&
    fit.rmsError / fit.patchSpan <= 0.014 &&
    (!hasNormals || fit.normalDotP08 >= 0.992);

  if (strictlyPlanar) {
    const frame = PlaneFrame.fromPlane(fit.plane, hit.point3D);
    return {
      mapping: {
        type: 'planar',
        frame,
        anchorPoint3D: hit.point3D,
        rmsError: fit.rmsError,
        depthToleranceRatio: 0.035,
        centerU: 0,
        centerV: 0,
        baseWidthWorld,
        baseHeightWorld,
        isApproximatelyPlanar: false,
      },
      initialScale,
    };
  }

  // Cover generous object extent around the artwork so wide banners and upscaled artwork
  // wrap across the entire visible curved silhouette without circular truncation.
  const scaledW = baseWidthWorld * initialScale;
  const scaledH = baseHeightWorld * initialScale;
  const artHalfDiag = Math.hypot(scaledW * 0.5, scaledH * 0.5);
  const patchRadius = Math.max(
    artHalfDiag * 2.8,
    Math.min(hit.depth * 0.75, Math.max(baseWidthWorld * 2.6, 0.28)),
  );

  const curvedOutcome = sharedParameterizer.build({
    geometry,
    gridIndex,
    imagePoint,
    imageSize,
    maxRadius: patchRadius,
  });

  if (curvedOutcome.valid && curvedOutcome.mapping) {
    return {
      mapping: {
        type: 'curved',
        surfaceMapping: curvedOutcome.mapping,
        centerU: 0,
        centerV: 0,
        baseWidthWorld,
        baseHeightWorld,
      },
      initialScale,
    };
  }

  // Fallback to bounded planar approximation if moderately flat locally.
  if (fit && fit.sampleCount >= 6 && fit.patchSpan > 1e-6) {
    const relRms = fit.rmsError / fit.patchSpan;
    const boundedPlanarApprox =
      relRms <= 0.045 &&
      fit.rmsError <= 0.035 * hit.depth &&
      (!hasNormals || fit.normalDotP08 >= 0.86);
    if (boundedPlanarApprox) {
      const frame = PlaneFrame.fromPlane(fit.plane, hit.point3D);
      return {
        mapping: {
          type: 'planar',
          frame,
          anchorPoint3D: hit.point3D,
          rmsError: fit.rmsError,
          depthToleranceRatio: 0.055,
          centerU: 0,
          centerV: 0,
          baseWidthWorld,
          baseHeightWorld,
          isApproximatelyPlanar: true,
        },
        initialScale,
      };
    }
  }

  return fallback2d(
    curvedOutcome.diagnostics.rejectionReason ??
      'Surface geometry is unsuitable for 3D wrapping; used 2D overlay.',
  );
}

function buildMappingAtPoint(params: {
  artwork: Artwork;
  imagePoint: Point2D;
  imageSize: Size2D;
  geometry: GeometryInput | null;
  targetImageWidthPx?: number;
}): { mapping: ArtworkMapping; initialScale: number } {
  const { artwork, imagePoint, imageSize, geometry, targetImageWidthPx } = params;
  const defaultWidthPx = computeDefaultPlacementImageWidth(artwork, imageSize);
  const desiredWidthPx = targetImageWidthPx ?? defaultWidthPx;
  const safeAspect = Math.max(1, artwork.height) / Math.max(1, artwork.width);
  return buildMappingFromAspectAndWidth({
    aspectHeightOverWidth: safeAspect,
    defaultWidthPx,
    desiredWidthPx,
    imagePoint,
    imageSize,
    geometry,
  });
}

export function createAutomaticArtworkPlacement(params: {
  artwork: Artwork;
  imagePoint: Point2D;
  imageSize: Size2D;
  geometry: GeometryInput | null;
  label?: string;
  opacity?: number;
  scale?: number;
  rotationRad?: number;
}): ArtworkPlacement {
  const {
    artwork,
    imagePoint,
    imageSize,
    geometry,
    label,
    opacity = 1.0,
    scale = 1.0,
    rotationRad = 0,
  } = params;
  const { mapping, initialScale } = buildMappingAtPoint({
    artwork,
    imagePoint,
    imageSize,
    geometry,
  });
  const seq = nextPlacementSeq++;
  return {
    id: `placement-${seq}`,
    label: label ?? `${artwork.name} #${seq}`,
    artworkId: artwork.id,
    visible: true,
    scale: clamp(scale * initialScale, MIN_PLACEMENT_SCALE, MAX_PLACEMENT_SCALE),
    rotationRad: normalizeAngleRad(rotationRad),
    opacity: clamp(opacity, 0.05, 1.0),
    mapping,
  };
}

export function estimatePlacementImageCenter(
  placement: ArtworkPlacement,
  imageSize: Size2D,
  geometry: GeometryInput | null,
): Point2D | null {
  if (placement.mapping.type === 'image-2d') {
    return { x: placement.mapping.centerX, y: placement.mapping.centerY };
  }
  if (!geometry) {
    return null;
  }
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  if (placement.mapping.type === 'curved') {
    return mapSurfaceUvToImagePoint(
      placement.mapping.surfaceMapping,
      placement.mapping.centerU,
      placement.mapping.centerV,
      imageSize,
      intr,
    );
  }
  const centerWorld = placement.mapping.frame.localToWorld(
    placement.mapping.centerU,
    placement.mapping.centerV,
  );
  return projectOpenCvPointToImage(centerWorld, intr, imageSize);
}

export function reanchorArtworkPlacement(params: {
  placement: ArtworkPlacement;
  artwork: Artwork;
  imagePoint: Point2D;
  imageSize: Size2D;
  geometry: GeometryInput | null;
}): ArtworkPlacement {
  const { placement, artwork, imagePoint, imageSize, geometry } = params;
  const quad = computePlacementImageQuad(placement, imageSize, geometry);
  const currentPxWidth = quad
    ? Math.hypot(quad.corners[1].x - quad.corners[0].x, quad.corners[1].y - quad.corners[0].y)
    : undefined;
  const { mapping, initialScale } = buildMappingAtPoint({
    artwork,
    imagePoint,
    imageSize,
    geometry,
    targetImageWidthPx: currentPxWidth,
  });
  return {
    ...placement,
    scale: initialScale,
    mapping,
  };
}

function buildPlacementOnSurface(params: {
  sourcePlacement: ArtworkPlacement;
  imagePoint: Point2D;
  imageSize: Size2D;
  geometry: GeometryInput;
  targetImageWidthPx: number;
  aspectHeightOverWidth: number;
}): ArtworkPlacement {
  const {
    sourcePlacement,
    imagePoint,
    imageSize,
    geometry,
    targetImageWidthPx,
    aspectHeightOverWidth,
  } = params;
  const oldDepth =
    sourcePlacement.mapping.type === 'planar'
      ? sourcePlacement.mapping.anchorPoint3D[2]
      : sourcePlacement.mapping.type === 'curved'
        ? sourcePlacement.mapping.surfaceMapping.anchorPoint3D[2]
        : 1.0;
  const defaultWidthPx =
    sourcePlacement.mapping.type === 'image-2d'
      ? sourcePlacement.mapping.baseWidthPx
      : Math.max(
          1,
          (sourcePlacement.mapping.baseWidthWorld *
            parseNormalizedIntrinsics(geometry.metadata.intrinsics).fx *
            imageSize.width) /
            Math.max(0.1, oldDepth),
        );

  const { mapping, initialScale } = buildMappingFromAspectAndWidth({
    aspectHeightOverWidth,
    defaultWidthPx,
    desiredWidthPx: targetImageWidthPx,
    imagePoint,
    imageSize,
    geometry,
  });

  return {
    ...sourcePlacement,
    scale: initialScale,
    mapping,
  };
}

function evaluatePlacementSurfaceSupport(
  placement: ArtworkPlacement,
  imageSize: Size2D,
  geometry: GeometryInput,
): number {
  if (placement.mapping.type === 'image-2d') {
    return 1.0;
  }

  const infW = geometry.metadata.inference_width;
  const infH = geometry.metadata.inference_height;
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const cosR = Math.cos(placement.rotationRad);
  const sinR = Math.sin(placement.rotationRad);
  const w = placement.mapping.baseWidthWorld * placement.scale;
  const h = placement.mapping.baseHeightWorld * placement.scale;
  const offsets = [-0.35, 0, 0.35];
  let visible = 0;

  if (placement.mapping.type === 'planar') {
    const { frame, centerU, centerV, depthToleranceRatio, rmsError } = placement.mapping;
    for (let oy = 0; oy < offsets.length; oy++) {
      const sy = offsets[oy] ?? 0;
      for (let ox = 0; ox < offsets.length; ox++) {
        const sx = offsets[ox] ?? 0;
        const lx = sx * w;
        const ly = sy * h;
        const u = centerU + lx * cosR - ly * sinR;
        const v = centerV + lx * sinR + ly * cosR;
        const worldPt = frame.localToWorld(u, v);
        if (worldPt[2] <= 0.05) continue;
        const imgPt = projectOpenCvPointToImage(worldPt, intr, imageSize);
        if (!imgPt) continue;
        const pix = imagePointToInferencePixel(imgPt, imageSize, infW, infH);
        const gridIdx = pix ? pix.y * infW + pix.x : -1;
        if (!pix || gridIdx < 0 || !isValidGridPixel(geometry, gridIdx)) {
          continue;
        }
        const t = worldPt[2];
        const obsDepth = geometry.depth[gridIdx] ?? 0;
        const tol = clamp(Math.max(depthToleranceRatio * t, 2.5 * rmsError), 0.015 * t, 0.14 * t);
        if (Math.abs(obsDepth - t) > tol) continue;
        if (geometry.normals) {
          const obsNorm = readNormalOpenCV(geometry, pix.x, pix.y);
          if (obsNorm && vecDot(obsNorm, frame.normal) < 0.45) {
            continue;
          }
        }
        visible += 1;
      }
    }
    return visible / 9;
  }

  const sm = placement.mapping.surfaceMapping;
  const { centerU, centerV } = placement.mapping;
  for (let oy = 0; oy < offsets.length; oy++) {
    const sy = offsets[oy] ?? 0;
    for (let ox = 0; ox < offsets.length; ox++) {
      const sx = offsets[ox] ?? 0;
      const lx = sx * w;
      const ly = sy * h;
      const u = centerU + lx * cosR - ly * sinR;
      const v = centerV + lx * sinR + ly * cosR;
      const imgPt = mapSurfaceUvToImagePoint(sm, u, v, imageSize, intr);
      if (!imgPt) continue;
      const sample = sampleSurfaceMappingAtImagePoint(sm, imgPt, imageSize);
      if (!sample || !sample.valid) continue;
      const pix = imagePointToInferencePixel(imgPt, imageSize, infW, infH);
      const gridIdx = pix ? pix.y * infW + pix.x : -1;
      if (!pix || gridIdx < 0 || !isValidGridPixel(geometry, gridIdx)) {
        continue;
      }
      const obsDepth = geometry.depth[gridIdx] ?? 0;
      const tol = Math.max(0.015, sample.depth * 0.055);
      if (Math.abs(obsDepth - sample.depth) <= tol) {
        visible += 1;
      }
    }
  }
  return visible / 9;
}

function tryTransitionPlacementAcrossSurfaces(params: {
  original: ArtworkPlacement;
  shifted: ArtworkPlacement;
  horizonMissed: boolean;
  nextImagePoint: Point2D;
  imageSize: Size2D;
  geometry: GeometryInput;
}): ArtworkPlacement {
  const { original, shifted, horizonMissed, nextImagePoint, imageSize, geometry } = params;
  const oldSupport = horizonMissed
    ? 0
    : evaluatePlacementSurfaceSupport(shifted, imageSize, geometry);
  if (oldSupport >= 0.45) {
    return shifted;
  }

  const hitNext = pickSurfaceHit(geometry, nextImagePoint, imageSize);
  if (!hitNext.valid || !hitNext.point3D || !hitNext.depth) {
    return shifted;
  }

  // Distinguish surface transitions (e.g. wall -> floor corner or moving off a foreground object)
  // from sliding behind a narrower same-facing foreground occluder on the current surface.
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  let shouldEvaluateTransition = horizonMissed;
  if (!shouldEvaluateTransition && shifted.mapping.type === 'planar') {
    const oldNormal = shifted.mapping.frame.normal;
    const planeHit = shifted.mapping.frame.imagePointToPlane(nextImagePoint, imageSize, intr);
    const oldDepth = planeHit?.depth ?? shifted.mapping.anchorPoint3D[2];
    const normalChanged = hitNext.normal !== null && vecDot(hitNext.normal, oldNormal) < 0.65;
    const behindOldSurface = hitNext.depth > oldDepth + Math.max(0.05, oldDepth * 0.08);
    shouldEvaluateTransition = normalChanged || behindOldSurface;
  } else if (!shouldEvaluateTransition && shifted.mapping.type === 'curved') {
    const onOldPatch = sampleSurfaceMappingAtImagePoint(
      shifted.mapping.surfaceMapping,
      nextImagePoint,
      imageSize,
    );
    shouldEvaluateTransition = onOldPatch === null;
  }

  if (!shouldEvaluateTransition || original.mapping.type === 'image-2d') {
    return shifted;
  }

  const oldDepth =
    original.mapping.type === 'planar'
      ? original.mapping.anchorPoint3D[2]
      : original.mapping.surfaceMapping.anchorPoint3D[2];
  const approxScreenWidthPx =
    (original.mapping.baseWidthWorld * original.scale * intr.fx * imageSize.width) /
    Math.max(0.1, oldDepth);
  const aspect = original.mapping.baseHeightWorld / Math.max(1e-6, original.mapping.baseWidthWorld);

  const candidate = buildPlacementOnSurface({
    sourcePlacement: original,
    imagePoint: nextImagePoint,
    imageSize,
    geometry,
    targetImageWidthPx: approxScreenWidthPx,
    aspectHeightOverWidth: aspect,
  });

  if (candidate.mapping.type === 'image-2d') {
    return shifted;
  }

  const newSupport = evaluatePlacementSurfaceSupport(candidate, imageSize, geometry);
  if (horizonMissed || newSupport >= 0.5) {
    return candidate;
  }
  return shifted;
}

export function movePlacementByImagePoints(
  placement: ArtworkPlacement,
  prevImagePoint: Point2D,
  nextImagePoint: Point2D,
  imageSize: Size2D,
  geometry: GeometryInput | null,
): ArtworkPlacement {
  if (placement.mapping.type === 'image-2d') {
    return {
      ...placement,
      mapping: {
        ...placement.mapping,
        centerX: clamp(
          placement.mapping.centerX + (nextImagePoint.x - prevImagePoint.x),
          -imageSize.width * 0.5,
          imageSize.width * 1.5,
        ),
        centerY: clamp(
          placement.mapping.centerY + (nextImagePoint.y - prevImagePoint.y),
          -imageSize.height * 0.5,
          imageSize.height * 1.5,
        ),
      },
    };
  }

  if (!geometry) {
    return placement;
  }

  let shifted = placement;
  let horizonMissed = false;

  if (placement.mapping.type === 'curved') {
    const sPrev = sampleSurfaceMappingAtImagePoint(
      placement.mapping.surfaceMapping,
      prevImagePoint,
      imageSize,
    );
    const sNext = sampleSurfaceMappingAtImagePoint(
      placement.mapping.surfaceMapping,
      nextImagePoint,
      imageSize,
    );
    if (sPrev && sNext) {
      shifted = {
        ...placement,
        mapping: {
          ...placement.mapping,
          centerU: placement.mapping.centerU + (sNext.u - sPrev.u),
          centerV: placement.mapping.centerV + (sNext.v - sPrev.v),
        },
      };
    } else {
      const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
      const tangentFrame = buildCurvedTangentFrameAtCenter(placement.mapping, imageSize, geometry);
      const hitPrev = tangentFrame.imagePointToPlane(prevImagePoint, imageSize, intr);
      const hitNext = tangentFrame.imagePointToPlane(nextImagePoint, imageSize, intr);
      if (!hitPrev || !hitNext || hitPrev.depth <= 0.05 || hitNext.depth <= 0.05) {
        horizonMissed = true;
      } else {
        shifted = {
          ...placement,
          mapping: {
            ...placement.mapping,
            centerU: placement.mapping.centerU + (hitNext.u - hitPrev.u),
            centerV: placement.mapping.centerV + (hitNext.v - hitPrev.v),
          },
        };
      }
    }
  } else {
    const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
    const prevHit = placement.mapping.frame.imagePointToPlane(prevImagePoint, imageSize, intr);
    const nextHit = placement.mapping.frame.imagePointToPlane(nextImagePoint, imageSize, intr);
    if (!prevHit || !nextHit || prevHit.depth <= 0.05 || nextHit.depth <= 0.05) {
      horizonMissed = true;
    } else {
      shifted = {
        ...placement,
        mapping: {
          ...placement.mapping,
          centerU: placement.mapping.centerU + (nextHit.u - prevHit.u),
          centerV: placement.mapping.centerV + (nextHit.v - prevHit.v),
        },
      };
    }
  }

  return tryTransitionPlacementAcrossSurfaces({
    original: placement,
    shifted,
    horizonMissed,
    nextImagePoint,
    imageSize,
    geometry,
  });
}

/**
 * If a completed drag or scale pushed a curved placement near or past the boundary
 * of its parameterized patch, extends the intrinsic surface patch cleanly without
 * rebuilding per-frame during active interaction.
 */
export function maybeExtendCurvedMappingOnCompletedTransform(
  placement: ArtworkPlacement,
  imageSize: Size2D,
  geometry: GeometryInput | null,
): ArtworkPlacement {
  if (placement.mapping.type !== 'curved' || !geometry) {
    return placement;
  }
  const m = placement.mapping;
  const sm = m.surfaceMapping;
  const halfW = m.baseWidthWorld * placement.scale * 0.5;
  const halfH = m.baseHeightWorld * placement.scale * 0.5;
  const artHalfDiag = Math.hypot(halfW, halfH);

  const uMinReq = m.centerU - artHalfDiag * 0.9;
  const uMaxReq = m.centerU + artHalfDiag * 0.9;
  const vMinReq = m.centerV - artHalfDiag * 0.9;
  const vMaxReq = m.centerV + artHalfDiag * 0.9;
  const centerDist = Math.hypot(m.centerU, m.centerV);

  const needsExtension =
    uMinReq < sm.minU ||
    uMaxReq > sm.maxU ||
    vMinReq < sm.minV ||
    vMaxReq > sm.maxV ||
    centerDist > sm.patchRadius * 0.42 ||
    artHalfDiag * 1.15 > sm.patchRadius;

  if (!needsExtension) {
    return placement;
  }

  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const newCenterImg = mapSurfaceUvToImagePoint(sm, m.centerU, m.centerV, imageSize, intr);
  if (!newCenterImg) {
    return placement;
  }

  const centerHit = pickSurfaceHit(geometry, newCenterImg, imageSize);
  if (!centerHit.valid || !centerHit.point3D) {
    return placement;
  }

  const oldSample = sampleSurfaceMappingAtImagePoint(sm, newCenterImg, imageSize);
  const desiredRadius = Math.max(sm.patchRadius * 1.35, artHalfDiag * 2.4);
  const outcome = sharedParameterizer.build({
    geometry,
    geometryId: sm.geometryId,
    imagePoint: newCenterImg,
    imageSize,
    maxRadius: desiredRadius,
    maxVisitedNodes: 75_000,
  });

  if (!outcome.valid || !outcome.mapping) {
    return placement;
  }

  let nextRotation = placement.rotationRad;
  if (oldSample && oldSample.valid) {
    const cosAngle = vecDot(oldSample.tangentU, outcome.mapping.anchorFrame.uAxis);
    const sinAngle = vecDot(oldSample.tangentU, outcome.mapping.anchorFrame.vAxis);
    if (Number.isFinite(cosAngle) && Number.isFinite(sinAngle)) {
      // Transport local artwork rotation continuously between the old tangent basis and the new one.
      nextRotation = normalizeAngleRad(placement.rotationRad + Math.atan2(sinAngle, cosAngle));
    }
  }

  return {
    ...placement,
    rotationRad: nextRotation,
    mapping: {
      ...m,
      surfaceMapping: outcome.mapping,
      centerU: 0,
      centerV: 0,
    },
  };
}

function resolveCurvedDragSurfaceCoords(
  mapping: Extract<ArtworkMapping, { type: 'curved' }>,
  dragImagePoint: Point2D,
  imageSize: Size2D,
  geometry: GeometryInput,
): { u: number; v: number } | null {
  const direct = sampleSurfaceMappingAtImagePoint(
    mapping.surfaceMapping,
    dragImagePoint,
    imageSize,
  );
  if (direct && direct.valid) {
    return { u: direct.u, v: direct.v };
  }
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const tangentFrame = buildCurvedTangentFrameAtCenter(mapping, imageSize, geometry);
  const hit = tangentFrame.imagePointToPlane(dragImagePoint, imageSize, intr);
  if (!hit || hit.depth <= 0.05) {
    return null;
  }
  return { u: mapping.centerU + hit.u, v: mapping.centerV + hit.v };
}

export function scalePlacementFromImagePoint(
  placement: ArtworkPlacement,
  dragImagePoint: Point2D,
  imageSize: Size2D,
  geometry: GeometryInput | null,
): ArtworkPlacement {
  const cosR = Math.cos(placement.rotationRad);
  const sinR = Math.sin(placement.rotationRad);

  if (placement.mapping.type === 'image-2d') {
    const dx = dragImagePoint.x - placement.mapping.centerX;
    const dy = dragImagePoint.y - placement.mapping.centerY;
    const lx = dx * cosR + dy * sinR;
    const ly = -dx * sinR + dy * cosR;
    const halfW = Math.max(1, placement.mapping.baseWidthPx * 0.5);
    const halfH = Math.max(1, placement.mapping.baseHeightPx * 0.5);
    const ratio = Math.hypot(lx / halfW, ly / halfH) / Math.SQRT2;
    return {
      ...placement,
      scale: clamp(ratio, MIN_PLACEMENT_SCALE, MAX_PLACEMENT_SCALE),
    };
  }

  if (!geometry) {
    return placement;
  }

  let dragU: number;
  let dragV: number;
  if (placement.mapping.type === 'curved') {
    const resolved = resolveCurvedDragSurfaceCoords(
      placement.mapping,
      dragImagePoint,
      imageSize,
      geometry,
    );
    if (!resolved) return placement;
    dragU = resolved.u;
    dragV = resolved.v;
  } else {
    const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
    const hit = placement.mapping.frame.imagePointToPlane(dragImagePoint, imageSize, intr);
    if (!hit || hit.depth <= 0.05) {
      // Fallback cleanly when a shallow floor handle extends past the horizon plane.
      const quad = computePlacementImageQuad(placement, imageSize, geometry);
      if (!quad) return placement;
      const baseSpan = Math.max(
        1,
        Math.hypot(quad.corners[1].x - quad.corners[0].x, quad.corners[1].y - quad.corners[0].y) /
          Math.max(MIN_PLACEMENT_SCALE, placement.scale),
      );
      const ratio =
        (Math.hypot(dragImagePoint.x - quad.center.x, dragImagePoint.y - quad.center.y) * 2) /
        (baseSpan * Math.SQRT2);
      return {
        ...placement,
        scale: clamp(ratio, MIN_PLACEMENT_SCALE, MAX_PLACEMENT_SCALE),
      };
    }
    dragU = hit.u;
    dragV = hit.v;
  }

  const du = dragU - placement.mapping.centerU;
  const dv = dragV - placement.mapping.centerV;
  const lx = du * cosR + dv * sinR;
  const ly = -du * sinR + dv * cosR;
  const halfW = Math.max(1e-5, placement.mapping.baseWidthWorld * 0.5);
  const halfH = Math.max(1e-5, placement.mapping.baseHeightWorld * 0.5);
  const ratio = Math.hypot(lx / halfW, ly / halfH) / Math.SQRT2;
  return {
    ...placement,
    scale: clamp(ratio, MIN_PLACEMENT_SCALE, MAX_PLACEMENT_SCALE),
  };
}

export function rotatePlacementFromImagePoint(
  placement: ArtworkPlacement,
  dragImagePoint: Point2D,
  imageSize: Size2D,
  geometry: GeometryInput | null,
): ArtworkPlacement {
  if (placement.mapping.type === 'image-2d') {
    const dx = dragImagePoint.x - placement.mapping.centerX;
    const dy = dragImagePoint.y - placement.mapping.centerY;
    if (Math.hypot(dx, dy) < 1e-3) {
      return placement;
    }
    return {
      ...placement,
      rotationRad: normalizeAngleRad(Math.atan2(dy, dx) + Math.PI * 0.5),
    };
  }

  if (!geometry) {
    return placement;
  }

  let dragU: number;
  let dragV: number;
  if (placement.mapping.type === 'curved') {
    const resolved = resolveCurvedDragSurfaceCoords(
      placement.mapping,
      dragImagePoint,
      imageSize,
      geometry,
    );
    if (!resolved) return placement;
    dragU = resolved.u;
    dragV = resolved.v;
  } else {
    const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
    const hit = placement.mapping.frame.imagePointToPlane(dragImagePoint, imageSize, intr);
    if (!hit || hit.depth <= 0.05) {
      const quad = computePlacementImageQuad(placement, imageSize, geometry);
      if (!quad) return placement;
      const dx = dragImagePoint.x - quad.center.x;
      const dy = dragImagePoint.y - quad.center.y;
      if (Math.hypot(dx, dy) < 1e-3) return placement;
      return {
        ...placement,
        rotationRad: normalizeAngleRad(Math.atan2(dy, dx) + Math.PI * 0.5),
      };
    }
    dragU = hit.u;
    dragV = hit.v;
  }

  const du = dragU - placement.mapping.centerU;
  const dv = dragV - placement.mapping.centerV;
  if (Math.hypot(du, dv) < 1e-6) {
    return placement;
  }
  return {
    ...placement,
    rotationRad: normalizeAngleRad(Math.atan2(dv, du) + Math.PI * 0.5),
  };
}

export function computePlacementImageQuad(
  placement: ArtworkPlacement,
  imageSize: Size2D,
  geometry: GeometryInput | null,
): PlacementImageQuad | null {
  const cosR = Math.cos(placement.rotationRad);
  const sinR = Math.sin(placement.rotationRad);

  if (placement.mapping.type === 'image-2d') {
    const m = placement.mapping;
    const hw = m.baseWidthPx * placement.scale * 0.5;
    const hh = m.baseHeightPx * placement.scale * 0.5;
    const mapPt = (lx: number, ly: number): Point2D => ({
      x: m.centerX + lx * cosR - ly * sinR,
      y: m.centerY + lx * sinR + ly * cosR,
    });
    return {
      corners: [mapPt(-hw, -hh), mapPt(hw, -hh), mapPt(hw, hh), mapPt(-hw, hh)],
      center: { x: m.centerX, y: m.centerY },
      topMid: mapPt(0, -hh),
    };
  }

  if (!geometry) {
    return null;
  }
  const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
  const hw = placement.mapping.baseWidthWorld * placement.scale * 0.5;
  const hh = placement.mapping.baseHeightWorld * placement.scale * 0.5;

  if (placement.mapping.type === 'curved') {
    const { surfaceMapping, centerU, centerV } = placement.mapping;
    const projectSurfaceLocal = (lx: number, ly: number): Point2D | null => {
      const u = centerU + lx * cosR - ly * sinR;
      const v = centerV + lx * sinR + ly * cosR;
      return mapSurfaceUvToImagePoint(surfaceMapping, u, v, imageSize, intr);
    };
    const c0 = projectSurfaceLocal(-hw, -hh);
    const c1 = projectSurfaceLocal(hw, -hh);
    const c2 = projectSurfaceLocal(hw, hh);
    const c3 = projectSurfaceLocal(-hw, hh);
    const center = projectSurfaceLocal(0, 0);
    const topMid = projectSurfaceLocal(0, -hh);
    if (!c0 || !c1 || !c2 || !c3 || !center || !topMid) {
      return null;
    }

    const outlinePolygon: Point2D[] = [];
    const edgeSteps = 7;
    const addEdge = (x0: number, y0: number, x1: number, y1: number) => {
      for (let i = 0; i < edgeSteps; i++) {
        const t = i / edgeSteps;
        const pt = projectSurfaceLocal(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
        if (pt) outlinePolygon.push(pt);
      }
    };
    addEdge(-hw, -hh, hw, -hh);
    addEdge(hw, -hh, hw, hh);
    addEdge(hw, hh, -hw, hh);
    addEdge(-hw, hh, -hw, -hh);

    return {
      corners: [c0, c1, c2, c3],
      center,
      topMid,
      outlinePolygon: outlinePolygon.length >= 4 ? outlinePolygon : undefined,
    };
  }

  const { frame, centerU, centerV } = placement.mapping;
  const projectLocal = (lx: number, ly: number): Point2D | null => {
    const u = centerU + lx * cosR - ly * sinR;
    const v = centerV + lx * sinR + ly * cosR;
    const worldPt = frame.localToWorld(u, v);
    if (worldPt[2] <= 0.05) {
      return null;
    }
    return projectOpenCvPointToImage(worldPt, intr, imageSize);
  };

  const c0 = projectLocal(-hw, -hh);
  const c1 = projectLocal(hw, -hh);
  const c2 = projectLocal(hw, hh);
  const c3 = projectLocal(-hw, hh);
  const center = projectLocal(0, 0);
  const topMid = projectLocal(0, -hh);
  if (!c0 || !c1 || !c2 || !c3 || !center || !topMid) {
    return null;
  }
  return {
    corners: [c0, c1, c2, c3],
    center,
    topMid,
  };
}

function pointInPolygon2D(pt: Point2D, poly: readonly Point2D[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const pi = poly[i];
    const pj = poly[j];
    if (!pi || !pj) continue;
    const intersects =
      pi.y > pt.y !== pj.y > pt.y &&
      pt.x < ((pj.x - pi.x) * (pt.y - pi.y)) / (pj.y - pi.y + 1e-12) + pi.x;
    if (intersects) {
      inside = !inside;
    }
  }
  return inside;
}

function pointInConvexQuad(
  pt: Point2D,
  quad: readonly [Point2D, Point2D, Point2D, Point2D],
): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    if (!a || !b) return false;
    const cross = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
    if (Math.abs(cross) < 1e-6) continue;
    const s = cross > 0 ? 1 : -1;
    if (sign === 0) {
      sign = s;
    } else if (sign !== s) {
      return false;
    }
  }
  return sign !== 0;
}

export function hitTestPlacements(
  placements: readonly ArtworkPlacement[],
  imagePoint: Point2D,
  imageSize: Size2D,
  geometry: GeometryInput | null,
): ArtworkPlacement | null {
  for (let i = placements.length - 1; i >= 0; i--) {
    const p = placements[i];
    if (!p || !p.visible) continue;
    if (p.mapping.type === 'curved') {
      const hitCoord = samplePlacementTextureCoord(p, imagePoint, imageSize, geometry);
      if (hitCoord) {
        return p;
      }
      const quad = computePlacementImageQuad(p, imageSize, geometry);
      if (quad) {
        const poly = quad.outlinePolygon ?? quad.corners;
        if (pointInPolygon2D(imagePoint, poly)) {
          return p;
        }
      }
      continue;
    }
    const quad = computePlacementImageQuad(p, imageSize, geometry);
    if (quad && pointInConvexQuad(imagePoint, quad.corners)) {
      return p;
    }
  }
  return null;
}

/**
 * Computes normalized artwork texture UV [0..1] for an image pixel using identical mapping math as the GPU shader.
 */
export function samplePlacementTextureCoord(
  placement: ArtworkPlacement,
  imagePoint: Point2D,
  imageSize: Size2D,
  geometry: GeometryInput | null,
): Point2D | null {
  const cosR = Math.cos(placement.rotationRad);
  const sinR = Math.sin(placement.rotationRad);

  if (placement.mapping.type === 'image-2d') {
    const dx = imagePoint.x - placement.mapping.centerX;
    const dy = imagePoint.y - placement.mapping.centerY;
    const rx = dx * cosR + dy * sinR;
    const ry = -dx * sinR + dy * cosR;
    const w = Math.max(1e-6, placement.mapping.baseWidthPx * placement.scale);
    const h = Math.max(1e-6, placement.mapping.baseHeightPx * placement.scale);
    const uTex = rx / w + 0.5;
    const vTex = ry / h + 0.5;
    if (uTex < 0 || uTex > 1 || vTex < 0 || vTex > 1) {
      return null;
    }
    return { x: uTex, y: vTex };
  }

  if (!geometry) {
    return null;
  }

  let surfU: number;
  let surfV: number;
  if (placement.mapping.type === 'curved') {
    const sample = sampleSurfaceMappingAtImagePoint(
      placement.mapping.surfaceMapping,
      imagePoint,
      imageSize,
    );
    if (!sample || !sample.valid) {
      return null;
    }
    surfU = sample.u;
    surfV = sample.v;
  } else {
    const intr = parseNormalizedIntrinsics(geometry.metadata.intrinsics);
    const hit = placement.mapping.frame.imagePointToPlane(imagePoint, imageSize, intr);
    if (!hit || hit.depth <= 1e-4) {
      return null;
    }
    surfU = hit.u;
    surfV = hit.v;
  }

  const du = surfU - placement.mapping.centerU;
  const dv = surfV - placement.mapping.centerV;
  const rx = du * cosR + dv * sinR;
  const ry = -du * sinR + dv * cosR;
  const w = Math.max(1e-6, placement.mapping.baseWidthWorld * placement.scale);
  const h = Math.max(1e-6, placement.mapping.baseHeightWorld * placement.scale);
  const uTex = rx / w + 0.5;
  const vTex = ry / h + 0.5;
  if (uTex < 0 || uTex > 1 || vTex < 0 || vTex > 1) {
    return null;
  }
  return { x: uTex, y: vTex };
}
