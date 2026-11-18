from dataclasses import replace

import numpy as np

from .depth import DepthMap
from .normals import NormalMap
from .points import Points3D
from .result import GeometryResult


def refine_depth_with_normals(geometry: GeometryResult, guide: NormalMap) -> GeometryResult:
    if guide.shape[:2] != geometry.shape:
        raise ValueError("The normal guide and depth map must share the same pixel grid.")
    if min(geometry.shape) < 2:
        return geometry

    depth = geometry.depth.data
    valid = geometry.mask.data & np.isfinite(depth) & (depth > 0)
    rays = geometry.intrinsics.pixel_rays(geometry.width, geometry.height)
    lengths = np.linalg.norm(guide.data, axis=-1, keepdims=True)
    normals = guide.data / np.maximum(lengths, 1e-6)
    facing = np.sum(normals * rays, axis=-1)
    normals = np.where(facing[..., None] > 0, -normals, normals)
    log_depth = np.log(np.where(valid, depth, 1))
    rhs = np.zeros_like(depth)
    diagonal = np.full_like(depth, 0.15)
    edges = []

    for start, end in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        agreement = np.sum(normals[start] * normals[end], axis=-1)
        normal = (normals[start] + normals[end]) * 0.5
        near = -np.sum(normal * rays[start], axis=-1)
        far = -np.sum(normal * rays[end], axis=-1)
        gradient = np.log(np.maximum(near, 0.1)) - np.log(np.maximum(far, 0.1))
        original = log_depth[end] - log_depth[start]
        # Across a depth jump, keep the two surfaces independent.
        use_edge = (
            valid[start] & valid[end] & (near > 0.1) & (far > 0.1)
            & (agreement > 0.7) & (np.abs(original) < np.log(1.1))
        )
        weight = np.where(use_edge, np.clip(agreement, 0, 1), 0)
        residual = np.where(use_edge, gradient - original, 0)
        rhs[start] -= weight * residual
        rhs[end] += weight * residual
        diagonal[start] += weight
        diagonal[end] += weight
        edges.append((start, end, weight))

    # Solve for log-depth corrections with a screen term anchoring the original depth.
    correction = np.zeros_like(depth)
    for _ in range(48):
        update = rhs.copy()
        for start, end, weight in edges:
            update[start] += weight * correction[end]
            update[end] += weight * correction[start]
        correction = update / diagonal

    refined_depth = np.where(valid, depth * np.exp(np.clip(correction, -0.15, 0.15)), depth)
    points = geometry.intrinsics.unproject_depth(np.where(valid, refined_depth, np.nan))
    return replace(
        geometry,
        depth=DepthMap(refined_depth),
        points=Points3D(points),
        normals=NormalMap.from_points(points),
    )
