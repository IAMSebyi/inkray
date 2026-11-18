from dataclasses import replace

import numpy as np

from .depth import DepthMap
from .normals import NormalMap
from .points import Points3D
from .result import GeometryResult

_NEIGHBOR_OFFSETS = tuple(
    (dy, dx)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dy, dx) != (0, 0)
)


def _neighbor_slice(height: int, width: int, dy: int, dx: int) -> tuple[slice, slice]:
    return slice(1 + dy, 1 + dy + height), slice(1 + dx, 1 + dx + width)


def smooth_geometry(
    geometry: GeometryResult,
    iterations: int = 3,
    max_depth_jump_ratio: float = 1.25,
) -> GeometryResult:
    """Uniformly smooth valid depth and normals across continuous neighborhoods.

    Recomputes 3D points from smoothed depth to keep depth and unprojected points consistent,
    then smooths and renormalizes unit surface normals without bleeding across depth jumps.
    """
    if iterations <= 0 or min(geometry.shape) < 2:
        return geometry

    depth = geometry.depth.data
    valid = geometry.mask.data & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return geometry

    height, width = geometry.shape
    max_log_jump = float(np.log(max(max_depth_jump_ratio, 1.01)))
    log_orig = np.log(np.where(valid, depth, 1.0))

    valid_pad = np.pad(valid, 1, mode="constant", constant_values=False)
    log_pad = np.pad(log_orig, 1, mode="constant", constant_values=0.0)
    neighbors = [
        (
            _neighbor_slice(height, width, dy, dx),
            valid
            & valid_pad[_neighbor_slice(height, width, dy, dx)]
            & (np.abs(log_pad[_neighbor_slice(height, width, dy, dx)] - log_orig) <= max_log_jump),
        )
        for dy, dx in _NEIGHBOR_OFFSETS
    ]

    work_depth = np.where(valid, depth, 0.0).astype(np.float32, copy=True)
    for _ in range(iterations):
        depth_pad = np.pad(work_depth, 1, mode="constant", constant_values=0.0)
        total = np.where(valid, work_depth, 0.0)
        count = valid.astype(np.float32)
        for (sy, sx), keep in neighbors:
            total += np.where(keep, depth_pad[sy, sx], 0.0)
            count += keep
        work_depth = np.where(valid & (count > 0), total / np.maximum(count, 1.0), work_depth)

    out_depth = np.where(valid, work_depth, depth).astype(np.float32, copy=False)
    points = geometry.intrinsics.unproject_depth(np.where(valid, out_depth, np.nan))

    work_normals = NormalMap.from_points(points).data.copy()
    finite_n = valid & np.all(np.isfinite(work_normals), axis=-1)
    for _ in range(iterations):
        normals_pad = np.pad(
            np.where(finite_n[..., None], work_normals, 0.0),
            ((1, 1), (1, 1), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        finite_pad = np.pad(finite_n, 1, mode="constant", constant_values=False)
        accum = np.where(finite_n[..., None], work_normals, 0.0)
        for (sy, sx), keep in neighbors:
            keep_n = keep & finite_pad[sy, sx]
            accum += np.where(keep_n[..., None], normals_pad[sy, sx], 0.0)
        lengths = np.linalg.norm(accum, axis=-1, keepdims=True)
        has_norm = valid & (lengths[..., 0] > 1e-6)
        np.divide(accum, lengths, out=work_normals, where=has_norm[..., None])
        work_normals[~has_norm] = np.nan
        finite_n = has_norm

    return replace(
        geometry,
        depth=DepthMap(out_depth),
        points=Points3D(points),
        normals=NormalMap(work_normals),
    )
