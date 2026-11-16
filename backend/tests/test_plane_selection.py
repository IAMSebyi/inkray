import math
import numpy as np

from app.geometry.depth import DepthMap
from app.geometry.intrinsics import CameraIntrinsics
from app.geometry.mask import ValidMask
from app.geometry.normals import NormalMap
from app.geometry.plane_fitter import LocalPlaneFitter
from app.geometry.points import Points3D
from app.geometry.result import GeometryResult
from app.geometry.surface_selector import SurfaceSelector


def _build_planar_samples(
    nu: int,
    nv: int,
    origin: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(n, ref)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(n, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    pts: list[np.ndarray] = []
    for iv in range(nv):
        v = (iv - (nv - 1) * 0.5) * 0.15
        for iu in range(nu):
            u = (iu - (nu - 1) * 0.5) * 0.15
            pts.append(origin + u * u_axis + v * v_axis)
    return np.stack(pts, axis=0)


def _create_synthetic_scene(
    width: int,
    height: int,
    plane_origin: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
    *,
    occluders: list[tuple[int, int, int, int, float]] | None = None,
    backgrounds: list[tuple[int, int, int, int, float]] | None = None,
    invalid_rects: list[tuple[int, int, int, int]] | None = None,
    noise_amp: float = 0.0,
) -> tuple[GeometryResult, np.ndarray]:
    n = np.array(plane_normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    origin = np.array(plane_origin, dtype=np.float64)
    intr = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)

    cols = (np.arange(width, dtype=np.float64) + 0.5) / width
    rows = (np.arange(height, dtype=np.float64) + 0.5) / height
    uu, vv = np.meshgrid(cols, rows)
    x_norm = uu - 0.5
    y_norm = vv - 0.5
    rays = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)

    denom = rays @ n
    t_plane = np.where(np.abs(denom) > 1e-6, float(np.dot(n, origin)) / denom, -1.0)

    depth = t_plane.copy()
    normals = np.broadcast_to(n, (height, width, 3)).copy()
    mask = t_plane > 0.0
    truth_plane = mask.copy()

    if occluders:
        for c0, c1, r0, r1, z_occ in occluders:
            sl = np.s_[r0 : r1 + 1, c0 : c1 + 1]
            front = (t_plane[sl] <= 0.0) | (z_occ < t_plane[sl])
            depth[sl] = np.where(front, z_occ, depth[sl])
            normals[sl][front] = np.array([0.0, 0.0, -1.0])
            truth_plane[sl][front] = False

    if backgrounds:
        for c0, c1, r0, r1, z_bg in backgrounds:
            sl = np.s_[r0 : r1 + 1, c0 : c1 + 1]
            depth[sl] = z_bg
            normals[sl] = np.array([0.0, 0.0, -1.0])
            truth_plane[sl] = False

    if noise_amp > 0.0:
        rng = np.random.default_rng(123)
        depth *= 1.0 + noise_amp * rng.uniform(-1.0, 1.0, size=(height, width))

    points = rays * depth[..., None]

    if invalid_rects:
        for c0, c1, r0, r1 in invalid_rects:
            sl = np.s_[r0 : r1 + 1, c0 : c1 + 1]
            mask[sl] = False
            depth[sl] = math.nan
            points[sl] = 0.0
            truth_plane[sl] = False

    geom = GeometryResult(
        depth=DepthMap(depth.astype(np.float32)),
        normals=NormalMap(normals.astype(np.float32)),
        points=Points3D(points.astype(np.float32)),
        intrinsics=CameraIntrinsics(intr),
        mask=ValidMask(mask),
    )
    return geom, truth_plane


def test_local_plane_fitter_least_squares_and_degeneracy() -> None:
    origin = np.array([0.2, -0.1, 4.0])
    normal = np.array([0.45, -0.2, -0.87])
    normal /= np.linalg.norm(normal)
    pts = _build_planar_samples(6, 6, origin, normal)

    fit = LocalPlaneFitter.fit_least_squares(pts)
    assert fit is not None
    assert float(np.dot(fit.plane.normal, normal)) > 0.9999
    assert float(np.dot(fit.plane.normal, -fit.plane.origin)) > 0.0
    assert fit.rms_error < 1e-6
    assert fit.inlier_indices.size == 36

    assert LocalPlaneFitter.fit_least_squares(pts[:2]) is None
    collinear = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    assert LocalPlaneFitter.fit_least_squares(collinear) is None


def test_local_plane_fitter_robust_handles_outliers() -> None:
    origin = np.array([0.0, 0.0, 4.0])
    normal = np.array([0.3, 0.2, -0.93])
    normal /= np.linalg.norm(normal)
    inliers = _build_planar_samples(8, 8, origin, normal)

    outliers = np.array(
        [[(i - 10) * 0.12, (i % 5) * 0.1, 1.5 if i % 2 == 0 else 8.5] for i in range(20)],
        dtype=np.float64,
    )
    mixed = np.vstack([inliers, outliers])

    fit = LocalPlaneFitter.fit_robust(mixed, seed=42, iterations=96)
    assert fit is not None
    assert float(np.dot(fit.plane.normal, normal)) > 0.999
    assert fit.inlier_indices.size >= inliers.shape[0] - 1
    assert fit.rms_error < 0.01


def test_surface_selector_fits_oblique_occluded_scenes_with_high_iou() -> None:
    w, h = 48, 48
    true_normal = (0.45, -0.15, -0.88)
    geom, truth_plane = _create_synthetic_scene(
        w,
        h,
        (0.0, 0.0, 4.0),
        true_normal,
        occluders=[(18, 26, 18, 28, 1.8)],
        backgrounds=[(38, 47, 0, 47, 9.0)],
        invalid_rects=[(2, 5, 2, 5)],
        noise_amp=0.003,
    )

    outcome = SurfaceSelector.select_at_image_point(geom, (10.0, 20.0), (w, h), ransac_seed=42)
    assert outcome.ok is True
    assert outcome.selection is not None

    sel = outcome.selection
    n_true = np.array(true_normal, dtype=np.float64)
    n_true /= np.linalg.norm(n_true)
    assert float(np.dot(sel.plane.normal, n_true)) > 0.998
    assert sel.rms_error < 0.02

    # Foreground occluders, distant background, and invalid mask regions must be excluded.
    assert not np.any(sel.support_mask[18:29, 18:27])
    assert not np.any(sel.support_mask[:, 38:48])
    assert not np.any(sel.support_mask[2:6, 2:6])

    intersection = int(np.count_nonzero(sel.support_mask & truth_plane))
    union = int(np.count_nonzero(sel.support_mask | truth_plane))
    iou = intersection / union
    assert iou >= 0.98

    # Frame axes must be orthonormal and footprint must strictly contain origin (0, 0).
    assert np.linalg.norm(sel.frame.u_axis) == 1.0 or math.isclose(
        float(np.linalg.norm(sel.frame.u_axis)), 1.0, abs_tol=1e-6
    )
    assert abs(float(np.dot(sel.frame.u_axis, sel.frame.v_axis))) < 1e-6
    assert abs(float(np.dot(sel.frame.u_axis, sel.frame.normal))) < 1e-6
    assert sel.suggested_footprint.min_u < 0.0 < sel.suggested_footprint.max_u
    assert sel.suggested_footprint.min_v < 0.0 < sel.suggested_footprint.max_v


def test_surface_selector_rejects_invalid_and_isolated_clicks() -> None:
    w, h = 24, 24
    geom, _ = _create_synthetic_scene(
        w,
        h,
        (0.0, 0.0, 3.0),
        (0.0, 0.0, -1.0),
        invalid_rects=[(10, 15, 10, 15)],
    )

    out_of_bounds = SurfaceSelector.select_at_image_point(geom, (-1.0, 4.0), (w, h))
    assert out_of_bounds.ok is False

    on_invalid = SurfaceSelector.select_at_image_point(geom, (12.0, 12.0), (w, h))
    assert on_invalid.ok is False

    # Create a tiny isolated 2x2 patch inside invalid region.
    isolated_mask = geom.mask.data.copy()
    isolated_mask[11:13, 11:13] = True
    isolated_depth = geom.depth.data.copy()
    isolated_depth[11:13, 11:13] = 3.0
    isolated_pts = geom.points.data.copy()
    isolated_pts[11:13, 11:13, :] = [0.0, 0.0, 3.0]
    tiny_geom = GeometryResult(
        depth=DepthMap(isolated_depth),
        normals=geom.normals,
        points=Points3D(isolated_pts),
        intrinsics=geom.intrinsics,
        mask=ValidMask(isolated_mask),
    )
    tiny_outcome = SurfaceSelector.select_at_image_point(tiny_geom, (11.5, 11.5), (w, h))
    assert tiny_outcome.ok is False
