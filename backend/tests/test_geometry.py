import numpy as np
import pytest

from app.geometry.depth import DepthMap
from app.geometry.intrinsics import CameraIntrinsics
from app.geometry.mask import ValidMask
from app.geometry.normal_refinement import refine_depth_with_normals
from app.geometry.normals import NormalMap
from app.geometry.points import Points3D
from app.geometry.result import GeometryResult
from app.geometry.smoothing import smooth_geometry
from app.geometry.transforms import (
    compute_inference_size,
    map_point_inference_to_original,
    map_point_original_to_inference,
    opencv_normals_to_threejs,
    opencv_points_to_threejs,
    threejs_points_to_opencv,
)


def test_depth_and_mask_sampling_respects_validity() -> None:
    depth = DepthMap(
        np.array(
            [
                [1.0, 3.0, 99.0],
                [5.0, 7.0, 99.0],
            ],
            dtype=np.float32,
        )
    )
    mask = ValidMask(
        np.array(
            [
                [True, True, False],
                [True, True, False],
            ],
            dtype=bool,
        )
    )

    assert depth.at_pixel(1, 0, mask) == pytest.approx(3.0)
    assert depth.at_pixel(2, 0, mask) is None
    assert depth.valid_range(mask) == pytest.approx((1.0, 7.0))

    # Exact pixel center (col + 0.5, row + 0.5) returns exact value.
    assert depth.sample(0.5, 0.5, mask) == pytest.approx(1.0)
    assert depth.sample(1.0, 1.0, mask) == pytest.approx(4.0)

    # Neighboring invalid pixels (col=2) must not bleed into valid sample at (1.75, 0.5).
    assert depth.sample(1.75, 0.5, mask) == pytest.approx(3.0)
    assert depth.sample(2.5, 0.5, mask) is None

    # Mask uses discrete lookup rather than blending.
    assert mask.is_valid(1.9, 0.5) is True
    assert mask.is_valid(2.1, 0.5) is False


def test_normal_map_sampling_renormalizes() -> None:
    n0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    normals = NormalMap(np.array([[n0, n1]], dtype=np.float32))
    sampled = normals.sample(1.0, 0.5)
    assert sampled is not None
    assert np.linalg.norm(sampled) == pytest.approx(1.0, abs=1e-5)
    expected = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0], dtype=np.float32)
    np.testing.assert_allclose(sampled, expected, atol=1e-5)


def test_unproject_depth_and_derive_normals_at_an_occlusion_edge() -> None:
    depth = np.array([[1, 1, 4, 4], [1, 1, 4, 4], [1, 1, 4, 4]], dtype=np.float32)
    height, width = depth.shape
    intrinsics = CameraIntrinsics.from_pixel_matrix(
        np.array([[3, 0, width / 2], [0, 3, height / 2], [0, 0, 1]]), width, height,
    )
    points = intrinsics.unproject_depth(depth)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    np.testing.assert_allclose(
        intrinsics.project(points, width, height), np.stack((u, v), axis=-1), atol=1e-6,
    )
    normals = NormalMap.from_points(points)
    np.testing.assert_allclose(normals.data[..., :2], 0)
    np.testing.assert_allclose(normals.data[..., 2], -1)


def test_normal_refinement_restores_planar_detail_without_smoothing_across_occlusions() -> None:
    height, width = 12, 24
    intrinsics = CameraIntrinsics.from_pixel_matrix(
        np.array([[22, 0, width / 2], [0, 22, height / 2], [0, 0, 1]]), width, height,
    )
    rays = intrinsics.pixel_rays(width, height)
    columns = np.arange(width)
    target = np.where(columns < width // 2, 2.0, 5.0) / (1 + 0.4 * rays[..., 0])
    noisy = np.asarray(target * (1 + 0.012 * (-1.0) ** columns), dtype=np.float32)
    geometry = GeometryResult(
        depth=DepthMap(noisy),
        normals=None,
        points=Points3D(intrinsics.unproject_depth(noisy)),
        intrinsics=intrinsics,
        mask=ValidMask(np.ones((height, width), dtype=bool)),
    )
    normal = np.array([-0.4, 0, -1], dtype=np.float32)
    refined = refine_depth_with_normals(geometry, NormalMap(np.broadcast_to(normal, rays.shape)))
    assert np.mean(np.abs(refined.depth.data - target)) < np.mean(np.abs(noisy - target)) * 0.35
    assert refined.depth.data[0, width // 2] / refined.depth.data[0, width // 2 - 1] > 2
    np.testing.assert_allclose(refined.points.data[..., 2], refined.depth.data)
    assert refined.normals is not None


def test_camera_intrinsics_conversion_and_projection() -> None:
    k_norm = np.array(
        [
            [0.8, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    intrinsics = CameraIntrinsics(k_norm)
    width, height = 1000, 800

    k_px = intrinsics.to_pixel_matrix(width, height)
    assert k_px[0, 0] == pytest.approx(800.0)
    assert k_px[1, 1] == pytest.approx(800.0)
    assert k_px[0, 2] == pytest.approx(500.0)
    assert k_px[1, 2] == pytest.approx(400.0)

    recovered = CameraIntrinsics.from_pixel_matrix(k_px, width, height)
    np.testing.assert_allclose(recovered.matrix, k_norm, atol=1e-6)

    point_3d = intrinsics.unproject(700.0, 200.0, 4.0, width, height)
    np.testing.assert_allclose(point_3d, [1.0, -1.0, 4.0], atol=1e-5)

    uv = intrinsics.project(point_3d, width, height)
    np.testing.assert_allclose(uv, [700.0, 200.0], atol=1e-4)


def test_opencv_to_threejs_conversion_preserves_input() -> None:
    pts = np.array([[[1.5, 2.0, 5.0], [-0.5, -1.0, 3.0]]], dtype=np.float32)
    pts_copy = pts.copy()
    three_pts = opencv_points_to_threejs(pts)

    np.testing.assert_array_equal(pts, pts_copy)
    np.testing.assert_allclose(three_pts[0, 0], [1.5, -2.0, -5.0])
    np.testing.assert_allclose(three_pts[0, 1], [-0.5, 1.0, -3.0])
    np.testing.assert_allclose(threejs_points_to_opencv(three_pts), pts)

    normals = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(opencv_normals_to_threejs(normals)[0, 0], [0.0, -1.0, 0.0])


def test_original_and_inference_resolution_mapping() -> None:
    assert compute_inference_size(4000, 3000, 1000) == (1000, 750)
    assert compute_inference_size(800, 600, 1024) == (800, 600)
    assert compute_inference_size(4000, 3000, 0) == (4000, 3000)

    orig = (4000, 3000)
    inf = (1000, 750)
    ix, iy = map_point_original_to_inference(2000.0, 1500.0, orig, inf)
    assert (ix, iy) == pytest.approx((500.0, 375.0))
    ox, oy = map_point_inference_to_original(ix, iy, inf, orig)
    assert (ox, oy) == pytest.approx((2000.0, 1500.0))


def test_geometry_result_bundle() -> None:
    depth = DepthMap(np.ones((4, 6), dtype=np.float32))
    normals = NormalMap(np.tile([0.0, 0.0, -1.0], (4, 6, 1)).astype(np.float32))
    points = Points3D(np.zeros((4, 6, 3), dtype=np.float32))
    intrinsics = CameraIntrinsics(np.eye(3, dtype=np.float32))
    mask = ValidMask(np.ones((4, 6), dtype=bool))

    res = GeometryResult(
        depth=depth,
        normals=normals,
        points=points,
        intrinsics=intrinsics,
        mask=mask,
    )
    assert res.shape == (4, 6)
    assert res.width == 6
    assert res.height == 4


def test_smooth_geometry_reduces_noise_and_respects_occlusions() -> None:
    height, width = 10, 16
    intrinsics = CameraIntrinsics.from_pixel_matrix(
        np.array([[20.0, 0.0, width / 2], [0.0, 20.0, height / 2], [0.0, 0.0, 1.0]]),
        width,
        height,
    )
    cols = np.arange(width)
    rows = np.arange(height)[:, None]
    base_depth = np.where(cols < width // 2, 2.0, 6.0).astype(np.float32)[None, :]
    noise = (0.04 * (((rows + cols) % 2) * 2 - 1)).astype(np.float32)
    noisy_depth = np.where(base_depth < 4.0, base_depth + noise, base_depth + noise).astype(np.float32)
    valid = np.ones((height, width), dtype=bool)
    valid[0, 0] = False

    noisy_points = intrinsics.unproject_depth(np.where(valid, noisy_depth, np.nan))
    geometry = GeometryResult(
        depth=DepthMap(noisy_depth),
        normals=NormalMap.from_points(noisy_points),
        points=Points3D(noisy_points),
        intrinsics=intrinsics,
        mask=ValidMask(valid),
    )

    assert smooth_geometry(geometry, iterations=0) is geometry

    smoothed = smooth_geometry(geometry, iterations=3)
    assert smoothed.normals is not None
    # Noise on each side is reduced cleanly without smearing the 2m <-> 6m occlusion step.
    left_err_before = np.mean(np.abs(noisy_depth[1:-1, 1 : width // 2 - 1] - 2.0))
    left_err_after = np.mean(np.abs(smoothed.depth.data[1:-1, 1 : width // 2 - 1] - 2.0))
    assert left_err_after < left_err_before * 0.4
    assert np.min(smoothed.depth.data[:, width // 2 :]) > 5.8
    assert np.max(smoothed.depth.data[:, : width // 2]) < 2.2
    assert np.isnan(smoothed.points.data[0, 0, 2])
    np.testing.assert_allclose(
        smoothed.points.data[valid, 2], smoothed.depth.data[valid], atol=1e-6,
    )
    unit_lengths = np.linalg.norm(smoothed.normals.data[1:-1, 1:-1], axis=-1)
    np.testing.assert_allclose(unit_lengths, 1.0, atol=1e-5)
