from pathlib import Path
import numpy as np

from app.cache.geometry_cache import (
    GeometryCache,
    InferenceParams,
    build_cache_key,
    build_metadata,
    compute_image_sha256,
)
from app.geometry.depth import DepthMap
from app.geometry.intrinsics import CameraIntrinsics
from app.geometry.mask import ValidMask
from app.geometry.normals import NormalMap
from app.geometry.points import Points3D
from app.geometry.result import GeometryResult


def _make_sample_result() -> GeometryResult:
    depth = DepthMap(np.array([[1.2, 2.4], [3.6, 4.8]], dtype=np.float32))
    normals = NormalMap(
        np.array(
            [
                [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            ],
            dtype=np.float32,
        )
    )
    points = Points3D(
        np.array(
            [
                [[-0.2, -0.1, 1.2], [0.2, -0.1, 2.4]],
                [[-0.2, 0.1, 3.6], [0.2, 0.1, 4.8]],
            ],
            dtype=np.float32,
        )
    )
    intrinsics = CameraIntrinsics(
        np.array([[0.9, 0.0, 0.5], [0.0, 0.9, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    )
    mask = ValidMask(np.array([[True, True], [True, False]], dtype=bool))
    return GeometryResult(
        depth=depth,
        normals=normals,
        points=points,
        intrinsics=intrinsics,
        mask=mask,
    )


def test_cache_key_changes_with_image_and_settings() -> None:
    sha_a = compute_image_sha256(b"image-a")
    sha_b = compute_image_sha256(b"image-b")
    params_1 = InferenceParams("Ruicheng/moge-3-vitl", 9, 3, True, 1024)
    params_2 = InferenceParams("Ruicheng/moge-3-vitg", 9, 3, True, 1024)
    params_3 = InferenceParams("Ruicheng/moge-3-vitl", 7, 3, True, 1024)

    assert build_cache_key(sha_a, params_1) == build_cache_key(sha_a, params_1)
    assert build_cache_key(sha_a, params_1) != build_cache_key(sha_b, params_1)
    assert build_cache_key(sha_a, params_1) != build_cache_key(sha_a, params_2)
    assert build_cache_key(sha_a, params_1) != build_cache_key(sha_a, params_3)


def test_cache_round_trip_and_binary_maps(tmp_path: Path) -> None:
    cache = GeometryCache(tmp_path / "cache")
    params = InferenceParams("Ruicheng/moge-3-vitl", 9, 3, True, 1024)
    sha = compute_image_sha256(b"sample-bytes")
    cache_id = build_cache_key(sha, params)

    assert not cache.has(cache_id)
    assert cache.get_metadata(cache_id) is None
    assert cache.load_result(cache_id) is None

    result = _make_sample_result()
    meta = build_metadata(cache_id, sha, params, (4, 4), result)
    cache.put(cache_id, meta, result)

    assert cache.has(cache_id)
    loaded_meta = cache.get_metadata(cache_id)
    assert loaded_meta == meta

    loaded_res = cache.load_result(cache_id)
    assert loaded_res is not None
    np.testing.assert_allclose(loaded_res.depth.data, result.depth.data)
    assert loaded_res.normals is not None
    assert result.normals is not None
    np.testing.assert_allclose(loaded_res.normals.data, result.normals.data)
    np.testing.assert_allclose(loaded_res.points.data, result.points.data)
    np.testing.assert_allclose(loaded_res.intrinsics.matrix, result.intrinsics.matrix)
    np.testing.assert_array_equal(loaded_res.mask.data, result.mask.data)

    depth_payload = cache.load_map_bytes(cache_id, "depth")
    assert depth_payload is not None
    depth_bytes, depth_shape, depth_dtype = depth_payload
    assert depth_shape == (2, 2)
    assert depth_dtype == "float32"
    np.testing.assert_allclose(
        np.frombuffer(depth_bytes, dtype="<f4").reshape(2, 2),
        result.depth.data,
    )

    mask_payload = cache.load_map_bytes(cache_id, "mask")
    assert mask_payload is not None
    mask_bytes, mask_shape, mask_dtype = mask_payload
    assert mask_shape == (2, 2)
    assert mask_dtype == "uint8"
    np.testing.assert_array_equal(
        np.frombuffer(mask_bytes, dtype=np.uint8).reshape(2, 2),
        result.mask.data.astype(np.uint8),
    )
