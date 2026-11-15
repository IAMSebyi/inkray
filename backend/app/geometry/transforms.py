import math
import numpy as np

from .mask import ValidMask

# OpenCV camera space (+x right, +y down, +z forward) to
# Three.js / WebGL space (+x right, +y up, -z forward).
# Determinant is +1 (180-deg rotation around X), so normals transform identically.
OPENCV_TO_THREEJS = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
THREEJS_TO_OPENCV = OPENCV_TO_THREEJS


def opencv_points_to_threejs(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dimension 3, got shape {arr.shape}.")
    return np.ascontiguousarray(arr @ OPENCV_TO_THREEJS.T)


def threejs_points_to_opencv(points: np.ndarray) -> np.ndarray:
    return opencv_points_to_threejs(points)


def opencv_normals_to_threejs(normals: np.ndarray) -> np.ndarray:
    return opencv_points_to_threejs(normals)


def compute_inference_size(width: int, height: int, max_image_size: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
    if max_image_size <= 0 or max(width, height) <= max_image_size:
        return (width, height)
    scale = max_image_size / float(max(width, height))
    return (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )


def map_point_original_to_inference(
    x: float,
    y: float,
    original_size: tuple[int, int],
    inference_size: tuple[int, int],
) -> tuple[float, float]:
    orig_w, orig_h = original_size
    inf_w, inf_h = inference_size
    if orig_w <= 0 or orig_h <= 0 or inf_w <= 0 or inf_h <= 0:
        raise ValueError("Image dimensions must be positive.")
    return (x * (inf_w / float(orig_w)), y * (inf_h / float(orig_h)))


def map_point_inference_to_original(
    x: float,
    y: float,
    inference_size: tuple[int, int],
    original_size: tuple[int, int],
) -> tuple[float, float]:
    return map_point_original_to_inference(x, y, inference_size, original_size)


def sample_bilinear_masked(
    data: np.ndarray,
    x: float,
    y: float,
    mask: ValidMask | None = None,
) -> np.ndarray | None:
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    height, width = data.shape[0], data.shape[1]
    if x < 0.0 or x >= width or y < 0.0 or y >= height:
        return None

    # Pixel (c, r) center sits at (c + 0.5, r + 0.5).
    u = min(max(x - 0.5, 0.0), float(width - 1))
    v = min(max(y - 0.5, 0.0), float(height - 1))

    x0 = int(math.floor(u))
    y0 = int(math.floor(v))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    wx = u - x0
    wy = v - y0

    corners = (
        (y0, x0, (1.0 - wy) * (1.0 - wx)),
        (y0, x1, (1.0 - wy) * wx),
        (y1, x0, wy * (1.0 - wx)),
        (y1, x1, wy * wx),
    )

    accum = np.zeros(data.shape[2:], dtype=np.float64)
    weight_sum = 0.0
    fallback_value: np.ndarray | None = None
    fallback_weight = -1.0

    for row, col, weight in corners:
        if mask is not None and not mask.data[row, col]:
            continue
        val = data[row, col]
        if not np.all(np.isfinite(val)):
            continue
        if weight > fallback_weight:
            fallback_weight = weight
            fallback_value = np.asarray(val, dtype=np.float64)
        if weight > 0.0:
            accum += val * weight
            weight_sum += weight

    if weight_sum > 0.0:
        return (accum / weight_sum).astype(np.float32)
    if fallback_value is not None:
        return fallback_value.astype(np.float32)
    return None
