from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Normalized 3x3 pinhole intrinsics matrix in OpenCV image convention."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.matrix, dtype=np.float32)
        if arr.shape != (3, 3):
            raise ValueError(f"CameraIntrinsics expects shape (3, 3), got {arr.shape}.")
        object.__setattr__(self, "matrix", np.ascontiguousarray(arr))

    @property
    def fx_norm(self) -> float:
        return float(self.matrix[0, 0])

    @property
    def fy_norm(self) -> float:
        return float(self.matrix[1, 1])

    @property
    def cx_norm(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def cy_norm(self) -> float:
        return float(self.matrix[1, 2])

    def to_pixel_matrix(self, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
        px = self.matrix.copy()
        px[0, :] *= float(width)
        px[1, :] *= float(height)
        return px

    @classmethod
    def from_pixel_matrix(
        cls,
        pixel_matrix: np.ndarray,
        width: int,
        height: int,
    ) -> "CameraIntrinsics":
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
        arr = np.asarray(pixel_matrix, dtype=np.float32).copy()
        if arr.shape != (3, 3):
            raise ValueError(f"Expected shape (3, 3), got {arr.shape}.")
        arr[0, :] /= float(width)
        arr[1, :] /= float(height)
        return cls(arr)

    def fov_x_degrees(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 / self.fx_norm))

    def fov_y_degrees(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 / self.fy_norm))

    def project(self, points_3d: np.ndarray, width: int, height: int) -> np.ndarray:
        """Projects OpenCV camera-space 3D points (..., 3) into pixel coordinates (..., 2)."""
        pts = np.asarray(points_3d, dtype=np.float32)
        if pts.shape[-1] != 3:
            raise ValueError(f"Expected last dimension 3, got shape {pts.shape}.")
        k_px = self.to_pixel_matrix(width, height)
        z = pts[..., 2]
        x = pts[..., 0] / z
        y = pts[..., 1] / z
        u = k_px[0, 0] * x + k_px[0, 1] * y + k_px[0, 2]
        v = k_px[1, 1] * y + k_px[1, 2]
        return np.stack([u, v], axis=-1).astype(np.float32)

    def unproject(self, u: float, v: float, depth: float, width: int, height: int) -> np.ndarray:
        """Unprojects a pixel coordinate (u, v) at metric depth into OpenCV camera space."""
        k_px = self.to_pixel_matrix(width, height)
        fx = float(k_px[0, 0])
        fy = float(k_px[1, 1])
        skew = float(k_px[0, 1])
        cx = float(k_px[0, 2])
        cy = float(k_px[1, 2])
        y_norm = (v - cy) / fy
        x_norm = (u - cx - skew * y_norm) / fx
        return np.array([x_norm * depth, y_norm * depth, depth], dtype=np.float32)

    def unproject_depth(self, depth: np.ndarray) -> np.ndarray:
        height, width = depth.shape
        return self.pixel_rays(width, height) * depth[..., None]

    def pixel_rays(self, width: int, height: int) -> np.ndarray:
        k = self.to_pixel_matrix(width, height)
        u = np.arange(width, dtype=np.float32) + 0.5
        v = np.arange(height, dtype=np.float32) + 0.5
        y = (v[:, None] - k[1, 2]) / k[1, 1]
        x = (u - k[0, 2] - k[0, 1] * y) / k[0, 0]
        return np.stack((x, np.broadcast_to(y, x.shape), np.ones_like(x)), axis=-1)
