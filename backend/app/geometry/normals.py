from dataclasses import dataclass
import numpy as np

from .mask import ValidMask
from .transforms import opencv_normals_to_threejs, sample_bilinear_masked


@dataclass(frozen=True)
class NormalMap:
    data: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"NormalMap expects shape (H, W, 3), got {arr.shape}.")
        object.__setattr__(self, "data", np.ascontiguousarray(arr))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]), 3)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def at_pixel(self, col: int, row: int, mask: ValidMask | None = None) -> np.ndarray | None:
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        if mask is not None and not mask.data[row, col]:
            return None
        vec = self.data[row, col]
        if not np.all(np.isfinite(vec)):
            return None
        return vec.copy()

    def sample(self, x: float, y: float, mask: ValidMask | None = None) -> np.ndarray | None:
        vec = sample_bilinear_masked(self.data, x, y, mask)
        if vec is None:
            return None
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return None
        return (vec / norm).astype(np.float32)

    def to_threejs(self) -> np.ndarray:
        return opencv_normals_to_threejs(self.data)

    @classmethod
    def from_camera_space(cls, normals: np.ndarray) -> "NormalMap":
        """Convert Y-up, Z-toward-viewer normals (Marigold convention) into Inkray OpenCV space."""
        return cls(np.asarray(normals, dtype=np.float32) * np.array([1.0, -1.0, -1.0], dtype=np.float32))

    @classmethod
    def from_points(cls, points: np.ndarray) -> "NormalMap":
        def tangent(axis: int) -> np.ndarray:
            grid = np.moveaxis(points, axis, 0)
            forward = np.full_like(grid, np.nan)
            backward = np.full_like(grid, np.nan)
            forward[:-1] = backward[1:] = grid[1:] - grid[:-1]
            # Prefer the smaller depth change when choosing a neighbor.
            ahead = np.nan_to_num(np.abs(forward[..., 2]), nan=np.inf)
            behind = np.nan_to_num(np.abs(backward[..., 2]), nan=np.inf)
            return np.moveaxis(np.where((ahead <= behind)[..., None], forward, backward), 0, axis)

        normals = np.cross(tangent(0), tangent(1))
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        np.divide(normals, lengths, out=normals, where=np.isfinite(lengths) & (lengths > 0))
        normals[~np.isfinite(lengths[..., 0]) | (lengths[..., 0] == 0)] = np.nan
        return cls(normals)
