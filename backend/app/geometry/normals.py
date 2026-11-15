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
