from dataclasses import dataclass
import numpy as np

from .mask import ValidMask
from .transforms import sample_bilinear_masked


@dataclass(frozen=True)
class DepthMap:
    data: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"DepthMap expects a 2D array, got shape {arr.shape}.")
        object.__setattr__(self, "data", np.ascontiguousarray(arr))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def at_pixel(self, col: int, row: int, mask: ValidMask | None = None) -> float | None:
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        if mask is not None and not mask.data[row, col]:
            return None
        value = float(self.data[row, col])
        return value if np.isfinite(value) else None

    def sample(self, x: float, y: float, mask: ValidMask | None = None) -> float | None:
        sampled = sample_bilinear_masked(self.data, x, y, mask)
        if sampled is None:
            return None
        return float(sampled)

    def valid_range(self, mask: ValidMask | None = None) -> tuple[float, float] | None:
        valid = np.isfinite(self.data)
        if mask is not None:
            valid = valid & mask.data
        if not np.any(valid):
            return None
        values = self.data[valid]
        return (float(np.min(values)), float(np.max(values)))
