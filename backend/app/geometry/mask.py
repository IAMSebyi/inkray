from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ValidMask:
    data: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=bool)
        if arr.ndim != 2:
            raise ValueError(f"ValidMask expects a 2D array, got shape {arr.shape}.")
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

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.data))

    def at_pixel(self, col: int, row: int) -> bool:
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return bool(self.data[row, col])

    def is_valid(self, x: float, y: float) -> bool:
        # Masks are discrete per-pixel flags and should not be bilinearly blended.
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if x < 0.0 or x >= self.width or y < 0.0 or y >= self.height:
            return False
        return bool(self.data[int(math.floor(y)), int(math.floor(x))])
