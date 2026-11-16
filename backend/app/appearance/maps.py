from dataclasses import dataclass
import numpy as np

from ..geometry.transforms import sample_bilinear_masked


@dataclass(frozen=True)
class RgbFloatMap:
    data: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"{type(self).__name__} expects an (H, W, 3) array, got shape {arr.shape}."
            )
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

    def at_pixel(self, col: int, row: int) -> tuple[float, float, float] | None:
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        val = self.data[row, col]
        if not np.all(np.isfinite(val)):
            return None
        return (float(val[0]), float(val[1]), float(val[2]))

    def sample(self, x: float, y: float) -> tuple[float, float, float] | None:
        sampled = sample_bilinear_masked(self.data, x, y)
        if sampled is None:
            return None
        return (float(sampled[0]), float(sampled[1]), float(sampled[2]))

    def value_range(self) -> tuple[float, float] | None:
        valid = np.isfinite(self.data)
        if not np.any(valid):
            return None
        values = self.data[valid]
        return (float(np.min(values)), float(np.max(values)))


@dataclass(frozen=True)
class AlbedoMap(RgbFloatMap):
    """Intrinsic diffuse reflectance map per pixel in linear [0, 1] space."""


@dataclass(frozen=True)
class ShadingMap(RgbFloatMap):
    """Diffuse illumination map per pixel predicted up to an unknown global scale."""


@dataclass(frozen=True)
class ResidualMap(RgbFloatMap):
    """Non-diffuse appearance residual (e.g., specular highlights or unmodeled lighting)."""
