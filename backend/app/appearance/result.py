from dataclasses import dataclass, field
import numpy as np

from .maps import AlbedoMap, ResidualMap, ShadingMap


def default_scale_conventions() -> dict[str, str]:
    return {
        "albedo": "bounded_0_1",
        "shading": "up_to_scale",
        "residual": "up_to_scale",
    }


@dataclass(frozen=True)
class AppearanceResult:
    albedo: AlbedoMap
    model_id: str
    shading: ShadingMap | None = None
    residual: ResidualMap | None = None
    denoising_steps: int = 0
    processing_resolution: int = 0
    ensemble_size: int = 1
    seed: int = 0
    prediction_space: str = "linear"
    scale_conventions: dict[str, str] = field(default_factory=default_scale_conventions)

    def __post_init__(self) -> None:
        for layer in (self.shading, self.residual):
            if layer is not None and layer.shape != self.albedo.shape:
                raise ValueError("Appearance maps must share matching spatial dimensions.")

    @classmethod
    def from_albedo(
        cls,
        albedo: np.ndarray,
        model_id: str,
        processing_resolution: int = 0,
        denoising_steps: int = 0,
        seed: int = 0,
        scale_convention: str = "bounded_0_1",
    ) -> "AppearanceResult":
        if not np.all(np.isfinite(albedo)):
            raise RuntimeError(f"{model_id} returned non-finite albedo values.")
        return cls(
            albedo=AlbedoMap(np.clip(albedo, 0, 1)),
            model_id=model_id,
            denoising_steps=denoising_steps,
            processing_resolution=processing_resolution,
            seed=seed,
            scale_conventions={"albedo": scale_convention},
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.albedo.height, self.albedo.width)

    @property
    def height(self) -> int:
        return self.albedo.height

    @property
    def width(self) -> int:
        return self.albedo.width
