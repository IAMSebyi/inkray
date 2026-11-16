from dataclasses import dataclass, field

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
    shading: ShadingMap
    residual: ResidualMap
    model_id: str
    denoising_steps: int
    processing_resolution: int
    ensemble_size: int
    seed: int
    prediction_space: str = "linear"
    scale_conventions: dict[str, str] = field(default_factory=default_scale_conventions)

    def __post_init__(self) -> None:
        a_shape = (self.albedo.height, self.albedo.width)
        s_shape = (self.shading.height, self.shading.width)
        r_shape = (self.residual.height, self.residual.width)
        if a_shape != s_shape or a_shape != r_shape:
            raise ValueError(
                "Appearance maps must share matching spatial dimensions, got "
                f"albedo={a_shape}, shading={s_shape}, residual={r_shape}."
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
