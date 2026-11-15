from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("INKRAY_MOGE_MODEL_ID", "Ruicheng/moge-3-vitl")
    device: str = os.getenv("INKRAY_DEVICE", "auto")
    use_fp16: bool = os.getenv("INKRAY_MOGE_FP16", "true").lower() in ("1", "true", "yes")
    resolution_level: int = int(os.getenv("INKRAY_MOGE_RESOLUTION_LEVEL", "9"))
    refine_steps: int = int(os.getenv("INKRAY_MOGE_REFINE_STEPS", "3"))
    max_image_size: int = int(os.getenv("INKRAY_MAX_IMAGE_SIZE", "1024"))
    cache_dir: Path = Path(
        os.getenv(
            "INKRAY_CACHE_DIR",
            str(Path(__file__).resolve().parents[1] / ".cache" / "geometry"),
        )
    )
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "INKRAY_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )
