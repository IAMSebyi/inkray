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
    appearance_model_id: str = os.getenv(
        "INKRAY_APPEARANCE_MODEL_ID", "prs-eth/marigold-iid-lighting-v1-1"
    )
    appearance_use_fp16: bool = os.getenv("INKRAY_APPEARANCE_FP16", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    appearance_denoising_steps: int = int(os.getenv("INKRAY_APPEARANCE_STEPS", "4"))
    appearance_processing_resolution: int = int(os.getenv("INKRAY_APPEARANCE_RES", "768"))
    appearance_ensemble_size: int = int(os.getenv("INKRAY_APPEARANCE_ENSEMBLE", "1"))
    appearance_seed: int = int(os.getenv("INKRAY_APPEARANCE_SEED", "42"))
    appearance_max_image_size: int = int(os.getenv("INKRAY_APPEARANCE_MAX_IMAGE_SIZE", "1024"))
    offload_inactive_models: bool = os.getenv("INKRAY_OFFLOAD_INACTIVE_MODELS", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    cache_dir: Path = Path(
        os.getenv(
            "INKRAY_CACHE_DIR",
            str(Path(__file__).resolve().parents[1] / ".cache" / "geometry"),
        )
    )
    appearance_cache_dir: Path = Path(
        os.getenv(
            "INKRAY_APPEARANCE_CACHE_DIR",
            str(Path(__file__).resolve().parents[1] / ".cache" / "appearance"),
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
