from dataclasses import dataclass, field
import os
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[1] / ".cache" / "models"


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = int(os.getenv("INKRAY_MAX_UPLOAD_BYTES", "33554432"))
    max_image_pixels: int = int(os.getenv("INKRAY_MAX_IMAGE_PIXELS", "64000000"))
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
    normal_model_id: str = os.getenv(
        "INKRAY_NORMALS_MODEL_ID", "prs-eth/marigold-normals-v1-1"
    )
    normal_use_fp16: bool = os.getenv("INKRAY_NORMALS_FP16", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    normal_denoising_steps: int = int(os.getenv("INKRAY_NORMALS_STEPS", "4"))
    normal_processing_resolution: int = int(os.getenv("INKRAY_NORMALS_RES", "768"))
    normal_ensemble_size: int = int(os.getenv("INKRAY_NORMALS_ENSEMBLE", "1"))
    normal_seed: int = int(os.getenv("INKRAY_NORMALS_SEED", "42"))
    marigold_v2_repo: str = field(default_factory=lambda: os.getenv(
        "INKRAY_MARIGOLD_V2_REPO", "",
    ))
    marigold_v2_python: str = field(default_factory=lambda: os.getenv(
        "INKRAY_MARIGOLD_V2_PYTHON", "",
    ))
    latent_intrinsics_repo: str = field(default_factory=lambda: os.getenv(
        "INKRAY_LATENT_INTRINSICS_REPO", str(MODELS_DIR / "latent-intrinsics"),
    ))
    latent_intrinsics_checkpoint: str = field(default_factory=lambda: os.getenv(
        "INKRAY_LATENT_INTRINSICS_CHECKPOINT", "",
    ))
    iid_in_the_wild_weights: str = field(default_factory=lambda: os.getenv(
        "INKRAY_IID_IN_THE_WILD_WEIGHTS", "v2",
    ))
    iid_in_the_wild_base_size: int = int(os.getenv("INKRAY_IID_IN_THE_WILD_BASE_SIZE", "384"))
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
