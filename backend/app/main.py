from contextlib import asynccontextmanager
import threading
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from .api.appearance import router as appearance_router
from .api.geometry import router as geometry_router
from .appearance.result import AppearanceResult
from .cache.appearance_cache import AppearanceCache
from .cache.geometry_cache import GeometryCache
from .config import Settings
from .geometry.result import GeometryResult
from .inference.base import InferenceModel


def create_app(
    settings: Settings | None = None,
    inference_model: InferenceModel[np.ndarray, GeometryResult] | None = None,
    appearance_model: InferenceModel[np.ndarray, AppearanceResult] | None = None,
) -> FastAPI:
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        fastapi_app.state.settings = cfg
        fastapi_app.state.geometry_cache = GeometryCache(cfg.cache_dir)
        fastapi_app.state.appearance_cache = AppearanceCache(cfg.appearance_cache_dir)
        fastapi_app.state.inference_lock = threading.Lock()
        fastapi_app.state.last_loaded_geometry = None

        if inference_model is not None:
            fastapi_app.state.inference_model = inference_model
        else:
            from .inference.moge import MoGeInference

            fastapi_app.state.inference_model = MoGeInference(
                model_id=cfg.model_id,
                device=cfg.device,
                resolution_level=cfg.resolution_level,
                refine_steps=cfg.refine_steps,
                use_fp16=cfg.use_fp16,
            )

        if appearance_model is not None:
            fastapi_app.state.appearance_model = appearance_model
        else:
            from .inference.marigold import MarigoldInference

            fastapi_app.state.appearance_model = MarigoldInference(
                model_id=cfg.appearance_model_id,
                device=cfg.device,
                use_fp16=cfg.appearance_use_fp16,
                denoising_steps=cfg.appearance_denoising_steps,
                processing_resolution=cfg.appearance_processing_resolution,
                ensemble_size=cfg.appearance_ensemble_size,
                seed=cfg.appearance_seed,
                lazy_load=True,
            )

        yield

    fastapi_app = FastAPI(title="Inkray Backend", lifespan=lifespan)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Array-Shape", "X-Array-Dtype"],
    )
    fastapi_app.include_router(geometry_router)
    fastapi_app.include_router(appearance_router)
    return fastapi_app


app = create_app()
