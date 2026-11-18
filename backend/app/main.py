from contextlib import asynccontextmanager
import threading
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from .api.appearance import router as appearance_router
from .api.geometry import router as geometry_router
from .appearance.result import AppearanceResult
from .cache.appearance_cache import AppearanceCache, AppearanceInferenceParams
from .cache.geometry_cache import GeometryCache, InferenceParams
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
        from .inference.depth_pro import DepthProInference
        from .inference.iid_in_the_wild import IidInTheWildInference
        from .inference.latent_intrinsics import LatentIntrinsicsInference
        from .inference.marigold import MarigoldInference, MarigoldNormalsInference
        from .inference.marigold_v2 import (
            MarigoldV2Inference,
            MarigoldV2NormalsInference,
            MarigoldV2Runner,
        )

        fastapi_app.state.settings = cfg
        fastapi_app.state.geometry_cache = GeometryCache(cfg.cache_dir)
        fastapi_app.state.appearance_cache = AppearanceCache(cfg.appearance_cache_dir)
        fastapi_app.state.inference_lock = threading.Lock()
        fastapi_app.state.last_loaded_geometry = None

        moge = inference_model
        moge_unavailable = None
        if moge is None:
            from .inference.moge import MoGeInference

            moge = MoGeInference(
                model_id=cfg.model_id,
                device=cfg.device,
                resolution_level=cfg.resolution_level,
                refine_steps=cfg.refine_steps,
                use_fp16=cfg.use_fp16,
                lazy_load=True,
            )
            moge_unavailable = moge.unavailable_reason()

        marigold = appearance_model
        if marigold is None:
            marigold = MarigoldInference(
                model_id=cfg.appearance_model_id,
                device=cfg.device,
                use_fp16=cfg.appearance_use_fp16,
                denoising_steps=cfg.appearance_denoising_steps,
                processing_resolution=cfg.appearance_processing_resolution,
                ensemble_size=cfg.appearance_ensemble_size,
                seed=cfg.appearance_seed,
                lazy_load=True,
            )

        depth_pro = DepthProInference(device=cfg.device)
        marigold_normals = MarigoldNormalsInference(
            model_id=cfg.normal_model_id,
            device=cfg.device,
            use_fp16=cfg.normal_use_fp16,
            denoising_steps=cfg.normal_denoising_steps,
            processing_resolution=cfg.normal_processing_resolution,
            ensemble_size=cfg.normal_ensemble_size,
            seed=cfg.normal_seed,
            lazy_load=True,
        )
        marigold_v2 = MarigoldV2Runner(cfg.marigold_v2_repo, cfg.marigold_v2_python, cfg.device)
        albedo = MarigoldV2Inference(marigold_v2)
        marigold_v2_normals = MarigoldV2NormalsInference(marigold_v2)
        latent = LatentIntrinsicsInference(
            cfg.latent_intrinsics_repo, cfg.latent_intrinsics_checkpoint, cfg.device,
        )
        iid_in_the_wild = IidInTheWildInference(
            weights=cfg.iid_in_the_wild_weights,
            base_size=cfg.iid_in_the_wild_base_size,
            device=cfg.device,
        )
        fastapi_app.state.normal_models = {
            "marigold-normals": marigold_normals,
            "marigold-v2": marigold_v2_normals,
        }
        fastapi_app.state.default_geometry_model = "depth-pro" if moge_unavailable else "moge-3"
        fastapi_app.state.default_normal_model = "marigold-normals"
        fastapi_app.state.unavailable_models = {
            name: reason
            for name, reason in (
                ("moge-3", moge_unavailable),
                ("marigold-v2", marigold_v2.unavailable_reason()),
                ("latent-intrinsics", latent.unavailable_reason()),
                ("iid-in-the-wild", iid_in_the_wild.unavailable_reason()),
            )
            if reason
        }
        fastapi_app.state.geometry_models = {
            "moge-3": (
                moge,
                InferenceParams(
                    model_id=cfg.model_id,
                    resolution_level=cfg.resolution_level,
                    refine_steps=cfg.refine_steps,
                    use_fp16=cfg.use_fp16,
                    max_image_size=cfg.max_image_size,
                ),
            ),
            "depth-pro": (
                depth_pro,
                InferenceParams(
                    model_id=depth_pro.model_id,
                    use_fp16=depth_pro.use_fp16,
                    max_image_size=cfg.max_image_size,
                ),
            ),
        }
        fastapi_app.state.appearance_models = {
            "marigold-iid": (
                marigold,
                AppearanceInferenceParams(
                    model_id=cfg.appearance_model_id,
                    denoising_steps=cfg.appearance_denoising_steps,
                    processing_resolution=cfg.appearance_processing_resolution,
                    ensemble_size=cfg.appearance_ensemble_size,
                    seed=cfg.appearance_seed,
                    use_fp16=cfg.appearance_use_fp16,
                    max_image_size=cfg.appearance_max_image_size,
                ),
            ),
            "marigold-v2": (
                albedo,
                AppearanceInferenceParams(
                    model_id=albedo.model_id,
                    denoising_steps=1,
                    seed=albedo.seed,
                    max_image_size=cfg.appearance_max_image_size,
                ),
            ),
            "latent-intrinsics": (
                latent,
                AppearanceInferenceParams(
                    model_id=latent.model_id,
                    processing_resolution=latent.patch_size,
                    max_image_size=cfg.appearance_max_image_size,
                    cache_variant=latent.cache_variant(),
                ),
            ),
            "iid-in-the-wild": (
                iid_in_the_wild,
                AppearanceInferenceParams(
                    model_id=iid_in_the_wild.model_id,
                    processing_resolution=iid_in_the_wild.base_size,
                    max_image_size=cfg.appearance_max_image_size,
                    cache_variant=iid_in_the_wild.weights,
                ),
            ),
        }
        fastapi_app.state.models = tuple(
            dict.fromkeys([
                *(model for model, _ in fastapi_app.state.geometry_models.values()),
                *(model for model, _ in fastapi_app.state.appearance_models.values()),
                *fastapi_app.state.normal_models.values(),
            ])
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

    @fastapi_app.get("/api/models")
    def get_models() -> dict[str, object]:
        return {
            "default_geometry": fastapi_app.state.default_geometry_model,
            "default_normal": fastapi_app.state.default_normal_model,
            "unavailable": fastapi_app.state.unavailable_models,
        }

    return fastapi_app


app = create_app()
