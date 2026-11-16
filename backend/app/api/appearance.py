from dataclasses import asdict
import threading
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from PIL import UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from ..cache.appearance_cache import (
    AppearanceCache,
    AppearanceInferenceParams,
    VALID_APPEARANCE_MAP_KINDS,
    build_appearance_cache_key,
    build_appearance_metadata,
    compute_image_sha256,
)
from ..config import Settings
from ..inference.memory import is_cuda_oom, prepare_model_for_inference
from ..utils.image import decode_image_bytes

router = APIRouter(prefix="/api/appearance", tags=["appearance"])


def _appearance_params(settings: Settings) -> AppearanceInferenceParams:
    return AppearanceInferenceParams(
        model_id=settings.appearance_model_id,
        denoising_steps=settings.appearance_denoising_steps,
        processing_resolution=settings.appearance_processing_resolution,
        ensemble_size=settings.appearance_ensemble_size,
        seed=settings.appearance_seed,
        use_fp16=settings.appearance_use_fp16,
        max_image_size=settings.appearance_max_image_size,
    )


def _infer_and_cache(
    model: object,
    peer_model: object | None,
    offload_inactive: bool,
    cache: AppearanceCache,
    lock: threading.Lock,
    cache_id: str,
    image_sha256: str,
    params: AppearanceInferenceParams,
    raw_bytes: bytes,
) -> tuple[dict[str, object], bool]:
    decoded = decode_image_bytes(raw_bytes, max_image_size=params.max_image_size)

    with lock:
        existing = cache.get_metadata(cache_id)
        if existing is not None:
            return (asdict(existing), True)

        prepare_model_for_inference(
            active_model=model,
            peer_models=(peer_model,),
            offload_inactive=offload_inactive,
        )
        result = model.run(decoded.rgb)  # type: ignore[attr-defined]
        metadata = build_appearance_metadata(
            cache_id=cache_id,
            image_sha256=image_sha256,
            params=params,
            original_size=(decoded.original_width, decoded.original_height),
            result=result,
        )
        cache.put(cache_id, metadata, result)
        return (asdict(metadata), False)


@router.post("/infer")
async def infer_appearance(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    settings: Settings = request.app.state.settings
    cache: AppearanceCache = request.app.state.appearance_cache
    model = request.app.state.appearance_model
    peer_model = getattr(request.app.state, "inference_model", None)
    lock: threading.Lock = request.app.state.inference_lock

    params = _appearance_params(settings)
    image_sha256 = compute_image_sha256(raw_bytes)
    cache_id = build_appearance_cache_key(image_sha256, params)

    cached_meta = cache.get_metadata(cache_id)
    if cached_meta is not None:
        return {**asdict(cached_meta), "cached": True}

    try:
        metadata_dict, was_cached = await run_in_threadpool(
            _infer_and_cache,
            model,
            peer_model,
            settings.offload_inactive_models,
            cache,
            lock,
            cache_id,
            image_sha256,
            params,
            raw_bytes,
        )
    except (ValueError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if is_cuda_oom(exc):
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory during appearance inference. Try a smaller max image size.",
            ) from exc
        raise

    return {**metadata_dict, "cached": was_cached}


@router.get("/{appearance_id}")
def get_appearance_metadata(request: Request, appearance_id: str) -> dict[str, object]:
    cache: AppearanceCache = request.app.state.appearance_cache
    metadata = cache.get_metadata(appearance_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Appearance record not found.")
    return {**asdict(metadata), "cached": True}


@router.get("/{appearance_id}/maps/{kind}")
def get_appearance_map(request: Request, appearance_id: str, kind: str) -> Response:
    if kind not in VALID_APPEARANCE_MAP_KINDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported appearance map kind {kind!r}. Expected one of: "
                f"{', '.join(VALID_APPEARANCE_MAP_KINDS)}."
            ),
        )

    cache: AppearanceCache = request.app.state.appearance_cache
    payload = cache.load_map_bytes(appearance_id, kind)  # type: ignore[arg-type]
    if payload is None:
        raise HTTPException(status_code=404, detail="Appearance map not found.")

    raw_bytes, shape, dtype = payload
    return Response(
        content=raw_bytes,
        media_type="application/octet-stream",
        headers={
            "X-Array-Shape": ",".join(str(dim) for dim in shape),
            "X-Array-Dtype": dtype,
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )
