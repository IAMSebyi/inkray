from dataclasses import asdict
import threading
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from PIL import UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
import torch

from ..cache.geometry_cache import (
    GeometryCache,
    InferenceParams,
    VALID_MAP_KINDS,
    build_cache_key,
    build_metadata,
    compute_image_sha256,
)
from ..config import Settings
from ..utils.image import decode_image_bytes

router = APIRouter(prefix="/api/geometry", tags=["geometry"])


def _inference_params(settings: Settings) -> InferenceParams:
    return InferenceParams(
        model_id=settings.model_id,
        resolution_level=settings.resolution_level,
        refine_steps=settings.refine_steps,
        use_fp16=settings.use_fp16,
        max_image_size=settings.max_image_size,
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _infer_and_cache(
    model: object,
    cache: GeometryCache,
    lock: threading.Lock,
    cache_id: str,
    image_sha256: str,
    params: InferenceParams,
    raw_bytes: bytes,
) -> tuple[dict[str, object], bool]:
    decoded = decode_image_bytes(raw_bytes, max_image_size=params.max_image_size)

    with lock:
        existing = cache.get_metadata(cache_id)
        if existing is not None:
            return (asdict(existing), True)

        result = model.run(decoded.rgb)  # type: ignore[attr-defined]
        metadata = build_metadata(
            cache_id=cache_id,
            image_sha256=image_sha256,
            params=params,
            original_size=(decoded.original_width, decoded.original_height),
            result=result,
        )
        cache.put(cache_id, metadata, result)
        return (asdict(metadata), False)


@router.post("/infer")
async def infer_geometry(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    settings: Settings = request.app.state.settings
    cache: GeometryCache = request.app.state.geometry_cache
    model = request.app.state.inference_model
    lock: threading.Lock = request.app.state.inference_lock

    params = _inference_params(settings)
    image_sha256 = compute_image_sha256(raw_bytes)
    cache_id = build_cache_key(image_sha256, params)

    cached_meta = cache.get_metadata(cache_id)
    if cached_meta is not None:
        return {**asdict(cached_meta), "cached": True}

    try:
        metadata_dict, was_cached = await run_in_threadpool(
            _infer_and_cache,
            model,
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
        if _is_cuda_oom(exc):
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory during geometry inference. Try a smaller max image size.",
            ) from exc
        raise

    return {**metadata_dict, "cached": was_cached}


@router.get("/{geometry_id}")
def get_geometry_metadata(request: Request, geometry_id: str) -> dict[str, object]:
    cache: GeometryCache = request.app.state.geometry_cache
    metadata = cache.get_metadata(geometry_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Geometry record not found.")
    return {**asdict(metadata), "cached": True}


@router.get("/{geometry_id}/maps/{kind}")
def get_geometry_map(request: Request, geometry_id: str, kind: str) -> Response:
    if kind not in VALID_MAP_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported map kind {kind!r}. Expected one of: {', '.join(VALID_MAP_KINDS)}.",
        )

    cache: GeometryCache = request.app.state.geometry_cache
    payload = cache.load_map_bytes(geometry_id, kind)  # type: ignore[arg-type]
    if payload is None:
        raise HTTPException(status_code=404, detail="Geometry map not found.")

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
