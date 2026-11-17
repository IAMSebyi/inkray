import base64
from dataclasses import asdict
import math
import threading
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
import numpy as np
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..cache.geometry_cache import (
    GeometryCache,
    GeometryRecordMetadata,
    InferenceParams,
    VALID_MAP_KINDS,
    build_cache_key,
    build_metadata,
    compute_image_sha256,
)
from ..config import Settings
from ..geometry.result import GeometryResult
from ..geometry.surface_selector import SurfaceSelectionOutcome, SurfaceSelector
from ..inference.memory import is_cuda_oom, prepare_model_for_inference
from ..utils.image import (
    ImageTooLargeError,
    InvalidImageError,
    decode_image_bytes,
    validate_image_pixel_count,
)
from .uploads import read_image_upload

router = APIRouter(prefix="/api/geometry", tags=["geometry"])


class SelectSurfaceRequest(BaseModel):
    image_x: float
    image_y: float


def _inference_params(settings: Settings) -> InferenceParams:
    return InferenceParams(
        model_id=settings.model_id,
        resolution_level=settings.resolution_level,
        refine_steps=settings.refine_steps,
        use_fp16=settings.use_fp16,
        max_image_size=settings.max_image_size,
    )


def _infer_and_cache(
    model: object,
    peer_model: object | None,
    offload_inactive: bool,
    cache: GeometryCache,
    lock: threading.Lock,
    cache_id: str,
    image_sha256: str,
    params: InferenceParams,
    raw_bytes: bytes,
    max_image_pixels: int,
) -> tuple[dict[str, object], bool]:
    with lock:
        existing = cache.get_metadata(cache_id)
        if existing is not None:
            validate_image_pixel_count(
                existing.original_width, existing.original_height, max_image_pixels,
            )
            return (asdict(existing), True)

        decoded = decode_image_bytes(
            raw_bytes, max_image_size=params.max_image_size, max_pixels=max_image_pixels,
        )

        prepare_model_for_inference(
            active_model=model,
            peer_models=(peer_model,),
            offload_inactive=offload_inactive,
        )
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
    settings: Settings = request.app.state.settings
    raw_bytes = await read_image_upload(file, settings.max_upload_bytes)
    cache: GeometryCache = request.app.state.geometry_cache
    model = request.app.state.inference_model
    peer_model = getattr(request.app.state, "appearance_model", None)
    lock: threading.Lock = request.app.state.inference_lock

    params = _inference_params(settings)
    image_sha256 = compute_image_sha256(raw_bytes)
    cache_id = build_cache_key(image_sha256, params)

    cached_meta = await run_in_threadpool(cache.get_metadata, cache_id)
    try:
        if cached_meta is not None:
            validate_image_pixel_count(
                cached_meta.original_width, cached_meta.original_height, settings.max_image_pixels,
            )
            return {**asdict(cached_meta), "cached": True}

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
            settings.max_image_pixels,
        )
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if is_cuda_oom(exc):
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


def _load_geometry_record(
    request: Request,
    geometry_id: str,
) -> tuple[GeometryRecordMetadata, GeometryResult] | None:
    cached = getattr(request.app.state, "last_loaded_geometry", None)
    if cached is not None and cached[0] == geometry_id:
        return (cached[1], cached[2])

    cache: GeometryCache = request.app.state.geometry_cache
    metadata = cache.get_metadata(geometry_id)
    if metadata is None:
        return None
    result = cache.load_result(geometry_id)
    if result is None:
        return None

    request.app.state.last_loaded_geometry = (geometry_id, metadata, result)
    return (metadata, result)


def _serialize_surface_outcome(outcome: SurfaceSelectionOutcome) -> dict[str, object]:
    if not outcome.ok or outcome.selection is None:
        return {
            "ok": False,
            "reason": outcome.reason or "Failed to select planar surface.",
        }

    sel = outcome.selection
    mask_b64 = base64.b64encode(sel.support_mask.astype(np.uint8).tobytes()).decode("ascii")
    return {
        "ok": True,
        "selection": {
            "seed_image_point": {
                "x": float(sel.seed_image_point[0]),
                "y": float(sel.seed_image_point[1]),
            },
            "seed_inference_pixel": {
                "x": int(sel.seed_inference_pixel[0]),
                "y": int(sel.seed_inference_pixel[1]),
            },
            "seed_point_3d": [float(v) for v in sel.seed_point_3d],
            "plane": {
                "origin": [float(v) for v in sel.plane.origin],
                "normal": [float(v) for v in sel.plane.normal],
            },
            "frame": {
                "origin": [float(v) for v in sel.frame.origin],
                "normal": [float(v) for v in sel.frame.normal],
                "u_axis": [float(v) for v in sel.frame.u_axis],
                "v_axis": [float(v) for v in sel.frame.v_axis],
            },
            "support_mask_b64": mask_b64,
            "support_width": int(sel.support_mask.shape[1]),
            "support_height": int(sel.support_mask.shape[0]),
            "inlier_count": sel.inlier_count,
            "rms_error": float(sel.rms_error),
            "suggested_footprint": {
                "min_u": float(sel.suggested_footprint.min_u),
                "max_u": float(sel.suggested_footprint.max_u),
                "min_v": float(sel.suggested_footprint.min_v),
                "max_v": float(sel.suggested_footprint.max_v),
            },
            "reference_brush_diameter": float(sel.reference_brush_diameter),
        },
    }


@router.post("/{geometry_id}/surface/select")
async def select_planar_surface(
    request: Request,
    geometry_id: str,
    body: SelectSurfaceRequest,
) -> dict[str, object]:
    if not (math.isfinite(body.image_x) and math.isfinite(body.image_y)):
        raise HTTPException(status_code=400, detail="Surface click coordinates must be finite.")

    record = await run_in_threadpool(_load_geometry_record, request, geometry_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Geometry record not found.")

    metadata, result = record
    outcome = await run_in_threadpool(
        SurfaceSelector.select_at_image_point,
        result,
        (body.image_x, body.image_y),
        (metadata.original_width, metadata.original_height),
    )
    return _serialize_surface_outcome(outcome)
