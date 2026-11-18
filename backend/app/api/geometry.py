import base64
from dataclasses import asdict, replace
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
from ..geometry.normal_refinement import refine_depth_with_normals
from ..geometry.normals import NormalMap
from ..geometry.result import GeometryResult
from ..geometry.smoothing import smooth_geometry
from ..geometry.surface_selector import SurfaceSelectionOutcome, SurfaceSelector
from ..inference.base import InferenceModel, ModelUnavailableError
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


def _infer_and_cache(
    model: InferenceModel[np.ndarray, GeometryResult],
    models: tuple[object, ...],
    offload_inactive: bool,
    cache: GeometryCache,
    lock: threading.Lock,
    cache_id: str,
    image_sha256: str,
    params: InferenceParams,
    raw_bytes: bytes,
    max_image_pixels: int,
    normal_model: InferenceModel[np.ndarray, NormalMap] | None,
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

        original_size = (decoded.original_width, decoded.original_height)
        unsmoothed_params = replace(params, smooth_iterations=0)
        unsmoothed_id = (
            build_cache_key(image_sha256, unsmoothed_params)
            if params.smooth_iterations > 0
            else cache_id
        )
        result = cache.load_result(unsmoothed_id) if params.smooth_iterations > 0 else None

        if result is None:
            base_params = replace(unsmoothed_params, normal_model_id=None)
            base_id = (
                build_cache_key(image_sha256, base_params)
                if normal_model is not None
                else unsmoothed_id
            )
            result = cache.load_result(base_id) if normal_model is not None else None
            if result is None:
                prepare_model_for_inference(model, models, offload_inactive)
                result = model.run(decoded.rgb)
                base_metadata = build_metadata(
                    base_id, image_sha256, base_params, original_size, result,
                )
                cache.put(base_id, base_metadata, result)
                if normal_model is None and params.smooth_iterations == 0:
                    return (asdict(base_metadata), False)

            if normal_model is not None:
                prepare_model_for_inference(normal_model, models, offload_inactive)
                result = refine_depth_with_normals(result, normal_model.run(decoded.rgb))
                unsmoothed_metadata = build_metadata(
                    unsmoothed_id, image_sha256, unsmoothed_params, original_size, result,
                )
                cache.put(unsmoothed_id, unsmoothed_metadata, result)
                if params.smooth_iterations == 0:
                    return (asdict(unsmoothed_metadata), False)

        refined_or_smoothed = (
            smooth_geometry(result, iterations=params.smooth_iterations)
            if params.smooth_iterations > 0
            else result
        )
        metadata = build_metadata(cache_id, image_sha256, params, original_size, refined_or_smoothed)
        cache.put(cache_id, metadata, refined_or_smoothed)
        return (asdict(metadata), False)


@router.post("/infer")
async def infer_geometry(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = None,
    refine_normals: bool = False,
    normal_model: str | None = None,
    smooth_iterations: int = 0,
) -> dict[str, object]:
    settings: Settings = request.app.state.settings
    if model is None:
        model = request.app.state.default_geometry_model
    selection = request.app.state.geometry_models.get(model)
    if selection is None:
        raise HTTPException(status_code=400, detail=f"Unknown geometry model {model!r}.")
    if not 0 <= smooth_iterations <= 10:
        raise HTTPException(
            status_code=400, detail="smooth_iterations must be between 0 and 10."
        )
    inference, params = selection
    selected_normal_model: InferenceModel[np.ndarray, NormalMap] | None = None
    if refine_normals:
        normal_name = normal_model or request.app.state.default_normal_model
        selected_normal_model = request.app.state.normal_models.get(normal_name)
        if selected_normal_model is None:
            raise HTTPException(
                status_code=400, detail=f"Unknown normal refinement model {normal_name!r}."
            )
        params = replace(params, normal_model_id=selected_normal_model.model_id)
    if smooth_iterations > 0:
        params = replace(params, smooth_iterations=smooth_iterations)
    raw_bytes = await read_image_upload(file, settings.max_upload_bytes)
    cache: GeometryCache = request.app.state.geometry_cache
    lock: threading.Lock = request.app.state.inference_lock

    image_sha256 = compute_image_sha256(raw_bytes)
    cache_id = build_cache_key(image_sha256, params)

    cached_meta = await run_in_threadpool(cache.get_metadata, cache_id)
    try:
        if cached_meta is not None:
            validate_image_pixel_count(
                cached_meta.original_width, cached_meta.original_height, settings.max_image_pixels,
            )
            return {**asdict(cached_meta), "model": model, "cached": True}

        metadata_dict, was_cached = await run_in_threadpool(
            _infer_and_cache,
            inference,
            request.app.state.models,
            settings.offload_inactive_models,
            cache,
            lock,
            cache_id,
            image_sha256,
            params,
            raw_bytes,
            settings.max_image_pixels,
            selected_normal_model,
        )
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        if is_cuda_oom(exc):
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory during geometry inference. Try a smaller max image size or INKRAY_DEVICE=cpu.",
            ) from exc
        raise

    return {**metadata_dict, "model": model, "cached": was_cached}


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
