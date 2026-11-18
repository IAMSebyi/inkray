from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal
import numpy as np

from ..appearance.maps import AlbedoMap, ResidualMap, ShadingMap
from ..appearance.result import AppearanceResult
from .geometry_cache import compute_image_sha256
from .storage import is_valid_cache_id, read_cache_metadata, write_npz_and_json

APPEARANCE_CACHE_FORMAT_VERSION = 1
AppearanceMapKind = Literal["albedo", "shading", "residual"]
VALID_APPEARANCE_MAP_KINDS = ("albedo", "shading", "residual")


@dataclass(frozen=True)
class AppearanceInferenceParams:
    model_id: str
    denoising_steps: int = 0
    processing_resolution: int = 0
    ensemble_size: int = 1
    seed: int = 0
    use_fp16: bool = False
    max_image_size: int = 0
    cache_variant: str | None = None


@dataclass(frozen=True)
class AppearanceRecordMetadata:
    id: str
    cache_version: int
    image_sha256: str
    model_id: str
    denoising_steps: int
    processing_resolution: int
    ensemble_size: int
    seed: int
    use_fp16: bool
    max_image_size: int
    original_width: int
    original_height: int
    appearance_width: int
    appearance_height: int
    prediction_space: str
    scale_conventions: dict[str, str]
    albedo_range: list[float] | None
    shading_range: list[float] | None
    residual_range: list[float] | None
    arrays: dict[str, dict[str, object]]
    has_shading: bool = True
    has_residual: bool = True


def build_appearance_cache_key(
    image_sha256: str,
    params: AppearanceInferenceParams,
) -> str:
    options = asdict(params)
    if params.cache_variant is None:
        del options["cache_variant"]
    payload = json.dumps(
        {
            "cache_version": APPEARANCE_CACHE_FORMAT_VERSION,
            "image_sha256": image_sha256,
            **options,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _range_list(bounds: tuple[float, float] | None) -> list[float] | None:
    return [bounds[0], bounds[1]] if bounds is not None else None


def build_appearance_metadata(
    cache_id: str,
    image_sha256: str,
    params: AppearanceInferenceParams,
    original_size: tuple[int, int],
    result: AppearanceResult,
) -> AppearanceRecordMetadata:
    arrays: dict[str, dict[str, object]] = {
        kind: {"shape": list(layer.shape), "dtype": "float32"}
        for kind in VALID_APPEARANCE_MAP_KINDS
        if (layer := getattr(result, kind)) is not None
    }
    return AppearanceRecordMetadata(
        id=cache_id,
        cache_version=APPEARANCE_CACHE_FORMAT_VERSION,
        image_sha256=image_sha256,
        model_id=params.model_id,
        denoising_steps=result.denoising_steps,
        processing_resolution=result.processing_resolution,
        ensemble_size=result.ensemble_size,
        seed=result.seed,
        use_fp16=params.use_fp16,
        max_image_size=params.max_image_size,
        original_width=original_size[0],
        original_height=original_size[1],
        appearance_width=result.width,
        appearance_height=result.height,
        prediction_space=result.prediction_space,
        scale_conventions=dict(result.scale_conventions),
        albedo_range=_range_list(result.albedo.value_range()),
        shading_range=_range_list(result.shading.value_range()) if result.shading is not None else None,
        residual_range=_range_list(result.residual.value_range()) if result.residual is not None else None,
        arrays=arrays,
        has_shading=result.shading is not None,
        has_residual=result.residual is not None,
    )


class AppearanceCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, cache_id: str) -> Path:
        return self.cache_dir / f"{cache_id}.json"

    def _npz_path(self, cache_id: str) -> Path:
        return self.cache_dir / f"{cache_id}.npz"

    def has(self, cache_id: str) -> bool:
        if not is_valid_cache_id(cache_id):
            return False
        if not (self._meta_path(cache_id).is_file() and self._npz_path(cache_id).is_file()):
            return False
        meta = self.get_metadata(cache_id)
        return meta is not None and meta.cache_version == APPEARANCE_CACHE_FORMAT_VERSION

    def get_metadata(self, cache_id: str) -> AppearanceRecordMetadata | None:
        if not is_valid_cache_id(cache_id):
            return None
        raw = read_cache_metadata(
            self._meta_path(cache_id), self._npz_path(cache_id), APPEARANCE_CACHE_FORMAT_VERSION,
            ("albedo",),
        )
        if raw is None:
            return None
        expected_maps = {"albedo", *(
            kind for kind in ("shading", "residual") if raw.get(f"has_{kind}", True)
        )}
        if set(raw["arrays"]) != expected_maps:
            return None
        try:
            return AppearanceRecordMetadata(**raw)
        except TypeError:
            return None

    def put(
        self,
        cache_id: str,
        metadata: AppearanceRecordMetadata,
        result: AppearanceResult,
    ) -> None:
        payload: dict[str, np.ndarray] = {
            kind: np.ascontiguousarray(getattr(result, kind).data, dtype=np.float32)
            for kind in metadata.arrays
        }
        write_npz_and_json(
            cache_dir=self.cache_dir,
            cache_id=cache_id,
            payload=payload,
            metadata_dict=asdict(metadata),
            npz_path=self._npz_path(cache_id),
            meta_path=self._meta_path(cache_id),
        )

    def load_result(self, cache_id: str) -> AppearanceResult | None:
        meta = self.get_metadata(cache_id)
        if meta is None or not self._npz_path(cache_id).is_file():
            return None
        with np.load(self._npz_path(cache_id), allow_pickle=False) as archive:
            return AppearanceResult(
                albedo=AlbedoMap(archive["albedo"]),
                shading=ShadingMap(archive["shading"]) if meta.has_shading else None,
                residual=ResidualMap(archive["residual"]) if meta.has_residual else None,
                model_id=meta.model_id,
                denoising_steps=meta.denoising_steps,
                processing_resolution=meta.processing_resolution,
                ensemble_size=meta.ensemble_size,
                seed=meta.seed,
                prediction_space=meta.prediction_space,
                scale_conventions=dict(meta.scale_conventions),
            )

    def load_map_bytes(
        self,
        cache_id: str,
        kind: AppearanceMapKind,
    ) -> tuple[bytes, tuple[int, ...], str] | None:
        metadata = self.get_metadata(cache_id)
        if metadata is None or kind not in metadata.arrays:
            return None
        with np.load(self._npz_path(cache_id), allow_pickle=False) as archive:
            if kind not in archive:
                return None
            out = np.ascontiguousarray(archive[kind], dtype="<f4")
            return (out.tobytes(order="C"), tuple(int(d) for d in out.shape), "float32")


__all__ = [
    "APPEARANCE_CACHE_FORMAT_VERSION",
    "AppearanceCache",
    "AppearanceInferenceParams",
    "AppearanceMapKind",
    "AppearanceRecordMetadata",
    "VALID_APPEARANCE_MAP_KINDS",
    "build_appearance_cache_key",
    "build_appearance_metadata",
    "compute_image_sha256",
]
