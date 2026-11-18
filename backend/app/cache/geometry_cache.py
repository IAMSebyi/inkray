from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal
import numpy as np

from ..geometry.depth import DepthMap
from ..geometry.intrinsics import CameraIntrinsics
from ..geometry.mask import ValidMask
from ..geometry.normals import NormalMap
from ..geometry.points import Points3D
from ..geometry.result import GeometryResult
from .storage import is_valid_cache_id, read_cache_metadata, write_npz_and_json

CACHE_FORMAT_VERSION = 1
MapKind = Literal["depth", "normals", "points", "mask"]
VALID_MAP_KINDS = ("depth", "normals", "points", "mask")


@dataclass(frozen=True)
class InferenceParams:
    model_id: str
    resolution_level: int = 0
    refine_steps: int = 0
    use_fp16: bool = False
    max_image_size: int = 0
    cache_variant: str | None = None
    normal_model_id: str | None = None
    smooth_iterations: int = 0


@dataclass(frozen=True)
class GeometryRecordMetadata:
    id: str
    cache_version: int
    image_sha256: str
    model_id: str
    resolution_level: int
    refine_steps: int
    use_fp16: bool
    max_image_size: int
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int
    coordinate_system: str
    intrinsics_convention: str
    intrinsics: list[list[float]]
    depth_range: list[float] | None
    valid_pixel_count: int
    has_normals: bool
    arrays: dict[str, dict[str, object]]
    normal_model_id: str | None = None
    smooth_iterations: int = 0


def compute_image_sha256(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def build_cache_key(image_sha256: str, params: InferenceParams) -> str:
    options = asdict(params)
    if params.cache_variant is None:
        del options["cache_variant"]
    if params.normal_model_id is None:
        del options["normal_model_id"]
    if params.smooth_iterations == 0:
        del options["smooth_iterations"]
    payload = json.dumps(
        {
            "cache_version": CACHE_FORMAT_VERSION,
            "image_sha256": image_sha256,
            **options,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def build_metadata(
    cache_id: str,
    image_sha256: str,
    params: InferenceParams,
    original_size: tuple[int, int],
    result: GeometryResult,
) -> GeometryRecordMetadata:
    depth_bounds = result.depth.valid_range(result.mask)
    arrays: dict[str, dict[str, object]] = {
        "depth": {"shape": list(result.depth.shape), "dtype": "float32"},
        "points": {"shape": list(result.points.shape), "dtype": "float32"},
        "mask": {"shape": list(result.mask.shape), "dtype": "uint8"},
    }
    if result.normals is not None:
        arrays["normals"] = {"shape": list(result.normals.shape), "dtype": "float32"}

    return GeometryRecordMetadata(
        id=cache_id,
        cache_version=CACHE_FORMAT_VERSION,
        image_sha256=image_sha256,
        model_id=params.model_id,
        resolution_level=params.resolution_level,
        refine_steps=params.refine_steps,
        use_fp16=params.use_fp16,
        max_image_size=params.max_image_size,
        original_width=original_size[0],
        original_height=original_size[1],
        inference_width=result.width,
        inference_height=result.height,
        coordinate_system="opencv",
        intrinsics_convention="normalized",
        intrinsics=result.intrinsics.matrix.tolist(),
        depth_range=[depth_bounds[0], depth_bounds[1]] if depth_bounds is not None else None,
        valid_pixel_count=result.mask.valid_count,
        has_normals=result.normals is not None,
        arrays=arrays,
        normal_model_id=params.normal_model_id,
        smooth_iterations=params.smooth_iterations,
    )


class GeometryCache:
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
        meta_path = self._meta_path(cache_id)
        npz_path = self._npz_path(cache_id)
        if not (meta_path.is_file() and npz_path.is_file()):
            return False
        meta = self.get_metadata(cache_id)
        return meta is not None and meta.cache_version == CACHE_FORMAT_VERSION

    def get_metadata(self, cache_id: str) -> GeometryRecordMetadata | None:
        if not is_valid_cache_id(cache_id):
            return None
        raw = read_cache_metadata(
            self._meta_path(cache_id), self._npz_path(cache_id), CACHE_FORMAT_VERSION,
            ("depth", "points", "mask", "intrinsics"),
        )
        if raw is None:
            return None
        required_maps = {"depth", "points", "mask"}
        if raw.get("has_normals"):
            required_maps.add("normals")
        if not required_maps.issubset(raw["arrays"]):
            return None
        try:
            return GeometryRecordMetadata(**raw)
        except TypeError:
            return None

    def put(
        self,
        cache_id: str,
        metadata: GeometryRecordMetadata,
        result: GeometryResult,
    ) -> None:
        npz_path = self._npz_path(cache_id)
        meta_path = self._meta_path(cache_id)

        payload: dict[str, np.ndarray] = {
            "depth": np.ascontiguousarray(result.depth.data, dtype=np.float32),
            "points": np.ascontiguousarray(result.points.data, dtype=np.float32),
            "intrinsics": np.ascontiguousarray(result.intrinsics.matrix, dtype=np.float32),
            "mask": np.ascontiguousarray(result.mask.data, dtype=np.bool_),
        }
        if result.normals is not None:
            payload["normals"] = np.ascontiguousarray(result.normals.data, dtype=np.float32)

        write_npz_and_json(
            cache_dir=self.cache_dir,
            cache_id=cache_id,
            payload=payload,
            metadata_dict=asdict(metadata),
            npz_path=npz_path,
            meta_path=meta_path,
        )

    def load_result(self, cache_id: str) -> GeometryResult | None:
        if not self.has(cache_id):
            return None
        npz_path = self._npz_path(cache_id)
        with np.load(npz_path, allow_pickle=False) as archive:
            normals = NormalMap(archive["normals"]) if "normals" in archive else None
            return GeometryResult(
                depth=DepthMap(archive["depth"]),
                normals=normals,
                points=Points3D(archive["points"]),
                intrinsics=CameraIntrinsics(archive["intrinsics"]),
                mask=ValidMask(archive["mask"]),
            )

    def load_map_bytes(
        self,
        cache_id: str,
        kind: MapKind,
    ) -> tuple[bytes, tuple[int, ...], str] | None:
        if kind not in VALID_MAP_KINDS or not self.has(cache_id):
            return None
        npz_path = self._npz_path(cache_id)
        with np.load(npz_path, allow_pickle=False) as archive:
            if kind not in archive:
                return None
            arr = archive[kind]
            if kind == "mask":
                out = np.ascontiguousarray(arr.astype(np.uint8))
                return (out.tobytes(order="C"), tuple(int(d) for d in out.shape), "uint8")
            out = np.ascontiguousarray(arr, dtype="<f4")
            return (out.tobytes(order="C"), tuple(int(d) for d in out.shape), "float32")
