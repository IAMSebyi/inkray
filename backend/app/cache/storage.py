from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile
from zlib import error as ZlibError
import numpy as np


def is_valid_cache_id(cache_id: str) -> bool:
    return bool(cache_id) and all(c in "0123456789abcdef" for c in cache_id)


def read_cache_metadata(
    meta_path: Path,
    npz_path: Path,
    version: int,
    required_arrays: tuple[str, ...],
) -> dict[str, object] | None:
    try:
        meta_stat = meta_path.stat()
        npz_stat = npz_path.stat()
    except FileNotFoundError:
        return None
    return _read_cache_metadata(
        meta_path,
        npz_path,
        version,
        required_arrays,
        (meta_stat.st_mtime_ns, meta_stat.st_size, npz_stat.st_mtime_ns, npz_stat.st_size),
    )


@lru_cache(maxsize=128)
def _read_cache_metadata(
    meta_path: Path,
    npz_path: Path,
    version: int,
    required_arrays: tuple[str, ...],
    fingerprint: tuple[int, int, int, int],
) -> dict[str, object] | None:
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("cache_version") != version:
            return None
        arrays = raw.get("arrays")
        if not isinstance(arrays, dict) or not arrays:
            return None
        with np.load(npz_path, allow_pickle=False) as archive:
            if not set(required_arrays).issubset(archive.files):
                return None
            for kind in set(required_arrays).union(arrays):
                array = archive[kind]
                if kind in arrays and list(array.shape) != arrays[kind]["shape"]:
                    return None
        return raw
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        BadZipFile,
        ZlibError,
        EOFError,
        ValueError,
        KeyError,
        TypeError,
    ):
        return None


def write_npz_and_json(
    cache_dir: Path,
    cache_id: str,
    payload: dict[str, np.ndarray],
    metadata_dict: dict[str, object],
    npz_path: Path,
    meta_path: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    fd_npz, tmp_npz = tempfile.mkstemp(dir=cache_dir, prefix=f".{cache_id}_", suffix=".npz")
    try:
        with os.fdopen(fd_npz, "wb") as f:
            np.savez_compressed(f, **payload)
        os.replace(tmp_npz, npz_path)
    except Exception:
        if os.path.exists(tmp_npz):
            os.unlink(tmp_npz)
        raise

    fd_meta, tmp_meta = tempfile.mkstemp(dir=cache_dir, prefix=f".{cache_id}_", suffix=".json")
    try:
        with os.fdopen(fd_meta, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, separators=(",", ":"))
        os.replace(tmp_meta, meta_path)
    except Exception:
        if os.path.exists(tmp_meta):
            os.unlink(tmp_meta)
        raise
