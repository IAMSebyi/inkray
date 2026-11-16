import json
import os
from pathlib import Path
import tempfile
import numpy as np


def is_valid_cache_id(cache_id: str) -> bool:
    return bool(cache_id) and all(c in "0123456789abcdef" for c in cache_id)


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
