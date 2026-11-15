"""Smoke test for real MoGe-3 inference on CUDA and disk cache round-trip."""

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
import torch

from app.cache.geometry_cache import (
    GeometryCache,
    InferenceParams,
    build_cache_key,
    build_metadata,
    compute_image_sha256,
)
from app.inference.moge import MoGeInference
from app.utils.image import decode_image_bytes


def _create_synthetic_scene_png(path: Path) -> bytes:
    img = Image.new("RGB", (640, 480), (45, 70, 110))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 260, 640, 480], fill=(95, 85, 75))
    draw.rectangle([180, 160, 440, 360], fill=(185, 115, 75))
    draw.ellipse([270, 210, 350, 290], fill=(220, 200, 170))
    img.save(path, format="PNG")
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoGe-3 CUDA + cache smoke test.")
    parser.add_argument("--image", type=Path, default=None, help="Optional JPEG/PNG path.")
    parser.add_argument("--model-id", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for GPU smoke test.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        raw_bytes = (
            args.image.read_bytes()
            if args.image is not None
            else _create_synthetic_scene_png(tmp_dir / "synthetic.png")
        )

        decoded = decode_image_bytes(raw_bytes, max_image_size=args.max_size)
        print(
            f"Decoded image: orig={decoded.original_width}x{decoded.original_height}, "
            f"inf={decoded.inference_width}x{decoded.inference_height}"
        )

        model = MoGeInference(model_id=args.model_id, device="cuda", use_fp16=True)
        result = model.run(decoded.rgb)

        h, w = decoded.inference_height, decoded.inference_width
        assert result.depth.shape == (h, w)
        assert result.points.shape == (h, w, 3)
        assert result.mask.shape == (h, w)
        assert result.intrinsics.matrix.shape == (3, 3)
        assert result.normals is not None and result.normals.shape == (h, w, 3)

        valid = result.mask.data
        assert np.any(valid), "Expected at least some valid geometry pixels."
        assert np.all(np.isfinite(result.depth.data[valid]))
        assert np.all(np.isfinite(result.points.data[valid]))
        assert np.all(np.isfinite(result.normals.data[valid]))

        cache = GeometryCache(tmp_dir / "cache")
        params = InferenceParams(args.model_id, 9, 3, True, args.max_size)
        sha = compute_image_sha256(raw_bytes)
        cache_id = build_cache_key(sha, params)
        meta = build_metadata(
            cache_id,
            sha,
            params,
            (decoded.original_width, decoded.original_height),
            result,
        )
        cache.put(cache_id, meta, result)

        loaded = cache.load_result(cache_id)
        assert loaded is not None
        np.testing.assert_allclose(loaded.depth.data, result.depth.data)
        np.testing.assert_allclose(loaded.points.data, result.points.data)
        print(
            f"Smoke test passed on {torch.cuda.get_device_name(0)}: "
            f"valid_pixels={result.mask.valid_count}/{h * w}, "
            f"depth_range={result.depth.valid_range(result.mask)}"
        )


if __name__ == "__main__":
    main()
