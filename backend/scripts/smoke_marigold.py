import argparse
import hashlib
from pathlib import Path
import tempfile
import time
import numpy as np
from PIL import Image
import torch

from app.appearance import (
    calibrate_shading_scale,
    compute_reconstruction_error,
    linear_to_srgb,
    srgb_to_linear,
)
from app.cache.appearance_cache import (
    AppearanceCache,
    AppearanceInferenceParams,
    build_appearance_cache_key,
    build_appearance_metadata,
)
from app.inference.marigold import MarigoldInference
from app.inference.memory import prepare_model_for_inference
from app.utils.image import decode_image_bytes


def _default_synthetic_image() -> bytes:
    h, w = 512, 640
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    albedo = np.stack(
        [
            0.55 + 0.1 * np.sin(xs * 0.08),
            0.48 + 0.08 * np.cos(ys * 0.08),
            0.42 + 0.05 * np.sin((xs + ys) * 0.05),
        ],
        axis=-1,
    )
    # Illumination gradient plus crisp cast shadow transition.
    shading = np.where(xs < w * 0.55, 1.05 - 0.35 * (ys / h), 0.32)[..., None]
    shading = shading * np.array([1.05, 0.98, 0.88], dtype=np.float32)
    srgb = (linear_to_srgb(np.clip(albedo * shading, 0.0, 1.0)) * 255.0).round().astype(np.uint8)
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(srgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Marigold IID Lighting pipeline and cache.")
    parser.add_argument("--image", type=Path, default=None, help="Optional JPEG/PNG path.")
    parser.add_argument("--model-id", default="prs-eth/marigold-iid-lighting-v1-1")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--res", type=int, default=768)
    parser.add_argument("--with-moge", action="store_true", help="Also verify VRAM swap with MoGe-3.")
    args = parser.parse_args()

    raw_bytes = args.image.read_bytes() if args.image else _default_synthetic_image()
    decoded = decode_image_bytes(raw_bytes, max_image_size=args.max_size)

    moge = None
    if args.with_moge:
        from app.inference.moge import MoGeInference

        moge = MoGeInference(device="auto", use_fp16=True)
        _ = moge.run(decoded.rgb)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    marigold = MarigoldInference(
        model_id=args.model_id,
        device="auto",
        use_fp16=True,
        denoising_steps=args.steps,
        processing_resolution=args.res,
    )
    prepare_model_for_inference(marigold, (moge,), offload_inactive=True)
    result = marigold.run(decoded.rgb)
    elapsed = time.perf_counter() - t0

    assert result.shape == (decoded.inference_height, decoded.inference_width)
    ref_linear = srgb_to_linear(decoded.rgb.astype(np.float32) / 255.0)
    calib = calibrate_shading_scale(ref_linear, result.albedo.data, result.shading.data, residual_linear=result.residual.data)
    recon = compute_reconstruction_error(
        ref_linear, result.albedo.data, result.shading.data, result.residual.data
    )

    print(
        f"device={marigold.device} elapsed={elapsed:.2f}s shape={result.shape} "
        f"albedo_range={result.albedo.value_range()} shading_range={result.shading.value_range()} "
        f"residual_range={result.residual.value_range()} "
        f"calib_scale={calib.scale:.3f} (reliable={calib.is_reliable}) "
        f"recon_mae={recon.mae:.4f} recon_rmse={recon.rmse:.4f}"
    )
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        res_gb = torch.cuda.memory_allocated() / 1e9
        print(f"cuda_mem resident={res_gb:.2f}GB peak={peak_gb:.2f}GB")

    if moge is not None:
        prepare_model_for_inference(moge, (marigold,), offload_inactive=True)
        geom_res = moge.run(decoded.rgb)
        assert geom_res.shape == (decoded.inference_height, decoded.inference_width)
        prepare_model_for_inference(marigold, (moge,), offload_inactive=True)
        app_res2 = marigold.run(decoded.rgb)
        np.testing.assert_allclose(app_res2.albedo.data, result.albedo.data, atol=1e-3)
        print(f"moge_roundtrip_swap_ok geom_shape={geom_res.shape}")

    with tempfile.TemporaryDirectory() as tmp:
        cache = AppearanceCache(Path(tmp))
        params = AppearanceInferenceParams(
            model_id=args.model_id,
            denoising_steps=args.steps,
            processing_resolution=args.res,
            ensemble_size=1,
            seed=42,
            use_fp16=True,
            max_image_size=args.max_size,
        )
        sha = hashlib.sha256(raw_bytes).hexdigest()
        key = build_appearance_cache_key(sha, params)
        meta = build_appearance_metadata(
            key, sha, params, (decoded.original_width, decoded.original_height), result
        )
        cache.put(key, meta, result)
        loaded = cache.load_result(key)
        assert loaded is not None
        np.testing.assert_allclose(loaded.albedo.data, result.albedo.data)
        np.testing.assert_allclose(loaded.shading.data, result.shading.data)
        np.testing.assert_allclose(loaded.residual.data, result.residual.data)
        print("appearance_cache_roundtrip_ok=True")


if __name__ == "__main__":
    main()
