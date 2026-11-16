from pathlib import Path
import numpy as np
import pytest

from app.appearance import (
    AlbedoMap,
    AppearanceResult,
    ResidualMap,
    ShadingMap,
    calibrate_shading_scale,
    composite_lighting_aware_linear,
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


def test_appearance_maps_and_result_validation() -> None:
    albedo = AlbedoMap(np.full((4, 6, 3), 0.5, dtype=np.float32))
    shading = ShadingMap(np.full((4, 6, 3), 0.8, dtype=np.float32))
    residual = ResidualMap(np.zeros((4, 6, 3), dtype=np.float32))

    result = AppearanceResult(
        albedo=albedo,
        shading=shading,
        residual=residual,
        model_id="stub",
        denoising_steps=4,
        processing_resolution=768,
        ensemble_size=1,
        seed=42,
    )
    assert result.shape == (4, 6)
    assert albedo.at_pixel(2, 1) == (0.5, 0.5, 0.5)
    assert albedo.sample(3.0, 2.0) is not None
    assert albedo.sample(-1.0, 0.0) is None

    with pytest.raises(ValueError):
        AlbedoMap(np.zeros((4, 6), dtype=np.float32))

    with pytest.raises(ValueError):
        AppearanceResult(
            albedo=albedo,
            shading=ShadingMap(np.zeros((2, 3, 3), dtype=np.float32)),
            residual=residual,
            model_id="stub",
            denoising_steps=4,
            processing_resolution=768,
            ensemble_size=1,
            seed=42,
        )


def test_srgb_linear_round_trip() -> None:
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    recovered = linear_to_srgb(srgb_to_linear(ramp))
    np.testing.assert_allclose(recovered, ramp, atol=1e-6)


def test_calibrate_shading_scale_handles_arbitrary_scale_and_outliers() -> None:
    h, w = 24, 32
    rng = np.random.default_rng(7)
    true_albedo = rng.uniform(0.25, 0.75, size=(h, w, 3)).astype(np.float32)
    true_shading = rng.uniform(0.6, 1.4, size=(h, w, 3)).astype(np.float32)
    # Use colored warm illumination to ensure calibration keeps relative RGB balance.
    true_shading *= np.array([1.15, 1.0, 0.78], dtype=np.float32)
    ref_linear = np.clip(true_albedo * true_shading, 0.0, 1.0).astype(np.float32)

    true_scale = 2.35
    raw_shading = (true_shading / true_scale).astype(np.float32)

    # Inject a clipped highlight block and a near-black patch that should not skew calibration.
    raw_albedo_noisy = true_albedo.copy()
    ref_noisy = ref_linear.copy()
    raw_albedo_noisy[:4, :4] = 0.005
    ref_noisy[:4, :4] = 0.002
    ref_noisy[-4:, -4:] = 0.999

    support = np.ones((h, w), dtype=np.uint8)
    calib = calibrate_shading_scale(ref_noisy, raw_albedo_noisy, raw_shading, support_mask=support)
    assert calib.is_reliable is True
    assert calib.scale == pytest.approx(true_scale, rel=0.05)

    # Degenerate zero-albedo mask returns safe fallback.
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    empty_calib = calibrate_shading_scale(ref_noisy, raw_albedo_noisy, raw_shading, empty_mask)
    assert empty_calib.is_reliable is False
    assert empty_calib.scale == 1.0


def test_composite_modes_preserve_unpainted_and_lighting() -> None:
    h, w = 16, 16
    albedo = np.full((h, w, 3), 0.35, dtype=np.float32)
    # Add fine alternating texture stripe in center albedo.
    albedo[6:10, 6:10] += np.where((np.indices((4, 4)).sum(axis=0) % 2)[..., None] == 0, 0.08, -0.08)

    shading_raw = np.ones((h, w, 3), dtype=np.float32)
    shading_raw[:, : w // 2] = 0.25  # Shadowed left half
    shading_raw[:, w // 2 :] = 0.75  # Lit right half
    scale = 2.0
    original = np.clip(albedo * shading_raw * scale, 0.0, 1.0).astype(np.float32)

    paint_color = np.full((h, w, 3), 0.55, dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[4:12, :] = 1.0

    out_shading = composite_lighting_aware_linear(
        original_linear=original,
        paint_linear=paint_color,
        paint_alpha=alpha,
        albedo_linear=albedo,
        shading_raw=shading_raw,
        shading_scale=scale,
        mode="shading",
    )
    # Unpainted rows remain untouched.
    np.testing.assert_allclose(out_shading[:4], original[:4], atol=1e-7)
    np.testing.assert_allclose(out_shading[12:], original[12:], atol=1e-7)
    # Lit side receives proportionally brighter paint than shadow side.
    shadow_mean = float(np.mean(out_shading[4:12, : w // 2]))
    lit_mean = float(np.mean(out_shading[4:12, w // 2 :]))
    assert lit_mean > shadow_mean * 2.0

    # Texture mode transfers fine local albedo variation relative to smooth neighborhood.
    out_texture = composite_lighting_aware_linear(
        original_linear=original,
        paint_linear=paint_color,
        paint_alpha=alpha,
        albedo_linear=albedo,
        shading_raw=shading_raw,
        shading_scale=scale,
        mode="texture",
        texture_preservation=1.0,
    )
    assert float(np.std(out_texture[6:10, 10:14])) > float(np.std(out_shading[6:10, 10:14]))

    metrics = compute_reconstruction_error(original, albedo, shading_raw)
    assert metrics.mae < 1e-4
    assert metrics.rmse < 1e-4


def test_appearance_cache_round_trip(tmp_path: Path) -> None:
    cache = AppearanceCache(tmp_path / "appearance")
    params = AppearanceInferenceParams(
        model_id="prs-eth/marigold-iid-lighting-v1-1",
        denoising_steps=4,
        processing_resolution=768,
        ensemble_size=1,
        seed=42,
        use_fp16=True,
        max_image_size=1024,
    )
    cache_id = build_appearance_cache_key("abc123", params)

    result = AppearanceResult(
        albedo=AlbedoMap(np.full((4, 6, 3), 0.4, dtype=np.float32)),
        shading=ShadingMap(np.full((4, 6, 3), 0.7, dtype=np.float32)),
        residual=ResidualMap(np.full((4, 6, 3), 0.05, dtype=np.float32)),
        model_id=params.model_id,
        denoising_steps=4,
        processing_resolution=768,
        ensemble_size=1,
        seed=42,
    )
    meta = build_appearance_metadata(cache_id, "abc123", params, (12, 8), result)
    cache.put(cache_id, meta, result)

    assert cache.has(cache_id) is True
    loaded = cache.load_result(cache_id)
    assert loaded is not None
    np.testing.assert_allclose(loaded.albedo.data, result.albedo.data)
    np.testing.assert_allclose(loaded.shading.data, result.shading.data)
    np.testing.assert_allclose(loaded.residual.data, result.residual.data)

    payload = cache.load_map_bytes(cache_id, "shading")
    assert payload is not None
    raw, shape, dtype = payload
    assert shape == (4, 6, 3)
    assert dtype == "float32"
    arr = np.frombuffer(raw, dtype="<f4").reshape(4, 6, 3)
    np.testing.assert_allclose(arr, 0.7)
