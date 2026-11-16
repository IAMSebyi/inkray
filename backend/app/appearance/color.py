from dataclasses import dataclass
import numpy as np

DEFAULT_MAX_CALIBRATED_SHADING = 2.5


@dataclass(frozen=True)
class ShadingCalibration:
    scale: float
    is_reliable: bool
    valid_sample_count: int
    median_residual: float


@dataclass(frozen=True)
class ReconstructionMetrics:
    mae: float
    rmse: float
    p95_abs_error: float
    calibrated_scale: float


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert standard IEC 61966-2-1 sRGB values in [0, 1] to linear RGB float32."""
    x = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    low = x / np.float32(12.92)
    high = np.power((x + np.float32(0.055)) / np.float32(1.055), np.float32(2.4))
    return np.where(x <= np.float32(0.04045), low, high).astype(np.float32, copy=False)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB float32 values to IEC 61966-2-1 sRGB in [0, 1]."""
    x = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    low = x * np.float32(12.92)
    high = np.float32(1.055) * np.power(
        np.maximum(x, np.float32(0.0)), np.float32(1.0 / 2.4)
    ) - np.float32(0.055)
    out = np.where(x <= np.float32(0.0031308), low, high)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def calibrate_shading_scale(
    linear_reference: np.ndarray,
    albedo_linear: np.ndarray,
    shading_raw: np.ndarray,
    support_mask: np.ndarray | None = None,
    residual_linear: np.ndarray | None = None,
    min_albedo: float = 0.05,
    max_reference: float = 0.96,
    min_scale: float = 0.15,
    max_scale: float = 8.0,
) -> ShadingCalibration:
    """Estimate deterministic global shading scale matching linear_reference ~ k * albedo * shading.

    Uses robust per-pixel projection ratios filtered to non-dark diffuse pixels so
    low-albedo shadows and clipped specular highlights do not destabilize scale.
    """
    ref = np.asarray(linear_reference, dtype=np.float32)
    alb = np.asarray(albedo_linear, dtype=np.float32)
    shd = np.asarray(shading_raw, dtype=np.float32)

    base = ref
    if residual_linear is not None:
        res = np.asarray(residual_linear, dtype=np.float32)
        base = np.where(np.isfinite(res), ref - np.clip(res, 0.0, 1.0), ref)

    finite = np.all(np.isfinite(base), axis=-1) & np.all(np.isfinite(alb), axis=-1) & np.all(
        np.isfinite(shd), axis=-1
    )
    if support_mask is not None:
        finite = finite & (np.asarray(support_mask) > 0)

    alb_mean = np.mean(alb, axis=-1)
    shd_mean = np.mean(shd, axis=-1)
    ref_mean = np.mean(ref, axis=-1)
    ref_max = np.max(ref, axis=-1)

    primary = (
        finite
        & (alb_mean >= min_albedo)
        & (shd_mean >= 1e-3)
        & (ref_mean >= 0.02)
        & (ref_max <= max_reference)
    )
    selected = primary
    if int(np.count_nonzero(selected)) < 16:
        selected = finite & (alb_mean >= 0.02) & (shd_mean >= 1e-4) & (ref_mean >= 0.01)

    count = int(np.count_nonzero(selected))
    if count == 0:
        return ShadingCalibration(
            scale=1.0,
            is_reliable=False,
            valid_sample_count=0,
            median_residual=0.0,
        )

    prod = alb[selected] * shd[selected]
    target = np.maximum(base[selected], 0.0)
    den = np.sum(prod * prod, axis=-1)
    valid_den = den > 1e-6
    valid_count = int(np.count_nonzero(valid_den))
    if valid_count == 0:
        return ShadingCalibration(
            scale=1.0,
            is_reliable=False,
            valid_sample_count=0,
            median_residual=0.0,
        )

    num = np.sum(target[valid_den] * prod[valid_den], axis=-1)
    ratios = num / den[valid_den]
    positive = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    if positive.size < 8:
        raw_scale = float(np.median(positive)) if positive.size > 0 else 1.0
        scale = float(np.clip(raw_scale, min_scale, max_scale)) if np.isfinite(raw_scale) else 1.0
        err = np.abs(target[valid_den] - scale * prod[valid_den])
        return ShadingCalibration(
            scale=scale,
            is_reliable=False,
            valid_sample_count=int(positive.size),
            median_residual=float(np.median(err)) if err.size > 0 else 0.0,
        )

    lo = float(np.percentile(positive, 10.0))
    hi = float(np.percentile(positive, 90.0))
    trimmed = positive[(positive >= lo) & (positive <= hi)]
    median_scale = float(np.median(trimmed if trimmed.size > 0 else positive))
    scale = float(np.clip(median_scale, min_scale, max_scale))

    err = np.abs(target[valid_den] - scale * prod[valid_den])
    return ShadingCalibration(
        scale=scale,
        is_reliable=True,
        valid_sample_count=int(positive.size),
        median_residual=float(np.median(err)),
    )


def composite_lighting_aware_linear(
    original_linear: np.ndarray,
    paint_linear: np.ndarray,
    paint_alpha: np.ndarray,
    albedo_linear: np.ndarray,
    shading_raw: np.ndarray,
    shading_scale: float,
    mode: str = "shading",
    texture_preservation: float = 1.0,
    max_calibrated_shading: float = DEFAULT_MAX_CALIBRATED_SHADING,
) -> np.ndarray:
    """Composite paint in linear RGB space while preserving surface illumination and optional texture."""
    orig = np.asarray(original_linear, dtype=np.float32)
    paint = np.asarray(paint_linear, dtype=np.float32)
    alpha = np.clip(np.asarray(paint_alpha, dtype=np.float32), 0.0, 1.0)
    if alpha.ndim == orig.ndim - 1:
        alpha = alpha[..., np.newaxis]

    if mode == "flat":
        return np.clip(orig * (1.0 - alpha) + paint * alpha, 0.0, 1.0).astype(np.float32)

    alb = np.clip(np.asarray(albedo_linear, dtype=np.float32), 0.0, 1.0)
    shd = np.maximum(np.asarray(shading_raw, dtype=np.float32), 0.0)
    s_cal = np.minimum(shd * np.float32(shading_scale), np.float32(max_calibrated_shading))

    target_albedo = paint
    if mode == "texture" and texture_preservation > 0.0:
        # Estimate edge-aware smooth albedo over a sparse 5x5 neighborhood so surface texture
        # (e.g., plaster, brick, wood grain) is preserved without bleeding across albedo seams.
        h, w = alb.shape[:2]
        step = max(1, min(h, w) // 40)
        pad = 2 * step
        padded = np.pad(alb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        local_sum = np.zeros_like(alb, dtype=np.float32)
        weight_sum = np.zeros((h, w, 1), dtype=np.float32)
        for dy in range(-2, 3):
            y0 = pad + dy * step
            for dx in range(-2, 3):
                x0 = pad + dx * step
                nbr = padded[y0 : y0 + h, x0 : x0 + w]
                diff_sq = np.sum((nbr - alb) ** 2, axis=-1, keepdims=True)
                w_nbr = np.exp(-diff_sq / np.float32(0.25)).astype(np.float32)
                local_sum += nbr * w_nbr
                weight_sum += w_nbr
        alb_smooth = local_sum / np.maximum(weight_sum, np.float32(1e-4))
        detail = np.clip(alb / np.maximum(alb_smooth, np.float32(0.05)), 0.5, 1.5)
        tp = np.float32(np.clip(texture_preservation, 0.0, 1.0))
        target_albedo = np.clip(paint * ((1.0 - tp) + tp * detail), 0.0, 1.0)

    out = orig + alpha * s_cal * (target_albedo - alb)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_reconstruction_error(
    linear_reference: np.ndarray,
    albedo_linear: np.ndarray,
    shading_raw: np.ndarray,
    residual_linear: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> ReconstructionMetrics:
    """Measure residual reconstruction accuracy of calibrated albedo * shading (+ residual)."""
    calib = calibrate_shading_scale(
        linear_reference=linear_reference,
        albedo_linear=albedo_linear,
        shading_raw=shading_raw,
        support_mask=valid_mask,
        residual_linear=residual_linear,
    )
    recon = calib.scale * np.asarray(albedo_linear, dtype=np.float32) * np.asarray(
        shading_raw, dtype=np.float32
    )
    if residual_linear is not None:
        recon = recon + np.asarray(residual_linear, dtype=np.float32)
    recon = np.clip(recon, 0.0, 1.0)

    diff = recon - np.asarray(linear_reference, dtype=np.float32)
    if valid_mask is not None:
        mask = np.asarray(valid_mask) > 0
        if np.any(mask):
            diff = diff[mask]
    abs_err = np.abs(diff).reshape(-1)
    return ReconstructionMetrics(
        mae=float(np.mean(abs_err)),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        p95_abs_error=float(np.percentile(abs_err, 95.0)),
        calibrated_scale=calib.scale,
    )
