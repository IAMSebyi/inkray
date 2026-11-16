import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

from app.appearance import (
    calibrate_shading_scale,
    composite_lighting_aware_linear,
    compute_reconstruction_error,
    linear_to_srgb,
    srgb_to_linear,
)
from app.inference.marigold import MarigoldInference


@dataclass(frozen=True)
class EvalScenario:
    name: str
    description: str
    srgb_uint8: np.ndarray
    lit_region: np.ndarray
    shadow_region: np.ndarray
    textured_region: np.ndarray
    paint_color_srgb: tuple[float, float, float]


def _to_uint8_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    srgb = linear_to_srgb(np.clip(linear_rgb, 0.0, 1.0))
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def _build_scenarios(width: int = 512, height: int = 384) -> list[EvalScenario]:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = xs / width
    yn = ys / height

    base_mask = np.ones((height, width), dtype=bool)
    left_lit = xn < 0.45
    right_shadow = xn >= 0.55

    # 1. Soft shadow across a wall
    alb1 = np.full((height, width, 3), [0.58, 0.52, 0.45], dtype=np.float32)
    soft_falloff = 1.0 - 0.65 / (1.0 + np.exp(-(xn - 0.5) * 16.0))
    shd1 = soft_falloff[..., None] * np.array([1.05, 1.0, 0.94], dtype=np.float32)
    s1 = EvalScenario(
        "01_soft_shadow_wall",
        "Soft shadow falloff across beige plaster wall",
        _to_uint8_srgb(alb1 * shd1),
        left_lit,
        right_shadow,
        base_mask,
        (0.22, 0.56, 0.85),
    )

    # 2. Sunlight + hard shadow boundary
    alb2 = np.full((height, width, 3), [0.62, 0.56, 0.48], dtype=np.float32)
    diag = (xn * 0.75 + yn * 0.35) < 0.52
    shd2 = np.where(
        diag[..., None],
        np.array([1.15, 1.05, 0.92], dtype=np.float32),
        np.array([0.24, 0.28, 0.38], dtype=np.float32),
    )
    s2 = EvalScenario(
        "02_sunlight_shadow_boundary",
        "Crisp sunlight vs cool sky shadow diagonal",
        _to_uint8_srgb(alb2 * shd2),
        diag,
        ~diag,
        base_mask,
        (0.85, 0.35, 0.25),
    )

    # 3. Dark wall + bright wall
    alb3 = np.where(
        (xn < 0.5)[..., None],
        np.array([0.11, 0.12, 0.14], dtype=np.float32),
        np.array([0.78, 0.76, 0.72], dtype=np.float32),
    )
    shd3 = (0.85 - 0.35 * yn)[..., None] * np.ones((1, 1, 3), dtype=np.float32)
    s3 = EvalScenario(
        "03_dark_and_bright_walls",
        "Dark charcoal panel beside bright white wall",
        _to_uint8_srgb(alb3 * shd3),
        yn < 0.35,
        yn > 0.65,
        base_mask,
        (0.25, 0.75, 0.42),
    )

    # 4. Corner crease between two planes
    alb4 = np.where(
        (xn < 0.5)[..., None],
        np.array([0.52, 0.48, 0.44], dtype=np.float32),
        np.array([0.48, 0.50, 0.53], dtype=np.float32),
    )
    crease_ao = 1.0 - 0.55 * np.exp(-np.abs(xn - 0.5) * 28.0)
    plane_dir = np.where(xn < 0.5, 0.92, 0.48)
    shd4 = (plane_dir * crease_ao)[..., None] * np.ones((1, 1, 3), dtype=np.float32)
    s4 = EvalScenario(
        "04_corner_crease",
        "Two planes meeting at shaded vertical corner crease",
        _to_uint8_srgb(alb4 * shd4),
        xn < 0.4,
        xn > 0.6,
        base_mask,
        (0.82, 0.62, 0.22),
    )

    # 5. Warm/cool colored illumination
    alb5 = np.full((height, width, 3), [0.65, 0.65, 0.65], dtype=np.float32)
    warm_w = (1.0 - xn)[..., None]
    cool_w = xn[..., None]
    shd5 = warm_w * np.array([1.25, 0.78, 0.38], dtype=np.float32) + cool_w * np.array(
        [0.42, 0.75, 1.25], dtype=np.float32
    )
    s5 = EvalScenario(
        "05_colored_illumination",
        "Warm tungsten left light and cool blue right light on white wall",
        _to_uint8_srgb(alb5 * shd5),
        left_lit,
        right_shadow,
        base_mask,
        (0.82, 0.82, 0.82),
    )

    # 6. Textured wall / patterned surface
    brick_y = np.sin(yn * np.pi * 16.0)
    brick_x = np.sin(xn * np.pi * 12.0 + np.floor(yn * 16.0) * 1.5)
    grain = 0.08 * brick_y + 0.06 * brick_x + 0.05 * np.sin(xs * 0.6) * np.cos(ys * 0.6)
    alb6 = np.clip(
        np.array([0.52, 0.38, 0.31], dtype=np.float32) + grain[..., None],
        0.12,
        0.85,
    )
    shd6 = (0.95 - 0.5 * xn)[..., None] * np.ones((1, 1, 3), dtype=np.float32)
    s6 = EvalScenario(
        "06_textured_patterned_wall",
        "Patterned masonry wall with lighting gradient",
        _to_uint8_srgb(alb6 * shd6),
        left_lit,
        right_shadow,
        base_mask,
        (0.32, 0.62, 0.78),
    )

    # 7. Partially occluded foreground object casting shadow onto plane
    pillar = (xn > 0.22) & (xn < 0.34) & (yn > 0.15)
    cast_shadow = (xn >= 0.34) & (xn < 0.62) & (yn > 0.25)
    alb7 = np.where(
        pillar[..., None],
        np.array([0.18, 0.22, 0.28], dtype=np.float32),
        np.array([0.62, 0.58, 0.52], dtype=np.float32),
    )
    shd7 = np.where(cast_shadow[..., None], 0.34, 0.92).astype(np.float32) * np.ones(
        (1, 1, 3), dtype=np.float32
    )
    s7 = EvalScenario(
        "07_occluded_cast_shadow",
        "Foreground post casting shadow over painted background wall",
        _to_uint8_srgb(alb7 * shd7),
        (xn > 0.68),
        cast_shadow,
        ~pillar,
        (0.86, 0.28, 0.42),
    )

    # 8. Specular / non-diffuse residual highlight region
    alb8 = np.full((height, width, 3), [0.42, 0.44, 0.48], dtype=np.float32)
    shd8 = (0.75 - 0.25 * yn)[..., None] * np.ones((1, 1, 3), dtype=np.float32)
    dist_sq = (xn - 0.35) ** 2 + (yn - 0.35) ** 2
    spec8 = (0.72 * np.exp(-dist_sq / 0.008))[..., None] * np.ones((1, 1, 3), dtype=np.float32)
    s8 = EvalScenario(
        "08_specular_residual_surface",
        "Glossy wall with strong localized specular highlight",
        _to_uint8_srgb(alb8 * shd8 + spec8),
        yn < 0.45,
        yn > 0.65,
        base_mask,
        (0.28, 0.72, 0.62),
    )

    return [s1, s2, s3, s4, s5, s6, s7, s8]


def _save_strip(
    out_path: Path,
    panels: list[np.ndarray],
) -> None:
    canvas = np.concatenate(panels, axis=1)
    Image.fromarray(canvas, mode="RGB").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Marigold IID Lighting v1.1 on 8 appearance compositing scenarios."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/inkray_appearance_eval"),
        help="Directory where comparison strips are written.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--res", type=int, default=768)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    marigold = MarigoldInference(
        device="auto",
        use_fp16=True,
        denoising_steps=args.steps,
        processing_resolution=args.res,
    )

    scenarios = _build_scenarios()
    for sc in scenarios:
        t0 = time.perf_counter()
        result = marigold.run(sc.srgb_uint8)
        elapsed = time.perf_counter() - t0

        ref_linear = srgb_to_linear(sc.srgb_uint8.astype(np.float32) / 255.0)
        calib = calibrate_shading_scale(
            ref_linear,
            result.albedo.data,
            result.shading.data,
            residual_linear=result.residual.data,
        )
        recon = compute_reconstruction_error(
            ref_linear, result.albedo.data, result.shading.data, result.residual.data
        )

        # Paint a central horizontal band leaving top/bottom 22% untouched to verify zero unpainted shift.
        h, w = sc.srgb_uint8.shape[:2]
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[int(h * 0.22) : int(h * 0.78), int(w * 0.1) : int(w * 0.9)] = 0.92
        paint_lin = np.broadcast_to(
            srgb_to_linear(np.array(sc.paint_color_srgb, dtype=np.float32)), (h, w, 3)
        )

        flat_out = composite_lighting_aware_linear(
            ref_linear, paint_lin, alpha, result.albedo.data, result.shading.data, calib.scale, "flat"
        )
        shd_out = composite_lighting_aware_linear(
            ref_linear,
            paint_lin,
            alpha,
            result.albedo.data,
            result.shading.data,
            calib.scale,
            "shading",
        )
        tex_out = composite_lighting_aware_linear(
            ref_linear,
            paint_lin,
            alpha,
            result.albedo.data,
            result.shading.data,
            calib.scale,
            "texture",
            texture_preservation=1.0,
        )

        unpainted = alpha == 0.0
        unpainted_max_diff = float(np.max(np.abs(shd_out[unpainted] - ref_linear[unpainted])))

        painted = alpha > 0.5
        lit_mask = painted & sc.lit_region
        shd_mask = painted & sc.shadow_region
        lit_luma = float(np.mean(shd_out[lit_mask])) if np.any(lit_mask) else 0.0
        shadow_luma = float(np.mean(shd_out[shd_mask])) if np.any(shd_mask) else 0.0
        tex_std = float(np.std(tex_out[painted & sc.textured_region]))
        shd_std = float(np.std(shd_out[painted & sc.textured_region]))

        shd_p98 = float(np.percentile(result.shading.data, 98.0))
        res_p99 = float(np.percentile(result.residual.data, 99.0))
        panels = [
            sc.srgb_uint8,
            _to_uint8_srgb(result.albedo.data),
            _to_uint8_srgb(result.shading.data / max(shd_p98, 1e-4)),
            _to_uint8_srgb(result.residual.data / max(res_p99, 0.25)),
            _to_uint8_srgb(flat_out),
            _to_uint8_srgb(shd_out),
            _to_uint8_srgb(tex_out),
        ]
        _save_strip(args.output_dir / f"{sc.name}.png", panels)

        print(
            f"[{sc.name}] time={elapsed:.2f}s calib={calib.scale:.2f} (ok={calib.is_reliable}) "
            f"recon_mae={recon.mae:.4f} unpainted_err={unpainted_max_diff:.1e} "
            f"lit_vs_shadow=({lit_luma:.3f} vs {shadow_luma:.3f}) "
            f"std(tex/shd)=({tex_std:.4f}/{shd_std:.4f})"
        )

    if torch.cuda.is_available():
        print(f"peak_cuda_vram={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")


if __name__ == "__main__":
    main()
