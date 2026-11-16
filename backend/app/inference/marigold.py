from diffusers import MarigoldIntrinsicsPipeline
import numpy as np
import torch

from ..appearance.maps import AlbedoMap, ResidualMap, ShadingMap
from ..appearance.result import AppearanceResult, default_scale_conventions
from .base import InferenceModel
from .moge import resolve_device


class MarigoldInference(InferenceModel[np.ndarray, AppearanceResult]):
    """Intrinsic image decomposition via Diffusers Marigold IID Lighting v1.1."""

    def __init__(
        self,
        model_id: str = "prs-eth/marigold-iid-lighting-v1-1",
        device: str = "auto",
        use_fp16: bool = True,
        denoising_steps: int = 4,
        processing_resolution: int = 768,
        ensemble_size: int = 1,
        seed: int = 42,
        lazy_load: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.torch_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.denoising_steps = max(1, int(denoising_steps))
        self.processing_resolution = max(0, int(processing_resolution))
        self.ensemble_size = max(1, int(ensemble_size))
        self.seed = int(seed)
        self._pipeline: MarigoldIntrinsicsPipeline | None = None
        self._active_device = torch.device("cpu")

        if not lazy_load:
            self.activate_device()

    def _ensure_loaded(self) -> MarigoldIntrinsicsPipeline:
        if self._pipeline is None:
            load_kwargs: dict[str, object] = {}
            if self.use_fp16:
                load_kwargs["variant"] = "fp16"
                load_kwargs["dtype"] = torch.float16
            self._pipeline = MarigoldIntrinsicsPipeline.from_pretrained(self.model_id, **load_kwargs)
            self._active_device = torch.device("cpu")
        return self._pipeline

    def activate_device(self) -> None:
        pipe = self._ensure_loaded()
        if self._active_device != self.device:
            pipe.to(self.device)
            self._active_device = self.device

    def offload_to_cpu(self) -> None:
        if self._pipeline is not None and self._active_device.type == "cuda":
            for attr in ("unet", "vae", "text_encoder", "safety_checker"):
                module = getattr(self._pipeline, attr, None)
                if isinstance(module, torch.nn.Module):
                    module.to("cpu")
            self._active_device = torch.device("cpu")
            torch.cuda.empty_cache()

    def run(self, input_data: np.ndarray) -> AppearanceResult:
        if input_data.ndim != 3 or input_data.shape[2] != 3:
            raise ValueError(f"Expected RGB image array (H, W, 3), got {input_data.shape}.")

        if np.issubdtype(input_data.dtype, np.floating):
            rgb = np.clip(input_data, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            rgb = (np.asarray(input_data, dtype=np.float32) / 255.0).clip(0.0, 1.0)
        rgb = np.ascontiguousarray(rgb)

        self.activate_device()
        assert self._pipeline is not None
        pipe = self._pipeline

        target_props = dict(getattr(pipe, "target_properties", {}) or {})
        target_names = list(target_props.get("target_names", ("albedo", "shading", "residual")))

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        with torch.inference_mode():
            output = pipe(
                rgb,
                num_inference_steps=self.denoising_steps,
                ensemble_size=self.ensemble_size,
                processing_resolution=self.processing_resolution,
                match_input_resolution=True,
                generator=generator,
                output_type="np",
            )

        pred = np.asarray(output.prediction, dtype=np.float32)
        if pred.ndim == 5:
            pred = pred[0]
        if pred.ndim != 4 or pred.shape[0] != len(target_names) or pred.shape[-1] != 3:
            raise RuntimeError(
                f"Unexpected Marigold prediction shape {pred.shape} for targets {target_names}."
            )

        by_name = {
            name: np.ascontiguousarray(pred[idx], dtype=np.float32)
            for idx, name in enumerate(target_names)
        }
        albedo = AlbedoMap(by_name["albedo"])
        shading = ShadingMap(by_name["shading"])
        residual = ResidualMap(by_name["residual"])

        scale_conventions: dict[str, str] = {}
        prediction_space = "linear"
        for name in target_names:
            meta = target_props.get(name, {})
            if isinstance(meta, dict):
                scale_conventions[name] = "up_to_scale" if meta.get("up_to_scale") else "bounded_0_1"
                if name == "albedo" and isinstance(meta.get("prediction_space"), str):
                    prediction_space = meta["prediction_space"]

        return AppearanceResult(
            albedo=albedo,
            shading=shading,
            residual=residual,
            model_id=self.model_id,
            denoising_steps=self.denoising_steps,
            processing_resolution=self.processing_resolution,
            ensemble_size=self.ensemble_size,
            seed=self.seed,
            prediction_space=prediction_space,
            scale_conventions=scale_conventions or default_scale_conventions(),
        )
