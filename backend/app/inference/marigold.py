from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np
import torch

from ..appearance.maps import AlbedoMap, ResidualMap, ShadingMap
from ..appearance.result import AppearanceResult, default_scale_conventions
from ..geometry.normals import NormalMap
from ..utils.image import as_rgb_float01
from .base import InferenceModel, ModelUnavailableError
from .memory import move_to_device, offload_to_cpu, resolve_device

OutputT = TypeVar("OutputT")


class _DiffusersMarigoldModel(InferenceModel[np.ndarray, OutputT], Generic[OutputT]):
    """Shared Diffusers Marigold pipeline lifecycle and deterministic inference."""

    model_id: str = ""
    pipeline_class_name: str = ""

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "auto",
        use_fp16: bool = True,
        denoising_steps: int = 4,
        processing_resolution: int = 768,
        ensemble_size: int = 1,
        seed: int = 42,
        lazy_load: bool = False,
    ) -> None:
        self.model_id = model_id or type(self).model_id
        self.device = resolve_device(device)
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.denoising_steps = denoising_steps
        self.processing_resolution = processing_resolution
        self.ensemble_size = ensemble_size
        self.seed = seed
        self._pipeline: Any = None
        self._active_device = torch.device("cpu")

        if not lazy_load:
            self.activate_device()

    def _ensure_loaded(self) -> Any:
        if self._pipeline is None:
            import diffusers

            pipeline_cls = getattr(diffusers, self.pipeline_class_name)
            load_kwargs: dict[str, object] = {}
            if self.use_fp16:
                load_kwargs["variant"] = "fp16"
                load_kwargs["torch_dtype"] = torch.float16
            try:
                self._pipeline = pipeline_cls.from_pretrained(self.model_id, **load_kwargs)
            except OSError as exc:
                raise ModelUnavailableError(
                    f"Could not load Marigold checkpoint {self.model_id!r}."
                ) from exc
            self._active_device = torch.device("cpu")
        return self._pipeline

    def activate_device(self) -> None:
        pipeline = self._ensure_loaded()
        self._active_device = move_to_device(pipeline, self._active_device, self.device)

    def offload_to_cpu(self) -> None:
        self._active_device = offload_to_cpu(self._pipeline, self._active_device)

    def _run_pipeline(self, input_data: np.ndarray) -> tuple[Any, Any]:
        rgb = as_rgb_float01(input_data)
        self.activate_device()
        assert self._pipeline is not None
        pipe = self._pipeline

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
        return pipe, output


class MarigoldInference(_DiffusersMarigoldModel[AppearanceResult]):
    """Intrinsic image decomposition via Diffusers Marigold IID Lighting v1.1."""

    model_id = "prs-eth/marigold-iid-lighting-v1-1"
    pipeline_class_name = "MarigoldIntrinsicsPipeline"

    def run(self, input_data: np.ndarray) -> AppearanceResult:
        pipe, output = self._run_pipeline(input_data)
        target_props = dict(pipe.config.target_properties or {})
        target_names = list(target_props.get("target_names", ("albedo", "shading", "residual")))

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
        scale_conventions: dict[str, str] = {}
        prediction_space = "linear"
        for name in target_names:
            meta = target_props.get(name, {})
            if isinstance(meta, dict):
                scale_conventions[name] = (
                    "up_to_scale" if meta.get("up_to_scale") else "bounded_0_1"
                )
                if name == "albedo" and isinstance(meta.get("prediction_space"), str):
                    prediction_space = meta["prediction_space"]

        return AppearanceResult(
            albedo=AlbedoMap(by_name["albedo"]),
            shading=ShadingMap(by_name["shading"]),
            residual=ResidualMap(by_name["residual"]),
            model_id=self.model_id,
            denoising_steps=self.denoising_steps,
            processing_resolution=self.processing_resolution,
            ensemble_size=self.ensemble_size,
            seed=self.seed,
            prediction_space=prediction_space,
            scale_conventions=scale_conventions or default_scale_conventions(),
        )


class MarigoldNormalsInference(_DiffusersMarigoldModel[NormalMap]):
    """Surface normal prediction via Diffusers Marigold Normals v1.1."""

    model_id = "prs-eth/marigold-normals-v1-1"
    pipeline_class_name = "MarigoldNormalsPipeline"

    def run(self, input_data: np.ndarray) -> NormalMap:
        _, output = self._run_pipeline(input_data)
        pred = np.asarray(output.prediction, dtype=np.float32)
        if pred.ndim == 4 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim != 3 or pred.shape[2] != 3 or not np.all(np.isfinite(pred)):
            raise RuntimeError(f"Unexpected Marigold normals prediction shape {pred.shape}.")
        return NormalMap.from_camera_space(pred)
