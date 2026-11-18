from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..appearance.maps import AlbedoMap, ResidualMap, ShadingMap
from ..appearance.result import AppearanceResult, default_scale_conventions
from ..utils.image import as_rgb_float01
from .base import InferenceModel, ModelUnavailableError
from .memory import move_to_device, offload_to_cpu, resolve_device


class _ModuleGroup:
    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        self.modules = modules

    def to(self, device: object) -> None:
        for module in self.modules.values():
            module.to(device)


def _match_spatial_shape(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape[:2] == target_hw:
        return np.ascontiguousarray(arr, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return np.ascontiguousarray(resized.squeeze(0).permute(1, 2, 0).numpy(), dtype=np.float32)


class IidInTheWildInference(InferenceModel[np.ndarray, AppearanceResult]):
    """Intrinsic image decomposition via Careaga & Aksoy's IID-in-the-Wild (academic only)."""

    model_id = "compphoto/Intrinsic/v2"

    def __init__(
        self,
        weights: str = "v2",
        base_size: int = 384,
        device: str = "auto",
    ) -> None:
        self.weights = weights or "v2"
        self.base_size = base_size
        self.device = resolve_device(device)
        self._models: dict[str, torch.nn.Module] | None = None
        self._module_group: _ModuleGroup | None = None
        self._active_device = torch.device("cpu")

    def unavailable_reason(self) -> str | None:
        for package in ("intrinsic", "chrislib", "altered_midas", "skimage"):
            if find_spec(package) is None:
                return (
                    'Install IID-in-the-Wild (non-commercial) with '
                    'pip install -e ".[iid-in-the-wild]".'
                )
        return None

    def _ensure_loaded(self) -> _ModuleGroup:
        if self._models is None or self._module_group is None:
            if reason := self.unavailable_reason():
                raise ModelUnavailableError(reason)
            from intrinsic.pipeline import load_models

            try:
                torch.hub.list(
                    "rwightman/gen-efficientnet-pytorch",
                    trust_repo=True,
                    verbose=False,
                )
                loaded: dict[str, Any] = load_models(self.weights, device="cpu")
            except (OSError, RuntimeError, ValueError) as exc:
                raise ModelUnavailableError(
                    "Could not download or load IID-in-the-Wild weights. "
                    "Check network access and retry."
                ) from exc
            self._models = loaded
            self._module_group = _ModuleGroup(loaded)
            self._active_device = torch.device("cpu")
        return self._module_group

    def activate_device(self) -> None:
        group = self._ensure_loaded()
        self._active_device = move_to_device(group, self._active_device, self.device)

    def offload_to_cpu(self) -> None:
        self._active_device = offload_to_cpu(self._module_group, self._active_device)

    def run(self, input_data: np.ndarray) -> AppearanceResult:
        rgb = as_rgb_float01(input_data)
        height, width = rgb.shape[:2]
        self.activate_device()
        assert self._models is not None

        from intrinsic.pipeline import run_pipeline

        with torch.inference_mode():
            results = run_pipeline(
                self._models,
                rgb,
                stage=4,
                resize_conf=None,
                base_size=self.base_size,
                linear=False,
                device=str(self.device),
            )

        target_hw = (height, width)
        albedo = _match_spatial_shape(results["hr_alb"], target_hw)
        shading = _match_spatial_shape(results["dif_shd"], target_hw)
        residual = _match_spatial_shape(results["residual"], target_hw)

        for layer in (albedo, shading, residual):
            if layer.shape != (height, width, 3) or not np.all(np.isfinite(layer)):
                raise RuntimeError("IID-in-the-Wild returned invalid appearance maps.")

        return AppearanceResult(
            albedo=AlbedoMap(np.clip(albedo, 0.0, 1.0)),
            shading=ShadingMap(np.maximum(shading, 0.0)),
            residual=ResidualMap(residual),
            model_id=self.model_id,
            processing_resolution=self.base_size,
            prediction_space="linear",
            scale_conventions=default_scale_conventions(),
        )
