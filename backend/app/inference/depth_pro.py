import math

import numpy as np
import torch

from ..geometry.depth import DepthMap
from ..geometry.intrinsics import CameraIntrinsics
from ..geometry.mask import ValidMask
from ..geometry.normals import NormalMap
from ..geometry.points import Points3D
from ..geometry.result import GeometryResult
from ..utils.image import as_rgb_image
from .base import InferenceModel, ModelUnavailableError
from .memory import move_to_device, offload_to_cpu, resolve_device


class DepthProInference(InferenceModel[np.ndarray, GeometryResult]):
    model_id = "apple/DepthPro-hf"

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.use_fp16 = self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self._model = None
        self._processor = None
        self._active_device = torch.device("cpu")

    def activate_device(self) -> None:
        if self._model is None:
            from transformers import AutoImageProcessor, DepthProForDepthEstimation

            processor = AutoImageProcessor.from_pretrained(self.model_id)
            try:
                model = DepthProForDepthEstimation.from_pretrained(
                    self.model_id, torch_dtype=self.dtype, use_safetensors=True,
                ).eval()
            except OSError as exc:
                raise ModelUnavailableError("Could not load the Apple Depth Pro checkpoint.") from exc
            self._processor, self._model = processor, model
        self._active_device = move_to_device(self._model, self._active_device, self.device)

    def offload_to_cpu(self) -> None:
        self._active_device = offload_to_cpu(self._model, self._active_device)

    @torch.inference_mode()
    def run(self, input_data: np.ndarray) -> GeometryResult:
        image = as_rgb_image(input_data)
        self.activate_device()
        assert self._model is not None and self._processor is not None

        width, height = image.size
        pixels = self._processor(images=image, return_tensors="pt").pixel_values
        output = self._model(pixel_values=pixels.to(device=self.device, dtype=self.dtype))
        if output.field_of_view is None or not 0 < float(output.field_of_view.item()) < 180:
            raise RuntimeError("Depth Pro returned an invalid horizontal field of view.")
        output.field_of_view = output.field_of_view.float()
        output.predicted_depth = output.predicted_depth.float()
        prediction = self._processor.post_process_depth_estimation(
            output, target_sizes=[(height, width)],
        )[0]

        focal = float(prediction["focal_length"].item())
        if not math.isfinite(focal) or focal <= 0:
            raise RuntimeError("Depth Pro returned an invalid focal length.")
        depth = prediction["predicted_depth"].cpu().numpy().reshape(height, width)
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            raise RuntimeError("Depth Pro returned no valid depth values.")
        depth = np.where(valid, depth, np.nan)
        intrinsics = CameraIntrinsics.from_pixel_matrix(
            np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]),
            width,
            height,
        )
        points = intrinsics.unproject_depth(depth)
        normals = NormalMap.from_points(points) if min(width, height) > 1 else None
        return GeometryResult(
            depth=DepthMap(depth),
            normals=normals,
            points=Points3D(points),
            intrinsics=intrinsics,
            mask=ValidMask(valid),
        )
