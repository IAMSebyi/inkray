from importlib.util import find_spec
import sys

import numpy as np
import torch

from ..geometry.depth import DepthMap
from ..geometry.intrinsics import CameraIntrinsics
from ..geometry.mask import ValidMask
from ..geometry.normals import NormalMap
from ..geometry.points import Points3D
from ..geometry.result import GeometryResult
from ..utils.image import as_rgb_float01
from .base import InferenceModel, ModelUnavailableError
from .memory import move_to_device, offload_to_cpu, resolve_device


class MoGeInference(InferenceModel[np.ndarray, GeometryResult]):
    def __init__(
        self,
        model_id: str = "Ruicheng/moge-3-vitl",
        device: str = "auto",
        resolution_level: int = 9,
        refine_steps: int = 3,
        use_fp16: bool = True,
        lazy_load: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.resolution_level = resolution_level
        self.refine_steps = refine_steps
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model = None
        self._active_device = torch.device("cpu")

        if not lazy_load:
            self.activate_device()

    def unavailable_reason(self) -> str | None:
        if sys.platform not in ("linux", "win32"):
            return "MoGe-3 requires Linux or Windows."
        if self.refine_steps > 0 and (self.device.type != "cuda" or not torch.cuda.is_available()):
            return "MoGe-3 sparse refinement requires CUDA."
        if find_spec("moge") is None:
            return 'Install the MoGe-3 extra with pip install -e ".[moge]".'
        return None

    def activate_device(self) -> None:
        if self.model is None:
            if reason := self.unavailable_reason():
                raise ModelUnavailableError(reason)
            try:
                from moge.model.v3 import MoGeModel
            except ImportError as exc:
                raise ModelUnavailableError("Check the MoGe-3, FlexGEMM, and Triton installation.") from exc

            self.model = MoGeModel.from_pretrained(self.model_id).eval()
        self._active_device = move_to_device(self.model, self._active_device, self.device)

    def offload_to_cpu(self) -> None:
        self._active_device = offload_to_cpu(self.model, self._active_device)

    def run(self, input_data: np.ndarray) -> GeometryResult:
        rgb = as_rgb_float01(input_data)
        self.activate_device()
        assert self.model is not None

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        with torch.inference_mode():
            output = self.model.infer(
                tensor,
                resolution_level=self.resolution_level,
                refine_steps=self.refine_steps,
                use_fp16=self.use_fp16,
            )

        depth = output["depth"].detach().cpu().numpy().astype(np.float32, copy=False)
        points = output["points"].detach().cpu().numpy().astype(np.float32, copy=False)
        intrinsics = output["intrinsics"].detach().cpu().numpy().astype(np.float32, copy=False)
        mask = output["mask"].detach().cpu().numpy().astype(bool, copy=False)

        normal_tensor = output.get("normal")
        normals = (
            NormalMap(normal_tensor.detach().cpu().numpy().astype(np.float32, copy=False))
            if normal_tensor is not None
            else None
        )

        return GeometryResult(
            depth=DepthMap(depth),
            normals=normals,
            points=Points3D(points),
            intrinsics=CameraIntrinsics(intrinsics),
            mask=ValidMask(mask),
        )
