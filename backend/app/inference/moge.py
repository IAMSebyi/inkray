from moge.model.v3 import MoGeModel
import numpy as np
import torch

from ..geometry.depth import DepthMap
from ..geometry.intrinsics import CameraIntrinsics
from ..geometry.mask import ValidMask
from ..geometry.normals import NormalMap
from ..geometry.points import Points3D
from ..geometry.result import GeometryResult
from .base import InferenceModel


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class MoGeInference(InferenceModel[np.ndarray, GeometryResult]):
    def __init__(
        self,
        model_id: str = "Ruicheng/moge-3-vitl",
        device: str = "auto",
        resolution_level: int = 9,
        refine_steps: int = 3,
        use_fp16: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.resolution_level = resolution_level
        self.refine_steps = refine_steps
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model = MoGeModel.from_pretrained(model_id).to(self.device).eval()

    def run(self, input_data: np.ndarray) -> GeometryResult:
        if input_data.ndim != 3 or input_data.shape[2] != 3:
            raise ValueError(f"Expected RGB image array (H, W, 3), got {input_data.shape}.")

        if np.issubdtype(input_data.dtype, np.floating):
            rgb = np.clip(input_data, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            rgb = input_data.astype(np.float32) / 255.0

        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(self.device)

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
