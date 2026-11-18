from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from ..appearance.result import AppearanceResult
from ..utils.image import as_rgb_image
from .base import InferenceModel, ModelUnavailableError
from .memory import move_to_device, offload_to_cpu, resolve_device


class LatentIntrinsicsInference(InferenceModel[np.ndarray, AppearanceResult]):
    model_id = "xiao7199/Latent_Intrinsics"
    patch_size = 256
    checkpoint_repo = "xyxingx/ALI_RADIO2.5h"
    checkpoint_name = "latent_intrinsics.pth.tar"
    checkpoint_revision = "d33406475dba2b3d20043d62b868918972f4bc6f"

    def __init__(self, repo: str, checkpoint: str, device: str = "auto") -> None:
        self.source = Path(repo).expanduser().resolve() / "unets.py" if repo else None
        self.checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else None
        self.device = resolve_device(device)
        self._active_device = torch.device("cpu")
        self._model: torch.nn.Module | None = None

    def cache_variant(self) -> str:
        if self.checkpoint is not None and self.checkpoint.is_file():
            stamp = self.checkpoint.stat()
            return f"latent-intrinsics:{self.checkpoint}:{stamp.st_size}:{stamp.st_mtime_ns}"
        return f"latent-intrinsics:{self.checkpoint_revision}"

    def unavailable_reason(self) -> str | None:
        if self.source is None or not self.source.is_file():
            return "Install the Latent Intrinsics checkout under backend/.cache/models/latent-intrinsics (see README)."
        if self.checkpoint is not None and not self.checkpoint.is_file():
            return "The checkpoint configured in INKRAY_LATENT_INTRINSICS_CHECKPOINT was not found."
        return None

    def _load_model(self) -> torch.nn.Module:
        if reason := self.unavailable_reason():
            raise ModelUnavailableError(reason)
        assert self.source is not None
        checkpoint = self.checkpoint
        if checkpoint is None:
            from huggingface_hub import hf_hub_download

            try:
                checkpoint = Path(hf_hub_download(
                    self.checkpoint_repo, filename=self.checkpoint_name,
                    revision=self.checkpoint_revision,
                ))
            except OSError as exc:
                raise ModelUnavailableError(
                    "Could not download Latent Intrinsics weights from Hugging Face. "
                    "Check the backend network connection or configure a local checkpoint."
                ) from exc
        spec = spec_from_file_location("inkray_latent_intrinsics_unets", self.source)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.UNet(
            img_resolution=self.patch_size, in_channels=3, out_channels=3,
            model_channels=32, channel_mult=[1, 2, 4, 4, 8, 16],
            num_blocks_list=[1, 2, 2, 4, 4, 4], attn_resolutions=[0], affine_scale=0,
        )
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state_dict = weights["ema_state_dict"]
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
        model.load_state_dict(state_dict)
        return model.eval()

    def activate_device(self) -> None:
        if self._model is None:
            self._model = self._load_model()
        self._active_device = move_to_device(self._model, self._active_device, self.device)

    def offload_to_cpu(self) -> None:
        self._active_device = offload_to_cpu(self._model, self._active_device)

    @torch.inference_mode()
    def run(self, input_data: np.ndarray) -> AppearanceResult:
        image = as_rgb_image(input_data)
        self.activate_device()
        assert self._model is not None
        rgb = torch.tensor(np.asarray(image), dtype=torch.float32, device=self.device)
        rgb = rgb.permute(2, 0, 1) / 127.5 - 1
        width, height = image.size
        combined = torch.zeros_like(rgb)
        weight_sum = rgb.new_zeros(height, width)

        # Blend local and large crops with the Gaussian window used in the IIW evaluation.
        axis = torch.linspace(-1, 1, self.patch_size, device=self.device)
        window = torch.exp(-4 * (axis[:, None].square() + axis.square()))[None, None]
        for size in sorted({self.patch_size, max(self.patch_size, min(width, height))}):
            tile_h, tile_w = min(size, height), min(size, width)
            stride = size // 8
            rows = [*range(0, height - tile_h, stride), height - tile_h]
            cols = [*range(0, width - tile_w, stride), width - tile_w]
            tiles = list(product(rows, cols))
            weight = F.interpolate(window, (tile_h, tile_w), mode="bilinear", align_corners=False)[0, 0]
            for start in range(0, len(tiles), 4):
                positions = tiles[start:start + 4]
                crops = torch.stack([
                    rgb[:, row:row + tile_h, col:col + tile_w] for row, col in positions
                ])
                patches = F.interpolate(
                    crops, (self.patch_size, self.patch_size), mode="bilinear", align_corners=False,
                )
                features = self._model(patches, run_encoder=True)
                albedo = self._model(features, run_encoder=False).clamp(-1, 1)
                resized = F.interpolate(albedo, (tile_h, tile_w), mode="bilinear", align_corners=False)
                for (row, col), tile in zip(positions, resized):
                    y, x = slice(row, row + tile_h), slice(col, col + tile_w)
                    combined[:, y, x] += tile * weight
                    weight_sum[y, x] += weight

        albedo = ((combined / weight_sum + 1) * 0.5).permute(1, 2, 0).cpu().numpy()
        return AppearanceResult.from_albedo(
            albedo, self.model_id, self.patch_size, scale_convention="up_to_scale",
        )
