import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from shutil import disk_usage
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download

from ..config import MODELS_DIR
from ..appearance.color import srgb_to_linear
from ..appearance.result import AppearanceResult
from ..geometry.normals import NormalMap
from ..utils.image import as_rgb_image
from .base import InferenceModel, ModelUnavailableError
from .memory import resolve_device


class MarigoldV2Runner:
    seed = 2025
    source_url = "https://github.com/huawei-bayerlab/marigold-v2.git"
    source_revision = "9926d93b0780de5f64bee20895b25a56a98ffbf8"
    qwen_repo = "Qwen/Qwen-Image-Edit-2509"
    qwen_revision = "d3968ef930e841f4c73640fb8afa3b306a78167e"
    adapter_repo = "huawei-bayerlab/marigold-v2-0"
    adapter_revision = "cdf9810fb690886391a63aec012b5f501064fb0d"
    qwen_files = (
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        *(f"transformer/diffusion_pytorch_model-{index:05d}-of-00005.safetensors"
          for index in range(1, 6)),
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )
    prompt_prefixes = {
        "albedo": "qwen_edit_2509_qwen_albedo_rgb_dummy512",
        "normals": "qwen_edit_2509_qwen_normals_dummy512",
    }

    def __init__(self, repo: str, python: str, device: str = "auto") -> None:
        self.repo = Path(repo).expanduser().resolve() if repo else MODELS_DIR / "marigold-v2"
        self.script = self.repo / "scripts/infer.py"
        self.use_local_checkout = bool(repo)
        self.python = Path(python or sys.executable).expanduser().absolute()
        self.use_backend_python = not python
        self.device = resolve_device(device)

    def unavailable_reason(self) -> str | None:
        if sys.platform != "linux" or self.device.type != "cuda" or not torch.cuda.is_available():
            return "Marigold V2 requires a Linux CUDA backend."
        if not self.python.is_file():
            return "The executable configured in INKRAY_MARIGOLD_V2_PYTHON was not found."
        if self.use_backend_python and find_spec("marigoldv2") is None:
            return 'Install Marigold V2 with pip install -e ".[marigold-v2]" in the backend environment.'
        if self.use_local_checkout and not self.script.is_file():
            return "The checkout configured in INKRAY_MARIGOLD_V2_REPO was not found."
        return None

    def _ensure_checkout(self) -> None:
        if self.script.is_file():
            return
        try:
            self.repo.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="marigold-v2-", dir=self.repo.parent) as directory:
                checkout = Path(directory) / "source"
                subprocess.run(["git", "init", "-q", str(checkout)], check=True)
                subprocess.run(
                    ["git", "fetch", "--depth=1", self.source_url, self.source_revision],
                    cwd=checkout, check=True,
                )
                subprocess.run(
                    ["git", "checkout", "-q", "--detach", "FETCH_HEAD"],
                    cwd=checkout, check=True,
                )
                try:
                    checkout.rename(self.repo)
                except OSError:
                    if not self.script.is_file():
                        raise
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ModelUnavailableError(
                "Could not prepare the Marigold V2 inference scripts. Check Git and network access."
            ) from exc

    def _ensure_weights(self, assets: Path, modality: Literal["albedo", "normals"]) -> None:
        qwen = assets / "checkpoints" / "Qwen-Image-Edit-2509"
        adapters = assets / "checkpoints" / "Marigold-V2"
        prefix = self.prompt_prefixes[modality]
        adapter_files = (
            f"{modality}/trainables.safetensors",
            f"qwen_text_embeddings/{prefix}_prompt_embeds.pt",
            f"qwen_text_embeddings/{prefix}_prompt_mask.pt",
        )
        try:
            assets.mkdir(parents=True, exist_ok=True)
            pending = []
            required = 0
            for repo, revision, destination, files in (
                (self.qwen_repo, self.qwen_revision, qwen, self.qwen_files),
                (self.adapter_repo, self.adapter_revision, adapters, adapter_files),
            ):
                if all((destination / name).is_file() for name in files):
                    continue
                options = dict(revision=revision, local_dir=destination, allow_patterns=list(files))
                planned = snapshot_download(repo, dry_run=True, **options)
                if {item.filename for item in planned} != set(files):
                    raise OSError(f"Missing files at {repo}@{revision}")
                required += sum(item.file_size for item in planned if item.will_download)
                pending.append((repo, options, destination, files))
            if pending and disk_usage(assets).free < required:
                raise ModelUnavailableError(
                    f"Marigold V2 needs {required / 1e9:.1f} GB of free space "
                    f"to download the missing weights into {assets}."
                )
            for repo, options, destination, files in pending:
                snapshot_download(repo, **options)
                if not all((destination / name).is_file() for name in files):
                    raise OSError(f"Missing files after download from {repo}")
        except OSError as exc:
            raise ModelUnavailableError(
                "Could not download Marigold V2 weights. Check network access and disk space, "
                "then retry."
            ) from exc

    def predict(self, input_data: np.ndarray, modality: Literal["albedo", "normals"]) -> np.ndarray:
        if reason := self.unavailable_reason():
            raise ModelUnavailableError(reason)
        image = as_rgb_image(input_data)
        env = os.environ.copy()
        if self.device.index is not None:
            visible = env.get("CUDA_VISIBLE_DEVICES")
            if self.device.index >= torch.cuda.device_count():
                raise ModelUnavailableError("The configured Marigold V2 CUDA device is unavailable.")
            env["CUDA_VISIBLE_DEVICES"] = (
                visible.split(",")[self.device.index] if visible else str(self.device.index)
            )

        self._ensure_checkout()
        assets = Path(env.get("DEPTH_ASSETS_DIR", "assets")).expanduser()
        assets = (self.repo / assets).resolve()
        self._ensure_weights(assets, modality)
        env["DEPTH_ASSETS_DIR"] = str(assets)
        # The upstream evaluator resolves its YAML files relative to this checkout.
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(self.repo), env.get("PYTHONPATH"))))

        with TemporaryDirectory(prefix="inkray-marigold-v2-") as directory:
            root = Path(directory)
            inputs = root / "input"
            outputs = root / "output"
            inputs.mkdir()
            image.save(inputs / "photo.png")
            try:
                subprocess.run(
                    [
                        str(self.python), str(self.script), "--modality", modality,
                        "--image_dir", str(inputs), "--output_dir", str(outputs),
                        "--seed", str(self.seed),
                    ],
                    cwd=self.repo,
                    env=env,
                    check=True,
                )
                prediction = np.load(outputs / "images/predictions_npy/photo.npy", allow_pickle=False)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ModelUnavailableError(
                    "Marigold V2 inference failed. Check GPU memory, the checkpoint, and backend logs."
                ) from exc

        if prediction.shape != (3, image.height, image.width) or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"Invalid Marigold V2 {modality} prediction with shape {prediction.shape}.")
        return prediction.transpose(1, 2, 0)


class MarigoldV2Inference(InferenceModel[np.ndarray, AppearanceResult]):
    model_id = "huawei-bayerlab/marigold-v2-0/albedo"
    seed = MarigoldV2Runner.seed

    def __init__(self, runner: MarigoldV2Runner) -> None:
        self.runner = runner

    def run(self, input_data: np.ndarray) -> AppearanceResult:
        # The upstream CLI saves its albedo arrays in sRGB.
        albedo = srgb_to_linear(self.runner.predict(input_data, "albedo"))
        return AppearanceResult.from_albedo(
            albedo, self.model_id, max(albedo.shape[:2]), denoising_steps=1, seed=self.seed,
        )


class MarigoldV2NormalsInference(InferenceModel[np.ndarray, NormalMap]):
    model_id = "huawei-bayerlab/marigold-v2-0/normals"

    def __init__(self, runner: MarigoldV2Runner) -> None:
        self.runner = runner

    def run(self, input_data: np.ndarray) -> NormalMap:
        return NormalMap.from_camera_space(self.runner.predict(input_data, "normals"))
