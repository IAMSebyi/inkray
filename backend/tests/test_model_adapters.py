import sys
from pathlib import Path
import shutil
import subprocess
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from app import config
from app.inference import iid_in_the_wild
from app.inference.depth_pro import DepthProInference
from app.inference.iid_in_the_wild import IidInTheWildInference
from app.inference.latent_intrinsics import LatentIntrinsicsInference
from app.inference.marigold import MarigoldInference, MarigoldNormalsInference
from app.inference.marigold_v2 import MarigoldV2NormalsInference, MarigoldV2Runner


def test_marigold_iid_uses_full_precision_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from diffusers import MarigoldIntrinsicsPipeline

    predictions = np.broadcast_to(
        np.array([0.7, 0.2, 0.05], dtype=np.float32)[None, :, None, None, None],
        (1, 3, 2, 4, 3),
    )
    pipe = Mock(return_value=SimpleNamespace(prediction=predictions))
    pipe.config.target_properties = {
        "target_names": ["shading", "albedo", "residual"],
        "albedo": {"prediction_space": "linear"},
        "shading": {"up_to_scale": True},
        "residual": {"up_to_scale": True},
    }
    load = Mock(return_value=pipe)
    monkeypatch.setattr(MarigoldIntrinsicsPipeline, "from_pretrained", load)

    model = MarigoldInference(device="cpu", use_fp16=True, seed=13, lazy_load=True)
    load.assert_not_called()
    result = model.run(np.full((2, 4, 3), 128, dtype=np.uint8))
    load.assert_called_once_with(model.model_id)
    rgb = pipe.call_args.args[0]
    assert rgb.dtype == np.float32
    np.testing.assert_allclose(rgb, 128 / 255)
    generator = pipe.call_args.kwargs["generator"]
    assert generator.device.type == "cpu"
    assert generator.initial_seed() == 13
    assert result.shape == (2, 4)
    np.testing.assert_allclose(result.albedo.data, 0.2)
    assert result.shading is not None and result.residual is not None
    np.testing.assert_allclose(result.shading.data, 0.7)
    np.testing.assert_allclose(result.residual.data, 0.05)


def test_marigold_iid_requests_fp16_checkpoint_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from diffusers import MarigoldIntrinsicsPipeline

    pipe = Mock()
    load = Mock(return_value=pipe)
    monkeypatch.setattr(MarigoldIntrinsicsPipeline, "from_pretrained", load)
    model = MarigoldInference(device="cuda", lazy_load=True)

    model.activate_device()

    load.assert_called_once_with(model.model_id, variant="fp16", torch_dtype=torch.float16)
    pipe.to.assert_called_once_with(torch.device("cuda"))


def test_marigold_normals_converts_camera_axes_and_uses_cpu_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import MarigoldNormalsPipeline

    predictions = np.broadcast_to(np.array([0.2, 0.4, 0.8], dtype=np.float32), (1, 2, 3, 3))
    pipe = Mock(return_value=SimpleNamespace(prediction=predictions))
    load = Mock(return_value=pipe)
    monkeypatch.setattr(MarigoldNormalsPipeline, "from_pretrained", load)

    model = MarigoldNormalsInference(device="cpu", use_fp16=True, seed=17, lazy_load=True)
    load.assert_not_called()
    normals = model.run(np.full((2, 3, 3), 128, dtype=np.uint8))

    load.assert_called_once_with(model.model_id)
    rgb = pipe.call_args.args[0]
    assert rgb.dtype == np.float32
    np.testing.assert_allclose(rgb, 128 / 255)
    assert pipe.call_args.kwargs["generator"].initial_seed() == 17
    assert normals.shape == (2, 3, 3)
    np.testing.assert_allclose(normals.data[0, 0], [0.2, -0.4, -0.8])


def test_marigold_normals_requests_fp16_on_cuda_and_offloads_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import MarigoldNormalsPipeline

    pipe = Mock()
    load = Mock(return_value=pipe)
    monkeypatch.setattr(MarigoldNormalsPipeline, "from_pretrained", load)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    model = MarigoldNormalsInference(device="cuda", lazy_load=True)

    model.activate_device()
    load.assert_called_once_with(model.model_id, variant="fp16", torch_dtype=torch.float16)
    pipe.to.assert_called_with(torch.device("cuda"))

    model.offload_to_cpu()
    pipe.to.assert_called_with("cpu")


def test_depth_pro_postprocessing_and_intrinsics_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoImageProcessor, DepthProForDepthEstimation, DepthProImageProcessor

    processor = DepthProImageProcessor(size={"height": 8, "width": 8})
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    model = Mock()
    model.eval.return_value = model
    model.return_value = SimpleNamespace(
        field_of_view=torch.tensor([90.0]), predicted_depth=torch.ones((1, 8, 8)),
    )
    checkpoint = Mock(return_value=model)
    monkeypatch.setattr(DepthProForDepthEstimation, "from_pretrained", checkpoint)

    inference = DepthProInference(device="cpu")
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    result = inference.run(image)
    assert result.shape == (2, 4)
    np.testing.assert_allclose(result.depth.data, 0.5, atol=1e-6)
    np.testing.assert_allclose(result.points.data[..., 2], result.depth.data)
    np.testing.assert_allclose(result.intrinsics.to_pixel_matrix(4, 2),
                               [[2, 0, 2], [0, 2, 1], [0, 0, 1]], atol=1e-6)
    assert result.mask.valid_count == 8
    assert result.normals is not None
    np.testing.assert_allclose(result.normals.data[..., 2], -1)
    assert model.call_args.kwargs["pixel_values"].device.type == "cpu"
    model.return_value.field_of_view = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="invalid horizontal field of view"):
        inference.run(image)
    checkpoint.assert_called_once()


def test_marigold_v2_normals_convert_camera_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = MarigoldV2Runner("", "", device="cpu")
    model = MarigoldV2NormalsInference(runner)
    predict = Mock(return_value=np.broadcast_to(np.array([0.2, 0.4, 0.8]), (2, 3, 3)))
    monkeypatch.setattr(runner, "predict", predict)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    result = model.run(image)
    assert predict.call_args.args[0] is image
    assert predict.call_args.args[1] == "normals"
    np.testing.assert_allclose(result.data[0, 0], [0.2, -0.4, -0.8])


def test_marigold_v2_uses_backend_python_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.inference import marigold_v2

    script = tmp_path / "scripts" / "infer.py"
    script.parent.mkdir()
    script.touch()
    python = marigold_v2.sys.executable
    monkeypatch.setattr(marigold_v2, "sys", SimpleNamespace(platform="linux", executable=python))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    find_spec = Mock(return_value=None)
    monkeypatch.setattr(marigold_v2, "find_spec", find_spec)
    runner = MarigoldV2Runner(str(tmp_path), "", device="cuda")

    assert runner.python == Path(python)
    assert 'pip install -e ".[marigold-v2]"' in (runner.unavailable_reason() or "")
    find_spec.assert_called_once_with("marigoldv2")
    find_spec.return_value = object()
    assert runner.unavailable_reason() is None
    monkeypatch.setattr(marigold_v2, "MODELS_DIR", tmp_path)
    assert MarigoldV2Runner("", "", device="cuda").unavailable_reason() is None


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required to prepare V2 source")
def test_marigold_v2_prepares_the_pinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.inference import marigold_v2

    source = tmp_path / "upstream"
    script = source / "scripts" / "infer.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('pinned')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "add", "scripts/infer.py"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "--no-gpg-sign", "-qm", "Inference script"],
        cwd=source, check=True,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    monkeypatch.setattr(marigold_v2, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(MarigoldV2Runner, "source_url", str(source))
    monkeypatch.setattr(MarigoldV2Runner, "source_revision", revision)
    runner = MarigoldV2Runner("", "", device="cpu")

    runner._ensure_checkout()
    runner._ensure_checkout()

    assert runner.script.read_text(encoding="utf-8") == script.read_text(encoding="utf-8")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runner.repo, text=True).strip() == revision


@pytest.mark.parametrize("shape", [(3, 2, 3), (3, 2, 2)])
def test_marigold_v2_rejects_invalid_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int, int],
) -> None:
    from app.inference import marigold_v2

    script = tmp_path / "scripts" / "infer.py"
    script.parent.mkdir()
    script.touch()
    python = Path(marigold_v2.sys.executable)
    monkeypatch.setattr(marigold_v2, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runner = MarigoldV2Runner(str(tmp_path), str(python), device="cuda")
    monkeypatch.setattr(runner, "_ensure_weights", Mock())

    def run(command, **kwargs):
        assert command[command.index("--modality") + 1] == "albedo"
        output = Path(command[command.index("--output_dir") + 1]) / "images" / "predictions_npy"
        output.mkdir(parents=True)
        np.save(output / "photo.npy", np.full(shape, np.nan, dtype=np.float32))

    monkeypatch.setattr(marigold_v2.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Invalid Marigold V2 albedo prediction"):
        runner.predict(np.zeros((2, 3, 3), dtype=np.uint8), "albedo")


def test_marigold_v2_downloads_only_requested_weights_and_reuses_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.inference import marigold_v2

    runner = MarigoldV2Runner(str(tmp_path), "", device="cpu")
    downloads = []

    def download(repo, *, revision, local_dir, allow_patterns, dry_run=False):
        if dry_run:
            return [
                SimpleNamespace(filename=name, file_size=1, will_download=True)
                for name in allow_patterns
            ]
        downloads.append((repo, revision, tuple(allow_patterns)))
        for name in allow_patterns:
            target = local_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    monkeypatch.setattr(marigold_v2, "snapshot_download", download)

    runner._ensure_weights(tmp_path, "albedo")
    runner._ensure_weights(tmp_path, "normals")
    runner._ensure_weights(tmp_path, "albedo")

    assert [repo for repo, _, _ in downloads] == [
        runner.qwen_repo, runner.adapter_repo, runner.adapter_repo,
    ]
    assert downloads[0][1] == runner.qwen_revision
    assert {name.split("/")[0] for name in downloads[0][2]} == {"transformer", "vae"}
    for (_, revision, files), modality in zip(downloads[1:], ("albedo", "normals")):
        assert revision == runner.adapter_revision
        assert files[0] == f"{modality}/trainables.safetensors"
        assert all(runner.prompt_prefixes[modality] in name for name in files[1:])


def test_marigold_v2_rejects_download_with_insufficient_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.inference import marigold_v2
    from app.inference.base import ModelUnavailableError

    def plan(repo, *, allow_patterns, **kwargs):
        size = 41_000_000_000 if repo == MarigoldV2Runner.qwen_repo else 2_000_000_000
        return [
            SimpleNamespace(filename=name, file_size=size if index == 0 else 0,
                            will_download=True)
            for index, name in enumerate(allow_patterns)
        ]

    download = Mock(side_effect=plan)
    monkeypatch.setattr(marigold_v2, "snapshot_download", download)
    monkeypatch.setattr(marigold_v2, "disk_usage", lambda _: SimpleNamespace(free=42_000_000_000))
    runner = MarigoldV2Runner(str(tmp_path), "", device="cpu")

    with pytest.raises(ModelUnavailableError, match="43.0 GB of free space"):
        runner._ensure_weights(tmp_path, "albedo")

    assert download.call_count == 2
    assert all(call.kwargs["dry_run"] for call in download.call_args_list)


@pytest.mark.parametrize("local_checkpoint", [True, False])
def test_latent_intrinsics_uses_ema_and_covers_overlapping_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_checkpoint: bool,
) -> None:
    source = tmp_path / "unets.py"
    source.write_text(
        "import torch\n"
        "class UNet(torch.nn.Module):\n"
        "    def __init__(self, **kwargs):\n"
        "        super().__init__()\n"
        "        self.value = torch.nn.Parameter(torch.zeros(1))\n"
        "    def forward(self, x, run_encoder):\n"
        "        return x if run_encoder else self.value.expand_as(x)\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "latent.pth.tar"
    torch.save({
        "state_dict": {"module.value": torch.tensor([-1.0])},
        "ema_state_dict": {"module.value": torch.tensor([0.25])},
    }, checkpoint)
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    model = LatentIntrinsicsInference(
        str(tmp_path), str(checkpoint) if local_checkpoint else "", device="cpu",
    )
    assert model.unavailable_reason() is None
    download.assert_not_called()
    model.patch_size = 8
    result = model.run(np.zeros((9, 11, 3), dtype=np.uint8))
    assert result.shape == (9, 11)
    assert result.shading is None and result.residual is None
    np.testing.assert_allclose(result.albedo.data, 0.625, atol=1e-6)
    if local_checkpoint:
        download.assert_not_called()
    else:
        download.assert_called_once_with(
            model.checkpoint_repo, filename=model.checkpoint_name, revision=model.checkpoint_revision,
        )
        assert model.cache_variant().endswith(model.checkpoint_revision)


def test_optional_model_paths_default_to_project_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    variables = (
        "INKRAY_MARIGOLD_V2_REPO", "INKRAY_MARIGOLD_V2_PYTHON",
        "INKRAY_LATENT_INTRINSICS_REPO", "INKRAY_LATENT_INTRINSICS_CHECKPOINT",
    )
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    settings = config.Settings()
    assert not settings.marigold_v2_repo
    assert not settings.marigold_v2_python
    assert settings.latent_intrinsics_repo == str(tmp_path / "latent-intrinsics")
    assert not settings.latent_intrinsics_checkpoint
    for variable in variables:
        monkeypatch.setenv(variable, str(tmp_path / variable))
    overrides = config.Settings()
    for variable in variables:
        assert getattr(overrides, variable.removeprefix("INKRAY_").lower()) == str(tmp_path / variable)


def test_iid_in_the_wild_reports_missing_package_and_runs_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = IidInTheWildInference(device="cpu")
    monkeypatch.setattr(iid_in_the_wild, "find_spec", lambda name: None if name == "intrinsic" else object())
    reason = model.unavailable_reason()
    assert reason is not None
    assert "iid-in-the-wild" in reason

    submodules = {name: torch.nn.Linear(2, 2) for name in ("ord_model", "iid_model", "col_model", "alb_model", "dif_model")}
    hub_list = Mock(return_value=["tf_efficientnet_lite3"])
    monkeypatch.setattr(torch.hub, "list", hub_list)
    load_models = Mock(return_value=submodules)

    def fake_run_pipeline(models, rgb, *, stage, resize_conf, base_size, linear, device):
        assert models is submodules
        assert rgb.dtype == np.float32
        assert stage == 4
        assert resize_conf is None
        assert base_size == 384
        assert linear is False
        assert device == "cpu"
        # Simulate round_32 internal shape differing from 5x7 input.
        return {
            "hr_alb": np.full((32, 32, 3), 0.4, dtype=np.float32),
            "dif_shd": np.full((32, 32, 3), 0.9, dtype=np.float32),
            "residual": np.full((32, 32, 3), 0.05, dtype=np.float32),
        }

    run_pipeline = Mock(side_effect=fake_run_pipeline)
    fake_pkg = ModuleType("intrinsic")
    fake_pipeline = ModuleType("intrinsic.pipeline")
    fake_pipeline.load_models = load_models
    fake_pipeline.run_pipeline = run_pipeline
    monkeypatch.setitem(sys.modules, "intrinsic", fake_pkg)
    monkeypatch.setitem(sys.modules, "intrinsic.pipeline", fake_pipeline)
    monkeypatch.setattr(iid_in_the_wild, "find_spec", lambda _name: object())

    result = model.run(np.full((5, 7, 3), 128, dtype=np.uint8))
    hub_list.assert_called_once_with(
        "rwightman/gen-efficientnet-pytorch",
        trust_repo=True,
        verbose=False,
    )
    load_models.assert_called_once_with("v2", device="cpu")
    assert result.shape == (5, 7)
    assert result.model_id == IidInTheWildInference.model_id
    np.testing.assert_allclose(result.albedo.data, 0.4, atol=1e-5)
    assert result.shading is not None and result.residual is not None
    np.testing.assert_allclose(result.shading.data, 0.9, atol=1e-5)
    np.testing.assert_allclose(result.residual.data, 0.05, atol=1e-5)
