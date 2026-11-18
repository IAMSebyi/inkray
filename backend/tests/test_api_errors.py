from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from app.appearance.maps import AlbedoMap, ResidualMap, ShadingMap
from app.appearance.result import AppearanceResult
from app.config import Settings
from app.geometry.depth import DepthMap
from app.geometry.intrinsics import CameraIntrinsics
from app.geometry.mask import ValidMask
from app.geometry.normals import NormalMap
from app.geometry.points import Points3D
from app.geometry.result import GeometryResult
from app.inference import moge
from app.inference.depth_pro import DepthProInference
from app.inference.marigold import MarigoldNormalsInference
from app.inference.marigold_v2 import MarigoldV2Inference, MarigoldV2NormalsInference, MarigoldV2Runner
from app.main import create_app


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), (100, 120, 140)).save(output, format="PNG")
    return output.getvalue()


def test_macos_defaults_to_depth_pro_without_importing_moge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: tuple[Mock, Mock],
) -> None:
    monkeypatch.setattr(moge, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(moge, "find_spec", Mock(side_effect=AssertionError("MoGe discovery on macOS")))
    monkeypatch.setattr(DepthProInference, "activate_device", lambda self: None)
    run_depth = models[0].run
    monkeypatch.setattr(DepthProInference, "run", run_depth)

    settings = Settings(
        device="cpu", cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance",
    )
    with TestClient(create_app(settings)) as client:
        capabilities = client.get("/api/models").json()
        assert capabilities["default_geometry"] == "depth-pro"
        assert capabilities["default_normal"] == "marigold-normals"
        assert "Linux or Windows" in capabilities["unavailable"]["moge-3"]
        assert "marigold-v2" in capabilities["unavailable"]
        assert "marigold-normals" not in capabilities["unavailable"]
        assert "marigold-iid" not in capabilities["unavailable"]
        files = {"file": ("image.png", png_bytes(), "image/png")}
        response = client.post("/api/geometry/infer", files=files)
        assert response.status_code == 200
        assert response.json()["model"] == "depth-pro"
        response = client.post("/api/geometry/infer?model=moge-3", files=files)
        assert response.status_code == 503
        assert "Linux or Windows" in response.json()["detail"]
    run_depth.assert_called_once()


@pytest.fixture
def models() -> tuple[Mock, Mock]:
    geometry = Mock(spec=["run"])
    geometry.run.return_value = GeometryResult(
        depth=DepthMap(np.ones((4, 4), dtype=np.float32)),
        normals=None,
        points=Points3D(np.ones((4, 4, 3), dtype=np.float32)),
        intrinsics=CameraIntrinsics(np.eye(3, dtype=np.float32)),
        mask=ValidMask(np.ones((4, 4), dtype=bool)),
    )
    appearance = Mock(spec=["run"])
    appearance.run.return_value = AppearanceResult(
        albedo=AlbedoMap(np.full((4, 4, 3), 0.5, dtype=np.float32)),
        shading=ShadingMap(np.ones((4, 4, 3), dtype=np.float32)),
        residual=ResidualMap(np.zeros((4, 4, 3), dtype=np.float32)),
        model_id="test",
        denoising_steps=1,
        processing_resolution=4,
        ensemble_size=1,
        seed=42,
    )
    return geometry, appearance


@pytest.mark.parametrize("kind", ["geometry", "appearance"])
@pytest.mark.parametrize("damage", ["missing", "metadata", "array_metadata", "archive", "member"])
def test_inference_rebuilds_damaged_cache(
    tmp_path: Path,
    models: tuple[Mock, Mock],
    kind: str,
    damage: str,
) -> None:
    settings = Settings(cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance")
    with TestClient(create_app(settings, *models)) as client:
        files = {"file": ("image.png", png_bytes(), "image/png")}
        first = client.post(f"/api/{kind}/infer", files=files)
        assert first.status_code == 200
        cache_id = first.json()["id"]
        directory = settings.cache_dir if kind == "geometry" else settings.appearance_cache_dir
        if damage == "missing":
            (directory / f"{cache_id}.npz").unlink()
        elif damage == "metadata":
            (directory / f"{cache_id}.json").write_text("{invalid", encoding="utf-8")
        elif damage == "array_metadata":
            path = directory / f"{cache_id}.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            del metadata["arrays"]["depth" if kind == "geometry" else "albedo"]
            path.write_text(json.dumps(metadata), encoding="utf-8")
        elif damage == "member":
            path = directory / f"{cache_id}.npz"
            missing = "intrinsics" if kind == "geometry" else "albedo"
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files if name != missing}
            np.savez_compressed(path, **arrays)
        else:
            (directory / f"{cache_id}.npz").write_bytes(b"invalid archive")
        rebuilt = client.post(f"/api/{kind}/infer", files=files)
        assert rebuilt.status_code == 200
        assert rebuilt.json()["cached"] is False
        assert models[0 if kind == "geometry" else 1].run.call_count == 2
        map_kind = "depth" if kind == "geometry" else "albedo"
        assert client.get(f"/api/{kind}/{cache_id}/maps/{map_kind}").status_code == 200


@pytest.mark.parametrize("kind", ["geometry", "appearance"])
@pytest.mark.parametrize("limit", ["bytes", "pixels", "pillow"])
def test_rejects_oversized_images_before_inference(
    tmp_path: Path,
    models: tuple[Mock, Mock],
    kind: str,
    limit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        cache_dir=tmp_path / "geometry",
        appearance_cache_dir=tmp_path / "appearance",
        max_upload_bytes=10 if limit == "bytes" else 1024,
        max_image_pixels=10 if limit == "pixels" else 64,
    )
    if limit == "pillow":
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)
    with TestClient(create_app(settings, *models)) as client:
        response = client.post(
            f"/api/{kind}/infer",
            files={"file": ("image.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 413
    for model in models:
        model.run.assert_not_called()


@pytest.mark.parametrize("kind", ["geometry", "appearance"])
def test_model_io_failure_is_a_server_error(
    tmp_path: Path,
    models: tuple[Mock, Mock],
    kind: str,
) -> None:
    models[0 if kind == "geometry" else 1].run.side_effect = OSError("Model unavailable")
    settings = Settings(cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance")
    with TestClient(create_app(settings, *models), raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/{kind}/infer",
            files={"file": ("image.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 500


@pytest.mark.parametrize("kind", ["geometry", "appearance"])
def test_pixel_limit_applies_to_cached_images(
    tmp_path: Path,
    models: tuple[Mock, Mock],
    kind: str,
) -> None:
    settings = Settings(cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance")
    files = {"file": ("image.png", png_bytes(), "image/png")}
    with TestClient(create_app(settings, *models)) as client:
        assert client.post(f"/api/{kind}/infer", files=files).status_code == 200
    with TestClient(create_app(replace(settings, max_image_pixels=10), *models)) as client:
        response = client.post(f"/api/{kind}/infer", files=files)
    assert response.status_code == 413
    assert models[0 if kind == "geometry" else 1].run.call_count == 1


def test_albedo_only_model_has_its_own_cache_entry(
    tmp_path: Path,
    models: tuple[Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infer_albedo = Mock(return_value=AppearanceResult.from_albedo(
        np.full((4, 4, 3), 0.25, dtype=np.float32), MarigoldV2Inference.model_id, 4,
    ))
    monkeypatch.setattr(MarigoldV2Inference, "run", infer_albedo)
    settings = Settings(cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance")
    files = {"file": ("image.png", png_bytes(), "image/png")}
    with TestClient(create_app(settings, *models)) as client:
        default = client.post("/api/appearance/infer", files=files).json()
        url = "/api/appearance/infer?model=marigold-v2"
        response = client.post(url, files=files)
        assert response.status_code == 200
        alternative = response.json()
        assert alternative["id"] != default["id"]
        assert alternative["model_id"] == MarigoldV2Inference.model_id
        assert set(alternative["arrays"]) == {"albedo"}
        assert client.post(url, files=files).json()["cached"] is True
        maps = f"/api/appearance/{alternative['id']}/maps"
        assert client.get(f"{maps}/albedo").status_code == 200
        assert client.get(f"{maps}/shading").status_code == 404
        assert client.get(f"{maps}/residual").status_code == 404
    assert infer_albedo.call_count == 1
    assert models[1].run.call_count == 1


def test_unconfigured_model_returns_actionable_error(
    tmp_path: Path,
    models: tuple[Mock, Mock],
) -> None:
    settings = Settings(
        cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance",
        latent_intrinsics_repo="",
    )
    with TestClient(create_app(settings, *models)) as client:
        response = client.post(
            "/api/appearance/infer?model=latent-intrinsics",
            files={"file": ("image.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 503
    assert "backend/.cache/models/latent-intrinsics" in response.json()["detail"]
    models[1].run.assert_not_called()


def test_normal_refinement_reuses_base_geometry_and_has_its_own_cache_entry(
    tmp_path: Path,
    models: tuple[Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guide_normals = NormalMap(
        np.broadcast_to(np.array([0.0, 0.0, -1.0], dtype=np.float32), (4, 4, 3))
    )
    run_v11 = Mock(return_value=guide_normals)
    monkeypatch.setattr(MarigoldNormalsInference, "run", run_v11)
    predict_v2 = Mock(return_value=np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 4, 3)))
    monkeypatch.setattr(MarigoldV2Runner, "predict", predict_v2)

    settings = Settings(cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance")
    files = {"file": ("image.png", png_bytes(), "image/png")}
    with TestClient(create_app(settings, *models)) as client:
        base = client.post("/api/geometry/infer", files=files).json()

        url_v11 = "/api/geometry/infer?refine_normals=true"
        resp_v11 = client.post(url_v11, files=files)
        assert resp_v11.status_code == 200
        refined_v11 = resp_v11.json()
        assert refined_v11["id"] != base["id"]
        assert refined_v11["normal_model_id"] == MarigoldNormalsInference.model_id
        assert client.post(url_v11, files=files).json()["cached"] is True

        url_v2 = "/api/geometry/infer?refine_normals=true&normal_model=marigold-v2"
        resp_v2 = client.post(url_v2, files=files)
        assert resp_v2.status_code == 200
        refined_v2 = resp_v2.json()
        assert refined_v2["id"] not in {base["id"], refined_v11["id"]}
        assert refined_v2["normal_model_id"] == MarigoldV2NormalsInference.model_id
        assert client.post(url_v2, files=files).json()["cached"] is True

        bad = client.post("/api/geometry/infer?refine_normals=true&normal_model=unknown", files=files)
        assert bad.status_code == 400

    assert models[0].run.call_count == 1
    run_v11.assert_called_once()
    predict_v2.assert_called_once()
    assert predict_v2.call_args.args[1] == "normals"


def test_geometry_smoothing_reuses_cached_base_and_validates_iterations(
    tmp_path: Path,
    models: tuple[Mock, Mock],
) -> None:
    settings = Settings(cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance")
    files = {"file": ("image.png", png_bytes(), "image/png")}
    with TestClient(create_app(settings, *models)) as client:
        base = client.post("/api/geometry/infer", files=files).json()
        smooth_url = "/api/geometry/infer?smooth_iterations=3"
        resp = client.post(smooth_url, files=files)
        assert resp.status_code == 200
        smoothed = resp.json()
        assert smoothed["id"] != base["id"]
        assert smoothed["smooth_iterations"] == 3
        assert client.post(smooth_url, files=files).json()["cached"] is True

        assert client.post("/api/geometry/infer?smooth_iterations=11", files=files).status_code == 400
        assert client.post("/api/geometry/infer?smooth_iterations=-1", files=files).status_code == 400

    assert models[0].run.call_count == 1


def test_iid_in_the_wild_reports_actionable_setup_when_uninstalled(
    tmp_path: Path,
    models: tuple[Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.inference import iid_in_the_wild

    monkeypatch.setattr(iid_in_the_wild, "find_spec", lambda name: None if name == "intrinsic" else object())
    settings = Settings(cache_dir=tmp_path / "geometry", appearance_cache_dir=tmp_path / "appearance")
    with TestClient(create_app(settings, *models)) as client:
        caps = client.get("/api/models").json()
        assert "iid-in-the-wild" in caps["unavailable"]
        resp = client.post(
            "/api/appearance/infer?model=iid-in-the-wild",
            files={"file": ("image.png", png_bytes(), "image/png")},
        )
    assert resp.status_code == 503
    assert "iid-in-the-wild" in resp.json()["detail"]
    models[1].run.assert_not_called()
