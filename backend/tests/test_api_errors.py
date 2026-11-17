from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
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
from app.geometry.points import Points3D
from app.geometry.result import GeometryResult
from app.main import create_app


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), (100, 120, 140)).save(output, format="PNG")
    return output.getvalue()


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
