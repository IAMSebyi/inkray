from io import BytesIO
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from app.appearance import AlbedoMap, AppearanceResult, ResidualMap, ShadingMap
from app.config import Settings
from app.geometry.depth import DepthMap
from app.geometry.intrinsics import CameraIntrinsics
from app.geometry.mask import ValidMask
from app.geometry.normals import NormalMap
from app.geometry.points import Points3D
from app.geometry.result import GeometryResult
from app.inference.base import InferenceModel
from app.main import create_app


class TrackingGeometryModel(InferenceModel[np.ndarray, GeometryResult]):
    def __init__(self) -> None:
        self.calls = 0
        self.activate_calls = 0
        self.offload_calls = 0
        self.on_cpu = False

    def activate_device(self) -> None:
        self.activate_calls += 1
        self.on_cpu = False

    def offload_to_cpu(self) -> None:
        self.offload_calls += 1
        self.on_cpu = True

    def run(self, input_data: np.ndarray) -> GeometryResult:
        self.calls += 1
        h, w = input_data.shape[:2]
        depth = np.full((h, w), 2.5, dtype=np.float32)
        normals = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (h, w, 1))
        xs = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) / 0.9
        ys = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) / 0.9
        gx, gy = np.meshgrid(xs, ys)
        points = np.stack([gx * 2.5, gy * 2.5, depth], axis=-1).astype(np.float32)
        intrinsics = np.array([[0.9, 0.0, 0.5], [0.0, 0.9, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)
        return GeometryResult(
            depth=DepthMap(depth),
            normals=NormalMap(normals),
            points=Points3D(points),
            intrinsics=CameraIntrinsics(intrinsics),
            mask=ValidMask(mask),
        )


class StubAppearanceModel(InferenceModel[np.ndarray, AppearanceResult]):
    def __init__(self) -> None:
        self.calls = 0
        self.activate_calls = 0
        self.offload_calls = 0
        self.on_cpu = True

    def activate_device(self) -> None:
        self.activate_calls += 1
        self.on_cpu = False

    def offload_to_cpu(self) -> None:
        self.offload_calls += 1
        self.on_cpu = True

    def run(self, input_data: np.ndarray) -> AppearanceResult:
        self.calls += 1
        h, w = input_data.shape[:2]
        albedo = np.full((h, w, 3), 0.35, dtype=np.float32)
        shading = np.full((h, w, 3), 0.65, dtype=np.float32)
        residual = np.full((h, w, 3), 0.02, dtype=np.float32)
        return AppearanceResult(
            albedo=AlbedoMap(albedo),
            shading=ShadingMap(shading),
            residual=ResidualMap(residual),
            model_id="prs-eth/marigold-iid-lighting-v1-1",
            denoising_steps=4,
            processing_resolution=768,
            ensemble_size=1,
            seed=42,
        )


def _encode_png(width: int, height: int, color: tuple[int, int, int] = (120, 160, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_appearance_infer_caches_and_serves_binary_maps(tmp_path: Path) -> None:
    geom_model = TrackingGeometryModel()
    app_model = StubAppearanceModel()
    settings = Settings(
        cache_dir=tmp_path / "geom_cache",
        appearance_cache_dir=tmp_path / "app_cache",
        max_image_size=8,
        appearance_max_image_size=8,
    )
    app = create_app(settings=settings, inference_model=geom_model, appearance_model=app_model)
    png_bytes = _encode_png(16, 12)

    with TestClient(app) as client:
        resp1 = client.post(
            "/api/appearance/infer",
            files={"file": ("sample.png", png_bytes, "image/png")},
        )
        assert resp1.status_code == 200
        meta1 = resp1.json()
        assert meta1["cached"] is False
        assert meta1["original_width"] == 16
        assert meta1["original_height"] == 12
        assert meta1["appearance_width"] == 8
        assert meta1["appearance_height"] == 6
        assert meta1["prediction_space"] == "linear"
        assert meta1["scale_conventions"]["shading"] == "up_to_scale"
        assert app_model.calls == 1
        assert geom_model.offload_calls == 1
        assert app_model.activate_calls == 1

        # Second identical request hits disk cache cleanly without calling model again.
        resp2 = client.post(
            "/api/appearance/infer",
            files={"file": ("sample.png", png_bytes, "image/png")},
        )
        assert resp2.status_code == 200
        meta2 = resp2.json()
        assert meta2["cached"] is True
        assert meta2["id"] == meta1["id"]
        assert app_model.calls == 1

        appearance_id = meta1["id"]
        meta_resp = client.get(f"/api/appearance/{appearance_id}")
        assert meta_resp.status_code == 200
        assert meta_resp.json()["id"] == appearance_id

        albedo_resp = client.get(f"/api/appearance/{appearance_id}/maps/albedo")
        assert albedo_resp.status_code == 200
        assert albedo_resp.headers["X-Array-Shape"] == "6,8,3"
        assert albedo_resp.headers["X-Array-Dtype"] == "float32"
        albedo_arr = np.frombuffer(albedo_resp.content, dtype="<f4").reshape(6, 8, 3)
        np.testing.assert_allclose(albedo_arr, 0.35)

        shading_resp = client.get(f"/api/appearance/{appearance_id}/maps/shading")
        assert shading_resp.status_code == 200
        shading_arr = np.frombuffer(shading_resp.content, dtype="<f4").reshape(6, 8, 3)
        np.testing.assert_allclose(shading_arr, 0.65)

        bad_kind = client.get(f"/api/appearance/{appearance_id}/maps/specular")
        assert bad_kind.status_code == 400
        missing_rec = client.get("/api/appearance/deadbeef")
        assert missing_rec.status_code == 404

        # Next geometry inference offloads appearance model first.
        geom_resp = client.post(
            "/api/geometry/infer",
            files={"file": ("other.png", _encode_png(16, 12, (200, 120, 90)), "image/png")},
        )
        assert geom_resp.status_code == 200
        assert app_model.offload_calls == 1
        assert geom_model.activate_calls == 1
