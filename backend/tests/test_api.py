from io import BytesIO
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from app.config import Settings
from app.geometry.depth import DepthMap
from app.geometry.intrinsics import CameraIntrinsics
from app.geometry.mask import ValidMask
from app.geometry.normals import NormalMap
from app.geometry.points import Points3D
from app.geometry.result import GeometryResult
from app.inference.base import InferenceModel
from app.main import create_app


class StubGeometryModel(InferenceModel[np.ndarray, GeometryResult]):
    def __init__(self) -> None:
        self.calls = 0

    def run(self, input_data: np.ndarray) -> GeometryResult:
        self.calls += 1
        h, w = input_data.shape[:2]
        depth = np.full((h, w), 2.5, dtype=np.float32)
        normals = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (h, w, 1))
        points = np.zeros((h, w, 3), dtype=np.float32)
        points[..., 2] = 2.5
        intrinsics = np.array([[0.9, 0.0, 0.5], [0.0, 0.9, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        mask = np.ones((h, w), dtype=bool)
        mask[0, 0] = False
        return GeometryResult(
            depth=DepthMap(depth),
            normals=NormalMap(normals),
            points=Points3D(points),
            intrinsics=CameraIntrinsics(intrinsics),
            mask=ValidMask(mask),
        )


def _encode_png(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 160, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_infer_caches_result_and_serves_binary_maps(tmp_path: Path) -> None:
    model = StubGeometryModel()
    settings = Settings(cache_dir=tmp_path / "cache", max_image_size=8)
    app = create_app(settings=settings, inference_model=model)
    png_bytes = _encode_png(16, 12)

    with TestClient(app) as client:
        resp1 = client.post(
            "/api/geometry/infer",
            files={"file": ("sample.png", png_bytes, "image/png")},
        )
        assert resp1.status_code == 200
        meta1 = resp1.json()
        assert meta1["cached"] is False
        assert meta1["original_width"] == 16
        assert meta1["original_height"] == 12
        assert meta1["inference_width"] == 8
        assert meta1["inference_height"] == 6
        assert meta1["coordinate_system"] == "opencv"
        assert model.calls == 1

        # Second upload of identical image hits cache without calling model again.
        resp2 = client.post(
            "/api/geometry/infer",
            files={"file": ("sample.png", png_bytes, "image/png")},
        )
        assert resp2.status_code == 200
        meta2 = resp2.json()
        assert meta2["cached"] is True
        assert meta2["id"] == meta1["id"]
        assert model.calls == 1

        geometry_id = meta1["id"]
        meta_resp = client.get(f"/api/geometry/{geometry_id}")
        assert meta_resp.status_code == 200
        assert meta_resp.json()["id"] == geometry_id

        depth_resp = client.get(f"/api/geometry/{geometry_id}/maps/depth")
        assert depth_resp.status_code == 200
        assert depth_resp.headers["X-Array-Shape"] == "6,8"
        assert depth_resp.headers["X-Array-Dtype"] == "float32"
        depth_arr = np.frombuffer(depth_resp.content, dtype="<f4").reshape(6, 8)
        np.testing.assert_allclose(depth_arr, 2.5)

        mask_resp = client.get(f"/api/geometry/{geometry_id}/maps/mask")
        assert mask_resp.status_code == 200
        assert mask_resp.headers["X-Array-Dtype"] == "uint8"
        mask_arr = np.frombuffer(mask_resp.content, dtype=np.uint8).reshape(6, 8)
        assert mask_arr[0, 0] == 0
        assert mask_arr[1, 1] == 1


def test_invalid_upload_and_missing_records(tmp_path: Path) -> None:
    model = StubGeometryModel()
    settings = Settings(cache_dir=tmp_path / "cache")
    app = create_app(settings=settings, inference_model=model)

    with TestClient(app) as client:
        bad = client.post(
            "/api/geometry/infer",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert bad.status_code == 400
        assert model.calls == 0

        missing_meta = client.get("/api/geometry/deadbeef")
        assert missing_meta.status_code == 404

        missing_map = client.get("/api/geometry/deadbeef/maps/depth")
        assert missing_map.status_code == 404

        bad_kind = client.get("/api/geometry/deadbeef/maps/albedo")
        assert bad_kind.status_code == 400
