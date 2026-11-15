from dataclasses import dataclass

from .depth import DepthMap
from .intrinsics import CameraIntrinsics
from .mask import ValidMask
from .normals import NormalMap
from .points import Points3D


@dataclass(frozen=True)
class GeometryResult:
    depth: DepthMap
    normals: NormalMap | None
    points: Points3D
    intrinsics: CameraIntrinsics
    mask: ValidMask

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape

    @property
    def height(self) -> int:
        return self.depth.height

    @property
    def width(self) -> int:
        return self.depth.width
