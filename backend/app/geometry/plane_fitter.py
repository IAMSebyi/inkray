from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PlaneModel:
    origin: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        origin_arr = np.asarray(self.origin, dtype=np.float64)
        normal_arr = np.asarray(self.normal, dtype=np.float64)
        if origin_arr.shape != (3,) or normal_arr.shape != (3,):
            raise ValueError("PlaneModel requires 3-element origin and normal vectors.")
        object.__setattr__(self, "origin", origin_arr)
        object.__setattr__(self, "normal", normal_arr)


@dataclass(frozen=True)
class PlaneFitResult:
    plane: PlaneModel
    inlier_indices: np.ndarray
    rms_error: float
    max_error: float


def orient_normal_toward_camera(normal: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Orients a plane normal to face the camera at the OpenCV origin."""
    n = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(n))
    if not np.isfinite(length) or length <= 1e-10:
        unit = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        unit = n / length
    to_camera = -np.asarray(origin, dtype=np.float64)
    if float(np.dot(unit, to_camera)) < 0.0:
        unit = -unit
    return unit


def _normalize_normals(normals: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    if normals is None:
        return None
    arr = np.asarray(normals, dtype=np.float64)
    lengths = np.linalg.norm(arr, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-6) & np.all(np.isfinite(arr), axis=1)
    unit = np.zeros_like(arr, dtype=np.float64)
    unit[valid] = arr[valid] / lengths[valid, None]
    return unit, valid


def _fit_from_three_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> PlaneModel | None:
    cross = np.cross(p1 - p0, p2 - p0)
    norm_val = float(np.linalg.norm(cross))
    if not np.isfinite(norm_val) or norm_val <= 1e-10:
        return None
    normal = orient_normal_toward_camera(cross / norm_val, p0)
    return PlaneModel(origin=p0, normal=normal)


class LocalPlaneFitter:
    """Least-squares PCA and deterministic RANSAC local 3D plane fitting."""

    @staticmethod
    def fit_least_squares(
        points: np.ndarray,
        subset_indices: np.ndarray | None = None,
    ) -> PlaneFitResult | None:
        pts_all = np.asarray(points, dtype=np.float64)
        if subset_indices is not None:
            idx = np.asarray(subset_indices, dtype=np.int64)
            pts = pts_all[idx]
        else:
            idx = np.arange(pts_all.shape[0], dtype=np.int64)
            pts = pts_all

        if pts.shape[0] < 3:
            return None

        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Reject rank-deficient / collinear patches where the second axis collapses.
        if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[1]) <= 1e-12:
            return None

        normal = orient_normal_toward_camera(eigenvectors[:, 0], centroid)
        plane = PlaneModel(origin=centroid, normal=normal)
        dists = np.abs(centered @ normal)
        rms_error = float(np.sqrt(np.mean(dists * dists)))
        max_error = float(np.max(dists))

        return PlaneFitResult(
            plane=plane,
            inlier_indices=idx,
            rms_error=rms_error,
            max_error=max_error,
        )

    @staticmethod
    def fit_robust(
        points: np.ndarray,
        *,
        distance_threshold: float | None = None,
        iterations: int = 72,
        seed: int = 42,
        normals: np.ndarray | None = None,
        min_normal_dot: float = 0.55,
    ) -> PlaneFitResult | None:
        pts = np.asarray(points, dtype=np.float64)
        count = pts.shape[0]
        if count < 3:
            return None
        if count < 6:
            return LocalPlaneFitter.fit_least_squares(pts)

        mean_depth = max(1e-4, float(np.mean(pts[:, 2])))
        threshold = (
            float(distance_threshold)
            if distance_threshold is not None
            else max(1e-4, 0.015 * mean_depth)
        )

        norm_bundle = _normalize_normals(normals)
        rng = np.random.default_rng(seed)

        best_inliers: np.ndarray | None = None
        best_score = float("-inf")

        for _ in range(iterations):
            tri = rng.choice(count, size=3, replace=False)
            candidate = _fit_from_three_points(pts[tri[0]], pts[tri[1]], pts[tri[2]])
            if candidate is None:
                continue

            dists = np.abs((pts - candidate.origin) @ candidate.normal)
            inlier_mask = dists <= threshold
            if norm_bundle is not None:
                unit_normals, valid_normals = norm_bundle
                normal_dots = unit_normals @ candidate.normal
                inlier_mask &= (~valid_normals) | (normal_dots >= min_normal_dot)

            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < 3:
                continue

            residual_penalty = float(np.sum(dists[inlier_mask]) / threshold)
            score = float(inlier_count) - 0.1 * residual_penalty
            if score > best_score:
                best_score = score
                best_inliers = np.flatnonzero(inlier_mask)

        if best_inliers is None or best_inliers.size < 3:
            return LocalPlaneFitter.fit_least_squares(pts)

        first_refined = LocalPlaneFitter.fit_least_squares(pts, best_inliers)
        if first_refined is None:
            return LocalPlaneFitter.fit_least_squares(pts)

        dists = np.abs((pts - first_refined.plane.origin) @ first_refined.plane.normal)
        final_mask = dists <= threshold
        if norm_bundle is not None:
            unit_normals, valid_normals = norm_bundle
            normal_dots = unit_normals @ first_refined.plane.normal
            final_mask &= (~valid_normals) | (normal_dots >= min_normal_dot)

        final_indices = np.flatnonzero(final_mask)
        if final_indices.size < 3:
            return first_refined
        return LocalPlaneFitter.fit_least_squares(pts, final_indices) or first_refined
