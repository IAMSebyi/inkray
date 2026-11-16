from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from .plane_fitter import LocalPlaneFitter, PlaneModel, orient_normal_toward_camera
from .result import GeometryResult


@dataclass(frozen=True)
class PlaneFootprint:
    min_u: float
    max_u: float
    min_v: float
    max_v: float


@dataclass(frozen=True)
class PlaneFrameSpec:
    origin: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray


@dataclass(frozen=True)
class SurfaceSelectionResult:
    seed_image_point: tuple[float, float]
    seed_inference_pixel: tuple[int, int]
    seed_point_3d: np.ndarray
    plane: PlaneModel
    frame: PlaneFrameSpec
    support_mask: np.ndarray
    inlier_count: int
    rms_error: float
    suggested_footprint: PlaneFootprint
    reference_brush_diameter: float


@dataclass(frozen=True)
class SurfaceSelectionOutcome:
    ok: bool
    selection: SurfaceSelectionResult | None = None
    reason: str | None = None


def _normalize_vec3(vec: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vec))
    if not np.isfinite(length) or length <= 1e-10:
        return None
    return vec / length


def _build_plane_frame(plane: PlaneModel, anchor_point: np.ndarray) -> PlaneFrameSpec:
    """Builds an orthonormal camera-facing frame with +u screen-right and +v screen-down."""
    unit_normal = orient_normal_toward_camera(plane.normal, plane.origin)
    anchor = np.asarray(anchor_point, dtype=np.float64)
    signed_dist = float(np.dot(unit_normal, anchor - plane.origin))
    frame_origin = anchor - unit_normal * signed_dist

    cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    proj_right = cam_right - unit_normal * float(np.dot(cam_right, unit_normal))
    u_axis = _normalize_vec3(proj_right)
    if u_axis is None:
        cam_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        proj_fwd = cam_forward - unit_normal * float(np.dot(cam_forward, unit_normal))
        u_axis = _normalize_vec3(proj_fwd)
        if u_axis is None:
            u_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    v_axis = _normalize_vec3(np.cross(u_axis, unit_normal))
    if v_axis is None:
        v_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    if float(v_axis[1]) < 0.0 and abs(float(unit_normal[1])) < 0.999:
        v_axis = -v_axis
        flipped_u = _normalize_vec3(np.cross(unit_normal, v_axis))
        if flipped_u is not None:
            u_axis = flipped_u

    return PlaneFrameSpec(
        origin=frame_origin,
        normal=unit_normal,
        u_axis=u_axis,
        v_axis=v_axis,
    )


def _grow_connected_grid(
    node_mask: np.ndarray,
    depth_grid: np.ndarray,
    seed_row: int,
    seed_col: int,
    jump_ratio: float,
    max_nodes: int,
) -> np.ndarray:
    """Finds 4-connected BFS nodes from seed respecting per-edge relative depth jumps."""
    rows_n, cols_n = node_mask.shape
    total = rows_n * cols_n
    seed_idx = seed_row * cols_n + seed_col

    active = node_mask.copy()
    active[seed_row, seed_col] = True
    safe_depth = np.where(active, depth_grid, 0.0)
    idx_grid = np.arange(total, dtype=np.int32).reshape(rows_n, cols_n)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []

    if cols_n > 1:
        h_ok = (
            active[:, :-1]
            & active[:, 1:]
            & (
                np.abs(safe_depth[:, :-1] - safe_depth[:, 1:])
                <= jump_ratio * np.minimum(safe_depth[:, :-1], safe_depth[:, 1:])
            )
        )
        row_parts.append(idx_grid[:, :-1][h_ok])
        col_parts.append(idx_grid[:, 1:][h_ok])

    if rows_n > 1:
        v_ok = (
            active[:-1, :]
            & active[1:, :]
            & (
                np.abs(safe_depth[:-1, :] - safe_depth[1:, :])
                <= jump_ratio * np.minimum(safe_depth[:-1, :], safe_depth[1:, :])
            )
        )
        row_parts.append(idx_grid[:-1, :][v_ok])
        col_parts.append(idx_grid[1:, :][v_ok])

    if not row_parts:
        return np.array([seed_idx], dtype=np.int32)

    edge_u = np.concatenate(row_parts)
    edge_v = np.concatenate(col_parts)
    if edge_u.size == 0:
        return np.array([seed_idx], dtype=np.int32)

    weights = np.ones(edge_u.size, dtype=np.uint8)
    graph = csr_matrix((weights, (edge_u, edge_v)), shape=(total, total))
    order = breadth_first_order(graph, i_start=seed_idx, directed=False, return_predecessors=False)
    if order.size > max_nodes:
        order = order[:max_nodes]
    return np.asarray(order, dtype=np.int32)


def _compute_candidate_mask(
    valid_grid: np.ndarray,
    points: np.ndarray,
    unit_normals: np.ndarray | None,
    valid_normals: np.ndarray | None,
    plane: PlaneModel,
    plane_residual_tol: float,
    min_normal_dot: float,
) -> np.ndarray:
    candidates = valid_grid.copy()
    valid_pts = points[valid_grid].astype(np.float64)
    dists = np.abs((valid_pts - plane.origin) @ plane.normal)
    keep = dists <= plane_residual_tol

    if unit_normals is not None and valid_normals is not None:
        sub_unit = unit_normals[valid_grid]
        sub_has_n = valid_normals[valid_grid]
        normal_ok = (~sub_has_n) | ((sub_unit @ plane.normal) >= min_normal_dot)
        keep &= normal_ok

    candidates[valid_grid] = keep
    return candidates


def _compute_footprint_from_support(
    support_points: np.ndarray,
    frame: PlaneFrameSpec,
    seed_depth: float,
    pixel_angle: float,
) -> PlaneFootprint:
    deltas = support_points.astype(np.float64) - frame.origin
    u_vals = np.sort(deltas @ frame.u_axis)
    v_vals = np.sort(deltas @ frame.v_axis)
    count = u_vals.size

    # Trim extreme 1% tails so sparse boundary outliers do not bloat the decal footprint.
    trim_low = math.floor((count - 1) * 0.01) if count >= 40 else 0
    trim_high = math.ceil((count - 1) * 0.99) if count >= 40 else count - 1

    raw_min_u = min(0.0, float(u_vals[trim_low]))
    raw_max_u = max(0.0, float(u_vals[trim_high]))
    raw_min_v = min(0.0, float(v_vals[trim_low]))
    raw_max_v = max(0.0, float(v_vals[trim_high]))

    min_half_span = max(1e-4, 4.0 * seed_depth * pixel_angle)
    span_u = max(min_half_span * 2.0, raw_max_u - raw_min_u)
    span_v = max(min_half_span * 2.0, raw_max_v - raw_min_v)
    pad_u = max(min_half_span, span_u * 0.05)
    pad_v = max(min_half_span, span_v * 0.05)

    return PlaneFootprint(
        min_u=raw_min_u - pad_u,
        max_u=raw_max_u + pad_u,
        min_v=raw_min_v - pad_v,
        max_v=raw_max_v + pad_v,
    )


class SurfaceSelector:
    """Selects a locally planar connected support region from MoGe-3 geometry."""

    @staticmethod
    def select_at_image_point(
        geometry: GeometryResult,
        seed_image_point: tuple[float, float],
        original_size: tuple[int, int],
        *,
        max_region_pixels: int | None = None,
        neighbor_depth_jump_ratio: float | None = None,
        min_normal_dot: float = 0.55,
        ransac_seed: int = 42,
    ) -> SurfaceSelectionOutcome:
        orig_w, orig_h = original_size
        inf_w, inf_h = geometry.width, geometry.height
        sx, sy = seed_image_point

        if (
            orig_w <= 0
            or orig_h <= 0
            or inf_w <= 0
            or inf_h <= 0
            or not math.isfinite(sx)
            or not math.isfinite(sy)
            or sx < 0.0
            or sx >= orig_w
            or sy < 0.0
            or sy >= orig_h
        ):
            return SurfaceSelectionOutcome(
                ok=False,
                reason="Clicked position is outside the image bounds.",
            )

        seed_col = min(inf_w - 1, max(0, math.floor((sx * inf_w) / orig_w)))
        seed_row = min(inf_h - 1, max(0, math.floor((sy * inf_h) / orig_h)))

        depth = geometry.depth.data
        points = geometry.points.data
        valid_grid = (
            geometry.mask.data
            & np.isfinite(depth)
            & (depth > 0.0)
            & np.all(np.isfinite(points), axis=2)
            & (points[:, :, 2] > 0.0)
        )

        if not valid_grid[seed_row, seed_col]:
            return SurfaceSelectionOutcome(
                ok=False,
                reason="Selected pixel has invalid or missing depth geometry. Try clicking inside the surface.",
            )

        seed_point = points[seed_row, seed_col].astype(np.float64)
        seed_depth = float(seed_point[2])

        intr = geometry.intrinsics
        step_x = 1.0 / (intr.fx_norm * inf_w)
        step_y = 1.0 / (intr.fy_norm * inf_h)
        pixel_angle = max(step_x, step_y)

        region_cap = (
            max_region_pixels if max_region_pixels is not None else min(inf_w * inf_h, 250_000)
        )
        jump_ratio = (
            neighbor_depth_jump_ratio
            if neighbor_depth_jump_ratio is not None
            else max(0.08, min(0.35, 8.0 * pixel_angle))
        )
        seed_radius = min(20, max(2, round(min(inf_w, inf_h) * 0.08)))

        unit_normals: np.ndarray | None = None
        valid_normals: np.ndarray | None = None
        if geometry.normals is not None:
            n_arr = geometry.normals.data.astype(np.float64)
            n_len = np.linalg.norm(n_arr, axis=2)
            valid_normals = np.isfinite(n_len) & (n_len > 1e-6) & np.all(np.isfinite(n_arr), axis=2)
            unit_normals = np.zeros_like(n_arr, dtype=np.float64)
            unit_normals[valid_normals] = n_arr[valid_normals] / n_len[valid_normals, None]

        r0 = max(0, seed_row - seed_radius)
        r1 = min(inf_h, seed_row + seed_radius + 1)
        c0 = max(0, seed_col - seed_radius)
        c1 = min(inf_w, seed_col + seed_radius + 1)

        patch_depth = depth[r0:r1, c0:c1]
        patch_nodes = valid_grid[r0:r1, c0:c1] & (
            np.abs(patch_depth - seed_depth) <= 0.45 * seed_depth
        )
        patch_order = _grow_connected_grid(
            patch_nodes,
            patch_depth,
            seed_row - r0,
            seed_col - c0,
            jump_ratio,
            patch_nodes.size,
        )

        if patch_order.size < 6:
            return SurfaceSelectionOutcome(
                ok=False,
                reason="Not enough connected valid geometry points around the selected pixel to fit a plane.",
            )

        patch_w = c1 - c0
        pr = patch_order // patch_w + r0
        pc = patch_order % patch_w + c0
        patch_points = points[pr, pc]
        patch_normals = geometry.normals.data[pr, pc] if geometry.normals is not None else None

        initial_fit = LocalPlaneFitter.fit_robust(
            patch_points,
            distance_threshold=max(1e-4, 0.02 * seed_depth),
            normals=patch_normals,
            min_normal_dot=min_normal_dot,
            seed=ransac_seed,
        )
        if initial_fit is None:
            return SurfaceSelectionOutcome(
                ok=False,
                reason="Could not fit a stable plane at this location (points may be degenerate).",
            )

        active_plane = initial_fit.plane
        active_rms = initial_fit.rms_error

        stage1_tol = max(0.025 * seed_depth, 3.0 * initial_fit.rms_error)
        stage1_mask = _compute_candidate_mask(
            valid_grid, points, unit_normals, valid_normals, active_plane, stage1_tol, min_normal_dot
        )
        final_indices = _grow_connected_grid(
            stage1_mask, depth, seed_row, seed_col, jump_ratio, region_cap
        )

        flat_points = points.reshape(-1, 3)
        if final_indices.size >= 6:
            refined = LocalPlaneFitter.fit_least_squares(flat_points[final_indices])
            if refined is not None:
                active_plane = refined.plane
                active_rms = refined.rms_error

                refined_tol = max(0.025 * seed_depth, 2.5 * refined.rms_error)
                stage2_mask = _compute_candidate_mask(
                    valid_grid,
                    points,
                    unit_normals,
                    valid_normals,
                    active_plane,
                    refined_tol,
                    min_normal_dot,
                )
                stage2_indices = _grow_connected_grid(
                    stage2_mask, depth, seed_row, seed_col, jump_ratio, region_cap
                )
                if stage2_indices.size >= 6:
                    final_refit = LocalPlaneFitter.fit_least_squares(flat_points[stage2_indices])
                    if final_refit is not None:
                        active_plane = final_refit.plane
                        active_rms = final_refit.rms_error
                    final_indices = stage2_indices

        if final_indices.size < 6:
            return SurfaceSelectionOutcome(
                ok=False,
                reason="Selected planar support region is too small for reliable painting.",
            )

        support_mask = np.zeros((inf_h, inf_w), dtype=bool)
        support_mask.ravel()[final_indices] = True

        frame = _build_plane_frame(active_plane, seed_point)
        suggested_footprint = _compute_footprint_from_support(
            flat_points[final_indices], frame, seed_depth, pixel_angle
        )

        footprint_span = min(
            suggested_footprint.max_u - suggested_footprint.min_u,
            suggested_footprint.max_v - suggested_footprint.min_v,
        )
        reference_brush_diameter = max(
            seed_depth * pixel_angle * 2.0,
            min(seed_depth * 0.08, footprint_span * 0.12),
        )

        return SurfaceSelectionOutcome(
            ok=True,
            selection=SurfaceSelectionResult(
                seed_image_point=(sx, sy),
                seed_inference_pixel=(seed_col, seed_row),
                seed_point_3d=seed_point,
                plane=active_plane,
                frame=frame,
                support_mask=support_mask,
                inlier_count=int(final_indices.size),
                rms_error=active_rms,
                suggested_footprint=suggested_footprint,
                reference_brush_diameter=reference_brush_diameter,
            ),
        )
