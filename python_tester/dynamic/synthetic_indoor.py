"""Synthetic indoor scenes for dynamic-GICP testing.

All geometry is generated procedurally with numpy (no external assets), with a
fixed seed for reproducibility. Each scene factory returns a Scene object with:

  source            Nx3 float64    source frame point cloud
  target            Mx3 float64    target frame point cloud
  gt_pose           4x4 float64    rigid pose T such that target ~= T @ source
                                   for static points (T transforms source -> target)
  source_is_dynamic N   bool       True for source points that belong to a
                                   moving object (those are NOT consistent with
                                   gt_pose); False for static structure.
  description       str
"""

from __future__ import annotations

import dataclasses
import numpy as np


# ---------- primitives ------------------------------------------------------


def _grid_face(u_axis, v_axis, normal_axis, normal_value, u_range, v_range, step):
    """Sample a rectangular face (uniform grid) and return Nx3 points."""
    uu = np.arange(u_range[0], u_range[1] + 1e-9, step)
    vv = np.arange(v_range[0], v_range[1] + 1e-9, step)
    U, V = np.meshgrid(uu, vv, indexing="xy")
    pts = np.zeros((U.size, 3), dtype=np.float64)
    pts[:, u_axis] = U.ravel()
    pts[:, v_axis] = V.ravel()
    pts[:, normal_axis] = normal_value
    return pts


def make_room(width=4.0, depth=4.0, height=2.5, step=0.06):
    """Six-sided rectangular room. Origin is center on floor; +z up."""
    half_w, half_d = width / 2, depth / 2
    floor = _grid_face(0, 1, 2, 0.0, (-half_w, half_w), (-half_d, half_d), step)
    ceil = _grid_face(0, 1, 2, height, (-half_w, half_w), (-half_d, half_d), step)
    wall_x_neg = _grid_face(1, 2, 0, -half_w, (-half_d, half_d), (0, height), step)
    wall_x_pos = _grid_face(1, 2, 0, half_w, (-half_d, half_d), (0, height), step)
    wall_y_neg = _grid_face(0, 2, 1, -half_d, (-half_w, half_w), (0, height), step)
    wall_y_pos = _grid_face(0, 2, 1, half_d, (-half_w, half_w), (0, height), step)
    return np.vstack([floor, ceil, wall_x_neg, wall_x_pos, wall_y_neg, wall_y_pos])


def make_box(center, size, step=0.05):
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    faces = [
        _grid_face(1, 2, 0, cx - hx, (cy - hy, cy + hy), (cz - hz, cz + hz), step),
        _grid_face(1, 2, 0, cx + hx, (cy - hy, cy + hy), (cz - hz, cz + hz), step),
        _grid_face(0, 2, 1, cy - hy, (cx - hx, cx + hx), (cz - hz, cz + hz), step),
        _grid_face(0, 2, 1, cy + hy, (cx - hx, cx + hx), (cz - hz, cz + hz), step),
        _grid_face(0, 1, 2, cz - hz, (cx - hx, cx + hx), (cy - hy, cy + hy), step),
        _grid_face(0, 1, 2, cz + hz, (cx - hx, cx + hx), (cy - hy, cy + hy), step),
    ]
    return np.vstack(faces)


def make_cylinder(center, radius=0.25, height=1.7, n_circ=40, n_h=30):
    """Vertical cylinder side surface (no caps)."""
    cx, cy, cz = center
    theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    zs = np.linspace(0, height, n_h)
    TH, Z = np.meshgrid(theta, zs, indexing="xy")
    x = cx + radius * np.cos(TH).ravel()
    y = cy + radius * np.sin(TH).ravel()
    z = cz + Z.ravel()
    return np.stack([x, y, z], axis=-1)


def make_corridor(length=10.0, width=1.4, height=2.5, step=0.06, door_period=2.0,
                  door_width=0.9):
    """Long corridor with periodic door openings on the +y wall."""
    half_w = width / 2
    floor = _grid_face(0, 1, 2, 0.0, (-length / 2, length / 2), (-half_w, half_w), step)
    ceil = _grid_face(0, 1, 2, height, (-length / 2, length / 2), (-half_w, half_w), step)
    wall_y_neg = _grid_face(0, 2, 1, -half_w, (-length / 2, length / 2), (0, height), step)
    # +y wall with door openings:
    pts = []
    xs = np.arange(-length / 2, length / 2 + 1e-9, step)
    zs = np.arange(0, height + 1e-9, step)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    # door mask: every door_period along x, opening of width door_width centered
    in_door = np.zeros_like(XX, dtype=bool)
    for door_center in np.arange(-length / 2 + door_period / 2, length / 2,
                                 door_period):
        in_door |= (np.abs(XX - door_center) < door_width / 2) & (ZZ < 2.0)
    keep = ~in_door
    wall_y_pos = np.stack([XX[keep], np.full(keep.sum(), half_w), ZZ[keep]], axis=-1)
    return np.vstack([floor, ceil, wall_y_neg, wall_y_pos])


# ---------- transforms ------------------------------------------------------


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(3)
    R[:2, :2] = np.array([[c, -s], [s, c]])
    return R


def rotation_about(axis, theta):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def make_pose(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def apply_pose(T, pts):
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def pose_error(T_gt, T_est):
    """Return (rotation_deg, translation_m) error between two 4x4 poses.

    Both represent source->target rigid transforms.
    """
    R_err = T_gt[:3, :3].T @ T_est[:3, :3]
    cos_arg = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    rot_deg = np.degrees(np.arccos(cos_arg))
    t_err = np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3])
    return rot_deg, t_err


# ---------- Scene container -------------------------------------------------


@dataclasses.dataclass
class Scene:
    source: np.ndarray             # Nx3 source frame
    target: np.ndarray             # Mx3 target frame
    gt_pose: np.ndarray            # 4x4 source->target
    source_is_dynamic: np.ndarray  # N bool
    description: str = ""


def add_noise(pts, sigma=0.005, seed=0):
    rng = np.random.default_rng(seed)
    return pts + rng.normal(0.0, sigma, size=pts.shape)


def downsample(pts, max_count, seed=0):
    if pts.shape[0] <= max_count:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_count, replace=False)
    return pts[idx]


# ---------- scene factories -------------------------------------------------


def _move_subset(pts, mask, T_obj):
    """Apply object-local rigid T to the masked subset; return new point cloud."""
    out = pts.copy()
    if mask.any():
        out[mask] = apply_pose(T_obj, pts[mask])
    return out


def scene_S1_single_mover(seed=1):
    """Room + a large furniture-like box that moves between source and target.

    ~15% of source points are dynamic (on the box). The room itself moves with
    gt_pose (camera/sensor motion). The box has additional independent motion.
    """
    rng = np.random.default_rng(seed)
    room = make_room(step=0.07)
    box = make_box((0.6, 0.4, 0.4), (0.8, 0.6, 0.8), step=0.04)
    n_room = room.shape[0]
    n_box = box.shape[0]
    source = np.vstack([room, box])
    source_is_dynamic = np.zeros(source.shape[0], dtype=bool)
    source_is_dynamic[n_room:] = True

    # Sensor motion (room moves rigidly with this).
    T_gt = make_pose(rotation_z(np.radians(8.0)), np.array([0.10, -0.05, 0.02]))

    # Object motion in target frame (extra rigid offset, defined in target coordinates).
    T_obj_in_target = make_pose(rotation_z(np.radians(20.0)), np.array([0.3, -0.2, 0.0]))

    # Build target: static points get gt_pose; dynamic points get gt_pose THEN object offset.
    target_static = apply_pose(T_gt, room)
    target_dynamic = apply_pose(T_obj_in_target, apply_pose(T_gt, box))
    target = np.vstack([target_static, target_dynamic])

    source = add_noise(downsample(source, 8000, seed), sigma=0.004, seed=seed + 1)
    # Recompute dynamic mask after downsampling: we lost the per-index map, so
    # re-downsample with the index trail tracked.
    rng2 = np.random.default_rng(seed)
    all_pts = np.vstack([room, box])
    all_dyn = np.zeros(all_pts.shape[0], dtype=bool)
    all_dyn[n_room:] = True
    if all_pts.shape[0] > 8000:
        idx = rng2.choice(all_pts.shape[0], size=8000, replace=False)
        source = add_noise(all_pts[idx], sigma=0.004, seed=seed + 1)
        source_is_dynamic = all_dyn[idx]
    else:
        source = add_noise(all_pts, sigma=0.004, seed=seed + 1)
        source_is_dynamic = all_dyn

    target = add_noise(downsample(target, 8000, seed + 2), sigma=0.004,
                       seed=seed + 3)

    return Scene(source, target, T_gt, source_is_dynamic,
                 description="S1: room + single large mover (~15% dynamic)")


def scene_S2_multi_movers(seed=2):
    """Room + 4 vertical cylinders (pedestrians) that all move differently."""
    room = make_room(step=0.07)
    cyls = []
    centers = [(0.5, 0.5, 0.0), (-1.2, 0.8, 0.0), (1.0, -1.3, 0.0), (-0.5, -0.5, 0.0)]
    for c in centers:
        cyls.append(make_cylinder(c, radius=0.18, height=1.6, n_circ=24, n_h=18))
    cyl_blocks = cyls

    source_parts = [room]
    source_parts.extend(cyl_blocks)
    source = np.vstack(source_parts)
    src_dyn = np.zeros(source.shape[0], dtype=bool)
    offset = room.shape[0]
    for b in cyl_blocks:
        src_dyn[offset:offset + b.shape[0]] = True
        offset += b.shape[0]

    T_gt = make_pose(rotation_z(np.radians(-5.0)), np.array([-0.08, 0.12, 0.0]))

    # Per-cyl motions.
    rng = np.random.default_rng(seed)
    cyl_motions = [
        make_pose(rotation_z(np.radians(rng.uniform(-30, 30))),
                  rng.uniform(-0.4, 0.4, size=3) * np.array([1, 1, 0]))
        for _ in cyl_blocks
    ]

    target_room = apply_pose(T_gt, room)
    target_cyls = []
    for b, m in zip(cyl_blocks, cyl_motions):
        target_cyls.append(apply_pose(m, apply_pose(T_gt, b)))
    target = np.vstack([target_room] + target_cyls)

    rng2 = np.random.default_rng(seed + 1)
    if source.shape[0] > 8000:
        idx = rng2.choice(source.shape[0], size=8000, replace=False)
        source = add_noise(source[idx], sigma=0.004, seed=seed + 2)
        src_dyn = src_dyn[idx]
    else:
        source = add_noise(source, sigma=0.004, seed=seed + 2)
    target = add_noise(downsample(target, 8000, seed + 3), sigma=0.004,
                       seed=seed + 4)
    return Scene(source, target, T_gt, src_dyn,
                 description="S2: room + 4 moving cylinder 'people'")


def scene_S3_corridor_with_cabinet(seed=3):
    """Corridor + a cabinet box that slides along x-axis between frames."""
    corridor = make_corridor(length=8.0, width=1.4, step=0.06)
    cabinet = make_box((-1.0, 0.0, 0.4), (0.6, 0.5, 0.8), step=0.04)
    n_corr = corridor.shape[0]
    source = np.vstack([corridor, cabinet])
    src_dyn = np.zeros(source.shape[0], dtype=bool)
    src_dyn[n_corr:] = True

    # Small sensor motion mostly along corridor.
    T_gt = make_pose(rotation_z(np.radians(2.0)), np.array([0.15, 0.0, 0.0]))
    T_cab = make_pose(np.eye(3), np.array([0.6, 0.0, 0.0]))  # cabinet slid along x

    target = np.vstack([apply_pose(T_gt, corridor),
                        apply_pose(T_cab, apply_pose(T_gt, cabinet))])

    rng = np.random.default_rng(seed)
    if source.shape[0] > 9000:
        idx = rng.choice(source.shape[0], size=9000, replace=False)
        source = add_noise(source[idx], sigma=0.004, seed=seed)
        src_dyn = src_dyn[idx]
    else:
        source = add_noise(source, sigma=0.004, seed=seed)
    target = add_noise(downsample(target, 9000, seed + 1), sigma=0.004,
                       seed=seed + 2)
    return Scene(source, target, T_gt, src_dyn,
                 description="S3: corridor + sliding cabinet")


def scene_S4_mirror_room(seed=4):
    """Symmetric room with one distinctive box to break mirror ambiguity."""
    room = make_room(width=4.0, depth=4.0, height=2.5, step=0.07)
    landmark = make_box((1.3, 0.0, 0.3), (0.3, 0.3, 0.6), step=0.04)
    source = np.vstack([room, landmark])
    src_dyn = np.zeros(source.shape[0], dtype=bool)

    T_gt = make_pose(rotation_z(np.radians(3.0)), np.array([0.05, 0.0, 0.0]))
    target = apply_pose(T_gt, source)

    rng = np.random.default_rng(seed)
    if source.shape[0] > 8000:
        idx = rng.choice(source.shape[0], size=8000, replace=False)
        source = add_noise(source[idx], sigma=0.004, seed=seed)
        src_dyn = src_dyn[idx]
    else:
        source = add_noise(source, sigma=0.004, seed=seed)
    target = add_noise(downsample(target, 8000, seed + 1), sigma=0.004,
                       seed=seed + 2)
    return Scene(source, target, T_gt, src_dyn,
                 description="S4: symmetric room with landmark (basin test)")


def scene_S5_planar_floor(seed=5):
    """Planar floor only — degenerate observability scene; no dynamic objects."""
    floor = _grid_face(0, 1, 2, 0.0, (-2.0, 2.0), (-2.0, 2.0), 0.05)
    source = floor
    src_dyn = np.zeros(source.shape[0], dtype=bool)
    T_gt = make_pose(np.eye(3), np.array([0.05, 0.04, 0.0]))
    target = apply_pose(T_gt, source)
    source = add_noise(downsample(source, 6000, seed), sigma=0.003, seed=seed)
    target = add_noise(downsample(target, 6000, seed + 1), sigma=0.003,
                       seed=seed + 2)
    return Scene(source, target, T_gt, src_dyn[:source.shape[0]],
                 description="S5: degenerate planar floor")


def scene_S6_big_init_offset(seed=6):
    """Furnished room; static. Returns a perturbed initial guess (caller uses it)."""
    room = make_room(step=0.07)
    box = make_box((0.5, 0.6, 0.3), (0.6, 0.4, 0.6), step=0.04)
    source = np.vstack([room, box])
    src_dyn = np.zeros(source.shape[0], dtype=bool)

    # Substantial sensor motion.
    T_gt = make_pose(rotation_z(np.radians(15.0)), np.array([0.20, -0.10, 0.0]))
    target = apply_pose(T_gt, source)

    rng = np.random.default_rng(seed)
    if source.shape[0] > 8000:
        idx = rng.choice(source.shape[0], size=8000, replace=False)
        source = add_noise(source[idx], sigma=0.004, seed=seed)
        src_dyn = src_dyn[idx]
    else:
        source = add_noise(source, sigma=0.004, seed=seed)
    target = add_noise(downsample(target, 8000, seed + 1), sigma=0.004,
                       seed=seed + 2)
    return Scene(source, target, T_gt, src_dyn,
                 description="S6: room+box, no dynamic; basin sweep harness")


def scene_S7_combined_worst_case(seed=7):
    """S1 dynamic + S6 large pose magnitude (caller will additionally perturb init)."""
    rng = np.random.default_rng(seed)
    room = make_room(step=0.07)
    box = make_box((0.6, 0.4, 0.4), (0.8, 0.6, 0.8), step=0.04)
    n_room = room.shape[0]
    source = np.vstack([room, box])
    src_dyn = np.zeros(source.shape[0], dtype=bool)
    src_dyn[n_room:] = True

    T_gt = make_pose(rotation_z(np.radians(18.0)), np.array([0.25, -0.15, 0.02]))
    T_obj = make_pose(rotation_z(np.radians(30.0)), np.array([0.4, -0.3, 0.0]))

    target = np.vstack([apply_pose(T_gt, room),
                        apply_pose(T_obj, apply_pose(T_gt, box))])

    rng2 = np.random.default_rng(seed + 1)
    if source.shape[0] > 8000:
        idx = rng2.choice(source.shape[0], size=8000, replace=False)
        source = add_noise(source[idx], sigma=0.004, seed=seed + 2)
        src_dyn = src_dyn[idx]
    else:
        source = add_noise(source, sigma=0.004, seed=seed + 2)
    target = add_noise(downsample(target, 8000, seed + 3), sigma=0.004,
                       seed=seed + 4)
    return Scene(source, target, T_gt, src_dyn,
                 description="S7: dynamic + large pose (worst case)")


def scene_S8_anchors_on_dynamic(seed=8, frac_dynamic_anchors=0.3, n_anchors=40):
    """S1 base + sparse anchors. A fraction lands on the moving box."""
    base = scene_S1_single_mover(seed=seed)
    # Sample anchors from source: split by dynamic mask.
    rng = np.random.default_rng(seed)
    static_idx = np.where(~base.source_is_dynamic)[0]
    dyn_idx = np.where(base.source_is_dynamic)[0]
    n_dyn = max(1, int(round(n_anchors * frac_dynamic_anchors)))
    n_stat = n_anchors - n_dyn
    src_anchor_idx = np.concatenate([
        rng.choice(static_idx, size=min(n_stat, static_idx.size), replace=False),
        rng.choice(dyn_idx, size=min(n_dyn, dyn_idx.size), replace=False),
    ])
    rng.shuffle(src_anchor_idx)

    src_anchors = base.source[src_anchor_idx]
    # For static anchors, target is GT-transformed source (no noise added).
    # For dynamic anchors, target is some wrong place (the dynamic-displaced point).
    is_dyn = base.source_is_dynamic[src_anchor_idx]
    tgt_anchors = apply_pose(base.gt_pose, src_anchors)  # default static target

    # For dynamic ones, replace with the "wrong" target (where the dynamic point
    # actually appears in the target frame).
    # We need the target-frame position of each dynamic source point. The scene
    # didn't expose per-point object motion; for testing purposes we synthesize
    # by sampling a nearby point in the actual target dynamic region.
    # Simpler: fabricate a residual by perturbing the static target position
    # with a known dynamic displacement, large enough to be an outlier.
    rng2 = np.random.default_rng(seed + 11)
    perturb = rng2.normal(0.0, 0.4, size=(int(is_dyn.sum()), 3))
    tgt_anchors[is_dyn] += perturb

    weights = np.ones(src_anchor_idx.shape[0], dtype=np.float64)
    sigmas = np.full(src_anchor_idx.shape[0], 0.05, dtype=np.float64)

    return base, src_anchors.astype(np.float64), tgt_anchors.astype(np.float64), \
        weights, sigmas, is_dyn


def scene_S9_2dgs_surfels(seed=9):
    """S1 base but expose 2DGS surfel covariances (rotation quaternions + 2D scales).

    Source points are treated as surfels with isotropic small in-plane scale and
    a thin normal direction. We attach unit quaternion rotations (identity) and
    2D scales ~ noise-level, so the GICP path with 2DGS covariances applies.
    """
    base = scene_S1_single_mover(seed=seed)
    # Identity rotation (xyzw) for every surfel; 2D scales.
    n = base.source.shape[0]
    rotations_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)).ravel()
    scales_2d = np.tile(np.array([0.04, 0.04], dtype=np.float32), (n, 1)).ravel()
    m = base.target.shape[0]
    tgt_rotations_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (m, 1)).ravel()
    tgt_scales_2d = np.tile(np.array([0.04, 0.04], dtype=np.float32), (m, 1)).ravel()
    return base, rotations_xyzw, scales_2d, tgt_rotations_xyzw, tgt_scales_2d


def scene_S10_static_regression(seed=10):
    """Static room+box, no dynamic — to test that enabling rejection on a clean
    static scene does not degrade alignment vs baseline."""
    room = make_room(step=0.07)
    box = make_box((0.5, 0.6, 0.3), (0.6, 0.4, 0.6), step=0.04)
    source = np.vstack([room, box])
    src_dyn = np.zeros(source.shape[0], dtype=bool)
    T_gt = make_pose(rotation_z(np.radians(4.0)), np.array([0.06, 0.03, 0.0]))
    target = apply_pose(T_gt, source)
    rng = np.random.default_rng(seed)
    if source.shape[0] > 8000:
        idx = rng.choice(source.shape[0], size=8000, replace=False)
        source = add_noise(source[idx], sigma=0.004, seed=seed)
        src_dyn = src_dyn[idx]
    else:
        source = add_noise(source, sigma=0.004, seed=seed)
    target = add_noise(downsample(target, 8000, seed + 1), sigma=0.004,
                       seed=seed + 2)
    return Scene(source, target, T_gt, src_dyn,
                 description="S10: static room+box regression")
