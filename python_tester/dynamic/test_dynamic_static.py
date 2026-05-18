"""S10 + S5: enabling dynamic rejection must not noticeably degrade static
alignment, including in the degenerate planar case."""

import json
import os

import numpy as np
import pytest

from _loader import load, results_dir
import synthetic_indoor as si
from _helpers import run_align
import viz


@pytest.fixture(scope="module")
def pg():
    return load()


def test_S10_static_room_regression(pg):
    s = si.scene_S10_static_regression()
    T_base, _, _ = run_align(pg, s)
    T_dyn, diag, g_dyn = run_align(
        pg, s,
        rejection_config={"enable": True, "kernel": "GEMAN_MCCLURE",
                          "warmup_iterations": 3, "min_inlier_ratio": 0.3},
        return_diag=True)

    rot_b, t_b = si.pose_error(s.gt_pose, T_base)
    rot_d, t_d = si.pose_error(s.gt_pose, T_dyn)
    out = dict(scene="S10_static_regression",
               baseline_rot_deg=rot_b, baseline_trans_m=t_b,
               dynrej_rot_deg=rot_d, dynrej_trans_m=t_d,
               diagnostics=diag)
    with open(os.path.join(results_dir(), "S10.json"), "w") as f:
        json.dump(out, f, indent=2)
    viz.render_full_scene(
        results_dir(), "S10",
        scene_desc=f"S10: {s.description}",
        source=s.source, target=s.target, gt_pose=s.gt_pose,
        est_pose_baseline=T_base, est_pose_dynrej=T_dyn,
        source_is_dynamic=s.source_is_dynamic,
        weights=np.asarray(g_dyn.get_dynamic_correspondence_weights()),
        residuals=np.asarray(g_dyn.get_dynamic_correspondence_residuals()),
        diagnostics=diag,
        rot_err_baseline=rot_b, trans_err_baseline=t_b,
        rot_err_dynrej=rot_d, trans_err_dynrej=t_d)

    # Allow at most 1° rotation degradation and 1 cm translation degradation.
    assert rot_d <= rot_b + 1.0, f"rotation degraded by {rot_d - rot_b:.3f} deg"
    assert t_d <= t_b + 0.01, f"translation degraded by {t_d - t_b:.4f} m"
    # And absolute correctness floor:
    assert rot_d < 1.0, f"dynrej rot {rot_d:.3f}"
    assert t_d < 0.02, f"dynrej trans {t_d:.4f}"


def test_S5_planar_floor_safety(pg):
    """On a degenerate planar floor, rejection must not blow up. We don't
    require a tight pose match (geometry doesn't constrain z-translation in
    direction tangent to the plane fully), but the optimizer must still
    converge and not erroneously reject the entire floor."""
    s = si.scene_S5_planar_floor()
    T_dyn, diag, g = run_align(
        pg, s,
        rejection_config={"enable": True, "kernel": "GEMAN_MCCLURE",
                          "warmup_iterations": 3, "min_inlier_ratio": 0.5},
        return_diag=True)
    out = dict(scene="S5_planar_floor", diagnostics=diag,
               gt_pose=s.gt_pose.tolist(), est_pose=T_dyn.tolist())
    with open(os.path.join(results_dir(), "S5.json"), "w") as f:
        json.dump(out, f, indent=2)
    viz.render_full_scene(
        results_dir(), "S5",
        scene_desc=f"S5: {s.description}",
        source=s.source, target=s.target, gt_pose=s.gt_pose,
        est_pose_dynrej=T_dyn,
        source_is_dynamic=s.source_is_dynamic,
        weights=np.asarray(g.get_dynamic_correspondence_weights()),
        residuals=np.asarray(g.get_dynamic_correspondence_residuals()),
        diagnostics=diag)

    # The optimizer must not have rejected everything: at least 50% must remain.
    weights = np.asarray(g.get_dynamic_correspondence_weights())
    assert weights.size > 0
    inlier_frac = float(np.mean(weights >= 0.5))
    assert inlier_frac >= 0.5, f"too aggressive: only {inlier_frac:.2f} kept on planar floor"
