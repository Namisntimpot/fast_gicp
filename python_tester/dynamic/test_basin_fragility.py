"""S4, S6, S7: basin fragility tests.

S4: mirror-symmetric room with one landmark — convergence basin around the
    correct alignment.
S6: substantial sensor motion + identity initialisation — sweeps initial-guess
    offsets and measures success rate.
S7: dynamic outliers + large initial offset combined.
"""

import json
import os

import numpy as np
import pytest

from _loader import load, results_dir
import synthetic_indoor as si
from _helpers import run_align, perturb_pose
import viz


@pytest.fixture(scope="module")
def pg():
    return load()


def _success(scene, T_est, rot_tol_deg=2.0, t_tol_m=0.05):
    rot, tr = si.pose_error(scene.gt_pose, T_est)
    return rot < rot_tol_deg and tr < t_tol_m, rot, tr


def _basin_sweep(pg, scene, perturb_rot_deg_list, perturb_trans_m_list,
                 rejection_config=None, max_dist=2.0, seeds=4):
    """Sweep init perturbations; return success rate AND a (rot, trans) grid."""
    n_success = 0
    n_total = 0
    sample_errs = []
    grid = np.zeros((len(perturb_rot_deg_list), len(perturb_trans_m_list)),
                    dtype=np.float64)
    counts = np.zeros_like(grid)
    for ri, theta in enumerate(perturb_rot_deg_list):
        for ti, tm in enumerate(perturb_trans_m_list):
            for seed in range(seeds):
                T_init = perturb_pose(scene.gt_pose, theta, tm,
                                      axis=(0, 0, 1), seed=seed).astype(np.float32)
                T_est, _, _ = run_align(pg, scene, rejection_config=rejection_config,
                                        max_dist=max_dist, initial_guess=T_init)
                ok, rot, tr = _success(scene, T_est)
                n_total += 1
                counts[ri, ti] += 1
                if ok:
                    n_success += 1
                    grid[ri, ti] += 1
                sample_errs.append((theta, tm, seed, rot, tr, bool(ok)))
    with np.errstate(invalid="ignore"):
        grid = np.where(counts > 0, grid / np.maximum(counts, 1), 0.0)
    return n_success / max(n_total, 1), sample_errs, grid


def test_S4_mirror_room_landmark(pg):
    """The landmark must keep the convergence basin around gt."""
    s = si.scene_S4_mirror_room()
    rot_grid = [0.0, 5.0, 15.0]
    trans_grid = [0.0, 0.05, 0.1]
    rate_base, _, gb = _basin_sweep(pg, s,
                                    perturb_rot_deg_list=rot_grid,
                                    perturb_trans_m_list=trans_grid,
                                    rejection_config=None, max_dist=1.0, seeds=2)
    rate_dyn, _, gd = _basin_sweep(pg, s,
                                   perturb_rot_deg_list=rot_grid,
                                   perturb_trans_m_list=trans_grid,
                                   rejection_config={"enable": True,
                                                     "kernel": "GEMAN_MCCLURE",
                                                     "warmup_iterations": 3,
                                                     "min_inlier_ratio": 0.4},
                                   max_dist=1.0, seeds=2)
    with open(os.path.join(results_dir(), "S4.json"), "w") as f:
        json.dump({"scene": "S4_mirror_room",
                   "baseline_success_rate": rate_base,
                   "dynrej_success_rate": rate_dyn}, f, indent=2)
    viz.render_basin_heatmap(
        os.path.join(results_dir(), "S4_basin.png"),
        scene_desc=f"S4: {s.description}",
        perturb_rot_deg=rot_grid, perturb_trans_m=trans_grid,
        success_grid_baseline=gb, success_grid_dynrej=gd)
    # Dynamic rejection on a static scene should not catastrophically hurt the
    # basin. Allow a generous tolerance.
    assert rate_dyn >= rate_base - 0.15, (rate_base, rate_dyn)


def test_S6_basin_sweep(pg):
    """Substantial sensor motion; measure success rate vs init offset for both
    plain GICP and GICP+rejection (no actual dynamics here)."""
    s = si.scene_S6_big_init_offset()
    rot_grid = [0.0, 10.0, 25.0, 40.0]
    trans_grid = [0.0, 0.1, 0.4, 0.8]
    sweep = dict(perturb_rot_deg_list=rot_grid, perturb_trans_m_list=trans_grid,
                 max_dist=2.0, seeds=2)
    rate_base, samples_base, gb = _basin_sweep(pg, s, rejection_config=None, **sweep)
    rate_dyn, samples_dyn, gd = _basin_sweep(
        pg, s,
        rejection_config={"enable": True, "kernel": "GEMAN_MCCLURE",
                          "warmup_iterations": 4, "min_inlier_ratio": 0.4},
        **sweep)
    out = dict(scene="S6_basin_sweep",
               baseline_success_rate=rate_base,
               dynrej_success_rate=rate_dyn,
               samples=samples_base[:8],
               samples_dyn=samples_dyn[:8])
    with open(os.path.join(results_dir(), "S6.json"), "w") as f:
        json.dump(out, f, indent=2)
    viz.render_basin_heatmap(
        os.path.join(results_dir(), "S6_basin.png"),
        scene_desc=f"S6: {s.description}",
        perturb_rot_deg=rot_grid, perturb_trans_m=trans_grid,
        success_grid_baseline=gb, success_grid_dynrej=gd)

    # On clean scenes, dynamic rejection must not appreciably shrink the basin.
    assert rate_dyn >= rate_base - 0.20, (
        f"basin shrinkage too large: base={rate_base:.2f} dyn={rate_dyn:.2f}")


def test_S7_worst_case_dynamic_and_bad_init(pg):
    """S7 has dynamic outliers and substantial sensor motion. Use multi-restart
    + rejection to recover. Compare against baseline GICP at identity init.
    """
    s = si.scene_S7_combined_worst_case()
    # Multiple init candidates: identity, gt-perturbed, opposite perturbation.
    rng = np.random.default_rng(7)
    cand_T = [np.eye(4, dtype=np.float32),
              perturb_pose(np.eye(4), 8.0, 0.10, seed=1).astype(np.float32),
              perturb_pose(np.eye(4), -8.0, 0.10, seed=2).astype(np.float32)]

    # Baseline: best of multi-restart with plain GICP (no rejection).
    best_T_base = None
    best_err = float("inf")
    for T_init in cand_T:
        T, _, _ = run_align(pg, s, max_dist=2.0, initial_guess=T_init)
        rot, tr = si.pose_error(s.gt_pose, T)
        err = rot + tr * 50.0
        if err < best_err:
            best_err = err
            best_T_base = T
    rot_b, t_b = si.pose_error(s.gt_pose, best_T_base)

    # Dynamic rejection with internal multi-restart.
    g = pg.FastGICP()
    g.set_num_threads(8)
    g.set_max_correspondence_distance(2.0)
    g.set_input_target(s.target)
    g.set_input_source(s.source)
    g.set_dynamic_rejection_config({"enable": True, "kernel": "GEMAN_MCCLURE",
                                    "warmup_iterations": 4, "min_inlier_ratio": 0.2})
    g.set_multi_restart_initial_guesses([T.astype(np.float32) for T in cand_T])
    T_dyn = g.align(np.eye(4, dtype=np.float32)).astype(np.float64)
    rot_d, t_d = si.pose_error(s.gt_pose, T_dyn)

    with open(os.path.join(results_dir(), "S7.json"), "w") as f:
        json.dump({"scene": "S7_worst_case",
                   "baseline_best_rot_deg": rot_b, "baseline_best_trans_m": t_b,
                   "dynrej_rot_deg": rot_d, "dynrej_trans_m": t_d}, f, indent=2)
    viz.render_full_scene(
        results_dir(), "S7",
        scene_desc=f"S7: {s.description}",
        source=s.source, target=s.target, gt_pose=s.gt_pose,
        est_pose_baseline=best_T_base, est_pose_dynrej=T_dyn,
        source_is_dynamic=s.source_is_dynamic,
        weights=np.asarray(g.get_dynamic_correspondence_weights()),
        residuals=np.asarray(g.get_dynamic_correspondence_residuals()),
        diagnostics=g.get_dynamic_rejection_diagnostics(),
        rot_err_baseline=rot_b, trans_err_baseline=t_b,
        rot_err_dynrej=rot_d, trans_err_dynrej=t_d)

    # Hard expectation: rejection should beat or match baseline.
    # In this challenging scene baseline may already be reasonable due to room
    # dominance; we require dynrej to be no worse and ideally better.
    assert rot_d <= rot_b + 0.5
    assert t_d <= t_b + 0.02
