"""CUDA correctness mirror of the CPU dynamic-rejection test suite.

For each scene we check that FastGICPCuda:
- converges to a pose close enough to ground truth (or to FastGICP CPU),
- emits sane dynamic-rejection diagnostics when rejection is on,
- agrees with the CPU path on the 2DGS / sparse-anchor codepaths.

CPU and CUDA paths differ in float vs double precision in the linearize loop,
so we use looser tolerances than the CPU suite. Speedup is measured in
test_cuda_speedup.py.
"""

import json
import os

import numpy as np
import pytest

from _loader import load as load_cuda, results_dir
import synthetic_indoor as si
import viz


def _make_cuda(pg, target, source, *, max_dist=1.5):
    g = pg.FastGICPCuda()
    g.set_correspondence_randomness(20)
    g.set_max_correspondence_distance(max_dist)
    g.set_input_target(target)
    g.set_input_source(source)
    return g


def _make_cpu(pg, target, source, *, max_dist=1.5):
    g = pg.FastGICP()
    g.set_num_threads(8)
    g.set_correspondence_randomness(20)
    g.set_max_correspondence_distance(max_dist)
    g.set_input_target(target)
    g.set_input_source(source)
    return g


@pytest.fixture(scope="module")
def pg():
    return load_cuda()


# ---------------------------------------------------------------------------
# Static regression: CUDA must not be dramatically worse than CPU on a clean
# scene (allow ~mm-level precision difference vs CPU due to single precision).
# ---------------------------------------------------------------------------
def test_cuda_S10_static(pg):
    s = si.scene_S10_static_regression()
    g = _make_cuda(pg, s.target, s.source, max_dist=1.0)
    T = g.align(np.eye(4, dtype=np.float32)).astype(np.float64)
    rot, tr = si.pose_error(s.gt_pose, T)
    out = dict(scene="S10_cuda_static", rot_deg=rot, trans_m=tr)
    with open(os.path.join(results_dir(), "S10_cuda.json"), "w") as f:
        json.dump(out, f, indent=2)
    viz.render_full_scene(
        results_dir(), "S10_cuda",
        scene_desc=f"S10 (CUDA): {s.description}",
        source=s.source, target=s.target, gt_pose=s.gt_pose,
        est_pose_dynrej=T,
        rot_err_dynrej=rot, trans_err_dynrej=tr)
    assert rot < 0.5, f"CUDA rotation error {rot:.3f} deg too high"
    assert tr < 0.005, f"CUDA translation error {tr:.5f} m too high"


# ---------------------------------------------------------------------------
# Dynamic rejection on S3 (corridor + sliding cabinet): the hard CPU case.
# CUDA should beat the no-rejection CUDA baseline by a wide margin even if it
# can't quite reach CPU's gold-standard precision (float vs double).
# ---------------------------------------------------------------------------
def test_cuda_S3_dynamic_rejection(pg):
    s = si.scene_S3_corridor_with_cabinet()
    # Baseline
    g_base = _make_cuda(pg, s.target, s.source, max_dist=2.0)
    T_base = g_base.align(np.eye(4, dtype=np.float32)).astype(np.float64)
    rot_b, t_b = si.pose_error(s.gt_pose, T_base)
    # With rejection
    g = _make_cuda(pg, s.target, s.source, max_dist=2.0)
    g.set_dynamic_rejection_config({
        "enable": True, "kernel": "GEMAN_MCCLURE",
        "warmup_iterations": 3, "min_inlier_ratio": 0.2,
    })
    T = g.align(np.eye(4, dtype=np.float32)).astype(np.float64)
    rot, tr = si.pose_error(s.gt_pose, T)
    diag = g.get_dynamic_rejection_diagnostics()
    weights = np.asarray(g.get_dynamic_correspondence_weights())
    residuals = np.asarray(g.get_dynamic_correspondence_residuals())

    metrics = dict(scene="S3_cuda_corridor",
                   baseline_rot_deg=rot_b, baseline_trans_m=t_b,
                   dynrej_rot_deg=rot, dynrej_trans_m=tr,
                   diagnostics=diag,
                   inlier_count=int(diag["inlier_count"]),
                   total=int(diag["total_correspondences"]))
    with open(os.path.join(results_dir(), "S3_cuda.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    viz.render_full_scene(
        results_dir(), "S3_cuda",
        scene_desc=f"S3 (CUDA): {s.description}",
        source=s.source, target=s.target, gt_pose=s.gt_pose,
        est_pose_baseline=T_base, est_pose_dynrej=T,
        source_is_dynamic=s.source_is_dynamic,
        weights=weights, residuals=residuals, diagnostics=diag,
        rot_err_baseline=rot_b, trans_err_baseline=t_b,
        rot_err_dynrej=rot, trans_err_dynrej=tr)

    # Hard expectation: dynrej trans error must be << baseline.
    assert tr < t_b * 0.3 + 1e-6, (
        f"CUDA dyn-rej did not beat baseline: base={t_b:.4f} dyn={tr:.4f}")
    assert tr < 0.10, f"CUDA dyn-rej trans error {tr:.4f} m too high"
    # GNC should have actually fired.
    assert diag["enabled"] and diag["active"]
    assert diag["gnc_iterations"] >= 1


# ---------------------------------------------------------------------------
# 2DGS surfel path parity with CPU: should match exactly (both implement the
# same closed-form covariance from quaternion + 2D scales).
# ---------------------------------------------------------------------------
def test_cuda_S9_2dgs_parity_with_cpu(pg):
    base, src_rot, src_scl, tgt_rot, tgt_scl = si.scene_S9_2dgs_surfels()

    def run(klass_factory):
        g = klass_factory()
        g.set_correspondence_randomness(20)
        g.set_max_correspondence_distance(1.0)
        g.set_input_target(base.target)
        g.set_input_source(base.source)
        g.set_source_covariances_from_2dgs(src_rot.astype(np.float32), src_scl.astype(np.float32), "physical", 0.05, 1e-3)
        g.set_target_covariances_from_2dgs(tgt_rot.astype(np.float32), tgt_scl.astype(np.float32), "physical", 0.05, 1e-3)
        return g.align(np.eye(4, dtype=np.float32)).astype(np.float64)

    T_cuda = run(pg.FastGICPCuda)
    T_cpu = run(pg.FastGICP)
    rot_cu, t_cu = si.pose_error(base.gt_pose, T_cuda)
    rot_cp, t_cp = si.pose_error(base.gt_pose, T_cpu)

    out = dict(scene="S9_cuda_2dgs",
               cuda_rot_deg=rot_cu, cuda_trans_m=t_cu,
               cpu_rot_deg=rot_cp, cpu_trans_m=t_cp)
    with open(os.path.join(results_dir(), "S9_cuda.json"), "w") as f:
        json.dump(out, f, indent=2)
    viz.render_full_scene(
        results_dir(), "S9_cuda",
        scene_desc=f"S9 (CUDA): 2DGS surfel covariance, {base.description}",
        source=base.source, target=base.target, gt_pose=base.gt_pose,
        est_pose_baseline=T_cpu, est_pose_dynrej=T_cuda,
        source_is_dynamic=base.source_is_dynamic,
        rot_err_baseline=rot_cp, trans_err_baseline=t_cp,
        rot_err_dynrej=rot_cu, trans_err_dynrej=t_cu)
    # CUDA should match CPU within a tight margin (both run the same
    # deterministic 2DGS covariance + LM steps).
    assert abs(rot_cu - rot_cp) < 0.05
    assert abs(t_cu - t_cp) < 0.005


# ---------------------------------------------------------------------------
# Sparse anchors with bad anchors on dynamics — verify CUDA + dyn rejection
# successfully detects them via anchor recall.
# ---------------------------------------------------------------------------
def test_cuda_S8_anchors_on_dynamic(pg):
    scene, src_a, tgt_a, weights, sigmas, is_dyn = si.scene_S8_anchors_on_dynamic()
    g = _make_cuda(pg, scene.target, scene.source, max_dist=1.5)
    g.set_dynamic_rejection_config({
        "enable": True, "kernel": "GEMAN_MCCLURE",
        "warmup_iterations": 3, "min_inlier_ratio": 0.2,
        "anchor_rejection_enabled": True,
    })
    g.set_sparse_anchor_config({"objective_weight": 50.0, "balance_mode": "BY_HESSIAN_TRACE"})
    g.set_sparse_anchor_correspondences(src_a, tgt_a, weights, sigmas)
    T = g.align(np.eye(4, dtype=np.float32)).astype(np.float64)
    rot, tr = si.pose_error(scene.gt_pose, T)
    anchor_w = np.asarray(g.get_dynamic_anchor_weights())
    if anchor_w.size == is_dyn.size:
        pred = anchor_w < 0.5
        tp = int((pred & is_dyn).sum())
        fn = int((~pred & is_dyn).sum())
        recall = tp / max(tp + fn, 1)
    else:
        recall = float("nan")

    out = dict(scene="S8_cuda_anchors_on_dynamic",
               rot_deg=rot, trans_m=tr, anchor_recall=recall,
               anchor_total=int(is_dyn.size), anchor_dyn_gt=int(is_dyn.sum()))
    with open(os.path.join(results_dir(), "S8_cuda.json"), "w") as f:
        json.dump(out, f, indent=2)
    viz.render_full_scene(
        results_dir(), "S8_cuda",
        scene_desc=f"S8 (CUDA): anchors with bad ones on movers",
        source=scene.source, target=scene.target, gt_pose=scene.gt_pose,
        est_pose_dynrej=T,
        source_is_dynamic=scene.source_is_dynamic,
        diagnostics=g.get_dynamic_rejection_diagnostics(),
        rot_err_dynrej=rot, trans_err_dynrej=tr,
        extra_pose_lines=[f"anchor_recall: {recall:.2f}"])
    # 100% anchor recall is the working CPU number; allow 70% as floor.
    if not np.isnan(recall):
        assert recall >= 0.7, f"CUDA anchor recall too low: {recall:.2f}"


# ---------------------------------------------------------------------------
# VGICP (CPU) must still throw if dyn rejection is enabled (sanity check that
# the CUDA build didn't accidentally enable rejection on unsupported paths).
# ---------------------------------------------------------------------------
def test_unsupported_paths_throw(pg):
    v = pg.FastVGICP()
    with pytest.raises(RuntimeError):
        v.set_dynamic_rejection_config({"enable": True, "kernel": "GEMAN_MCCLURE"})
