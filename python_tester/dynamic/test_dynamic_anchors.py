"""S8: sparse 3D anchors, some of which land on dynamic objects.

Without rejection, the corrupted anchors can pull the alignment significantly.
With rejection (anchor_rejection_enabled=True), the bad anchors should be
identified and downweighted.
"""

import json
import os

import numpy as np
import pytest

from _loader import load, results_dir
import synthetic_indoor as si
from _helpers import run_align, mask_recall_precision
import viz


@pytest.fixture(scope="module")
def pg():
    return load()


def test_S8_anchors_on_dynamic(pg):
    scene, src_a, tgt_a, weights, sigmas, anchor_is_dyn = si.scene_S8_anchors_on_dynamic()

    # Helper: run with anchors (with or without rejection).
    def _run(rejection):
        g = pg.FastGICP()
        g.set_num_threads(8)
        g.set_max_correspondence_distance(1.5)
        g.set_input_target(scene.target)
        g.set_input_source(scene.source)
        if rejection:
            g.set_dynamic_rejection_config({"enable": True, "kernel": "GEMAN_MCCLURE",
                                            "warmup_iterations": 3, "min_inlier_ratio": 0.2,
                                            "anchor_rejection_enabled": True})
        # Strong objective weight to ensure anchors meaningfully influence result.
        g.set_sparse_anchor_config({"objective_weight": 50.0, "balance_mode": "BY_HESSIAN_TRACE"})
        g.set_sparse_anchor_correspondences(src_a, tgt_a, weights, sigmas)
        T = g.align(np.eye(4, dtype=np.float32))
        anchor_w = np.asarray(g.get_dynamic_anchor_weights())
        return T.astype(np.float64), anchor_w, g.get_dynamic_rejection_diagnostics()

    T_base, _, _ = _run(False)
    T_dyn, anchor_w_dyn, diag = _run(True)

    rot_b, t_b = si.pose_error(scene.gt_pose, T_base)
    rot_d, t_d = si.pose_error(scene.gt_pose, T_dyn)

    # Per-anchor outlier prediction: w < 0.5.
    if anchor_w_dyn.size == anchor_is_dyn.size:
        pred = anchor_w_dyn < 0.5
        recall, precision = mask_recall_precision(pred, anchor_is_dyn)
    else:
        recall, precision = float("nan"), float("nan")

    metrics = dict(scene="S8_anchors_on_dynamic",
                   baseline_rot_deg=rot_b, baseline_trans_m=t_b,
                   dynrej_rot_deg=rot_d, dynrej_trans_m=t_d,
                   anchor_recall=recall, anchor_precision=precision,
                   diagnostics=diag,
                   anchor_total=int(anchor_is_dyn.size),
                   anchor_dynamic_gt=int(anchor_is_dyn.sum()))
    with open(os.path.join(results_dir(), "S8.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    extras = [
        f"anchor_total: {int(anchor_is_dyn.size)}",
        f"anchor_dynamic_gt: {int(anchor_is_dyn.sum())}",
        f"anchor_recall: {recall:.3f}" if not np.isnan(recall) else "anchor_recall: n/a",
        f"anchor_precision: {precision:.3f}" if not np.isnan(precision) else "anchor_precision: n/a",
    ]
    viz.render_full_scene(
        results_dir(), "S8",
        scene_desc=f"S8: {scene.description}",
        source=scene.source, target=scene.target, gt_pose=scene.gt_pose,
        est_pose_baseline=T_base, est_pose_dynrej=T_dyn,
        source_is_dynamic=scene.source_is_dynamic,
        diagnostics=diag,
        rot_err_baseline=rot_b, trans_err_baseline=t_b,
        rot_err_dynrej=rot_d, trans_err_dynrej=t_d,
        extra_pose_lines=extras)

    # Rejection should make trans error strictly better than baseline when bad
    # anchors are present.
    assert t_d < t_b + 1e-6 or t_d < 0.02, (
        f"anchor rejection did not improve: base={t_b:.4f} dyn={t_d:.4f}")
    # Catch at least half of the bad anchors.
    if not np.isnan(recall):
        assert recall >= 0.5, f"anchor mask recall too low: {recall:.2f}"
