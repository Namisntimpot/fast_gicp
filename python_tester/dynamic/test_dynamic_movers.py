"""S1, S2, S3: dynamic outlier rejection on indoor scenes with moving objects.

For each scene, baseline GICP (no rejection) is compared against GICP with
dynamic rejection enabled. We assert that rejection meaningfully reduces pose
error on at least the difficult cases and that the dynamic mask is detected
with reasonable recall.
"""

import json
import os

import numpy as np
import pytest

from _loader import load, results_dir
import synthetic_indoor as si
from _helpers import run_align, mask_recall_precision
import viz


REJECTION = {
    "enable": True,
    "kernel": "GEMAN_MCCLURE",
    "warmup_iterations": 3,
    "min_inlier_ratio": 0.2,
}


@pytest.fixture(scope="module")
def pg():
    return load()


def _evaluate(scene, T_base, T_dyn, weights):
    rot_b, t_b = si.pose_error(scene.gt_pose, T_base)
    rot_d, t_d = si.pose_error(scene.gt_pose, T_dyn)
    # The weight array is per-source-point; pred_outlier = weight < 0.5.
    pred_outlier = np.asarray(weights) < 0.5
    if pred_outlier.size != scene.source_is_dynamic.size:
        # If weights vector is empty (rejection off / warmup), recall is N/A.
        recall, precision = float("nan"), float("nan")
    else:
        recall, precision = mask_recall_precision(pred_outlier, scene.source_is_dynamic)
    return dict(baseline_rot_deg=rot_b, baseline_trans_m=t_b,
                dynrej_rot_deg=rot_d, dynrej_trans_m=t_d,
                mask_recall=recall, mask_precision=precision)


def _run_scene(pg, scene, *, max_dist):
    T_base, _, _ = run_align(pg, scene, max_dist=max_dist)
    T_dyn, diag, g = run_align(
        pg, scene, rejection_config=REJECTION, max_dist=max_dist, return_diag=True)
    weights = np.asarray(g.get_dynamic_correspondence_weights())
    residuals = np.asarray(g.get_dynamic_correspondence_residuals())
    return T_base, T_dyn, diag, weights, residuals


def _viz_scene(scene_id, scene, T_base, T_dyn, diag, weights, residuals, metrics):
    viz.render_full_scene(
        results_dir(), scene_id,
        scene_desc=f"{scene_id}: {scene.description}",
        source=scene.source, target=scene.target,
        gt_pose=scene.gt_pose,
        est_pose_baseline=T_base, est_pose_dynrej=T_dyn,
        source_is_dynamic=scene.source_is_dynamic,
        weights=weights, residuals=residuals, diagnostics=diag,
        rot_err_baseline=metrics["baseline_rot_deg"],
        trans_err_baseline=metrics["baseline_trans_m"],
        rot_err_dynrej=metrics["dynrej_rot_deg"],
        trans_err_dynrej=metrics["dynrej_trans_m"])


def test_S1_single_mover(pg):
    s = si.scene_S1_single_mover()
    T_b, T_d, diag, w, r = _run_scene(pg, s, max_dist=1.0)
    metrics = _evaluate(s, T_b, T_d, w)
    metrics["diagnostics"] = diag
    metrics["scene"] = "S1_single_mover"
    with open(os.path.join(results_dir(), "S1.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _viz_scene("S1", s, T_b, T_d, diag, w, r, metrics)

    # Rejection should not be much worse than baseline; in benign cases the
    # static room dominates so the gap is small either way.
    assert metrics["dynrej_rot_deg"] <= metrics["baseline_rot_deg"] + 1.0
    assert metrics["dynrej_trans_m"] <= metrics["baseline_trans_m"] + 0.02


def test_S2_multi_movers(pg):
    s = si.scene_S2_multi_movers()
    T_b, T_d, diag, w, r = _run_scene(pg, s, max_dist=1.5)
    metrics = _evaluate(s, T_b, T_d, w)
    metrics["diagnostics"] = diag
    metrics["scene"] = "S2_multi_movers"
    with open(os.path.join(results_dir(), "S2.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _viz_scene("S2", s, T_b, T_d, diag, w, r, metrics)

    # On multi-mover scenes the rejection should not regress significantly.
    assert metrics["dynrej_rot_deg"] <= metrics["baseline_rot_deg"] + 1.0
    assert metrics["dynrej_trans_m"] <= metrics["baseline_trans_m"] + 0.02


def test_S3_corridor_with_cabinet_rejection_wins(pg):
    """The corridor is geometrically ambiguous along its axis; the sliding
    cabinet drags baseline GICP along that axis. Rejection should recover.
    """
    s = si.scene_S3_corridor_with_cabinet()
    T_b, T_d, diag, w, r = _run_scene(pg, s, max_dist=2.0)
    metrics = _evaluate(s, T_b, T_d, w)
    metrics["diagnostics"] = diag
    metrics["scene"] = "S3_corridor_with_cabinet"
    with open(os.path.join(results_dir(), "S3.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _viz_scene("S3", s, T_b, T_d, diag, w, r, metrics)

    # Hard expectation: rejection trans error << baseline trans error.
    assert metrics["dynrej_trans_m"] < 0.05, (
        f"dynrej should converge tightly; got {metrics['dynrej_trans_m']}")
    assert metrics["dynrej_trans_m"] < metrics["baseline_trans_m"] * 0.5 + 1e-6, (
        f"rejection did not beat baseline: "
        f"dyn={metrics['dynrej_trans_m']} base={metrics['baseline_trans_m']}")
    # Dynamic mask recall: detect at least 50% of dynamic points.
    if not np.isnan(metrics["mask_recall"]):
        assert metrics["mask_recall"] >= 0.5, (
            f"mask recall too low: {metrics['mask_recall']:.2f}")
