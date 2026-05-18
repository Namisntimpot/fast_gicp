"""Common helpers for tests."""

import numpy as np


def make_gicp(pg, target, source, max_dist=2.0, threads=8):
    g = pg.FastGICP()
    g.set_num_threads(threads)
    g.set_max_correspondence_distance(max_dist)
    g.set_input_target(target)
    g.set_input_source(source)
    return g


def run_align(pg, scene, *, rejection_config=None, max_dist=2.0,
              initial_guess=None, anchors=None, return_diag=False):
    g = make_gicp(pg, scene.target, scene.source, max_dist=max_dist)
    if rejection_config is not None:
        g.set_dynamic_rejection_config(rejection_config)
    if anchors is not None:
        src_a, tgt_a, w, sig = anchors
        g.set_sparse_anchor_correspondences(src_a, tgt_a, w, sig)
    if initial_guess is None:
        initial_guess = np.eye(4, dtype=np.float32)
    T = g.align(initial_guess.astype(np.float32))
    diag = g.get_dynamic_rejection_diagnostics() if return_diag else None
    return T.astype(np.float64), diag, g


def perturb_pose(T, rot_deg, trans_m, axis=(0.0, 0.0, 1.0), seed=0):
    rng = np.random.default_rng(seed)
    theta = np.radians(rot_deg)
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    Rp = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    tp = trans_m * rng.normal(0.0, 1.0, size=3)
    if np.linalg.norm(tp) > 1e-9:
        tp = tp / np.linalg.norm(tp) * trans_m
    P = np.eye(4)
    P[:3, :3] = Rp
    P[:3, 3] = tp
    return P @ T


def mask_recall_precision(predicted_outlier_mask, gt_outlier_mask):
    """Return (recall, precision) on the dynamic/outlier class."""
    if gt_outlier_mask.sum() == 0:
        return float("nan"), float("nan")
    tp = (predicted_outlier_mask & gt_outlier_mask).sum()
    fp = (predicted_outlier_mask & ~gt_outlier_mask).sum()
    fn = (~predicted_outlier_mask & gt_outlier_mask).sum()
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else float("nan")
    return float(recall), float(precision)
