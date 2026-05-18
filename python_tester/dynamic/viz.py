"""Visualization helpers for dynamic-GICP tests.

Each renderer produces a PNG into ``results/<scene>_<view>.png`` and contributes
an entry to ``results/index.html`` (regenerated from the JSON metric files +
PNG files on disk).

Design goals:
- Keep rendering CPU-only and matplotlib-only — no Open3D, no plotly — so the
  same code can run on headless servers without extra config.
- Each public function takes pre-computed numpy arrays + a scene object and
  writes a PNG. No state is shared between renderers.
- The HTML aggregator scans ``results/*.json`` and groups PNGs by scene id.
"""

from __future__ import annotations

import glob
import html
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _equal_3d(ax, pts: np.ndarray) -> None:
    """Set equal aspect ratio for a matplotlib 3D axis given a point set."""
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2
    spans = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - spans, centers[0] + spans)
    ax.set_ylim(centers[1] - spans, centers[1] + spans)
    ax.set_zlim(centers[2] - spans, centers[2] + spans)


def _subsample(pts: np.ndarray, max_pts: int, seed: int = 0):
    """Random subsample to keep figures responsive."""
    if pts.shape[0] <= max_pts:
        return pts, np.arange(pts.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_pts, replace=False)
    return pts[idx], idx


def _add_pose_overlay(ax, T: np.ndarray, scale: float = 0.3, label: str = "T"):
    """Draw a coordinate frame for pose T at its translation."""
    origin = T[:3, 3]
    R = T[:3, :3]
    colors = ["r", "g", "b"]
    for k in range(3):
        axis = R[:, k] * scale
        ax.plot([origin[0], origin[0] + axis[0]],
                [origin[1], origin[1] + axis[1]],
                [origin[2], origin[2] + axis[2]],
                color=colors[k], linewidth=2)
    ax.text(origin[0], origin[1], origin[2], label, fontsize=7)


def apply_pose(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


# ---------------------------------------------------------------------------
# core renderers
# ---------------------------------------------------------------------------


def render_scene_overview(
    out_path: str,
    *,
    scene_desc: str,
    source: np.ndarray,
    target: np.ndarray,
    source_is_dynamic: Optional[np.ndarray] = None,
    gt_pose: Optional[np.ndarray] = None,
    est_pose: Optional[np.ndarray] = None,
    title_suffix: str = "",
) -> None:
    """Three-panel figure: source+target (initial), source+target after est align,
    source coloured by GT dynamic mask."""
    fig = plt.figure(figsize=(14, 4.5))
    src_view, _ = _subsample(source, 4000, seed=1)
    tgt_view, _ = _subsample(target, 4000, seed=2)

    # Panel 1: initial overlay
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(src_view[:, 0], src_view[:, 1], src_view[:, 2],
                s=0.6, c="#1f77b4", alpha=0.6, label="source")
    ax1.scatter(tgt_view[:, 0], tgt_view[:, 1], tgt_view[:, 2],
                s=0.6, c="#ff7f0e", alpha=0.4, label="target")
    _equal_3d(ax1, np.vstack([src_view, tgt_view]))
    ax1.set_title("Initial overlay")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.tick_params(labelsize=6)

    # Panel 2: aligned overlay using est pose
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    if est_pose is not None:
        src_aligned = apply_pose(est_pose, src_view)
    else:
        src_aligned = src_view
    ax2.scatter(src_aligned[:, 0], src_aligned[:, 1], src_aligned[:, 2],
                s=0.6, c="#1f77b4", alpha=0.6, label="src (T_est)")
    ax2.scatter(tgt_view[:, 0], tgt_view[:, 1], tgt_view[:, 2],
                s=0.6, c="#ff7f0e", alpha=0.4, label="target")
    if gt_pose is not None:
        _add_pose_overlay(ax2, gt_pose, scale=0.3, label="GT")
    if est_pose is not None:
        _add_pose_overlay(ax2, est_pose, scale=0.3, label="est")
    _equal_3d(ax2, np.vstack([src_aligned, tgt_view]))
    ax2.set_title("After alignment")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.tick_params(labelsize=6)

    # Panel 3: GT dynamic mask
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    if source_is_dynamic is not None and source_is_dynamic.size == source.shape[0]:
        src_v, idx = _subsample(source, 4000, seed=3)
        dyn = source_is_dynamic[idx]
        ax3.scatter(src_v[~dyn, 0], src_v[~dyn, 1], src_v[~dyn, 2],
                    s=0.5, c="#888888", alpha=0.5, label="static")
        ax3.scatter(src_v[dyn, 0], src_v[dyn, 1], src_v[dyn, 2],
                    s=2.0, c="#d62728", alpha=0.9, label="GT dynamic")
        _equal_3d(ax3, src_v)
        ax3.set_title("GT dynamic mask (source)")
        ax3.legend(loc="upper right", fontsize=7)
    else:
        ax3.text2D(0.1, 0.5, "no GT dynamic mask", transform=ax3.transAxes)
        ax3.set_title("GT dynamic mask (n/a)")
    ax3.tick_params(labelsize=6)

    fig.suptitle(f"{scene_desc}{title_suffix}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_weight_overlay(
    out_path: str,
    *,
    scene_desc: str,
    source: np.ndarray,
    weights: np.ndarray,
    source_is_dynamic: Optional[np.ndarray] = None,
) -> None:
    """Source colored by runtime weight + (optionally) GT outlier mask comparison."""
    if weights.size != source.shape[0]:
        # weights vector empty (rejection off / warmup) — write a placeholder.
        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)
        ax.text(0.1, 0.5, f"weights vector size {weights.size} != source size {source.shape[0]}\n"
                "(rejection disabled or only warmup reached)",
                transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        fig.suptitle(f"{scene_desc}\nRuntime weights (n/a)", fontsize=10)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return

    src_v, idx = _subsample(source, 6000, seed=7)
    w_v = weights[idx]
    fig = plt.figure(figsize=(12, 4.5))

    # Panel 1: weights as color
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax1.scatter(src_v[:, 0], src_v[:, 1], src_v[:, 2],
                     c=w_v, cmap="RdYlGn", vmin=0, vmax=1, s=0.8, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax1, shrink=0.6, pad=0.04)
    cbar.set_label("runtime weight", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _equal_3d(ax1, src_v)
    ax1.set_title(f"Runtime weights (mean={float(weights.mean()):.3f}, "
                  f"inlier@≥0.5={float((weights >= 0.5).mean()):.2f})")
    ax1.tick_params(labelsize=6)

    # Panel 2: confusion plot — predicted outlier (w<0.5) vs GT dynamic
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if source_is_dynamic is not None and source_is_dynamic.size == source.shape[0]:
        gt = source_is_dynamic[idx]
        pred = w_v < 0.5
        # Categories: TP (true positive: predicted dynamic, GT dynamic),
        # FP (predicted dynamic, GT static), FN (kept, GT dynamic), TN (kept, GT static).
        tp = pred & gt
        fp = pred & ~gt
        fn = ~pred & gt
        tn = ~pred & ~gt
        for mask, color, label in [
            (tn, "#d0d0d0", "TN kept static"),
            (tp, "#2ca02c", "TP rejected"),
            (fp, "#ff7f0e", "FP rejected static"),
            (fn, "#d62728", "FN missed dyn"),
        ]:
            if mask.any():
                ax2.scatter(src_v[mask, 0], src_v[mask, 1], src_v[mask, 2],
                            c=color, s=0.8 if "TN" in label else 2.0,
                            alpha=0.4 if "TN" in label else 0.9, label=label)
        recall = float(tp.sum() / max((tp.sum() + fn.sum()), 1))
        precision = float(tp.sum() / max((tp.sum() + fp.sum()), 1)) if (tp.sum() + fp.sum()) > 0 else float("nan")
        _equal_3d(ax2, src_v)
        ax2.set_title(f"Pred-vs-GT (recall={recall:.2f}, precision={precision:.2f})")
        ax2.legend(loc="upper right", fontsize=6)
    else:
        ax2.text2D(0.1, 0.5, "no GT mask", transform=ax2.transAxes)
        ax2.set_title("Pred-vs-GT (n/a)")
    ax2.tick_params(labelsize=6)

    fig.suptitle(scene_desc, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_pose_error_bar(
    out_path: str,
    *,
    scene_desc: str,
    baseline_rot_deg: Optional[float],
    baseline_trans_m: Optional[float],
    dynrej_rot_deg: Optional[float],
    dynrej_trans_m: Optional[float],
    extras: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Compare baseline vs dynrej pose error (and optional extras)."""
    fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(9, 3.5))
    labels = []
    rot_vals = []
    trans_vals = []
    if baseline_rot_deg is not None:
        labels.append("baseline")
        rot_vals.append(baseline_rot_deg)
        trans_vals.append(baseline_trans_m or 0.0)
    if dynrej_rot_deg is not None:
        labels.append("dyn rejection")
        rot_vals.append(dynrej_rot_deg)
        trans_vals.append(dynrej_trans_m or 0.0)
    if extras:
        for k, vals in extras.items():
            labels.append(k)
            rot_vals.append(vals.get("rot_deg", 0.0))
            trans_vals.append(vals.get("trans_m", 0.0))
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"][:len(labels)]
    ax_r.bar(labels, rot_vals, color=colors)
    ax_r.set_ylabel("rotation error (deg)")
    ax_r.set_title("Rotation error")
    for i, v in enumerate(rot_vals):
        ax_r.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax_t.bar(labels, trans_vals, color=colors)
    ax_t.set_ylabel("translation error (m)")
    ax_t.set_title("Translation error")
    for i, v in enumerate(trans_vals):
        ax_t.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(scene_desc, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_diagnostics_panel(
    out_path: str,
    *,
    scene_desc: str,
    diagnostics: Dict,
    extra_lines: Optional[Sequence[str]] = None,
) -> None:
    """Render the dynamic-rejection diagnostic dict as a text figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    lines = [scene_desc, ""]
    if diagnostics:
        for k in [
            "enabled", "active", "gnc_iterations", "final_mu", "inlier_sigma",
            "total_correspondences", "inlier_count", "anchor_total",
            "anchor_inlier_count", "mean_inlier_residual", "mean_outlier_residual"
        ]:
            if k in diagnostics:
                v = diagnostics[k]
                if isinstance(v, float):
                    lines.append(f"{k:>26s}: {v:.6g}")
                else:
                    lines.append(f"{k:>26s}: {v}")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=9, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_basin_heatmap(
    out_path: str,
    *,
    scene_desc: str,
    perturb_rot_deg: Sequence[float],
    perturb_trans_m: Sequence[float],
    success_grid_baseline: np.ndarray,  # shape (len(rot), len(trans))
    success_grid_dynrej: np.ndarray,
) -> None:
    """Two heatmaps showing success rate vs init perturbation."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, grid, title in [
        (axes[0], success_grid_baseline, "baseline success rate"),
        (axes[1], success_grid_dynrej, "dyn rejection success rate"),
    ]:
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto",
                       origin="lower",
                       extent=(min(perturb_trans_m), max(perturb_trans_m),
                               min(perturb_rot_deg), max(perturb_rot_deg)))
        ax.set_xlabel("init translation perturbation (m)")
        ax.set_ylabel("init rotation perturbation (deg)")
        ax.set_title(title)
        for i, r in enumerate(perturb_rot_deg):
            for j, t in enumerate(perturb_trans_m):
                ax.text(t, r, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(scene_desc, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_residual_histogram(
    out_path: str,
    *,
    scene_desc: str,
    residuals: np.ndarray,
    weights: Optional[np.ndarray] = None,
    source_is_dynamic: Optional[np.ndarray] = None,
) -> None:
    """Histogram of per-correspondence Mahalanobis residuals, coloured by
    runtime weight (or GT mask if provided)."""
    valid = residuals[(residuals >= 0) & np.isfinite(residuals)]
    if valid.size == 0:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.1, 0.5, "no valid residuals", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7, 3.8))
    if (source_is_dynamic is not None and source_is_dynamic.size == residuals.size):
        gt_static = ~source_is_dynamic & (residuals >= 0)
        gt_dyn = source_is_dynamic & (residuals >= 0)
        bins = np.linspace(0, np.percentile(valid, 99), 50)
        ax.hist(residuals[gt_static], bins=bins, color="#1f77b4", alpha=0.7,
                label=f"static (n={int(gt_static.sum())})")
        ax.hist(residuals[gt_dyn], bins=bins, color="#d62728", alpha=0.7,
                label=f"dynamic (n={int(gt_dyn.sum())})")
    else:
        bins = np.linspace(0, np.percentile(valid, 99), 50)
        ax.hist(valid, bins=bins, color="#1f77b4", alpha=0.7)
    ax.set_xlabel("Mahalanobis residual (squared)")
    ax.set_ylabel("count")
    ax.set_title(scene_desc)
    ax.legend(fontsize=7)
    ax.set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# scene-level convenience wrapper
# ---------------------------------------------------------------------------


def render_full_scene(
    results_dir: str,
    scene_id: str,
    *,
    scene_desc: str,
    source: np.ndarray,
    target: np.ndarray,
    gt_pose: np.ndarray,
    est_pose_baseline: Optional[np.ndarray] = None,
    est_pose_dynrej: Optional[np.ndarray] = None,
    source_is_dynamic: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    residuals: Optional[np.ndarray] = None,
    diagnostics: Optional[Dict] = None,
    rot_err_baseline: Optional[float] = None,
    trans_err_baseline: Optional[float] = None,
    rot_err_dynrej: Optional[float] = None,
    trans_err_dynrej: Optional[float] = None,
    extra_pose_lines: Optional[Sequence[str]] = None,
) -> List[str]:
    """One-call helper that writes all standard PNGs for a single scene.

    Returns the list of file paths written so the caller can include them
    explicitly in any aggregator.
    """
    os.makedirs(results_dir, exist_ok=True)
    paths = []

    p = os.path.join(results_dir, f"{scene_id}_overview.png")
    render_scene_overview(
        p, scene_desc=scene_desc, source=source, target=target,
        source_is_dynamic=source_is_dynamic, gt_pose=gt_pose,
        est_pose=est_pose_dynrej if est_pose_dynrej is not None else est_pose_baseline)
    paths.append(p)

    if weights is not None:
        p = os.path.join(results_dir, f"{scene_id}_weights.png")
        render_weight_overlay(p, scene_desc=scene_desc, source=source,
                              weights=weights, source_is_dynamic=source_is_dynamic)
        paths.append(p)

    if (rot_err_baseline is not None) or (rot_err_dynrej is not None):
        p = os.path.join(results_dir, f"{scene_id}_pose_error.png")
        render_pose_error_bar(
            p, scene_desc=scene_desc,
            baseline_rot_deg=rot_err_baseline, baseline_trans_m=trans_err_baseline,
            dynrej_rot_deg=rot_err_dynrej, dynrej_trans_m=trans_err_dynrej)
        paths.append(p)

    if diagnostics:
        p = os.path.join(results_dir, f"{scene_id}_diagnostics.png")
        render_diagnostics_panel(p, scene_desc=scene_desc, diagnostics=diagnostics,
                                  extra_lines=extra_pose_lines)
        paths.append(p)

    if residuals is not None and residuals.size > 0:
        p = os.path.join(results_dir, f"{scene_id}_residuals.png")
        render_residual_histogram(
            p, scene_desc=scene_desc, residuals=residuals,
            weights=weights, source_is_dynamic=source_is_dynamic)
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# HTML aggregator
# ---------------------------------------------------------------------------


def write_index_html(results_dir: str, out_path: Optional[str] = None) -> str:
    """Walk results_dir, group images by scene id, render an index.html."""
    out_path = out_path or os.path.join(results_dir, "index.html")
    png_paths = sorted(glob.glob(os.path.join(results_dir, "*.png")))
    json_paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))

    # Group PNGs by scene id (filename prefix before first underscore-suffix).
    scenes: Dict[str, List[str]] = {}
    for p in png_paths:
        base = os.path.basename(p)
        # e.g. "S3_corridor_with_cabinet_weights.png" — split off the trailing
        # known suffixes; otherwise use the basename without extension.
        for suffix in ["_overview", "_weights", "_pose_error",
                       "_diagnostics", "_residuals", "_basin"]:
            if suffix in base:
                scene_id = base.split(suffix, 1)[0]
                scenes.setdefault(scene_id, []).append(p)
                break
        else:
            scenes.setdefault(base.rsplit(".", 1)[0], []).append(p)

    # Pull metrics text out of each JSON.
    metrics_by_scene: Dict[str, str] = {}
    for j in json_paths:
        name = os.path.basename(j).rsplit(".", 1)[0]
        # try to match prefix to scene ids
        try:
            data = json.load(open(j))
            metrics_by_scene[name] = json.dumps(data, indent=2)
        except Exception:
            continue

    lines: List[str] = []
    lines.append("<!doctype html><meta charset='utf-8'>")
    lines.append("<title>Dynamic GICP test results</title>")
    lines.append("<style>")
    lines.append(
        "body{font-family:sans-serif;max-width:1300px;margin:0 auto;padding:1em;}"
        "h1{border-bottom:2px solid #333;}"
        "h2{border-bottom:1px dotted #888;margin-top:2em;}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:0.8em;}"
        ".grid img{width:100%;border:1px solid #ccc;}"
        "pre{background:#f5f5f5;padding:0.6em;font-size:11px;overflow-x:auto;}"
        ".desc{color:#666;font-style:italic;}")
    lines.append("</style>")
    lines.append("<h1>Dynamic GICP — test result gallery</h1>")
    lines.append(f"<p class='desc'>Generated from <code>{html.escape(results_dir)}</code> "
                 f"({len(scenes)} scenes, {len(png_paths)} images).</p>")
    for scene_id, imgs in sorted(scenes.items()):
        lines.append(f"<h2>{html.escape(scene_id)}</h2>")
        lines.append("<div class='grid'>")
        for img in imgs:
            rel = os.path.relpath(img, results_dir)
            lines.append(f"<a href='{html.escape(rel)}'>"
                         f"<img src='{html.escape(rel)}' alt='{html.escape(rel)}'></a>")
        lines.append("</div>")
        # Inline JSON if available
        if scene_id in metrics_by_scene:
            lines.append("<pre>" + html.escape(metrics_by_scene[scene_id]) + "</pre>")
    html_text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(html_text)
    return out_path
