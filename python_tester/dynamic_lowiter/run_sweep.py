"""Low-iteration sweep: time + accuracy across S1-S10 x iters x backends x dyn variants.

Designed for the real-time-SLAM use case where only 1-3 LM iterations are
affordable per frame. Each (scene, iters, backend, dyn) cell is timed
best-of-N and pose error is recorded.

Outputs:
  results/sweep.json   raw per-cell numbers
  results/summary.csv  flat table for downstream analysis
  results/*.png        summary plots (quality-vs-iters, time-vs-iters,
                       Pareto, per-scene drilldown)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dynamic"))   # synthetic_indoor.py
sys.path.insert(0, os.path.join(HERE, "..", "..", "build_dyn_cuda"))

import pygicp                                              # noqa: E402
import synthetic_indoor as si                              # noqa: E402

assert "build_dyn_cuda" in pygicp.__file__, "wrong pygicp loaded"
assert hasattr(pygicp, "FastGICPCuda"), "pygicp.FastGICPCuda not built"

ITER_GRID = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
DYN_VARIANTS = [
    ("off",        None),
    ("warmup3",    {"enable": True, "kernel": "GEMAN_MCCLURE",
                    "warmup_iterations": 3, "min_inlier_ratio": 0.2}),
    ("warmup0",    {"enable": True, "kernel": "GEMAN_MCCLURE",
                    "warmup_iterations": 0, "min_inlier_ratio": 0.2}),
]
N_TIMING_RUNS = 4     # discard run 0 (warmup); take min of remaining
MAX_CORR_DIST = 2.0


@dataclass
class Cell:
    scene: str
    n_source: int
    n_target: int
    backend: str          # "cpu" | "cuda_bf"
    dyn: str              # "off" | "warmup3" | "warmup0"
    iters: int
    time_min_ms: float
    time_mean_ms: float
    rot_deg: float
    trans_m: float
    final_cost: float
    converged: bool
    gnc_iterations: int
    inlier_count: int
    total_corr: int


# ---------------------------------------------------------------------------
# Scene assembly. Each entry yields:
#   id, description, source(np.ndarray Nx3), target(np.ndarray Mx3), gt_pose
# Some scenes also need extra state (anchors, 2DGS covariance arrays); those
# return a small "setup_extra" closure that mutates a registration instance.
# ---------------------------------------------------------------------------
def _scene_list():
    scenes = []
    for fac, sid in [
        (si.scene_S1_single_mover, "S1"),
        (si.scene_S2_multi_movers, "S2"),
        (si.scene_S3_corridor_with_cabinet, "S3"),
        (si.scene_S4_mirror_room, "S4"),
        (si.scene_S5_planar_floor, "S5"),
        (si.scene_S6_big_init_offset, "S6"),
        (si.scene_S7_combined_worst_case, "S7"),
        (si.scene_S10_static_regression, "S10"),
    ]:
        s = fac()
        scenes.append((sid, s, None))

    # S8 — sparse anchors
    base, src_a, tgt_a, w, sig, _ = si.scene_S8_anchors_on_dynamic()
    def s8_extra(reg, src_a=src_a, tgt_a=tgt_a, w=w, sig=sig):
        reg.set_sparse_anchor_config({"objective_weight": 50.0,
                                      "balance_mode": "BY_HESSIAN_TRACE"})
        reg.set_sparse_anchor_correspondences(src_a, tgt_a, w, sig)
    scenes.append(("S8", base, s8_extra))

    # S9 — 2DGS surfel covariances
    s9, src_rot, src_scl, tgt_rot, tgt_scl = si.scene_S9_2dgs_surfels()
    def s9_extra(reg, sr=src_rot, ss=src_scl, tr=tgt_rot, ts=tgt_scl):
        reg.set_source_covariances_from_2dgs(sr.astype(np.float32),
                                              ss.astype(np.float32),
                                              "physical", 0.05, 1e-3)
        reg.set_target_covariances_from_2dgs(tr.astype(np.float32),
                                              ts.astype(np.float32),
                                              "physical", 0.05, 1e-3)
    scenes.append(("S9", s9, s9_extra))
    return scenes


def _make_reg(backend: str):
    if backend == "cpu":
        g = pygicp.FastGICP()
        g.set_num_threads(8)
    elif backend == "cuda_bf":
        g = pygicp.FastGICPCuda()
        g.set_knn_backend("brute_force")
    else:
        raise ValueError(backend)
    g.set_correspondence_randomness(20)
    g.set_max_correspondence_distance(MAX_CORR_DIST)
    return g


def _time_align(scene_pts, setup_extra, backend, iters, dyn_cfg):
    """Return (time_min_s, time_mean_s, T_est, diagnostics)."""
    src, tgt = scene_pts
    times = []
    diag = {}
    T_est = None
    for r in range(N_TIMING_RUNS):
        g = _make_reg(backend)
        g.set_input_target(tgt)
        g.set_input_source(src)
        if setup_extra is not None:
            setup_extra(g)
        if dyn_cfg is not None:
            g.set_dynamic_rejection_config(dyn_cfg)
        g.set_max_iterations(iters)
        t0 = time.perf_counter()
        T_est = g.align(np.eye(4, dtype=np.float32))
        dt = time.perf_counter() - t0
        if r > 0:
            times.append(dt)
        if r == N_TIMING_RUNS - 1:
            try:
                diag = g.get_dynamic_rejection_diagnostics() if dyn_cfg else {}
                rep = g.get_alignment_quality_report()
                diag["final_cost"] = rep.get("final_cost", float("nan"))
                diag["converged"] = bool(rep.get("converged", False))
            except Exception:
                diag = {"final_cost": float("nan"), "converged": False}
    return float(min(times)), float(np.mean(times)), T_est.astype(np.float64), diag


def run() -> Dict:
    scenes = _scene_list()
    cells: List[Cell] = []
    n_total = len(scenes) * len(ITER_GRID) * 2 * len(DYN_VARIANTS)
    progress = 0
    print(f"[lowiter] sweeping {len(scenes)} scenes x {len(ITER_GRID)} iters "
          f"x 2 backends x {len(DYN_VARIANTS)} dyn variants = {n_total} cells")

    for sid, scene, setup_extra in scenes:
        n_s, n_t = scene.source.shape[0], scene.target.shape[0]
        for backend in ["cpu", "cuda_bf"]:
            for dyn_name, dyn_cfg in DYN_VARIANTS:
                for iters in ITER_GRID:
                    progress += 1
                    try:
                        t_min, t_mean, T_est, diag = _time_align(
                            (scene.source, scene.target), setup_extra,
                            backend, iters, dyn_cfg)
                        rot, tr = si.pose_error(scene.gt_pose, T_est)
                        cell = Cell(
                            scene=sid, n_source=int(n_s), n_target=int(n_t),
                            backend=backend, dyn=dyn_name, iters=iters,
                            time_min_ms=t_min * 1000.0,
                            time_mean_ms=t_mean * 1000.0,
                            rot_deg=float(rot), trans_m=float(tr),
                            final_cost=float(diag.get("final_cost", float("nan"))),
                            converged=bool(diag.get("converged", False)),
                            gnc_iterations=int(diag.get("gnc_iterations", 0)),
                            inlier_count=int(diag.get("inlier_count", 0)),
                            total_corr=int(diag.get("total_correspondences", n_s)),
                        )
                    except Exception as e:
                        print(f"  [{progress}/{n_total}] FAIL {sid}/{backend}/"
                              f"{dyn_name}/iter={iters}: {e}")
                        cell = Cell(scene=sid, n_source=int(n_s), n_target=int(n_t),
                                    backend=backend, dyn=dyn_name, iters=iters,
                                    time_min_ms=float("nan"),
                                    time_mean_ms=float("nan"),
                                    rot_deg=float("nan"), trans_m=float("nan"),
                                    final_cost=float("nan"), converged=False,
                                    gnc_iterations=0, inlier_count=0,
                                    total_corr=n_s)
                    cells.append(cell)
        print(f"  [{sid:>3}] done ({progress}/{n_total})")

    out = dict(scenes=[asdict(c) for c in cells], iter_grid=ITER_GRID,
               dyn_variants=[name for name, _ in DYN_VARIANTS])
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "sweep.json"), "w") as f:
        json.dump(out, f, indent=2)

    # CSV
    with open(os.path.join(results_dir, "summary.csv"), "w") as f:
        f.write("scene,n_source,n_target,backend,dyn,iters,time_min_ms,"
                "time_mean_ms,rot_deg,trans_m,final_cost,converged,"
                "gnc_iterations,inlier_count,total_corr\n")
        for c in cells:
            f.write(f"{c.scene},{c.n_source},{c.n_target},{c.backend},{c.dyn},"
                    f"{c.iters},{c.time_min_ms:.4f},{c.time_mean_ms:.4f},"
                    f"{c.rot_deg:.6f},{c.trans_m:.6f},{c.final_cost:.6e},"
                    f"{c.converged},{c.gnc_iterations},{c.inlier_count},"
                    f"{c.total_corr}\n")
    print(f"[lowiter] wrote {len(cells)} cells to {results_dir}/sweep.json + summary.csv")
    return out


if __name__ == "__main__":
    run()
