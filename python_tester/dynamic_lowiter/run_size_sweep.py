"""Low-iteration size sweep: iter=[1,2,3,5,10] x size=[10K,50K,200K,500K]
x backend={cpu, cuda_bf} x dyn={off, warmup0}.

The synthetic scene is identical to test_cuda_speedup.py so results are
comparable. dyn=warmup3 is skipped because at iter<=3 it never activates;
warmup0 lets us see whether immediate rejection helps or hurts at low iters.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dynamic"))
sys.path.insert(0, os.path.join(HERE, "..", "dynamic_cuda"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "build_dyn_cuda"))

import pygicp                                              # noqa: E402
import synthetic_indoor as si                              # noqa: E402
from test_cuda_speedup import _make_synthetic_pair          # noqa: E402

assert "build_dyn_cuda" in pygicp.__file__

ITER_GRID = [1, 2, 3, 5, 10]
SIZE_GRID = [10_000, 50_000, 200_000, 500_000]
SRC_SIZE = 20_000
DYN_VARIANTS = [
    ("off",     None),
    ("warmup0", {"enable": True, "kernel": "GEMAN_MCCLURE",
                 "warmup_iterations": 0, "min_inlier_ratio": 0.2}),
]
N_TIMING_RUNS = 4


@dataclass
class Cell:
    n_target: int
    n_source: int
    backend: str
    dyn: str
    iters: int
    time_min_ms: float
    time_mean_ms: float
    rot_deg: float
    trans_m: float


def _make_reg(backend):
    if backend == "cpu":
        g = pygicp.FastGICP(); g.set_num_threads(8)
    else:
        g = pygicp.FastGICPCuda(); g.set_knn_backend("brute_force")
    g.set_correspondence_randomness(20)
    g.set_max_correspondence_distance(2.0)
    return g


def _time_align(target, source, backend, iters, dyn_cfg):
    times = []
    T_est = None
    for r in range(N_TIMING_RUNS):
        g = _make_reg(backend)
        g.set_input_target(target); g.set_input_source(source)
        if dyn_cfg is not None:
            g.set_dynamic_rejection_config(dyn_cfg)
        g.set_max_iterations(iters)
        t0 = time.perf_counter()
        T_est = g.align(np.eye(4, dtype=np.float32))
        dt = time.perf_counter() - t0
        if r > 0: times.append(dt)
    return float(min(times)), float(np.mean(times)), T_est.astype(np.float64)


def run():
    cells: List[Cell] = []
    for n_tgt in SIZE_GRID:
        source, target, T_gt = _make_synthetic_pair(n_target=n_tgt,
                                                     src_subsample=SRC_SIZE)
        actual_t, actual_s = target.shape[0], source.shape[0]
        for backend in ["cpu", "cuda_bf"]:
            for dyn_name, dyn_cfg in DYN_VARIANTS:
                for iters in ITER_GRID:
                    t_min, t_mean, T_est = _time_align(
                        target, source, backend, iters, dyn_cfg)
                    rot, tr = si.pose_error(T_gt, T_est)
                    cells.append(Cell(n_target=actual_t, n_source=actual_s,
                                       backend=backend, dyn=dyn_name,
                                       iters=iters,
                                       time_min_ms=t_min * 1000,
                                       time_mean_ms=t_mean * 1000,
                                       rot_deg=float(rot),
                                       trans_m=float(tr)))
        print(f"  size {actual_t} done")
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "size_sweep.json"), "w") as f:
        json.dump([asdict(c) for c in cells], f, indent=2)

    with open(os.path.join(results_dir, "size_summary.csv"), "w") as f:
        f.write("n_target,n_source,backend,dyn,iters,time_min_ms,time_mean_ms,rot_deg,trans_m\n")
        for c in cells:
            f.write(f"{c.n_target},{c.n_source},{c.backend},{c.dyn},{c.iters},"
                    f"{c.time_min_ms:.4f},{c.time_mean_ms:.4f},"
                    f"{c.rot_deg:.6f},{c.trans_m:.6f}\n")
    print(f"[size] wrote {len(cells)} cells")


if __name__ == "__main__":
    run()
