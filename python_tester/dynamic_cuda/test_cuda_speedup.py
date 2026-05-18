"""Speedup benchmarks for FastGICPCuda vs FastGICP on indoor scenes.

Sweeps target sizes up to 500K to show the CUDA advantage on large keyframe
windows (matching the user's typical SLAM use case: source ~ a few tens of K,
target frequently exceeds 100K when many keyframes accumulate).

Each case is run 3 times with the first run discarded to amortise GPU warmup,
JIT, and KD-tree construction. Pose error is verified to remain bounded so we
catch any silent precision blow-up at scale.
"""

import json
import os
import time

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _loader import load as load_cuda, results_dir
import synthetic_indoor as si


@pytest.fixture(scope="module")
def pg():
    return load_cuda()


def _make_synthetic_pair(n_target, src_subsample, seed=11):
    """Furnished room + 4 box pieces. Returns (source, target, gt_pose).

    Target size scales by sampling step in `make_room` then subsampling boxes.
    Source is a downsampled view of the same scene (smaller and noisier).
    """
    rng = np.random.default_rng(seed)
    # Build a "dense" room of ~n_target points by tuning the grid step.
    # Approximate surface area of a 6x6x3 room ~ 144 m^2; step s gives
    # ~144/s^2 points. So step = sqrt(144 / n_target).
    step = float(np.sqrt(144.0 / max(n_target, 100)))
    room = si.make_room(width=6.0, depth=6.0, height=3.0, step=step)
    box1 = si.make_box((1.5, 1.0, 0.5), (0.8, 0.5, 1.0), step=step)
    box2 = si.make_box((-1.0, 1.4, 0.3), (0.6, 0.6, 0.6), step=step)
    box3 = si.make_box((0.8, -1.5, 0.4), (0.5, 1.0, 0.8), step=step)
    box4 = si.make_box((-1.4, -0.8, 0.5), (0.7, 0.7, 0.9), step=step)
    scene = np.vstack([room, box1, box2, box3, box4])
    # Subsample to target sizes.
    if scene.shape[0] > n_target:
        idx = rng.choice(scene.shape[0], size=n_target, replace=False)
        target = scene[idx]
    else:
        target = scene
    target = si.add_noise(target, 0.003, seed + 1)

    # Source is independent draw on the same scene, smaller.
    if scene.shape[0] > src_subsample:
        idx_src = rng.choice(scene.shape[0], size=src_subsample, replace=False)
        source = scene[idx_src]
    else:
        source = scene
    source = si.add_noise(source, 0.003, seed + 2)
    # Apply a known sensor motion (gt_pose maps source to target frame).
    T_gt = si.make_pose(si.rotation_z(np.radians(8.0)), np.array([0.12, -0.05, 0.01]))
    target = si.apply_pose(T_gt, target)
    return source, target, T_gt


def _time_align(make_align, runs=3):
    times = []
    last_T = None
    for r in range(runs + 1):
        t0 = time.perf_counter()
        T = make_align()
        dt = time.perf_counter() - t0
        if r > 0:
            times.append(dt)
        last_T = T
    return float(min(times)), float(np.mean(times)), last_T


def _cuvs_available(pg):
    """Probe whether the CUDA build was compiled with cuVS."""
    try:
        g = pg.FastGICPCuda()
        g.set_knn_backend("cuvs")
        return True
    except RuntimeError:
        return False


@pytest.mark.parametrize("n_target", [10_000, 50_000, 200_000, 500_000])
def test_cuda_vs_cpu_speedup(pg, n_target):
    src_n = 20_000
    source, target, T_gt = _make_synthetic_pair(n_target=n_target,
                                                src_subsample=src_n)
    actual_target = target.shape[0]
    actual_source = source.shape[0]

    def cpu_align():
        g = pg.FastGICP()
        g.set_num_threads(16)
        g.set_correspondence_randomness(20)
        g.set_max_correspondence_distance(2.0)
        g.set_input_target(target)
        g.set_input_source(source)
        return g.align(np.eye(4, dtype=np.float32))

    def cuda_align(backend):
        g = pg.FastGICPCuda()
        g.set_knn_backend(backend)
        g.set_correspondence_randomness(20)
        g.set_max_correspondence_distance(2.0)
        g.set_input_target(target)
        g.set_input_source(source)
        return g.align(np.eye(4, dtype=np.float32))

    cpu_min, cpu_mean, T_cpu = _time_align(cpu_align)
    bf_min, bf_mean, T_bf = _time_align(lambda: cuda_align("brute_force"))

    cuvs_min = cuvs_mean = float("nan")
    cuvs_rot = cuvs_trans = float("nan")
    has_cuvs = _cuvs_available(pg)
    if has_cuvs:
        cuvs_min, cuvs_mean, T_cuvs = _time_align(lambda: cuda_align("cuvs"))
        cuvs_rot, cuvs_trans = si.pose_error(T_gt, T_cuvs.astype(np.float64))

    rot_cpu, t_cpu = si.pose_error(T_gt, T_cpu.astype(np.float64))
    rot_bf, t_bf = si.pose_error(T_gt, T_bf.astype(np.float64))
    speedup_bf = cpu_min / max(bf_min, 1e-9)
    speedup_cuvs = cpu_min / max(cuvs_min, 1e-9) if has_cuvs else float("nan")

    out = dict(
        scene=f"speedup_n{n_target}",
        n_target_requested=n_target,
        n_target=int(actual_target),
        n_source=int(actual_source),
        cpu_min_s=cpu_min, cpu_mean_s=cpu_mean,
        cuda_brute_min_s=bf_min, cuda_brute_mean_s=bf_mean,
        cuda_cuvs_min_s=cuvs_min, cuda_cuvs_mean_s=cuvs_mean,
        speedup_brute_vs_cpu=speedup_bf,
        speedup_cuvs_vs_cpu=speedup_cuvs,
        cpu_rot_deg=rot_cpu, cpu_trans_m=t_cpu,
        cuda_brute_rot_deg=rot_bf, cuda_brute_trans_m=t_bf,
        cuda_cuvs_rot_deg=cuvs_rot, cuda_cuvs_trans_m=cuvs_trans,
        has_cuvs=has_cuvs,
    )
    with open(os.path.join(results_dir(), f"speedup_n{n_target}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [n_target={actual_target}, n_source={actual_source}]")
    print(f"    CPU         {cpu_min*1000:7.1f} ms / {cpu_mean*1000:7.1f} ms   "
          f"rot={rot_cpu:.4f} deg  trans={t_cpu:.5f} m")
    print(f"    CUDA brute  {bf_min*1000:7.1f} ms / {bf_mean*1000:7.1f} ms   "
          f"rot={rot_bf:.4f} deg  trans={t_bf:.5f} m   speedup={speedup_bf:.2f}x")
    if has_cuvs:
        print(f"    CUDA cuvs   {cuvs_min*1000:7.1f} ms / {cuvs_mean*1000:7.1f} ms   "
              f"rot={cuvs_rot:.4f} deg  trans={cuvs_trans:.5f} m   speedup={speedup_cuvs:.2f}x")

    # Sanity floors so a silent regression is caught.
    assert rot_bf < 1.0, f"CUDA brute rotation diverged at n={n_target}"
    assert t_bf < 0.10, f"CUDA brute translation diverged at n={n_target}"
    if has_cuvs:
        # Allow up to a few mm difference vs brute_force at large scales (top-k
        # ties may be resolved differently).
        assert abs(t_bf - cuvs_trans) < 0.01, (
            f"cuVS pose drift vs brute: bf={t_bf:.5f} cuvs={cuvs_trans:.5f}")


def test_summary_plot(pg):
    """Aggregate per-size speedup JSON files into a summary plot."""
    cases = []
    for n in [10_000, 50_000, 200_000, 500_000]:
        path = os.path.join(results_dir(), f"speedup_n{n}.json")
        if os.path.exists(path):
            cases.append(json.load(open(path)))
    if not cases:
        pytest.skip("no speedup results to summarise — run parameterised tests first")

    cases.sort(key=lambda c: c["n_target"])
    ns = [c["n_target"] for c in cases]
    cpu_ms = [c["cpu_min_s"] * 1000 for c in cases]
    bf_ms = [c["cuda_brute_min_s"] * 1000 for c in cases]
    sp_bf = [c["speedup_brute_vs_cpu"] for c in cases]
    has_cuvs = any(c.get("has_cuvs") for c in cases)
    cuvs_ms = [c.get("cuda_cuvs_min_s", float("nan")) * 1000 for c in cases]
    sp_cuvs = [c.get("speedup_cuvs_vs_cpu", float("nan")) for c in cases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ns, cpu_ms, "o-", color="#1f77b4", label="CPU (16 threads)")
    ax1.plot(ns, bf_ms, "s-", color="#2ca02c", label="CUDA brute_force")
    if has_cuvs:
        ax1.plot(ns, cuvs_ms, "^-", color="#9467bd", label="CUDA cuvs (brute_force backend)")
    ax1.set_xlabel("target cloud size")
    ax1.set_ylabel("align() wall time (ms, best of 3)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("Wall time vs target size")
    for n, c in zip(ns, cpu_ms): ax1.annotate(f"{c:.0f}", (n, c), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6, color="#1f77b4")
    for n, c in zip(ns, bf_ms):  ax1.annotate(f"{c:.0f}", (n, c), textcoords="offset points", xytext=(0,-14), ha="center", fontsize=6, color="#2ca02c")
    if has_cuvs:
        for n, c in zip(ns, cuvs_ms):
            if not np.isnan(c):
                ax1.annotate(f"{c:.0f}", (n, c), textcoords="offset points", xytext=(8, 2), ha="left", fontsize=6, color="#9467bd")

    x = np.arange(len(ns))
    w = 0.4 if has_cuvs else 0.6
    ax2.bar(x - w/2, sp_bf, w, color="#2ca02c", label="brute_force")
    if has_cuvs:
        ax2.bar(x + w/2, sp_cuvs, w, color="#9467bd", label="cuvs")
    ax2.set_xticks(x); ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("target cloud size")
    ax2.set_ylabel("speedup vs CPU (best-of-N)")
    ax2.axhline(1.0, color="#888", linestyle="--", linewidth=1)
    ax2.set_title("CUDA speedup over CPU")
    for i, v in enumerate(sp_bf): ax2.text(x[i] - w/2, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=7)
    if has_cuvs:
        for i, v in enumerate(sp_cuvs):
            if not np.isnan(v): ax2.text(x[i] + w/2, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=7)
    ax2.legend(fontsize=8)
    fig.suptitle("FastGICPCuda vs FastGICP — synthetic indoor scene, source ~ 20K",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(results_dir(), "speedup_summary.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def test_cuvs_parity_with_brute_force(pg):
    """Side-by-side parity: cuVS and brute_force backends should produce the
    same pose (up to top-k tie-breaking and float associativity)."""
    if not _cuvs_available(pg):
        pytest.skip("cuVS backend not compiled in (build with -DUSE_CUVS=ON)")
    source, target, T_gt = _make_synthetic_pair(n_target=80_000, src_subsample=15_000)

    def run(backend):
        g = pg.FastGICPCuda()
        g.set_knn_backend(backend)
        g.set_correspondence_randomness(20)
        g.set_max_correspondence_distance(2.0)
        g.set_input_target(target)
        g.set_input_source(source)
        return g.align(np.eye(4, dtype=np.float32)).astype(np.float64)

    T_bf = run("brute_force")
    T_cu = run("cuvs")
    rot_bf, t_bf = si.pose_error(T_gt, T_bf)
    rot_cu, t_cu = si.pose_error(T_gt, T_cu)
    print(f"\n  parity: brute_force rot={rot_bf:.4f}deg trans={t_bf:.5f}m | "
          f"cuvs rot={rot_cu:.4f}deg trans={t_cu:.5f}m")
    out = dict(scene="parity", brute_force_rot_deg=rot_bf, brute_force_trans_m=t_bf,
               cuvs_rot_deg=rot_cu, cuvs_trans_m=t_cu)
    with open(os.path.join(results_dir(), "parity.json"), "w") as f:
        json.dump(out, f, indent=2)
    assert abs(rot_bf - rot_cu) < 0.2
    assert abs(t_bf - t_cu) < 0.005
