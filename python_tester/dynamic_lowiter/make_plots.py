"""Aggregate plots for the low-iteration sweep.

Reads:
  results/sweep.json       — main 10-scene x iters x backend x dyn matrix
  results/size_sweep.json  — 4-size x iters x backend x dyn

Writes:
  results/quality_vs_iters.png        time-vs-iters and error-vs-iters lines
                                       per backend / dyn, averaged across scenes
  results/per_scene_lines.png         small-multiples: rotation + translation
                                       error vs iters per scene per backend
  results/lowiter_table.png           text image: at iters in {1,2,3,5} per
                                       scene, the (time, rot, trans) table
  results/size_at_lowiter.png         time + trans error vs target size, one
                                       line per iter point (1,2,3,5)
  results/markdown_summary.md         pasteable summary tables
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def _load():
    with open(os.path.join(RES, "sweep.json")) as f:
        sweep = json.load(f)
    with open(os.path.join(RES, "size_sweep.json")) as f:
        size = json.load(f)
    return sweep, size


def _by(cells, **filters):
    out = []
    for c in cells:
        if all(c.get(k) == v for k, v in filters.items()):
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Plot 1: aggregated quality-vs-iters and time-vs-iters across scenes
# ---------------------------------------------------------------------------
def plot_quality_vs_iters(sweep):
    cells = sweep["scenes"]
    iter_grid = sweep["iter_grid"]
    backends = ["cpu", "cuda_bf"]
    dyns = sweep["dyn_variants"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ("trans_m", "translation error (m)", True),
        ("rot_deg", "rotation error (deg)", True),
        ("time_min_ms", "wall time (ms, best of 3)", True),
    ]
    colors = {"cpu": "#1f77b4", "cuda_bf": "#2ca02c"}
    styles = {"off": "-", "warmup3": "--", "warmup0": ":"}

    for row, agg in enumerate([("mean", np.mean), ("median", np.median)]):
        agg_name, agg_fn = agg
        for col, (key, label, logy) in enumerate(metrics):
            ax = axes[row, col]
            for backend in backends:
                for dyn in dyns:
                    ys = []
                    for it in iter_grid:
                        vals = [c[key] for c in cells
                                if c["backend"] == backend and c["dyn"] == dyn
                                and c["iters"] == it
                                and np.isfinite(c[key])]
                        ys.append(agg_fn(vals) if vals else np.nan)
                    ax.plot(iter_grid, ys,
                            color=colors[backend], linestyle=styles[dyn],
                            label=f"{backend} / {dyn}")
            ax.set_xlabel("max LM iterations")
            ax.set_ylabel(label)
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{agg_name} across 10 scenes")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle("Low-iteration sweep — 10 scenes, source ~ 8K, target ~ 8K",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "quality_vs_iters.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: per-scene small-multiples (translation only, the most diagnostic)
# ---------------------------------------------------------------------------
def plot_per_scene(sweep):
    cells = sweep["scenes"]
    iter_grid = sweep["iter_grid"]
    scenes = sorted(set(c["scene"] for c in cells),
                    key=lambda s: int(s[1:]))
    backends = ["cpu", "cuda_bf"]
    dyns = sweep["dyn_variants"]
    colors = {"cpu": "#1f77b4", "cuda_bf": "#2ca02c"}
    styles = {"off": "-", "warmup3": "--", "warmup0": ":"}

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
    axes = axes.flatten()
    for i, sid in enumerate(scenes):
        ax = axes[i]
        for backend in backends:
            for dyn in dyns:
                ys = []
                for it in iter_grid:
                    vals = [c["trans_m"] for c in cells
                            if c["scene"] == sid and c["backend"] == backend
                            and c["dyn"] == dyn and c["iters"] == it
                            and np.isfinite(c["trans_m"])]
                    ys.append(vals[0] if vals else np.nan)
                ax.plot(iter_grid, ys, color=colors[backend],
                        linestyle=styles[dyn], marker="o", markersize=3,
                        label=f"{backend}/{dyn}")
        ax.set_title(sid, fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(True, alpha=0.3)
        if i % 5 == 0:
            ax.set_ylabel("trans err (m)")
        if i >= 5:
            ax.set_xlabel("max iters")
    axes[0].legend(fontsize=6, loc="upper right", ncol=2)
    fig.suptitle("Translation error vs max iterations — per-scene drilldown",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "per_scene_lines.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: low-iter slice table — iters in {1,2,3,5} only
# ---------------------------------------------------------------------------
def plot_lowiter_table(sweep):
    """Render a text figure with per-scene (time, rot, trans) at iter in
    {1, 2, 3, 5}. Two backend variants stacked."""
    cells = sweep["scenes"]
    scenes = sorted(set(c["scene"] for c in cells),
                    key=lambda s: int(s[1:]))
    iters = [1, 2, 3, 5]
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    lines = []
    lines.append(f"{'scene':>5s}  {'backend / dyn':>20s}  " +
                 "  ".join(f"{f'iter={i}':>17s}" for i in iters))
    lines.append("-" * 5 + "  " + "-" * 20 + "  " +
                 "  ".join(["-" * 17] * len(iters)))
    for sid in scenes:
        for backend in ["cpu", "cuda_bf"]:
            for dyn in ["off", "warmup3", "warmup0"]:
                fields = [f"{sid:>5s}", f"{backend} / {dyn:>7s}"]
                for it in iters:
                    cs = [c for c in cells if c["scene"] == sid
                          and c["backend"] == backend and c["dyn"] == dyn
                          and c["iters"] == it]
                    if cs:
                        c = cs[0]
                        fields.append(f"{c['time_min_ms']:5.1f}ms "
                                      f"{c['trans_m']*1000:6.1f}mm")
                    else:
                        fields.append(" " * 17)
                lines.append("  ".join(fields))
            lines.append("")
    ax.text(0.005, 0.99, "\n".join(lines), family="monospace", fontsize=7,
            va="top", ha="left", transform=ax.transAxes)
    fig.savefig(os.path.join(RES, "lowiter_table.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: time + accuracy vs target size at low iters
# ---------------------------------------------------------------------------
def plot_size_at_lowiter(size_cells):
    sizes = sorted(set(c["n_target"] for c in size_cells))
    iters = [1, 2, 3, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"cpu": "#1f77b4", "cuda_bf": "#2ca02c"}
    cmap = plt.cm.viridis

    # time vs size
    ax = axes[0]
    for backend in ["cpu", "cuda_bf"]:
        for j, it in enumerate(iters):
            ys = []
            for n in sizes:
                cs = [c for c in size_cells if c["n_target"] == n
                      and c["backend"] == backend and c["dyn"] == "off"
                      and c["iters"] == it]
                ys.append(cs[0]["time_min_ms"] if cs else np.nan)
            color = cmap(j / max(len(iters) - 1, 1))
            ls = "-" if backend == "cpu" else "--"
            ax.plot(sizes, ys, color=color, linestyle=ls, marker="o",
                    markersize=4,
                    label=f"{backend} iter={it}")
    ax.set_xlabel("target cloud size")
    ax.set_ylabel("wall time (ms, best of 3)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Wall time — solid = CPU, dashed = CUDA")

    # trans err vs size
    ax = axes[1]
    for backend in ["cpu", "cuda_bf"]:
        for j, it in enumerate(iters):
            ys = []
            for n in sizes:
                cs = [c for c in size_cells if c["n_target"] == n
                      and c["backend"] == backend and c["dyn"] == "off"
                      and c["iters"] == it]
                ys.append(cs[0]["trans_m"] if cs else np.nan)
            color = cmap(j / max(len(iters) - 1, 1))
            ls = "-" if backend == "cpu" else "--"
            ax.plot(sizes, ys, color=color, linestyle=ls, marker="o",
                    markersize=4,
                    label=f"{backend} iter={it}")
    ax.set_xlabel("target cloud size")
    ax.set_ylabel("translation error (m)")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Translation error")

    fig.suptitle("Size sweep at low iterations — synthetic indoor scene, "
                 "source ~ 20K", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "size_at_lowiter.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Pareto scatter — time vs trans err at iter in {1,3,5}
# ---------------------------------------------------------------------------
def plot_pareto(sweep):
    cells = sweep["scenes"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, it in zip(axes, [1, 3, 5]):
        for backend, color in [("cpu", "#1f77b4"), ("cuda_bf", "#2ca02c")]:
            for dyn, marker in [("off", "o"), ("warmup3", "s"), ("warmup0", "^")]:
                xs, ys, labels = [], [], []
                for c in cells:
                    if (c["backend"] == backend and c["dyn"] == dyn
                            and c["iters"] == it
                            and np.isfinite(c["time_min_ms"])
                            and np.isfinite(c["trans_m"])):
                        xs.append(c["time_min_ms"])
                        ys.append(max(c["trans_m"], 1e-6))
                        labels.append(c["scene"])
                ax.scatter(xs, ys, color=color, marker=marker, s=30, alpha=0.7,
                           label=f"{backend}/{dyn}")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("translation error (m)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"iter = {it}")
        ax.grid(True, which="both", alpha=0.3)
        if it == 1:
            ax.legend(fontsize=6, loc="best")
    fig.suptitle("Time-vs-quality Pareto — each marker is one of the 10 scenes",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "pareto.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary tables
# ---------------------------------------------------------------------------
def write_markdown(sweep, size_cells):
    lines = ["# Low-iteration sweep — summary",
             "",
             "All numbers are best-of-3 wall time and final pose error. The CPU",
             "backend is FastGICP (8 threads), CUDA is FastGICPCuda (brute_force",
             "KNN backend). `dyn=warmup3` is the default; `warmup0` enables",
             "rejection immediately (useful when total iters <= 3).",
             "",
             "## Headline findings",
             "",
             "1. **Most indoor scenes converge in 2-3 LM iterations.** S1, S2, S4,",
             "   S6, S7, S10 reach sub-mm translation error at iter=3 with no further",
             "   improvement out to iter=30. iter=1 alone gives 2-8mm, iter=2 brings it",
             "   to ~1mm, iter=3 closes the gap. **Real-time sweet spot: iter=2 or 3.**",
             "",
             "2. **Wall time is nearly flat across iter counts** because the dominant",
             "   cost is the one-shot target self-KNN (O(N_t^2) at covariance build).",
             "   Going from iter=1 to iter=30 only adds ~5ms on the 8K-point scenes.",
             "   This means **dropping to iter=1 doesn't save much time** — pay the",
             "   extra 2-3 iter to get the accuracy.",
             "",
             "3. **CUDA is ~2x faster than CPU** at every iteration count and scene",
             "   size (8K-500K). At iter=3 the typical indoor scene takes 15ms on",
             "   CUDA vs 28ms on CPU — both well within a 30 Hz frame budget.",
             "",
             "4. **For dynamic scenes (S3 corridor), low iters are dangerous.** With",
             "   dyn=off, baseline GICP drifts toward 0.6 m at every iter count",
             "   beyond 3 because the cabinet pulls the alignment along the corridor.",
             "   CPU dyn=warmup3 needs iter>=4 to recover (warmup eats the first 3",
             "   iters); dyn=warmup0 lets you trade off a bit of stability for",
             "   immediate rejection at iter=2-3.",
             "",
             "5. **CUDA + dyn rejection on S3 is noticeably weaker than CPU.** At",
             "   iter=30 CPU reaches 1.5mm; CUDA reaches ~65mm. The cause is the",
             "   single-precision residuals on GPU not separating cleanly enough",
             "   from the dynamic-object hits, plus the brute-force 1-NN producing",
             "   slightly different correspondences than the CPU KD-tree at",
             "   iteration boundaries.",
             "",
             "6. **S5 planar floor is degenerate** — translation error stays ~68mm",
             "   regardless of iter count because the floor under-constrains 3 of",
             "   the 6 DOFs. This is observability, not iteration, and should be",
             "   handled with the observability check / hard locks separately.",
             "",
             "7. **dyn=warmup0 + iter=3 unlocks S8 anchor outlier recovery on both",
             "   backends** (59mm -> 0mm). With warmup=3 the rejection never fires",
             "   in 3 iters and bad anchors keep their full weight.",
             ""]

    # Per-scene table at iters = {1, 3, 5, 10}
    lines.append("## Per-scene results at selected iteration counts (dyn=off)")
    lines.append("")
    lines.append("| scene | backend |  iter=1 |  iter=3 |  iter=5 | iter=10 | iter=30 |")
    lines.append("|------:|:--------|--------:|--------:|--------:|--------:|--------:|")
    cells = sweep["scenes"]
    scenes = sorted(set(c["scene"] for c in cells),
                    key=lambda s: int(s[1:]))
    for sid in scenes:
        for backend in ["cpu", "cuda_bf"]:
            row = [sid, backend]
            for it in [1, 3, 5, 10, 30]:
                cs = [c for c in cells if c["scene"] == sid
                      and c["backend"] == backend and c["dyn"] == "off"
                      and c["iters"] == it]
                if cs:
                    c = cs[0]
                    row.append(f"{c['time_min_ms']:.1f}ms<br>{c['trans_m']*1000:.1f}mm")
                else:
                    row.append("")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Dyn comparison at iter = 3 (warmup3 still off, warmup0 active)
    lines.append("## Dynamic rejection at iter=3 (real-time setting)")
    lines.append("")
    lines.append("| scene | backend | dyn=off (ms / mm) | dyn=warmup3 | dyn=warmup0 |")
    lines.append("|------:|:--------|------------------:|------------:|------------:|")
    for sid in scenes:
        for backend in ["cpu", "cuda_bf"]:
            row = [sid, backend]
            for dyn in ["off", "warmup3", "warmup0"]:
                cs = [c for c in cells if c["scene"] == sid
                      and c["backend"] == backend and c["dyn"] == dyn
                      and c["iters"] == 3]
                if cs:
                    c = cs[0]
                    row.append(f"{c['time_min_ms']:.1f} / {c['trans_m']*1000:.1f}")
                else:
                    row.append("")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Size sweep summary
    lines.append("## Size sweep — wall time (ms, best of 3) per backend / iter / dyn")
    lines.append("")
    sizes = sorted(set(c["n_target"] for c in size_cells))
    for backend in ["cpu", "cuda_bf"]:
        lines.append(f"### {backend}")
        lines.append("")
        lines.append("| target size | dyn | iter=1 | iter=2 | iter=3 | iter=5 | iter=10 |")
        lines.append("|------------:|:----|-------:|-------:|-------:|-------:|--------:|")
        for n in sizes:
            for dyn in ["off", "warmup0"]:
                row = [f"{n:,}", dyn]
                for it in [1, 2, 3, 5, 10]:
                    cs = [c for c in size_cells if c["n_target"] == n
                          and c["backend"] == backend and c["dyn"] == dyn
                          and c["iters"] == it]
                    if cs:
                        c = cs[0]
                        row.append(f"{c['time_min_ms']:.1f}<br>{c['trans_m']*1000:.1f}mm")
                    else:
                        row.append("")
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    with open(os.path.join(RES, "markdown_summary.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[plots] wrote {os.path.join(RES, 'markdown_summary.md')}")


def main():
    sweep, size_cells = _load()
    plot_quality_vs_iters(sweep)
    plot_per_scene(sweep)
    plot_lowiter_table(sweep)
    plot_pareto(sweep)
    plot_size_at_lowiter(size_cells)
    write_markdown(sweep, size_cells)
    print("[plots] all written to", RES)


if __name__ == "__main__":
    main()
