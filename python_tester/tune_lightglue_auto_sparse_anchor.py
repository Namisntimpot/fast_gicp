import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


def load_module(module_path: str):
    spec = importlib.util.spec_from_file_location("pygicp", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rotation_angle(matrix: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))


def transform_delta(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(reference) @ estimate
    return float(np.linalg.norm(delta[:3, 3])), rotation_angle(delta)


def run_case(pygicp, cache, balance_mode: str, config_overrides: dict | None):
    reg = pygicp.FastGICP()
    reg.set_input_target(cache["target_cloud"])
    reg.set_input_source(cache["source_cloud"])
    reg.set_max_correspondence_distance(2.0)
    if balance_mode != "BASELINE":
        reg.set_sparse_anchor_correspondences(
            cache["source_anchors"],
            cache["target_anchors"],
            cache["anchor_weights"],
            None,
        )
        config = {"balance_mode": balance_mode}
        if config_overrides:
            config.update(config_overrides)
        reg.set_sparse_anchor_config(config)
    transform = np.asarray(reg.align(), dtype=np.float64)
    report = reg.get_alignment_quality_report()
    return transform, report


def load_caches(cache_dir: Path):
    caches = []
    for path in sorted(cache_dir.glob("*.npz")):
        data = np.load(path)
        caches.append({key: data[key] for key in data.files} | {"path": str(path)})
    return caches


def pair_metric(reference: np.ndarray, estimate: np.ndarray) -> float:
    t_delta, r_delta = transform_delta(reference, estimate)
    return t_delta + 0.5 * r_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--objective-weights", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--auto-balance-maxes", type=float, nargs="+", default=[6.0, 10.0, 16.0, 24.0])
    parser.add_argument("--auto-count-powers", type=float, nargs="+", default=[0.25, 0.35, 0.5])
    parser.add_argument("--auto-ambiguity-floors", type=float, nargs="+", default=[0.15, 0.25, 0.35])
    parser.add_argument("--auto-ambiguity-gains", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    args = parser.parse_args()

    pygicp = load_module(args.module_path)
    caches = load_caches(Path(args.cache_dir))

    pair_refs = []
    for cache in caches:
        baseline_transform, baseline_report = run_case(pygicp, cache, "BASELINE", None)
        none_transform, none_report = run_case(
            pygicp, cache, "NONE", {"objective_weight": 1.0}
        )
        count_transform, count_report = run_case(
            pygicp, cache, "BY_COUNT", {"objective_weight": 1.0}
        )
        pair_refs.append(
            {
                "cache": cache,
                "baseline_transform": baseline_transform,
                "baseline_report": baseline_report,
                "none_metric": pair_metric(baseline_transform, none_transform),
                "count_metric": pair_metric(baseline_transform, count_transform),
                "none_anchor_residual": float(none_report["anchor_mean_residual"]),
                "count_anchor_residual": float(count_report["anchor_mean_residual"]),
            }
        )

    best = None
    results = []
    grid = itertools.product(
        args.objective_weights,
        args.auto_balance_maxes,
        args.auto_count_powers,
        args.auto_ambiguity_floors,
        args.auto_ambiguity_gains,
    )
    for objective_weight, auto_balance_max, auto_count_power, auto_ambiguity_floor, auto_ambiguity_gain in grid:
        cfg = {
            "objective_weight": objective_weight,
            "auto_balance_max": auto_balance_max,
            "auto_count_power": auto_count_power,
            "auto_ambiguity_floor": auto_ambiguity_floor,
            "auto_ambiguity_gain": auto_ambiguity_gain,
        }
        pair_stats = []
        score = 0.0
        for ref in pair_refs:
            auto_transform, auto_report = run_case(pygicp, ref["cache"], "AUTO", cfg)
            auto_metric = pair_metric(ref["baseline_transform"], auto_transform)
            target_metric = np.sqrt(max(ref["none_metric"], 1e-6) * max(ref["count_metric"], 1e-6))
            score += abs(np.log(max(auto_metric, 1e-6) / target_metric))
            score += 0.5 * float(auto_report["anchor_mean_residual"])
            score += 0.05 * float(auto_report["normalized_cost_per_match"])
            pair_stats.append(
                {
                    "path": ref["cache"]["path"],
                    "auto_metric": auto_metric,
                    "target_metric": target_metric,
                    "anchor_mean_residual": float(auto_report["anchor_mean_residual"]),
                    "anchor_effective_scale": float(auto_report["anchor_effective_scale"]),
                    "auto_balance_factor": float(auto_report["anchor_auto_balance_factor"]),
                }
            )
        result = {"config": cfg, "score": score, "pairs": pair_stats}
        results.append(result)
        if best is None or score < best["score"]:
            best = result

    output = {
        "best": best,
        "top5": sorted(results, key=lambda item: item["score"])[:5],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
