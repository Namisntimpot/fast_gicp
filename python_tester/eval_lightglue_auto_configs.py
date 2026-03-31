import argparse
import importlib.util
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


def delta(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    diff = np.linalg.inv(reference) @ estimate
    return float(np.linalg.norm(diff[:3, 3])), rotation_angle(diff)


def run_case(pygicp, cache, config: dict | None):
    reg = pygicp.FastGICP()
    reg.set_input_target(cache["target_cloud"])
    reg.set_input_source(cache["source_cloud"])
    reg.set_max_correspondence_distance(2.0)
    if config is not None:
        reg.set_sparse_anchor_correspondences(
            cache["source_anchors"], cache["target_anchors"], cache["anchor_weights"], None
        )
        sparse_config = {"balance_mode": "AUTO"}
        sparse_config.update(config)
        reg.set_sparse_anchor_config(sparse_config)
    transform = np.asarray(reg.align(), dtype=np.float64)
    return transform, reg.get_alignment_quality_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--module-path", required=True)
    args = parser.parse_args()

    pygicp = load_module(args.module_path)
    configs = {
        "current_default": {
            "objective_weight": 1.0,
            "auto_balance_max": 1000.0,
            "auto_count_power": 0.5,
            "auto_ambiguity_floor": 0.25,
            "auto_ambiguity_gain": 1.0,
        },
        "conservative": {
            "objective_weight": 1.0,
            "auto_balance_max": 6.0,
            "auto_count_power": 0.25,
            "auto_ambiguity_floor": 0.15,
            "auto_ambiguity_gain": 0.5,
        },
        "medium": {
            "objective_weight": 1.5,
            "auto_balance_max": 10.0,
            "auto_count_power": 0.35,
            "auto_ambiguity_floor": 0.15,
            "auto_ambiguity_gain": 0.75,
        },
    }

    results = {}
    for name, config in configs.items():
        pair_results = []
        for path in sorted(Path(args.cache_dir).glob("*.npz")):
            data = np.load(path)
            cache = {key: data[key] for key in data.files}
            baseline_transform, baseline_report = run_case(pygicp, cache, None)
            auto_transform, auto_report = run_case(pygicp, cache, config)
            delta_t, delta_r = delta(baseline_transform, auto_transform)
            pair_results.append(
                {
                    "pair": path.name,
                    "anchor_count": int(cache["source_anchors"].shape[0]),
                    "delta_translation": delta_t,
                    "delta_rotation_rad": delta_r,
                    "anchor_effective_scale": float(auto_report["anchor_effective_scale"]),
                    "anchor_auto_balance_factor": float(auto_report["anchor_auto_balance_factor"]),
                    "anchor_mean_residual": float(auto_report["anchor_mean_residual"]),
                    "fitness": float(auto_report["fitness_score"]),
                    "baseline_fitness": float(baseline_report["fitness_score"]),
                }
            )
        results[name] = pair_results

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
