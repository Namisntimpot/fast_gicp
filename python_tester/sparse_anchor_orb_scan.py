import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real_data_default_compare import load_caminfo, depth_to_points


def build_cloud(depth: np.ndarray, intrinsics, pixel_stride: int) -> np.ndarray:
    return depth_to_points(depth, intrinsics, pixel_stride)


def backproject_pixel(
    depth: np.ndarray,
    u: float,
    v: float,
    intrinsics,
) -> np.ndarray | None:
    width, height, fx, fy, cx, cy, depth_trunc, depth_scale = intrinsics
    uu = int(round(u))
    vv = int(round(v))
    if uu < 0 or uu >= width or vv < 0 or vv >= height:
      return None

    depth_raw = float(depth[vv, uu])
    if depth_raw <= 0.0:
      return None

    z = depth_raw / depth_scale
    if z <= 0.0 or z >= depth_trunc:
      return None

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def extract_orb_anchors(
    rgb_target: np.ndarray,
    rgb_source: np.ndarray,
    depth_target: np.ndarray,
    depth_source: np.ndarray,
    intrinsics,
    max_matches: int,
) -> tuple[np.ndarray, np.ndarray]:
    orb = cv2.ORB_create(nfeatures=max_matches * 4)
    keypoints_target, descriptors_target = orb.detectAndCompute(rgb_target, None)
    keypoints_source, descriptors_source = orb.detectAndCompute(rgb_source, None)
    if descriptors_target is None or descriptors_source is None:
      return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_source, descriptors_target)
    matches = sorted(matches, key=lambda match: match.distance)

    source_points = []
    target_points = []
    for match in matches:
      source_kp = keypoints_source[match.queryIdx].pt
      target_kp = keypoints_target[match.trainIdx].pt
      source_point = backproject_pixel(depth_source, source_kp[0], source_kp[1], intrinsics)
      target_point = backproject_pixel(depth_target, target_kp[0], target_kp[1], intrinsics)
      if source_point is None or target_point is None:
        continue

      source_points.append(source_point)
      target_points.append(target_point)
      if len(source_points) >= max_matches:
        break

    if not source_points:
      return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    return np.asarray(source_points), np.asarray(target_points)


def filter_anchor_inliers_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    ransac_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(source_points) < 4:
      return source_points, target_points

    retval, _, inliers = cv2.estimateAffine3D(
        source_points.astype(np.float32),
        target_points.astype(np.float32),
        ransacThreshold=ransac_threshold,
        confidence=0.99,
    )
    if retval == 0 or inliers is None:
      return source_points, target_points

    mask = inliers.ravel().astype(bool)
    return source_points[mask], target_points[mask]


def rotation_angle(matrix: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))


def transform_delta(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(reference) @ estimate
    return float(np.linalg.norm(delta[:3, 3])), rotation_angle(delta)


def run_case(
    pygicp,
    target_cloud: np.ndarray,
    source_cloud: np.ndarray,
    source_anchors: np.ndarray | None,
    target_anchors: np.ndarray | None,
    balance_mode: str | None,
    objective_weight: float,
    auto_balance_max: float | None,
) -> tuple[np.ndarray, dict]:
    reg = pygicp.FastGICP()
    reg.set_input_target(target_cloud)
    reg.set_input_source(source_cloud)
    reg.set_max_correspondence_distance(2.0)

    if source_anchors is not None and len(source_anchors) > 0:
      reg.set_sparse_anchor_correspondences(source_anchors, target_anchors)
      reg.set_sparse_anchor_config(
          {
              "balance_mode": balance_mode,
              "objective_weight": objective_weight,
              **({"auto_balance_max": auto_balance_max} if auto_balance_max is not None else {}),
          }
      )

    transform = np.asarray(reg.align(), dtype=np.float64)
    report = reg.get_alignment_quality_report()
    return transform, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--target-frame", type=int, default=30)
    parser.add_argument("--source-frame", type=int, default=40)
    parser.add_argument("--pixel-stride", type=int, default=10)
    parser.add_argument("--max-matches", type=int, default=300)
    parser.add_argument("--objective-weight", type=float, default=1.0)
    parser.add_argument("--auto-balance-max", type=float, default=None)
    parser.add_argument("--anchor-ransac-threshold", type=float, default=0.05)
    args = parser.parse_args()

    import importlib.util

    spec = importlib.util.spec_from_file_location("pygicp", args.module_path)
    pygicp = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(pygicp)

    prepared_dir = Path(args.prepared_dir)
    intrinsics = load_caminfo(prepared_dir / "caminfo.txt")

    rgb_target = cv2.imread(str(prepared_dir / "rgb" / f"{args.target_frame:06d}.png"), cv2.IMREAD_COLOR)
    rgb_source = cv2.imread(str(prepared_dir / "rgb" / f"{args.source_frame:06d}.png"), cv2.IMREAD_COLOR)
    depth_target = cv2.imread(str(prepared_dir / "depth" / f"{args.target_frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    depth_source = cv2.imread(str(prepared_dir / "depth" / f"{args.source_frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if rgb_target is None or rgb_source is None or depth_target is None or depth_source is None:
      raise FileNotFoundError("failed to load rgb/depth frames")

    target_cloud = build_cloud(depth_target, intrinsics, args.pixel_stride)
    source_cloud = build_cloud(depth_source, intrinsics, args.pixel_stride)
    source_anchors, target_anchors = extract_orb_anchors(
        rgb_target, rgb_source, depth_target, depth_source, intrinsics, args.max_matches
    )
    source_anchors, target_anchors = filter_anchor_inliers_ransac(
        source_anchors, target_anchors, args.anchor_ransac_threshold
    )
    if len(source_anchors) == 0:
      raise RuntimeError("failed to build sparse anchors from ORB matches")

    baseline_transform, baseline_report = run_case(
        pygicp, target_cloud, source_cloud, None, None, None, args.objective_weight, args.auto_balance_max
    )

    results = {
        "meta": {
            "target_frame": args.target_frame,
            "source_frame": args.source_frame,
            "pixel_stride": args.pixel_stride,
            "target_cloud_size": int(target_cloud.shape[0]),
            "source_cloud_size": int(source_cloud.shape[0]),
            "anchor_count": int(source_anchors.shape[0]),
        },
        "baseline": {
            "fitness": float(baseline_report["fitness_score"]),
            "translation": baseline_transform[:3, 3].tolist(),
            "rotation_rad": rotation_angle(baseline_transform),
        },
        "anchor_cases": {},
    }

    for balance_mode in ["NONE", "AUTO", "BY_COUNT", "BY_HESSIAN_TRACE"]:
      transform, report = run_case(
          pygicp,
          target_cloud,
          source_cloud,
          source_anchors,
          target_anchors,
          balance_mode,
          args.objective_weight,
          args.auto_balance_max,
      )
      delta_translation, delta_rotation = transform_delta(baseline_transform, transform)
      results["anchor_cases"][balance_mode] = {
          "fitness": float(report["fitness_score"]),
          "delta_vs_baseline_translation": delta_translation,
          "delta_vs_baseline_rotation_rad": delta_rotation,
          "anchor_effective_scale": float(report["anchor_effective_scale"]),
          "anchor_auto_balance_factor": float(report["anchor_auto_balance_factor"]),
          "geometry_hessian_trace": float(report["geometry_hessian_trace"]),
          "anchor_hessian_trace": float(report["anchor_hessian_trace"]),
          "anchor_scaled_cost": float(report["anchor_scaled_cost"]),
          "anchor_raw_cost": float(report["anchor_raw_cost"]),
          "normalized_cost_per_match": float(report["normalized_cost_per_match"]),
          "anchor_mean_residual": float(report["anchor_mean_residual"]),
          "anchor_p95_residual": float(report["anchor_p95_residual"]),
          "translation": transform[:3, 3].tolist(),
      }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
