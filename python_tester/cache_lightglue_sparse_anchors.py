import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/nvme1/jiaheng/dev/_DeepSL_ICP")

from recon.anchor_quality import (
    filter_3d_ransac,
    filter_by_depth_gradient,
    filter_by_patch_brightness,
)
from recon.backend.features import (
    attach_feature_depths,
    create_matcher,
    filter_feature_set,
    filter_match_result,
)
from recon.backend.io_utils import backproject_depth_at_keypoints, sample_depth_at_keypoints

from real_data_default_compare import depth_to_points, load_caminfo


def parse_pair(text: str) -> tuple[int, int]:
    left, right = text.split("-")
    return int(left), int(right)


def load_rgb_depth(prepared_dir: Path, frame_idx: int):
    rgb = cv2.cvtColor(
        cv2.imread(str(prepared_dir / "rgb" / f"{frame_idx:06d}.png"), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    depth = cv2.imread(
        str(prepared_dir / "depth" / f"{frame_idx:06d}.png"), cv2.IMREAD_UNCHANGED
    ).astype(np.float32)
    if rgb is None or depth is None:
        raise FileNotFoundError(f"failed to load frame {frame_idx}")
    return rgb, depth


def detect_features(matcher, image_rgb, depth_m, brightness_lo, brightness_hi, brightness_patch_r, gradient_r, gradient_max):
    feat = matcher.detect(image_rgb)
    if len(feat.keypoints) == 0:
        return feat

    depths, valid = sample_depth_at_keypoints(feat.keypoints, depth_m, sample_mode="nearest")
    feat = filter_feature_set(attach_feature_depths(feat, depths), valid)
    if len(feat.keypoints) == 0:
        return feat

    feat = filter_feature_set(
        feat,
        filter_by_patch_brightness(
            image_rgb,
            feat.keypoints,
            patch_r=brightness_patch_r,
            lo=brightness_lo,
            hi=brightness_hi,
        ),
    )
    if len(feat.keypoints) == 0:
        return feat

    feat = filter_feature_set(
        feat,
        filter_by_depth_gradient(
            depth_m,
            feat.keypoints,
            radius=gradient_r,
            max_gradient=gradient_max,
        ),
    )
    return feat


def build_pair_cache(prepared_dir: Path, pair: tuple[int, int], matcher, intrinsics, pixel_stride: int, confidence_th: float,
                     brightness_lo: int, brightness_hi: int, brightness_patch_r: int, gradient_r: int, gradient_max: float,
                     ransac_th: float, ransac_min_inliers: int):
    target_idx, source_idx = pair
    rgb_target, depth_target_raw = load_rgb_depth(prepared_dir, target_idx)
    rgb_source, depth_source_raw = load_rgb_depth(prepared_dir, source_idx)
    depth_scale = intrinsics[-1]
    depth_target = depth_target_raw.astype(np.float32) / depth_scale
    depth_source = depth_source_raw.astype(np.float32) / depth_scale

    feat_target = detect_features(
        matcher, rgb_target, depth_target, brightness_lo, brightness_hi, brightness_patch_r, gradient_r, gradient_max
    )
    feat_source = detect_features(
        matcher, rgb_source, depth_source, brightness_lo, brightness_hi, brightness_patch_r, gradient_r, gradient_max
    )
    match = matcher.match(feat_source, feat_target)
    raw_matches = int(match.mkpts_a.shape[0])
    if raw_matches:
      match = filter_match_result(
          match,
          (match.depths_a > 0) & (match.depths_b > 0) & (match.confidence >= confidence_th),
      )
    used_matches = int(match.mkpts_a.shape[0])

    K = np.array(
        [[intrinsics[2], 0.0, intrinsics[4]], [0.0, intrinsics[3], intrinsics[5]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    source_anchors = np.zeros((0, 3), dtype=np.float64)
    target_anchors = np.zeros((0, 3), dtype=np.float64)
    anchor_weights = np.zeros((0,), dtype=np.float64)
    inlier_matches = 0
    if used_matches >= 3:
        source_anchors, _ = backproject_depth_at_keypoints(match.mkpts_a, depth_source, K, sample_mode="nearest")
        target_anchors, _ = backproject_depth_at_keypoints(match.mkpts_b, depth_target, K, sample_mode="nearest")
        inlier_mask = filter_3d_ransac(
            source_anchors,
            target_anchors,
            match.confidence,
            threshold=ransac_th,
            min_inliers=ransac_min_inliers,
        )
        source_anchors = source_anchors[inlier_mask].astype(np.float64)
        target_anchors = target_anchors[inlier_mask].astype(np.float64)
        anchor_weights = match.confidence[inlier_mask].astype(np.float64)
        inlier_matches = int(np.count_nonzero(inlier_mask))

    target_cloud = depth_to_points(depth_target_raw, intrinsics, pixel_stride)
    source_cloud = depth_to_points(depth_source_raw, intrinsics, pixel_stride)
    return {
        "target_frame": target_idx,
        "source_frame": source_idx,
        "target_cloud": target_cloud,
        "source_cloud": source_cloud,
        "source_anchors": source_anchors,
        "target_anchors": target_anchors,
        "anchor_weights": anchor_weights,
        "raw_matches": raw_matches,
        "used_matches": used_matches,
        "inlier_matches": inlier_matches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="frame pairs like 30-31 30-40")
    parser.add_argument("--pixel-stride", type=int, default=10)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--confidence-th", type=float, default=0.2)
    parser.add_argument("--brightness-lo", type=int, default=20)
    parser.add_argument("--brightness-hi", type=int, default=240)
    parser.add_argument("--brightness-patch-r", type=int, default=7)
    parser.add_argument("--depth-gradient-r", type=int, default=3)
    parser.add_argument("--depth-gradient-max", type=float, default=0.3)
    parser.add_argument("--ransac-th", type=float, default=0.05)
    parser.add_argument("--ransac-min-inliers", type=int, default=6)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = load_caminfo(prepared_dir / "caminfo.txt")
    matcher = create_matcher(
        "lightglue",
        max_keypoints=args.max_keypoints,
        match_ratio_th=0.75,
        min_match_confidence=args.confidence_th,
    )

    summary = []
    for pair_text in args.pairs:
        pair = parse_pair(pair_text)
        cache = build_pair_cache(
            prepared_dir,
            pair,
            matcher,
            intrinsics,
            args.pixel_stride,
            args.confidence_th,
            args.brightness_lo,
            args.brightness_hi,
            args.brightness_patch_r,
            args.depth_gradient_r,
            args.depth_gradient_max,
            args.ransac_th,
            args.ransac_min_inliers,
        )
        stem = f"{pair[0]:06d}_{pair[1]:06d}"
        np.savez_compressed(out_dir / f"{stem}.npz", **cache)
        summary.append(
            {
                "pair": pair_text,
                "raw_matches": cache["raw_matches"],
                "used_matches": cache["used_matches"],
                "inlier_matches": cache["inlier_matches"],
                "target_cloud_size": int(cache["target_cloud"].shape[0]),
                "source_cloud_size": int(cache["source_cloud"].shape[0]),
            }
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
