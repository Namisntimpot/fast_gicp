import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def load_module(module_path: str):
    spec = importlib.util.spec_from_file_location("pygicp", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_caminfo(caminfo_path: Path):
    with caminfo_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]

    values = lines[-1].split()
    width = int(values[0])
    height = int(values[1])
    fx = float(values[2])
    fy = float(values[3])
    cx = float(values[4])
    cy = float(values[5])
    depth_trunc = float(values[6])
    depth_scale = float(values[8])
    return width, height, fx, fy, cx, cy, depth_trunc, depth_scale


def depth_to_points(depth: np.ndarray, intrinsics, pixel_stride: int):
    width, height, fx, fy, cx, cy, depth_trunc, depth_scale = intrinsics
    if depth.shape != (height, width):
        raise ValueError(f"unexpected depth shape {depth.shape}, expected {(height, width)}")

    z = depth.astype(np.float32) / depth_scale
    if pixel_stride > 1:
        sampled = np.zeros_like(z, dtype=bool)
        sampled[::pixel_stride, ::pixel_stride] = True
    else:
        sampled = np.ones_like(z, dtype=bool)
    valid = (z > 0.0) & (z < depth_trunc)
    valid &= sampled
    v_coords, u_coords = np.nonzero(valid)
    z = z[valid]
    x = (u_coords.astype(np.float32) - cx) * z / fx
    y = (v_coords.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float64)


def pose_delta(reference: np.ndarray, estimated: np.ndarray):
    delta = np.linalg.inv(reference) @ estimated
    translation = np.linalg.norm(delta[:3, 3])
    rotation = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    return translation, rotation, delta


def run_alignment(module, target_points: np.ndarray, source_points: np.ndarray):
    return np.asarray(module.align_points(target_points, source_points), dtype=np.float64)


def run_module_once(
    script_path: Path,
    module_path: str,
    prepared_dir: str,
    target_frame: int,
    source_frame: int,
    pixel_stride: int,
):
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--single-module",
            module_path,
            "--prepared-dir",
            prepared_dir,
            "--target-frame",
            str(target_frame),
            "--source-frame",
            str(source_frame),
            "--pixel-stride",
            str(pixel_stride),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return np.asarray(json.loads(result.stdout), dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--current-module")
    parser.add_argument("--main-module")
    parser.add_argument("--target-frame", type=int, default=30)
    parser.add_argument("--source-frame", type=int, default=31)
    parser.add_argument("--single-module")
    parser.add_argument("--pixel-stride", type=int, default=1)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    intrinsics = load_caminfo(prepared_dir / "caminfo.txt")

    target_depth = cv2.imread(str(prepared_dir / "depth" / f"{args.target_frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    source_depth = cv2.imread(str(prepared_dir / "depth" / f"{args.source_frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if target_depth is None or source_depth is None:
        raise FileNotFoundError("failed to load depth frames")

    target_points = depth_to_points(target_depth, intrinsics, args.pixel_stride)
    source_points = depth_to_points(source_depth, intrinsics, args.pixel_stride)

    if args.single_module:
        module = load_module(args.single_module)
        transform = run_alignment(module, target_points, source_points)
        print(json.dumps(transform.tolist()))
        return

    script_path = Path(__file__).resolve()
    current_transform = run_module_once(
        script_path, args.current_module, str(prepared_dir), args.target_frame, args.source_frame, args.pixel_stride
    )
    main_transform = run_module_once(
        script_path, args.main_module, str(prepared_dir), args.target_frame, args.source_frame, args.pixel_stride
    )
    translation, rotation, delta = pose_delta(main_transform, current_transform)

    np.set_printoptions(precision=6, suppress=True)
    print(f"target_points={target_points.shape[0]} source_points={source_points.shape[0]}")
    print("main_transform=")
    print(main_transform)
    print("current_transform=")
    print(current_transform)
    print(f"delta_translation={translation:.6f}")
    print(f"delta_rotation_rad={rotation:.6f}")
    print("delta_transform=")
    print(delta)


if __name__ == "__main__":
    main()
