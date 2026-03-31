import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_metric(output: str, key: str) -> float:
    prefix = f"{key}="
    for line in output.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):])
    raise ValueError(f"missing {key} in output:\n{output}")


def run_pair(
    compare_script: Path,
    prepared_dir: Path,
    current_module: Path,
    main_module: Path,
    target_frame: int,
    source_frame: int,
    pixel_stride: int,
) -> tuple[float, float]:
    result = subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--prepared-dir",
            str(prepared_dir),
            "--current-module",
            str(current_module),
            "--main-module",
            str(main_module),
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
    translation = parse_metric(result.stdout, "delta_translation")
    rotation = parse_metric(result.stdout, "delta_rotation_rad")
    return translation, rotation


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--current-module", required=True)
    parser.add_argument("--main-module", required=True)
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 10, 20])
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--end-frame", type=int, default=120)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--pixel-stride", type=int, default=1)
    args = parser.parse_args()

    compare_script = Path(__file__).resolve().with_name("real_data_default_compare.py")
    prepared_dir = Path(args.prepared_dir)
    current_module = Path(args.current_module)
    main_module = Path(args.main_module)

    results = {}
    for interval in args.intervals:
      translations = []
      rotations = []
      frame_pairs = []
      last_target = args.end_frame - interval
      for target_frame in range(args.start_frame, last_target + 1, args.stride):
          source_frame = target_frame + interval
          translation, rotation = run_pair(
              compare_script,
              prepared_dir,
              current_module,
              main_module,
              target_frame,
              source_frame,
              args.pixel_stride,
          )
          translations.append(translation)
          rotations.append(rotation)
          frame_pairs.append(
              {
                  "target": target_frame,
                  "source": source_frame,
                  "delta_translation": translation,
                  "delta_rotation_rad": rotation,
              }
          )

      results[str(interval)] = {
          "translation": summarize(translations),
          "rotation_rad": summarize(rotations),
          "pairs": frame_pairs,
      }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
