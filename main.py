"""Default entry point: FOREST-aligned ten-case OpenXLA experiment."""

import argparse
from pathlib import Path
from runner import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New directory; existing directories are never overwritten.",
    )
    parser.add_argument("--forest-root", type=Path, default=Path.home() / "forest")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument(
        "--no-gpu-trace",
        action="store_true",
        help="Only uninstrumented standard do_bench, including the required XLA completion barrier.",
    )
    args = parser.parse_args()
    run(args.output_dir, args.forest_root, args.timeout_s, not args.no_gpu_trace)
