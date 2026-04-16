from __future__ import annotations

import argparse
from pathlib import Path

from runner import BenchmarkConfig, run_benchmark


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run torch vs torch-xla performance benchmarks")
    parser.add_argument(
        "--backend",
        dest="backends",
        action="append",
        choices=["torch_flash", "torch_cudnn", "torch_xla"],
        help="Backend to run. Can be passed multiple times. Default: both.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Tensor dtype used by all benchmark cases.",
    )
    parser.add_argument(
        "--timer",
        default="triton_do_bench",
        choices=["triton_do_bench", "perf_counter"],
        help=(
            "Timing backend. 'triton_do_bench' matches triton_eval's timer on CUDA; "
            "'perf_counter' uses iteration-averaged wall time."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        default="on",
        choices=["on", "off"],
        help="Enable/disable torch.compile for eligible torch CUDA cases.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for eligible torch CUDA cases.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per case.")
    parser.add_argument("--rep", type=int, default=30, help="Measured iterations per case.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save benchmark CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backends = tuple(args.backends) if args.backends else ("torch_flash", "torch_cudnn", "torch_xla")
    config = BenchmarkConfig(
        backends=backends,
        dtype=args.dtype,
        timer=args.timer,
        torch_compile=args.torch_compile == "on",
        torch_compile_mode=args.torch_compile_mode,
        warmup=args.warmup,
        rep=args.rep,
        output_dir=Path(args.output_dir),
    )
    artifacts = run_benchmark(config)
    print(artifacts.dataframe)
    print(f"Saved CSV: {artifacts.csv_path}")


if __name__ == "__main__":
    main()
