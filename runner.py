"""Isolated cases, bounded workers, validated results and both timing reports."""

from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
CASES = (
    "attention",
    "gqa-attention",
    "layernorm-matmul",
    "rms-ffn-swiglu",
    "efficient-vit",
)


def write_reports(directory):
    rows = []
    for b in (1, 16):
        for name in CASES:
            d = directory / name / f"B{b}"
            if not (d / "settings.json").exists():
                continue
            row = {"workload": name, "B": b}
            for source, file in (
                ("xla", "event-result.json"),
                ("xla_gpu_trace", "result.json"),
            ):
                p = d / file
                if not p.exists():
                    row[source] = {"status": "missing"}
                    continue
                row[source] = json.loads(p.read_text())
            rows.append(row)
    (directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for file, metric in (
        ("tflops.csv", "effective_tflops"),
        ("latency.csv", "latency_ms"),
    ):
        with (directory / file).open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["workload", "B", "xla", "xla_gpu_trace"],
                lineterminator="\n",
            )
            w.writeheader()
            for row in rows:
                output = {k: row[k] for k in ("workload", "B")}
                for source in ("xla", "xla_gpu_trace"):
                    r = row[source]
                    if r["status"] == "ok":
                        output[source] = (
                            r.get(metric)
                            if metric != "latency_ms"
                            else r.get(metric, {}).get("median")
                        )
                w.writerow(output)
    for row in rows:
        print(
            row["workload"],
            row["B"],
            {
                s: round(row[s].get("effective_tflops", 0), 4)
                if row[s]["status"] == "ok"
                else row[s]["status"]
                for s in ("xla", "xla_gpu_trace")
            },
            flush=True,
        )
    return rows


def run(
    output_dir=None,
    forest_root=Path.home() / "forest",
    timeout_s=600,
    gpu_trace=True,
):
    directory = (
        Path(output_dir)
        if output_dir
        else ROOT
        / "outputs"
        / (
            "forest-aligned-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    for file in ("main.py", "runner.py", "pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / file, directory / file)
    shutil.copytree(
        ROOT / "scripts",
        directory / "scripts",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        ROOT / "kernels",
        directory / "kernels",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    with (directory / "prepare.log").open("w") as log:
        subprocess.run(
            [
                str(forest_root / "benchmarks/.venv/bin/python"),
                str(ROOT / "scripts/prepare_forest_inputs.py"),
                "--forest-root",
                str(forest_root),
                "--output-dir",
                str(directory),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    for b in (1, 16):
        for name in CASES:
            d = directory / name / f"B{b}"
            for profiled in [True, False] if gpu_trace else [False]:
                file = "result.json" if profiled else "event-result.json"
                cmd = [
                    str(ROOT / "scripts/xla_python.sh"),
                    str(ROOT / "scripts/profile_forest_aligned.py"),
                    "--directory",
                    str(d),
                    "--result-name",
                    file,
                ]
                if profiled:
                    cmd = [
                        "/software/cuda-13.0/bin/nsys",
                        "profile",
                        "--trace=cuda,nvtx",
                        "--sample=none",
                        "--cpuctxsw=none",
                        "--force-overwrite=true",
                        "-o",
                        str(d / "profile"),
                        *cmd,
                    ]
                print(
                    "START",
                    name,
                    "B=" + str(b),
                    "gpu-trace" if profiled else "do_bench",
                    flush=True,
                )
                try:
                    with (d / ("worker.log" if profiled else "event-worker.log")).open(
                        "w"
                    ) as log:
                        subprocess.run(
                            ["timeout", "--kill-after=15s", str(timeout_s) + "s", *cmd],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={
                                **os.environ,
                                "FOREST_ROOT": str(forest_root.resolve()),
                            },
                            check=True,
                        )
                    if profiled:
                        with (d / "export.log").open("w") as log:
                            subprocess.run(
                                [
                                    "/software/cuda-13.0/bin/nsys",
                                    "export",
                                    "--type",
                                    "sqlite",
                                    "--force-overwrite=true",
                                    "--output",
                                    str(d / "profile.sqlite"),
                                    str(d / "profile.nsys-rep"),
                                ],
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                        subprocess.run(
                            [
                                sys.executable,
                                str(ROOT / "scripts/summarize_gpu_trace.py"),
                                str(d),
                            ],
                            check=True,
                        )
                except Exception as e:
                    p = d / file
                    r = (
                        json.loads(p.read_text())
                        if p.exists()
                        else {"settings": json.loads((d / "settings.json").read_text())}
                    )
                    r.update(
                        status="timeout"
                        if isinstance(e, subprocess.CalledProcessError)
                        and e.returncode in (124, 137)
                        else "failed",
                        orchestration_error=str(e),
                    )
                    p.write_text(json.dumps(r, indent=2) + "\n")
    rows = write_reports(directory)
    (directory / "elapsed.json").write_text(
        json.dumps(
            {"elapsed_s": time.monotonic() - started, "timeout_s": timeout_s}, indent=2
        )
        + "\n"
    )
    print("Results:", directory, flush=True)
    return rows
