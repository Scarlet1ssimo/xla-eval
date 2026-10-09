"""GPU latency from all executable launches, excluding cache flush/host submission."""

import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np


def summarize(directory):
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "ok":
        return result
    conn = sqlite3.connect(directory / "profile.sqlite")
    tables = {
        r[0] for r in conn.execute("select name from sqlite_master where type='table'")
    }
    ranges = conn.execute(
        "select start,end from NVTX_EVENTS where text='FOREST_XLA_EXECUTE' and end is not null order by start"
    ).fetchall()
    n = len(result["raw_cuda_event_ms"])
    ranges = ranges[-n:]
    if len(ranges) != n or not n:
        raise RuntimeError("measurement range count mismatch")
    samples = []
    for a, b in ranges:
        activities = []
        counts = {}
        for table in (
            "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        ):
            if table not in tables:
                continue
            events = conn.execute(
                "select start,end,streamId,contextId from "
                + table
                + " where start>=? and end<=? order by start",
                (a, b),
            ).fetchall()
            counts[table] = len(events)
            activities.extend(events)
        if not activities:
            raise RuntimeError("no GPU work in an executable measurement")
        first = min(e[0] for e in activities)
        last = max(e[1] for e in activities)
        samples.append(
            dict(
                gpu_span_ms=(last - first) / 1e6,
                host_range_ms=(b - a) / 1e6,
                counts=counts,
                streams=sorted({e[2] for e in activities}),
                contexts=sorted({e[3] for e in activities}),
            )
        )
    quantiles = np.quantile([s["gpu_span_ms"] for s in samples], [0.5, 0.2, 0.8])
    result["submission_inclusive_latency_ms"] = result.pop("latency_ms")
    result["submission_inclusive_tflops"] = result.pop("effective_tflops")
    result["latency_ms"] = dict(zip(("median", "p20", "p80"), map(float, quantiles)))
    result["effective_tflops"] = result["settings"]["effective_flops"] / (
        quantiles[0] * 1e9
    )
    result["gpu_trace_samples"] = samples
    result["gpu_timer"] = (
        "Nsight Systems full executable GPU span, first GPU activity start to last end, including all kernels/graphs/device copies/memsets; excludes host submission and L2 flush"
    )
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        result["settings"]["case"],
        result["settings"]["B"],
        round(result["effective_tflops"], 3),
        result["latency_ms"],
        len(samples),
        flush=True,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    summarize(a.directory)
