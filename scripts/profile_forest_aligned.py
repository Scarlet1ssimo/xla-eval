"""Profile cached OpenXLA executables against FOREST's ten-case protocol."""

from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch_xla.utils.dlpack import from_dlpack, from_xla_cuda_to_cuda
import triton.testing

ROOT = Path(__file__).resolve().parents[1]
FOREST = Path(os.environ.get("FOREST_ROOT", str(Path.home() / "forest")))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cooling = load_file("cooldown", FOREST / "benchmarks/src/forest_benchmarks/cooldown.py")
numerics = load_file("numerics", FOREST / "codegen/src/forest/codegen/numerics.py")

sys.path.insert(0, str(ROOT))
from kernels.flashattn.torch import flashattn_pytorch
from kernels.gqa.torch import gqa_pytorch
from kernels.layernorm_matmul.torch import layernorm_matmul_pytorch
from kernels.rmsnorm_swiglu.torch import rmsnorm_swiglu_pytorch_ref
from kernels.efficientvit.torch import efficientvit_pytorch

FUNCTIONS = {
    "attention": flashattn_pytorch,
    "gqa-attention": gqa_pytorch,
    "layernorm-matmul": layernorm_matmul_pytorch,
    "rms-ffn-swiglu": rmsnorm_swiglu_pytorch_ref,
    "efficient-vit": efficientvit_pytorch,
}


def profile_case(directory, *, trace=False, result_name="result.json"):
    torch.set_grad_enabled(False)
    started = time.monotonic()
    settings = json.loads((directory / "settings.json").read_text())
    result = dict(
        settings=settings,
        status="running",
        runtime=dict(
            torch=torch.__version__,
            torch_xla=torch_xla.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(0),
            xla_repo_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            xla_flags=os.environ.get("XLA_FLAGS"),
            timer="triton.testing.do_bench with XLA completion barrier; GPU trace reports executable span; flush L2",
        ),
    )

    def persist():
        result["elapsed_s"] = time.monotonic() - started
        (directory / result_name).write_text(json.dumps(result, indent=2) + "\n")

    try:
        fixtures = np.load(directory / "inputs.npz")
        cuda_inputs = [
            torch.from_numpy(fixtures[f"input{i}"].copy()).cuda()
            for i in range(len(fixtures.files))
        ]
        inputs = [from_dlpack(t) for t in cuda_inputs]
        xm.mark_step()
        xm.wait_device_ops()
        met.clear_all()
        t = time.monotonic()
        output = FUNCTIONS[settings["case"]](*inputs)
        (directory / "input-hlo.txt").write_text(
            torch_xla._XLAC._get_xla_tensors_hlo([output])
        )
        _, graph_inputs = torch_xla._XLAC._get_tensors_xla_device_data_node([output])
        graph_hash = torch_xla._XLAC._get_graph_hash([output])
        torch_xla._XLAC._xla_warm_up_cache([output], [])
        result["compile_s"] = time.monotonic() - t
        result["graph_hash"] = graph_hash.hex()
        result["compile_metrics"] = met.metrics_report()
        result["fallback_ops"] = [
            n for n in met.counter_names() if n.startswith("aten::")
        ]
        if result["fallback_ops"]:
            raise RuntimeError(
                "XLA CPU fallback detected: " + str(result["fallback_ops"])
            )
        holder = []

        def run():
            holder[:] = torch_xla._XLAC._run_cached_graph(graph_hash, graph_inputs)
            xm.wait_device_ops()

        run()
        xm.wait_device_ops()
        actual = from_xla_cuda_to_cuda(holder[0])
        reference = torch.from_numpy(np.load(directory / "reference.npy")).cuda()
        result["actual_abi"] = {
            "inputs": [
                dict(
                    shape=list(t.shape),
                    strides=list(t.stride()),
                    contiguous=t.is_contiguous(),
                    dtype=str(t.dtype),
                )
                for t in cuda_inputs
            ],
            "output": dict(
                shape=list(actual.shape),
                strides=list(actual.stride()),
                contiguous=actual.is_contiguous(),
                dtype=str(actual.dtype),
            ),
        }
        if not actual.is_contiguous():
            raise RuntimeError(
                "XLA output layout differs from the contiguous FOREST ABI"
            )
        check = numerics.numerical_metrics(actual, reference)
        check["passed"] = (
            check["output_nonfinite"] == 0
            and check["reference_nonfinite"] == 0
            and check["relative_l2"] < settings["relative_l2_limit"]
            and check["normalized_max_abs"] < settings["normalized_max_limit"]
        )
        result["validation"] = check
        if not check["passed"]:
            result["status"] = "incorrect"
            persist()
            return result
        stream_ptr = torch_xla._XLAC._get_stream_for_cuda_device(0)
        stream = torch.cuda.ExternalStream(stream_ptr if stream_ptr else 1)
        result["stream_ptr"] = stream_ptr
        for _ in range(settings["warmup_calls"]):
            run()
        xm.wait_device_ops()
        torch.cuda.synchronize()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        if not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        result["cooldown"] = {}
        persist()
        cooling.cooldown(uuid, settings["cooldown_s"], report=result["cooldown"])
        if trace:
            from torch.profiler import profile, ProfilerActivity

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                with torch.cuda.stream(stream):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    run()
                    end.record()
                    end.synchronize()
                    result["trace_event_ms"] = start.elapsed_time(end)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(directory / "cuda-trace.json"))

        def marked_run():
            # Finish do_bench's L2 flush before XLA launches on its own stream.
            torch.cuda.current_stream().synchronize()
            torch.cuda.nvtx.range_push("FOREST_XLA_EXECUTE")
            try:
                run()
            finally:
                torch.cuda.nvtx.range_pop()

        original_summary = triton.testing._summarize_statistics

        def collect_samples(times, quantiles, return_mode):
            result["raw_cuda_event_ms"] = [float(t) for t in times]
            return original_summary(times, quantiles, return_mode)

        triton.testing._summarize_statistics = collect_samples
        torch.cuda.nvtx.range_push("FOREST_MEASUREMENT")
        try:
            with torch.cuda.stream(stream):
                median, p20, p80 = triton.testing.do_bench(
                    marked_run,
                    rep=settings["measurement_ms"],
                    quantiles=[0.5, 0.2, 0.8],
                )
        finally:
            torch.cuda.nvtx.range_pop()
            triton.testing._summarize_statistics = original_summary
        xm.wait_device_ops()
        torch.cuda.synchronize()
        result["latency_ms"] = dict(median=median, p20=p20, p80=p80)
        result["effective_tflops"] = settings["effective_flops"] / (median * 1e9)
        result["after_measurement"] = cooling.snapshot(uuid)
        actual = from_xla_cuda_to_cuda(holder[0])
        post = numerics.numerical_metrics(actual, reference)
        post["passed"] = (
            post["output_nonfinite"] == 0
            and post["reference_nonfinite"] == 0
            and post["relative_l2"] < settings["relative_l2_limit"]
            and post["normalized_max_abs"] < settings["normalized_max_limit"]
        )
        result["post_validation"] = post
        result["runtime_metrics"] = met.metrics_report()
        result["status"] = "ok" if post["passed"] else "incorrect"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
    persist()
    print(
        settings["case"],
        settings["B"],
        result["status"],
        result.get("effective_tflops"),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--result-name", default="result.json")
    args = parser.parse_args()
    profile_case(args.directory, trace=args.trace, result_name=args.result_name)
