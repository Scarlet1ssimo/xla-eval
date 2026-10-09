"""Run with FOREST's Python: current cases, input generator, reference and FLOPs."""

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from forest_benchmarks.abi import ProgramABI, TensorABI
from forest_benchmarks.metrics import workload_flops
from forest_benchmarks.workloads import WORKLOADS

parser = argparse.ArgumentParser()
parser.add_argument("--forest-root", type=Path, default=Path.home() / "forest")
parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
)
a = parser.parse_args()
spec = importlib.util.spec_from_file_location(
    "five", a.forest_root / "benchmarks/scripts/profile_five_workloads.py"
)
five = importlib.util.module_from_spec(spec)
spec.loader.exec_module(five)
torch.set_grad_enabled(False)
for case in five.cases():
    name, batch, problem = case["case"], case["B"], case["problem"]
    directory = a.output_dir / name / f"B{batch}"
    directory.mkdir(parents=True, exist_ok=True)
    d = problem.to_dict()["dimensions"]
    workload = problem.workload
    if name == "attention":
        shapes = [
            (d["B"], d["H"], d["M"], d["D"]),
            (d["B"], d["H"], d["N"], d["D"]),
            (d["B"], d["H"], d["N"], d["L"]),
        ]
        output = (d["B"], d["H"], d["M"], d["L"])
    elif name == "gqa-attention":
        shapes = [
            (d["B"], d["G"] * d["R"], d["M"], d["D"]),
            (d["B"], d["G"], d["N"], d["D"]),
            (d["B"], d["G"], d["N"], d["L"]),
        ]
        output = (d["B"], d["G"] * d["R"], d["M"], d["L"])
    elif name == "layernorm-matmul":
        shapes = [(d["M"], d["K"]), (d["K"], d["N"])]
        output = (d["M"], d["N"])
    elif name == "rms-ffn-swiglu":
        shapes = [
            (d["M"], d["D"]),
            (d["D"], d["K"]),
            (d["D"], d["K"]),
            (d["K"], d["N"]),
        ]
        output = (d["M"], d["N"])
    else:
        shapes = [
            (d["B"], d["N"], d["D"]),
            (d["B"], d["D"], d["N"]),
            (d["B"], d["E"], d["N"]),
        ]
        output = (d["B"], d["N"], d["E"])
    semantic = WORKLOADS[workload]

    def tensor(name, shape):
        return TensorABI(
            name, shape, shape, "f16", tuple((i,) for i in range(len(shape)))
        )

    abi = ProgramABI(
        semantic.program,
        tuple(tensor(n, s) for n, s in zip(semantic.input_names, shapes)),
        (tensor("output0", output),),
    )
    semantic.check_abi(abi)
    inputs = semantic.make_inputs(abi, "cuda", five.SEED)
    print("REFERENCE", name, batch, [tuple(t.shape) for t in inputs], flush=True)
    reference = semantic.reference(inputs)
    np.savez(
        directory / "inputs.npz",
        **{f"input{i}": t.cpu().numpy() for i, t in enumerate(inputs)},
    )
    np.save(directory / "reference.npy", reference.cpu().numpy())
    settings = dict(
        case=name,
        B=batch,
        workload=workload,
        problem=problem.to_dict(),
        abi=abi.external(),
        effective_flops=workload_flops(workload, abi),
        seed=five.SEED,
        dtype="float16",
        warmup_calls=five.WARMUP,
        measurement_ms=five.BENCHMARK_MS,
        quantiles=[0.5, 0.2, 0.8],
        cooldown_s=five.COOLDOWN_S,
        relative_l2_limit=semantic.relative_l2_limit,
        normalized_max_limit=semantic.normalized_max_limit,
    )
    (directory / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    del inputs, reference
    torch.cuda.empty_cache()
