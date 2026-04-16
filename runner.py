from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Callable, Literal

import pandas as pd
import torch

try:
    import triton  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    triton = None

from kernels.efficientvit.torch import efficientvit_pytorch
from kernels.flashattn.torch import flashattn_pytorch, set_flashattn_sdpa_backend
from kernels.gqa.torch import gqa_pytorch, set_gqa_sdpa_backend
from kernels.layernorm_matmul.torch import layernorm_matmul_pytorch
from kernels.rmsnorm_swiglu.torch import rmsnorm_swiglu_pytorch

try:
    import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - optional dependency
    xm = None
    _xla_import_error = f"ImportError: {exc}"
else:
    _xla_import_error = None


BackendName = Literal["torch_flash", "torch_cudnn", "torch_xla"]
TimerName = Literal["triton_do_bench", "perf_counter"]
CompileModeName = Literal["default", "reduce-overhead", "max-autotune"]


@dataclass(frozen=True)
class BenchmarkArtifacts:
    csv_path: Path
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class BenchmarkConfig:
    backends: tuple[BackendName, ...] = ("torch_flash", "torch_cudnn", "torch_xla")
    batches: tuple[int, ...] = (1, 16)
    dtype: str = "float16"
    timer: TimerName = "triton_do_bench"
    torch_compile: bool = True
    torch_compile_mode: CompileModeName = "max-autotune"
    warmup: int = 25
    rep: int = 100
    output_dir: Path = Path("outputs")
    output_prefix: str = "perf"


@dataclass(frozen=True)
class _CaseDef:
    name: str
    build: Callable[[torch.device, torch.dtype, int], "_CaseRuntime"]


@dataclass(frozen=True)
class _CaseRuntime:
    fn: Callable[..., torch.Tensor]
    args: tuple[torch.Tensor, ...]
    flops: float


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _backend_device(backend: BackendName) -> torch.device:
    if backend in ("torch_flash", "torch_cudnn"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if xm is None:
        detail = f" ({_xla_import_error})" if _xla_import_error else ""
        raise RuntimeError(
            "torch_xla is unavailable for backend=torch_xla"
            f"{detail}. This is often caused by torch / torch-xla version mismatch."
        )
    return xm.xla_device()


def _sync(backend: BackendName, device: torch.device) -> None:
    if backend in ("torch_flash", "torch_cudnn"):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return

    # torch-xla uses lazy execution; explicit step + wait makes timing stable.
    xm.mark_step()
    xm.wait_device_ops()


def _maybe_compile_torch_fn(
    fn: Callable[..., torch.Tensor],
    *,
    mode: CompileModeName,
) -> Callable[..., torch.Tensor]:
    # Keep benchmark runnable on older PyTorch versions.
    if not hasattr(torch, "compile"):
        return fn

    try:
        compiled_fn = torch.compile(fn, mode=mode)
    except RuntimeError as exc:
        print(f"Warning: torch.compile failed; falling back to eager mode ({exc})")
        return fn

    state = {"use_eager": False}

    def _wrapped(*args: torch.Tensor) -> torch.Tensor:
        if state["use_eager"]:
            return fn(*args)
        try:
            return compiled_fn(*args)
        except RuntimeError as exc:
            # Some kernels can fail at runtime for specific shapes/dtypes.
            print(f"Warning: compiled run failed; switching to eager mode ({exc})")
            state["use_eager"] = True
            return fn(*args)

    return _wrapped


def _run_triton_do_bench(
    fn: Callable[..., torch.Tensor],
    *,
    warmup: int,
    rep: int,
) -> float:
    if triton is None:
        raise RuntimeError(
            "timer=triton_do_bench requires triton to be installed in xla_eval environment"
        )

    result = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.1, 0.9])
    if result is None:
        raise RuntimeError("triton.testing.do_bench returned no timing result")

    ms, _, _ = result
    return float(ms)


def _benchmark_fn(
    *,
    backend: BackendName,
    device: torch.device,
    timer: TimerName,
    warmup: int,
    rep: int,
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    flops: float,
) -> tuple[float, float]:
    if warmup < 0 or rep <= 0:
        raise ValueError("warmup must be >= 0 and rep must be > 0")

    if timer == "triton_do_bench" and backend in ("torch_flash", "torch_cudnn") and device.type == "cuda":
        ms = _run_triton_do_bench(lambda: fn(*args), warmup=warmup, rep=rep)
        tflops = flops / (ms * 1e-3) / 1e12
        return ms, tflops

    for _ in range(warmup):
        _ = fn(*args)
        _sync(backend, device)

    start = time.perf_counter()
    for _ in range(rep):
        _ = fn(*args)
        _sync(backend, device)
    elapsed = time.perf_counter() - start

    ms = elapsed * 1e3 / rep
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


def _flashattn_case(
    device: torch.device, dtype: torch.dtype, batch: int
) -> _CaseRuntime:
    # Base shape (llama2-7b): M=4096, N=4096, D=128, L=128, H=32
    # m, n, d, h = 4096, 4096, 128, 32
    m, n, d, h = 16, 4096, 64, 16
    bh = batch * h
    # q = torch.randn((bh, m, d), device=device, dtype=dtype)
    # k = torch.randn((bh, n, d), device=device, dtype=dtype)
    # v = torch.randn((bh, n, d), device=device, dtype=dtype)
    q = torch.randn((batch, h, m, d), device=device, dtype=dtype)
    k = torch.randn((batch, h, n, d), device=device, dtype=dtype)
    v = torch.randn((batch, h, n, d), device=device, dtype=dtype)
    # print(f"{bh=}, {m=}, {n=}, {d=}")
    # print(f"{q.shape=}, {k.shape=}, {v.shape=}")

    return _CaseRuntime(
        fn=flashattn_pytorch,
        args=(q, k, v),
        flops=4.0 * bh * m * n * d,
    )


def _gqa_case(
    device: torch.device, dtype: torch.dtype, batch: int
) -> _CaseRuntime:
    # Base shape (llama3-8b): M=4096, N=4096, D=128, L=128, G=8, R=4
    # m, n, d, l, g, r = 4096, 4096, 128, 128, 8, 4
    m, n, d, l, g, r = 256, 4096, 64, 64, 1, 2
    bg = batch * g
    q = torch.randn((batch, g, r, m, d), device=device, dtype=dtype)
    k = torch.randn((batch, g, n, d), device=device, dtype=dtype)
    v = torch.randn((batch, g, n, l), device=device, dtype=dtype)
    
    print(f"{bg=}, {m=}, {n=}, {d=}, {l=}, {g=}, {r=}")
    print(f"{q.shape=}, {k.shape=}, {v.shape=}")

    return _CaseRuntime(
        fn=gqa_pytorch,
        args=(q, k, v),
        flops=2.0 * bg * r * m * n * (d + l),
    )


def _efficientvit_case(
    device: torch.device, dtype: torch.dtype, batch: int
) -> _CaseRuntime:
    # Base shape: B=1,16, N=128
    n = 128
    q = torch.randn((batch, n, n), device=device, dtype=dtype)
    k = torch.randn((batch, n, n), device=device, dtype=dtype)
    v = torch.randn((batch, n, n), device=device, dtype=dtype)

    return _CaseRuntime(
        fn=efficientvit_pytorch,
        args=(q, k, v),
        flops=4.0 * batch * n * n * n + 2.0 * batch * n * n,
    )


def _layernorm_matmul_case(
    device: torch.device, dtype: torch.dtype, batch: int
) -> _CaseRuntime:
    # Base shape (llama2-70b): M=1, K=8192, N=28672
    m, k, n = batch * 1, 8192, 28672
    
    x = torch.randn((m, k), device=device, dtype=dtype)
    y = torch.randn((n, k), device=device, dtype=dtype)

    return _CaseRuntime(
        fn=layernorm_matmul_pytorch,
        args=(x, y),
        flops=2.0 * m * n * k,
    )


def _rmsnorm_swiglu_case(
    device: torch.device, dtype: torch.dtype, batch: int
) -> _CaseRuntime:
    # Base shape (mobilevit-xxs): M=4096, D=96, N=96, K=192
    m, d, n, k = batch * 4096, 96, 96, 192
    x = torch.randn((m, d), device=device, dtype=dtype)
    wv = torch.randn((2 * k, d), device=device, dtype=dtype)
    u = torch.randn((n, k), device=device, dtype=dtype)

    return _CaseRuntime(
        fn=rmsnorm_swiglu_pytorch,
        args=(x, wv, u),
        flops=2.0 * m * k * d * 2 + 2.0 * m * n * k,
    )


def _case_defs() -> tuple[_CaseDef, ...]:
    return (
        _CaseDef("flashattn", _flashattn_case),
        _CaseDef("gqa", _gqa_case),
        _CaseDef("layernorm", _layernorm_matmul_case),
        _CaseDef("rms_norm", _rmsnorm_swiglu_case),
        _CaseDef("efficientvit", _efficientvit_case),
    )


def run_benchmark(config: BenchmarkConfig) -> BenchmarkArtifacts:
    torch.set_grad_enabled(False)
    dtype = _dtype_from_name(config.dtype)
    rows: list[dict[str, object]] = []

    for backend in config.backends:
        print(f"Running benchmark for backend={backend} on device={_backend_device(backend)} with dtype={dtype}")
        device = _backend_device(backend)

        if backend == "torch_flash":
            set_flashattn_sdpa_backend("flash")
            set_gqa_sdpa_backend("flash")
        elif backend == "torch_cudnn":
            set_flashattn_sdpa_backend("cudnn")
            set_gqa_sdpa_backend("cudnn")

        for case in _case_defs():
            print(f"  Case: {case.name}")
            for batch in config.batches:
                runtime = case.build(device, dtype, batch)
                fn = runtime.fn
                if (
                    config.torch_compile
                    and backend in ("torch_flash", "torch_cudnn")
                    and case.name not in ("flashattn", "gqa")
                ):
                    print(f"    Compiling with torch.compile (mode={config.torch_compile_mode})...")
                    fn = _maybe_compile_torch_fn(fn, mode=config.torch_compile_mode)

                ms, tflops = _benchmark_fn(
                    backend=backend,
                    device=device,
                    timer=config.timer,
                    warmup=config.warmup,
                    rep=config.rep,
                    fn=fn,
                    args=runtime.args,
                    flops=runtime.flops,
                )
                rows.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "backend": backend,
                        "device": str(device),
                        "case": case.name,
                        "batch": batch,
                        "dtype": config.dtype,
                        "warmup": config.warmup,
                        "rep": config.rep,
                        "estimated_flops": runtime.flops,
                        "ms_per_iter": ms,
                        "tflops": tflops,
                    }
                )

    df = pd.DataFrame(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / f"{config.output_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_path, index=False)
    return BenchmarkArtifacts(csv_path=out_path, dataframe=df)
