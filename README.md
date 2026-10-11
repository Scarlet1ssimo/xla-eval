# FOREST-aligned OpenXLA profiling

This checkout profiles **PyTorch/XLA 2.7.0 CUDA**, using an isolated Python 3.10
uv environment. PyTorch 2.7.0+cu126 requires Triton 3.3.0; the incompatible
original `triton>=3.6.0` requirement is corrected and locked. FOREST's upstream
and Tawa environments are unchanged.

```bash
cd ~/xla-eval
uv sync --locked
uv run --locked python main.py
```

The default generates fresh input/reference fixtures with
`~/forest/benchmarks/.venv/bin/python`, then runs each case in its own process,
sequentially on GPU 0. It writes a new uniquely named directory under `outputs/`.
Existing experiment directories are never overwritten. `--forest-root` and
`--output-dir` can specify the local checkout and a new destination.

The CUDA launcher loads the available `gcc/12.3.0-ubuntu` and
`cuda-toolkit/13.0` modules for ptxas/libdevice. CUDA runtime and cuDNN come from
the pinned wheel environment, not the module's shared libraries. It also exposes
the uv-managed Python's shared library required by `_XLAC`.

| Workload | B | Physical inputs / established dimensions |
|---|---|---|
| Native MHA | 1,16 | Q/K/V BHSD; H=32, M=N=4096, D=L=128 |
| GQA | 1,16 | Q=[B,32,4096,128], K/V=[B,8,4096,128]; G=8,R=4 |
| LayerNorm+Matmul | 1,16 | X=[B,8192], Y=[8192,28672]; shared weights |
| RMSNorm-FFN-SwiGLU | 1,16 | X=[B*4096,96], W/V=[96,192], U=[192,96]; shared weights |
| EfficientViT | 1,16 | Q=[B,N,D], K=[B,D,N], V=[B,E,N]; N=D=E=128 |

`prepare_forest_inputs.py` imports the current FOREST five-workload script, its
input generators, FP64 references, validation thresholds and effective FLOPs
formula. It does not require historical winner protobufs or optimizer discovery.
All inputs are FP16, contiguous, seed 1729. Normalization has no affine parameters
and epsilon 0. EfficientViT uses positive Q/K and epsilon 1e-15. V's transpose is
inside its operator. No weight packing or layout copies are moved outside the
operator to improve its timing. Attention is non-causal with scale 1/sqrt(D).

The canonical workload implementations in `kernels/*/torch.py` are traced by
XLA. Cached compiled executables run repeatedly, without rebuilding lazy Python
IR each iteration. The complete compiled operator is measured; it may contain
multiple GPU launches or XLA's internal command buffer. No CUDA graph benchmark
helper is used. CPU fallback, numerical failures and worker errors are retained
and excluded from performance tables. Output must satisfy the contiguous ABI.

## Timing columns

- **`xla`**: uninstrumented `triton.testing.do_bench`, flush L2, 10 initial warmup
  calls, a measurement budget inherited from FOREST's `BENCHMARK_MS` (currently
  200 ms), median/p20/p80. Existing results retain the budget recorded in their
  per-case `settings.json`. XLA uses another CUDA stream,
  so the wrapper finishes the L2 flush before submitting XLA work, and waits for
  XLA completion before recording the end event. These CUDA-event samples include
  the necessary host submission/wait gap. They cover the complete invocation.
- **`xla_gpu_trace`**: a separate Nsight Systems run of the same protocol. It
  reports first GPU activity start to last GPU activity end for each measured
  invocation, including **all** kernels, graph executions, device copies and
  memsets. It excludes the preceding L2 flush and host submission/completion
  overhead. Median/p20/p80 come from the same number of measured observations as
  `do_bench`; initial estimation and warmup observations are excluded. This is a
  GPU diagnostic and should not silently replace the `xla` column in a table of
  ordinary `do_bench` latency.

Both modes exclude compilation, input generation/transfer and correctness from
timing. Numerical checks run before and after measurement with FOREST's original
relative-L2 <0.005, normalized-max <0.02 and finite-output gates. Before each
measurement, the unchanged FOREST cooldown requires at least 20 seconds, GPU
<=55 C, utilization <=5%, stable for 1 second, capped at 60 seconds. Worker budget
is 600 seconds; no unsuccessful partial result is published. Clock/power settings
are not modified. Shared GPU contexts must be disclosed; cooldown alone does not
prove exclusive GPU use. New fixtures inherit both `BENCHMARK_MS` and
`COOLDOWN_S` from FOREST; regenerate fixtures after a protocol change. No
minimum sample count is imposed; existing fixtures/results retain their saved
settings. The shared protocol is documented in FOREST's `benchmarks/README.md`
under "Shared final measurement protocol".

Artifacts include `tflops.csv`, `latency.csv` (**milliseconds**), `summary.json`,
per-case settings, input/reference fixtures, pre/post validation, cooldown samples,
software/GPU versions, source snapshots, input/optimized HLO, XLA autotuning dumps,
PTX, logs and Nsight reports/SQLite samples. Separate HLO directories preserve
profiled and uninstrumented compilation runs. Use `--no-gpu-trace` to omit Nsight
and produce only the standard invocation measurement.

The first measured report, CSVs and acceptance metadata are committed under
`results/forest-aligned-20261009/`. Large raw artifacts remain in the ignored
local `outputs/forest-aligned-20261009/` directory.
