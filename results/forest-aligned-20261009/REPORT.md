# XLA: FOREST-aligned ten-case experiment

Repository: `git@github.com:Scarlet1ssimo/xla-eval.git`, original commit `69db82c`.

Runtime: Python 3.10.22; torch 2.7.0+cu126; torch-xla 2.7.0 official CUDA 12.6 wheel; Triton 3.3.0; runtime cuDNN 9.5.1. CUDA toolkit 13.0 supplies ptxas/libdevice. H100 NVL, enforced power limit 350 W, clocks unlocked. The FOREST repository/environments were not modified.

All ten shapes and physical input directions match the current FOREST five-workload script. Attention is native B,H MHA. FP16 external tensors, seed 1729, 10 warmup calls, 200 ms do_bench budget, flush L2, unchanged 5 s thermal/idle cooldown, worker budget 600 s. Numerical thresholds were not relaxed. All 20 final measurements pass correctness before and after timing; no CPU fallback.

**Timing distinction:** `xla` is an uninstrumented CUDA-event do_bench observation of the complete cached executable invocation with the required XLA completion barrier. It includes host submission/wait gaps. `xla_gpu_trace` is a separate Nsight observation of first GPU activity to last GPU activity, including the complete operator and all graph/kernel/copy/memset work, excluding preceding flush and host gaps. It must not be substituted silently into a standard do_bench comparison. Neither is the original repository's lazy-IR/perf_counter timer.

| Workload | B | XLA do_bench TFLOPS | do_bench latency (ms) | GPU trace TFLOPS | GPU trace latency (ms) | trace samples |
|---|---:|---:|---:|---:|---:|---:|
| attention | 1 | 70.004192 | 3.926592 | 71.639568 | 3.836957 | 38 |
| gqa-attention | 1 | 69.849359 | 3.935296 | 73.670641 | 3.731173 | 39 |
| layernorm-matmul | 1 | 1.805000 | 0.260256 | 2.711485 | 0.173249 | 113 |
| rms-ffn-swiglu | 1 | 3.871930 | 0.116992 | 24.280919 | 0.018656 | 129 |
| efficient-vit | 1 | 0.073419 | 0.114256 | 0.894689 | 0.009376 | 126 |
| attention | 16 | 69.552999 | 63.233025 | 71.519370 | 61.494481 | 3 |
| gqa-attention | 16 | 62.655219 | 70.194416 | 66.109430 | 66.526765 | 2 |
| layernorm-matmul | 16 | 29.692310 | 0.253136 | 50.155432 | 0.149858 | 119 |
| rms-ffn-swiglu | 16 | 40.072967 | 0.180864 | 95.526114 | 0.075872 | 117 |
| efficient-vit | 16 | 1.316893 | 0.101920 | 10.922667 | 0.012288 | 129 |

The attention/GQA XLA path materializes score matrices and softmax rather than producing a fused FlashAttention kernel. Input/optimized HLO and CUDA traces retain the actual generated work. This is the pinned repository build's default path, not a manually tuned XLA kernel.

A preliminary explicit-FP16 GQA B1 variant failed normalized-max validation (0.020527 >0.02); it is preserved under `gqa-attention/B1/attempts/explicit-fp16/` and excluded. The final GQA path uses the original SDPA API. An initial DLPack-stream event smoke missed compute work and was discarded; completion synchronization and full GPU trace now verify the complete operation.

**Shared GPU limitation:** other users' GPU contexts were observed during this experiment. Thermal/idle cooldown passed, but GPU exclusivity was not enforced. Treat these as shared-machine initial measurements; they do not establish an exclusive-GPU performance bound. Only 2–3 trace samples fit the 200 ms budget for B16 attention/GQA; quantiles are retained but are sparsely sampled.

Reproduce from the isolated checkout:

```bash
cd ~/xla-eval
uv sync --locked
uv run --locked python main.py
```

The command creates a fresh artifact directory and new fixtures. It runs both timing modes and writes `tflops.csv`, `latency.csv` (ms), and `summary.json`. `--no-gpu-trace` retains only the uninstrumented standard invocation timing.

The current reproduction code is committed in this repository. Full input/reference, HLO and trace artifacts remain on the measurement machine under `outputs/forest-aligned-20261009/` (ignored by Git). The local `measured-source.py` contains the equivalent function definitions used in the measured runs; all five functions were checked for exact equivalence to the wired canonical modules on additional small, non-square shapes. GQA, EVT, and LayerNorm B16 also pass GPU adapter smoke checks. `validation.json` preserves acceptance and rejected-attempt diagnostics.
