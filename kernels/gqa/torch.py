import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Literal

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - depends on torch version
    SDPBackend = None
    sdpa_kernel = None


_SdpaMode = Literal["flash", "cudnn"]
_SDPA_STATE: dict[str, _SdpaMode] = {"mode": "flash"}


def set_gqa_sdpa_backend(mode: _SdpaMode) -> None:
    _SDPA_STATE["mode"] = mode


def _sdpa_ctx(query: torch.Tensor):
    if sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    if query.is_cuda:
        if _SDPA_STATE["mode"] == "flash":
            backend = SDPBackend.FLASH_ATTENTION
        else:
            backend = SDPBackend.CUDNN_ATTENTION
    else:
        backend = SDPBackend.MATH
    return sdpa_kernel(backend)


def gqa_pytorch(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """Q=[B,G*R,M,D], K=[B,G,N,D], V=[B,G,N,L], contiguous BHSD."""
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("GQA requires physical rank-four Q/K/V")
    if Q.shape[1] % K.shape[1] or K.shape[2] != V.shape[2]:
        raise ValueError("Inconsistent GQA heads or sequence lengths")
    with _sdpa_ctx(Q):
        return F.scaled_dot_product_attention(
            Q, K, V, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=True
        )
