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
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Grouped Query Attention (GQA) implementation in PyTorch.
    
    Args:
        Q: Query tensor of shape [B, G, R, M, D]
        K: Key tensor of shape [B, G, N, D]
        V: Value tensor of shape [B, G, N, L]
        scale: Scaling factor for attention scores (default: 1/sqrt(D))
    
    Returns:
        Output tensor of shape [B, G, R, M, L]
    """
    B, G, R, M, D = Q.shape
    _, _, N, _ = K.shape
    _, _, N_v, L = V.shape
    if N_v != N:
        raise ValueError("Inconsistent GQA shapes: expected V.shape[2] == K.shape[2]")

    if scale is None:
        scale = D ** -0.5

    # SDPA expects [B, Hq, M, D] for query and [B, Hk, N, D] / [B, Hk, N, L] for key/value.
    q = Q.reshape(B, G * R, M, D)
    k = K
    v = V
    with _sdpa_ctx(q):
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
    return out.reshape(B, G, R, M, L)
