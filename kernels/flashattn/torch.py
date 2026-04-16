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


def set_flashattn_sdpa_backend(mode: _SdpaMode) -> None:
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


@torch.inference_mode()
def flashattn_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    FlashAttention implementation in PyTorch.

    Args:
        Q: Query tensor of shape (..., M, D).
        K: Key tensor of shape (..., N, D).
        V: Value tensor of shape (..., N, L).
    Returns:
        Output tensor of shape (..., M, L).
    """
    # Fused CUDA SDPA backends expect 4D layout [B, H, T, D].
    return F.scaled_dot_product_attention(
        query=Q,
        key=K,
        value=V,
    )


@torch.inference_mode()
def flashattn_pytorch_plain(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    FlashAttention implementation in PyTorch without using built-in scaled_dot_product_attention.

    Args:
        Q: Query tensor of shape (..., M, D).
        K: Key tensor of shape (..., N, D).
        V: Value tensor of shape (..., N, L).
    Returns:
        Output tensor of shape (..., M, L).
    """
    # Compute attention scores
    attn_scores = torch.bmm(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)

    # Apply softmax to get attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # Compute output
    output = torch.bmm(attn_weights, V)

    return output
