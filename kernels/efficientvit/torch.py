from __future__ import annotations
import torch


def efficientvit_pytorch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-15
) -> torch.Tensor:
    """Q=[B,N,D], K=[B,D,N], V=[B,E,N], output=[B,N,E]."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("EfficientViT requires rank-three Q, K, V")
    if (
        q.shape[0] != k.shape[0]
        or q.shape[0] != v.shape[0]
        or q.shape[2] != k.shape[1]
        or k.shape[2] != v.shape[2]
    ):
        raise ValueError("EfficientViT batch/key/sequence dimensions disagree")
    kv = k @ v.transpose(-1, -2)
    k_sum = k.sum(dim=-1, keepdim=True)
    return (q @ kv) / ((q @ k_sum) + eps)
