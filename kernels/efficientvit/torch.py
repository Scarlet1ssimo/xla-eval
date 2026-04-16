from __future__ import annotations

import torch


def efficientvit_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-15,
) -> torch.Tensor:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must all be rank-3 tensors [B, N, N]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must all have the same shape [B, N, N]")

    kv = torch.matmul(k, v)
    k_sum = torch.sum(k, dim=2, keepdim=True)
    qkv = torch.matmul(q, kv)
    scale = torch.matmul(q, k_sum)
    return qkv / (scale + eps)