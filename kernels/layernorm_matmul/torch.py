import torch


def layernorm_matmul_pytorch(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 0.0,
):
    """
    LayerNorm + MatMul implementation in PyTorch.

    Args:
        X: Input tensor of shape (M, K).
        Y: Weight tensor of shape (N, K).
    Returns:
        Output tensor of shape (M, N).
    """
    # LayerNorm
    X_normalized = torch.nn.functional.layer_norm(X, X.shape[-1:], eps=eps)

    # MatMul
    output = torch.matmul(X_normalized, Y.T)
    return output


def layernorm_matmul_pytorch_plain(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 0.0,
):
    """
    LayerNorm + MatMul implementation in PyTorch without using built-in LayerNorm.

    Args:
        X: Input tensor of shape (M, K).
        Y: Weight tensor of shape (N, K).
    Returns:
        Output tensor of shape (M, N).
    """
    # LayerNorm
    mean = X.mean(dim=-1, keepdim=True)
    var = X.var(dim=-1, keepdim=True, unbiased=False)
    X_normalized = (X - mean) / torch.sqrt(var + eps)

    # MatMul
    output = torch.matmul(X_normalized, Y.T)
    return output
