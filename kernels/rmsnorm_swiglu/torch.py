import torch
import torch.nn.functional as F


@torch.inference_mode()
def rmsnorm_swiglu_pytorch_ref(
    X: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    eps: float = 0.0,
):
    """
    RMS + SwiGLU implementation in PyTorch.

    Args:
        X: Input tensor of shape (M, D).
        W: Gate weight tensor of shape (D, K).
        V: Weight tensor of shape (D, K).
        U: Weight tensor of shape (K, N).
    Returns:
        Output tensor of shape (M, N).
    """
    # RMSNorm
    normalized_X = F.rms_norm(X, normalized_shape=X.shape[-1:], eps=eps)

    # SwiGLU
    gate = torch.matmul(normalized_X, W)
    value = torch.matmul(normalized_X, V)
    swiglu = F.silu(gate) * value

    # Final linear transformation
    output = torch.matmul(swiglu, U)
    return output


@torch.inference_mode()
def rmsnorm_swiglu_pytorch_ref_plain(
    X: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    eps: float = 0.0,
):
    """
    RMS + SwiGLU implementation in PyTorch without using built-in RMSNorm or SiLU.

    Args:
        X: Input tensor of shape (M, D).
        W: Gate weight tensor of shape (D, K).
        V: Weight tensor of shape (D, K).
        U: Weight tensor of shape (K, N).
    Returns:
        Output tensor of shape (M, N).
    """
    # RMSNorm
    rms = torch.sqrt(torch.mean(X * X, dim=-1, keepdim=True) + eps)
    normalized_X = X / rms

    # SwiGLU
    gate = torch.matmul(normalized_X, W)
    value = torch.matmul(normalized_X, V)
    swiglu = (gate * torch.sigmoid(gate)) * value

    # Final linear transformation
    output = torch.matmul(swiglu, U)
    return output


def rmsnorm_swiglu_pytorch(
    X: torch.Tensor,
    WV: torch.Tensor,
    U: torch.Tensor,
    eps: float = 0.0,
):
    """
    RMS + SwiGLU implementation in PyTorch with combined weights.

    Args:
        X: Input tensor of shape (M, D).
        WV: Gate and value weight tensor of shape (2*K, D).
        U: Weight tensor of shape (K, N).
    Returns:
        Output tensor of shape (M, N).
    """
    # RMSNorm
    normalized_X = F.rms_norm(X, normalized_shape=X.shape[-1:], eps=eps)

    # SwiGLU
    X_WV = torch.matmul(normalized_X, WV.T)
    gate, value = X_WV.chunk(2, dim=-1)
    swiglu = F.silu(gate) * value

    # Final linear transformation
    output = torch.matmul(swiglu, U)
    return output


def rmsnorm_swiglu_pytorch_plain(
    X: torch.Tensor,
    WV: torch.Tensor,
    U: torch.Tensor,
    eps: float = 0.0,
):
    """
    RMS + SwiGLU implementation in PyTorch with combined weights without using built-in RMSNorm or SiLU.

    Args:
        X: Input tensor of shape (M, D).
        WV: Gate and value weight tensor of shape (2*K, D).
        U: Weight tensor of shape (K, N).
    Returns:
        Output tensor of shape (M, N).
    """
    # RMSNorm
    rms = torch.sqrt(torch.mean(X * X, dim=-1, keepdim=True) + eps)
    normalized_X = X / rms

    # SwiGLU
    X_WV = torch.matmul(normalized_X, WV.T)
    gate, value = X_WV.chunk(2, dim=-1)
    swiglu = (gate * torch.sigmoid(gate)) * value

    # Final linear transformation
    output = torch.matmul(swiglu, U)
    return output
