# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Linear algebra utilities."""

import torch
from jaxtyping import Float
from torch import Tensor


def svd(
    x: Float[Tensor, "*batch M N"],
    *,
    topk: int | None = None,
    deterministic: bool = True,
) -> tuple[
    Float[Tensor, "*batch M K"],
    Float[Tensor, "*batch K"],
    Float[Tensor, "*batch K N"],
]:
    """Computes the singular value decomposition (SVD) of a batch of matrices.

    Args:
        x: Tensor of shape `(*, M, N)`, where `*` is zero or more batch dimensions.
        topk: If not `None`, only return the top `k` singular values and corresponding
            singular vectors. If `None`, return all singular values and vectors.
        deterministic: If `True`, ensure deterministic output by fixing the signs
            of the singular vectors. This is done by making the largest absolute value
            in each column of `U` positive. If `False`, the output may vary across runs
            due to the inherent ambiguity in the sign of singular vectors.

    Returns:
        A tuple `(U, S, V^H)`, where `U` is of shape `(*, M, K)`, `S` is of shape
        `(*, K)`, and `V^H` is of shape `(*, K, N)`, where `K` is `topk` if `topk` is
        not `None`, else `min(M, N)`.
    """
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    if topk is not None:
        u = u[..., :topk]
        s = s[..., :topk]
        vh = vh[..., :topk, :]

    if deterministic:
        max_abs_cols = torch.argmax(torch.abs(u), dim=-2, keepdim=True)  # (*, M, 1)
        signs = torch.sign(torch.gather(u, -2, max_abs_cols))  # (*, M, 1)
        u = u * signs
        vh = vh * signs.transpose(-2, -1)

    return u, s, vh


def lstsq(
    A: Float[Tensor, "batch m n"],
    b: Float[Tensor, "batch m"],
    *,
    positive: bool = False,
    nnls_max_iter: int = 100,
) -> Float[Tensor, "batch n"]:
    """Solves the least squares problem `min_x ||Ax - b||_2` for `x`.

    Args:
        A: The coefficient matrix of shape. The shape can be `(m, n)` or `(batch, m, n)`
            for batched inputs.
        b: The target vector. The shape can be `(m,)` or `(batch, m)` for batched
            inputs.
        positive: If `True`, enforces non-negativity constraints on the solution `x`.
        nnls_max_iter: The maximum number of coordinate descent iterations to perform
            when `positive=True`.

    Returns:
        The least squares solution `x` of shape `(n,)`.
    """
    if not positive:
        # use torch.linalg.lstsq when no constraints are needed
        x = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
        return x

    if A.ndim == 2:
        A = A.unsqueeze(0)  # pyright: ignore[reportConstantRedefinition]
        squeeze = True
    else:
        squeeze = False

    x = _batched_nnls(A.transpose(1, 2), b, max_iter=nnls_max_iter)
    return x.squeeze(0) if squeeze else x


def _batched_nnls(
    A: Float[Tensor, "batch n m"],
    b: Float[Tensor, "batch m"],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Float[Tensor, "batch n"]:
    """Batched non-negative least squares solver using block coordinate descent."""
    B, N, _ = A.shape

    # precompute: AtA = A @ A^T, Atb = A @ b
    AtA = torch.bmm(A, A.transpose(1, 2))  # (B, N, N)
    Atb = torch.bmm(A, b.unsqueeze(-1)).squeeze(-1)  # (B, N)

    # initialize with unconstrained solution projected to non-negative
    try:
        L = torch.linalg.cholesky(AtA + 1e-6 * torch.eye(N, device=A.device))
        x = torch.cholesky_solve(Atb.unsqueeze(-1), L).squeeze(-1)
    except RuntimeError:
        x = torch.zeros((B, N), device=A.device, dtype=A.dtype)

    x = torch.clamp(x, min=0)

    # block coordinate descent
    for _ in range(max_iter):
        x_old = x.clone()

        for j in range(N):
            grad_j = torch.sum(AtA[:, j, :] * x, dim=1) - Atb[:, j]
            step = grad_j / (AtA[:, j, j] + 1e-10)
            x[:, j] = torch.clamp(x[:, j] - step, min=0)

        if torch.norm(x - x_old) < tol * torch.norm(x_old).clamp(min=1e-10):
            break

    return x
