# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import torch
from jaxtyping import Float
from sklearn.linear_model import Lasso
from torch import Tensor


def lasso(
    x: Float[Tensor, "batch_size dim"],
    dictionary: Float[Tensor, "num_atoms dim"],
    *,
    l1_penalty: float = 0.1,
    positive: bool = False,
    max_iter: int = 10000,
    tol: float = 1e-6,
) -> Float[Tensor, "batch_size num_atoms"]:
    """Sparsely encode a vector using Lasso regression and the given dictionary.

    Note:
        This function is currently a wrapper around the `sklearn.linear_model.Lasso`
        class, which does not support neither batched inputs nor GPU tensors.
        As a result, the input tensors will be moved to CPU and processed one by one,
        which can be inefficient for large batches.

    Args:
        x: The input vector to encode. This can be either a 1D tensor of shape `(dim,)`
            or a 2D tensor of shape `(batch_size, dim)`.
        dictionary: The dictionary to use for encoding. This should be a tensor of
            shape `(num_atoms, dim)`.
        l1_penalty: The L1 penalty term for the Lasso regression. This controls the
            sparsity of the encoding with a higher value leading to sparser solutions.
        positive: Whether to enforce non-negativity constraints on the sparse codes.
        max_iter: The maximum number of iterations for the Lasso solver.
        tol: The tolerance for the optimization. The solver will stop when the change
            in the loss function is less than this value.

    Returns:
        A tensor containing the sparse codes for the input vector(s). The shape of the
        output will be `(num_atoms,)` if the input was a 1D tensor, or
        `(batch_size, num_atoms)` if the input was a 2D tensor.
    """
    if x.ndim == 1:
        squeeze = True
        x = x.unsqueeze(0)
    else:
        squeeze = False

    # see SpLiCE: https://github.com/AI4LIFE-GROUP/SpLiCE/blob/9a498102ce7c6701f4afe361ffaa86c39b47fa5f/splice/model.py#L47
    alpha = l1_penalty / (2 * dictionary.shape[-1])
    lasso = Lasso(
        alpha=alpha,
        fit_intercept=False,
        positive=positive,
        max_iter=max_iter,
        tol=tol,
    )

    B, M = x.shape[0], dictionary.shape[0]
    x_np = x.cpu().numpy()
    dictionary_np = dictionary.T.cpu().numpy()

    coefs = torch.empty((B, M), dtype=x.dtype, device=x.device)
    for i in range(B):
        lasso.fit(dictionary_np, x_np[i])
        coefs[i] = torch.as_tensor(lasso.coef_, dtype=x.dtype, device=x.device)

    if squeeze:
        coefs = coefs.squeeze(0)

    return coefs
