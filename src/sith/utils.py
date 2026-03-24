# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Utility functions."""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


def accuracy(
    logits: Float[Tensor, "batch_size num_classes"],
    labels: Int[Tensor, " batch_size"],
    k: int = 1,
) -> float:
    """Computes the top-k accuracy.

    Args:
        logits: The model's output logits of shape `(batch_size, num_classes)`.
        labels: The true labels of shape `(batch_size,)`.
        k: The number of top elements to consider for accuracy.

    Returns:
        The top-k accuracy as a float.
    """
    labels = labels.unsqueeze(1)  # (batch_size, 1)
    if k == 1:
        # For top-1 accuracy, we can use argmax which is more efficient
        pred = logits.argmax(dim=1, keepdim=True)  # (batch_size, 1)
    else:
        _, pred = torch.topk(logits, k, dim=1, largest=True)  # (batch_size, k)

    matches = pred == labels  # (batch_size, k)
    correct = matches.any(dim=1)  # (batch_size,)
    accuracy = correct.float().mean().item()
    return accuracy


def cosine_similarity(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    """Computes the pairwise cosine similarity between two tensors.

    Args:
        x: The first input tensor.
        y: The second input tensor.
        dim: The dimension along which to compute the cosine similarity.

    Returns:
        The pairwise cosine similarity between the two tensors.
    """
    return F.cosine_similarity(x, y, dim=dim)


def all_pairs_cosine_similarity(x: Tensor, y: Tensor) -> Tensor:
    """Computes the cosine similarity between all pairs of vectors in two tensors.

    Args:
        x: The first input tensor.
        y: The second input tensor.

    Returns:
        The cosine similarity between all pairs of vectors in the two tensors.
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return x @ y.transpose(-2, -1)
