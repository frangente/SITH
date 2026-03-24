# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """A linear classification head for CLIP-based models."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        bias: bool = True,
        normalize: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes the classification head.

        Args:
            input_dim: The dimensionality of the input features.
            num_classes: The number of output classes.
            bias: Whether to include a bias term in the linear layer.
            normalize: Whether to L2-normalize the input features before
                classification.
            device: The device on which to initialize the parameters.
            dtype: The data type for the parameters.
        """
        super().__init__()

        self.in_features = input_dim
        self.out_features = num_classes
        self.normalize = normalize

        self.weight = nn.Parameter(
            torch.empty(num_classes, input_dim, device=device, dtype=dtype)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.empty(num_classes, device=device, dtype=dtype)
            )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initializes the parameters of the classification head."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"num_classes={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"normalize={self.normalize}"
        )

    def forward(
        self, x: Float[Tensor, "*batch input_dim"]
    ) -> Float[Tensor, "*batch num_classes"]:
        """Forward pass through the classification head."""
        if self.normalize:
            x = F.normalize(x, dim=-1)

        x = F.linear(x, self.weight, self.bias)
        return x
