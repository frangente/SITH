# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import math
from typing import Self

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


class LinearWithLoRA(nn.Module):
    """A linear layer with low-rank adaptation (LoRA)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        bias: bool = True,
        alpha: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.lora_a = nn.Parameter(
            torch.zeros((r, in_features), dtype=dtype, device=device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros((out_features, r), dtype=dtype, device=device)
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.linear.reset_parameters()
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, alpha: float = 1.0) -> Self:
        """Creates a LinearWithLoRA from an existing nn.Linear layer.

        Args:
            linear: The existing nn.Linear layer to convert.
            r: The rank of the LoRA adaptation.
            alpha: The scaling factor for the LoRA adaptation.
        """
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            bias=linear.bias is not None,  # pyright: ignore[reportUnnecessaryComparison]
            alpha=alpha,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        lora.linear.weight = nn.Parameter(linear.weight.clone())
        if linear.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            lora.linear.bias = nn.Parameter(linear.bias.clone())

        return lora

    @property
    def weight(self) -> Float[Tensor, "out_features in_features"]:
        return self.linear.weight + self.scaling * (self.lora_b @ self.lora_a)

    @property
    def bias(self) -> Float[Tensor, " out_features"] | None:
        return self.linear.bias

    def forward(
        self, x: Float[Tensor, "*batch in_features"]
    ) -> Float[Tensor, "*batch out_features"]:
        return F.linear(x, self.weight, self.bias)
