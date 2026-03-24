# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

from typing import Any

from torch import nn

from ._attn import MultiheadAttention
from ._clip import ImageEncoder, TextEncoder
from ._lora import LinearWithLoRA


def replace_attn(
    model: ImageEncoder | TextEncoder,
    attn_type: type[Any] = MultiheadAttention,
    *,
    fold_ln: bool = False,
    center_weights: bool = False,
    layer_indices: list[int] | None = None,
) -> None:
    """Replaces the attention layers in the model with a custom implementation.

    Args:
        model: The ImageEncoder or TextEncoder instance to modify.
        attn_type: The type of attention layer to replace with. Must implement
            `from_torch_mha` class method.
        fold_ln: Whether to fold the LayerNorm into the attention weights.
        center_weights: Whether to center the weights by making them orthogonal to the
            vector of ones.
        layer_indices: List of layer indices to replace. If `None`, replaces all
            layers.
    """
    if layer_indices is None:
        layer_indices = list(range(len(model.transformer.resblocks)))

    for idx in set(layer_indices):
        idx = idx if idx >= 0 else len(model.transformer.resblocks) + idx
        block = model.transformer.resblocks[idx]

        attn = block.attn
        ln = block.ln_1 if fold_ln else None
        if not isinstance(attn, nn.MultiheadAttention):
            msg = "Expected block.attn to be an instance of nn.MultiheadAttention."
            raise TypeError(msg)
        if ln is not None and not isinstance(ln, nn.LayerNorm):
            msg = "Expected block.ln_1 to be an instance of nn.LayerNorm."
            raise TypeError(msg)

        block.attn = attn_type.from_torch_mha(
            attn, ln=ln, center_weights=center_weights
        )
        if ln is not None:
            ln.elementwise_affine = False
            ln.weight = None  # pyright: ignore[reportAttributeAccessIssue]
            ln.bias = None  # pyright: ignore[reportAttributeAccessIssue]


def replace_linear_with_lora(
    model: ImageEncoder | TextEncoder,
    r: int,
    alpha: float,
    names: list[str],
    layer_indices: list[int] | None = None,
) -> None:
    """Replaces the linear layers in the model with `LinearWithLoRA`.

    Note:
        This function only replaces the linear layers in the attention modules.

    Args:
        model: The image or text encoder to modify.
        r: The rank of the LoRA decomposition.
        alpha: The scaling factor for the LoRA updates.
        names: The list of linear layer names to replace (e.g., ["q", "k", "v", "o"]
            for attention layers).
        layer_indices: The indices of the layers where the linear layers should be
            replaced. If `None`, replaces all layers.
    """
    if layer_indices is None:
        layer_indices = list(range(len(model.transformer.resblocks)))

    for idx in set(layer_indices):
        idx = idx if idx >= 0 else len(model.transformer.resblocks) + idx
        block = model.transformer.resblocks[idx]

        attn = block.attn
        for name in names:
            if not hasattr(attn, f"{name}_lin"):
                msg = f"Attention layer does not have a '{name}_lin' attribute."
                raise AttributeError(msg)

            linear_layer = getattr(attn, f"{name}_lin")
            if not isinstance(linear_layer, nn.Linear):
                msg = f"Expected '{name}_lin' to be an instance of nn.Linear."
                raise TypeError(msg)

            lora_layer = LinearWithLoRA.from_linear(linear_layer, r=r, alpha=alpha)
            setattr(attn, f"{name}_lin", lora_layer)
