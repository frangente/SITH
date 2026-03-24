# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

from typing import Self, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

import sith.linalg


class MultiheadAttention(nn.Module):
    """A multi-head attention module.

    This differs from `torch.nn.MultiheadAttention` in that it does not use a combined
    query-key-value projection, but instead has separate linear layers for the query,
    key, value, and output projections even when the dimensions are the same.
    """

    num_heads: int
    embed_dim: int
    head_dim: int
    dropout: float

    q_lin: nn.Linear
    k_lin: nn.Linear
    v_lin: nn.Linear
    o_lin: nn.Linear

    def __init__(self) -> None:
        msg = "Use from_torch_mha to create an instance of this class"
        raise NotImplementedError(msg)

    @classmethod
    @torch.no_grad()
    def from_torch_mha(
        cls,
        mha: nn.MultiheadAttention,
        *,
        ln: nn.LayerNorm | None = None,
        center_weights: bool = False,
    ) -> Self:
        """Converts a torch.nn.MultiheadAttention to this custom MultiheadAttention.

        Args:
            mha: The torch.nn.MultiheadAttention instance to convert.
            ln: An optional `LayerNorm` instance to fold in the attention weights.
            center_weights: Whether to center the attention weights by subtracting the
                mean from the rows of reading matrices (query, key, value) and the
                columns of writing matrices (output). This ensures that the attention
                operation does not read/write along the all-ones direction.

        Returns:
            An instance of the custom MultiheadAttention class.
        """
        c_mha = cls.__new__(cls)
        nn.Module.__init__(c_mha)

        c_mha.num_heads = mha.num_heads
        c_mha.embed_dim = mha.embed_dim
        c_mha.head_dim = mha.head_dim
        c_mha.dropout = mha.dropout

        device = mha.in_proj_weight.device
        dtype = mha.in_proj_weight.dtype
        c_mha.q_lin = nn.Linear(
            c_mha.embed_dim, c_mha.embed_dim, device=device, dtype=dtype
        )
        c_mha.k_lin = nn.Linear(
            c_mha.embed_dim, c_mha.embed_dim, device=device, dtype=dtype
        )
        c_mha.v_lin = nn.Linear(
            c_mha.embed_dim, c_mha.embed_dim, device=device, dtype=dtype
        )
        c_mha.o_lin = nn.Linear(
            c_mha.embed_dim, c_mha.embed_dim, device=device, dtype=dtype
        )

        w_qkv = mha.in_proj_weight  # (3 * embed_dim, embed_dim)

        b_qkv = mha.in_proj_bias  # (3 * embed_dim,)

        if ln is not None:
            if ln.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                b_qkv = b_qkv + w_qkv @ ln.bias
            if ln.weight is not None:  # pyright: ignore[reportUnnecessaryComparison]
                w_qkv = w_qkv * ln.weight

        if center_weights:
            w_qkv = w_qkv - w_qkv.mean(dim=1, keepdim=True)

        w_q, w_k, w_v = w_qkv.chunk(3, dim=0)  # 3 x (embed_dim, embed_dim)
        c_mha.q_lin.weight.data.copy_(w_q)
        c_mha.k_lin.weight.data.copy_(w_k)
        c_mha.v_lin.weight.data.copy_(w_v)

        b_q, b_k, b_v = b_qkv.chunk(3, dim=0)
        c_mha.q_lin.bias.data.copy_(b_q)
        c_mha.k_lin.bias.data.copy_(b_k)
        c_mha.v_lin.bias.data.copy_(b_v)

        w_o = mha.out_proj.weight  # (embed_dim, embed_dim)
        if center_weights:
            w_o = w_o - w_o.mean(dim=0, keepdim=True)
        c_mha.o_lin.weight.data.copy_(w_o)

        b_o = mha.out_proj.bias  # (embed_dim,)
        c_mha.o_lin.bias.data.copy_(b_o)

        return c_mha

    def get_qk_matrix(self) -> Float[Tensor, "num_heads embed_dim embed_dim"]:
        """Returns the query-key weight matrix."""
        # nn.Linear stores weights for left-multiplication: (out_features, in_features)
        w_q = self.q_lin.weight  # (embed_dim, embed_dim)
        w_q = w_q.reshape(self.num_heads, self.head_dim, self.embed_dim)
        w_q = w_q.transpose(1, 2)  # (num_heads, embed_dim, head_dim)

        w_k = self.k_lin.weight  # (embed_dim, embed_dim)
        w_k = w_k.reshape(self.num_heads, self.head_dim, self.embed_dim)
        w_k = w_k.transpose(1, 2)  # (num_heads, embed_dim, head_dim)

        return torch.matmul(w_q, w_k.transpose(-2, -1))

    def get_vo_matrix(self) -> Float[Tensor, "num_heads embed_dim embed_dim"]:
        """Returns the value-output weight matrix."""
        w_v = self.v_lin.weight  # (embed_dim, embed_dim)
        w_v = w_v.reshape(self.num_heads, self.head_dim, self.embed_dim)
        w_v = w_v.transpose(1, 2)  # (num_heads, embed_dim, head_dim)

        w_o = self.o_lin.weight  # (embed_dim, embed_dim)
        w_o = w_o.reshape(self.embed_dim, self.num_heads, self.head_dim)
        w_o = w_o.permute(1, 2, 0)  # (num_heads, head_dim, embed_dim)

        return torch.matmul(w_v, w_o)

    @torch.no_grad()
    def set_vo_matrix(self, vo: Float[Tensor, "num_heads embed_dim embed_dim"]) -> None:
        """Updates the value-output weight matrix.

        Since this class does not actually use an explicit value-output matrix but
        instead has separate value and output matrices, this method decomposes the given
        value-output matrix using SVD, and sets the value and output matrices to the two
        resulting factors.

        Note:
            This method folds the value bias into the output bias before
            updating the value and output matrices.
        """
        self.fold_value_bias()
        # u: (num_heads, embed_dim, head_dim)
        # s: (num_heads, head_dim)
        # vh: (num_heads, head_dim, embed_dim)
        u, s, vh = sith.linalg.svd(vo, topk=self.head_dim)
        s_sqrt = s.sqrt()
        w_v = u * s_sqrt.unsqueeze(1)  # (num_heads, embed_dim, head_dim)
        w_v = w_v.transpose(1, 2)  # (num_heads, head_dim, embed_dim)
        w_v = w_v.flatten(0, 1)  # (num_heads * head_dim, embed_dim)
        self.v_lin.weight.data.copy_(w_v)

        w_o = vh * s_sqrt.unsqueeze(2)  # (num_heads, head_dim, embed_dim)
        w_o = w_o.flatten(0, 1)  # (num_heads * head_dim, embed_dim)
        w_o = w_o.transpose(0, 1)  # (embed_dim, num_heads * head_dim)
        self.o_lin.weight.data.copy_(w_o)

    @torch.no_grad()
    def fold_value_bias(self) -> None:
        """Folds the value bias into the output bias."""
        w_o = self.o_lin.weight.T  # (embed_dim, embed_dim)
        b_v = self.v_lin.bias @ w_o  # (embed_dim,)
        self.o_lin.bias.data += b_v
        self.v_lin.bias.data.zero_()

    @override
    def forward(
        self,
        query: Float[Tensor, "batch_size q_len embed_dim"],
        key: Float[Tensor, "batch_size k_len embed_dim"],
        value: Float[Tensor, "batch_size k_len embed_dim"],
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Applies multi-head attention to the input tensors."""
        B, Q, _ = query.shape
        K = key.shape[1]

        q = self.q_lin(query)
        q = q.view(B, Q, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (B, num_heads, Q, head_dim)

        k = self.k_lin(key)
        k = k.view(B, K, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, K, head_dim)

        v = self.v_lin(value)
        v = v.view(B, K, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, K, head_dim)

        if need_weights:
            msg = "need_weights=True is not supported"
            raise NotImplementedError(msg)

        output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout,
        )  # (B, num_heads, Q, head_dim)

        output = output.transpose(1, 2)  # (B, Q, num_heads, head_dim)
        output = output.flatten(2)  # (B, Q, embed_dim)
        output = self.o_lin(output)  # (B, Q, embed_dim)
        return output, None

    @override
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"dropout={self.dropout}"
        )


class MultiheadAttentionWithOV(nn.Module):
    """A multi-head attention module with an explicit value-output matrix."""

    num_heads: int
    embed_dim: int
    head_dim: int
    dropout: float

    q_lin: nn.Linear
    k_lin: nn.Linear
    vo_lin: nn.Linear

    def __init__(self) -> None:
        msg = "Use from_torch_mha to create an instance of this class"
        raise NotImplementedError(msg)

    @classmethod
    @torch.no_grad()
    def from_torch_mha(
        cls,
        mha: nn.MultiheadAttention,
        *,
        ln: nn.LayerNorm | None = None,
        center_weights: bool = False,
    ) -> Self:
        c_mha = cls.__new__(cls)
        nn.Module.__init__(c_mha)

        c_mha.num_heads = mha.num_heads
        c_mha.embed_dim = mha.embed_dim
        c_mha.head_dim = mha.head_dim
        c_mha.dropout = mha.dropout

        device = mha.in_proj_weight.device
        dtype = mha.in_proj_weight.dtype
        c_mha.q_lin = nn.Linear(
            c_mha.embed_dim, c_mha.embed_dim, device=device, dtype=dtype
        )
        c_mha.k_lin = nn.Linear(
            c_mha.embed_dim, c_mha.embed_dim, device=device, dtype=dtype
        )
        c_mha.vo_lin = nn.Linear(
            c_mha.embed_dim,
            c_mha.embed_dim * c_mha.num_heads,
            device=device,
            dtype=dtype,
        )

        w_qkv = mha.in_proj_weight  # (3 * embed_dim, embed_dim)

        b_qkv = mha.in_proj_bias  # (3 * embed_dim,)

        if ln is not None:
            if ln.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                b_qkv = b_qkv + w_qkv @ ln.bias
            if ln.weight is not None:  # pyright: ignore[reportUnnecessaryComparison]
                w_qkv = w_qkv * ln.weight

        if center_weights:
            w_qkv = w_qkv - w_qkv.mean(dim=1, keepdim=True)

        w_q, w_k, w_v = w_qkv.chunk(3, dim=0)  # 3 x (embed_dim, embed_dim)
        c_mha.q_lin.weight.data.copy_(w_q)
        c_mha.k_lin.weight.data.copy_(w_k)

        b_q, b_k, b_v = b_qkv.chunk(3, dim=0)
        c_mha.q_lin.bias.data.copy_(b_q)
        c_mha.k_lin.bias.data.copy_(b_k)

        w_v = w_v.reshape(c_mha.num_heads, c_mha.head_dim, c_mha.embed_dim)

        w_o = mha.out_proj.weight  # (embed_dim, embed_dim)
        if center_weights:
            w_o = w_o - w_o.mean(dim=0, keepdim=True)
        b_v = w_o @ b_v  # (embed_dim,) fold value bias into output bias

        w_o = w_o.reshape(c_mha.embed_dim, c_mha.num_heads, c_mha.head_dim)
        w_o = w_o.permute(1, 0, 2)  # (num_heads, embed_dim, head_dim)
        w_ov = torch.matmul(w_o, w_v)  # (num_heads, embed_dim, embed_dim)
        w_ov = w_ov.flatten(0, 1)  # (num_heads * embed_dim, embed_dim)
        c_mha.vo_lin.weight.data.copy_(w_ov)

        b_o = mha.out_proj.bias  # (embed_dim,)
        b_o = b_o + b_v
        b_o = b_o / c_mha.num_heads  # average bias across heads
        b_o = b_o.repeat(c_mha.num_heads)  # (num_heads * embed_dim,)
        c_mha.vo_lin.bias.data.copy_(b_o)

        return c_mha

    def get_qk_matrix(self) -> Float[Tensor, "num_heads embed_dim embed_dim"]:
        """Returns the query-key weight matrix.

        Returns:
            A tensor of shape `(num_heads, embed_dim, embed_dim)`.
        """
        # nn.Linear stores weights for left-multiplication: (out_features, in_features)
        w_q = self.q_lin.weight  # (embed_dim, embed_dim)
        w_q = w_q.reshape(self.num_heads, self.head_dim, self.embed_dim)
        w_q = w_q.transpose(1, 2)  # (num_heads, embed_dim, head_dim)

        w_k = self.k_lin.weight  # (embed_dim, embed_dim)
        w_k = w_k.reshape(self.num_heads, self.head_dim, self.embed_dim)
        w_k = w_k.transpose(1, 2)  # (num_heads, embed_dim, head_dim)

        return torch.matmul(w_q, w_k.transpose(-2, -1))

    def get_vo_matrix(self) -> Float[Tensor, "num_heads embed_dim embed_dim"]:
        """Returns the value-output weight matrix."""
        w = self.vo_lin.weight  # (num_heads * embed_dim, embed_dim)

        w = w.view(self.num_heads, self.embed_dim, self.embed_dim)
        # from left-multiplication to right-multiplication
        w = w.transpose(1, 2)  # (num_heads, embed_dim, embed_dim)
        return w

    @torch.no_grad()
    def set_vo_matrix(self, w: Float[Tensor, "num_heads embed_dim embed_dim"]) -> None:
        """Updates the value-output weight matrix."""
        w = w.transpose(1, 2)  # (num_heads, embed_dim, embed_dim)
        w = w.flatten(0, 1)  # (num_heads * embed_dim, embed_dim)
        self.vo_lin.weight.data.copy_(w)

    @override
    def forward(
        self,
        query: Float[Tensor, "batch_size q_len embed_dim"],
        key: Float[Tensor, "batch_size k_len embed_dim"],
        value: Float[Tensor, "batch_size k_len embed_dim"],
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Applies multi-head attention to the input tensors."""
        B, Q, _ = query.shape
        K = key.shape[1]

        q = self.q_lin(query)
        q = q.view(B, Q, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (B, num_heads, Q, head_dim)

        k = self.k_lin(key)
        k = k.view(B, K, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, K, head_dim)

        v = self.vo_lin(value)  # (B, K, num_heads * embed_dim)
        v = v.view(B, K, self.num_heads, self.embed_dim)
        v = v.transpose(1, 2)  # (B, num_heads, K, embed_dim)

        if not need_weights:
            output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=self.dropout,
            )
        else:
            raise NotImplementedError

        # output: (B, num_heads, Q, embed_dim)
        output = torch.sum(output, dim=1)  # (B, Q, embed_dim)

        return output, None
