# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

from typing import Self, cast

import torch.nn.functional as F
from open_clip.model import CLIP as OpenCLIP  # noqa: N811
from open_clip.transformer import (
    AttentionalPooler,
    PatchDropout,
    Transformer,
    VisionTransformer,
    text_global_pool,
)
from torch import Tensor, nn


class ImageEncoder(nn.Module):
    """A transformer-based image encoder for CLIP-like models."""

    grid_size: tuple[int, int]
    final_ln_after_pool: bool
    output_dim: int

    conv1: nn.Conv2d
    class_embedding: nn.Parameter
    positional_embedding: nn.Parameter
    patch_dropout: PatchDropout | nn.Identity
    ln_pre: nn.Module
    transformer: Transformer

    attn_pool_type: str | None
    pool_type: str
    attn_pool: AttentionalPooler | None
    attn_pool_contrastive: AttentionalPooler | None
    ln_post: nn.Module

    proj: nn.Linear | nn.Identity

    @classmethod
    def from_open_clip(cls, transformer: VisionTransformer) -> Self:
        """Creates an ImageEncoder from an OpenCLIP VisionTransformer."""
        encoder = cls()

        encoder.grid_size = transformer.grid_size
        encoder.final_ln_after_pool = transformer.final_ln_after_pool
        encoder.output_dim = transformer.output_dim

        encoder.conv1 = transformer.conv1
        encoder.class_embedding = transformer.class_embedding
        encoder.positional_embedding = transformer.positional_embedding
        encoder.patch_dropout = transformer.patch_dropout
        encoder.ln_pre = transformer.ln_pre
        encoder.transformer = transformer.transformer

        encoder.attn_pool_type = getattr(transformer, "attn_pool_type", None)
        encoder.pool_type = transformer.pool_type
        encoder.attn_pool = transformer.attn_pool
        encoder.attn_pool_contrastive = getattr(
            transformer, "attn_pool_contrastive", None
        )
        encoder.ln_post = transformer.ln_post

        proj = transformer.proj
        if proj is not None:
            in_features, out_features = proj.shape
            encoder.proj = nn.Linear(in_features, out_features, bias=False)
            encoder.proj.weight = nn.Parameter(proj.T)
        else:
            encoder.proj = nn.Identity()

        return encoder

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass through the image encoder."""
        x = self._embeds(image)  # type: ignore
        x = self.transformer(x)
        pooled, _ = self._pool(x)  # type: ignore
        pooled = self.proj(pooled)
        return pooled

    _embeds = VisionTransformer._embeds  # pyright: ignore[reportPrivateUsage]

    _pool = VisionTransformer._pool  # pyright: ignore[reportPrivateUsage]

    _global_pool = VisionTransformer._global_pool  # pyright: ignore[reportPrivateUsage]


class TextEncoder(nn.Module):
    """A transformer-based text encoder for CLIP-like models."""

    context_length: int
    vocab_size: int
    eos_id: int | None

    token_embedding: nn.Embedding
    positional_embedding: nn.Parameter
    attn_mask: Tensor
    transformer: Transformer
    pool_type: str
    ln_final: nn.Module

    proj: nn.Linear | nn.Identity

    @classmethod
    def from_open_clip(cls, model: OpenCLIP) -> Self:
        """Creates a TextEncoder from an OpenCLIP TextTransformer."""
        encoder = cls()

        encoder.context_length = model.context_length
        encoder.vocab_size = model.vocab_size
        encoder.eos_id = getattr(model, "text_eos_id", None)

        encoder.token_embedding = cast("nn.Embedding", model.token_embedding)
        encoder.positional_embedding = cast("nn.Parameter", model.positional_embedding)
        encoder.register_buffer(
            "attn_mask", cast("Tensor", model.attn_mask), persistent=False
        )
        encoder.transformer = model.transformer
        encoder.pool_type = cast("str", model.text_pool_type)
        encoder.ln_final = cast("nn.Module", model.ln_final)

        proj = model.text_projection
        if isinstance(proj, nn.Linear):
            encoder.proj = proj
        elif isinstance(proj, Tensor):
            in_features, out_features = proj.shape
            encoder.proj = nn.Linear(in_features, out_features, bias=False)
            encoder.proj.weight = nn.Parameter(proj.T)
        else:
            encoder.proj = nn.Identity()

        return encoder

    def forward(self, text: Tensor) -> Tensor:
        """Forward pass through the text encoder."""
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = text_global_pool(x, text, self.pool_type, self.eos_id)
        x = self.proj(x)

        return x


class CLIP(nn.Module):
    """A CLIP-like model."""

    def __init__(
        self,
        model_name: str,
        pretrained: str | None,
        image: ImageEncoder,
        text: TextEncoder,
        logit_scale: Tensor,
        logit_bias: Tensor | None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained

        self.image_encoder = image
        self.text_encoder = text

        self.logit_scale = nn.Parameter(logit_scale)
        self.logit_bias = nn.Parameter(logit_bias) if logit_bias is not None else None

    def encode_image(self, x: Tensor, *, normalize: bool = False) -> Tensor:
        """Encodes an image tensor into a feature vector."""
        features = self.image_encoder(x)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, x: Tensor, *, normalize: bool = False) -> Tensor:
        """Encodes a text tensor into a feature vector."""
        features = self.text_encoder(x)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features
