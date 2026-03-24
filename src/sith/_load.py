# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

from collections.abc import Sequence
from typing import Any

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from open_clip.model import CLIP as OpenCLIP  # noqa: N811
from open_clip.tokenizer import HFTokenizer, SigLipTokenizer, SimpleTokenizer
from open_clip.transformer import VisionTransformer
from tqdm import tqdm

from .modeling import (
    CLIP,
    ClassificationHead,
    ImageEncoder,
    TextEncoder,
)
from .prompts import Template


def get_tokenizer(
    model_name: str,
    context_length: int | None = None,
    cache_dir: str | None = None,
    **kwargs: Any,
) -> HFTokenizer | SigLipTokenizer | SimpleTokenizer:
    """Returns the tokenizer for a given model."""
    return open_clip.get_tokenizer(
        model_name,
        context_length=context_length,
        cache_dir=cache_dir,
        **kwargs,
    )


def create_model(
    model_name: str,
    pretrained: str | None = None,
    *,
    device: str | torch.device = "cpu",
    force_quick_gelu: bool | None = None,
) -> CLIP:
    """Creates a CLIP model.

    Args:
        model_name: The name of the CLIP model to create.
        pretrained: The source of the pretrained weights for the model.
        device: The device to load the model on.
        force_quick_gelu: Whether to force the use of QuickGELU in the model.
            If `None`, a default value is determined based on the model name
            and the pretrained weights.

    Returns:
        A CLIP model instance.
    """
    if force_quick_gelu is None:
        force_quick_gelu = _force_quick_gelu(model_name, pretrained)

    oc_model = open_clip.create_model(
        model_name=model_name,
        pretrained=pretrained,
        force_quick_gelu=force_quick_gelu,
        device=device,
    )
    if not isinstance(oc_model, OpenCLIP):
        msg = "Currently only standard CLIP models are supported."
        raise TypeError(msg)
    if not isinstance(oc_model.visual, VisionTransformer):
        msg = "Currently only ViT-based encoders are supported."
        raise TypeError(msg)

    image_encoder = ImageEncoder.from_open_clip(oc_model.visual)
    text_encoder = TextEncoder.from_open_clip(oc_model)

    model = CLIP(
        model_name=model_name,
        pretrained=pretrained,
        image=image_encoder,
        text=text_encoder,
        logit_scale=oc_model.logit_scale,
        logit_bias=oc_model.logit_bias,
    )
    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: str | None = None,
    *,
    device: str | torch.device = "cpu",
    force_quick_gelu: bool | None = None,
) -> tuple[CLIP, T.Compose, T.Compose]:
    """Creates a CLIP model and its associated transforms.

    Args:
        model_name: The name of the CLIP model to create.
        pretrained: The source of the pretrained weights for the model.
        device: The device to load the model on.
        force_quick_gelu: Whether to force the use of QuickGELU in the model.
            If `None`, a default value is determined based on the model name
            and the pretrained weights.

    Returns:
        A tuple containing the CLIP model, training transforms, and validation
        transforms.
    """
    if force_quick_gelu is None:
        force_quick_gelu = _force_quick_gelu(model_name, pretrained)

    oc_model, train_t, val_t = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        force_quick_gelu=force_quick_gelu,
        device=device,
    )
    if not isinstance(oc_model, OpenCLIP):
        msg = "Currently only standard CLIP models are supported."
        raise TypeError(msg)
    if not isinstance(oc_model.visual, VisionTransformer):
        msg = "Currently only ViT-based encoders are supported."
        raise TypeError(msg)

    image_encoder = ImageEncoder.from_open_clip(oc_model.visual)
    text_encoder = TextEncoder.from_open_clip(oc_model)

    model = CLIP(
        model_name=model_name,
        pretrained=pretrained,
        image=image_encoder,
        text=text_encoder,
        logit_scale=oc_model.logit_scale,
        logit_bias=oc_model.logit_bias,
    )

    return model, train_t, val_t  # pyright: ignore[reportReturnType]


def create_classification_head(
    model: CLIP,
    classes: Sequence[str] | int,
    *,
    templates: Sequence[Template] | Template = lambda c: f"a photo of a {c}.",
    bias: bool | None = None,
    use_logit_scale: bool = True,
    use_logit_bias: bool = True,
    quiet: bool = False,
    device: str | torch.device = "cpu",
) -> ClassificationHead:
    """Creates a classification head for a CLIP model.

    Args:
        model: The CLIP model to create the classification head for.
        classes: The number of classes or a sequence of class names to classify.
            If a list of class names is provided, the provided model will be used to
            encode the class names and initialize the weight matrix of the
            classification head.
        templates: The templates to use as input to the text encoder to generate the
            class embeddings. If multiple templates are provided, the class embeddings
            obtained from each template will be averaged together to form the final
            class embedding.
        bias: Whether to include a bias term in the classification head. If `None`,
            whether to include a bias term will be determined as follows
            - if the class names are provided, the bias term will always be included;
            - if the class names are provided, the bias term will be included only if
                the CLIP model has a logit bias and `use_logit_bias` is `True`.
        use_logit_scale: Whether to multiply the class embeddings by the logit scale
            of the CLIP model.
        use_logit_bias: Whether to initialize the bias term of the classification head
            with the logit bias of the CLIP model (if available).
        quiet: If `True`, suppresses the progress bar when encoding class names.
        device: The device to load the classification head on.

    Returns:
        The classification head.
    """
    dim = model.image_encoder.output_dim
    if isinstance(classes, int):
        bias = bias if bias is not None else True
        return ClassificationHead(dim, classes, bias=bias, device=device)

    if bias is None:
        bias = model.logit_bias is not None and use_logit_bias
    head = ClassificationHead(dim, len(classes), bias=bias, device=device)

    with torch.no_grad():
        tokenizer = get_tokenizer(model.model_name)
        templates = templates if isinstance(templates, Sequence) else [templates]
        weights = []
        for class_name in tqdm(classes, desc="Encoding class names", disable=quiet):
            tokens = tokenizer([t(class_name) for t in templates])
            embeds = model.encode_text(tokens.to(device))  # (N, D)
            embeds = F.normalize(embeds, dim=-1)
            embeds = torch.mean(embeds, dim=0)  # (D,)
            embeds = F.normalize(embeds, dim=-1)
            weights.append(embeds)

        weights = torch.stack(weights, dim=0)  # (N, D)
        if use_logit_scale:
            weights *= model.logit_scale.exp()

        head.weight.data.copy_(weights)
        if bias and use_logit_bias and model.logit_bias is not None:
            head.bias.data.copy_(model.logit_bias)  # type: ignore[]

    return head


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _force_quick_gelu(model_name: str, pretrained: str | None) -> bool:
    """Returns whether to force QuickGELU for a given model."""
    return pretrained == "openai" or "quickgelu" in model_name
