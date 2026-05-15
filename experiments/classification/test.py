# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Script to evaluate the effect of editing CLIP on classification benchmarks."""

import argparse
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sith
from sith import linalg
from sith.datasets import (
    ClassificationDataset,
    DescribableTexturesDataset,
    FGVCAircraftDataset,
    Flowers102Dataset,
)
from sith.modeling import CLIP, ImageEncoder, MultiheadAttentionWithOV, replace_attn
from sith.prompts import Template
from sith.prompts.openai import (
    DESCRIBABLE_TEXTURES_TEMPLATES,
    FGVC_AIRCRAFT_TEMPLATES,
    FLOWERS102_TEMPLATES,
)
from sith.sdl import coherent_orthogonal_matching_pursuit as comp


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test classification model")

    parser.add_argument("--model-name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument("--fold-ln", action="store_true")
    parser.add_argument("--dictionary", type=str, default="conceptnet")
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 21, 22, 23])

    parser.add_argument("--class-template", type=str, default="{class_name}")
    parser.add_argument("--class-sparsity", type=int, default=10)
    parser.add_argument("--filter-dictionary", action="store_true")
    parser.add_argument("--min-threshold", type=float, default=0.8)
    parser.add_argument("--max-threshold", type=float, default=None)
    parser.add_argument("--offset", type=float, default=0.3)

    parser.add_argument(
        "--dataset",
        choices=["flowers", "aircraft", "textures"],
        default="flowers",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    device: torch.device,
) -> float:
    """Returns the accuracy of the model on the given data loader."""
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def get_dataset(
    args: argparse.Namespace, transform: Any
) -> tuple[ClassificationDataset, list[Template]]:
    """Returns the dataset and the corresponding templates."""
    match args.dataset:
        case "flowers":
            dataset = Flowers102Dataset(
                "data", split="test", transform=transform, download=True
            )
            templates = FLOWERS102_TEMPLATES
        case "aircraft":
            dataset = FGVCAircraftDataset(
                "data", split="test", transform=transform, download=True
            )
            templates = FGVC_AIRCRAFT_TEMPLATES
        case "textures":
            dataset = DescribableTexturesDataset(
                "data", split="test", transform=transform, download=True
            )
            templates = DESCRIBABLE_TEXTURES_TEMPLATES
        case _:
            msg = f"Unsupported dataset {args.dataset}"
            raise ValueError(msg)

    return dataset, templates


def edit_model(
    clip: CLIP,
    classes: list[str],
    args: argparse.Namespace,
) -> ImageEncoder:
    tokenizer = sith.get_tokenizer(clip.model_name)

    class_prompts = [
        args.class_template.format(class_name=class_) for class_ in classes
    ]
    class_embeds = clip.encode_text(tokenizer(class_prompts).to(args.device))
    class_embeds = F.normalize(class_embeds, dim=-1)

    with open(Path("data/dictionaries") / f"{args.dictionary}.txt") as f:
        concepts = [line.strip() for line in f]

    dict_path = (
        Path("data/models")
        / args.model_name
        / args.pretrained
        / f"dictionaries/{args.dictionary}.pt"
    )
    dictionary = torch.load(dict_path, map_location=args.device)
    dictionary = F.normalize(dictionary, dim=-1)
    if args.filter_dictionary:
        # remove all concepts which are equal to any class
        class_set = set(classes)
        mask = torch.tensor([c not in class_set for c in concepts], device=args.device)
        f_dictionary = dictionary[mask]
    else:
        f_dictionary = dictionary

    codes = comp(class_embeds, f_dictionary, max_sparsity=args.class_sparsity, coh=0.3)
    selected = codes.any(dim=0)  # (num_concepts,)
    task_dictionary = f_dictionary[selected]  # (num_selected_concepts, dim)

    model = clip.image_encoder
    replace_attn(
        model,
        MultiheadAttentionWithOV,
        fold_ln=args.fold_ln,
        center_weights=args.fold_ln,
        layer_indices=args.layers,
    )

    for layer_idx in args.layers:
        codes_path = (
            Path("data/models")
            / args.model_name
            / args.pretrained
            / "decompositions"
            / f"layer-{layer_idx}_right_{'foldln' if args.fold_ln else 'nofoldln'}_"
            f"{args.dictionary}_comp-0.3_sparsity-5.pt"
        )
        codes = torch.load(codes_path, map_location=args.device)
        scores, indices = codes["scores"], codes["indices"]  # (heads, sv, k)

        # compute weighted similarity between the decomposition of each right SV
        # and the task-related dictionary, and use it to modulate the SV scores
        sel_concepts = dictionary[indices.flatten()]  # (heads * sv * k, dim)
        sel_concepts = sel_concepts.view(*indices.shape, -1)  # (heads, sv, k, dim)
        sims = sel_concepts @ task_dictionary.T  # (heads, sv, k, num_concepts)
        sims = sims.amax(dim=-1)  # (heads, sv, k)
        sims = sims * scores  # (heads, sv, k)
        sims = sims.sum(dim=-1)  # (heads, sv)

        sims = sims + args.offset
        if args.min_threshold is not None:
            sims.clamp_min_(args.min_threshold)
        if args.max_threshold is not None:
            sims.clamp_max_(args.max_threshold)

        attn = model.transformer.resblocks[layer_idx].attn
        attn = cast("MultiheadAttentionWithOV", attn)
        w_vo = attn.get_vo_matrix()  # (num_heads, emb_dim, emb_dim)
        u, s, vh = linalg.svd(w_vo, topk=sims.shape[1])
        s = s * sims
        new_w_vo = u @ torch.diag_embed(s) @ vh
        attn.set_vo_matrix(new_w_vo)

    return model


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    clip, _, transform = sith.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    clip = clip.eval()

    dataset, templates = get_dataset(args, transform)

    classifier = sith.create_classification_head(
        clip,
        classes=dataset.classes,
        templates=templates,
        device=args.device,
    )
    model = nn.Sequential(clip.image_encoder, classifier)

    loader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # evaluate the unmodified model
    acc = evaluate(model, loader, torch.device(args.device))
    print(f"Original model accuracy: {acc:.4f}")

    # edit the model and evaluate again
    model[0] = edit_model(clip, dataset.classes, args)
    acc = evaluate(model, loader, torch.device(args.device))
    print(f"Edited model accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
