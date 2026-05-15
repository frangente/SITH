# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Script to evaluate the effect of editing CLIP on the Waterbirds dataset."""

import argparse
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import sith
from sith import linalg, utils
from sith.datasets import BinaryWaterbirdsDataset
from sith.modeling import (
    ClassificationHead,
    ImageEncoder,
    MultiheadAttentionWithOV,
    replace_attn,
)
from sith.prompts.openai import IMAGENET_TEMPLATES


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser


def get_multiplier(decisions: list[int], idx: int) -> float:
    """Returns the multiplier corresponding to a decision."""
    match decisions[idx]:
        case 1:
            return 1.0
        case 2:
            return 1.0
        case 3:
            return 0.0
        case 4:
            return 0.0
        case 5:
            return 0.0
        case _:
            msg = f"Invalid decision {decisions[idx]} at index {idx}"
            raise ValueError(msg)


def edit_model(model: ImageEncoder) -> ImageEncoder:
    replace_attn(
        model,
        MultiheadAttentionWithOV,
        fold_ln=True,
        center_weights=True,
        layer_indices=[20, 21, 22, 23],
    )

    for l_idx in [20, 21, 22, 23]:
        file = f"layer-{l_idx}_right_foldln_conceptnet_comp-0.3_sparsity-5.txt"
        file = Path(__file__).parent / "spurious-decisions" / file
        if not file.exists():
            msg = f"File {file} not found. Please run the query_gpt.py script first to generate it."  # noqa: E501
            raise FileNotFoundError(msg)

        with open(file) as f:
            decisions = [int(line.strip()) for line in f]

        attn = model.transformer.resblocks[l_idx].attn
        attn = cast("MultiheadAttentionWithOV", attn)
        w_vo = attn.get_vo_matrix()  # (num_heads, emb_dim, emb_dim)
        u_vo, s_vo, vh_vo = linalg.svd(w_vo, topk=len(decisions) // w_vo.shape[0])

        new_s_vo = s_vo.flatten(0, 1)
        multipliers = [get_multiplier(decisions, i) for i in range(new_s_vo.shape[0])]
        multipliers = torch.as_tensor(multipliers, device=new_s_vo.device)
        new_s_vo = new_s_vo * multipliers
        new_s_vo = new_s_vo.reshape_as(s_vo)
        new_w_vo = u_vo @ torch.diag_embed(new_s_vo) @ vh_vo

        attn.set_vo_matrix(new_w_vo)

    return model


def evaluate(
    model: ImageEncoder,
    classifier: ClassificationHead,
    loader: DataLoader[tuple[Tensor, int]],
    dataset: BinaryWaterbirdsDataset,
    device: str,
) -> dict[str, float]:
    model.eval()
    classifier.eval()

    all_logits = []
    for images, _ in tqdm(loader):
        images = images.to(device, non_blocking=True)
        logits = classifier(model(images))
        all_logits.append(logits.cpu())

    labels = torch.as_tensor(dataset.labels)
    places_labels = torch.as_tensor(dataset.places_labels)
    all_logits = torch.cat(all_logits, dim=0)

    metrics: dict[str, float] = {}
    for bl in range(2):  # [land, water]
        for pl in range(2):  # [land bird, water bird]
            # compute the accuracy for each group to see if the model is biased by
            # the background
            mask = (places_labels == pl) & (labels == bl)
            group_acc = utils.accuracy(all_logits[mask], labels[mask], k=1)
            metrics[f"{(bl, pl)}"] = group_acc * 100

    acc = utils.accuracy(all_logits, labels, k=1)
    metrics["full"] = acc * 100

    return metrics


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    dataset = BinaryWaterbirdsDataset("data", "test", download=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    clip, _, transform = sith.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    clip = clip.eval()
    dataset.transform = transform

    classifier = sith.create_classification_head(
        clip,
        classes=dataset.classes,
        templates=IMAGENET_TEMPLATES,
        device=args.device,
    )
    model = clip.image_encoder
    del clip

    # Evaluate the unmodified model
    base_results = evaluate(model, classifier, loader, dataset, args.device)
    print("Base model results:")
    for group, acc in base_results.items():
        print(f"{group}: {acc:.2f}%")

    # Edit the model to remove the bias towards the background
    edited_model = edit_model(model)
    edited_results = evaluate(edited_model, classifier, loader, dataset, args.device)
    print("\nEdited model results:")
    for group, acc in edited_results.items():
        print(f"{group}: {acc:.2f}%")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
