# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: Apache-2.0

"""Script to evaluate the effect of editing CLIP on the NSFW benchmark."""

import argparse
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import sith
from sith import linalg
from sith.datasets import ViSUDataset
from sith.modeling import (
    CLIP,
    ImageEncoder,
    MultiheadAttentionWithOV,
    replace_attn,
)


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
        case 1 | 2:
            return 1.0
        case 3:
            return -1.0
        case 4 | 5:
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
        file = Path("nsfw-decisions") / file
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


def compute_embeds(clip: CLIP, loader: DataLoader, device: str) -> dict[str, Tensor]:  # pyright: ignore[reportMissingTypeArgument]
    safe_img_embeds = []
    unsafe_img_embeds = []
    safe_txt_embeds = []
    unsafe_txt_embeds = []

    with torch.inference_mode():
        for safe_imgs, safe_txts, unsafe_imgs, unsafe_txts in tqdm(loader):
            safe_ims = safe_imgs.to(device, non_blocking=True)
            unsafe_ims = unsafe_imgs.to(device, non_blocking=True)
            safe_txts = safe_txts.to(device, non_blocking=True)
            unsafe_txts = unsafe_txts.to(device, non_blocking=True)

            safe_img_embeds.append(clip.encode_image(safe_ims, normalize=True))
            unsafe_img_embeds.append(clip.encode_image(unsafe_ims, normalize=True))
            safe_txt_embeds.append(clip.encode_text(safe_txts, normalize=True))
            unsafe_txt_embeds.append(clip.encode_text(unsafe_txts, normalize=True))

    safe_img = torch.cat(safe_img_embeds, dim=0)
    unsafe_img = torch.cat(unsafe_img_embeds, dim=0)
    safe_txt = torch.cat(safe_txt_embeds, dim=0)
    unsafe_txt = torch.cat(unsafe_txt_embeds, dim=0)
    img = torch.cat([safe_img, unsafe_img], dim=0)
    txt = torch.cat([safe_txt, unsafe_txt], dim=0)

    return {
        "safe_img": safe_img,
        "unsafe_img": unsafe_img,
        "safe_txt": safe_txt,
        "unsafe_txt": unsafe_txt,
        "img": img,
        "txt": txt,
    }


def recall(sims: Tensor, k: int = 1) -> float:
    """Computes recall@k for similarity matrix."""
    topk = sims.topk(k, dim=1).indices
    targets = torch.arange(sims.shape[0], device=sims.device).unsqueeze(1)
    correct = (topk == targets).any(dim=1).float()
    return correct.mean().item() * 100.0


def print_recall_table(embeds: dict[str, Tensor]) -> None:
    """Prints recall table for all settings and ks."""
    # (source, target) pairs for which to compute recall
    # T = text, V = image, T* = unsafe text, V* = unsafe image
    settings = [
        ("safe_txt", "safe_img"),  # T -> V
        ("safe_img", "safe_txt"),  # V -> T
        ("safe_txt", "img"),  # T -> V + V*
        ("safe_img", "txt"),  # V -> T + T*
        ("unsafe_txt", "img"),  # T* -> V + V*
        ("unsafe_img", "txt"),  # V* -> T + T*
    ]

    print(f"{'Setting':<20} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>10}")
    for source, target in settings:
        sims = embeds[source] @ embeds[target].T
        recalls = [recall(sims, k) for k in [1, 5, 10]]
        print(
            f"{f'{source} -> {target}':<20} {recalls[0]:>10.2f} {recalls[1]:>10.2f} {recalls[2]:>10.2f}"  # noqa: E501
        )


def collate_fn(tokenizer: Any, batch: Any) -> Any:
    safe_imgs, safe_txts, unsafe_imgs, unsafe_txts = zip(*batch, strict=True)
    safe_imgs = torch.stack(safe_imgs, dim=0)
    unsafe_imgs = torch.stack(unsafe_imgs, dim=0)

    safe_txts = tokenizer(safe_txts)
    unsafe_txts = tokenizer(unsafe_txts)

    return safe_imgs, safe_txts, unsafe_imgs, unsafe_txts


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    clip, _, transform = sith.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    clip = clip.eval()
    tokenizer = sith.get_tokenizer(args.model_name)

    dataset = ViSUDataset("data", split="test", transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, tokenizer),
    )

    # Evaluate the unmodified model
    base_embeds = compute_embeds(clip, loader, args.device)
    print("Base model results:")
    print_recall_table(base_embeds)

    # Edit the model to remove unsafe concepts and evaluate again
    clip.image_encoder = edit_model(clip.image_encoder)
    edited_embeds = compute_embeds(clip, loader, args.device)
    print("\nEdited model results:")
    print_recall_table(edited_embeds)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
