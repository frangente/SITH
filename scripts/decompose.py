# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Script to decompose the singular vectors of the VO matrices."""

import argparse
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

import sith
import sith.linalg
from sith.modeling import ImageEncoder, MultiheadAttentionWithOV, replace_attn
from sith.sdl import coherent_orthogonal_matching_pursuit as comp
from sith.sdl import matching_pursuit as mp
from sith.sdl import orthogonal_matching_pursuit as omp


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-L-14",
        help="The name of the model from which to extract the VO matrices.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s32b_b82k",
        help="The pretrained weights of the model.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[-1],
        help="The layers from which to extract the VO matrices.",
    )
    parser.add_argument(
        "--fold-ln",
        action="store_true",
        help="Whether to fold the layer normalization into the VO matrices.",
    )
    parser.add_argument(
        "--sv-type",
        choices=["left", "right"],
        default="right",
        help="Whether to decompose the left or right singular vectors of the VO matrices.",  # noqa: E501
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="The rank of the matrix to decompose (i.e., the number of singular vectors to keep).",  # noqa: E501
    )

    parser.add_argument(
        "--method",
        type=str,
        default="comp-0.3",
        help="The sparse dictionary learning method to use for the decomposition.",
    )
    parser.add_argument(
        "--sparsity",
        type=int,
        default=5,
        help="The sparsity level to use for the decomposition.",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="conceptnet",
        help="The name of the dictionary to use for the decomposition.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The number of singular vectors to decompose simultaneously.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to use for the decomposition.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to force the decomposition even if the output file already exists.",  # noqa: E501
    )

    return parser


def load_model_and_data(
    args: argparse.Namespace,
) -> tuple[ImageEncoder, Tensor, Tensor]:
    """Load the CLIP model, dictionary, and image mean."""
    clip = sith.create_model(args.model_name, pretrained=args.pretrained)
    model = clip.image_encoder
    replace_attn(
        model,
        MultiheadAttentionWithOV,
        fold_ln=args.fold_ln,
        center_weights=args.fold_ln,
        layer_indices=args.layers,
    )

    model_dir = Path("models") / f"{args.model_name}_{args.pretrained}"

    dictionary = torch.load(model_dir / "dictionaries" / f"{args.dictionary}.pt")
    dictionary = F.normalize(dictionary, dim=-1)
    dictionary = dictionary - dictionary.mean(dim=0)
    dictionary = F.normalize(dictionary, dim=-1)
    dictionary = dictionary.to(args.device)

    image_mean = torch.load(model_dir / "means" / "cc12m.pt", device=args.device)

    return model, dictionary, image_mean


def decompose_batch(
    x_proj: Float[Tensor, "batch dim"],
    dictionary: Float[Tensor, "dict_size dim"],
    method: str,
    sparsity: int,
) -> tuple[Float[Tensor, "batch sparsity"], Int[Tensor, "batch sparsity"]]:
    """Runs the chosen sparse coding method on a single batch."""
    if method.startswith("comp"):
        coh = float(method.rsplit("-", maxsplit=1)[-1])
        codes = comp(
            x_proj,
            dictionary,
            max_sparsity=sparsity,
            coh=coh,
            positive=True,
        )
        scores, indices = torch.topk(codes, k=sparsity, dim=-1)
    elif method == "omp":
        codes = omp(x_proj, dictionary, max_sparsity=sparsity, positive=True)
        scores, indices = torch.topk(codes, k=sparsity, dim=-1)
    elif method == "mp":
        codes = mp(x_proj, dictionary, max_sparsity=sparsity, positive=True)
        scores, indices = torch.topk(codes, k=sparsity, dim=-1)
    elif method == "topk":
        assert len(x_proj) == 1, "Top-k method only supports batch size of 1."  # noqa: S101
        sims = x_proj[0] @ dictionary.T
        _, indices = torch.topk(sims, k=sparsity, dim=-1)
        basis = dictionary[indices]
        scores = sith.linalg.lstsq(basis.T, x_proj[0], positive=True)
        scores, indices = scores.unsqueeze(0), indices.unsqueeze(0)
    else:
        msg = (
            f"Unknown method {method}. "
            "Supported methods are: comp-{{coh}}, omp, mp, topk."
        )
        raise ValueError(msg)

    return scores, indices


def decompose_layer(
    model: ImageEncoder,
    dictionary: Tensor,
    image_mean: Tensor,
    layer_idx: int,
    args: argparse.Namespace,
) -> None:
    """Decompose the singular vectors of a single layer's VO matrix."""
    attn = model.transformer.resblocks[layer_idx].attn
    attn = cast("MultiheadAttentionWithOV", attn)

    w_vo = attn.get_vo_matrix()
    u, _, vh = sith.linalg.svd(w_vo, topk=args.rank)
    x = u.transpose(-2, -1) if args.sv_type == "left" else vh  # (heads, rank, dim)
    x = x.flatten(0, 1)  # (heads * rank, dim)

    output_file = (
        Path("models")
        / f"{args.model_name}_{args.pretrained}"
        / "decompositions"
        / f"layer-{layer_idx:02d}_{args.sv_type}-{'foldln' if args.fold_ln else 'nofoldln'}_"  # noqa: E501
        f"{args.dictionary}_{args.method}_sparsity-{args.sparsity}.pt"
    )

    if output_file.exists() and not args.force:
        tmp = torch.load(output_file, device="cpu")
        all_scores, all_indices = tmp["scores"], tmp["indices"]
    else:
        all_scores = torch.zeros(len(x), args.sparsity)
        all_indices = torch.zeros(len(x), args.sparsity, dtype=torch.long)

    start = int(all_scores.any(dim=1).sum())
    if start >= len(x):
        print(f"Layer {layer_idx}: all codes already computed, skipping.")
        return

    x_proj = model.proj(model.ln_post(x))
    x_proj = F.normalize(x_proj, dim=-1)
    x_proj = x - image_mean
    x_proj = F.normalize(x_proj, dim=-1)

    for i in tqdm(
        range(start, len(x_proj), args.batch_size), desc=f"Layer {layer_idx}"
    ):
        batch_proj = x_proj[i : i + args.batch_size]
        scores, indices = decompose_batch(
            batch_proj,
            dictionary,
            args.method,
            args.sparsity,
        )

        all_scores[i : i + args.batch_size] = scores
        all_indices[i : i + args.batch_size] = indices

        if (i + args.batch_size) % (args.batch_size * 10) == 0:
            torch.save({"scores": all_scores, "indices": all_indices}, output_file)

    torch.save({"scores": all_scores, "indices": all_indices}, output_file)


def main(args: argparse.Namespace) -> None:
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    model, dictionary, image_mean = load_model_and_data(args)

    for layer_idx in args.layers:
        if layer_idx < 0:
            layer_idx += len(model.transformer.resblocks)

        decompose_layer(model, dictionary, image_mean, layer_idx, args)

    print("Decomposition completed.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
