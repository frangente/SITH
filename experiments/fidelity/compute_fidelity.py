# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Script to compute the fidelity of the reconstructions."""

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sith
import sith.linalg
from sith.modeling import MultiheadAttentionWithOV, replace_attn


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        type=Path,
        help="The file containing the reconstructions to evaluate.",
    )

    return parser


def get_info_from_path(path: Path) -> dict[str, Any]:
    layer, type_, fold_ln, dictionary, method, sparsity = path.stem.split("_")
    return {
        "model_name": path.parent.parent.parent.name,
        "pretrained": path.parent.parent.name,
        "layer": int(layer.split("-")[-1]),
        "type": type_,
        "fold_ln": fold_ln == "foldln",
        "dictionary": dictionary,
        "method": method,
        "sparsity": int(sparsity.split("-")[-1]),
    }


def main(file: Path) -> None:
    file = file.resolve()
    print(f"Evaluating fidelity for {file}...")

    info = get_info_from_path(file)

    codes = torch.load(file, map_location="cpu")
    scores, indices = codes["scores"], codes["indices"]  # (num_heads, rank, sparsity)
    if not scores.any(dim=-1).all():
        msg = "Found zero scores, cannot compute fidelity."
        raise RuntimeError(msg)

    clip = sith.create_model(info["model_name"], pretrained=info["pretrained"])
    model = clip.image_encoder
    replace_attn(
        model,
        MultiheadAttentionWithOV,
        fold_ln=info["fold_ln"],
        center_weights=info["fold_ln"],
        layer_indices=[info["layer"]],
    )

    dict_path = (
        Path("data/models")
        / info["model_name"]
        / info["pretrained"]
        / "dictionaries"
        / f"{info['dictionary']}.pt"
    )
    dictionary = torch.load(dict_path, map_location="cpu")
    dictionary = F.normalize(dictionary, dim=-1)
    dictionary = dictionary - dictionary.mean(dim=0)
    dictionary = F.normalize(dictionary, dim=-1)

    inv_proj = torch.linalg.pinv(model.proj.weight.T)
    image_mean = torch.load(file.parent.parent / "image_mean.pt", map_location="cpu")

    attn = model.transformer.resblocks[info["layer"]].attn
    assert isinstance(attn, MultiheadAttentionWithOV)  # noqa: S101
    w_vos = attn.get_vo_matrix()  # (num_heads, in_dim, out_dim)

    assert w_vos.shape[0] == scores.shape[0] == indices.shape[0]  # noqa: S101
    rank = scores.shape[1]

    all_sims = []
    num_print_zeros = len(str(w_vos.shape[0]))
    for head_idx in range(w_vos.shape[0]):
        w_vo = w_vos[head_idx]  # (in_dim, out_dim)
        head_scores = scores[head_idx]  # (rank, sparsity)
        head_indices = indices[head_idx]  # (rank, sparsity)

        u_vo, _, vh_vo = sith.linalg.svd(w_vo, topk=rank)
        x_vo = u_vo.T if info["type"] == "left" else vh_vo  # (rank, dim)

        head_codes = torch.zeros(rank, len(dictionary))
        head_codes.scatter_(1, head_indices, head_scores)
        head_recon = head_codes @ dictionary
        head_recon = F.normalize(head_recon, dim=-1)
        head_recon = head_recon + image_mean
        head_recon = F.normalize(head_recon, dim=-1)
        head_recon = head_recon @ inv_proj
        head_recon = F.normalize(head_recon, dim=-1)

        head_sims = F.cosine_similarity(head_recon, x_vo, dim=-1)
        print(
            f"Head {head_idx:0{num_print_zeros}d}: "
            f"mean sim = {head_sims.mean().item():.4f}, "
            f"std sim = {head_sims.std().item():.4f}, "
            f"max sim = {head_sims.max().item():.4f}, "
            f"min sim = {head_sims.min().item():.4f}",
        )

        all_sims.append(head_sims)

    all_sims = torch.cat(all_sims, dim=0)
    print(f"Overall fidelity: {all_sims.mean().item():.4f}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.file)
