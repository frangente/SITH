# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Script to encode a text pool using an OpenCLIP text-encoder."""

import argparse
from pathlib import Path

import open_clip
import torch
from tqdm import tqdm


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode a text pool using an OpenCLIP text-encoder."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-L-14",
        help="The name of the model to use for encoding.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s32b_b82k",
        help="The pretrained weights of the model.",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="conceptnet",
        help="The name of the dictionary to encode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="The batch size to use for encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to use for encoding.",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    input_path = Path(f"data/dictionaries/{args.dictionary}.txt")
    with input_path.open("r") as f:
        dictionary = f.read().splitlines()

    print(
        f"Encoding {len(dictionary)} texts from the '{args.dictionary}' dictionary "
        f"using {args.model_name} ({args.pretrained})"
    )

    model = open_clip.create_model(
        args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    all_embeddings = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(dictionary), args.batch_size)):
            batch = dictionary[i : i + args.batch_size]
            tokens = tokenizer(batch).to(args.device, non_blocking=True)
            embeddings = model.encode_text(tokens, normalize=False)  # pyright: ignore[reportCallIssue]
            all_embeddings.append(embeddings.to("cpu", non_blocking=True))

    all_embeddings = torch.cat(all_embeddings, dim=0)
    output_path = (
        Path("data/models")
        / args.model_name
        / args.pretrained
        / f"dictionaries/{args.dictionary}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_embeddings, output_path)

    print(f"Encoded text pool saved to {output_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
