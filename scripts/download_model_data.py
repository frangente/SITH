# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Download all pre-computed resources for a given model."""

import argparse
from pathlib import Path

import gdown

URLS = {
    "ViT-L-14/laion2b_s32b_b82k": "https://drive.google.com/drive/folders/1zmNAyKwMW4UqXowePB4xIcdSCDgXT0aA",
    "ViT-H-14/laion2b_s32b_b79k": "https://drive.google.com/drive/folders/1eIM_lfh1C2KdPGvZpxIsX4zyVOGeVWKZ",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-L-14",
        help="The name of the model for which to download the pre-computed data.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s32b_b82k",
        help="The pretrained weights of the model.",
    )

    return parser


def main(args: argparse.Namespace) -> None:

    url = URLS.get(f"{args.model_name}/{args.pretrained}")
    if url is None:
        msg = (
            f"No pre-computed data available for model {args.model_name} "
            f"with pretrained weights {args.pretrained}."
        )
        raise ValueError(msg)

    output = Path("data/models") / args.model_name / args.pretrained
    output.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(url, output=str(output), quiet=False, resume=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
