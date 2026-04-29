# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Script to compute the mean of image embeddings."""

import argparse
from pathlib import Path

import datasets
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-L-14",
        help="The name of the model from which to extract the image embeddings.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s32b_b82k",
        help="The pretrained weights of the model.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="The batch size to use for computing the image embeddings.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="The number of workers to use for loading the dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to use for computation.",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    model, _, transform = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )

    dataset = datasets.load_dataset("pixparse/cc12m-wds", split="train", streaming=True)
    dataset = dataset.map(lambda x: {"jpg": transform(x["jpg"])})  # pyright: ignore[reportCallIssue]

    loader = DataLoader(
        dataset,  # pyright: ignore[reportArgumentType]
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: torch.stack([item["jpg"] for item in x]),
    )

    with torch.inference_mode():
        total = 0
        embeds_sum = None
        for images in tqdm(loader, desc="Computing image embeddings"):
            images = images.to(args.device, non_blocking=True)
            embeddings = model.encode_image(images, normalize=True)  # pyright: ignore[reportCallIssue]
            embeddings = embeddings.cpu()

            if embeds_sum is None:
                embeds_sum = embeddings.sum(dim=0)
            else:
                embeds_sum += embeddings.sum(dim=0)
            total += images.size(0)

    if embeds_sum is None:
        print("No images found in the dataset.")
        return

    mean = embeds_sum / total
    output_path = (
        Path("data/models") / args.model_name / args.pretrained / "image_mean.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean, output_path)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
