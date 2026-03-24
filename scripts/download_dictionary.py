# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Script to download the text pools used as dictionaries."""

import argparse
from pathlib import Path

import gdown

URLS = {
    "conceptnet": "https://drive.google.com/file/d/1v364gM1r8I_AhFhQZBJvYLiIcPPC_CIJ/view",
    "efros": "https://drive.google.com/file/d/1plIjcMxQEYc0GZ6BoT7NyvQVkdyiGVE-/view",
    "laion": "https://drive.google.com/file/d/1KrVHzyepf3eCLBNKKD85anGNUYccakMa/view",
    "wordnet": "https://drive.google.com/file/d/1mCspIEEh6NhtnjeLECmRv3oN2Nh7A1fI/view",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the text pools used as dictionaries."
    )
    parser.add_argument(
        "--dictionary",
        choices=["conceptnet", "efros", "laion", "wordnet"],
        default="conceptnet",
        help="The dictionary to download.",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    url = URLS[args.dictionary]

    output_path = Path(f"data/dictionaries/{args.dictionary}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdown.download(url, output=output_path, quiet=False, fuzzy=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
