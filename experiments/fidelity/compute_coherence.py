# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Script to query GPT to assess the coherence of the reconstructions."""

import argparse
import asyncio
import logging
from pathlib import Path

import torch
from openai import AsyncOpenAI
from torch import Tensor
from tqdm.asyncio import tqdm

_logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file",
        type=Path,
        help="The file containing the reconstructions to evaluate.",
    )

    parser.add_argument("--gpt-model", type=str, default="gpt-5-mini")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path(__file__).parent / "prompt.txt",
    )

    parser.add_argument(
        "--num-parallel-queries",
        type=int,
        default=256,
        help="The number of parallel queries to send to GPT.",
    )

    return parser


def fill_prompt(
    template: str,
    scores: Tensor,
    indices: Tensor,
    atoms: list[str],
    idx: int,
) -> str:
    """Fills the prompt template with the concepts corresponding to the given index."""
    i_scores, sorted_idx = torch.sort(scores[idx], descending=True)
    i_indices = indices[idx][sorted_idx]

    mask = i_scores > 0
    i_scores = i_scores[mask]
    i_indices = i_indices[mask]
    if len(i_scores) == 0:
        msg = f"No positive concepts for index {idx}."
        raise ValueError(msg)

    i_scores = i_scores.tolist()
    i_indices = i_indices.tolist()

    concepts = [
        f"- {i_scores[i]:.2f} {atoms[i_indices[i]]}" for i in range(len(i_scores))
    ]
    prompt_filled = template.format(concepts="\n".join(concepts))
    return prompt_filled


async def query_one(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
) -> tuple[int, str] | None:
    try:
        response = await client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "high"},
            text={"verbosity": "low"},
        )

        text = response.output[-1].content[0].text  # pyright: ignore[]

        theme, score = text.split("\n")
        theme = theme.removeprefix("Theme: ").strip()
        score = int(score.removeprefix("Score: ").strip())
    except Exception:
        _logger.exception("Failed to query GPT")
        return None

    return score, theme


async def query_all(
    client: AsyncOpenAI,
    prompts: list[str],
    model: str,
    num_parallel_queries: int,
) -> list[tuple[int, str] | None]:
    decisions = []

    for i in tqdm(range(0, len(prompts), num_parallel_queries)):
        batch_prompts = prompts[i : i + num_parallel_queries]
        batch_decisions = await asyncio.gather(*[
            query_one(client, prompt, model) for prompt in batch_prompts
        ])
        decisions.extend(batch_decisions)

    return decisions


def main(args: argparse.Namespace) -> None:
    file = args.file.resolve()

    out_file = Path(f"coherence-results/{file.stem}.txt")
    out_file.parent.mkdir(exist_ok=True)
    if out_file.exists():
        print(f"Output file {out_file} already exists, skipping...")
        return

    print(f"Evaluating coherence for {file}...")

    codes = torch.load(file, map_location="cpu")
    scores, indices = codes["scores"], codes["indices"]  # (num_heads, rank, sparsity)
    scores = scores.flatten(0, 1)  # (num_heads * rank, sparsity)
    indices = indices.flatten(0, 1)  # (num_heads * rank, sparsity)

    dict_name = file.stem.split("_")[3]
    dict_path = Path("data/dictionaries") / f"{dict_name}.txt"
    with open(dict_path) as f:
        atoms = f.read().splitlines()

    with open(args.prompt_file) as f:
        prompt_template = f.read()

    prompts = [
        fill_prompt(prompt_template, scores, indices, atoms, idx)
        for idx in range(scores.shape[0])
    ]
    client = AsyncOpenAI(api_key=args.api_key)
    decisions = asyncio.run(
        query_all(
            client,
            prompts,
            model=args.gpt_model,
            num_parallel_queries=args.num_parallel_queries,
        )
    )

    with open(out_file, "w") as f:
        for decision in decisions:
            if decision is None:
                f.write("Error\n")
            else:
                score, theme = decision
                f.write(f"{theme},{score}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
