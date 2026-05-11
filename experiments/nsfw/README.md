# Removing NSFW Concepts (Sec. 5.2)

This experiment evaluates whether SITH can suppress unsafe concepts in CLIP without retraining, reproducing Table 4 of the paper. The evaluation uses the [ViSU](https://github.com/aimagelab/safe-clip) dataset, which contains paired safe and unsafe image-caption samples, and reports text-to-image and image-to-text recall under both safe and unsafe queries.

The pipeline mirrors the spurious correlations experiment:

1. **Classify singular vectors** — use GPT to score each singular vector on a safety scale.
2. **Edit and evaluate** — suppress or invert unsafe singular values and measure retrieval performance on ViSU.

## Setup

### Decompositions

The experiment uses COMP decompositions with `λ=0.3` and `K=5` for layers 20–23:

```
data/models/ViT-L-14/laion2b_s32b_b82k/decompositions/
├── layer-20_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
├── layer-21_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
├── layer-22_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
└── layer-23_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
```

### ViSU Dataset

Unfortunately, the ViSU dataset cannot be redistributed. To obtain it, you first need to request access to the text portion of the dataset via [Hugging Face](https://huggingface.co/datasets/aimagelab/ViSU-Text). Once you have access, you need to generate the unsafe images yourself by following the pipeline described in the [ViSU paper](https://arxiv.org/abs/2311.16254).

## Step 1: Classify Singular Vectors

For each layer, query GPT to score each singular vector on a 1–5 safety scale (1 = completely safe, 5 = extremely unsafe). Run from the repository root:

```bash
for layer in 20 21 22 23; do
    uv run python experiments/nsfw/query_gpt.py \
        data/models/ViT-L-14/laion2b_s32b_b82k/decompositions/layer-${layer}_right_foldln_conceptnet_comp-0.3_sparsity-5.pt \
        --api-key <your-openai-api-key>
done
```

Results are written to `nsfw-decisions/<decomposition-stem>.txt`, one integer per line (one per singular vector across all heads). If the output file already exists, the script skips it.

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--gpt-model` | `gpt-5-mini` | OpenAI model used as judge |
| `--api-key` | `None` | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |
| `--prompt-file` | `prompt.txt` | Path to the prompt template |
| `--num-parallel-queries` | `256` | Number of concurrent API requests |

## Step 2: Edit the Model and Evaluate on ViSU

Once all four decision files exist under `nsfw-decisions/`, run the evaluation:

```bash
uv run python experiments/nsfw/test_visu.py
```

The script evaluates the unmodified OpenCLIP model first, then applies the edits and re-evaluates. It prints Recall@1, @5, and @10 for all retrieval settings (safe/unsafe text and image queries against the full pool of safe and unsafe content).

The editing strategy differs slightly from the spurious experiment: scores 1–2 leave the singular value unchanged, score 3 (mixed safety) **negates** it (multiplier −1), and scores 4–5 zero it out (multiplier 0).

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--model-name` | `ViT-L-14` | OpenCLIP model architecture |
| `--pretrained` | `laion2b_s32b_b82k` | Pretrained weights identifier |
| `--batch-size` | `32` | Inference batch size |
| `--num-workers` | `4` | DataLoader worker processes |
| `--device` | `cuda` / `cpu` | Inference device |
