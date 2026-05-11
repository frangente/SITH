# Suppressing Spurious Correlations (Sec. 5.1)

This experiment evaluates whether SITH can identify and suppress background-related features in CLIP to reduce spurious correlations, reproducing Table 3 of the paper. The evaluation uses the [Waterbirds](https://github.com/kohpangwei/group_DRO) dataset, which contains birds placed on backgrounds that do not match their natural habitats, and reports both overall and worst-group zero-shot classification accuracy.

The pipeline has two steps:

1. **Classify singular vectors** — use GPT to decide which singular vectors encode background/location information.
2. **Edit and evaluate** — zero out the singular values of background-related vectors and measure accuracy on Waterbirds.

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

### Waterbirds Dataset

Download the Waterbirds dataset and place it under `data/waterbirds/` (if the dataset does not exist, the evaluation script will attempt to download it automatically).

## Step 1: Classify Singular Vectors

For each layer, query GPT to score each singular vector on a 1–5 scale indicating how likely its concept set describes a background or outdoor location (1 = definitely not background, 5 = definitely background). Run from the repository root:

```bash
for layer in 20 21 22 23; do
    uv run python experiments/spurious/query_gpt.py \
        data/models/ViT-L-14/laion2b_s32b_b82k/decompositions/layer-${layer}_right_foldln_conceptnet_comp-0.3_sparsity-5.pt \
        --api-key <your-openai-api-key>
done
```

Results are written to `spurious-decisions/<decomposition-stem>.txt`, one integer per line (one per singular vector across all heads). If the output file already exists, the script skips it.

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--gpt-model` | `gpt-5-mini` | OpenAI model used as judge |
| `--api-key` | `None` | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |
| `--prompt-file` | `prompt.txt` | Path to the prompt template |
| `--num-parallel-queries` | `256` | Number of concurrent API requests |

## Step 2: Edit the Model and Evaluate on Waterbirds

Once all four decision files exist under `spurious-decisions/`, run the evaluation:

```bash
uv run python experiments/spurious/test_waterbirds.py
```

The script evaluates the unmodified OpenCLIP model first, then applies the edits (singular values with score ≥ 3 are zeroed out) and re-evaluates. It prints per-group accuracy for each combination of bird label × background label, as well as overall accuracy.

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--model-name` | `ViT-L-14` | OpenCLIP model architecture |
| `--pretrained` | `laion2b_s32b_b82k` | Pretrained weights identifier |
| `--batch-size` | `32` | Inference batch size |
| `--num-workers` | `4` | DataLoader worker processes |
| `--device` | `cuda` / `cpu` | Inference device |
