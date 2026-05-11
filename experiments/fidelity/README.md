# Interpretability-Fidelity Analysis (Sec. 4.1)

This experiment evaluates the quality of SITH decompositions along two complementary axes:

- **Fidelity** — how faithfully the selected concepts reconstruct the original singular vector, measured as cosine similarity between the original vector and its reconstruction.
- **Coherence** — how semantically coherent the concept set is, assessed by an LLM-as-a-judge (GPT-5-mini) that rates each concept set on a 1–5 Likert scale (1 = incoherent, 5 = clearly monosemantic).

These two metrics correspond to Figure 3 of the paper, which compares COMP, NNOMP, and top-k across varying sparsity levels K ∈ {5, 10, 20, 50} on the last four layers of OpenCLIP ViT-L/14.

## Setup

The scripts expect pre-computed decompositions under `data/` (see the [main README](../../README.md) for how to obtain or recompute them). The analysis in the paper uses the following decompositions:

```
data/models/ViT-L-14/laion2b_s32b_b82k/decompositions/
├── layer-{20..23}_{right}_{foldln}_{conceptnet}_{top-k}_sparsity-{5,10,20,50}.pt
├── layer-{20..23}_{right}_{foldln}_{conceptnet}_{nnomp}_sparsity-{5,10,20,50}.pt
└── layer-{20..23}_{right}_{foldln}_{conceptnet}_{comp-0.2,comp-0.3,comp-0.4}_sparsity-{5,10,20,50}.pt
```

## Computing Fidelity

Run from the repository root:

```bash
uv run python experiments/fidelity/compute_fidelity.py \
    data/models/ViT-L-14/laion2b_s32b_b82k/decompositions/layer-23_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
```

The script prints per-head fidelity statistics (mean, std, min, max cosine similarity) and the overall mean fidelity across all heads and singular vectors.

## Computing Coherence

Coherence evaluation queries the OpenAI API (GPT-5-mini by default) and requires an API key:

```bash
uv run python experiments/fidelity/compute_coherence.py \
    data/models/ViT-L-14/laion2b_s32b_b82k/decompositions/layer-23_right_foldln_conceptnet_comp-0.3_sparsity-5.pt \
    --api-key <your-openai-api-key>
```

Results are written to `coherence-results/<decomposition-stem>.txt` (relative to the script directory), with one line per singular vector in the format `<theme>,<score>`. If the output file already exists, the script skips the evaluation.

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--gpt-model` | `gpt-5-mini` | OpenAI model used as judge |
| `--api-key` | `None` | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |
| `--prompt-file` | `prompt.txt` | Path to the prompt template |
| `--num-parallel-queries` | `256` | Number of concurrent API requests |
