# Improving Classification Performance (Sec. 5.3)

This experiment evaluates whether SITH can improve zero-shot classification accuracy by amplifying task-relevant singular vectors and dampening irrelevant ones, reproducing Table 5 of the paper. Unlike the spurious and NSFW experiments, no GPT calls are required: task relevance is computed automatically from the class label embeddings.

The evaluation runs on three datasets: Flowers 102, FGVC-Aircraft, and DTD (Describable Textures).

## Setup

### Decompositions

The experiment requires COMP decompositions with `λ=0.3` and `K=5` for layers 20–23:

```
data/models/ViT-L-14/laion2b_s32b_b82k/decompositions/
├── layer-20_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
├── layer-21_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
├── layer-22_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
└── layer-23_right_foldln_conceptnet_comp-0.3_sparsity-5.pt
```

### Datasets

Flowers 102, FGVC-Aircraft, and DTD are downloaded automatically under `data/` on first run.

## Running the Experiment

Run from the repository root, chaning the `--dataset` argument to evaluate on each dataset:

```bash
# see below for all options
uv run python experiments/classification/test.py --dataset flowers --fold-ln --filter-dictionary
```

Each run prints the accuracy of the unmodified OpenCLIP model followed by the accuracy of the edited model.

## How It Works

The editing pipeline (Sec. A.7 of the supplementary) has two stages:

1. **Build the task concept pool.** Each class label is encoded with CLIP's text encoder and decomposed into constituent concepts using COMP (`--class-sparsity` controls sparsity, `--filter-dictionary` removes the class names themselves from the candidate pool to surface fundamental attributes like colors and textures rather than trivial label matches). The union of concepts across all classes forms the task pool Γ_task.

2. **Scale singular values.** For each right singular vector, a relevance score R(v) is computed as the weighted cosine similarity between its COMP concepts and Γ_task. The singular value is then rescaled by α = max(`--min-threshold`, R(v) + `--offset`), amplifying task-relevant vectors (α > 1) and dampening irrelevant ones (α < 1) without zeroing any out entirely.

## Options

| Argument | Default | Description |
|---|---|---|
| `--model-name` | `ViT-L-14` | OpenCLIP model architecture |
| `--pretrained` | `laion2b_s32b_b82k` | Pretrained weights identifier |
| `--dataset` | `flowers` | Dataset to evaluate (`flowers`, `aircraft`, `textures`) |
| `--fold-ln` | off | Fold layer norm into VO matrices (must match decomposition files) |
| `--dictionary` | `conceptnet` | Concept dictionary to use |
| `--layers` | `20 21 22 23` | Layers to edit |
| `--class-template` | `{class_name}` | Prompt template for encoding class labels (must contain `{class_name}`) |
| `--class-sparsity` | `10` | Number of concepts used to decompose each class embedding |
| `--filter-dictionary` | off | Remove class names from the concept pool before decomposing class embeddings |
| `--offset` | `0.3` | τ: shifts relevance scores so that above-average vectors are amplified |
| `--min-threshold` | `0.8` | Floor on the scaling factor (prevents complete suppression) |
| `--max-threshold` | `None` | Ceiling on the scaling factor |
| `--batch-size` | `32` | Inference batch size |
| `--num-workers` | `8` | DataLoader worker processes |
| `--device` | `cuda` / `cpu` | Inference device |
