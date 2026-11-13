# Expected values (development check)

These numbers come from a **development run made by the AI assistant on a CPU while the code was written**
(float32, PyTorch 2.15, open_clip 4.2.2, spaCy 3.8.18; COCO-Order with shuffle seed 0 only).
They are here so that a new run can be checked: a run on a GPU should agree to within
about 0.1 points (COCO-Order and anything that depends on the spaCy version may move a
little more). **They are not the results of the project**; the reported results are the
files under `results/`, produced by the Kaggle notebook.

All values are percentages. "Shuffled image" means the same captions scored against a
different image.

## Released checkpoints on the four benchmarks

| Model | VG-Relation (macro) | VG-Attribution (macro) | COCO-Order (seed 0) | SugarCrepe (mean of 7) |
|---|---|---|---|---|
| CLIP ViT-B/32 (OpenAI), QuickGELU | 59.90 | 63.20 | 47.20 | 73.05 |
| NegCLIP released, QuickGELU | 81.76 | 72.07 | 82.49 | 83.47 |
| NegCLIP released, GELU | 80.22 | 70.52 | 86.80 | 82.50 |
| CLIP ViT-B/32 (OpenAI), QuickGELU, shuffled image | 45.47 | 47.24 | 6.78 | 51.44 |
| NegCLIP released, QuickGELU, shuffled image | 64.40 | 53.28 | 16.67 | 59.28 |
| NegCLIP released, GELU, shuffled image | 53.58 | 53.03 | 22.42 | 58.46 |

Published for comparison (ARO, Table 6): CLIP 59 / 62 / 46, NegCLIP 81 / 71 / 86. SugarCrepe mean of subsets (C2LIP, Table A3): CLIP 73.1, NegCLIP 82.5.

Per-item (micro) accuracy: CLIP ViT-B/32 (OpenAI): VG-Relation 51.12, VG-Attribution 61.33; NegCLIP released, QuickGELU: VG-Relation 58.75, VG-Attribution 72.11; NegCLIP released, GELU: VG-Relation 58.33, VG-Attribution 70.87.

## SugarCrepe subsets

| Model | replace_obj | replace_att | replace_rel | swap_obj | swap_att | add_obj | add_att |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (OpenAI), QuickGELU | 90.92 | 80.08 | 69.20 | 61.22 | 63.96 | 77.21 | 68.79 |
| NegCLIP released, QuickGELU | 93.83 | 87.94 | 74.40 | 75.92 | 77.03 | 90.06 | 85.12 |
| NegCLIP released, GELU | 92.68 | 85.91 | 76.46 | 75.51 | 75.38 | 88.80 | 82.80 |

Published: CLIP 90.92 / 80.08 / 69.20 / 61.38 / 63.96 / 77.21 / 68.79 (SugarCrepe, Table 6; the repository now has 245 instead of 246 swap_obj items). NegCLIP 92.7 / 85.9 / 76.5 / 75.2 / 75.4 / 88.8 / 82.8 (C2LIP, Table A3).

## Text-only baselines (no image)

| Scorer | VG-Relation (macro / per item) | VG-Attribution (macro / per item) | COCO-Order (seed 0) | SugarCrepe (mean of 7) |
|---|---|---|---|---|
| GPT-2, mean log-probability | 82.80 / 58.79 | 72.35 / 74.71 | 96.62 | 49.78 |
| GPT-2, total log-probability | 82.80 / 58.79 | 72.35 / 74.71 | 97.68 | 67.04 |
| Grammar model (CoLA) | 83.44 / 59.55 | 56.23 / 58.38 | 92.38 | 50.03 |

Published grammar model, per item (SugarCrepe, Table 8): VG-Relation 59.55, VG-Attribution 58.38.

Share of test items whose image is in the released NegCLIP training file: vg_relation 42.2%, vg_attribution 42.9%, coco_order 0.0%, sugarcrepe 75.4%.

## Zero-shot classification and COCO retrieval

| Model | CIFAR-10 top-1 | CIFAR-100 top-1 | COCO text R@1 | COCO image R@1 |
|---|---|---|---|---|
| CLIP ViT-B/32 (OpenAI), QuickGELU | 89.83 | 64.20 | 50.20 | 30.18 |
| NegCLIP released, QuickGELU | 88.88 | 62.94 | 59.26 | 44.81 |
| NegCLIP released, GELU | 85.88 | 60.92 | 56.00 | 41.21 |

Published with CLIP_benchmark (C2LIP, Table A6): CLIP 89.8 / 64.3, NegCLIP 85.9 / 60.9 on CIFAR-10 / CIFAR-100. Published COCO Recall@1 (ARO, Table 6): CLIP text 50, image 30; NegCLIP text 56, image 41.

The published NegCLIP values agree with the model built with GELU. Built with QuickGELU, the activation it was fine-tuned with, the same checkpoint loses much less zero-shot accuracy and retrieves better.

## Training negatives (a sample of 1500 images, not the full pool)

- Swap rules reproduce 95.2% of the authors' released negatives (1500 captions);
  GPT-2 alone separates the released negatives from the true caption 94.7% of the time.
- All candidate swaps: GPT-2 alone picks the true caption 96.1% of the time.
- 958 fluency-matched pairs from 1500 images (64% of the images).
- Text-only accuracy on the selected training pairs, GPT-2 / grammar model:
  swap 84.4 / 84.4,
  swap_nosym 83.1 / 86.5,
  swap_fluent 50.0 / 71.8.
- Leak checks: all zero.

## Training loop (smoke test only)

52 steps of `swap` on CPU with batch 32 and a learning rate of 1e-5 (ten times the experiment's),
on 862 pairs, raised the accuracy on 96 held-out pairs from 68.8% to 83.3% against NegCLIP-style
negatives and from 61.5% to 72.9% against fluency-matched negatives. This only shows that the
training signal has the right sign; it says nothing about the experiment's outcome.
