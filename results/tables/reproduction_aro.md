### Reproduction on ARO: published vs. measured

| Benchmark | Model | Published (%) | Source | Measured (%) | 95% CI | Std over shuffle seeds | Measured − published |
|---|---|---|---|---|---|---|---|
| VG-Relation (macro accuracy) | CLIP ViT-B/32 (OpenAI) | 59 | ARO, Table 6 | 59.90 | [57.14, 63.02] |  | +0.9 |
| VG-Relation (macro accuracy) | NegCLIP released, QuickGELU (as trained) | 81 | ARO, Table 6 | 81.76 | [79.78, 83.61] |  | +0.8 |
| VG-Relation (macro accuracy) | NegCLIP released, GELU (as the released evaluation code builds it) | 81 | ARO, Table 6 | 80.22 | [78.18, 82.14] |  | -0.8 |
| VG-Relation (per-item accuracy) | CLIP ViT-B/32 (OpenAI) | 51.12 | SugarCrepe, Table 8 | 51.12 | [50.61, 51.64] |  | +0.0 |
| VG-Relation (per-item accuracy) | NegCLIP released, QuickGELU (as trained) |  |  | 58.75 | [58.25, 59.24] |  |  |
| VG-Relation (per-item accuracy) | NegCLIP released, GELU (as the released evaluation code builds it) |  |  | 58.33 | [57.82, 58.83] |  |  |
| VG-Attribution (macro accuracy) | CLIP ViT-B/32 (OpenAI) | 62 | ARO, Table 6 | 63.20 | [61.64, 64.73] |  | +1.2 |
| VG-Attribution (macro accuracy) | NegCLIP released, QuickGELU (as trained) | 71 | ARO, Table 6 | 72.07 | [70.63, 73.55] |  | +1.1 |
| VG-Attribution (macro accuracy) | NegCLIP released, GELU (as the released evaluation code builds it) | 71 | ARO, Table 6 | 70.52 | [69.12, 72.03] |  | -0.5 |
| VG-Attribution (per-item accuracy) | CLIP ViT-B/32 (OpenAI) |  |  | 61.33 | [60.40, 62.21] |  |  |
| VG-Attribution (per-item accuracy) | NegCLIP released, QuickGELU (as trained) |  |  | 72.11 | [71.21, 72.97] |  |  |
| VG-Attribution (per-item accuracy) | NegCLIP released, GELU (as the released evaluation code builds it) |  |  | 70.87 | [69.95, 71.73] |  |  |
| COCO-Order (accuracy, 5 choices) | CLIP ViT-B/32 (OpenAI) | 46 | ARO, Table 6 | 47.39 | [46.52, 47.96] | 0.26 | +1.4 |
| COCO-Order (accuracy, 5 choices) | NegCLIP released, QuickGELU (as trained) | 86 | ARO, Table 6 | 82.49 | [81.94, 82.99] | 0.15 | -3.5 |
| COCO-Order (accuracy, 5 choices) | NegCLIP released, GELU (as the released evaluation code builds it) | 86 | ARO, Table 6 | 86.82 | [86.43, 87.33] | 0.25 | +0.8 |
| COCO-Order (accuracy, 5 choices) | CLIP ViT-B/32 (OpenAI) | 37.14 | SugarCrepe, Table 8 | 47.39 | [46.52, 47.96] | 0.26 | +10.3 |

- Published and measured values are in separate columns; the published ones were run once by the authors and are given with two decimals.
- 95% CI: cluster bootstrap over images. For COCO-Order the measured value is the mean over the shuffle seeds and the interval is for the first seed.
- NegCLIP was fine-tuned from OpenAI CLIP, which uses the QuickGELU activation. Building the model with GELU instead is what `open_clip.create_model_and_transforms('ViT-B-32', pretrained=<file>)` does in the released evaluation code.
