### Cost of fine-tuning: zero-shot accuracy, retrieval and resources

| Model | Type | Seeds | Zero-shot cifar10 top-1 (%) | Zero-shot cifar100 top-1 (%) | COCO text R@1 (%) | COCO image R@1 (%) | Training time (min) | Seconds per step | Peak GPU memory (GB) | Texts per step |
|---|---|---|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (OpenAI) [not trained here] | measured |  | 89.8 | 64.2 | 50.2 | 30.2 |  |  |  |  |
| NegCLIP released, QuickGELU (as trained) [not trained here] | measured |  | 88.9 | 62.9 | 59.2 | 44.8 |  |  |  |  |
| NegCLIP released, GELU (as the released evaluation code builds it) [not trained here] | measured |  | 85.9 | 60.9 | 56.0 | 41.2 |  |  |  |  |
| none (CLIP-FT control) | measured | 3 | 90.7 ± 0.2 | 67.5 ± 0.3 | 59.7 ± 0.2 | 42.9 ± 0.1 | 24.0 ± 0.3 | 1.45 ± 0.02 | 4.0 ± 0.0 | 256 ± 0 |
| random caption (control) | measured | 3 | 90.7 ± 0.1 | 67.5 ± 0.3 | 60.2 ± 0.2 | 42.8 ± 0.0 | 34.2 ± 0.4 | 2.06 ± 0.02 | 4.4 ± 0.0 | 512 ± 0 |
| swap (NegCLIP-style) | measured | 3 | 90.8 ± 0.3 | 67.4 ± 0.5 | 58.2 ± 0.0 | 42.9 ± 0.1 | 34.3 ± 0.2 | 2.07 ± 0.01 | 4.4 ± 0.0 | 512 ± 0 |
| swap_nosym (ablation) | measured | 3 | 90.8 ± 0.3 | 67.4 ± 0.4 | 58.2 ± 0.1 | 42.9 ± 0.1 | 34.1 ± 0.6 | 2.06 ± 0.04 | 4.4 ± 0.0 | 512 ± 0 |
| swap_fluent (fluency-matched) | measured | 3 | 90.7 ± 0.2 | 67.7 ± 0.5 | 59.1 ± 0.2 | 42.8 ± 0.1 | 34.3 ± 0.9 | 2.07 ± 0.05 | 4.4 ± 0.0 | 512 ± 0 |
| CLIP ViT-B/32 | published (C2LIP, Table A6) |  | 89.8 | 64.3 |  |  |  |  |  |  |
| NegCLIP | published (C2LIP, Table A6) |  | 85.9 | 60.9 |  |  |  |  |  |  |
| CLIP ViT-B/32 | published (ARO, Table 6) |  | 95 | 80 | 50 | 30 |  |  |  |  |
| CLIP-FT | published (ARO, Table 6) |  | 95 | 80 | 59 | 42 |  |  |  |  |
| NegCLIP | published (ARO, Table 6) |  | 94 | 79 | 56 | 41 |  |  |  |  |

- Measured rows: mean ± standard deviation over training seeds. Zero-shot accuracy uses CLIP_benchmark's class names and prompt templates; retrieval is Recall@1 on the COCO Karpathy test split (5,000 images).
- Published rows are other people's measurements and are listed for reference only; ARO does not state its zero-shot protocol.
- 'Texts per step' shows the extra text-encoder work of conditions with negative captions; the 'random' condition has the same amount of work as the swap conditions.
