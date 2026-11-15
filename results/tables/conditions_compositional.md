### Fine-tuning conditions on the compositional benchmarks

| Model | Seeds | VG-Relation | VG-Attribution | COCO-Order | SugarCrepe | SugarCrepe replace (pooled) | SugarCrepe swap (pooled) | SugarCrepe add (pooled) |
|---|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (OpenAI) [not trained here] |  | 59.9 | 63.2 | 47.2 | 73.1 | 80.8 | 63.2 | 75.1 |
| NegCLIP released, QuickGELU (as trained) [not trained here] |  | 81.8 | 72.1 | 82.5 | 83.5 | 85.5 | 76.7 | 88.8 |
| none (CLIP-FT control) | 3 | 59.5 ± 0.2 | 64.5 ± 0.3 | 34.0 ± 0.1 | 79.2 ± 0.2 | 84.0 ± 0.1 | 68.6 ± 0.7 | 86.9 ± 0.1 |
| random caption (control) | 3 | 59.8 ± 0.0 | 64.5 ± 0.2 | 33.8 ± 0.5 | 79.1 ± 0.2 | 84.1 ± 0.1 | 68.4 ± 0.6 | 86.7 ± 0.3 |
| swap (NegCLIP-style) | 3 | 80.1 ± 0.2 | 71.2 ± 0.2 | 85.5 ± 0.2 | 81.8 ± 0.0 | 84.8 ± 0.2 | 73.6 ± 0.5 | 88.2 ± 0.1 |
| swap_nosym (ablation) | 3 | 80.1 ± 0.2 | 71.4 ± 0.1 | 85.4 ± 0.2 | 81.8 ± 0.1 | 84.8 ± 0.1 | 73.5 ± 0.6 | 87.9 ± 0.1 |
| swap_fluent (fluency-matched) | 3 | 75.9 ± 0.2 | 70.1 ± 0.1 | 69.7 ± 0.4 | 81.8 ± 0.1 | 84.6 ± 0.1 | 73.8 ± 0.2 | 87.7 ± 0.2 |

- Mean ± standard deviation over training seeds, in %. All conditions use the same images, captions, steps, batch size and optimiser; only the negative caption differs.
- Headline metrics: VG-Relation: macro accuracy; VG-Attribution: macro accuracy; COCO-Order: accuracy, 5 choices; SugarCrepe: mean of 7 subsets. COCO-Order uses the first shuffle seed for every model.
- The released NegCLIP row is not comparable in compute: it was trained on more pairs and with hard negative images.
