### Text-side preference: accuracy when the image is replaced by another item's image

| Model | Seeds | VG-Relation | VG-Attribution | COCO-Order | SugarCrepe |
|---|---|---|---|---|---|
| CLIP ViT-B/32 (OpenAI) [not trained here] |  | 45.5 | 47.2 | 6.8 | 51.4 |
| NegCLIP released, QuickGELU (as trained) [not trained here] |  | 64.4 | 53.3 | 16.7 | 59.3 |
| none (CLIP-FT control) | 3 | 32.7 ± 0.4 | 45.9 ± 0.2 | 2.2 ± 0.0 | 54.7 ± 0.1 |
| random caption (control) | 3 | 32.1 ± 0.5 | 45.9 ± 0.1 | 2.2 ± 0.1 | 54.9 ± 0.0 |
| swap (NegCLIP-style) | 3 | 52.0 ± 0.9 | 53.3 ± 0.3 | 14.1 ± 0.2 | 58.1 ± 0.1 |
| swap_nosym (ablation) | 3 | 52.3 ± 0.5 | 53.1 ± 0.1 | 14.2 ± 0.3 | 58.1 ± 0.2 |
| swap_fluent (fluency-matched) | 3 | 45.9 ± 0.7 | 50.9 ± 0.3 | 6.9 ± 0.1 | 57.4 ± 0.1 |

- Accuracy (%) when each item's captions are scored against the image of a different test item; mean ± standard deviation over training seeds; COCO-Order with shuffle seed 0. If a model's choice depended only on the image, it would be near chance (50%; 20% for COCO-Order). A higher value means the model prefers the true caption from the text alone.
- Pre-registered secondary check of H3 (HYPOTHESES.md): on the ARO tasks, none < swap_fluent < swap.
