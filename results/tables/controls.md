### Controls: what can be solved without looking at the image

| Method | Type | VG-Relation | VG-Attribution | COCO-Order | SugarCrepe |
|---|---|---|---|---|---|
| Chance | by construction | 50.0 | 50.0 | 20.0 | 50.0 |
| Text only: GPT-2, mean log-probability | measured | 82.8 | 72.4 | 96.7 | 49.8 |
| Text only: GPT-2, total log-probability | measured | 82.8 | 72.4 | 97.7 | 67.0 |
| Text only: grammar model (CoLA) | measured | 83.4 | 56.2 | 92.2 | 50.0 |
| CLIP ViT-B/32 (OpenAI), image replaced by another image | measured | 45.5 | 47.2 | 6.9 | 51.4 |
| NegCLIP released, QuickGELU (as trained), image replaced by another image | measured | 64.4 | 53.3 | 16.8 | 59.3 |
| CLIP ViT-B/32 (OpenAI) | measured | 59.9 | 63.2 | 47.4 | 73.1 |
| NegCLIP released, QuickGELU (as trained) | measured | 81.8 | 72.1 | 82.5 | 83.5 |

- All values are percentages on the benchmark's headline metric (VG-Relation: macro accuracy; VG-Attribution: macro accuracy; COCO-Order: accuracy, 5 choices; SugarCrepe: mean of 7 subsets).
- A text-only model picks the caption a language model prefers and never sees the image.
- Shuffled image: the model scores the same captions against a different image, so only its text-side preference can produce a correct answer.
- Published text-only numbers use per-item accuracy and are compared like for like in the table 'reproduction_text_only'.
