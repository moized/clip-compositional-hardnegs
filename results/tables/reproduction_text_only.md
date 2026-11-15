### Text-only models on ARO: published vs. measured

| Benchmark | Grammar model, published | Grammar model, measured | Vera, published (not re-run) | GPT-2 mean log-probability, measured | Headline metric, grammar model / GPT-2, measured |
|---|---|---|---|---|---|
| VG-Relation (per-item accuracy) | 59.55 | 59.55 | 61.71 | 58.79 | 83.4 / 82.8 |
| VG-Attribution (per-item accuracy) | 58.38 | 58.38 | 82.59 | 74.71 | 56.2 / 72.4 |
| COCO-Order (per-item accuracy) | 74.33 | 92.25 | 59.81 | 96.72 | 92.2 / 96.7 |

- Published values: SugarCrepe, Table 8. Text-only models (no image). Vera is a plausibility model, Grammar is a CoLA classifier. This project re-implements the Grammar model and adds GPT-2; Vera is not re-run.
- The measured values in the first columns are per-item accuracies. On the two Visual Genome tasks they agree with the published grammar-model values to within 0.00 points, which indicates that the published values are per-item accuracies as well. The last column gives the same measured models on ARO's own headline metric (macro accuracy for the two Visual Genome tasks).
- COCO-Order: the value published by the SugarCrepe authors differs from ours, as does their CLIP accuracy on this task (37.14, against 46 in ARO), so their version of the task is probably not identical to the released ARO code used here.
