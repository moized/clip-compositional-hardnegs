### Validation pairs (held out from training, not a test set)

| Condition | When | Seeds | True caption vs. fluency-matched negative (%) | True caption vs. NegCLIP-style negative (%) | True caption vs. swap_nosym negative (%) | Caption retrieval within a batch (%) |
|---|---|---|---|---|---|---|
| none (CLIP-FT control) | before training (epoch 0) | 3 | 62.0 ± 0.0 | 65.2 ± 0.0 | 65.3 ± 0.0 | 72.5 ± 0.0 |
| none (CLIP-FT control) | after the last epoch | 3 | 63.0 ± 0.3 | 67.0 ± 0.2 | 65.5 ± 0.4 | 80.9 ± 0.1 |
| random caption (control) | before training (epoch 0) | 3 | 62.0 ± 0.0 | 65.2 ± 0.0 | 65.3 ± 0.0 | 72.5 ± 0.0 |
| random caption (control) | after the last epoch | 3 | 63.0 ± 0.3 | 67.2 ± 0.1 | 65.7 ± 0.2 | 81.0 ± 0.1 |
| swap (NegCLIP-style) | before training (epoch 0) | 3 | 62.0 ± 0.0 | 65.2 ± 0.0 | 65.3 ± 0.0 | 72.5 ± 0.0 |
| swap (NegCLIP-style) | after the last epoch | 3 | 76.8 ± 0.3 | 88.2 ± 0.0 | 89.0 ± 0.2 | 79.4 ± 0.1 |
| swap_nosym (ablation) | before training (epoch 0) | 3 | 62.0 ± 0.0 | 65.2 ± 0.0 | 65.3 ± 0.0 | 72.5 ± 0.0 |
| swap_nosym (ablation) | after the last epoch | 3 | 76.6 ± 0.2 | 88.3 ± 0.1 | 89.1 ± 0.2 | 79.5 ± 0.2 |
| swap_fluent (fluency-matched) | before training (epoch 0) | 3 | 62.0 ± 0.0 | 65.2 ± 0.0 | 65.3 ± 0.0 | 72.5 ± 0.0 |
| swap_fluent (fluency-matched) | after the last epoch | 3 | 74.2 ± 0.3 | 83.7 ± 0.2 | 83.3 ± 0.1 | 80.3 ± 0.3 |

- Validation images come from COCO train2014 and are disjoint from the training images and from every test set. Use these numbers, not the test sets, for any choice of settings.
