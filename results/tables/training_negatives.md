### Training negatives: how much can a text-only model tell?

| Negative set | Pairs | Negatives per caption | Text-only accuracy, GPT-2 (%) | Text-only accuracy, grammar model (%) | Median fluency gap | Flagged meaning-preserving (%) |
|---|---|---|---|---|---|---|
| all candidate swaps (before selection) | 1820904 |  | 95.8 |  | 0.85 | 13.4 |
| swap (NegCLIP-style, one per swap type) | 89661 | 1.76 | 82.9 | 82.1 | 0.41 | 13.1 |
| swap_nosym | 84068 | 1.65 | 83.0 | 82.7 | 0.43 | 0.0 |
| swap_fluent (fluency-matched) | 51078 | 1.00 | 50.0 | 67.6 | -0.00 | 0.0 |

- Training pairs: 51078; validation pairs: 2054; every condition trains on the same images and captions.
- Fluency gap = mean GPT-2 log-probability of the caption minus that of the negative (positive: the negative reads worse). Text-only accuracy = how often the language model alone prefers the true caption; 50% means no text-only signal for that model.
- The fluency-matched set is balanced with GPT-2, so its GPT-2 accuracy is 50% by construction; the grammar model was not used for the selection and is an independent check.
- Leak check (training images that also appear in a test set): sugarcrepe images: 0, karpathy test images: 0, karpathy val images: 0, aro vg images: 0.
- Check against the authors' released NegCLIP training file (20000 captions): our swap rules reproduce 95.3% of their negatives; GPT-2 alone separates their negatives from the true caption 94.8% of the time.
