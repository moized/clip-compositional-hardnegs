### Audit of the training negatives

| Negative set | Labelled pairs | D: different scene (valid) (%) | S: same meaning (false negative) (%) | N: not a meaningful sentence (%) |
|---|---|---|---|---|
| swap (NegCLIP-style) | 100 | 54 [44, 63] | 17 [11, 26] | 29 [21, 39] |
| swap_nosym | 100 | 47 [38, 57] | 10 [6, 17] | 43 [34, 53] |
| swap_fluent | 100 | 61 [51, 70] | 19 [13, 28] | 20 [13, 29] |

- 300 of 300 sampled pairs labelled by the AI assistant (Claude), after the model results were known, blind to the negative set.
- Share of each label in %, with a 95% Wilson interval.
