### Paired differences between fine-tuning conditions

| Comparison | Seeds | VG-Relation | VG-Attribution | COCO-Order | SugarCrepe |
|---|---|---|---|---|---|
| swap − none | 3 vs 3 | +20.6 [+17.8, +23.0], p<0.001 | +6.7 [+5.7, +7.8], p<0.001 | +51.5 [+50.8, +52.2], p<0.001 | +2.6 [+1.6, +3.6], p<0.001 |
| swap_fluent − none | 3 vs 3 | +16.3 [+13.6, +19.0], p<0.001 | +5.6 [+4.7, +6.5], p<0.001 | +35.7 [+35.1, +36.3], p<0.001 | +2.6 [+1.7, +3.5], p<0.001 |
| swap_fluent − swap | 3 vs 3 | -4.2 [-5.8, -2.6], p<0.001 | -1.1 [-1.6, -0.5], p<0.001 | -15.8 [-16.3, -15.3], p<0.001 | -0.0 [-0.6, +0.6], p=0.948 |
| swap_nosym − swap | 3 vs 3 | -0.0 [-0.2, +0.1], p=0.805 | +0.2 [-0.0, +0.4], p=0.052 | -0.1 [-0.2, +0.0], p=0.083 | -0.1 [-0.2, +0.1], p=0.435 |
| random − none | 3 vs 3 | +0.3 [-0.1, +0.8], p=0.183 | +0.0 [-0.2, +0.2], p=0.996 | -0.2 [-0.3, -0.1], p=0.001 | -0.1 [-0.3, +0.1], p=0.155 |
| swap_fluent − random | 3 vs 3 | +16.0 [+13.2, +18.6], p<0.001 | +5.6 [+4.7, +6.5], p<0.001 | +35.9 [+35.3, +36.6], p<0.001 | +2.7 [+1.8, +3.7], p<0.001 |

- Difference of the headline metrics in percentage points, [95% interval], two-sided bootstrap p-value. Each item's correctness is first averaged over the training seeds; then images are resampled, with the same resampled images for both conditions.
- The interval covers the uncertainty from the finite test set. Seed-to-seed variation is shown by the standard deviations in the table of conditions.
- Several comparisons are tested; treat p-values near 0.05 with caution.
