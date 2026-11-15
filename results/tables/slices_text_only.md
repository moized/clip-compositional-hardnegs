### Where does NegCLIP gain? Items split by a text-only model

| Benchmark | Slice | Items | Share (%) | CLIP | NegCLIP | NegCLIP − CLIP |
|---|---|---|---|---|---|---|
| VG-Relation | GPT-2 right (text alone suffices) | 14073 | 58.8 | 51.0 [50.1, 51.9] | 61.5 [60.5, 62.4] | +10.5 [+9.3, +11.5] |
| VG-Relation | GPT-2 wrong (text alone misleads) | 9864 | 41.2 | 51.3 [50.3, 52.4] | 54.9 [53.8, 56.0] | +3.6 [+2.4, +4.8] |
| VG-Relation | grammar model right (text alone suffices) | 14255 | 59.6 | 51.7 [50.8, 52.7] | 67.7 [66.8, 68.6] | +16.0 [+14.9, +17.0] |
| VG-Relation | grammar model wrong (text alone misleads) | 9682 | 40.4 | 50.2 [49.1, 51.3] | 45.6 [44.4, 46.7] | -4.7 [-6.0, -3.5] |
| VG-Attribution | GPT-2 right (text alone suffices) | 21477 | 74.7 | 62.3 [61.3, 63.3] | 75.0 [74.1, 75.9] | +12.8 [+11.9, +13.6] |
| VG-Attribution | GPT-2 wrong (text alone misleads) | 7271 | 25.3 | 58.5 [57.1, 60.1] | 63.5 [62.1, 64.8] | +4.9 [+3.5, +6.3] |
| VG-Attribution | grammar model right (text alone suffices) | 16782 | 58.4 | 61.4 [60.3, 62.6] | 73.1 [72.1, 74.2] | +11.7 [+10.7, +12.7] |
| VG-Attribution | grammar model wrong (text alone misleads) | 11966 | 41.6 | 61.2 [60.0, 62.3] | 70.7 [69.6, 71.8] | +9.5 [+8.5, +10.7] |
| COCO-Order | GPT-2 right (text alone suffices) | 24165 | 96.6 | 47.5 [46.7, 48.3] | 83.1 [82.6, 83.6] | +35.6 [+34.8, +36.4] |
| COCO-Order | GPT-2 wrong (text alone misleads) | 845 | 3.4 | 40.5 [37.2, 43.6] | 64.4 [61.1, 67.7] | +23.9 [+20.2, +27.7] |
| COCO-Order | grammar model right (text alone suffices) | 23082 | 92.3 | 48.0 [47.2, 48.7] | 83.7 [83.2, 84.2] | +35.8 [+34.9, +36.6] |
| COCO-Order | grammar model wrong (text alone misleads) | 1928 | 7.7 | 38.8 [36.6, 41.0] | 67.8 [65.5, 69.9] | +28.9 [+26.3, +31.7] |
| SugarCrepe | GPT-2 right (text alone suffices) | 3275 | 43.6 | 76.0 [74.5, 77.5] | 85.3 [84.0, 86.6] | +9.3 [+7.9, +10.7] |
| SugarCrepe | GPT-2 wrong (text alone misleads) | 4236 | 56.4 | 77.0 [75.5, 78.3] | 85.9 [84.7, 87.1] | +9.0 [+7.7, +10.2] |
| SugarCrepe | grammar model right (text alone suffices) | 3756 | 50.0 | 77.4 [75.9, 78.7] | 86.1 [84.9, 87.2] | +8.7 [+7.4, +9.9] |
| SugarCrepe | grammar model wrong (text alone misleads) | 3755 | 50.0 | 75.7 [74.3, 77.1] | 85.2 [84.1, 86.4] | +9.5 [+8.2, +10.8] |

- Per-item accuracy (%) with a 95% cluster-bootstrap interval; NegCLIP is the released checkpoint with QuickGELU. COCO-Order uses the first shuffle seed.
- If NegCLIP's gain comes from reading the image, it should appear in both slices. If it comes from preferring fluent captions, it should be concentrated where the text-only model is right and turn negative where the text-only model is wrong.
