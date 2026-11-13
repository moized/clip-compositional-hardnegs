# Deviations from the original work

What this project does differently from the papers it builds on, and why. "ARO" is
Yuksekgonul et al. (ICLR 2023); the released code is
https://github.com/mertyg/vision-language-models-are-bows at commit `81fdd20`.

## A. Reproduction of the evaluation (released checkpoints)

| # | Original | Here | Reason / expected effect |
|---|---|---|---|
| A1 | ARO's evaluation scripts | Re-implemented in `src/compbind/` from the released code: same test files, same candidate order, same tie rule (first candidate wins), same metric definitions | One code path for all benchmarks and per-item outputs. Checked against the published numbers (`docs/EXPECTED_VALUES.md`) |
| A2 | The released code builds NegCLIP with `open_clip.create_model_and_transforms("ViT-B-32", pretrained=<file>)`, i.e. with the GELU activation | Evaluated twice, on the benchmarks and on the downstream tasks: with QuickGELU (the activation NegCLIP was trained with) and with GELU | The two give different numbers (also for zero-shot accuracy and retrieval); both are reported, and which one matches the published values is a finding of the reproduction |
| A3 | Library versions of 2022 | open_clip 4.2.2, PyTorch 2.x, full precision (float32) | Small numerical differences only |
| A4 | COCO-Order: shuffles drawn from the global random generators, seeded once per run (default seed 1) | Generators owned by the shuffling object, so a seed fixes the captions whatever else runs; seeds 0, 1, 2 reported; spaCy 3.8 and NLTK 3.10 | The shuffled captions are not the authors' exact ones; the paper reports ±0.001 over 3 seeds for CLIP |
| A5 | Four tasks, including Flickr30k-Order | Flickr30k-Order is not run | Flickr30k requires a separate access request; COCO-Order tests the same thing |
| A6 | No uncertainty estimates (one run, fixed seed) | 95% cluster-bootstrap intervals over images | Added information, no change to the values |
| A7 | SugarCrepe paper: 7,512 items | The repository at commit `0047054` has 7,511 (Swap-Obj 245 instead of 246) | Swap-Obj can differ from the paper by up to 0.4 points |
| A8 | SugarCrepe's text-only models: Vera and a grammar model | The grammar model (same checkpoint) re-implemented with batching; GPT-2 added; Vera not run | Vera is a multi-billion-parameter model, too large for the available GPU budget |
| A9 | Zero-shot protocol not stated in ARO | CLIP_benchmark's datasets, class names and prompt templates (as in the CVPR 2026 paper), accuracy computed by this code; CIFAR-10/100, ImageNet optional | Our zero-shot numbers are comparable with the CVPR 2026 paper, not with ARO's Table 6 |

## B. Fine-tuning (a controlled comparison in NegCLIP's recipe, not a re-run of NegCLIP)

| # | NegCLIP (paper, released command, released training file) | Here | Reason / expected effect |
|---|---|---|---|
| B1 | Hard negative **images** (one of the 3 nearest neighbours, added with its captions) and hard negative captions | Hard negative captions only | The question is about the text negatives; SugarCrepe's and TripletCLIP's NegCLIP baselines make the same simplification. Gains can be smaller than the released checkpoint's |
| B2 | 109,468 COCO pairs from the Karpathy training split, which includes val2014 images | COCO train2014 images only, minus the COCO images in ARO's Visual Genome tests; then only the images with a fluency-matched pair | Keeps every test image unseen (75% of SugarCrepe's and 43% of ARO's VG images are in NegCLIP's training file). Fewer pairs, so fewer steps at 5 epochs |
| B3 | Negative captions made with spaCy; code not released | Rules inferred from the released training file (`src/compbind/negatives.py`); about 95% of the released negatives are reproduced; original capitalisation and spacing are kept | A caption and its negative differ only by the swap, which matters for fluency scores |
| B4 | One negative chosen at random per step from the caption's list | Same for `swap` and `swap_nosym`; `swap_fluent` has exactly one negative per caption | By design of the improvement |
| B5 | Learning rate chosen from {1e-5, 5e-6, 1e-6} on COCO validation retrieval; the released checkpoint is the last of 5 epochs | Fixed at 1e-6, the value of the released command; the last of 5 epochs is used; held-out caption/negative pairs are monitored but select nothing | No tuning budget, and no choice that could favour one condition |
| B6 | One run, fixed seed | Three seeds per condition | Needed to separate effects from seed noise |
| B7 | OpenCLIP fork of 2022 | OpenCLIP 4.2.2 with the same optimiser settings (AdamW, betas 0.9/0.98, eps 1e-6, weight decay 0.2 on weight matrices, cosine schedule, 50 warm-up steps, mixed precision, temperature capped at 100) | Small numerical differences only |

## C. What is new here (not in any of the five papers)

- The text-only and shuffled-image controls applied to both the benchmarks and the training negatives.
- Fluency-matched selection of training negatives (`src/compbind/refine.py`), an adaptation of SugarCrepe's adversarial refinement from an evaluation set to a training set, with one language model (GPT-2).
- The `random` control (same number of texts per step, no hard negative) and the `swap_nosym` ablation.
- Paired, seed-averaged comparisons on the items a text-only model gets right / wrong.
