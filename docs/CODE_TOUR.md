# Code tour

A map of the code, written so that you can explain, defend and change it. Read the
files in the order below; each has a docstring at the top that says what it does
and where the idea comes from.

## 1. The pipeline in one picture

```
00_download_data          benchmarks, released NegCLIP checkpoint, COCO evaluation images
        │
01_eval_compositional ──► results/<model>/summary.json + one row per test item
02_eval_text_only     ──► results/text_only/…        (language model alone, no image)
05_eval_downstream    ──► results/<model>/downstream.json   (zero-shot, retrieval)
        │
03_build_training_data ─► data/training/pairs.csv.gz  (caption + three kinds of negatives)
        │
run_grid = for every condition and seed: 04_train → 01 → 05
        │
06_make_tables, 07_make_figures ─► results/tables, results/figures
```

Scripts are thin: they parse options, call functions from `src/compbind/` and save
files. All logic is in the modules.

## 2. Modules, in reading order

| File | What it does | Functions to know |
|---|---|---|
| `items.py` | The shape of a test case: one image, several captions, the index of the right one | `Item` |
| `scoring.py` | Encodes images and captions with CLIP, picks the most similar caption, records the result per item. Also the shuffled-image control | `Embedder`, `score_items`, `shuffled_image_index` |
| `models.py` | Loads ViT-B/32 from OpenAI weights or from a checkpoint; makes the QuickGELU / GELU choice explicit | `load_clip`, `read_state_dict` |
| `aro.py`, `perturb.py` | VG-Relation, VG-Attribution, COCO-Order as lists of `Item`s; ARO's macro-accuracy rules | `load_vg_relation`, `load_coco_order`, `relation_groups`, `attribution_groups` |
| `sugarcrepe.py` | SugarCrepe as a list of `Item`s | `load_items` |
| `benchmarks.py` | One entry point for the four benchmarks and their headline metrics | `get_items`, `summarize` |
| `stats.py` | Confidence intervals and paired differences by resampling images | `accuracy_ci`, `macro_accuracy_ci`, `paired_difference` |
| `textonly.py` | Scores a caption without the image: GPT-2 log-probability, grammar classifier | `Gpt2Scorer`, `GrammarScorer` |
| `overlap.py` | Which test images were in NegCLIP's training file | `negclip_training_set`, `aro_coco_ids` |
| `negatives.py` | NegCLIP-style swapped captions made with spaCy; flag for swaps that keep the meaning | `candidate_swaps`, `is_symmetric`, `one_per_kind` |
| `refine.py` | Fluency-matched selection of one negative per image | `build_matched_pairs` |
| `traindata.py` | Leak-free training pool; assembles the table of pairs | `leak_free_pool`, `assemble_pairs`, `leak_report` |
| `losses.py` | CLIP loss and NegCLIP loss | `clip_loss`, `negclip_loss` |
| `training.py` | The fine-tuning loop and the validation on held-out pairs | `train`, `validate`, `PairDataset` |
| `downstream.py` | Zero-shot classification and COCO retrieval | `zero_shot`, `coco_retrieval` |
| `report.py`, `figures.py` | Tables and figures from saved files | `ALL_TABLES`, `ALL_FIGURES` |

## 3. How one number is produced: "CLIP, VG-Relation, 59.9%"

1. `aro.load_vg_relation` reads `visual_genome_relation.json`. Each case becomes an
   `Item` with captions `(false_caption, true_caption)`, label 1, the relation name as
   `group`, and a crop box around the two objects.
2. `scoring.score_items` encodes every cropped image and every distinct caption once,
   L2-normalises both, and computes the cosine similarity of each image with its two
   captions. The prediction is the caption with the higher similarity; a tie goes to
   the first caption (the false one), so a tie counts as an error.
3. The same captions are also scored against **another** image (`shuffled_image_index`).
   If a model is still right, it did not need the image.
4. `benchmarks.summarize` drops the relations ARO calls symmetric, computes the accuracy
   of each remaining relation, and averages those 48 accuracies (**macro** accuracy,
   ARO's metric). It also reports the plain per-item accuracy, because SugarCrepe's
   re-evaluation of ARO used that one (51.1% for the same model).
5. `stats.macro_accuracy_ci` resamples **images** with replacement 2,000 times and
   recomputes the macro accuracy. The 2.5% and 97.5% quantiles are the interval.
   Images, not items, are resampled because the ~4 items of one image are not independent.

## 4. The two losses (`losses.py`)

A batch has N images and N captions; all vectors have length 1, so a dot product is
a cosine similarity, multiplied by CLIP's temperature (about 100) before the softmax.

- **CLIP loss.** Build the N×N table of similarities. Row *i* should select caption *i*
  (cross-entropy over the row), and column *i* should select image *i* (cross-entropy
  over the column). Average the two.
- **NegCLIP loss.** Each caption also has a hard negative (the swapped caption). Append
  the N negatives as extra columns: the table is N×2N. Row *i* must now select caption
  *i* among 2N captions, including its own swapped version. The column part uses only
  the N true captions, because a negative caption has no image of its own.

The only difference between training conditions is what is put in those N extra
columns: nothing (`none`), another image's caption (`random`), a NegCLIP-style swap
(`swap`, `swap_nosym`) or the fluency-matched swap (`swap_fluent`).

## 5. Fluency-matched negatives (`negatives.py` → `refine.py`)

1. `candidate_swaps` lists every swap NegCLIP's rules allow: two words with the same
   tag (noun, plural noun, adjective, adverb, -ing verb) or two noun phrases of at least
   three words. The rules were inferred from the authors' released training file; the
   builder script measures the agreement (about 95%).
2. `is_symmetric` flags swaps that probably keep the meaning ("a dog and a cat").
   It is a heuristic on an automatic parse, which is why there is a manual audit.
3. Fluency gap = mean GPT-2 log-probability of the caption − that of the swapped
   caption. A positive gap means the swap reads worse. For ordinary swaps the gap is
   positive about 95% of the time: a language model can spot the negative without the image.
4. `build_matched_pairs` picks one (caption, negative) pair per image so that in every
   small range of |gap| there are exactly as many pairs with a negative gap as with a
   positive gap. "Pick the more fluent caption" is then right on exactly half of the
   pairs. This is SugarCrepe's adversarial refinement, applied to training data and
   with one language model.

Limit to state openly: the balance holds for GPT-2 by construction. The grammar model,
which was not used in the selection, still does better than chance on the selected
pairs (see `results/training_data/stats.json`), so the cue is reduced, not removed.

## 6. Why the training data cannot leak (`traindata.py`, `overlap.py`)

- SugarCrepe uses COCO val2017 images, which are part of val2014. COCO-Order and the
  retrieval test use the Karpathy test split, also part of val2014.
- ARO's Visual Genome images partly come from COCO; Visual Genome's metadata gives
  the COCO id of each. The ids of ARO's 6,603 images (3,225 from COCO) are also stored in
  `src/compbind/resources/aro_vg_to_coco.csv`, so the check works when the Visual Genome
  server is down (`overlap.py`).
- So the pool is: train2014 images only, minus the COCO images used by ARO.
  `leak_report` counts the overlap with every test set and the builder stops if any is
  not zero. Training and validation images are different images.

## 7. Questions you should be able to answer

| Question | Where to look |
|---|---|
| What exactly is the NegCLIP loss, and why is there no column loss for negatives? | `losses.py`, section 4 above |
| Why two numbers for NegCLIP (QuickGELU and GELU)? | docstring of `models.py`, `docs/DEVIATIONS.md` A2 |
| Why macro accuracy, and what changes with per-item accuracy? | `aro.py` docstring, `benchmarks.summarize` |
| How is a tie between two captions handled? | `scoring.py` docstring |
| What does the shuffled-image control show that the text-only baseline does not? | The text-only baseline uses an outside language model; the shuffled-image control asks whether *the evaluated model itself* prefers the right caption without the right image |
| Why resample images and not items in the bootstrap? | `stats.py` docstring |
| How do you know no test image was used for training? | section 6 above, `results/training_data/stats.json` → `leak_checks` |
| Why is `random` a necessary control? | Conditions with negatives encode twice as many texts per step; `random` has the same amount but no hard negative |
| Why can't you claim the fluency cue is fully removed? | section 5, last paragraph |
| What did you not reproduce? | `docs/DEVIATIONS.md` (no hard negative images, no Flickr30k-Order, no Vera) |

## 8. Changes you could be asked to make

- **Add a training condition.** Add its name to `CONDITIONS` in `src/compbind/__init__.py`,
  give `PairDataset` in `training.py` a rule for its negative, and add a label in
  `report.CONDITION_LABELS` and a style in `figures.STYLE`.
- **Use another fluency model.** `Gpt2Scorer(model_name=...)` accepts any causal language
  model on the Hugging Face hub (for example `distilgpt2` or `gpt2-medium`).
- **Balance on two text-only models.** In `refine.build_matched_pairs`, bin on the sign
  of a second gap as well, the way SugarCrepe balances on Vera and the grammar model.
- **Add a benchmark.** Write a loader that returns `Item`s, register it in
  `benchmarks.get_items` and give `benchmarks.summarize` its headline metric.
- **Change a hyperparameter.** Edit `configs/main.yaml`; record the change and the
  validation evidence in `HYPOTHESES.md`.

## 9. Tests

`pytest tests` runs in a few seconds and needs no GPU or download (the swap tests need
the spaCy model). Each test file checks one module against values that can be computed
by hand, for example `test_losses.py` compares the NegCLIP loss with a formula for a
2-image batch. `python scripts/quick_test.py` runs the real model on 100 test items and
two training steps.
