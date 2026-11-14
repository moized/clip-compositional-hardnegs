# Hypotheses and analysis plan

**STATUS: FINAL**

Date finalised: 2026-11-14, before the full fine-tuning run.
Author: Mohammed Izedin Mohammed.

How this file was written: the predictions are the author's answers to a questionnaire
on 2026-11-14. For three of them (the primary test of H3, where the gap in H3c should
appear, and which conditions lose zero-shot accuracy in H5) the author chose between
options prepared by the AI assistant (Claude). The assistant wrote the wording and the
test criteria, and checked that the predictions are consistent with each other; the
author accepted the result.

Any later change is recorded in the table at the end, with its date and reason.

## Research question

Do NegCLIP's gains on compositional benchmarks come from better visual binding, or from
learning to reject captions that a language model can already recognise as unlikely? If
the text-only cue is removed from the training negatives (fluency-matched hard negatives),
does fine-tuning still improve binding compared with a fine-tuning-only control under the
same data, steps and optimiser? What does it cost in zero-shot accuracy?

## What had been seen when this was written

- **The author** had seen only the AI assistant's written summaries of a development check
  of the two released checkpoints (OpenAI CLIP and NegCLIP): accuracies on ARO and
  SugarCrepe, the text-only baselines, the statistics of the training negatives and the
  zero-shot CIFAR accuracies (`docs/EXPECTED_VALUES.md`).
- **Two rehearsal runs** on Kaggle checked that the pipeline works (12 training steps,
  500 test items per benchmark). The assistant checked their output files for errors.
  After 12 steps the models have barely moved from CLIP, so these numbers say nothing
  about H3 to H5, and the author did not use them.
- **Not seen:** any result of the full fine-tuning runs.

So H1 and H2 are **checks whose outcome is already known**; they are listed to state the
criteria and the values. H3 to H5 are **predictions**.

## Checks (outcome known from the development check)

**H1 (reproduction).** With the released checkpoint, NegCLIP beats CLIP on VG-Relation,
VG-Attribution and COCO-Order, with values close to ARO, Table 6 (CLIP 59 / 62 / 46,
NegCLIP 81 / 71 / 86).
*Criterion:* each measured value within 2 points of the published one, for at least one
of the two ways of building NegCLIP (QuickGELU, as trained, or GELU, as the released
evaluation code builds it), and NegCLIP > CLIP on all three.
*Known result:* holds with GELU (80.2 / 70.5 / 86.8; CLIP 59.9 / 63.2 / 47.2). With
QuickGELU, COCO-Order is 3.5 points below the published value (82.5).

**H2 (text-only shortcut in the benchmarks).** A language model that never sees the image
scores far above chance on the three ARO tasks and at chance on SugarCrepe. NegCLIP's gain
over CLIP is larger on items the text-only model (GPT-2) gets right than on items it gets
wrong.
*Known result:* GPT-2 alone scores 82.8 / 72.4 / 96.6 on VG-Relation / VG-Attribution /
COCO-Order and 49.8 on SugarCrepe. NegCLIP's gain over CLIP, on items GPT-2 gets right
vs. wrong: VG-Relation +10.5 vs +3.6, VG-Attribution +12.8 vs +4.9, COCO-Order +35.6 vs
+25.2 points. On SugarCrepe the gains are equal (+9.3 vs +9.0), as expected where GPT-2 is
at chance and the split carries no information.

**The author's reading of H2, which motivates H3 to H5:** both mechanisms contribute.
NegCLIP gains even where the text misleads (some visual binding), but gains much more
where the text alone gives the answer away (a text-only cue).

## Predictions (no result of the full fine-tuning runs has been seen)

Conditions, all fine-tuned from OpenAI CLIP on the same images and captions, with the same
steps, batch size and optimiser; only the negative caption differs:
`none` (no extra negatives, the CLIP-FT control), `random` (a caption of another image),
`swap` (NegCLIP-style swaps), `swap_nosym` (swaps without meaning-preserving ones),
`swap_fluent` (fluency-matched swaps). Three seeds each (0, 1, 2).

**Decision rules used throughout.** A comparison "A − B" is the difference of the headline
metric (VG tasks: macro accuracy; COCO-Order: accuracy, shuffle seed 0; SugarCrepe: mean of
the 7 subsets), with each item's correctness first averaged over the seeds and a 95%
cluster-bootstrap interval over images (`results/tables/conditions_differences.md`).
- **"A > B"** means the interval excludes zero on the positive side *and* A's headline
  value is higher than B's in each seed pair (seed 0 vs seed 0, and so on; runs with the
  same seed see the training images in the same order).
- **"no difference"** means the interval includes zero. This is not proof of equality, so
  the interval itself is always reported.

### H3: fluency-matched negatives improve binding (primary hypothesis)

**H3 (primary test).** On SugarCrepe, `swap_fluent` > `none`.
SugarCrepe was chosen because text-only models score at chance on it, so a gain there
cannot come from the fluency cue.
*Supported* if the rule "A > B" holds. *Not supported* if the interval includes zero.
*Contradicted* if the interval excludes zero on the negative side.

**H3b (secondary).** `swap_fluent` improves less than NegCLIP-style negatives on the ARO
tasks: on each of VG-Relation, VG-Attribution and COCO-Order, `none` < `swap_fluent` <
`swap`. A consequence of H3c below; reported per task.

**H3c (where the gap appears; the author's choice: mainly where the text helps).**
If `swap`'s extra gain comes from the text-only cue:
1. on ARO items GPT-2 gets right, `swap` > `swap_fluent`;
2. on ARO items GPT-2 gets wrong, `swap_fluent` − `swap` is zero or positive (its interval
   includes zero or lies above it);
3. on SugarCrepe, `swap` and `swap_fluent` do not differ.

*Evidence against H3c:* `swap` > `swap_fluent` by a similar amount on both kinds of items
and on SugarCrepe. That would point to `swap` simply being a stronger training signal, for
example because it offers about 1.8 different negatives per caption against 1 for
`swap_fluent`.
COCO-Order has only 846 items that GPT-2 gets wrong, so its part of point 2 has little
power; VG-Relation and VG-Attribution carry most of the weight.

**Consequence used as an extra check (secondary).** If `swap` learns more of the text-only
cue, it should also prefer the right caption more often when shown a *wrong* image. The
shuffled-image accuracy on the ARO tasks should therefore be ordered `none` < `swap_fluent`
< `swap`. These values are saved in every run's `summary.json`.

### H4: controls

**H4a.** `random` does not improve over `none`: on each benchmark, `random` − `none` shows
no difference or a negative one. *Contradicted* if `random` > `none` on any benchmark.
Reason: a caption of another image is the kind of negative every batch already contains,
so it adds text-encoder work but no compositional signal. This control shows whether a
gain of the swap conditions is due to the hard negatives or only to encoding twice as
many captions per step.

**H4b.** Removing swaps flagged as meaning-preserving does not hurt: on each benchmark,
`swap_nosym` − `swap` shows no difference or a positive one. *Contradicted* if `swap` >
`swap_nosym` on any benchmark. Reason: about 13% of NegCLIP-style negatives are flagged as
keeping the caption's meaning ("a dog and a cat" / "a cat and a dog"), so training to
reject them teaches something false.

### H5: zero-shot cost

Measured on CIFAR-100 (10,000 test images; zero-shot top-1 accuracy). The reference is
pretrained CLIP in the same run (64.2% in the development check).

**H5.** The conditions with hard negatives lose accuracy; plain fine-tuning does not:
- `swap`, `swap_nosym` and `swap_fluent`: seed-mean accuracy more than 1 point below CLIP;
- `none` and `random`: seed-mean accuracy within 1 point of CLIP.

*Contradicted* if `none` or `random` also drops by more than 1 point (then the cost comes
from fine-tuning on COCO, not from the hard negatives), or if a hard-negative condition
stays within 1 point.
Known context, not part of the test: the released NegCLIP scores 62.9% (QuickGELU) or
60.9% (GELU) against CLIP's 64.2%. It was trained differently (more pairs, hard negative
images), so it does not decide H5.

## Fixed settings (not tuned)

- **Model and optimiser** (from the released NegCLIP command and `configs/main.yaml`):
  ViT-B/32 from OpenAI weights; 5 epochs; batch 256; learning rate 1e-6; 50 warm-up
  steps, then cosine decay; AdamW with weight decay 0.2; mixed precision; one GPU per run;
  gradient checkpointing (memory only, same gradients).
- **Training data:** COCO train2014 images used by no test set; reduced image copies with
  a shorter side of 256 pixels; one fluency-matched pair per image (bin width 0.02,
  |gap| ≤ 1.0); 2,000 validation pairs.
- **Evaluation of fine-tuned models:** all four benchmarks (COCO-Order with shuffle
  seed 0), zero-shot CIFAR-10 and CIFAR-100, COCO retrieval.
- **Time budget:** no run is started that would end more than 7.5 hours after the grid
  started.

## What may be chosen on validation data only

Nothing is tuned on a test set. If a setting has to change (for example the number of
epochs), the choice is made from the validation pairs in `results/<run>/train_summary.json`
and recorded below with the date and the reason.

## Analysis plan

1. Tables and figures from `scripts/06_make_tables.py` and `scripts/07_make_figures.py`,
   unchanged.
2. Mean ± standard deviation over the three seeds for every condition.
3. Paired differences with 95% cluster-bootstrap intervals, and the per-seed signs that the
   decision rule requires.
4. **One primary test** (H3 on SugarCrepe). All other comparisons are secondary: they are
   reported with their intervals and read as consistent or not with the predictions, not as
   separate confirmations.
5. **Missing runs:** if a run did not finish (time budget or error), comparisons between two
   conditions use only the seeds both have completed, and the number of seeds is reported.
6. **Manual audit** of 100 pairs per negative set (`docs/AUDIT.md`). If `swap_fluent` has a
   clearly higher share of meaning-preserving "negatives" than `swap`, part of any
   difference between them may come from false negatives; this is reported as a limitation.
7. **Known confounds of `swap` vs `swap_fluent`,** reported with the results: `swap_fluent`
   has one negative per caption against about 1.8 for `swap`, and a different mix of swap
   types (more adjective and noun-phrase swaps; `results/training_data/stats.json`). The
   grammar model, which was not used for the selection, still prefers the true caption for
   about 70% of the fluency-matched pairs, so the text-only cue is reduced, not removed.
8. **Failure analysis** with random examples on which two models disagree
   (`results/examples/`, drawn without choosing favourable cases).
9. All results are reported, including those that contradict the predictions.

## Changes after finalising

| Date | Change | Reason (validation evidence) |
|---|---|---|
| | | |
