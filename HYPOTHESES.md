# Hypotheses and analysis plan

**STATUS: DRAFT**

> This file must be finished and committed **before** the fine-tuning grid is run.
> Edit every part until it says what *you* expect and why, then change the status
> line above to `STATUS: FINAL`, fill in the date, commit and push. The commit time
> is the evidence that the hypotheses came before the results. The Kaggle notebook
> refuses to start the grid while the status is DRAFT.
>
> The text below is a draft written with AI assistance from the registered research
> topic. Change anything you do not agree with.

Date finalised: _(fill in)_
Author: Mohammed Izedin Mohammed

## Research question

Do NegCLIP's gains on compositional benchmarks come from better visual binding, or
from learning to reject captions that a language model can already recognise as
unlikely? And if the text-only cue is removed from the training negatives
(fluency-matched hard negatives), does fine-tuning still improve binding compared
with a fine-tuning-only control, under the same data, steps and optimiser? What
does it cost in zero-shot accuracy?

## What had been seen when this was written

Be exact here; it decides which statements below are predictions.

- Seen: results of the two *released* checkpoints (OpenAI CLIP, NegCLIP) on ARO and
  SugarCrepe, the text-only baselines on those benchmarks, and the statistics of
  the training negatives. A development run of these steps was made on CPU while
  the code was written (`docs/EXPECTED_VALUES.md`).
- Seen: a smoke test of the training loop (52 small steps on CPU, validation pairs
  only, no test set).
- Not seen: any test-set result of a model fine-tuned in this project.

So H1 and H2 are **checks whose outcome was already known** from the development
run; they are listed to state the criteria. H3, H4 and H5 are **predictions**.

## Checks (outcome known from the development run)

**H1 (reproduction).** With the released checkpoint, NegCLIP beats CLIP on
VG-Relation, VG-Attribution and COCO-Order by margins close to those in ARO,
Table 6 (0.59 → 0.81, 0.62 → 0.71, 0.46 → 0.86).
*Counts as reproduced if* each measured value is within 2 points of the published
one for at least one of the two ways of building the model (QuickGELU or GELU),
and the ranking NegCLIP > CLIP holds on all three.

**H2 (text-only shortcut in the benchmarks).** A language model that never sees the
image scores far above chance on the three ARO tasks and at chance on SugarCrepe.
NegCLIP's gain over CLIP is larger on items the text-only model gets right than on
items it gets wrong.
*Evidence against:* text-only accuracy within 5 points of chance on ARO, or equal
gains on both kinds of items.

## Predictions (no fine-tuned model has been evaluated yet)

**H3 (the improvement).** Fine-tuning with fluency-matched negatives (`swap_fluent`)
gives higher accuracy than the fine-tuning-only control (`none`) on
(a) SugarCrepe (mean of 7 subsets) and (b) the ARO items the text-only model gets
wrong. On items the text-only model gets right, NegCLIP-style swaps (`swap`) gain
more than `swap_fluent`.
*Primary comparison:* `swap_fluent − none` on SugarCrepe, paired bootstrap over
images, seed-averaged. *Supported if* the 95% interval excludes zero and the sign
is the same for every seed. *Evidence against:* an interval that includes zero or
a negative difference.

**H4 (controls).** `random` (an extra unrelated caption as negative) does not
improve over `none`, so any gain of the swap conditions is due to the hard
negatives and not to having twice as many texts per step. Removing swaps flagged
as meaning-preserving (`swap_nosym`) does not hurt compared with `swap`.

**H5 (cost).** All fine-tuned models lose zero-shot accuracy relative to pretrained
CLIP. The loss of `swap_fluent` is not larger than the loss of `swap` by more than
1 point (CIFAR-100, and ImageNet if run).

## Fixed settings (not tuned)

Taken from the released NegCLIP command and `configs/main.yaml`: ViT-B/32 from
OpenAI weights, 5 epochs, batch 256, learning rate 1e-6, 50 warm-up steps, cosine
decay, AdamW (weight decay 0.2), mixed precision, one GPU, seeds 0, 1, 2. The
fluency-matching settings (bin width 0.02, |gap| ≤ 1.0) were fixed before training.

## What may be chosen on validation data only

Nothing is tuned on a test set. If a setting has to change (for example the number
of epochs), the choice is made from the validation pairs in
`results/<run>/train_summary.json` and recorded here with the date and the reason.

| Date | Change | Reason (validation evidence) |
|---|---|---|
| | | |

## Planned analysis

1. Tables and figures from `scripts/06_make_tables.py` and `scripts/07_make_figures.py`, unchanged.
2. Mean ± standard deviation over the three seeds for every condition.
3. Paired differences with 95% cluster-bootstrap intervals (images resampled).
4. Manual audit of 100 pairs per negative set (`docs/AUDIT.md`) to estimate how many
   "negatives" keep the caption's meaning.
5. All results are reported, including those that contradict the hypotheses.
