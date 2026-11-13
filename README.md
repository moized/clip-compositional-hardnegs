# Compositional binding vs. text-only shortcuts in CLIP

Research project for **YZM5110 Multimodal Learning** (Yıldız Technical University, Fall 2026).
Author: Mohammed Izedin Mohammed.

**Topic.** Reproducing NegCLIP on ARO and SugarCrepe, and testing whether
fluency-matched hard negatives improve binding over a fine-tuning-only control under
matched compute, including the zero-shot cost.

**Reproduction target.** Yuksekgonul, Bianchi, Kalluri, Jurafsky & Zou, *When and why
vision-language models behave like bags-of-words, and what to do about it?*, ICLR 2023
([paper](https://arxiv.org/abs/2210.01936),
[code](https://github.com/mertyg/vision-language-models-are-bows)).

**Status.** Code complete and tested. The experiments are run with the Kaggle notebook;
results are added to `results/` as they are produced (see `results/tables/README.md`).

## Research question

NegCLIP fine-tunes CLIP with "hard negative" captions made by swapping words
("the horse is eating the grass" → "the grass is eating the horse") and reports large
gains on the ARO benchmark. Many of those negatives are sentences that a language
model can recognise as unlikely without seeing the image. So:

1. Do NegCLIP's gains come from better visual binding, or from learning to reject
   unlikely sentences?
2. If the text-only cue is removed from the training negatives (fluency-matched
   negatives), does fine-tuning still help, compared with fine-tuning without hard
   negatives, when data, steps and optimiser are identical?
3. What does it cost in zero-shot accuracy?

The hypotheses, fixed settings and decision rules are in [HYPOTHESES.md](HYPOTHESES.md),
committed before the fine-tuning runs.

## Experiments

| Part | What is done | Script |
|---|---|---|
| Reproduction | OpenAI CLIP ViT-B/32 and the released NegCLIP checkpoint on VG-Relation, VG-Attribution, COCO-Order (ARO) and SugarCrepe; published and measured values in separate columns | `01_eval_compositional.py` |
| Controls | Text-only baselines (GPT-2, grammar model), shuffled-image control, accuracy split by "text alone suffices / misleads", train/test image overlap of the released checkpoint | `02_eval_text_only.py` |
| Training data | Leak-free COCO pairs with three kinds of negatives: NegCLIP-style swaps, swaps without meaning-preserving ones, fluency-matched swaps | `03_build_training_data.py` |
| Fine-tuning | Five conditions × three seeds, same images, captions, steps and optimiser: `none` (CLIP-FT control), `random` (control for extra texts), `swap` (NegCLIP-style), `swap_nosym` (ablation), `swap_fluent` (the improvement) | `run_grid.py` → `04_train.py` |
| Cost | Zero-shot CIFAR-10/100 (ImageNet optional), COCO retrieval, training time, peak GPU memory | `05_eval_downstream.py` |
| Report assets | Every table and figure, regenerated from saved files | `06_make_tables.py`, `07_make_figures.py` |

## Repository layout

```
configs/      published.yaml (published numbers with sources), main.yaml (experiment settings), quick.yaml
src/compbind/ the library: benchmarks, scoring, negatives, fluency matching, losses, training, statistics, tables, figures
scripts/      numbered command-line steps, run_grid.py, quick_test.py
notebooks/    kaggle_run.ipynb: the whole pipeline on Kaggle
tests/        unit tests (pytest)
docs/         CODE_TOUR.md, DEVIATIONS.md, EXPECTED_VALUES.md, AUDIT.md
results/      saved outputs: summaries, per-item results, tables, figures (small files only)
report/       report sources
```

No dataset, image or model checkpoint is stored in this repository.

## Setup

Python 3.10 or newer. On a machine with a GPU, install the PyTorch build for your CUDA
version first, then:

```bash
git clone https://github.com/moized/clip-compositional-hardnegs.git
```

```bash
cd clip-compositional-hardnegs
```

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions the code was developed with. On Kaggle the
notebook installs `requirements-kaggle.txt` (three packages on top of Kaggle's image)
and saves `pip freeze` to `results/environment.txt`.

## Quick test

```bash
pytest tests
```

```bash
python scripts/quick_test.py
```

The unit tests need no GPU and no download. The quick test downloads CLIP's weights
(about 350 MB) and 100 COCO images, scores 100 SugarCrepe items and runs two training
steps; it takes about two minutes on a laptop CPU. With `--full` it scores the whole
Swap-Att subset and checks the result against the published 63.96%.

## Data access

Everything is downloaded by `scripts/00_download_data.py` into `data/` (ignored by git);
file sizes and SHA-256 hashes are written to `data/data_manifest.json`.

| Data | Source | Size |
|---|---|---|
| SugarCrepe annotations | github.com/RAIVNLab/sugar-crepe, commit `0047054` | 2 MB |
| ARO: VG-Relation, VG-Attribution annotations and images | Google Drive links of the ARO repository, commit `81fdd20` | 0.9 GB |
| NegCLIP released checkpoint and training file | ARO repository | 1.7 GB |
| COCO Karpathy split files | BLIP project storage | 84 MB |
| Visual Genome image metadata | visualgenome.org | 17 MB |
| COCO images: 6,377 for evaluation, then the training subset | images.cocodataset.org, one file at a time; or any local COCO copy via `--coco-roots` | about 1 GB + 8 GB |
| CIFAR-10 / CIFAR-100 | torchvision | 365 MB |
| ImageNet validation set (optional) | not downloaded; pass `--imagenet-root` | |

If Google Drive refuses a download (it limits popular files), the script prints the
link and the place to put the file by hand. Licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Reproducing the results

The easiest way is `notebooks/kaggle_run.ipynb` (GPU T4, Internet on). The same steps
as commands, run from the repository root:

```bash
python scripts/00_download_data.py
```

```bash
python scripts/01_eval_compositional.py --model openai --name clip
```

```bash
python scripts/01_eval_compositional.py --model negclip --name negclip
```

```bash
python scripts/01_eval_compositional.py --model negclip --name negclip_gelu --gelu
```

```bash
python scripts/05_eval_downstream.py --model openai --name clip
```

```bash
python scripts/05_eval_downstream.py --model negclip --name negclip
```

```bash
python scripts/02_eval_text_only.py
```

```bash
python scripts/03_build_training_data.py --config configs/main.yaml --check-against-negclip
```

```bash
python scripts/run_grid.py --config configs/main.yaml
```

```bash
python scripts/06_make_tables.py
```

```bash
python scripts/07_make_figures.py
```

Approximate time on one Kaggle T4: downloads 20 min, reproduction and controls 40 min,
training data 20 min, grid (15 runs) 4 to 5 hours. `run_grid.py` skips finished steps,
so it can be stopped and started again. A rehearsal of everything on a small sample:
`--config configs/quick.yaml` with `--results-dir results_quick`.

After the manual audit of the negatives ([docs/AUDIT.md](docs/AUDIT.md)):

```bash
python scripts/08_audit_summary.py
```

Random examples on which two models disagree, for the failure analysis (favourable and
unfavourable cases are drawn the same way):

```bash
python scripts/09_examples.py --benchmark sugarcrepe --a swap_fluent_seed0 --b none_seed0
```

**Which results need a GPU.** The unit tests, the quick test, the tables and the figures
run on a laptop. The reproduction and the controls run on a CPU in a few hours, or on a
GPU in under an hour. The fine-tuning grid needs a GPU with at least 12 GB of memory
(estimated at 15 to 25 minutes per run on a T4, evaluation included).

## What is recorded for reproducibility

| Item | Where |
|---|---|
| Code revision, command line, library versions, GPU | `run` block of every `summary.json`, `downstream.json`, `train_summary.json`, `stats.json` |
| Upstream revisions | `src/compbind/paths.py` (ARO `81fdd20`, SugarCrepe `0047054`) |
| Checkpoints | OpenAI CLIP ViT-B/32 through OpenCLIP; released NegCLIP file, SHA-256 in `data_manifest.json` |
| Data versions | `data_manifest.json`; test-item fingerprints (`items_sha1`) in every summary |
| Seeds | training seeds 0, 1, 2; COCO-Order shuffle seeds 0, 1, 2; bootstrap seed 0 |
| Configuration | `configs/main.yaml`; every option is also in the `run` block |
| Training split | `results/training_data/split_manifest.json` (COCO image ids) |
| Published numbers | `configs/published.yaml`, each with its source table |
| Deviations from the original | [docs/DEVIATIONS.md](docs/DEVIATIONS.md) |

Published and measured numbers never share a column: see `results/tables/reproduction_*.md`.
Values from a development run on CPU, for checking a new run, are in
[docs/EXPECTED_VALUES.md](docs/EXPECTED_VALUES.md).

## Understanding the code

[docs/CODE_TOUR.md](docs/CODE_TOUR.md) explains each module, how one number is produced
from the raw files, the two losses, the fluency matching and the statistics.

## AI assistance

> Draft. Edit this section until it describes what was actually done, as the course
> brief requires (tools used and the specific contribution of each).

- **Claude Code (Anthropic's coding assistant), November 2026.**
  - Drafted the literature comparison matrix from full readings of the five papers.
    I checked the entries against the PDFs.
  - Wrote the code in this repository under my direction: the library in
    `src/compbind/`, the scripts, the tests, the Kaggle notebook and the documentation.
  - Ran a development check on CPU that compared the evaluation code with published
    numbers (`docs/EXPECTED_VALUES.md`). The results in `results/` and in the report
    come from my own runs on Kaggle.
  - Drafted `HYPOTHESES.md`; I edited and finalised it before the fine-tuning runs.
- **Gemini and ChatGPT** were asked for alternative topic and paper suggestions at the
  planning stage. _(State here what, if anything, was kept from them.)_
- I ran the experiments, did the manual audit of the negatives, wrote the report and
  can explain and modify every part of the code.

## References

- Yuksekgonul et al., *When and why vision-language models behave like bags-of-words, and what to do about it?*, ICLR 2023. arXiv:2210.01936
- Hsieh et al., *SugarCrepe: Fixing hackable benchmarks for vision-language compositionality*, NeurIPS 2023. arXiv:2306.14610
- Patel et al., *TripletCLIP: Improving compositional reasoning of CLIP via synthetic vision-language negatives*, NeurIPS 2024. arXiv:2411.02545
- Pham et al., *No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models*, CVPR 2026. arXiv:2603.25722
- Sbrolli, Yamasaki & Matteucci, *Beyond Bag-of-Words: Diagnosing Compositional Binding Failures in Vision-Language Models*, NeurIPS 2026. arXiv:2602.02043
- Radford et al., *Learning transferable visual models from natural language supervision*, ICML 2021. arXiv:2103.00020
