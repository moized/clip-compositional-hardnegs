# Compositional binding vs. text-only shortcuts in CLIP

Research project for **YZM5110 Multimodal Learning** (Yıldız Technical University, Fall 2026).

**Status:** in progress. No results yet.

## Research question

Do NegCLIP's compositional gains on ARO come from better visual binding, or from learning to reject implausible sentences? Do fluency-matched hard negatives help when compute is held equal?

## Reproduction target

Yuksekgonul, Bianchi, Kalluri, Jurafsky & Zou, *When and why vision-language models behave like bags-of-words, and what to do about it?*, ICLR 2023.
- Paper: https://arxiv.org/abs/2210.01936
- Original code: https://github.com/mertyg/vision-language-models-are-bows

## Planned experiments

1. **Reproduction.** Compare OpenAI CLIP ViT-B/32 with the released NegCLIP checkpoint on ARO (VG-Relation, VG-Attribution, COCO-Order) and SugarCrepe. Published and measured results go in separate columns.
2. **Controls.**
   - A text-only baseline (a language model picks the caption without seeing the image).
   - A shuffled-image control.
   - Accuracy reported separately for items the text-only baseline gets right and items it gets wrong.
3. **Fine-tuning under matched compute.**
   - (a) No hard negatives (CLIP-FT).
   - (b) NegCLIP-style swapped negatives.
   - (c) Fluency-matched swapped negatives.
4. **Cost.** Zero-shot ImageNet and CIFAR accuracy (CLIP_benchmark), training time and peak GPU memory.

## Repository layout

| Folder | Contents |
|---|---|
| `configs/` | Experiment configurations |
| `src/` | Evaluation and training code |
| `scripts/` | Exact commands that reproduce each table and figure |
| `results/` | Saved outputs (small files only) |
| `report/` | Report sources |

## Reproducing the results

To be added: environment and pinned dependency versions, data-access instructions, and exact commands. No datasets or large checkpoints are stored in this repository. See each dataset's own licence.

## AI assistance

To be completed, as the course brief requires: the tools used and the specific contribution of each.
