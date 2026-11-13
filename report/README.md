# Report

File to submit: `Mohammed_YZM5110_Research.pdf`, with the repository URL
(https://github.com/moized/clip-compositional-hardnegs) and the commit or tag of the
submitted version on the first page.

The brief asks for eight sections. The table lists, for each one, the material in this
repository that supports it. The text of the report is yours to write.

| # | Section (from the brief) | Material |
|---|---|---|
| 1 | Research question, scope, motivation, and contributions | `README.md` (research question), `HYPOTHESES.md` |
| 2 | Literature search approach and critical synthesis | your comparison matrix (one row per paper, seven fields); mark every statement as the authors' finding, your interpretation or your measurement |
| 3 | Technical explanation of the selected concept or method | `docs/CODE_TOUR.md` sections 3 to 5 (scoring, CLIP and NegCLIP losses, swapped negatives); `src/compbind/losses.py` |
| 4 | Reproduction target, implementation, and deviations from the source | `results/tables/reproduction_aro.md`, `reproduction_sugarcrepe.md`, `reproduction_text_only.md`; `results/figures/fig1_reproduction`; `docs/DEVIATIONS.md` part A; what you expected and what would have made you doubt the claim: `HYPOTHESES.md` H1 |
| 5 | Proposed improvement and its rationale | `HYPOTHESES.md` H3 to H5; `results/tables/training_negatives.md`; `results/figures/fig3_fluency_gap`; `docs/CODE_TOUR.md` section 5 |
| 6 | Experimental design, datasets, splits, baselines, metrics, and resources | `configs/main.yaml`; `docs/DEVIATIONS.md` part B; `results/training_data/stats.json` (sizes, leak checks); `docs/CODE_TOUR.md` section 6; hardware and library versions from the `run` block of any result file |
| 7 | Results, controls, uncertainty, and failure analysis | `results/tables/controls.md`, `slices_text_only.md`, `slices_seen_images.md`, `conditions_compositional.md`, `conditions_differences.md`, `conditions_slices.md`, `cost.md`, `validation.md`, `audit.md`; figures 2, 4, 5, 6; examples from `scripts/09_examples.py` |
| 8 | Limitations, conclusions, and unresolved questions | `docs/DEVIATIONS.md`; the audit result; the grammar-model accuracy on the fluency-matched set (the cue is reduced, not removed) |

Rules from the brief to keep in mind while writing:

- Published scores and your measured scores are shown separately (the reproduction tables already do this).
- Describe every difference from the original setup and say which comparisons still hold.
- Report unfavourable results and examples as well as favourable ones; a negative result is acceptable if the experiment is sound.
- Nothing is tuned on a test set.
- Disclose AI use: which tools, and what each contributed (`README.md`, section "AI assistance").
