# Manual audit of the training negatives

A swapped caption is only a useful negative if it describes a **different** scene.
Some swaps keep the meaning ("a dog and a cat" / "a cat and a dog"); training the
model to reject those teaches it something false. Fluent swaps are more likely to
be of this kind than clumsy ones, so the fluency-matched set must be checked by a
person. This is the main threat to the validity of the improvement.

## What to do (about 30 minutes)

1. Run `scripts/03_build_training_data.py`. It writes
   `results/training_data/audit_sample.csv` with 300 rows: the same 100 captions,
   each with one negative from the three sets (`swap`, `swap_nosym`, `swap_fluent`),
   in random order. The file does not say which set a row comes from.
2. Open the file in a spreadsheet and fill the `label` column of every row with one letter:

   | Label | Meaning | Example (caption → negative) |
   |---|---|---|
   | `D` | the negative describes a **different** scene: a valid negative | "a black dog and a white cat" → "a white dog and a black cat" |
   | `S` | the negative means the **same** as the caption: a false negative | "a dog and a cat on a sofa" → "a cat and a dog on a sofa" |
   | `N` | the negative is **not** a meaningful sentence: valid, but recognisable without the image | "a close up of a person" → "a person up of a close" |

   Judge from the two sentences only. An unlikely but understandable sentence ("a horse
   riding a man") is `D`, not `N`. If you cannot decide between `D` and `S`, choose `S`
   (the conservative choice for the improvement).
3. Save the file as CSV under the same name, commit it, and run

   ```bash
   python scripts/08_audit_summary.py
   ```

   It joins your labels with `audit_key.csv` and writes `results/tables/audit.md`:
   the share of each label per negative set with 95% intervals.

## How to read the result

- A clearly higher `S` share for `swap_fluent` than for `swap` means that part of
  any difference between the two conditions could come from false negatives, not
  from fluency. Report it as a limitation.
- A high `N` share for `swap` and a low one for `swap_fluent` is the intended effect
  of the fluency matching, confirmed by a person instead of a language model.
- Do not relabel after looking at the model results.
