# Audit of the training negatives

A swapped caption is only a useful negative if it describes a **different** scene.
Some swaps keep the meaning ("a dog and a cat" / "a cat and a dog"); training the
model to reject those teaches it something false. Fluent swaps may be of this kind
more often than clumsy ones, so the three sets of training negatives were audited.
This is the main threat to the validity of the improvement.

## Sample

`scripts/03_build_training_data.py` draws 100 training captions and, for each, one
negative from each set (`swap`, `swap_nosym`, `swap_fluent`): 300 rows in random order
in `results/training_data/audit_sample.csv`. That file does not say which set a row comes
from; the key is in `results/training_data/audit_key.csv`.

## Labels

| Label | Meaning | Example (caption → negative) |
|---|---|---|
| `D` | the negative describes a **different** scene: a valid negative | "a black dog and a white cat" → "a white dog and a black cat" |
| `S` | the negative means the **same** as the caption: a false negative | "a dog and a cat on a sofa" → "a cat and a dog on a sofa" |
| `N` | the negative is **not** a meaningful sentence: valid, but recognisable without the image | "a close up of a person" → "a person up of a close" |

Each pair is judged from the two sentences only. An unlikely but understandable sentence
("a horse riding a man") is `D`, not `N`. Between `D` and `S`, a doubtful case is `S`, the
conservative choice for the improvement.

## Labeller

One labeller, the AI assistant (Claude), labelled the 300 rows blind to the set each row came
from, but after the model results were known; there is no agreement measure. This is recorded
in `HYPOTHESES.md` (changes after finalising) and stated as a limitation in the report.

## Result

`scripts/08_audit_summary.py` joins the labels with the key and writes
`results/tables/audit.md`: the share of each label per set, with 95% Wilson intervals.

| Set | `D` | `S` | `N` |
|---|---|---|---|
| `swap` | 54% | 17% | 29% |
| `swap_nosym` | 47% | 10% | 43% |
| `swap_fluent` | 61% | 19% | 20% |

Fluency matching did not add false negatives (19% against 17%, with overlapping intervals)
and reduced the share of non-sentences (20% against 29%), its intended effect.
