#!/usr/bin/env python
"""Summarise the hand-labelled audit of the training negatives.

    python scripts/08_audit_summary.py

Before running it, open results/training_data/audit_sample.csv and fill the `label`
column of every row with one letter (see docs/AUDIT.md):
    D   the negative describes a Different scene from the caption (a valid negative)
    S   the negative means the Same as the caption (a false negative)
    N   the negative is Not a meaningful sentence (valid, but recognisable from text alone)
The file does not say which negative set a row comes from; that is stored in
audit_key.csv and joined here, so the labelling is blind.

Writes results/tables/audit.md and audit.csv: the share of each label per negative
set, with 95% Wilson intervals.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd  # noqa: E402

from compbind.report import Table  # noqa: E402
from compbind.utils import ensure_dir  # noqa: E402

LABELS = {"D": "different scene (valid)", "S": "same meaning (false negative)", "N": "not a meaningful sentence"}
SETS = {"neg_swap": "swap (NegCLIP-style)", "neg_swap_nosym": "swap_nosym", "neg_fluent": "swap_fluent"}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    folder = os.path.join(args.results_dir, "training_data")
    sample = pd.read_csv(os.path.join(folder, "audit_sample.csv"), dtype=str, keep_default_na=False)
    key = pd.read_csv(os.path.join(folder, "audit_key.csv"), dtype=str)
    sample["label"] = sample["label"].str.strip().str.upper()
    unknown = sorted(set(sample["label"]) - set(LABELS) - {""})
    if unknown:
        raise SystemExit(f"Unknown labels {unknown}; use only D, S or N.")
    merged = sample.merge(key, on="audit_id")
    done = merged[merged["label"] != ""]
    if done.empty:
        raise SystemExit("No row is labelled yet. Fill the `label` column of audit_sample.csv first.")
    rows = []
    for name, title in SETS.items():
        part = done[done["negative_set"] == name]
        row = {"Negative set": title, "Labelled pairs": len(part)}
        for label, description in LABELS.items():
            count = int((part["label"] == label).sum())
            low, high = wilson(count, len(part))
            row[f"{label}: {description} (%)"] = (f"{100 * count / len(part):.0f} [{100 * low:.0f}, {100 * high:.0f}]"
                                                 if len(part) else "")
        rows.append(row)
    table = Table("audit", "Manual audit of the training negatives", pd.DataFrame(rows), [
        f"{len(done)} of {len(merged)} sampled pairs labelled by hand, blind to the negative set.",
        "Share of each label in %, with a 95% Wilson interval.",
    ])
    out_dir = ensure_dir(os.path.join(args.results_dir, "tables"))
    with open(os.path.join(out_dir, "audit.md"), "w", encoding="utf-8") as f:
        f.write(table.to_markdown())
    table.frame.to_csv(os.path.join(out_dir, "audit.csv"), index=False)
    print(table.to_markdown())


if __name__ == "__main__":
    main()
