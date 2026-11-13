#!/usr/bin/env python
"""Collect test items on which two models disagree, for the failure analysis.

    python scripts/09_examples.py --benchmark sugarcrepe --a swap_fluent_seed0 --b none_seed0
    python scripts/09_examples.py --benchmark vg_relation --a negclip --b clip --n 15

`--a` and `--b` are result folders under results/. For each of three categories
  A right, B wrong  |  A wrong, B right  |  both wrong
the script draws N random items and writes them, with their captions, to
results/examples/<benchmark>__<a>__vs__<b>.csv. Favourable and unfavourable cases are
drawn the same way, so the sample is not cherry-picked. The text-only model's
preference (gpt2_gap > 0: GPT-2 alone prefers the true caption) is added when
results/text_only exists.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd  # noqa: E402

from compbind.benchmarks import BENCHMARKS, get_items  # noqa: E402
from compbind.coco import CocoImages  # noqa: E402
from compbind.paths import DataPaths, add_common_args  # noqa: E402
from compbind.report import Results, aligned  # noqa: E402
from compbind.utils import ensure_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--a", required=True, help="Result folder of model A, e.g. swap_fluent_seed0.")
    parser.add_argument("--b", required=True, help="Result folder of model B, e.g. none_seed0.")
    parser.add_argument("--n", type=int, default=10, help="Items per category.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random draw.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results = Results(args.results_dir)
    table_a, table_b = results.items(args.a, args.benchmark), results.items(args.b, args.benchmark)
    if table_a is None or table_b is None:
        raise SystemExit(f"Per-item results of {args.a} or {args.b} are missing for {args.benchmark}.")
    tables = [table_a, table_b]
    blind = results.blind(args.benchmark)
    if blind is not None:
        tables.append(blind)
    tables = aligned(tables)
    merged = pd.DataFrame({
        "item_id": tables[0]["item_id"], "group": tables[0]["group"], "image_id": tables[0]["image_id"],
        "a_correct": tables[0]["correct"], "b_correct": tables[1]["correct"],
        "a_margin": tables[0]["margin"], "b_margin": tables[1]["margin"],
    })
    if blind is not None:
        merged["gpt2_gap"] = tables[2]["gpt2_gap"].round(3)

    paths = DataPaths(args.data_dir)
    items = {item.item_id: item for item in
             get_items(args.benchmark, paths, CocoImages(args.coco_roots, download_dir=paths.coco_download))}
    categories = (("A right, B wrong", (merged["a_correct"] == 1) & (merged["b_correct"] == 0)),
                  ("A wrong, B right", (merged["a_correct"] == 0) & (merged["b_correct"] == 1)),
                  ("both wrong", (merged["a_correct"] == 0) & (merged["b_correct"] == 0)))
    rows = []
    for label, mask in categories:
        part = merged[mask]
        print(f"{label}: {len(part)} of {len(merged)} items")
        for _, row in part.sample(min(args.n, len(part)), random_state=args.seed).iterrows():
            item = items[row["item_id"]]
            wrong = [c for i, c in enumerate(item.captions) if i != item.label]
            rows.append({"category": label, **row.to_dict(), "true_caption": item.captions[item.label],
                         "false_captions": " || ".join(wrong), "image_file": os.path.basename(item.image_path)})
    out_dir = ensure_dir(os.path.join(args.results_dir, "examples"))
    out_path = os.path.join(out_dir, f"{args.benchmark}__{args.a}__vs__{args.b}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Wrote", out_path, "(A =", args.a + ", B =", args.b + ")")


if __name__ == "__main__":
    main()
