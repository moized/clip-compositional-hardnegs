#!/usr/bin/env python
"""Text-only ("blind") baselines and per-item metadata for every benchmark.

    python scripts/02_eval_text_only.py

A blind model never sees the image. For each test item it picks the caption that
a language model finds most fluent. Three variants are reported:
  gpt2      highest mean log-probability per token under GPT-2
  gpt2sum   highest total log-probability under GPT-2 (favours shorter captions)
  grammar   highest grammatical-acceptability probability (SugarCrepe's "Grammar model")

If a blind model beats chance, the benchmark can be partly solved from text alone.

Writes, under <out-dir>/text_only/:
  <benchmark>_items.csv.gz   per item: blind predictions, fluency gap, and whether the item's
                             image (or caption) is in the released NegCLIP training set
  summary.json               blind accuracies with 95% intervals and overlap statistics
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compbind import overlap  # noqa: E402
from compbind.benchmarks import BENCHMARKS, get_items, items_fingerprint, summarize  # noqa: E402
from compbind.coco import CocoImages  # noqa: E402
from compbind.paths import DataPaths, add_common_args, pick_device  # noqa: E402
from compbind.textonly import GrammarScorer, Gpt2Scorer, score_unique  # noqa: E402
from compbind.utils import ensure_dir, run_record, save_json, seed_everything  # noqa: E402


def blind_table(items: list, gpt2: Gpt2Scorer, grammar: GrammarScorer) -> pd.DataFrame:
    """Per-item blind predictions. Ties go to the first candidate, as in scoring.py."""
    k = len(items[0].captions)
    flat = [caption for item in items for caption in item.captions]
    unique = list(dict.fromkeys(flat))
    gpt2_scores = gpt2.score(unique, progress=True)
    grammar_scores = grammar.score(unique, progress=True)
    index = {text: i for i, text in enumerate(unique)}
    where = np.array([index[t] for t in flat]).reshape(len(items), k)
    labels = np.array([item.label for item in items])
    rows = np.arange(len(items))
    table = pd.DataFrame({
        "item_id": [item.item_id for item in items],
        "group": [item.group for item in items],
        "image_id": [item.image_id for item in items],
        "label": labels,
    })
    for name, values in (("gpt2", gpt2_scores["mean"]), ("gpt2sum", gpt2_scores["sum"]),
                         ("grammar", grammar_scores["prob"])):
        scores = values[where]
        table[f"{name}_correct"] = (scores.argmax(axis=1) == labels).astype(int)
        wrong = scores.copy()
        wrong[rows, labels] = -np.inf
        # > 0 means the true caption looks better to the blind model than every wrong one
        table[f"{name}_gap"] = (scores[rows, labels] - wrong.max(axis=1)).astype(np.float32)
    table["n_tokens_true"] = gpt2_scores["n_tokens"][where[rows, labels]]
    return table


def add_overlap(table: pd.DataFrame, items: list, benchmark: str, paths: DataPaths) -> None:
    """Flag items whose image / true caption is in the released NegCLIP training file."""
    if not overlap.have_overlap_files(paths):
        return
    train_ids, train_captions = overlap.negclip_training_set(paths.negclip_train_tsv)
    vg_map = overlap.vg_to_coco(paths.vg_image_data)
    coco_ids = [overlap.item_coco_id(benchmark, item.image_id, vg_map) for item in items]
    table["image_is_coco"] = [int(c is not None) for c in coco_ids]
    table["image_in_negclip_train"] = [int(c is not None and c in train_ids) for c in coco_ids]
    table["caption_in_negclip_train"] = [
        int(" ".join(item.captions[item.label].lower().split()) in train_captions) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--order-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="Quick test: use N random items.")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    seed_everything(0)
    device = pick_device(args.device)
    paths = DataPaths(args.data_dir)
    images = CocoImages(args.coco_roots, download_dir=paths.coco_download)
    gpt2 = Gpt2Scorer(device=device, batch_size=args.batch_size)
    grammar = GrammarScorer(device=device, batch_size=args.batch_size)
    out_dir = ensure_dir(os.path.join(args.out_dir, "text_only"))
    summary = {"run": run_record(args, device=device), "benchmarks": {}}

    for name in args.benchmarks:
        seeds = args.order_seeds if name == "coco_order" else [0]
        entries = []
        for seed in seeds:
            items = get_items(name, paths, images, seed=seed, limit=args.limit)
            table = blind_table(items, gpt2, grammar)
            add_overlap(table, items, name, paths)
            suffix = f"_seed{seed}" if name == "coco_order" else ""
            table.to_csv(os.path.join(out_dir, f"{name}{suffix}_items.csv.gz"), index=False)
            entry = {"seed": seed, "n_items": int(len(table)), "items_sha1": items_fingerprint(items)}
            for scorer in ("gpt2", "gpt2sum", "grammar"):
                entry[scorer] = summarize(name, table, column=f"{scorer}_correct", n_boot=args.n_boot)
            for flag in ("image_is_coco", "image_in_negclip_train", "caption_in_negclip_train"):
                if flag in table:
                    entry[flag] = float(table[flag].mean())
            entries.append(entry)
            print(f"{name:15s} seed {seed}: blind accuracy  gpt2 {100 * entry['gpt2']['headline']:.2f}%  "
                  f"gpt2sum {100 * entry['gpt2sum']['headline']:.2f}%  "
                  f"grammar {100 * entry['grammar']['headline']:.2f}%"
                  + (f"  | image in NegCLIP train: {100 * entry['image_in_negclip_train']:.1f}%"
                     if "image_in_negclip_train" in entry else ""))
        summary["benchmarks"][name] = entries[0] if len(entries) == 1 else {"per_seed": entries}
        save_json(summary, os.path.join(out_dir, "summary.json"))
    print("Saved", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
