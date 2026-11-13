#!/usr/bin/env python
"""Build the fine-tuning data and the fluency-matched negatives.

    python scripts/03_build_training_data.py --coco-roots /kaggle/input

Steps:
 1. Take the captions of COCO train2014 images that no benchmark uses.
 2. Generate every NegCLIP-style swap for every caption (spaCy).
 3. Score captions and swaps with GPT-2; gap = caption fluency - swap fluency.
 4. Select one fluency-matched pair per image (as many gap<0 as gap>0 in each bin).
 5. Split into train / validation, attach NegCLIP-style negatives for the same
    captions, and make sure the images are on disk.

Outputs in <data-dir>/training/:
  pairs.csv.gz          one row per image: caption, neg_fluent, neg_swap, neg_swap_nosym, split
  stats.json            sizes, text-only accuracy of each negative set, leak checks
  split_manifest.json   image ids of the train and validation sets
  gap_histograms.json   distribution of the fluency gap for each negative set (for a figure)
  audit_sample.csv      random (caption, negative) pairs to label by hand; which negative
                        set a row comes from is hidden in audit_key.csv (see docs/AUDIT.md)
  negclip_check.json    with --check-against-negclip: agreement with the authors' negatives
The small files (everything except pairs.csv.gz) are also copied to
<results-dir>/training_data/ so that they can be committed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compbind import refine, traindata  # noqa: E402
from compbind.coco import CocoImages, folder_of  # noqa: E402
from compbind.negatives import generate_candidates  # noqa: E402
from compbind.paths import DataPaths, add_common_args, pick_device  # noqa: E402
from compbind.textonly import GrammarScorer, Gpt2Scorer  # noqa: E402
from compbind.utils import (Timer, add_config_option, ensure_dir, load_json, run_record,  # noqa: E402
                            save_json, seed_everything)


def grammar_blind_accuracy(grammar: GrammarScorer, captions: list, negatives: list) -> float:
    """Text-only accuracy of the grammar model on (caption, negative) pairs."""
    if not captions:
        return float("nan")
    scores = grammar.score(captions + negatives)["prob"]
    n = len(captions)
    return float((scores[:n] > scores[n:]).mean())


GAP_EDGES = np.round(np.arange(-2.0, 4.0001, 0.1), 2)   # histogram bins for the fluency gap
# Options that decide which pairs are built. An existing table is reused only if they are unchanged.
SETTINGS = ("seed", "max_images", "max_train_pairs", "val_pairs", "bin_width", "max_abs_gap")


def gap_histogram(gap) -> dict:
    """Counts of the fluency gap in fixed bins (values outside the range go to the end bins)."""
    clipped = np.clip(np.asarray(gap, dtype=np.float64), GAP_EDGES[0], GAP_EDGES[-1] - 1e-9)
    counts, _ = np.histogram(clipped, bins=GAP_EDGES)
    return {"counts": counts.tolist(), "n": int(len(clipped))}


def check_against_negclip(paths: DataPaths, gpt2: Gpt2Scorer, sample_size: int, seed: int) -> dict:
    """How well do our swap rules reproduce the authors' released NegCLIP negatives?"""
    csv.field_size_limit(10 ** 9)
    with open(paths.negclip_train_tsv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    sample = random.Random(seed).sample(rows, min(sample_size, len(rows)))
    generated = generate_candidates([r["title"] for r in sample])
    squash = lambda s: re.sub(r"\s+", "", s.lower())  # noqa: E731  (ignore spacing and case)
    total = found = 0
    captions, matched_negatives = [], []
    released_per_caption, ours_per_caption = [], []
    for row, (caption, swaps) in zip(sample, generated):
        ours = {squash(s.text): s.text for s in swaps}
        released = ast.literal_eval(row["neg_caption"])
        released_per_caption.append(len(released))
        ours_per_caption.append(len({s.kind for s in swaps}))
        for negative in released:
            total += 1
            if squash(negative) in ours:
                found += 1
                captions.append(caption)
                matched_negatives.append(ours[squash(negative)])
    lp_caption = gpt2.score(captions)["mean"]
    lp_negative = gpt2.score(matched_negatives)["mean"]
    return {
        "captions_sampled": len(sample),
        "released_negatives": total,
        "reproduced_by_our_rules": found,
        "reproduced_share": found / max(total, 1),
        "released_negatives_per_caption": float(np.mean(released_per_caption)),
        "our_negatives_per_caption_one_per_type": float(np.mean(ours_per_caption)),
        "gpt2_blind_accuracy_on_released_negatives": refine.blind_accuracy(lp_caption - lp_negative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Use only this many train2014 images (default: all, about 80k).")
    parser.add_argument("--max-train-pairs", type=int, default=None,
                        help="Cap on the number of training pairs (default: all matched pairs).")
    parser.add_argument("--val-pairs", type=int, default=2000)
    parser.add_argument("--bin-width", type=float, default=0.02, help="Gap bin width (nats/token).")
    parser.add_argument("--max-abs-gap", type=float, default=1.0)
    parser.add_argument("--spacy-procs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-against-negclip", action="store_true")
    parser.add_argument("--check-sample", type=int, default=20000)
    parser.add_argument("--audit-size", type=int, default=100, help="Pairs per negative set to audit.")
    parser.add_argument("--results-dir", default="results",
                        help="Small output files are copied to <results-dir>/training_data/.")
    parser.add_argument("--no-image-download", action="store_true",
                        help="Do not fetch missing training images (they must exist already).")
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="Rebuild even if pairs.csv.gz exists.")
    add_config_option(parser, "training_data")
    args = parser.parse_args()

    seed_everything(args.seed)
    paths = DataPaths(args.data_dir)
    out_dir = ensure_dir(paths.training)
    pairs_path = os.path.join(out_dir, "pairs.csv.gz")
    settings = {name: getattr(args, name) for name in SETTINGS}
    stats_path = os.path.join(out_dir, "stats.json")
    if os.path.isfile(pairs_path) and not args.force:
        previous = load_json(stats_path).get("settings") if os.path.isfile(stats_path) else None
        if previous == settings:
            print(f"{pairs_path} already exists with the same settings (use --force to rebuild it).")
            finish(args, paths, pd.read_csv(pairs_path))
            return
        print(f"{pairs_path} was built with other settings ({previous}); rebuilding with {settings}.")
    stats: dict = {"settings": settings}

    # 1-2: pool and candidate swaps. spaCy may start extra processes here, so this step
    # comes before anything touches the GPU (a process that holds a GPU must not be forked).
    pool, stats["pool"] = traindata.leak_free_pool(paths, args.max_images, args.seed)
    print(f"Pool: {stats['pool']['pool_images']} images, {stats['pool']['pool_captions']} captions "
          f"({stats['pool']['images_removed_because_in_aro_test']} images removed: in ARO test sets)")
    with Timer() as timer:
        candidates = traindata.build_candidates(pool, n_process=args.spacy_procs)
    stats["candidates"] = {"n": int(len(candidates)), "by_kind": traindata.kind_counts(candidates),
                           "symmetric_share": float(candidates["symmetric"].mean()),
                           "seconds": round(timer.seconds, 1)}

    device = pick_device(args.device)
    stats["run"] = run_record(args, device=device)
    gpt2 = Gpt2Scorer(device=device, batch_size=args.batch_size)
    if args.check_against_negclip:
        check = check_against_negclip(paths, gpt2, args.check_sample, args.seed)
        save_json(check, os.path.join(out_dir, "negclip_check.json"))
        print(f"Our rules reproduce {100 * check['reproduced_share']:.1f}% of the authors' negatives; "
              f"GPT-2 alone separates them from the true caption "
              f"{100 * check['gpt2_blind_accuracy_on_released_negatives']:.1f}% of the time.")

    # 3: fluency gaps
    with Timer() as timer:
        traindata.add_fluency_gap(pool, candidates, gpt2)
    stats["candidates"]["gpt2_seconds"] = round(timer.seconds, 1)
    stats["all_candidates"] = refine.describe_pool(candidates["gap"].values, candidates["kind"],
                                                   candidates["symmetric"])
    print(f"{len(candidates)} candidate swaps; GPT-2 alone picks the true caption "
          f"{100 * stats['all_candidates']['blind_accuracy_gpt2']:.1f}% of the time")

    # 4-5: fluency-matched selection, split, NegCLIP-style negatives for the same captions
    matched = refine.build_matched_pairs(candidates, bin_width=args.bin_width,
                                         max_abs_gap=args.max_abs_gap, seed=args.seed)
    pairs = traindata.assemble_pairs(pool, candidates, matched, n_val=args.val_pairs,
                                     max_train=args.max_train_pairs, seed=args.seed)
    stats["selection"] = {"matched_pairs_available": int(len(matched)),
                          "train_pairs": int((pairs["split"] == "train").sum()),
                          "val_pairs": int((pairs["split"] == "val").sum()),
                          "bin_width": args.bin_width, "max_abs_gap": args.max_abs_gap}

    grammar = GrammarScorer(device=device, batch_size=args.batch_size)
    stats["negative_sets"] = {}
    histograms = {"edges": GAP_EDGES.tolist(), "split": "train",
                  "sets": {"all_candidates": gap_histogram(candidates["gap"].values)}}
    for split in ("train", "val"):
        part = pairs[pairs["split"] == split]
        entry = {}
        fluent = refine.describe_pool(part["gap_fluent"].values, part["kind_fluent"])
        fluent["blind_accuracy_grammar"] = grammar_blind_accuracy(
            grammar, part["caption"].tolist(), part["neg_fluent"].tolist())
        entry["neg_fluent"] = fluent
        if split == "train":
            histograms["sets"]["neg_fluent"] = gap_histogram(part["gap_fluent"].values)
        for column in ("neg_swap", "neg_swap_nosym"):
            info = traindata.pool_gaps(part, candidates, column)
            described = refine.describe_pool(info["gap"].values, info["kind"], info["symmetric"])
            captions = pool["caption"].values[info["caption_index"].values].tolist()
            described["blind_accuracy_grammar"] = grammar_blind_accuracy(
                grammar, captions, info["text"].tolist())
            described["negatives_per_caption"] = float(len(info) / max(len(part), 1))
            entry[column] = described
            if split == "train":
                histograms["sets"][column] = gap_histogram(info["gap"].values)
        stats["negative_sets"][split] = entry
        print(f"{split}: {len(part)} pairs | text-only accuracy (GPT-2 / grammar model): "
              + " | ".join(f"{c} {100 * entry[c]['blind_accuracy_gpt2']:.1f}% / "
                           f"{100 * entry[c]['blind_accuracy_grammar']:.1f}%"
                           for c in ("neg_swap", "neg_swap_nosym", "neg_fluent")))

    # leak checks and manifests
    stats["leak_checks"] = traindata.leak_report(pairs["image_id"], paths)
    if not stats["leak_checks"]["leak_free"]:
        raise RuntimeError(f"Training images overlap a test set: {stats['leak_checks']}")
    pairs.drop(columns=["caption_index"]).to_csv(pairs_path, index=False)
    save_json({split: sorted(int(i) for i in pairs.loc[pairs["split"] == split, "image_id"])
               for split in ("train", "val")}, os.path.join(out_dir, "split_manifest.json"))

    save_json(histograms, os.path.join(out_dir, "gap_histograms.json"))

    # Audit sample: the same captions with one negative from each set, in random order.
    # The set a row comes from is kept in a separate key file, so labelling is blind.
    rng = random.Random(args.seed)
    audit = []
    train = pairs[pairs["split"] == "train"]
    for column in ("neg_fluent", "neg_swap", "neg_swap_nosym"):
        for _, row in train.sample(min(args.audit_size, len(train)), random_state=args.seed).iterrows():
            negative = row[column] if column == "neg_fluent" else rng.choice(json.loads(row[column]))
            audit.append({"negative_set": column, "image_file": row["image_file"],
                          "caption": row["caption"], "negative": negative})
    rng.shuffle(audit)
    audit = pd.DataFrame(audit)
    audit.insert(0, "audit_id", np.arange(1, len(audit) + 1))
    audit[["audit_id", "negative_set", "image_file"]].to_csv(
        os.path.join(out_dir, "audit_key.csv"), index=False)
    audit[["audit_id", "caption", "negative"]].assign(label="").to_csv(
        os.path.join(out_dir, "audit_sample.csv"), index=False)

    save_json(stats, stats_path)
    print("Wrote", pairs_path, "and stats.json")
    finish(args, paths, pairs)


def finish(args, paths: DataPaths, pairs: pd.DataFrame) -> None:
    """Make sure the training images are on disk and copy the small files to the results folder."""
    out_dir = paths.training
    if not args.no_image_download:
        images = CocoImages(args.coco_roots, download_dir=paths.coco_download)
        wanted = {int(i): folder_of(f) for i, f in zip(pairs["image_id"], pairs["image_file"])}
        fetched = images.ensure(wanted, workers=args.download_workers)
        print(f"Training images: {fetched['present']} present, {fetched['downloaded']} downloaded")
    if args.results_dir:
        target = ensure_dir(os.path.join(args.results_dir, "training_data"))
        for name in ("stats.json", "split_manifest.json", "gap_histograms.json", "negclip_check.json",
                     "audit_sample.csv", "audit_key.csv"):
            source, copy = os.path.join(out_dir, name), os.path.join(target, name)
            if not os.path.isfile(source):
                continue
            if name == "audit_key.csv" and os.path.isfile(copy):
                with open(source, "rb") as new, open(copy, "rb") as old:
                    if new.read() != old.read():
                        print(f"Kept the audit files already in {target}; they belong to an earlier build of "
                              "the training data. Delete them and run this script again to audit the new pairs.")
            # never overwrite an audit file: it may already contain hand-made labels
            if not (name.startswith("audit") and os.path.isfile(copy)):
                shutil.copy2(source, copy)
        print("Copied the small files to", target)


if __name__ == "__main__":
    main()
