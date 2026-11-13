#!/usr/bin/env python
"""Evaluate one CLIP model on ARO (VG-Relation, VG-Attribution, COCO-Order) and SugarCrepe.

    python scripts/01_eval_compositional.py --model openai  --name clip
    python scripts/01_eval_compositional.py --model negclip --name negclip
    python scripts/01_eval_compositional.py --model negclip --name negclip_gelu --gelu
    python scripts/01_eval_compositional.py --model path/to/checkpoint.pt --name my_run

For each benchmark it writes, under <out-dir>/<name>/:
  <benchmark>_items.csv.gz   one row per test item (prediction, margin, shuffled-image control)
  summary.json               headline numbers with 95% intervals, plus the run record

`--gelu` builds the model with the standard GELU activation instead of QuickGELU.
OpenAI CLIP and NegCLIP were trained with QuickGELU; see compbind/models.py.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from compbind.benchmarks import BENCHMARKS, get_items, items_fingerprint, summarize  # noqa: E402
from compbind.coco import CocoImages  # noqa: E402
from compbind.models import activation_name, load_clip  # noqa: E402
from compbind.paths import DataPaths, add_common_args, pick_device  # noqa: E402
from compbind.scoring import Embedder, score_items  # noqa: E402
from compbind.stats import mean_std  # noqa: E402
from compbind.utils import (Timer, ensure_dir, load_json, run_record, save_json,  # noqa: E402
                            seed_everything)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--model", required=True,
                        help='"openai", "negclip" (released checkpoint) or a checkpoint path.')
    parser.add_argument("--name", required=True, help="Folder name for the results of this model.")
    parser.add_argument("--gelu", action="store_true", help="Use GELU instead of QuickGELU.")
    parser.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--order-seeds", nargs="*", type=int, default=[0, 1, 2],
                        help="Seeds for COCO-Order's random shuffles.")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="Quick test: use N random items.")
    parser.add_argument("--amp", action="store_true", help="Half precision on GPU (faster).")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates.")
    parser.add_argument("--save-scores", action="store_true",
                        help="Also store every caption's similarity score (larger files).")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow full unpickling of the checkpoint (only for trusted files).")
    args = parser.parse_args()

    seed_everything(0)
    device = pick_device(args.device)
    paths = DataPaths(args.data_dir)
    source = paths.negclip_checkpoint if args.model == "negclip" else args.model
    bundle = load_clip(source, device=device, quick_gelu=not args.gelu,
                       trust_checkpoint=args.trust_checkpoint)
    print(f"Model: {args.model} | arch {bundle.arch} | activation {activation_name(bundle.model)} "
          f"| device {device}")
    images = CocoImages(args.coco_roots, download_dir=paths.coco_download)
    embedder = Embedder(bundle, batch_size=args.batch_size, num_workers=args.workers, amp=args.amp)
    out_dir = ensure_dir(os.path.join(args.out_dir, args.name))
    summary_path = os.path.join(out_dir, "summary.json")
    summary = {"run": run_record(args, model_source=source, arch=bundle.arch,
                                 activation=activation_name(bundle.model), device=device),
               "benchmarks": {}}
    if os.path.isfile(summary_path):   # keep benchmarks evaluated by an earlier call
        previous = load_json(summary_path)
        summary["benchmarks"] = dict(previous.get("benchmarks", {}))
        summary["earlier_runs"] = previous.get("earlier_runs", []) + [previous.get("run")]

    for name in args.benchmarks:
        seeds = args.order_seeds if name == "coco_order" else [0]
        per_seed = []
        for seed in seeds:
            with Timer() as timer:
                items = get_items(name, paths, images, seed=seed, limit=args.limit)
                table = score_items(embedder, items, shuffle_seed=0)
            suffix = f"_seed{seed}" if name == "coco_order" else ""
            saved = table if args.save_scores else table.drop(
                columns=[c for c in table.columns if c[0] == "s" and c[1:].isdigit()])
            saved.assign(margin=saved["margin"].round(4)).to_csv(
                os.path.join(out_dir, f"{name}{suffix}_items.csv.gz"), index=False)
            result = summarize(name, table, n_boot=args.n_boot)
            result["shuffled_image"] = summarize(name, table, column="correct_shuffled",
                                                 n_boot=args.n_boot)["headline"]
            result["seconds"] = round(timer.seconds, 1)
            result["seed"] = seed
            result["items_sha1"] = items_fingerprint(items)
            per_seed.append(result)
            print(f"{name:15s} seed {seed}: {100 * result['headline']:.2f}%  "
                  f"(shuffled image: {100 * result['shuffled_image']:.2f}%, "
                  f"chance {100 * result['chance']:.0f}%, {len(table)} items, {timer.seconds:.0f}s)")
        entry = per_seed[0] if len(per_seed) == 1 else {
            "per_seed": per_seed,
            "headline": float(np.mean([r["headline"] for r in per_seed])),
            "headline_over_seeds": mean_std([r["headline"] for r in per_seed]),
            "shuffled_image": float(np.mean([r["shuffled_image"] for r in per_seed])),
            "chance": per_seed[0]["chance"], "n_items": per_seed[0]["n_items"],
        }
        summary["benchmarks"][name] = entry
        save_json(summary, summary_path)

    print("Saved", summary_path)


if __name__ == "__main__":
    main()
