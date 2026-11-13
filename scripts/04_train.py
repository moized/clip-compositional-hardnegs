#!/usr/bin/env python
"""Fine-tune CLIP ViT-B/32 with one kind of negative caption.

    python scripts/04_train.py --negatives swap_fluent --seed 0 --coco-roots /kaggle/input

Conditions: none | random | swap | swap_nosym | swap_fluent   (see compbind/training.py)
Defaults follow the released NegCLIP command: 5 epochs, batch 256, lr 1e-6, 50 warm-up steps.

Writes <runs-dir>/<negatives>_seed<seed>/:
  model.pt             fine-tuned weights (not committed to git)
  train_log.jsonl      loss and learning rate during training
  train_summary.json   settings, training time, peak GPU memory, validation accuracies
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd  # noqa: E402

from compbind.coco import CocoImages  # noqa: E402
from compbind.paths import DataPaths, add_common_args, pick_device  # noqa: E402
from compbind.training import CONDITIONS, train  # noqa: E402
from compbind.utils import run_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--negatives", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop early (for quick tests).")
    parser.add_argument("--no-amp", action="store_true", help="Full precision instead of mixed.")
    parser.add_argument("--no-save", action="store_true", help="Do not write model.pt.")
    parser.add_argument("--full-size-images", action="store_true",
                        help="Decode the original images instead of the reduced copies made by step 03.")
    parser.add_argument("--no-grad-checkpointing", action="store_true",
                        help="Keep all activations in memory (faster, needs far more GPU memory).")
    parser.add_argument("--pairs", default=None, help="Pair table (default: <data-dir>/training/pairs.csv.gz).")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    device = pick_device(args.device)
    paths = DataPaths(args.data_dir)
    pairs = pd.read_csv(args.pairs or os.path.join(paths.training, "pairs.csv.gz"))
    images = CocoImages(args.coco_roots, download_dir=paths.coco_download)
    out_dir = os.path.join(args.runs_dir, f"{args.negatives}_seed{args.seed}")
    summary = train(pairs, images, out_dir, negatives=args.negatives, seed=args.seed,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, warmup=args.warmup, workers=args.workers,
                    device=device, amp=not args.no_amp, max_steps=args.max_steps,
                    save_model=not args.no_save, run_info=run_record(args, device=device),
                    small_dir=None if args.full_size_images else paths.train_images_small,
                    grad_checkpointing=not args.no_grad_checkpointing)
    print(f"Done: {summary['steps']} steps in {summary['train_seconds']:.0f}s, "
          f"peak GPU memory {summary['peak_gpu_memory_mb']} MB -> {out_dir}")


if __name__ == "__main__":
    main()
