#!/usr/bin/env python
"""Zero-shot classification and COCO retrieval for one model (the cost of fine-tuning).

    python scripts/05_eval_downstream.py --model openai  --name clip
    python scripts/05_eval_downstream.py --model negclip --name negclip
    python scripts/05_eval_downstream.py --model runs/swap_fluent_seed0/model.pt --name swap_fluent_seed0

Zero-shot datasets are CLIP_benchmark names; cifar10 and cifar100 download by themselves.
ImageNet is not downloaded. To include it, pass the folder of the validation set:

    --zero-shot cifar10 cifar100 imagenet1k \
    --imagenet-root /kaggle/input/imagenet-object-localization-challenge

(the Kaggle layout, or one folder per class under <root>/val). `--imagenet-limit 10000`
uses a fixed random subset of the 50,000 validation images, which is faster and still
precise to about one percentage point.

Writes <out-dir>/<name>/downstream.json. Parts that are already in that file are kept
and not computed again (use --force to recompute).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from compbind.coco import CocoImages, load_karpathy  # noqa: E402
from compbind.downstream import coco_retrieval, zero_shot  # noqa: E402
from compbind.models import activation_name, load_clip  # noqa: E402
from compbind.paths import DataPaths, add_common_args, pick_device  # noqa: E402
from compbind.scoring import Embedder  # noqa: E402
from compbind.utils import (Timer, ensure_dir, load_json, run_record, save_json,  # noqa: E402
                            seed_everything)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--model", required=True, help='"openai", "negclip" or a checkpoint path.')
    parser.add_argument("--name", required=True)
    parser.add_argument("--gelu", action="store_true", help="Use GELU instead of QuickGELU.")
    parser.add_argument("--zero-shot", nargs="*", default=["cifar10", "cifar100"],
                        help="CLIP_benchmark dataset names.")
    parser.add_argument("--imagenet-root", default=None, help="Folder with the ImageNet validation set.")
    parser.add_argument("--imagenet-limit", type=int, default=None,
                        help="Use a fixed random subset of N ImageNet validation images.")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute parts that already exist.")
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()

    seed_everything(0)
    device = pick_device(args.device)
    paths = DataPaths(args.data_dir)
    source = paths.negclip_checkpoint if args.model == "negclip" else args.model
    out_dir = ensure_dir(os.path.join(args.out_dir, args.name))
    out_path = os.path.join(out_dir, "downstream.json")
    previous = load_json(out_path) if os.path.isfile(out_path) and not args.force else {}
    todo = [d for d in args.zero_shot if d not in previous.get("zero_shot", {})]
    need_retrieval = not args.skip_retrieval and "coco_retrieval" not in previous
    if not todo and not need_retrieval:
        print(f"{out_path} is complete; nothing to do.")
        return

    bundle = load_clip(source, device=device, quick_gelu=not args.gelu,
                       trust_checkpoint=args.trust_checkpoint)
    result = {"run": run_record(args, model_source=source, arch=bundle.arch,
                                activation=activation_name(bundle.model), device=device),
              "zero_shot": dict(previous.get("zero_shot", {}))}
    if "coco_retrieval" in previous:
        result["coco_retrieval"] = previous["coco_retrieval"]

    for dataset in todo:
        if dataset == "imagenet1k":
            if not args.imagenet_root:
                raise SystemExit("imagenet1k needs --imagenet-root (the validation set is not downloaded).")
            root, limit = args.imagenet_root, args.imagenet_limit
        else:
            root, limit = os.path.join(paths.root, "zeroshot", dataset.replace("/", "_")), None
        with Timer() as timer:
            metrics = zero_shot(bundle, dataset, root, batch_size=args.batch_size,
                                workers=args.workers, amp=args.amp, limit=limit)
        metrics["seconds"] = round(timer.seconds, 1)
        result["zero_shot"][dataset] = metrics
        print(f"zero-shot {dataset}: top-1 {100 * metrics['acc1']:.2f}%, top-5 {100 * metrics['acc5']:.2f}% "
              f"on {metrics['n_images']} images ({timer.seconds:.0f}s)")
        save_json(result, out_path)

    if need_retrieval:
        images = CocoImages(args.coco_roots, download_dir=paths.coco_download)
        embedder = Embedder(bundle, batch_size=args.batch_size, num_workers=args.workers, amp=args.amp)
        with Timer() as timer:
            retrieval = coco_retrieval(embedder, load_karpathy(paths.karpathy, "test"), images)
        retrieval["seconds"] = round(timer.seconds, 1)
        result["coco_retrieval"] = retrieval
        print(f"COCO retrieval: text R@1 {100 * retrieval['text_r1']:.2f}%  "
              f"image R@1 {100 * retrieval['image_r1']:.2f}%  ({timer.seconds:.0f}s)")
    save_json(result, out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
