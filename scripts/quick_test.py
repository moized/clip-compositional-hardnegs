#!/usr/bin/env python
"""Quick end-to-end check that the code runs (about two minutes on a laptop CPU).

    python scripts/quick_test.py            # 100 test items, 2 training steps
    python scripts/quick_test.py --full     # all 666 Swap-Att items, compared with the published value

What it does:
 1. generates swapped captions for three sentences (spaCy) and checks the NegCLIP loss;
 2. downloads the SugarCrepe Swap-Att file and the COCO images it needs, loads OpenAI
    CLIP ViT-B/32 and measures its accuracy (with --full: must equal the published 63.96%);
 3. runs two fine-tuning steps with swapped negatives on a handful of pairs.
Nothing is saved except the downloaded files under --data-dir. Step 3 reuses the test
images of step 2 only because they are already on disk; it is a smoke test, not an experiment.

Downloads: about 350 MB of CLIP weights plus 15-100 MB of images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SUBSET = "swap_att"
PUBLISHED_CLIP_SWAP_ATT = 63.96      # SugarCrepe (Hsieh et al., 2023), Table 6, CLIP ViT-B/32


def check(condition: bool, message: str) -> None:
    print(("  ok    " if condition else "  FAIL  ") + message)
    if not condition:
        raise SystemExit(f"Quick test failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("COMPBIND_DATA", "data"))
    parser.add_argument("--coco-roots", nargs="*", default=[])
    parser.add_argument("--items", type=int, default=100, help="Number of test items to score.")
    parser.add_argument("--full", action="store_true", help="Score the whole subset and check the value.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    start = time.time()

    import pandas as pd
    import torch

    from compbind.coco import CocoImages, coco_id, download_file
    from compbind.losses import clip_loss, negclip_loss
    from compbind.models import activation_name, load_clip
    from compbind.negatives import generate_candidates
    from compbind.paths import SUGARCREPE_COMMIT, SUGARCREPE_REPO, DataPaths, pick_device
    from compbind.scoring import Embedder, Item, score_items
    from compbind.training import train
    from compbind.utils import load_json, seed_everything

    seed_everything(0)
    device = pick_device(args.device)
    paths = DataPaths(args.data_dir)

    print("1. Swapped captions and the NegCLIP loss")
    captions = ["A black dog chases a white cat.", "the horse is eating the grass",
                "a man in a red shirt holding a blue umbrella"]
    generated = generate_candidates(captions)
    swaps = {swap.text for _, candidates in generated for swap in candidates}
    check("A white dog chases a black cat." in swaps, "adjective swap keeps spacing and capital letter")
    check("the grass is eating the horse" in swaps, "noun swap: 'the grass is eating the horse'")
    features = torch.nn.functional.normalize(torch.randn(4, 8, generator=torch.Generator().manual_seed(0)), dim=-1)
    scale = torch.tensor(100.0)
    check(abs(float(clip_loss(features, features, scale))) < 1e-3, "CLIP loss is ~0 for perfectly matched pairs")
    check(float(negclip_loss(features, features, features, scale)) > 0.3,
          "NegCLIP loss is large when the negative caption equals the true caption")

    print(f"2. CLIP ViT-B/32 on SugarCrepe {SUBSET}")
    url = f"https://raw.githubusercontent.com/{SUGARCREPE_REPO}/{SUGARCREPE_COMMIT}/data/{SUBSET}.json"
    examples = load_json(download_file(url, os.path.join(paths.sugarcrepe, f"{SUBSET}.json")))
    keys = list(examples) if args.full else list(examples)[:args.items]
    images = CocoImages(args.coco_roots, download_dir=paths.coco_download)
    images.ensure({coco_id(examples[k]["filename"]): "val2017" for k in keys}, workers=16)
    items = [Item(item_id=f"{SUBSET}/{k}", image_path=images.path(coco_id(examples[k]["filename"])),
                  captions=(examples[k]["negative_caption"], examples[k]["caption"]), label=1,
                  group=SUBSET, image_id=str(coco_id(examples[k]["filename"]))) for k in keys]
    bundle = load_clip("openai", device=device)
    check(activation_name(bundle.model) == "QuickGELU", "OpenAI weights are loaded with QuickGELU")
    table = score_items(Embedder(bundle, batch_size=64, num_workers=0), items)
    accuracy = 100 * table["correct"].mean()
    shuffled = 100 * table["correct_shuffled"].mean()
    print(f"        accuracy {accuracy:.2f}% on {len(items)} items (same captions, wrong image: {shuffled:.2f}%)")
    if args.full:
        check(abs(accuracy - PUBLISHED_CLIP_SWAP_ATT) < 0.2,
              f"matches the published {PUBLISHED_CLIP_SWAP_ATT}% (SugarCrepe, Table 6)")
    else:
        check(50 < accuracy < 80, "accuracy is in the expected range (published full-subset value: 63.96%)")

    if not args.skip_training:
        print("3. Two fine-tuning steps with swapped negatives")
        rows = []
        for item, (caption, candidates) in zip(items, generate_candidates([i.captions[1] for i in items])):
            texts = [swap.text for swap in candidates if not swap.symmetric]
            if texts:
                rows.append({"image_id": int(item.image_id), "caption": caption, "neg_fluent": texts[0],
                             "neg_swap": json.dumps(texts), "neg_swap_nosym": json.dumps(texts)})
            if len(rows) == 12:
                break
        pairs = pd.DataFrame(rows)
        pairs["split"] = ["train"] * 8 + ["val"] * (len(pairs) - 8)
        with tempfile.TemporaryDirectory() as out_dir:
            summary = train(pairs, images, out_dir, negatives="swap", seed=0, epochs=1, batch_size=4,
                            workers=0, device=device, amp=False, max_steps=2, save_model=False, log_every=1)
        check(summary["steps"] == 2 and summary["final_loss"] == summary["final_loss"],
              f"training ran 2 steps, loss {summary['final_loss']:.3f}")
        check(len(summary["validation"]) == 2, "validation ran before and after training")

    print(f"Quick test passed in {time.time() - start:.0f} s.")


if __name__ == "__main__":
    main()
