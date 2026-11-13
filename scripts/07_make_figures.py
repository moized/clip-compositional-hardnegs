#!/usr/bin/env python
"""Regenerate every figure of the report from the saved result files.

    python scripts/07_make_figures.py

Reads the same files as scripts/06_make_tables.py and writes PNG and PDF files to
results/figures/. A figure whose inputs do not exist yet is skipped with a message.
No model is run and no GPU is needed.

  fig1_reproduction   published vs. measured accuracy (CLIP, released NegCLIP)
  fig2_controls       text-only models and the wrong-image control on each benchmark
  fig3_fluency_gap    fluency gap of NegCLIP-style and fluency-matched training negatives
  fig4_conditions     every fine-tuning condition on every metric
  fig5_text_slices    accuracy on items a text-only model gets right / wrong
  fig6_validation     validation accuracy per epoch
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from compbind.figures import make_all  # noqa: E402
from compbind.report import Results, load_published  # noqa: E402
from compbind.utils import ensure_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--published", default=os.path.join("configs", "published.yaml"))
    parser.add_argument("--out-dir", default=None, help="Default: <results-dir>/figures")
    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir or os.path.join(args.results_dir, "figures"))
    written, skipped = make_all(Results(args.results_dir), load_published(args.published), out_dir)
    for path in written:
        print("wrote", path)
    for name, reason in skipped:
        print(f"skipped {name}: {reason}")


if __name__ == "__main__":
    main()
