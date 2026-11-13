"""The table builders run on a tiny synthetic results folder."""
import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yaml")

from compbind.benchmarks import BENCHMARKS, summarize  # noqa: E402
from compbind.report import ALL_TABLES, MissingInput, Results, load_published  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GROUPS = {"vg_relation": ["on", "has", "next to"], "vg_attribution": ["red_blue", "big_small"],
          "coco_order": ["coco_order"], "sugarcrepe": ["swap_obj", "swap_att", "add_obj", "replace_rel"]}


def item_table(benchmark, accuracy, seed):
    rng = np.random.RandomState(seed)
    n = 240
    correct = (rng.rand(n) < accuracy).astype(int)
    return pd.DataFrame({
        "item_id": [f"{benchmark}/{i}" for i in range(n)],
        "group": [GROUPS[benchmark][i % len(GROUPS[benchmark])] for i in range(n)],
        "image_id": [str(i // 2) for i in range(n)],
        "label": 1, "pred": correct, "correct": correct, "margin": 0.1,
        "pred_shuffled": 0, "correct_shuffled": (rng.rand(n) < 0.5).astype(int),
    })


def write_model(root, name, accuracy, seed, train=False):
    folder = os.path.join(root, name)
    os.makedirs(folder)
    summary = {"benchmarks": {}}
    for benchmark in BENCHMARKS:
        table = item_table(benchmark, accuracy, seed)
        suffix = "_seed0" if benchmark == "coco_order" else ""
        table.to_csv(os.path.join(folder, f"{benchmark}{suffix}_items.csv.gz"), index=False)
        entry = summarize(benchmark, table, n_boot=50)
        entry["shuffled_image"] = summarize(benchmark, table, column="correct_shuffled", n_boot=50)["headline"]
        summary["benchmarks"][benchmark] = entry
    json.dump(summary, open(os.path.join(folder, "summary.json"), "w"))
    json.dump({"zero_shot": {"cifar10": {"acc1": 0.9}}, "coco_retrieval": {"text_r1": 0.5, "image_r1": 0.3}},
              open(os.path.join(folder, "downstream.json"), "w"))
    if train:
        validation = [{"epoch": e, "pair_acc_fluent": 0.6, "pair_acc_swap": 0.7, "pair_acc_swap_nosym": 0.7,
                       "retrieval_acc": 0.9, "n": 10} for e in (0, 1)]
        json.dump({"train_seconds": 600, "seconds_per_step": 0.5, "peak_gpu_memory_mb": 8000,
                   "texts_per_step": 512, "validation": validation},
                  open(os.path.join(folder, "train_summary.json"), "w"))


@pytest.fixture()
def results_dir(tmp_path):
    root = str(tmp_path)
    for name, accuracy in (("clip", 0.6), ("negclip", 0.8), ("negclip_gelu", 0.78)):
        write_model(root, name, accuracy, seed=1)
    for condition, accuracy in (("none", 0.62), ("swap", 0.75), ("swap_fluent", 0.74)):
        for seed in (0, 1):
            write_model(root, f"{condition}_seed{seed}", accuracy, seed=10 + seed, train=True)
    os.makedirs(os.path.join(root, "text_only"))
    summary = {"benchmarks": {}}
    for benchmark in BENCHMARKS:
        table = item_table(benchmark, 0.7, seed=5)[["item_id", "group", "image_id", "label"]]
        rng = np.random.RandomState(7)
        entry = {}
        for scorer in ("gpt2", "gpt2sum", "grammar"):
            table[f"{scorer}_correct"] = (rng.rand(len(table)) < 0.7).astype(int)
            entry[scorer] = summarize(benchmark, table, column=f"{scorer}_correct", n_boot=50)
        table["image_in_negclip_train"] = (rng.rand(len(table)) < 0.5).astype(int)
        suffix = "_seed0" if benchmark == "coco_order" else ""
        table.to_csv(os.path.join(root, "text_only", f"{benchmark}{suffix}_items.csv.gz"), index=False)
        summary["benchmarks"][benchmark] = entry
    json.dump(summary, open(os.path.join(root, "text_only", "summary.json"), "w"))
    return root


def test_every_table_builds_or_reports_what_is_missing(results_dir):
    results = Results(results_dir)
    published = load_published(os.path.join(REPO, "configs", "published.yaml"))
    built = {}
    for builder in ALL_TABLES:
        try:
            table = builder(results, published, n_boot=50)
        except MissingInput:
            continue
        assert len(table.frame) > 0 and table.to_markdown().startswith("### ")
        built[table.name] = table
    # everything except the training-data table has its inputs in this fixture
    assert set(built) == {"reproduction_aro", "reproduction_sugarcrepe", "controls", "slices_text_only",
                          "slices_seen_images", "conditions_compositional", "conditions_differences",
                          "conditions_slices", "cost", "validation"}
    conditions = built["conditions_compositional"].frame
    assert conditions["Model"].str.contains("swap_fluent").any()
    assert "swap − none" in set(built["conditions_differences"].frame["Comparison"])


def test_published_and_measured_are_separate_columns(results_dir):
    published = load_published(os.path.join(REPO, "configs", "published.yaml"))
    frame = ALL_TABLES[0](Results(results_dir), published, n_boot=50).frame
    assert {"Published (%)", "Measured (%)", "Source"} <= set(frame.columns)
    assert (frame.loc[frame["Published (%)"] != "", "Source"] != "").all()


def test_empty_results_folder_reports_missing_inputs(tmp_path):
    published = load_published(os.path.join(REPO, "configs", "published.yaml"))
    for builder in ALL_TABLES:
        with pytest.raises(MissingInput):
            builder(Results(str(tmp_path)), published, n_boot=10)
