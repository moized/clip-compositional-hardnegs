"""Build the report's tables from saved result files. No model is run here.

Every table mixes two kinds of numbers and keeps them apart:
* "published" values are read from configs/published.yaml (with the source table);
* "measured" values are read from the files written by scripts 01-05.

Folder layout expected under the results directory:
  clip/ negclip/ negclip_gelu/     reference models (summary.json, *_items.csv.gz, downstream.json)
  <condition>_seed<k>/             our fine-tuned models (the same files plus train_summary.json)
  text_only/                       text-only baselines per item
  training_data/                   statistics of the training negatives
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import CONDITIONS, aro
from .benchmarks import BENCHMARKS, CHANCE
from .stats import accuracy_ci, paired_difference
from .utils import load_json

BENCHMARK_LABELS = {"vg_relation": "VG-Relation", "vg_attribution": "VG-Attribution",
                    "coco_order": "COCO-Order", "sugarcrepe": "SugarCrepe"}
METRIC_LABELS = {"vg_relation": "macro accuracy", "vg_attribution": "macro accuracy",
                 "coco_order": "accuracy, 5 choices", "sugarcrepe": "mean of 7 subsets"}
REFERENCE_LABELS = {
    "clip": "CLIP ViT-B/32 (OpenAI)",
    "negclip": "NegCLIP released, QuickGELU (as trained)",
    "negclip_gelu": "NegCLIP released, GELU (as the released evaluation code builds it)",
}
CONDITION_LABELS = {
    "none": "none (CLIP-FT control)",
    "random": "random caption (control)",
    "swap": "swap (NegCLIP-style)",
    "swap_nosym": "swap_nosym (ablation)",
    "swap_fluent": "swap_fluent (fluency-matched)",
}
SUBSET_LABELS = {"replace_obj": "Replace-Obj", "replace_att": "Replace-Att", "replace_rel": "Replace-Rel",
                 "swap_obj": "Swap-Obj", "swap_att": "Swap-Att", "add_obj": "Add-Obj", "add_att": "Add-Att"}
# Pairs of conditions compared in the paired-difference table: (A, B) means "A minus B".
COMPARISONS = (("swap", "none"), ("swap_fluent", "none"), ("swap_fluent", "swap"),
               ("swap_nosym", "swap"), ("random", "none"), ("swap_fluent", "random"))


class MissingInput(Exception):
    """A table cannot be built because a result file does not exist yet."""


@dataclass
class Table:
    name: str                 # file name without extension
    title: str
    frame: pd.DataFrame
    notes: list = field(default_factory=list)

    def to_markdown(self) -> str:
        columns = [str(c) for c in self.frame.columns]
        lines = [f"### {self.title}", "", "| " + " | ".join(columns) + " |",
                 "|" + "|".join("---" for _ in columns) + "|"]
        for row in self.frame.itertuples(index=False):
            lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
        if self.notes:
            lines += [""] + [f"- {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- reading files

class Results:
    """Read-only access to a results folder."""

    def __init__(self, results_dir: str):
        self.dir = results_dir
        self._tables: dict = {}

    def _json(self, *parts: str):
        path = os.path.join(self.dir, *parts)
        return load_json(path) if os.path.isfile(path) else None

    def summary(self, name: str):
        return self._json(name, "summary.json")

    def downstream(self, name: str):
        return self._json(name, "downstream.json")

    def train_summary(self, name: str):
        return self._json(name, "train_summary.json")

    def training_data(self, filename: str):
        return self._json("training_data", filename)

    def entry(self, name: str, benchmark: str):
        """The summary of one benchmark for one model, or None."""
        summary = self.summary(name)
        return (summary or {}).get("benchmarks", {}).get(benchmark)

    def _read_items(self, folder: str, benchmark: str, seed: int):
        key = (folder, benchmark, seed)
        if key not in self._tables:
            suffix = f"_seed{seed}" if benchmark == "coco_order" else ""
            path = os.path.join(self.dir, folder, f"{benchmark}{suffix}_items.csv.gz")
            self._tables[key] = pd.read_csv(
                path, dtype={"item_id": str, "image_id": str, "group": str},
                keep_default_na=False) if os.path.isfile(path) else None
        return self._tables[key]

    def items(self, name: str, benchmark: str, seed: int = 0):
        """Per-item results of a model (one row per test item), or None."""
        return self._read_items(name, benchmark, seed)

    def blind(self, benchmark: str, seed: int = 0):
        """Per-item results of the text-only baselines, or None."""
        return self._read_items("text_only", benchmark, seed)

    def grid(self) -> dict:
        """Fine-tuned runs found on disk: {condition: {seed: folder name}}."""
        runs: dict = {}
        if not os.path.isdir(self.dir):
            return runs
        for name in sorted(os.listdir(self.dir)):
            match = re.fullmatch(r"(.+)_seed(\d+)", name)
            if match and match.group(1) in CONDITIONS and self.summary(name) is not None:
                runs.setdefault(match.group(1), {})[int(match.group(2))] = name
        return {c: dict(sorted(runs[c].items())) for c in CONDITIONS if c in runs}


def load_published(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------- small helpers

def first_seed(entry: dict) -> dict:
    """COCO-Order is evaluated with several shuffle seeds; this returns the first one."""
    return entry["per_seed"][0] if "per_seed" in entry else entry


def pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{100 * value:.{digits}f}"


def pct_ci(interval: dict, digits: int = 1) -> str:
    if not interval or interval.get("ci_low") is None or np.isnan(interval["ci_low"]):
        return ""
    return f"[{100 * interval['ci_low']:.{digits}f}, {100 * interval['ci_high']:.{digits}f}]"


def pct_with_ci(interval: dict, digits: int = 1) -> str:
    return f"{pct(interval['value'], digits)} {pct_ci(interval, digits)}".strip() if interval else ""


def mean_pm_std(values, digits: int = 1, scale: float = 100.0) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return ""
    if len(values) == 1:
        return f"{scale * values[0]:.{digits}f}"
    return f"{scale * np.mean(values):.{digits}f} ± {scale * np.std(values, ddof=1):.{digits}f}"


def signed_ci(diff: dict, digits: int = 1) -> str:
    return (f"{100 * diff['difference']:+.{digits}f} "
            f"[{100 * diff['ci_low']:+.{digits}f}, {100 * diff['ci_high']:+.{digits}f}]")


def aligned(tables: list) -> list:
    """Restrict per-item tables to their common items and put them in the same order."""
    common = set(tables[0]["item_id"])
    for table in tables[1:]:
        common &= set(table["item_id"])
    if not common:
        raise MissingInput("the result files have no test item in common")
    return [t[t["item_id"].isin(common)].sort_values("item_id").reset_index(drop=True) for t in tables]


def metric_parts(benchmark: str, table: pd.DataFrame) -> tuple:
    """(rows, groups, clusters) that define the benchmark's headline metric.

    VG-Relation drops the relations ARO calls symmetric and VG-Attribution drops
    attribute pairs with fewer than 25 cases; both then average over groups. SugarCrepe
    averages over its seven subsets. COCO-Order is a plain accuracy (groups = None).
    """
    if benchmark == "vg_relation":
        keep = aro.relation_groups(table).values
    elif benchmark == "vg_attribution":
        keep = aro.attribution_groups(table).values
    else:
        keep = np.ones(len(table), dtype=bool)
    groups = None if benchmark == "coco_order" else table["group"].values[keep]
    return keep, groups, table["image_id"].values[keep]


def headline_difference(benchmark: str, table: pd.DataFrame, correct_a, correct_b, n_boot: int) -> dict:
    """Headline metric of A minus that of B on the same items, with a paired bootstrap interval."""
    keep, groups, clusters = metric_parts(benchmark, table)
    return paired_difference(np.asarray(correct_a, dtype=float)[keep], np.asarray(correct_b, dtype=float)[keep],
                             clusters=clusters, groups=groups, n_boot=n_boot)


def seed_average(results: Results, runs: dict, benchmark: str):
    """(table, mean correctness over seeds) for one condition, or None if a file is missing."""
    tables = [results.items(name, benchmark) for name in runs.values()]
    if not tables or any(t is None for t in tables):
        return None
    tables = aligned(tables)
    return tables[0], np.mean([t["correct"].values for t in tables], axis=0)


def _need(value, what: str):
    if value is None:
        raise MissingInput(what)
    return value


# ----------------------------------------------------------------------------- reproduction

def table_reproduction_aro(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """ARO: published vs. measured, for CLIP and the released NegCLIP checkpoint."""
    rows = []
    for benchmark in ("vg_relation", "vg_attribution", "coco_order"):
        for name, key in (("clip", "clip"), ("negclip", "negclip"), ("negclip_gelu", "negclip")):
            entry = results.entry(name, benchmark)
            if entry is None:
                continue
            reference = published["aro"]["models"][key][benchmark]
            seeds = entry.get("headline_over_seeds", {})
            rows.append({
                "Benchmark": f"{BENCHMARK_LABELS[benchmark]} ({METRIC_LABELS[benchmark]})",
                "Model": REFERENCE_LABELS[name],
                "Published (%)": f"{reference:g}",
                "Source": published["aro"]["source"],
                "Measured (%)": pct(entry["headline"], 2),
                "95% CI": pct_ci(first_seed(entry)["headline_ci"], 2),
                "Std over shuffle seeds": pct(seeds.get("std"), 2) if seeds else "",
                "Measured − published": f"{100 * entry['headline'] - reference:+.1f}",
            })
        if benchmark == "coco_order":
            entry = results.entry("clip", benchmark)
            other = published["aro_remeasured_by_sugarcrepe"]
            if entry is not None:                             # a second published value for CLIP
                reference = other["models"]["clip"]["coco_order"]
                clip_row = next(r for r in rows if r["Model"] == REFERENCE_LABELS["clip"]
                                and r["Benchmark"].startswith(BENCHMARK_LABELS[benchmark]))
                rows.append({**clip_row, "Published (%)": f"{reference:g}", "Source": other["source"],
                             "Measured − published": f"{100 * entry['headline'] - reference:+.1f}"})
            continue
        for name in ("clip", "negclip", "negclip_gelu"):      # the per-item ("micro") average
            entry = results.entry(name, benchmark)
            if entry is None:
                continue
            reference = published["aro_remeasured_by_sugarcrepe"]["models"].get(name, {}).get(
                f"{benchmark}_micro")
            micro = entry["micro_accuracy"]
            rows.append({
                "Benchmark": f"{BENCHMARK_LABELS[benchmark]} (per-item accuracy)",
                "Model": REFERENCE_LABELS[name],
                "Published (%)": f"{reference:g}" if reference is not None else "",
                "Source": published["aro_remeasured_by_sugarcrepe"]["source"] if reference is not None else "",
                "Measured (%)": pct(micro["value"], 2),
                "95% CI": pct_ci(micro, 2),
                "Std over shuffle seeds": "",
                "Measured − published": f"{100 * micro['value'] - reference:+.1f}" if reference is not None else "",
            })
    if not rows:
        raise MissingInput("no summary.json for clip / negclip / negclip_gelu")
    return Table("reproduction_aro", "Reproduction on ARO: published vs. measured", pd.DataFrame(rows), [
        "Published and measured values are in separate columns; the published ones were run once by "
        "the authors and are given with two decimals.",
        "95% CI: cluster bootstrap over images. For COCO-Order the measured value is the mean over the "
        "shuffle seeds and the interval is for the first seed.",
        "NegCLIP was fine-tuned from OpenAI CLIP, which uses the QuickGELU activation. Building the "
        "model with GELU instead is what `open_clip.create_model_and_transforms('ViT-B-32', "
        "pretrained=<file>)` does in the released evaluation code.",
    ])


def table_reproduction_sugarcrepe(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """SugarCrepe: published vs. measured per subset, per category and overall."""
    entries = {name: results.entry(name, "sugarcrepe") for name in ("clip", "negclip", "negclip_gelu")}
    if all(e is None for e in entries.values()):
        raise MissingInput("no SugarCrepe results for clip / negclip / negclip_gelu")
    subsets = published["sugarcrepe_subsets"]
    order = subsets["order"]
    reference = {model: dict(zip(order, subsets["models"][model]["values"])) for model in ("clip", "negclip")}
    pooled = published["sugarcrepe_categories_pooled"]["models"]
    overall = published["sugarcrepe_mean_of_subsets"]["models"]

    def measured(name: str, getter) -> str:
        entry = entries[name]
        try:
            return pct_with_ci(getter(entry), 2) if entry is not None else ""
        except KeyError:
            return ""

    rows = []
    for subset in order:
        rows.append({"Subset": SUBSET_LABELS[subset],
                     "CLIP published": f"{reference['clip'][subset]:g}",
                     "CLIP measured": measured("clip", lambda e: e["subsets"][subset]),
                     "NegCLIP published": f"{reference['negclip'][subset]:g}",
                     "NegCLIP measured (QuickGELU)": measured("negclip", lambda e: e["subsets"][subset]),
                     "NegCLIP measured (GELU)": measured("negclip_gelu", lambda e: e["subsets"][subset])})
    for category in ("replace", "swap", "add"):
        rows.append({"Subset": f"{category.capitalize()} (all items pooled)",
                     "CLIP published": f"{pooled['clip'][category]:g}",
                     "CLIP measured": measured("clip", lambda e: e["categories"][category]["pooled"]),
                     "NegCLIP published": f"{pooled['negclip'][category]:g}",
                     "NegCLIP measured (QuickGELU)": measured(
                         "negclip", lambda e: e["categories"][category]["pooled"]),
                     "NegCLIP measured (GELU)": measured(
                         "negclip_gelu", lambda e: e["categories"][category]["pooled"])})
    rows.append({"Subset": "Mean of the 7 subsets",
                 "CLIP published": f"{overall['clip']['overall']:g}",
                 "CLIP measured": measured("clip", lambda e: e["overall_mean_of_subsets"]),
                 "NegCLIP published": f"{overall['negclip']['overall']:g}",
                 "NegCLIP measured (QuickGELU)": measured("negclip", lambda e: e["overall_mean_of_subsets"]),
                 "NegCLIP measured (GELU)": measured("negclip_gelu", lambda e: e["overall_mean_of_subsets"])})
    return Table("reproduction_sugarcrepe", "Reproduction on SugarCrepe: published vs. measured",
                 pd.DataFrame(rows), [
        f"Published per-subset values: CLIP from {subsets['models']['clip']['source']}; "
        f"NegCLIP from {subsets['models']['negclip']['source']}.",
        f"Published pooled category values: {published['sugarcrepe_categories_pooled']['source']}. "
        f"Published mean of subsets: {published['sugarcrepe_mean_of_subsets']['source']}.",
        "Measured values are followed by a 95% cluster-bootstrap interval (images resampled).",
        published["benchmark_sizes"]["note"],
    ])


# ----------------------------------------------------------------------------- controls

def _blind_entry(results: Results, benchmark: str):
    summary = results._json("text_only", "summary.json")
    entry = (summary or {}).get("benchmarks", {}).get(benchmark)
    if entry is None:
        return None
    return entry["per_seed"] if "per_seed" in entry else [entry]


def table_controls(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Chance, text-only baselines and the shuffled-image control next to the real models."""
    rows = []

    def add(label: str, kind: str, getter) -> None:
        row = {"Method": label, "Type": kind}
        found = False
        for benchmark in BENCHMARKS:
            value = getter(benchmark)
            row[BENCHMARK_LABELS[benchmark]] = "" if value is None else (
                f"{value:g}" if kind.startswith("published") else pct(value))
            found = found or value is not None
        if found:
            rows.append(row)

    def blind(scorer: str):
        def getter(benchmark: str):
            entries = _blind_entry(results, benchmark)
            return None if entries is None else float(np.mean([e[scorer]["headline"] for e in entries]))
        return getter

    def model(name: str, key: str):
        def getter(benchmark: str):
            entry = results.entry(name, benchmark)
            return None if entry is None else entry[key]
        return getter

    add("Chance", "by construction", lambda b: CHANCE[b])
    add("Text only: GPT-2, mean log-probability", "measured", blind("gpt2"))
    add("Text only: GPT-2, total log-probability", "measured", blind("gpt2sum"))
    add("Text only: grammar model (CoLA)", "measured", blind("grammar"))
    add(f"{REFERENCE_LABELS['clip']}, image replaced by another image", "measured",
        model("clip", "shuffled_image"))
    add(f"{REFERENCE_LABELS['negclip']}, image replaced by another image", "measured",
        model("negclip", "shuffled_image"))
    add(REFERENCE_LABELS["clip"], "measured", model("clip", "headline"))
    add(REFERENCE_LABELS["negclip"], "measured", model("negclip", "headline"))
    if len(rows) <= 3:
        raise MissingInput("no text-only or model results yet")
    return Table("controls", "Controls: what can be solved without looking at the image", pd.DataFrame(rows), [
        "All values are percentages on the benchmark's headline metric (" +
        "; ".join(f"{BENCHMARK_LABELS[b]}: {METRIC_LABELS[b]}" for b in BENCHMARKS) + ").",
        "A text-only model picks the caption a language model prefers and never sees the image.",
        "Shuffled image: the model scores the same captions against a different image, so only "
        "its text-side preference can produce a correct answer.",
        "Published text-only numbers use per-item accuracy and are compared like for like in the "
        "table 'reproduction_text_only'.",
    ])


def table_reproduction_text_only(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Text-only models on ARO: published (SugarCrepe, Table 8) vs. measured, per-item accuracy."""
    reference = published["text_only_on_aro"]
    rows, gaps = [], []
    for benchmark in ("vg_relation", "vg_attribution", "coco_order"):
        entries = _blind_entry(results, benchmark)
        if entries is None:
            continue
        key = "accuracy" if benchmark == "coco_order" else "micro_accuracy"

        def per_item(scorer: str) -> str:
            return pct(float(np.mean([e[scorer][key]["value"] for e in entries])), 2)

        measured = 100 * float(np.mean([e["grammar"][key]["value"] for e in entries]))
        if benchmark != "coco_order":
            gaps.append(abs(measured - reference["models"]["grammar"][benchmark]))
        rows.append({"Benchmark": f"{BENCHMARK_LABELS[benchmark]} (per-item accuracy)",
                     "Grammar model, published": f"{reference['models']['grammar'][benchmark]:g}",
                     "Grammar model, measured": per_item("grammar"),
                     "Vera, published (not re-run)": f"{reference['models']['vera'][benchmark]:g}",
                     "GPT-2 mean log-probability, measured": per_item("gpt2"),
                     "Headline metric, grammar model / GPT-2, measured":
                         f"{pct(_mean_headline(entries, 'grammar'))} / {pct(_mean_headline(entries, 'gpt2'))}"})
    if not rows:
        raise MissingInput("needs results/text_only")
    return Table("reproduction_text_only", "Text-only models on ARO: published vs. measured", pd.DataFrame(rows), [
        f"Published values: {reference['source']}. {reference['note']}",
        "The measured values in the first columns are per-item accuracies. "
        + ("On the two Visual Genome tasks they agree with the published grammar-model values to "
           f"within {max(gaps):.2f} points, which indicates that the published values are per-item "
           "accuracies as well. " if gaps and max(gaps) < 0.5 else
           "They differ from the published grammar-model values on the Visual Genome tasks. ")
        + "The last column gives the same measured models on ARO's own headline metric (macro "
          "accuracy for the two Visual Genome tasks).",
        "COCO-Order: the value published by the SugarCrepe authors differs from ours, as does their "
        "CLIP accuracy on this task (37.14, against 46 in ARO), so their version of the task is "
        "probably not identical to the released ARO code used here.",
    ])


def _mean_headline(entries: list, scorer: str) -> float:
    return float(np.mean([e[scorer]["headline"] for e in entries]))


def table_per_relation(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """VG-Relation broken down by relation (the most frequent ones)."""
    clip, negclip = results.items("clip", "vg_relation"), results.items("negclip", "vg_relation")
    if clip is None or negclip is None:
        raise MissingInput("needs per-item VG-Relation results of clip and negclip")
    blind = results.blind("vg_relation")
    tables = aligned([clip, negclip] + ([blind] if blind is not None else []))
    frame = pd.DataFrame({"relation": tables[0]["group"].values, "CLIP": tables[0]["correct"].values,
                          "NegCLIP": tables[1]["correct"].values})
    if blind is not None:
        frame["Text only: GPT-2"] = tables[2]["gpt2_correct"].values
    models = [c for c in frame.columns if c != "relation"]
    grouped = frame.groupby("relation")[models].mean()
    grouped["items"] = frame.groupby("relation").size()
    grouped = grouped.sort_values("items", ascending=False)
    rows = []
    for relation, row in grouped.head(12).iterrows():
        rows.append({"Relation": relation, "Items": int(row["items"]),
                     "Share of all items (%)": pct(row["items"] / len(frame)),
                     "In ARO's score": "no" if relation in aro.SYMMETRIC_RELATIONS else "yes",
                     **{model: pct(row[model]) for model in models}})
    counted = grouped[~grouped.index.isin(aro.SYMMETRIC_RELATIONS)]
    rows.append({"Relation": f"Mean over the {len(counted)} relations in ARO's score (macro accuracy)",
                 "Items": int(counted["items"].sum()), "Share of all items (%)": pct(counted["items"].sum() / len(frame)),
                 "In ARO's score": "", **{model: pct(counted[model].mean()) for model in models}})
    rows.append({"Relation": "All items (per-item accuracy)", "Items": len(frame), "Share of all items (%)": "100.0",
                 "In ARO's score": "", **{model: pct(frame[model].mean()) for model in models}})
    return Table("per_relation", "VG-Relation by relation: the twelve most frequent relations", pd.DataFrame(rows), [
        "Accuracy in %; chance is 50%. NegCLIP is the released checkpoint with QuickGELU.",
        "ARO's score gives every relation the same weight, whatever its number of items; the last two "
        "rows show how much this choice matters.",
    ])


def _slice_rows(benchmark: str, slice_name: str, mask: np.ndarray, base: pd.DataFrame,
                tables: dict, n_boot: int) -> dict:
    row = {"Benchmark": BENCHMARK_LABELS[benchmark], "Slice": slice_name,
           "Items": int(mask.sum()), "Share (%)": pct(mask.mean())}
    clusters = base["image_id"].values[mask]
    for name, table in tables.items():
        row[name] = pct_with_ci(accuracy_ci(table["correct"].values[mask], clusters, n_boot=n_boot))
    return row


def table_slices_text_only(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Accuracy on items a text-only model gets right vs. items it gets wrong."""
    rows = []
    for benchmark in BENCHMARKS:
        blind, clip, negclip = (results.blind(benchmark), results.items("clip", benchmark),
                                results.items("negclip", benchmark))
        if blind is None or clip is None or negclip is None:
            continue
        blind, clip, negclip = aligned([blind, clip, negclip])
        for scorer, label in (("gpt2", "GPT-2"), ("grammar", "grammar model")):
            solved = blind[f"{scorer}_correct"].values == 1
            for slice_name, mask in ((f"{label} right (text alone suffices)", solved),
                                     (f"{label} wrong (text alone misleads)", ~solved)):
                if mask.sum() == 0:
                    continue
                row = _slice_rows(benchmark, slice_name, mask, blind,
                                  {"CLIP": clip, "NegCLIP": negclip}, n_boot)
                diff = paired_difference(negclip["correct"].values[mask], clip["correct"].values[mask],
                                         clusters=blind["image_id"].values[mask], n_boot=n_boot)
                row["NegCLIP − CLIP"] = signed_ci(diff)
                rows.append(row)
    if not rows:
        raise MissingInput("needs results/text_only plus per-item results of clip and negclip")
    return Table("slices_text_only", "Where does NegCLIP gain? Items split by a text-only model",
                 pd.DataFrame(rows), [
        "Per-item accuracy (%) with a 95% cluster-bootstrap interval; NegCLIP is the released "
        "checkpoint with QuickGELU. COCO-Order uses the first shuffle seed.",
        "If NegCLIP's gain comes from reading the image, it should appear in both slices. If it comes "
        "from preferring fluent captions, it should be concentrated where the text-only model is right "
        "and turn negative where the text-only model is wrong.",
    ])


def table_slices_seen(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Accuracy on test items whose image was / was not in NegCLIP's released training file."""
    rows = []
    for benchmark in ("vg_relation", "vg_attribution", "sugarcrepe"):
        blind, clip, negclip = (results.blind(benchmark), results.items("clip", benchmark),
                                results.items("negclip", benchmark))
        if blind is None or clip is None or negclip is None or "image_in_negclip_train" not in blind:
            continue
        blind, clip, negclip = aligned([blind, clip, negclip])
        seen = blind["image_in_negclip_train"].values.astype(int) == 1
        for slice_name, mask in (("image in NegCLIP's training file", seen),
                                 ("image not in NegCLIP's training file", ~seen)):
            if mask.sum() == 0:
                continue
            row = _slice_rows(benchmark, slice_name, mask, blind, {"CLIP": clip, "NegCLIP": negclip}, n_boot)
            diff = paired_difference(negclip["correct"].values[mask], clip["correct"].values[mask],
                                     clusters=blind["image_id"].values[mask], n_boot=n_boot)
            row["NegCLIP − CLIP"] = signed_ci(diff)
            rows.append(row)
    if not rows:
        raise MissingInput("needs results/text_only (with overlap flags) and per-item results")
    return Table("slices_seen_images", "Train/test overlap of the released NegCLIP checkpoint",
                 pd.DataFrame(rows), [
        "Per-item accuracy (%) with a 95% cluster-bootstrap interval.",
        "The released NegCLIP was fine-tuned on the COCO Karpathy training split, which contains "
        "images from COCO val2014; SugarCrepe's images and part of Visual Genome come from there.",
    ])


# ----------------------------------------------------------------------------- training data

def table_training_negatives(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Properties of the three sets of training negatives."""
    stats = _need(results.training_data("stats.json"), "results/training_data/stats.json")
    rows = []
    everything = stats["all_candidates"]
    rows.append({"Negative set": "all candidate swaps (before selection)", "Pairs": everything["n_pairs"],
                 "Negatives per caption": "", "Text-only accuracy, GPT-2 (%)": pct(everything["blind_accuracy_gpt2"]),
                 "Text-only accuracy, grammar model (%)": "", "Median fluency gap": f"{everything['gap_median']:.2f}",
                 "Flagged meaning-preserving (%)": pct(everything.get("symmetric_share"))})
    labels = {"neg_swap": "swap (NegCLIP-style, one per swap type)", "neg_swap_nosym": "swap_nosym",
              "neg_fluent": "swap_fluent (fluency-matched)"}
    for key, label in labels.items():
        entry = stats["negative_sets"]["train"][key]
        rows.append({"Negative set": label, "Pairs": entry["n_pairs"],
                     "Negatives per caption": f"{entry.get('negatives_per_caption', 1.0):.2f}",
                     "Text-only accuracy, GPT-2 (%)": pct(entry["blind_accuracy_gpt2"]),
                     "Text-only accuracy, grammar model (%)": pct(entry["blind_accuracy_grammar"]),
                     "Median fluency gap": f"{entry['gap_median']:.2f}",
                     "Flagged meaning-preserving (%)": pct(entry.get("symmetric_share", 0.0))})
    notes = [
        f"Training pairs: {stats['selection']['train_pairs']}; validation pairs: "
        f"{stats['selection']['val_pairs']}; every condition trains on the same images and captions.",
        "Fluency gap = mean GPT-2 log-probability of the caption minus that of the negative "
        "(positive: the negative reads worse). Text-only accuracy = how often the language model "
        "alone prefers the true caption; 50% means no text-only signal for that model.",
        "The fluency-matched set is balanced with GPT-2, so its GPT-2 accuracy is 50% by construction; "
        "the grammar model was not used for the selection and is an independent check.",
        "Leak check (training images that also appear in a test set): "
        + ", ".join(f"{key.replace('overlap_with_', '').replace('_', ' ')}: {value}"
                    for key, value in stats["leak_checks"].items() if key != "leak_free") + ".",
    ]
    check = results.training_data("negclip_check.json")
    if check:
        notes.append(
            f"Check against the authors' released NegCLIP training file ({check['captions_sampled']} captions): "
            f"our swap rules reproduce {100 * check['reproduced_share']:.1f}% of their negatives; GPT-2 alone "
            f"separates their negatives from the true caption "
            f"{100 * check['gpt2_blind_accuracy_on_released_negatives']:.1f}% of the time.")
    return Table("training_negatives", "Training negatives: how much can a text-only model tell?",
                 pd.DataFrame(rows), notes)


# ----------------------------------------------------------------------------- fine-tuning grid

def _grid(results: Results) -> dict:
    grid = results.grid()
    if not grid:
        raise MissingInput("no fine-tuned runs (<condition>_seed<k>/summary.json) yet")
    return grid


def _column_values(results: Results, names: list, getter) -> list:
    values = []
    for name in names:
        try:
            values.append(getter(name))
        except (KeyError, TypeError):
            values.append(None)
    return values


def table_conditions(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Compositional benchmarks for every training condition (mean ± std over seeds)."""
    grid = _grid(results)

    def headline(benchmark: str):
        return lambda name: first_seed(results.entry(name, benchmark))["headline"]

    def category(which: str):
        return lambda name: results.entry(name, "sugarcrepe")["categories"][which]["pooled"]["value"]

    columns = [(f"{BENCHMARK_LABELS[b]}", headline(b)) for b in BENCHMARKS]
    columns += [(f"SugarCrepe {c} (pooled)", category(c)) for c in ("replace", "swap", "add")]
    rows = []
    for name in ("clip", "negclip"):
        if results.summary(name) is None:
            continue
        row = {"Model": f"{REFERENCE_LABELS[name]} [not trained here]", "Seeds": ""}
        for label, getter in columns:
            row[label] = mean_pm_std(_column_values(results, [name], getter))
        rows.append(row)
    for condition, runs in grid.items():
        row = {"Model": CONDITION_LABELS[condition], "Seeds": len(runs)}
        for label, getter in columns:
            row[label] = mean_pm_std(_column_values(results, list(runs.values()), getter))
        rows.append(row)
    return Table("conditions_compositional", "Fine-tuning conditions on the compositional benchmarks",
                 pd.DataFrame(rows), [
        "Mean ± standard deviation over training seeds, in %. All conditions use the same images, "
        "captions, steps, batch size and optimiser; only the negative caption differs.",
        "Headline metrics: " + "; ".join(f"{BENCHMARK_LABELS[b]}: {METRIC_LABELS[b]}" for b in BENCHMARKS)
        + ". COCO-Order uses the first shuffle seed for every model.",
        "The released NegCLIP row is not comparable in compute: it was trained on more pairs and with "
        "hard negative images.",
    ])


def table_condition_differences(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Paired differences between conditions on each benchmark."""
    grid = _grid(results)
    rows = []
    for a, b in COMPARISONS:
        if a not in grid or b not in grid:
            continue
        row = {"Comparison": f"{a} − {b}", "Seeds": f"{len(grid[a])} vs {len(grid[b])}"}
        for benchmark in BENCHMARKS:
            first, second = seed_average(results, grid[a], benchmark), seed_average(results, grid[b], benchmark)
            if first is None or second is None:
                row[BENCHMARK_LABELS[benchmark]] = ""
                continue
            table_a, table_b = aligned([first[0].assign(correct=first[1]), second[0].assign(correct=second[1])])
            diff = headline_difference(benchmark, table_a, table_a["correct"].values,
                                       table_b["correct"].values, n_boot)
            p_value = "p<0.001" if diff["p_value"] < 0.001 else f"p={diff['p_value']:.3f}"
            row[BENCHMARK_LABELS[benchmark]] = f"{signed_ci(diff)}, {p_value}"
        rows.append(row)
    if not rows:
        raise MissingInput("needs at least two fine-tuned conditions")
    return Table("conditions_differences", "Paired differences between fine-tuning conditions",
                 pd.DataFrame(rows), [
        "Difference of the headline metrics in percentage points, [95% interval], two-sided bootstrap "
        "p-value. Each item's correctness is first averaged over the training seeds; then images are "
        "resampled, with the same resampled images for both conditions.",
        "The interval covers the uncertainty from the finite test set. Seed-to-seed variation is "
        "shown by the standard deviations in the table of conditions.",
        "Several comparisons are tested; treat p-values near 0.05 with caution.",
    ])


def table_condition_slices(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Accuracy of each condition on items a text-only model gets right / wrong."""
    grid = _grid(results)
    rows = []
    for benchmark in BENCHMARKS:
        blind = results.blind(benchmark)
        averages = {c: seed_average(results, runs, benchmark) for c, runs in grid.items()}
        averages = {c: v for c, v in averages.items() if v is not None}
        if blind is None or not averages:
            continue
        tables = aligned([blind] + [table.assign(correct=correct) for table, correct in averages.values()])
        blind, per_condition = tables[0], dict(zip(averages, tables[1:]))
        solved = blind["gpt2_correct"].values == 1
        for slice_name, mask in (("GPT-2 right (text alone suffices)", solved),
                                 ("GPT-2 wrong (text alone misleads)", ~solved)):
            if mask.sum() == 0:
                continue
            row = {"Benchmark": BENCHMARK_LABELS[benchmark], "Slice": slice_name, "Items": int(mask.sum())}
            clusters = blind["image_id"].values[mask]
            for condition, table in per_condition.items():
                row[condition] = pct(float(table["correct"].values[mask].mean()))
            for a, b in (("swap", "none"), ("swap_fluent", "none"), ("swap_fluent", "swap")):
                if a in per_condition and b in per_condition:
                    diff = paired_difference(per_condition[a]["correct"].values[mask],
                                             per_condition[b]["correct"].values[mask],
                                             clusters=clusters, n_boot=n_boot)
                    row[f"{a} − {b}"] = signed_ci(diff)
            rows.append(row)
    if not rows:
        raise MissingInput("needs results/text_only and per-item results of the fine-tuned runs")
    return Table("conditions_slices", "Fine-tuning conditions on text-solvable and text-misleading items",
                 pd.DataFrame(rows), [
        "Per-item accuracy (%), averaged over training seeds. Differences are in percentage points "
        "with a 95% paired cluster-bootstrap interval.",
        "The research question is answered here: gains that come from visual binding should show "
        "up where the text-only model is wrong, not only where it is right.",
    ])


def _downstream_value(results: Results, name: str, path: tuple):
    node = results.downstream(name)
    for key in path:
        node = node[key]
    return node


def table_cost(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """What fine-tuning costs: zero-shot accuracy, retrieval and training resources."""
    grid = results.grid()
    datasets = []
    for name in ["clip", "negclip"] + [n for runs in grid.values() for n in runs.values()]:
        for dataset in ((results.downstream(name) or {}).get("zero_shot") or {}):
            if dataset not in datasets:
                datasets.append(dataset)
    quality = [(f"Zero-shot {d} top-1 (%)", ("zero_shot", d, "acc1"), 100.0, 1) for d in datasets]
    quality += [("COCO text R@1 (%)", ("coco_retrieval", "text_r1"), 100.0, 1),
                ("COCO image R@1 (%)", ("coco_retrieval", "image_r1"), 100.0, 1)]
    training = [("Training time (min)", "train_seconds", 1 / 60.0, 1), ("Seconds per step", "seconds_per_step", 1.0, 2),
                ("Peak GPU memory (GB)", "peak_gpu_memory_mb", 1 / 1024.0, 1), ("Texts per step", "texts_per_step", 1.0, 0)]

    def row_for(label: str, names: list, seeds) -> dict:
        row = {"Model": label, "Type": "measured", "Seeds": seeds}
        for column, path, scale, digits in quality:
            values = _column_values(results, names, lambda n: _downstream_value(results, n, path))
            row[column] = mean_pm_std(values, digits, scale)
        for column, key, scale, digits in training:
            values = _column_values(results, names, lambda n: results.train_summary(n)[key])
            row[column] = mean_pm_std(values, digits, scale)
        return row

    rows = []
    for name in ("clip", "negclip"):
        if results.downstream(name) is not None:
            rows.append(row_for(f"{REFERENCE_LABELS[name]} [not trained here]", [name], ""))
    for condition, runs in grid.items():
        if any(results.downstream(n) is not None or results.train_summary(n) is not None for n in runs.values()):
            rows.append(row_for(CONDITION_LABELS[condition], list(runs.values()), len(runs)))
    if not rows:
        raise MissingInput("no downstream.json or train_summary.json yet")
    for block, models in ((published["zero_shot_clip_benchmark"], ("clip", "negclip")),
                          (published["aro_downstream"], ("clip", "clip_ft", "negclip"))):
        for model in models:
            values = block["models"][model]
            row = {"Model": {"clip": "CLIP ViT-B/32", "clip_ft": "CLIP-FT", "negclip": "NegCLIP"}[model],
                   "Type": f"published ({block['source']})", "Seeds": ""}
            for column, path, _, _ in quality:
                key = path[1] if path[0] == "zero_shot" else f"coco_{path[1]}"
                row[column] = f"{values[key]:g}" if key in values else ""
            for column, _, _, _ in training:
                row[column] = ""
            rows.append(row)
    return Table("cost", "Cost of fine-tuning: zero-shot accuracy, retrieval and resources", pd.DataFrame(rows), [
        "Measured rows: mean ± standard deviation over training seeds. Zero-shot accuracy uses "
        "CLIP_benchmark's class names and prompt templates; retrieval is Recall@1 on the COCO Karpathy "
        "test split (5,000 images).",
        "Published rows are other people's measurements and are listed for reference only; ARO does "
        "not state its zero-shot protocol.",
        "'Texts per step' shows the extra text-encoder work of conditions with negative captions; the "
        "'random' condition has the same amount of work as the swap conditions.",
    ])


def table_validation(results: Results, published: dict, n_boot: int = 2000) -> Table:
    """Held-out validation accuracies before and after fine-tuning (never the test sets)."""
    grid = _grid(results)
    metrics = (("pair_acc_fluent", "True caption vs. fluency-matched negative (%)"),
               ("pair_acc_swap", "True caption vs. NegCLIP-style negative (%)"),
               ("pair_acc_swap_nosym", "True caption vs. swap_nosym negative (%)"),
               ("retrieval_acc", "Caption retrieval within a batch (%)"))
    rows = []
    for condition, runs in grid.items():
        histories = [(results.train_summary(n) or {}).get("validation") for n in runs.values()]
        histories = [h for h in histories if h]
        if not histories:
            continue
        for label, index in (("before training (epoch 0)", 0), ("after the last epoch", -1)):
            row = {"Condition": CONDITION_LABELS[condition], "When": label, "Seeds": len(histories)}
            for key, column in metrics:
                row[column] = mean_pm_std([h[index][key] for h in histories])
            rows.append(row)
    if not rows:
        raise MissingInput("no train_summary.json with validation results yet")
    return Table("validation", "Validation pairs (held out from training, not a test set)", pd.DataFrame(rows), [
        "Validation images come from COCO train2014 and are disjoint from the training images and "
        "from every test set. Use these numbers, not the test sets, for any choice of settings.",
    ])


ALL_TABLES = (table_reproduction_aro, table_reproduction_sugarcrepe, table_reproduction_text_only,
              table_controls, table_per_relation,
              table_slices_text_only, table_slices_seen, table_training_negatives,
              table_conditions, table_condition_differences, table_condition_slices,
              table_cost, table_validation)
