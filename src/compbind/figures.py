"""Figures for the report, drawn from the saved result files with matplotlib.

Visual conventions (kept the same in every figure):
* hue = the kind of negative caption a model was trained with
    blue    fluency-matched swaps (swap_fluent, the improvement)
    orange  rule-based swaps (swap, swap_nosym, the released NegCLIP)
    aqua    no hard negatives (none = CLIP-FT, random)
    gray    the pretrained CLIP starting point
  The three hues were checked for colour-vision deficiencies (all pairs distinct);
* marker shape = the variant inside a family: circle = main condition, square =
  control / ablation, diamond = a checkpoint that was not trained in this project;
* every row or line is also labelled in text, so colour never carries identity alone;
* published numbers are drawn as open diamonds, measured numbers as filled dots.
Each figure has a table twin written by scripts/06_make_tables.py.
"""
from __future__ import annotations

import numpy as np

from . import report
from .benchmarks import BENCHMARKS, CHANCE
from .report import BENCHMARK_LABELS, MissingInput, Results, first_seed

SURFACE, INK, INK_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# model or condition -> (label, colour, marker)
STYLE = {
    "clip": ("CLIP, pretrained", MUTED, "D"),
    "negclip": ("NegCLIP, released", ORANGE, "D"),
    "none": ("none (CLIP-FT)", AQUA, "o"),
    "random": ("random caption", AQUA, "s"),
    "swap": ("swap (NegCLIP-style)", ORANGE, "o"),
    "swap_nosym": ("swap_nosym", ORANGE, "s"),
    "swap_fluent": ("swap_fluent (fluency-matched)", BLUE, "o"),
}
MARKER = dict(markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle="")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "legend.frameon": False,
    })
    return plt


def _style(ax, grid: str = "x") -> None:
    """Recessive axes: no box, hairline solid grid behind the marks, no tick marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left" if grid == "x" else "bottom"].set_visible(grid != "x")
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _title(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.985 - 0.26 / fig.get_figheight(), subtitle, ha="left", va="top",
                 fontsize=8.5, color=INK_SECONDARY)


def _legend_handle(label: str, color: str, marker: str = "o", filled: bool = True, line: bool = False):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color if line else "none", linewidth=2 if line else 0, marker=marker,
                  markersize=7, markerfacecolor=color if filled else SURFACE,
                  markeredgecolor=SURFACE if filled else color, markeredgewidth=1.4, label=label)


def _model_rows(results: Results) -> list:
    """[(key, [result folders])]: reference checkpoints first, then the fine-tuned conditions."""
    rows = [(name, [name]) for name in ("clip", "negclip") if results.summary(name) is not None]
    rows += [(condition, list(runs.values())) for condition, runs in results.grid().items()]
    return rows


def _save(fig, path_without_extension: str) -> list:
    paths = []
    for extension in ("png", "pdf"):
        path = f"{path_without_extension}.{extension}"
        fig.savefig(path, dpi=200)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------- figure 1

def figure_reproduction(results: Results, published: dict):
    """Published vs. measured accuracy of CLIP and the released NegCLIP on four benchmarks."""
    plt = _pyplot()
    variants = (("clip", "clip", "CLIP"), ("negclip", "negclip", "NegCLIP, QuickGELU (as trained)"),
                ("negclip_gelu", "negclip", "NegCLIP, GELU (as released code loads it)"))
    rows, y = [], 0.0
    for benchmark in BENCHMARKS:
        group = []
        for name, key, label in variants:
            entry = results.entry(name, benchmark)
            if entry is None:
                continue
            if benchmark == "sugarcrepe":
                reference = published["sugarcrepe_mean_of_subsets"]["models"][key]["overall"]
            else:
                reference = published["aro"]["models"][key][benchmark]
            interval = first_seed(entry)["headline_ci"]
            group.append((label, reference, 100 * entry["headline"], 100 * interval["ci_low"],
                          100 * interval["ci_high"]))
        if group:
            rows.append((benchmark, y, group))
            y += len(group) + 1.3
    if not rows:
        raise MissingInput("no results for clip / negclip / negclip_gelu")

    fig, ax = plt.subplots(figsize=(7.4, 0.9 + 0.27 * y))
    ticks, labels = [], []
    for benchmark, top, group in rows:
        ax.text(28.5, top - 0.75, f"{BENCHMARK_LABELS[benchmark]}  ({report.METRIC_LABELS[benchmark]})",
                fontsize=8.5, fontweight="bold", color=INK, va="center")
        for index, (label, reference, measured, low, high) in enumerate(group):
            position = top + index
            ax.plot([low, high], [position, position], color=INK, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(reference, position, marker="D", markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY,
                    markeredgewidth=1.3, markersize=6.5, linestyle="", zorder=3)
            ax.plot(measured, position, marker="o", markerfacecolor=INK, zorder=4, **MARKER)
            ax.text(101.5, position, f"{round(measured - reference, 1) + 0.0:+.1f}", va="center", ha="left",
                    fontsize=8, color=INK_SECONDARY)
            ticks.append(position)
            labels.append(label)
    ax.text(101.5, -0.75, "measured −\npublished", va="center", ha="left", fontsize=7.5, color=MUTED)
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 1.0, -1.5)
    ax.set_xlim(28, 101)
    ax.set_xlabel("accuracy (%)")
    _style(ax, "x")
    ax.legend(handles=[_legend_handle("published (one run, rounded)", INK_SECONDARY, "D", filled=False),
                       _legend_handle("measured here, with 95% interval", INK, "o")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8, handletextpad=0.3)
    _title(fig, "Reproduction: published and measured accuracy")
    fig.subplots_adjust(left=0.36, right=0.9, top=1 - 0.75 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
    return fig


# ----------------------------------------------------------------------------- dot matrices

def _dot_matrix(plt, row_labels: list, panels: list, title: str, subtitle: str, legend=None,
                shared_range=None, width: float = 7.4):
    """Rows of dots in several panels that share the row labels.

    `panels` is a list of dicts: {"title", "draw": function(ax, row_index), "reference": x or None,
    "reference_label": str}. With `shared_range` every panel uses that x-range.
    """
    n_rows = len(row_labels)
    top, bottom = 0.95, (0.66 if legend else 0.36)                    # margins in inches
    left = 0.25 + 0.064 * max(len(label) for label in row_labels)
    height = top + bottom + 0.3 * n_rows + 0.25
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels):
        for row in range(n_rows):
            panel["draw"](ax, row)
        if panel.get("reference") is not None:
            ax.axvline(panel["reference"], color=AXIS, linewidth=1.0, zorder=1)
            ax.text(panel["reference"], -0.72, panel.get("reference_label", ""), ha="center", va="bottom",
                    fontsize=7, color=MUTED)
        ax.set_title(panel["title"], fontsize=8.5, color=INK, loc="left", pad=14)
        if shared_range is not None:
            ax.set_xlim(*shared_range)
            ticks = np.linspace(shared_range[0], shared_range[1], 5)
            ax.set_xticks(ticks, [f"{t:g}" if i % 2 == 0 else "" for i, t in enumerate(ticks)])
            labels = ax.get_xticklabels()
            labels[0].set_ha("left")                                  # keep the end labels inside the panel
            labels[-1].set_ha("right")
        else:
            ax.margins(x=0.18)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_ylim(n_rows - 0.4, -0.6)
        _style(ax, "x")
    axes[0].set_yticks(range(n_rows), row_labels)
    if legend:
        fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=len(legend),
                   fontsize=8, handletextpad=0.3, columnspacing=1.4)
    _title(fig, title, subtitle)
    fig.subplots_adjust(left=left / width, right=0.985, top=1 - top / height, bottom=bottom / height,
                        wspace=0.12 if shared_range is not None else 0.2)
    return fig


def figure_controls(results: Results, published: dict):
    """What can be answered without the image: text-only models and the wrong-image control."""
    plt = _pyplot()
    blind = {b: report._blind_entry(results, b) for b in BENCHMARKS}
    if all(v is None for v in blind.values()) or results.summary("clip") is None:
        raise MissingInput("needs results/text_only and the clip results")
    rows = [("No image: GPT-2 picks the caption", INK, "o", lambda b: _blind_mean(blind[b], "gpt2")),
            ("No image: grammar model picks", INK, "s", lambda b: _blind_mean(blind[b], "grammar"))]
    for name in ("clip", "negclip"):
        if results.summary(name) is None:
            continue
        label, color, marker = STYLE[name]
        rows.append((f"Wrong image: {label}", color, marker,
                     lambda b, n=name: _entry_value(results, n, b, "shuffled_image")))
    for name in ("clip", "negclip"):
        if results.summary(name) is None:
            continue
        label, color, marker = STYLE[name]
        rows.append((f"Right image: {label}", color, marker,
                     lambda b, n=name: _entry_value(results, n, b, "headline")))

    def panel(benchmark: str) -> dict:
        def draw(ax, row: int) -> None:
            _, color, marker, getter = rows[row]
            value = getter(benchmark)
            if value is not None:
                ax.plot(100 * value, row, marker=marker, markerfacecolor=color, zorder=3, **MARKER)
        return {"title": BENCHMARK_LABELS[benchmark], "draw": draw, "reference": 100 * CHANCE[benchmark],
                "reference_label": "chance"}

    return _dot_matrix(plt, [r[0] for r in rows], [panel(b) for b in BENCHMARKS],
                       "How much of each benchmark can be solved without the right image?",
                       "Accuracy (%) on each benchmark's headline metric. A text-only model never sees an image.",
                       shared_range=(0, 100))


def _blind_mean(entries, scorer: str):
    return None if entries is None else float(np.mean([e[scorer]["headline"] for e in entries]))


def _entry_value(results: Results, name: str, benchmark: str, key: str):
    entry = results.entry(name, benchmark)
    return None if entry is None else entry[key]


def figure_conditions(results: Results, published: dict):
    """Every model on every metric: compositional benchmarks and the downstream cost."""
    plt = _pyplot()
    rows = _model_rows(results)
    if not results.grid():
        raise MissingInput("no fine-tuned runs yet")
    metrics = [(BENCHMARK_LABELS[b], lambda n, b=b: first_seed(results.entry(n, b))["headline"]) for b in BENCHMARKS]
    datasets = sorted({d for _, names in rows for n in names
                       for d in ((results.downstream(n) or {}).get("zero_shot") or {})})
    for dataset in datasets[-1:]:      # one zero-shot panel (the last name alphabetically, e.g. cifar100)
        metrics.append((f"Zero-shot {dataset}", lambda n, d=dataset: results.downstream(n)["zero_shot"][d]["acc1"]))
    if any((results.downstream(n) or {}).get("coco_retrieval") for _, names in rows for n in names):
        metrics.append(("COCO text R@1", lambda n: results.downstream(n)["coco_retrieval"]["text_r1"]))

    def panel(label: str, getter) -> dict:
        values = [report._column_values(results, names, getter) for _, names in rows]

        def draw(ax, row: int) -> None:
            _, color, marker = STYLE[rows[row][0]]
            seeds = [100 * v for v in values[row] if v is not None]
            if not seeds:
                return
            if len(seeds) > 1:
                ax.plot(seeds, [row] * len(seeds), marker="o", markersize=3.5, markerfacecolor=color,
                        markeredgecolor="none", alpha=0.45, linestyle="", zorder=2)
            ax.plot(np.mean(seeds), row, marker=marker, markerfacecolor=color, zorder=3, **MARKER)

        clip = [100 * v for (key, _), row in zip(rows, values) if key == "clip" for v in row if v is not None]
        return {"title": label, "draw": draw, "reference": clip[0] if clip else None, "reference_label": "CLIP"}

    return _dot_matrix(plt, [STYLE[key][0] for key, _ in rows], [panel(label, getter) for label, getter in metrics],
                       "Fine-tuning conditions: compositional benchmarks and downstream cost",
                       "Large marker: mean over training seeds; small dots: single seeds. All values in %. "
                       "Each panel has its own scale.", width=9.6)


def figure_slices(results: Results, published: dict):
    """Accuracy on items a text-only model gets right vs. items it gets wrong, per model."""
    plt = _pyplot()
    rows = _model_rows(results)
    data = {}
    for benchmark in BENCHMARKS:
        blind = results.blind(benchmark)
        if blind is None:
            continue
        for key, names in rows:
            average = report.seed_average(results, dict(enumerate(names)), benchmark)
            if average is None:
                continue
            table, correct = average
            blind_aligned, table_aligned = report.aligned([blind, table.assign(correct=correct)])
            solved = blind_aligned["gpt2_correct"].values == 1
            values = table_aligned["correct"].values
            if solved.any() and (~solved).any():
                data[(benchmark, key)] = (100 * values[solved].mean(), 100 * values[~solved].mean())
    if not data:
        raise MissingInput("needs results/text_only and per-item results")

    def panel(benchmark: str) -> dict:
        def draw(ax, row: int) -> None:
            key = rows[row][0]
            if (benchmark, key) not in data:
                return
            _, color, marker = STYLE[key]
            right, wrong = data[(benchmark, key)]
            ax.plot([wrong, right], [row, row], color=AXIS, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(right, row, marker=marker, markerfacecolor=color, zorder=4, **MARKER)
            ax.plot(wrong, row, marker=marker, markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=1.6, markersize=6.5, linestyle="", zorder=3)
        return {"title": BENCHMARK_LABELS[benchmark], "draw": draw, "reference": 100 * CHANCE[benchmark],
                "reference_label": "chance"}

    legend = [_legend_handle("items GPT-2 gets right without the image", INK_SECONDARY, "o"),
              _legend_handle("items GPT-2 gets wrong without the image", INK_SECONDARY, "o", filled=False)]
    return _dot_matrix(plt, [STYLE[key][0] for key, _ in rows], [panel(b) for b in BENCHMARKS if results.blind(b) is not None],
                       "Does a model's accuracy depend on whether the text alone gives the answer away?",
                       "Per-item accuracy (%). A model that reads the image should score the same on both kinds of items.",
                       legend=legend, shared_range=(0, 100))


# ----------------------------------------------------------------------------- distributions and curves

def figure_fluency_gap(results: Results, published: dict):
    """Distribution of the fluency gap for the NegCLIP-style and the fluency-matched negatives."""
    plt = _pyplot()
    histograms = results.training_data("gap_histograms.json")
    stats = results.training_data("stats.json")
    if histograms is None or stats is None:
        raise MissingInput("results/training_data/gap_histograms.json and stats.json")
    edges = np.asarray(histograms["edges"])
    centres = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    series = (("all_candidates", "all possible swaps", MUTED, None),
              ("neg_swap", "swap (NegCLIP-style)", ORANGE, "neg_swap"),
              ("neg_fluent", "swap_fluent (fluency-matched)", BLUE, "neg_fluent"))
    handles = []
    for key, label, color, stats_key in series:
        if key not in histograms["sets"]:
            continue
        counts = np.asarray(histograms["sets"][key]["counts"], dtype=float)
        share = 100 * counts / max(counts.sum(), 1)
        ax.fill_between(centres, share, color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(centres, share, color=color, linewidth=2, solid_joinstyle="round", zorder=3)
        accuracy = (stats["all_candidates"] if stats_key is None
                    else stats["negative_sets"]["train"][stats_key])["blind_accuracy_gpt2"]
        handles.append(_legend_handle(f"{label}: GPT-2 alone picks the true caption {100 * accuracy:.0f}% of the time",
                                      color, "", line=True))
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    top = ax.get_ylim()[1]
    ax.text(0.04, top * 0.97, "negative reads worse\nthan the caption →", ha="left", va="top", fontsize=7.5, color=MUTED)
    ax.text(-0.04, top * 0.97, "← negative reads\nbetter", ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(-1.5, 3.0)
    ax.set_ylim(0, top)
    ax.set_xlabel("fluency gap: GPT-2 mean log-probability of the caption minus that of its negative (nats per token)")
    ax.set_ylabel("share of pairs (%)")
    _style(ax, "y")
    ax.legend(handles=handles, loc="upper right", fontsize=8, handlelength=1.6)
    _title(fig, "Rule-based swaps can be told from the true caption by fluency alone")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.15)
    return fig


def figure_validation(results: Results, published: dict):
    """Held-out validation accuracy per epoch for each training condition."""
    plt = _pyplot()
    grid = results.grid()
    curves = {}
    for condition, runs in grid.items():
        histories = [(results.train_summary(n) or {}).get("validation") for n in runs.values()]
        histories = [h for h in histories if h]
        if histories:
            length = min(len(h) for h in histories)
            curves[condition] = {key: 100 * np.mean([[e[key] for e in h[:length]] for h in histories], axis=0)
                                 for key in ("pair_acc_fluent", "pair_acc_swap", "retrieval_acc")}
    if not curves:
        raise MissingInput("no train_summary.json with validation results yet")
    panels = (("pair_acc_fluent", "True caption vs. fluency-matched negative"),
              ("pair_acc_swap", "True caption vs. NegCLIP-style negative"),
              ("retrieval_acc", "Caption retrieval within a batch"))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    for ax, (key, label) in zip(axes, panels):
        for condition, values in curves.items():
            _, color, marker = STYLE[condition]
            ax.plot(range(len(values[key])), values[key], color=color, linewidth=2, marker=marker,
                    markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2, solid_joinstyle="round")
        ax.set_title(label, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("epoch (0 = before fine-tuning)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        _style(ax, "y")
    axes[0].set_ylabel("validation accuracy (%)")
    fig.legend(handles=[_legend_handle(STYLE[c][0], STYLE[c][1], STYLE[c][2], line=True) for c in curves],
               loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=len(curves), fontsize=8, columnspacing=1.4)
    _title(fig, "Validation pairs during fine-tuning (held out from training; not a test set)",
           "Mean over training seeds. Each panel has its own scale.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.27, wspace=0.22)
    return fig


ALL_FIGURES = (("fig1_reproduction", figure_reproduction), ("fig2_controls", figure_controls),
               ("fig3_fluency_gap", figure_fluency_gap), ("fig4_conditions", figure_conditions),
               ("fig5_text_slices", figure_slices), ("fig6_validation", figure_validation))


def make_all(results: Results, published: dict, out_dir: str) -> tuple:
    """Draw every figure whose inputs exist. Returns (written paths, [(name, reason skipped)])."""
    import os

    plt = _pyplot()
    written, skipped = [], []
    for name, builder in ALL_FIGURES:
        try:
            fig = builder(results, published)
        except MissingInput as missing:
            skipped.append((name, str(missing)))
            continue
        written += _save(fig, os.path.join(out_dir, name))
        plt.close(fig)
    return written, skipped
