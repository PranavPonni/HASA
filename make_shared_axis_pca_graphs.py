#!/usr/bin/env python3
"""Regenerate before/after PCA plots with shared axes and cluster circles."""

import csv
import math
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis" / "wandb_ablation_graphs" / "shared_axis_pca"
ARCHIVE = ROOT / "analysis" / "shared_axis_pca_graphs.tar.gz"

BEFORE = {
    "title": "Before contrastive\nPlain FCN baseline",
    "variant": "selftouch_fcn",
    "run": "balmy-spaceship-8",
    "label": "before_plain_fcn_best",
}
AFTER = {
    "title": "After contrastive\nContrastive FCN: best latent cluster separation",
    "variant": "selftouch_contrastive_fcn",
    "run": "zesty-sound-7",
    "label": "after_contrastive_best_latent_separation",
}

COMBO_COLORS = {
    "thumb-index": "#1f77b4",
    "thumb-middle": "#ff7f0e",
    "index-middle": "#2ca02c",
    "middle-ring": "#d62728",
    "index-middle-ring": "#9467bd",
    "thumb-index-middle": "#8c564b",
}


def plot_dir(run):
    return ROOT / "model_weight" / run["variant"] / run["run"] / "plots"


def read_points(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["pc1"] = float(row["pc1"])
        row["pc2"] = float(row["pc2"])
    return rows


def read_final_latent_metrics(run):
    path = plot_dir(run) / "latent_combination_metrics.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    final = rows[-1]
    return {
        "separation_ratio": float(final["latent_combo_separation_ratio"]),
        "within_cluster_spread": float(final["latent_combo_within_spread"]),
        "nearest_separation": float(final["latent_combo_nearest_separation_ratio"]),
    }


def add_metrics_box(ax, metrics, prefix="", loc="lower left"):
    lines = []
    if prefix:
        lines.append(prefix)
    lines.extend(
        [
            "separation ratio: %.2f" % metrics["separation_ratio"],
            "within-cluster spread: %.2f" % metrics["within_cluster_spread"],
            "nearest separation: %.2f" % metrics["nearest_separation"],
        ]
    )
    if loc == "upper right":
        x, y = 0.98, 0.98
        va, ha = "top", "right"
    else:
        x, y = 0.02, 0.02
        va, ha = "bottom", "left"
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7.5,
        va=va,
        ha=ha,
        bbox={
            "facecolor": "white",
            "edgecolor": "#dddddd",
            "alpha": 0.88,
            "boxstyle": "round,pad=0.35",
        },
    )


def bounds_for(datasets, pad_fraction=0.08):
    xs = [row["pc1"] for rows in datasets for row in rows]
    ys = [row["pc2"] for rows in datasets for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * pad_fraction, 1.0)
    y_pad = max((y_max - y_min) * pad_fraction, 1.0)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups[key].append(row)
    return groups


def percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return sorted_values[low]
    weight = index - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def add_cluster_circle(ax, rows, color, alpha=0.15, linestyle="-"):
    if len(rows) < 2:
        return
    center_x = sum(row["pc1"] for row in rows) / len(rows)
    center_y = sum(row["pc2"] for row in rows) / len(rows)
    distances = [
        math.sqrt((row["pc1"] - center_x) ** 2 + (row["pc2"] - center_y) ** 2)
        for row in rows
    ]
    radius = max(percentile(distances, 0.90), 0.25)
    circle = Circle(
        (center_x, center_y),
        radius,
        facecolor=color,
        edgecolor=color,
        linewidth=1.2,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(circle)
    ax.scatter([center_x], [center_y], s=28, marker="+", color=color, linewidths=1.6, zorder=5)


def plot_latent_panel(ax, rows, title, metrics, xlim, ylim):
    for combo_name, group_rows in sorted(grouped(rows, ["combo_name"]).items()):
        combo = combo_name[0]
        color = COMBO_COLORS.get(combo, "#333333")
        add_cluster_circle(ax, group_rows, color, alpha=0.18)
        ax.scatter(
            [row["pc1"] for row in group_rows],
            [row["pc2"] for row in group_rows],
            s=16,
            color=color,
            alpha=0.75,
            edgecolors="none",
            label=combo,
        )
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Latent principal component 1")
    ax.set_ylabel("Latent principal component 2")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.45)
    ax.axvline(0, color="#777777", linewidth=0.7, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    add_metrics_box(ax, metrics)


def plot_combination_panel(ax, rows, title, metrics, xlim, ylim):
    marker_by_source = {"raw": "o", "predicted": "x"}
    alpha_by_source = {"raw": 0.72, "predicted": 0.52}
    line_by_source = {"raw": "-", "predicted": "--"}

    for key, group_rows in sorted(grouped(rows, ["combo_name", "source"]).items()):
        combo, source = key
        color = COMBO_COLORS.get(combo, "#333333")
        add_cluster_circle(
            ax,
            group_rows,
            color,
            alpha=0.11 if source == "raw" else 0.07,
            linestyle=line_by_source[source],
        )
        ax.scatter(
            [row["pc1"] for row in group_rows],
            [row["pc2"] for row in group_rows],
            s=15,
            color=color,
            marker=marker_by_source[source],
            alpha=alpha_by_source[source],
            linewidths=0.8,
            label="%s %s" % (combo, source),
        )

    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Tactile principal component 1")
    ax.set_ylabel("Tactile principal component 2")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.45)
    ax.axvline(0, color="#777777", linewidth=0.7, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    add_metrics_box(ax, metrics, prefix="encoder latent metrics", loc="upper right")


def combo_legend_handles():
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7,
            label=name,
        )
        for name, color in COMBO_COLORS.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="#333333", linestyle="none", label="raw points"),
            Line2D([0], [0], marker="x", color="#333333", linestyle="none", label="predicted points"),
            Line2D([0], [0], color="#333333", linewidth=6, alpha=0.18, label="90% cluster circle"),
        ]
    )
    return handles


def write_pair_plot(kind, before_rows, after_rows, before_metrics, after_metrics, xlim, ylim):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), dpi=170, sharex=True, sharey=True)
    if kind == "latent":
        plot_latent_panel(axes[0], before_rows, BEFORE["title"], before_metrics, xlim, ylim)
        plot_latent_panel(axes[1], after_rows, AFTER["title"], after_metrics, xlim, ylim)
        fig.suptitle("Best Latent PCA Before vs After Contrastive Training", fontsize=15)
        dst = OUT_DIR / "before_after_latent_pca_shared_axes_circles.png"
    else:
        plot_combination_panel(axes[0], before_rows, BEFORE["title"], before_metrics, xlim, ylim)
        plot_combination_panel(axes[1], after_rows, AFTER["title"], after_metrics, xlim, ylim)
        fig.suptitle("Tactile Output PCA Before vs After Contrastive Training", fontsize=15)
        dst = OUT_DIR / "before_after_combination_pca_shared_axes_circles.png"

    fig.legend(
        handles=combo_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(dst)
    plt.close(fig)
    return dst


def write_contact_sheet(
    combo_before,
    combo_after,
    latent_before,
    latent_after,
    before_metrics,
    after_metrics,
    combo_xlim,
    combo_ylim,
    latent_xlim,
    latent_ylim,
):
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 11.0), dpi=170)
    plot_combination_panel(axes[0, 0], combo_before, BEFORE["title"], before_metrics, combo_xlim, combo_ylim)
    plot_combination_panel(axes[0, 1], combo_after, AFTER["title"], after_metrics, combo_xlim, combo_ylim)
    plot_latent_panel(axes[1, 0], latent_before, BEFORE["title"], before_metrics, latent_xlim, latent_ylim)
    plot_latent_panel(axes[1, 1], latent_after, AFTER["title"], after_metrics, latent_xlim, latent_ylim)
    axes[0, 0].set_ylabel("Combination PCA PC2")
    axes[1, 0].set_ylabel("Latent PCA PC2")
    fig.suptitle(
        "Before vs After Contrastive PCA: after run selected for strongest latent cluster separation",
        fontsize=15,
    )
    fig.legend(
        handles=combo_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout(rect=(0, 0.075, 1, 0.95))
    dst = OUT_DIR / "before_after_pca_shared_axes_circles_contact_sheet.png"
    fig.savefig(dst)
    plt.close(fig)
    return dst


def write_readme(outputs):
    lines = [
        "# Shared-Axis Before/After PCA Graphs",
        "",
        "These plots redraw the PCA CSV exports with fixed before/after axis limits and translucent 90% cluster circles.",
        "The after-contrastive run is the contrastive FCN with the strongest latent cluster separation, so it is the fairest PCA plot for showing the representation effect.",
        "",
        "The axes are shared within each PCA family:",
        "- combination PCA before vs after uses one shared x/y range.",
        "- encoder latent PCA before vs after uses one shared x/y range.",
        "- every panel includes separation ratio, within-cluster spread, and nearest separation for the run shown.",
        "",
        "The combination PCA and encoder latent PCA are not put on one single global scale because they are different coordinate spaces.",
        "",
        "## Outputs",
        "",
    ]
    for output in outputs:
        lines.append("- `%s`" % output.relative_to(ROOT))
    lines.extend(
        [
            "",
            "Before run: `%s / %s`, plain FCN baseline." % (BEFORE["variant"], BEFORE["run"]),
            "After run: `%s / %s`, best latent cluster separation among contrastive FCN runs." % (AFTER["variant"], AFTER["run"]),
            "",
        ]
    )
    path = OUT_DIR / "README.md"
    path.write_text("\n".join(lines))
    return path


def make_archive():
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(OUT_DIR, arcname=OUT_DIR.name)
    return ARCHIVE


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combo_before = read_points(plot_dir(BEFORE) / "combination_pca_epoch_0500.csv")
    combo_after = read_points(plot_dir(AFTER) / "combination_pca_epoch_0500.csv")
    latent_before = read_points(plot_dir(BEFORE) / "latent_combination_pca_epoch_0500.csv")
    latent_after = read_points(plot_dir(AFTER) / "latent_combination_pca_epoch_0500.csv")
    before_metrics = read_final_latent_metrics(BEFORE)
    after_metrics = read_final_latent_metrics(AFTER)

    combo_xlim, combo_ylim = bounds_for([combo_before, combo_after])
    latent_xlim, latent_ylim = bounds_for([latent_before, latent_after])

    outputs = [
        write_pair_plot(
            "combination",
            combo_before,
            combo_after,
            before_metrics,
            after_metrics,
            combo_xlim,
            combo_ylim,
        ),
        write_pair_plot(
            "latent",
            latent_before,
            latent_after,
            before_metrics,
            after_metrics,
            latent_xlim,
            latent_ylim,
        ),
        write_contact_sheet(
            combo_before,
            combo_after,
            latent_before,
            latent_after,
            before_metrics,
            after_metrics,
            combo_xlim,
            combo_ylim,
            latent_xlim,
            latent_ylim,
        ),
    ]
    outputs.append(write_readme(outputs))
    outputs.append(make_archive())

    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
