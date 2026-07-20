#!/usr/bin/env python3
"""Create a complete comparison report for the 14 self-touch FCN variants."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


VARIANTS = [
    "selftouch_fcn_pos",
    "selftouch_fcn_vel",
    "selftouch_fcn_trq",
    "selftouch_fcn_cmd",
    "selftouch_fcn_posvel",
    "selftouch_fcn_postrq",
    "selftouch_fcn_poscmd",
    "selftouch_fcn_velcmd",
    "selftouch_fcn_veltrq",
    "selftouch_fcn_trqcmd",
    "selftouch_fcn_posveltrq",
    "selftouch_fcn_postrqcmd",
    "selftouch_fcn_poscmdvel",
    "selftouch_fcn_posveltrqcmd",
]
FINGERS = ("index", "thumb", "middle", "ring")


def number(value, default=math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_history(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if math.isfinite(number(row.get("prediction_mae")))]


def choose_completed_run(root: Path, variant: str, minimum_epoch: int) -> tuple[Path, List[Dict[str, str]]]:
    candidates = []
    for path in (root / variant).glob("*/plots/raw_prediction_metrics.csv"):
        rows = read_history(path)
        if not rows:
            continue
        max_epoch = max(int(number(row.get("epoch"), 0)) for row in rows)
        if max_epoch >= minimum_epoch:
            candidates.append((path.stat().st_mtime, max_epoch, path, rows))
    if not candidates:
        raise FileNotFoundError(f"No completed >= {minimum_epoch}-epoch metric history for {variant}")
    _, _, path, rows = max(candidates, key=lambda item: (item[0], item[1]))
    rows.sort(key=lambda row: number(row.get("epoch")))
    return path, rows


def metric(row: Dict[str, str], name: str) -> float:
    return number(row.get(name))


def first_epoch_within(rows: List[Dict[str, str]], best_mae: float, tolerance: float) -> int:
    threshold = best_mae * (1.0 + tolerance)
    for row in rows:
        if metric(row, "prediction_mae") <= threshold:
            return int(metric(row, "epoch"))
    return int(metric(rows[-1], "epoch"))


def overfit_label(degradation_pct: float, late_trend_pct_per_100: float) -> str:
    if degradation_pct > 5.0 or late_trend_pct_per_100 > 5.0:
        return "high"
    if degradation_pct > 2.0 or late_trend_pct_per_100 > 2.0:
        return "watch"
    return "low"


def summarize(variant: str, path: Path, rows: List[Dict[str, str]]) -> tuple[Dict, List[Dict]]:
    maes = np.asarray([metric(row, "prediction_mae") for row in rows], dtype=np.float64)
    epochs = np.asarray([metric(row, "epoch") for row in rows], dtype=np.float64)
    best_index = int(np.nanargmin(maes))
    best = rows[best_index]
    final = rows[-1]
    initial = rows[0]
    best_mae = float(maes[best_index])
    final_mae = metric(final, "prediction_mae")
    initial_mae = metric(initial, "prediction_mae")
    degradation_pct = 100.0 * (final_mae - best_mae) / max(best_mae, 1e-12)
    improvement_pct = 100.0 * (initial_mae - best_mae) / max(initial_mae, 1e-12)
    late_count = min(5, len(rows))
    late_epochs = epochs[-late_count:]
    late_maes = maes[-late_count:]
    slope = float(np.polyfit(late_epochs, late_maes, 1)[0]) if late_count >= 2 else 0.0
    late_trend_pct_per_100 = slope * 10000.0 / max(best_mae, 1e-12)
    late_cv_pct = 100.0 * float(np.std(late_maes)) / max(float(np.mean(late_maes)), 1e-12)
    run_dir = path.parents[1]
    checkpoint_epochs = []
    for checkpoint in run_dir.glob("epoch*.pth"):
        try:
            checkpoint_epochs.append((int(checkpoint.stem.removeprefix("epoch")) + 1, checkpoint))
        except ValueError:
            continue
    exact_checkpoint = next(
        (checkpoint for epoch, checkpoint in checkpoint_epochs if epoch == int(metric(best, "epoch"))),
        None,
    )
    checkpoint_by_epoch = dict(checkpoint_epochs)
    evaluated_checkpoints = [
        (int(metric(row, "epoch")), metric(row, "prediction_mae"))
        for row in rows
        if int(metric(row, "epoch")) in checkpoint_by_epoch
    ]
    recommended_epoch, _ = min(
        evaluated_checkpoints,
        key=lambda item: item[1],
    ) if evaluated_checkpoints else (0, math.inf)
    recommended_checkpoint = checkpoint_by_epoch.get(recommended_epoch)

    summary = {
        "variant": variant,
        "run": path.parents[1].name,
        "history_rows": len(rows),
        "initial_epoch": int(metric(initial, "epoch")),
        "initial_mae": initial_mae,
        "best_epoch": int(metric(best, "epoch")),
        "best_mae": best_mae,
        "best_accuracy": metric(best, "prediction_raw_accuracy"),
        "best_profile_accuracy": metric(best, "prediction_profile_accuracy"),
        "best_rmse": metric(best, "prediction_rmse"),
        "best_corr": metric(best, "prediction_corr"),
        "best_r2": metric(best, "prediction_r2"),
        "best_p95": metric(best, "prediction_error_p95"),
        "best_abs_bias": metric(best, "prediction_abs_bias"),
        "best_checkpoint_saved": bool(exact_checkpoint),
        "recommended_checkpoint_epoch": recommended_epoch,
        "recommended_checkpoint": str(exact_checkpoint or recommended_checkpoint or ""),
        "final_epoch": int(metric(final, "epoch")),
        "final_mae": final_mae,
        "final_accuracy": metric(final, "prediction_raw_accuracy"),
        "final_profile_accuracy": metric(final, "prediction_profile_accuracy"),
        "final_corr": metric(final, "prediction_corr"),
        "final_r2": metric(final, "prediction_r2"),
        "improvement_pct": improvement_pct,
        "epoch_within_10pct_best": first_epoch_within(rows, best_mae, 0.10),
        "epoch_within_5pct_best": first_epoch_within(rows, best_mae, 0.05),
        "epoch_within_2pct_best": first_epoch_within(rows, best_mae, 0.02),
        "final_degradation_pct": degradation_pct,
        "late_trend_pct_per_100_epochs": late_trend_pct_per_100,
        "late_mae_cv_pct": late_cv_pct,
        "overfit_risk": overfit_label(degradation_pct, late_trend_pct_per_100),
        "source": str(path),
    }

    finger_rows = []
    for finger in FINGERS:
        finger_maes = np.asarray([metric(row, f"{finger}_mae") for row in rows], dtype=np.float64)
        finger_best_index = int(np.nanargmin(finger_maes))
        finger_best = rows[finger_best_index]
        finger_rows.append({
            "variant": variant,
            "run": path.parents[1].name,
            "finger": finger,
            "model_best_epoch": summary["best_epoch"],
            "mae_at_model_best": metric(best, f"{finger}_mae"),
            "accuracy_at_model_best": metric(best, f"{finger}_raw_accuracy"),
            "profile_accuracy_at_model_best": metric(best, f"{finger}_profile_accuracy"),
            "corr_at_model_best": metric(best, f"{finger}_corr"),
            "r2_at_model_best": metric(best, f"{finger}_r2"),
            "bias_at_model_best": metric(best, f"{finger}_bias"),
            "best_finger_epoch": int(metric(finger_best, "epoch")),
            "best_finger_mae": metric(finger_best, f"{finger}_mae"),
            "best_finger_accuracy": metric(finger_best, f"{finger}_raw_accuracy"),
            "final_finger_mae": metric(final, f"{finger}_mae"),
            "final_finger_accuracy": metric(final, f"{finger}_raw_accuracy"),
        })
    return summary, finger_rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def short_name(variant: str) -> str:
    return variant.removeprefix("selftouch_fcn_")


def make_plots(out_dir: Path, summaries: List[Dict], finger_rows: List[Dict], histories: Dict[str, List[Dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(summaries, key=lambda row: row["best_mae"])
    colors = plt.cm.tab20(np.linspace(0, 1, len(ordered)))

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    for color, row in zip(colors, ordered):
        history = histories[row["variant"]]
        ax.plot(
            [metric(item, "epoch") for item in history],
            [metric(item, "prediction_mae") for item in history],
            marker="o", markersize=2.8, linewidth=1.7, color=color,
            label=short_name(row["variant"]),
        )
    ax.set_title("Validation raw MAE convergence — 14 self-touch FCN variants")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Raw taxel MAE (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out_dir / "mae_convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    for color, row in zip(colors, ordered):
        history = histories[row["variant"]]
        ax.plot(
            [metric(item, "epoch") for item in history],
            [metric(item, "prediction_raw_accuracy") for item in history],
            marker="o", markersize=2.8, linewidth=1.7, color=color,
            label=short_name(row["variant"]),
        )
    ax.set_title("Validation raw accuracy convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Raw accuracy (%) — higher is better")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out_dir / "accuracy_convergence.png", dpi=180)
    plt.close(fig)

    labels = [short_name(row["variant"]) for row in ordered]
    best_values = np.asarray([row["best_mae"] for row in ordered])
    final_values = np.asarray([row["final_mae"] for row in ordered])
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    ax.bar(x - 0.2, best_values, width=0.4, label="best checkpoint", color="#3f7fba")
    ax.bar(x + 0.2, final_values, width=0.4, label="epoch 500", color="#ff9b73")
    ax.set_xticks(x, labels, rotation=42, ha="right")
    ax.set_ylabel("Raw taxel MAE")
    ax.set_title("Best versus final validation MAE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "best_vs_final_mae.png", dpi=180)
    plt.close(fig)

    by_key = {(row["variant"], row["finger"]): row for row in finger_rows}
    for field, title, filename, cmap, fmt in (
        ("mae_at_model_best", "Per-finger MAE at each model's best epoch", "per_finger_mae_heatmap.png", "magma_r", ".0f"),
        ("accuracy_at_model_best", "Per-finger raw accuracy at each model's best epoch", "per_finger_accuracy_heatmap.png", "viridis", ".1f"),
    ):
        matrix = np.asarray([
            [by_key[(row["variant"], finger)][field] for finger in FINGERS]
            for row in ordered
        ], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(FINGERS)), [name.capitalize() for name in FINGERS])
        ax.set_yticks(np.arange(len(ordered)), labels)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > np.nanmedian(matrix) else "black")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


def fmt(value, places=2) -> str:
    value = number(value)
    return "—" if not math.isfinite(value) else f"{value:.{places}f}"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable]) -> str:
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(out_dir: Path, summaries: List[Dict], finger_rows: List[Dict]) -> None:
    ranked = sorted(summaries, key=lambda row: (row["best_mae"], -row["best_accuracy"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    by_key = {(row["variant"], row["finger"]): row for row in finger_rows}
    best_by_finger = {
        finger: min(finger_rows, key=lambda row: row["best_finger_mae"] if row["finger"] == finger else math.inf)
        for finger in FINGERS
    }
    fastest = min(ranked, key=lambda row: row["epoch_within_5pct_best"])
    most_improved = max(ranked, key=lambda row: row["improvement_pct"])
    stable = min(ranked, key=lambda row: row["late_mae_cv_pct"])
    risks = [row for row in ranked if row["overfit_risk"] != "low"]
    singles = [row for row in ranked if short_name(row["variant"]) in {"pos", "vel", "trq", "cmd"}]
    multis = [row for row in ranked if row not in singles]
    single_mean = float(np.mean([row["best_mae"] for row in singles]))
    multi_mean = float(np.mean([row["best_mae"] for row in multis]))
    finger_mean_mae = {
        finger: float(np.mean([row["mae_at_model_best"] for row in finger_rows if row["finger"] == finger]))
        for finger in FINGERS
    }

    lines = [
        "# Self-touch FCN: 14-model analysis",
        "",
        "## Scope and method",
        "",
        "This report compares the newest completed 500-epoch run for each requested FCN input variant. "
        "Ranking is by the minimum aggregate validation raw-taxel MAE. Accuracy is the exported raw "
        "closeness score and is strongly related to MAE; correlation and R² provide shape/variance checks.",
        "",
        "Overfitting risk is a validation-only diagnostic: **high** means final degradation or the normalized "
        "late rising trend exceeds 5%; **watch** means it exceeds 2%; **low** means neither threshold is crossed. "
        "This can detect late regression but cannot prove train/test overfitting without a matching train-metric series.",
        "",
        "## Executive findings",
        "",
        f"- Best overall: **{ranked[0]['variant']}** at epoch {ranked[0]['best_epoch']} "
        f"(MAE {fmt(ranked[0]['best_mae'])}, accuracy {fmt(ranked[0]['best_accuracy'])}%).",
        f"- Fastest to within 5% of its own best: **{fastest['variant']}** at epoch {fastest['epoch_within_5pct_best']}.",
        f"- Largest improvement from first evaluation: **{most_improved['variant']}** "
        f"({fmt(most_improved['improvement_pct'])}%).",
        f"- Most stable late MAE: **{stable['variant']}** (CV {fmt(stable['late_mae_cv_pct'])}%).",
        "- Best finger-specific checkpoints: " + ", ".join(
            f"{finger} = **{best_by_finger[finger]['variant']}** "
            f"(epoch {best_by_finger[finger]['best_finger_epoch']}, MAE {fmt(best_by_finger[finger]['best_finger_mae'])})"
            for finger in FINGERS
        ) + ".",
        "- Validation regression flags: " + (
            ", ".join(f"**{row['variant']}** ({row['overfit_risk']})" for row in risks)
            if risks else "none"
        ) + ".",
        f"- Multi-input variants average MAE {fmt(multi_mean)} versus {fmt(single_mean)} for single-input variants "
        f"({fmt(100.0 * (single_mean - multi_mean) / single_mean)}% lower), although this is descriptive rather than a seeded significance test.",
        "- Average finger difficulty by MAE: " + ", ".join(
            f"{finger} {fmt(finger_mean_mae[finger])}" for finger in FINGERS
        ) + ".",
        "",
        "## Overall ranking",
        "",
        markdown_table(
            ["Rank", "Variant", "Run", "Best epoch", "Best MAE ↓", "Accuracy ↑", "RMSE ↓", "Corr ↑", "R² ↑", "p95 ↓", "Abs bias ↓"],
            [
                [row["rank"], row["variant"], row["run"], row["best_epoch"], fmt(row["best_mae"]),
                 fmt(row["best_accuracy"]), fmt(row["best_rmse"]), fmt(row["best_corr"], 3),
                 fmt(row["best_r2"], 3), fmt(row["best_p95"]), fmt(row["best_abs_bias"])]
                for row in ranked
            ],
        ),
        "",
        "## Convergence and late-regression diagnostics",
        "",
        markdown_table(
            ["Variant", "Initial MAE", "Improvement", "≤10% best", "≤5% best", "≤2% best", "Final MAE", "Final Δ", "Late trend/100 ep", "Late CV", "Risk"],
            [
                [row["variant"], fmt(row["initial_mae"]), f"{fmt(row['improvement_pct'])}%",
                 row["epoch_within_10pct_best"], row["epoch_within_5pct_best"], row["epoch_within_2pct_best"],
                 fmt(row["final_mae"]), f"{fmt(row['final_degradation_pct'])}%",
                 f"{fmt(row['late_trend_pct_per_100_epochs'])}%", f"{fmt(row['late_mae_cv_pct'])}%", row["overfit_risk"]]
                for row in ranked
            ],
        ),
        "",
        "## Recommended checkpoint files",
        "",
        markdown_table(
            ["Variant", "Best evaluated epoch", "Exact checkpoint?", "Recommended saved epoch", "Checkpoint"],
            [
                [row["variant"], row["best_epoch"], "yes" if row["best_checkpoint_saved"] else "no",
                 row["recommended_checkpoint_epoch"], f"`{row['recommended_checkpoint']}`"]
                for row in ranked
            ],
        ),
        "",
        "## Per-finger performance at each model's overall-best epoch",
        "",
        markdown_table(
            ["Variant", "Index MAE / Acc", "Thumb MAE / Acc", "Middle MAE / Acc", "Ring MAE / Acc"],
            [
                [row["variant"]] + [
                    f"{fmt(by_key[(row['variant'], finger)]['mae_at_model_best'])} / "
                    f"{fmt(by_key[(row['variant'], finger)]['accuracy_at_model_best'])}%"
                    for finger in FINGERS
                ]
                for row in ranked
            ],
        ),
        "",
        "## Output files",
        "",
        "- `model_summary.csv`: overall/best/final/convergence diagnostics.",
        "- `per_finger_metrics.csv`: per-finger metrics at model-best, finger-best, and final epochs.",
        "- `mae_convergence.png` and `accuracy_convergence.png`: all histories.",
        "- `best_vs_final_mae.png`: late degradation comparison.",
        "- `per_finger_mae_heatmap.png` and `per_finger_accuracy_heatmap.png`: cross-model finger comparison.",
        "",
        "## Selection guidance",
        "",
        "Use the recommended saved checkpoint rather than assuming every best evaluation epoch has a corresponding weight file. "
        "For deployment, repeat the top candidates with multiple random seeds: these results represent one run per input variant, "
        "so small ranking differences are not evidence of a reproducible advantage.",
        "",
        "## Compatibility caveat",
        "",
        "These results describe the checkpoint files listed above. They were trained before the latest shared FCN backbone change "
        "that ported the `selftouch_fcn_pos` temporal-difference/depthwise architecture to the other variants. New runs from the "
        "current code are a new experiment and should not be mixed into this ranking without regenerating the report.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-weight-root", type=Path, default=Path("model_weight"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/selftouch_fcn_14"))
    parser.add_argument("--minimum-epoch", type=int, default=500)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    finger_rows = []
    histories = {}
    for variant in VARIANTS:
        path, rows = choose_completed_run(args.model_weight_root, variant, args.minimum_epoch)
        summary, fingers = summarize(variant, path, rows)
        summaries.append(summary)
        finger_rows.extend(fingers)
        histories[variant] = rows

    summaries.sort(key=lambda row: (row["best_mae"], -row["best_accuracy"]))
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank
    write_csv(args.out_dir / "model_summary.csv", summaries)
    write_csv(args.out_dir / "per_finger_metrics.csv", finger_rows)
    make_plots(args.out_dir, summaries, finger_rows, histories)
    write_report(args.out_dir, summaries, finger_rows)
    print(f"Wrote analysis for {len(summaries)} models to {args.out_dir}")


if __name__ == "__main__":
    main()
