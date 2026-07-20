#!/usr/bin/env python3
"""Aggregate fixed-seed FCN ablations into mean/SD/95% CI tables and plots."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


VARIANTS = (
    "selftouch_fcn_pos", "selftouch_fcn_vel", "selftouch_fcn_trq", "selftouch_fcn_cmd",
    "selftouch_fcn_posvel", "selftouch_fcn_postrq", "selftouch_fcn_poscmd",
    "selftouch_fcn_velcmd", "selftouch_fcn_veltrq", "selftouch_fcn_trqcmd",
    "selftouch_fcn_posveltrq", "selftouch_fcn_postrqcmd", "selftouch_fcn_poscmdvel",
    "selftouch_fcn_posveltrqcmd",
)
FINGERS = ("index", "thumb", "middle", "ring")
METRICS = (
    "prediction_mae", "prediction_raw_accuracy", "prediction_profile_accuracy",
    "prediction_rmse", "prediction_corr", "prediction_r2", "prediction_error_p95",
    "prediction_abs_bias",
)
T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def number(value, default=math.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_history(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if math.isfinite(number(row.get("prediction_mae")))]


def stats(values):
    values = np.asarray([number(value) for value in values], dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if not n:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "ci95": math.nan}
    mean = float(np.mean(values))
    if n == 1:
        return {"n": 1, "mean": mean, "sd": math.nan, "ci95": math.nan}
    sd = float(np.std(values, ddof=1))
    critical = T_CRITICAL_975.get(n - 1, 1.96)
    return {"n": n, "mean": mean, "sd": sd, "ci95": critical * sd / math.sqrt(n)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, places=2):
    value = number(value)
    return "—" if not math.isfinite(value) else f"{value:.{places}f}"


def short(variant):
    return variant.removeprefix("selftouch_fcn_")


def discover(root, prefix, minimum_epoch):
    pattern = re.compile(rf"^{re.escape(prefix)}_(with_history|no_history)_seed(\d+)$")
    runs = []
    histories = {}
    for variant in VARIANTS:
        for path in sorted((root / variant).glob(f"{prefix}_*_seed*/plots/raw_prediction_metrics.csv")):
            run_name = path.parents[1].name
            match = pattern.match(run_name)
            if not match:
                continue
            history_mode, seed_text = match.groups()
            history = read_history(path)
            if not history:
                continue
            history.sort(key=lambda row: number(row.get("epoch")))
            max_epoch = int(max(number(row.get("epoch"), 0) for row in history))
            if max_epoch < minimum_epoch:
                continue
            best = min(history, key=lambda row: (number(row.get("prediction_mae")), number(row.get("epoch"))))
            final = history[-1]
            run = {
                "variant": variant,
                "history_mode": history_mode,
                "seed": int(seed_text),
                "run": run_name,
                "best_epoch": int(number(best.get("epoch"), 0)),
                "best_optimizer_step": int(number(best.get("optimizer_step"), 0)),
                "final_epoch": int(number(final.get("epoch"), 0)),
                "final_optimizer_step": int(number(final.get("optimizer_step"), 0)),
                "source": str(path),
            }
            for metric in METRICS:
                run[f"best_{metric}"] = number(best.get(metric))
                run[f"final_{metric}"] = number(final.get(metric))
            for finger in FINGERS:
                run[f"best_{finger}_mae"] = number(best.get(f"{finger}_mae"))
                run[f"best_{finger}_corr"] = number(best.get(f"{finger}_corr"))
                run[f"best_{finger}_r2"] = number(best.get(f"{finger}_r2"))
            runs.append(run)
            histories[(variant, history_mode, int(seed_text))] = history
    return runs, histories


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["history_mode"], run["variant"])].append(run)
    summary = []
    fingers = []
    for (history_mode, variant), group in sorted(grouped.items()):
        row = {
            "history_mode": history_mode,
            "variant": variant,
            "seeds": " ".join(str(item["seed"]) for item in sorted(group, key=lambda x: x["seed"])),
        }
        for metric in METRICS:
            result = stats(item[f"best_{metric}"] for item in group)
            for field, value in result.items():
                row[f"best_{metric}_{field}"] = value
            final_result = stats(item[f"final_{metric}"] for item in group)
            for field, value in final_result.items():
                row[f"final_{metric}_{field}"] = value
        summary.append(row)
        for finger in FINGERS:
            finger_row = {"history_mode": history_mode, "variant": variant, "finger": finger}
            for metric in ("mae", "corr", "r2"):
                result = stats(item[f"best_{finger}_{metric}"] for item in group)
                for field, value in result.items():
                    finger_row[f"{metric}_{field}"] = value
            fingers.append(finger_row)
    for history_mode in {row["history_mode"] for row in summary}:
        ordered = sorted(
            (row for row in summary if row["history_mode"] == history_mode),
            key=lambda row: row["best_prediction_mae_mean"],
        )
        for rank, row in enumerate(ordered, 1):
            row["rank"] = rank
    return summary, fingers


def completeness_rows(runs, expected_seeds, modes):
    present = {(run["variant"], run["history_mode"], run["seed"]) for run in runs}
    rows = []
    for mode in modes:
        for variant in VARIANTS:
            missing = [seed for seed in expected_seeds if (variant, mode, seed) not in present]
            rows.append({
                "history_mode": mode,
                "variant": variant,
                "expected_n": len(expected_seeds),
                "complete_n": len(expected_seeds) - len(missing),
                "missing_seeds": " ".join(map(str, missing)),
            })
    return rows


def make_plots(out_dir, summary, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted({row["history_mode"] for row in summary})
    for mode in modes:
        rows = sorted(
            (row for row in summary if row["history_mode"] == mode),
            key=lambda row: row["best_prediction_mae_mean"],
        )
        labels = [short(row["variant"]) for row in rows]
        means = np.asarray([row["best_prediction_mae_mean"] for row in rows])
        ci = np.asarray([number(row["best_prediction_mae_ci95"], 0.0) for row in rows])
        y = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
        ax.errorbar(means, y, xerr=ci, fmt="o", capsize=4, color="#2468a2")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Best raw-taxel MAE: mean ± 95% CI")
        ax.set_title(f"Fixed-seed FCN ablation — {mode.replace('_', ' ')}")
        ax.grid(True, axis="x", alpha=0.25)
        fig.savefig(out_dir / f"ranking_mae_{mode}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(VARIANTS)))
        for color, variant in zip(colors, VARIANTS):
            seed_histories = [
                history for (item_variant, item_mode, _), history in histories.items()
                if item_variant == variant and item_mode == mode
            ]
            by_step = defaultdict(list)
            for history in seed_histories:
                for item in history:
                    step = int(number(item.get("optimizer_step"), number(item.get("epoch"), 0) * 300))
                    by_step[step].append(number(item.get("prediction_mae")))
            steps = sorted(by_step)
            if not steps:
                continue
            means = np.asarray([stats(by_step[step])["mean"] for step in steps])
            cis = np.asarray([number(stats(by_step[step])["ci95"], 0.0) for step in steps])
            ax.plot(steps, means, color=color, linewidth=1.7, label=short(variant))
            ax.fill_between(steps, means - cis, means + cis, color=color, alpha=0.08)
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Raw-taxel MAE")
        ax.set_title(f"Convergence by optimizer updates — {mode.replace('_', ' ')}")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(out_dir / f"convergence_updates_{mode}.png", dpi=180)
        plt.close(fig)

    if {"with_history", "no_history"}.issubset(modes):
        by_key = {(row["variant"], row["history_mode"]): row for row in summary}
        variants = [variant for variant in VARIANTS if (variant, "with_history") in by_key and (variant, "no_history") in by_key]
        x = np.arange(len(variants))
        with_values = [by_key[(variant, "with_history")]["best_prediction_mae_mean"] for variant in variants]
        without_values = [by_key[(variant, "no_history")]["best_prediction_mae_mean"] for variant in variants]
        fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
        ax.bar(x - 0.2, with_values, 0.4, label="with tactile history")
        ax.bar(x + 0.2, without_values, 0.4, label="no tactile history")
        ax.set_xticks(x, [short(variant) for variant in variants], rotation=40, ha="right")
        ax.set_ylabel("Best raw-taxel MAE (seed mean)")
        ax.set_title("Contribution of shared tactile history")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.savefig(out_dir / "tactile_history_ablation.png", dpi=180)
        plt.close(fig)


def write_report(out_dir, summary, completeness):
    lines = [
        "# Fixed-seed self-touch FCN ablation",
        "",
        "All rankings use the mean of each seed's best held-out raw-taxel MAE. "
        "SD is the sample standard deviation and CI is a two-sided 95% Student-t interval.",
        "",
    ]
    for mode in sorted({row["history_mode"] for row in summary}):
        rows = sorted(
            (row for row in summary if row["history_mode"] == mode),
            key=lambda row: row["best_prediction_mae_mean"],
        )
        lines.extend([
            f"## {mode.replace('_', ' ').title()}", "",
            "| Rank | Variant | n | MAE mean ± SD | 95% CI half-width | Raw closeness | RMSE | Corr | R² | p95 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['rank']} | {short(row['variant'])} | {int(row['best_prediction_mae_n'])} | "
                f"{fmt(row['best_prediction_mae_mean'])} ± {fmt(row['best_prediction_mae_sd'])} | "
                f"±{fmt(row['best_prediction_mae_ci95'])} | {fmt(row['best_prediction_raw_accuracy_mean'])}% | "
                f"{fmt(row['best_prediction_rmse_mean'])} | {fmt(row['best_prediction_corr_mean'], 3)} | "
                f"{fmt(row['best_prediction_r2_mean'], 3)} | {fmt(row['best_prediction_error_p95_mean'])} |"
            )
        lines.append("")
    missing = [row for row in completeness if row["missing_seeds"]]
    lines.extend(["## Completeness", ""])
    if missing:
        lines.append("Incomplete cells: " + "; ".join(
            f"{row['variant']}/{row['history_mode']} missing {row['missing_seeds']}" for row in missing
        ))
    else:
        lines.append("Every requested variant/history cell contains every expected seed.")
    lines.extend([
        "", "## Visual interpretation", "",
        "Each run's `tactile_profile_epoch_*.png` overlays the predicted and raw mean tactile trace. "
        "Each `tactile_identity_epoch_*.png` checks every taxel against the pred=raw identity line, and "
        "the taxel-error heatmap exposes errors that averaging can hide.", "",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("model_weight"))
    parser.add_argument("--run-prefix", default="fcn_ablation_v2")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/selftouch_fcn_ablation_rerun"))
    parser.add_argument("--minimum-epoch", type=int, default=500)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument(
        "--expected-modes",
        nargs="+",
        choices=["with_history", "no_history"],
        default=["with_history"],
    )
    parser.add_argument("--strict", action="store_true", help="fail if an expected seed is missing")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs, histories = discover(args.root, args.run_prefix, args.minimum_epoch)
    if not runs:
        raise SystemExit(f"No completed runs found for prefix {args.run_prefix!r}")
    summary, fingers = aggregate_runs(runs)
    completeness = completeness_rows(runs, args.expected_seeds, args.expected_modes)
    write_csv(args.out_dir / "run_metrics.csv", runs)
    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "per_finger_summary.csv", fingers)
    write_csv(args.out_dir / "completeness.csv", completeness)
    make_plots(args.out_dir, summary, histories)
    write_report(args.out_dir, summary, completeness)
    missing = [row for row in completeness if row["missing_seeds"]]
    print(f"Wrote aggregate analysis to {args.out_dir} ({len(runs)} completed runs).")
    if args.strict and missing:
        raise SystemExit(f"Missing {sum(len(row['missing_seeds'].split()) for row in missing)} expected runs")


if __name__ == "__main__":
    main()
