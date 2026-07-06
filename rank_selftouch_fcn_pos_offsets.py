"""Summarize selftouch_fcn_pos temporal-offset ablation runs."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from selftouch_offset_utils import dropped_target_count, target_window


GROUPS = {
    "Baseline": [("selftouch_fcn_pos", 0)],
    "Future": [
        ("selftouch_fcn_pos_tplus2", 2),
        ("selftouch_fcn_pos_tplus5", 5),
        ("selftouch_fcn_pos_tplus10", 10),
    ],
    "Past": [
        ("selftouch_fcn_pos_tminus2", -2),
        ("selftouch_fcn_pos_tminus5", -5),
        ("selftouch_fcn_pos_tminus10", -10),
    ],
}


def num(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def raw_acc(row, finger):
    return num(row.get(f"{finger}_raw_accuracy", row.get(f"{finger}_accuracy")))


def aggregate_raw_acc(row):
    vals = [raw_acc(row, finger) for finger in ("index", "thumb", "middle")]
    vals = [value for value in vals if math.isfinite(value)]
    if vals:
        return sum(vals) / len(vals)
    return num(row.get("prediction_raw_accuracy"))


def read_rows(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def latest_row(rows):
    return max(rows, key=lambda row: num(row.get("epoch")))


def final_record(path):
    rows = read_rows(path)
    if not rows:
        return None
    row = latest_row(rows)
    return {
        "run": path.parents[1].name,
        "epoch": int(num(row.get("epoch"))),
        "mae": num(row.get("prediction_mae")),
        "raw_acc": aggregate_raw_acc(row),
        "corr": num(row.get("prediction_corr")),
        "p95": num(row.get("prediction_error_p95")),
    }


def collect_variant(variant, min_epoch):
    run_records = []
    for path in sorted((Path("model_weight") / variant).glob("*/plots/raw_prediction_metrics.csv")):
        record = final_record(path)
        if record is None:
            continue
        if record["epoch"] >= min_epoch and math.isfinite(record["mae"]):
            run_records.append(record)
    if not run_records:
        return {
            "variant": variant,
            "completed": 0,
            "best": None,
            "median_mae": math.nan,
            "mean_mae": math.nan,
            "std_mae": math.nan,
        }

    best = min(run_records, key=lambda rec: (rec["mae"], -rec["raw_acc"], rec["p95"]))
    maes = [rec["mae"] for rec in run_records]
    return {
        "variant": variant,
        "completed": len(run_records),
        "best": best,
        "median_mae": statistics.median(maes),
        "mean_mae": statistics.mean(maes),
        "std_mae": statistics.stdev(maes) if len(maes) > 1 else 0.0,
    }


def fmt(value, places=2):
    value = num(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{places}f}"


def print_table(title, rows):
    ranked = sorted(
        rows,
        key=lambda row: (
            math.inf if row["best"] is None else row["best"]["mae"],
            0.0 if row["best"] is None else -row["best"]["raw_acc"],
        ),
    )
    print(f"\n## {title}")
    print("| Rank | Variant | Completed | Best run | MAE | Raw acc % | Corr | p95 | Median MAE | Mean MAE | Std MAE |")
    print("|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(ranked, 1):
        best = row["best"] or {}
        print(
            f"| {rank} | `{row['variant']}` | {row['completed']} | "
            f"`{best.get('run', '')}` | {fmt(best.get('mae'))} | "
            f"{fmt(best.get('raw_acc'))} | {fmt(best.get('corr'), 3)} | "
            f"{fmt(best.get('p95'))} | {fmt(row['median_mae'])} | "
            f"{fmt(row['mean_mae'])} | {fmt(row['std_mae'])} |"
        )


def load_diagnostic(variant):
    diag_paths = sorted((Path("parameter") / variant).glob("*/temporal_offset_diagnostics.json"))
    for path in diag_paths:
        try:
            return json.loads(path.read_text())
        except Exception:
            continue
    return None


def print_notes(sequence_length):
    print("\n## Offset Notes")
    for group, variants in GROUPS.items():
        for variant, offset in variants:
            if offset == 0:
                continue
            diag = load_diagnostic(variant)
            if diag:
                drop = diag.get("drop", {}).get("all", {})
                drop_text = (
                    f"{drop.get('dropped_samples', 0)}/{drop.get('base_samples', 0)} "
                    f"({float(drop.get('drop_fraction', 0.0)):.3%})"
                )
                leakage = diag.get("leakage")
                if leakage:
                    leakage_text = (
                        f"; leakage {leakage.get('future_contact_or_precursor_samples', 0)}/"
                        f"{leakage.get('target_contact_samples', 0)} "
                        f"({float(leakage.get('fraction', 0.0)):.1%})"
                    )
                    if leakage.get("flagged"):
                        leakage_text += " FLAGGED"
                else:
                    leakage_text = ""
            else:
                start, stop = target_window(sequence_length, offset)
                dropped = dropped_target_count(sequence_length, offset)
                base = max(sequence_length - 1, 0)
                drop_text = f"{dropped}/{base} ({(dropped / base if base else 0.0):.3%}) expected per episode"
                leakage_text = ""
                if offset in (5, 10):
                    leakage_text = "; leakage pending until the first run builds diagnostics"
            print(f"- {group} `{variant}` offset {offset:+d}: drop {drop_text}{leakage_text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-epoch", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=400)
    args = parser.parse_args()

    for title, variants in GROUPS.items():
        rows = [collect_variant(variant, args.min_epoch) for variant, _ in variants]
        print_table(title, rows)
    print_notes(args.sequence_length)


if __name__ == "__main__":
    main()
