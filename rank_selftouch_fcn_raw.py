"""Rank selftouch_fcn input variants by raw tactile prediction metrics."""

import argparse
import csv
import math
from pathlib import Path


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


def num(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def raw_acc(row, finger):
    return num(row.get(f"{finger}_raw_accuracy", row.get(f"{finger}_accuracy")))


def aggregate_raw_acc(row):
    vals = [raw_acc(row, finger) for finger in ("index", "thumb", "middle", "ring")]
    vals = [value for value in vals if math.isfinite(value)]
    if vals:
        return sum(vals) / len(vals)
    return num(row.get("prediction_raw_accuracy"))


def best_row(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return min(
        rows,
        key=lambda r: (
            num(r.get("prediction_mae")),
            -aggregate_raw_acc(r),
            num(r.get("prediction_rmse")),
            num(r.get("prediction_error_p95")),
            -num(r.get("prediction_profile_accuracy", r.get("prediction_accuracy"))),
        ),
    )


def fmt(value, places=2):
    value = num(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{places}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="parameter_base")
    args = parser.parse_args()

    records = []
    for variant in VARIANTS:
        metrics = Path("model_weight") / variant / args.run_name / "plots" / "raw_prediction_metrics.csv"
        if not metrics.is_file():
            continue
        row = best_row(metrics)
        if row is None:
            continue
        records.append(
            {
                "variant": variant,
                "epoch": int(num(row.get("epoch"))),
                "mae": num(row.get("prediction_mae")),
                "raw_acc": aggregate_raw_acc(row),
                "index_raw_acc": raw_acc(row, "index"),
                "thumb_raw_acc": raw_acc(row, "thumb"),
                "middle_raw_acc": raw_acc(row, "middle"),
                "ring_raw_acc": raw_acc(row, "ring"),
                "rmse": num(row.get("prediction_rmse")),
                "p95": num(row.get("prediction_error_p95")),
                "bias": num(row.get("prediction_bias")),
                "abs_bias": num(row.get("prediction_abs_bias")),
                "profile_acc": num(row.get("prediction_profile_accuracy", row.get("prediction_accuracy"))),
            }
        )

    records.sort(
        key=lambda r: (
            r["mae"],
            -r["raw_acc"],
            r["rmse"],
            r["p95"],
            -r["profile_acc"],
        )
    )

    print(
        "| Rank | Model/input | Epoch | MAE ↓ | Overall raw taxel acc % ↑ | "
        "Index raw % | Thumb raw % | Middle raw % | Ring raw % | RMSE ↓ | p95 error ↓ | Bias raw-pred | Abs bias | Profile acc % tie-break |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rank, rec in enumerate(records, 1):
        print(
            f"| {rank} | `{rec['variant']}` | {rec['epoch']} | {fmt(rec['mae'])} | "
            f"{fmt(rec['raw_acc'])} | {fmt(rec['index_raw_acc'])} | "
            f"{fmt(rec['thumb_raw_acc'])} | {fmt(rec['middle_raw_acc'])} | "
            f"{fmt(rec['ring_raw_acc'])} | "
            f"{fmt(rec['rmse'])} | {fmt(rec['p95'])} | "
            f"{fmt(rec['bias'])} | {fmt(rec['abs_bias'])} | "
            f"{fmt(rec['profile_acc'])} |"
        )

    if not records:
        print("\nNo completed selftouch_fcn raw metric CSVs found yet.")


if __name__ == "__main__":
    main()
