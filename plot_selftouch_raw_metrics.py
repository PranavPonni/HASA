"""Plot raw self-touch prediction metrics."""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first_finite(row: Dict[str, str], *keys: str) -> float:
    for key in keys:
        value = _float_or_nan(row.get(key))
        if np.isfinite(value):
            return value
    return float("nan")


RAW_REQUIRED_COLUMNS = {"prediction_mae", "prediction_rmse", "prediction_corr"}


def _metrics_from_history(path: str) -> Optional[Dict]:
    rows = _read_csv(path)
    if not rows:
        return None
    if not RAW_REQUIRED_COLUMNS.issubset(set(rows[0].keys())):
        return None

    epochs = np.array([int(float(r.get("epoch", i + 1))) for i, r in enumerate(rows)], dtype=np.int32)
    mae = np.array([_float_or_nan(r.get("prediction_mae")) for r in rows], dtype=np.float32)
    raw_accuracy = np.array([
        _first_finite(r, "prediction_raw_accuracy", "prediction_accuracy")
        for r in rows
    ], dtype=np.float32)
    profile_accuracy = np.array([
        _first_finite(r, "prediction_profile_accuracy")
        for r in rows
    ], dtype=np.float32)
    accuracy = raw_accuracy
    rmse = np.array([_float_or_nan(r.get("prediction_rmse")) for r in rows], dtype=np.float32)
    corr = np.array([_float_or_nan(r.get("prediction_corr")) for r in rows], dtype=np.float32)
    r2 = np.array([_float_or_nan(r.get("prediction_r2")) for r in rows], dtype=np.float32)
    error_p95 = np.array([_float_or_nan(r.get("prediction_error_p95")) for r in rows], dtype=np.float32)
    if not np.isfinite(mae).any():
        return None

    best_i = int(np.nanargmin(mae))
    latest_i = len(rows) - 1
    return {
        "best_epoch": int(epochs[best_i]),
        "best_mae": float(mae[best_i]),
        "best_accuracy": float(accuracy[best_i]) if np.isfinite(accuracy[best_i]) else "",
        "best_profile_accuracy": float(profile_accuracy[best_i]) if np.isfinite(profile_accuracy[best_i]) else "",
        "best_rmse": float(rmse[best_i]) if np.isfinite(rmse[best_i]) else "",
        "best_corr": float(corr[best_i]) if np.isfinite(corr[best_i]) else "",
        "best_r2": float(r2[best_i]) if np.isfinite(r2[best_i]) else "",
        "best_error_p95": float(error_p95[best_i]) if np.isfinite(error_p95[best_i]) else "",
        "latest_epoch": int(epochs[latest_i]),
        "latest_mae": float(mae[latest_i]) if np.isfinite(mae[latest_i]) else "",
        "latest_accuracy": float(accuracy[latest_i]) if np.isfinite(accuracy[latest_i]) else "",
        "latest_profile_accuracy": float(profile_accuracy[latest_i]) if np.isfinite(profile_accuracy[latest_i]) else "",
        "latest_rmse": float(rmse[latest_i]) if np.isfinite(rmse[latest_i]) else "",
        "latest_corr": float(corr[latest_i]) if np.isfinite(corr[latest_i]) else "",
        "latest_r2": float(r2[latest_i]) if np.isfinite(r2[latest_i]) else "",
        "latest_error_p95": float(error_p95[latest_i]) if np.isfinite(error_p95[latest_i]) else "",
        "source": path,
    }


def _metrics_from_epoch_metrics(path: str) -> Optional[Dict]:
    rows = _read_csv(path)
    if not rows or "tactile_line_mae" not in rows[0]:
        return None

    epochs = np.array([int(float(r.get("epoch", i + 1))) for i, r in enumerate(rows)], dtype=np.int32)
    mae = np.array([_float_or_nan(r.get("tactile_line_mae")) for r in rows], dtype=np.float32)
    raw_accuracy = np.array([
        _first_finite(r, "tactile_line_raw_accuracy", "tactile_line_accuracy")
        for r in rows
    ], dtype=np.float32)
    profile_accuracy = np.array([
        _first_finite(r, "tactile_line_profile_accuracy")
        for r in rows
    ], dtype=np.float32)
    accuracy = raw_accuracy
    rmse = np.array([_float_or_nan(r.get("tactile_line_rmse")) for r in rows], dtype=np.float32)
    corr = np.array([_float_or_nan(r.get("tactile_line_corr")) for r in rows], dtype=np.float32)
    r2 = np.array([_float_or_nan(r.get("tactile_line_r2")) for r in rows], dtype=np.float32)
    error_p95 = np.array([_float_or_nan(r.get("tactile_line_error_p95")) for r in rows], dtype=np.float32)
    if not np.isfinite(mae).any():
        return None

    best_i = int(np.nanargmin(mae))
    latest_i = len(rows) - 1
    return {
        "best_epoch": int(epochs[best_i]),
        "best_mae": float(mae[best_i]),
        "best_accuracy": float(accuracy[best_i]) if np.isfinite(accuracy[best_i]) else "",
        "best_profile_accuracy": float(profile_accuracy[best_i]) if np.isfinite(profile_accuracy[best_i]) else "",
        "best_rmse": float(rmse[best_i]) if np.isfinite(rmse[best_i]) else "",
        "best_corr": float(corr[best_i]) if np.isfinite(corr[best_i]) else "",
        "best_r2": float(r2[best_i]) if np.isfinite(r2[best_i]) else "",
        "best_error_p95": float(error_p95[best_i]) if np.isfinite(error_p95[best_i]) else "",
        "latest_epoch": int(epochs[latest_i]),
        "latest_mae": float(mae[latest_i]) if np.isfinite(mae[latest_i]) else "",
        "latest_accuracy": float(accuracy[latest_i]) if np.isfinite(accuracy[latest_i]) else "",
        "latest_profile_accuracy": float(profile_accuracy[latest_i]) if np.isfinite(profile_accuracy[latest_i]) else "",
        "latest_rmse": float(rmse[latest_i]) if np.isfinite(rmse[latest_i]) else "",
        "latest_corr": float(corr[latest_i]) if np.isfinite(corr[latest_i]) else "",
        "latest_r2": float(r2[latest_i]) if np.isfinite(r2[latest_i]) else "",
        "latest_error_p95": float(error_p95[latest_i]) if np.isfinite(error_p95[latest_i]) else "",
        "source": path,
    }


def collect_raw_metrics(model_weight_root: str) -> List[Dict]:
    records: List[Dict] = []
    for family in sorted(os.listdir(model_weight_root)):
        if not family.startswith("selftouch"):
            continue
        family_dir = os.path.join(model_weight_root, family)
        if not os.path.isdir(family_dir):
            continue
        for sweep in sorted(os.listdir(family_dir)):
            sweep_dir = os.path.join(family_dir, sweep)
            if not os.path.isdir(sweep_dir) or sweep == "parameter_base":
                continue

            result = None
            for path in (
                os.path.join(sweep_dir, "plots", "raw_prediction_metrics.csv"),
                os.path.join(sweep_dir, "raw_prediction_metrics.csv"),
            ):
                if os.path.isfile(path):
                    result = _metrics_from_history(path)
                    if result is not None:
                        break
            if result is None:
                metrics_path = os.path.join(sweep_dir, "epoch_metrics.csv")
                if os.path.isfile(metrics_path):
                    result = _metrics_from_epoch_metrics(metrics_path)
            if result is None:
                continue
            result.update({"folder": family, "sweep": sweep})
            records.append(result)
    return records


def collect_raw_metrics_from_summary(path: str) -> List[Dict]:
    records: List[Dict] = []
    for row in _read_csv(path):
        if row.get("status") and row.get("status") != "ok":
            continue
        try:
            mae = float(row["prediction_mae"])
        except (KeyError, TypeError, ValueError):
            continue
        records.append(
            {
                "rank_by_mae": row.get("rank_by_mae", ""),
                "folder": row.get("family", ""),
                "sweep": row.get("sweep", ""),
                "best_epoch": int(float(row.get("epoch", 0) or 0)),
                "best_mae": mae,
                "best_accuracy": _first_finite(row, "prediction_raw_accuracy", "prediction_accuracy"),
                "best_profile_accuracy": _first_finite(row, "prediction_profile_accuracy"),
                "best_rmse": _float_or_nan(row.get("prediction_rmse")),
                "best_corr": _float_or_nan(row.get("prediction_corr")),
                "best_r2": _float_or_nan(row.get("prediction_r2")),
                "best_error_p95": _float_or_nan(row.get("prediction_error_p95")),
                "latest_epoch": int(float(row.get("epoch", 0) or 0)),
                "latest_mae": mae,
                "latest_accuracy": _first_finite(row, "prediction_raw_accuracy", "prediction_accuracy"),
                "latest_profile_accuracy": _first_finite(row, "prediction_profile_accuracy"),
                "latest_rmse": _float_or_nan(row.get("prediction_rmse")),
                "latest_corr": _float_or_nan(row.get("prediction_corr")),
                "latest_r2": _float_or_nan(row.get("prediction_r2")),
                "latest_error_p95": _float_or_nan(row.get("prediction_error_p95")),
                "source": path,
            }
        )
    return records


def save_outputs(records: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    records = sorted(records, key=lambda row: float(row.get("best_mae") or np.inf))
    csv_path = os.path.join(out_dir, "selftouch_raw_metrics_all_sweeps.csv")
    fieldnames = [
        "rank_by_mae",
        "folder",
        "sweep",
        "best_epoch",
        "best_mae",
        "best_accuracy",
        "best_profile_accuracy",
        "best_rmse",
        "best_corr",
        "best_r2",
        "best_error_p95",
        "latest_epoch",
        "latest_mae",
        "latest_accuracy",
        "latest_profile_accuracy",
        "latest_rmse",
        "latest_corr",
        "latest_r2",
        "latest_error_p95",
        "source",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if not records:
        print(f"No raw selftouch metric histories found. Wrote empty CSV: {csv_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder_to_rows = defaultdict(list)
    for idx, row in enumerate(records):
        folder_to_rows[row["folder"]].append((idx, row))

    cmap = plt.cm.get_cmap("tab10")
    plt.figure(figsize=(max(10, min(28, len(records) * 0.18)), 6))
    for color_i, (folder, rows) in enumerate(sorted(folder_to_rows.items())):
        xs = [idx for idx, _ in rows]
        ys = [row["best_mae"] for _, row in rows]
        plt.scatter(xs, ys, s=28, alpha=0.8, color=cmap(color_i % 10), label=folder)
    plt.title("Self-touch raw MAE by sweep")
    plt.xlabel("Sweep index (see CSV for names)")
    plt.ylabel("Best raw MAE")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "selftouch_raw_mae_all_sweeps.png")
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {plot_path}")

    accuracy_records = [
        row for row in records
        if np.isfinite(_float_or_nan(row.get("best_accuracy")))
    ]
    if accuracy_records:
        folder_to_rows = defaultdict(list)
        for idx, row in enumerate(accuracy_records):
            folder_to_rows[row["folder"]].append((idx, row))
        plt.figure(figsize=(max(10, min(28, len(accuracy_records) * 0.18)), 6))
        for color_i, (folder, rows) in enumerate(sorted(folder_to_rows.items())):
            xs = [idx for idx, _ in rows]
            ys = [float(row["best_accuracy"]) for _, row in rows]
            plt.scatter(xs, ys, s=28, alpha=0.8, color=cmap(color_i % 10), label=folder)
        plt.title("Self-touch raw taxel accuracy by sweep")
        plt.xlabel("Sweep index (see CSV for names)")
        plt.ylabel("Best raw taxel accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        accuracy_plot_path = os.path.join(out_dir, "selftouch_prediction_accuracy_all_sweeps.png")
        plt.savefig(accuracy_plot_path, dpi=180)
        plt.close()
        print(f"Wrote: {accuracy_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot raw prediction metrics for all selftouch sweeps.")
    parser.add_argument("--model-weight-root", default="model_weight")
    parser.add_argument("--summary-csv", default=None, help="Use a raw all-checkpoints CSV instead of run histories.")
    parser.add_argument("--out-dir", default="log/graphs")
    args = parser.parse_args()
    if args.summary_csv:
        records = collect_raw_metrics_from_summary(args.summary_csv)
    else:
        records = collect_raw_metrics(args.model_weight_root)
    save_outputs(records, args.out_dir)


if __name__ == "__main__":
    main()
