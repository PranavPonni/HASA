import csv
import os
import pickle
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data_preproc import unscale_data


FINGER_ORDER = ("index", "thumb", "middle", "ring")
FINGER_TO_KEY = {
    "index": "tactile_index_tip",
    "thumb": "tactile_thumb_tip",
    "middle": "tactile_middle_tip",
    "ring": "tactile_ring_tip",
}
FINGER_COLORS = {
    "index": "#3f7fba",
    "thumb": "#ff7f50",
    "middle": "#35b779",
    "ring": "#8d63e8",
}
TACTILE_PLOT_TITLE = "Predicted self-touch vs raw tactile (taxel-mean raw values)"
TAXELS_PER_FINGER = 90

TACTILE_XMAX = 400
TACTILE_YMIN = -30.0
TACTILE_YMAX = 30.0
TACTILE_YTICKS = [-30, -20, -10, 0, 10, 20, 30]
PROFILE_ACCURACY_SCALE = TACTILE_YMAX - TACTILE_YMIN


def included_fingers_from_combinations(combinations: Optional[Sequence[str]]) -> set:
    """Infer the fingers represented by the configured self-touch combinations."""
    if not combinations:
        return set(FINGER_ORDER)

    active = set()
    for combo in combinations:
        text = str(combo).lower().replace("_", "-")
        for finger in FINGER_ORDER:
            if finger in text:
                active.add(finger)
    return active or set(FINGER_ORDER)


def active_loss_coef(loss_coef: Optional[Mapping[str, float]], combinations: Optional[Sequence[str]]) -> Dict[str, float]:
    """Copy loss coefficients and disable tactile losses for excluded fingers."""
    coef = dict(loss_coef or {})
    active = included_fingers_from_combinations(combinations)
    for finger, key in FINGER_TO_KEY.items():
        if finger not in active:
            coef[key] = 0.0
    return coef


def load_scaling_param(dataset_param: Mapping) -> Dict:
    param_dir = dataset_param.get("param_file_dir", "")
    path = os.path.join(param_dir, "scaling_param.pkl")
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def maybe_unscale(arr: np.ndarray, key: str, dataset_param: Mapping, scaling_param: Mapping) -> np.ndarray:
    if key not in scaling_param or key not in dataset_param.get("modality", {}):
        return arr.astype(np.float32, copy=False)
    try:
        return unscale_data(arr, scaling_param[key], dataset_param["modality"][key]).astype(np.float32, copy=False)
    except Exception as exc:
        print(f"[warn] failed to unscale {key}; plotting stored values instead: {exc}")
        return arr.astype(np.float32, copy=False)


def _clean_metric_arrays(raw: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return raw, pred


def _safe_float(value, default=np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _as_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "mean"}:
            return True
        if text in {"0", "false", "no", "off", "none"}:
            return False
    return bool(value)


def _finger_set(value, fallback: Sequence[str]) -> set:
    if value is None:
        return set(fallback)
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.replace(";", ",").split(",")]
    else:
        items = [str(part).strip().lower() for part in value]
    selected = {item for item in items if item}
    return selected or set(fallback)


def rmse(raw: np.ndarray, pred: np.ndarray) -> float:
    raw, pred = _clean_metric_arrays(raw, pred)
    if raw.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(raw - pred))))


def r2_score(raw: np.ndarray, pred: np.ndarray) -> float:
    raw, pred = _clean_metric_arrays(raw, pred)
    if raw.size == 0:
        return 0.0
    target_var = float(np.sum(np.square(raw - np.mean(raw))))
    if target_var <= 1e-8:
        return 0.0
    residual = float(np.sum(np.square(raw - pred)))
    score = 1.0 - residual / target_var
    return score if np.isfinite(score) else 0.0


def _robust_signal_spread(raw: np.ndarray) -> float:
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    lo, hi = np.percentile(values, [5, 95])
    spread = float(hi - lo)
    if spread > 1e-8:
        return spread
    magnitude = float(np.percentile(np.abs(values), 95))
    return max(magnitude, 1.0)


def prediction_accuracy(raw: np.ndarray, pred: np.ndarray, *, scale: Optional[float] = None) -> float:
    """Return a 0-100 closeness score where higher means lower absolute error."""
    raw, pred = _clean_metric_arrays(raw, pred)
    raw = raw.reshape(-1)
    pred = pred.reshape(-1)
    count = min(raw.size, pred.size)
    if count == 0:
        return 0.0
    raw = raw[:count]
    pred = pred[:count]
    mae = float(np.mean(np.abs(raw - pred)))
    denom = float(scale) if scale is not None else _robust_signal_spread(raw)
    denom = max(denom, 1e-8)
    return float(np.clip(100.0 * (1.0 - mae / denom), 0.0, 100.0))


def taxel_mean_trace(arr: np.ndarray) -> np.ndarray:
    """Average every taxel in the tactile array into one trace per timestep."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr.mean(axis=(0, 2))
    if arr.ndim == 2:
        return arr.mean(axis=0)
    return arr.reshape(-1)


def temporal_profile(arr: np.ndarray) -> np.ndarray:
    return taxel_mean_trace(arr)


def safe_corr(raw: np.ndarray, pred: np.ndarray) -> float:
    raw = np.asarray(raw).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if raw.size < 2 or pred.size < 2:
        return 0.0
    if float(np.std(raw)) <= 1e-8 or float(np.std(pred)) <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(raw, pred)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _truncate_profile_window(ts, *arrays):
    ts = np.asarray(ts)
    keep = ts <= TACTILE_XMAX
    if keep.size == 0 or not np.any(keep):
        return (ts, *arrays)
    return (ts[keep], *[np.asarray(arr)[keep] for arr in arrays])


def _shared_ylim(series: Iterable[np.ndarray]) -> Tuple[float, float]:
    values = []
    for arr in series:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        if flat.size:
            values.append(flat)
    if not values:
        return -1.0, 1.0
    all_values = np.concatenate(values)
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size == 0:
        return -1.0, 1.0
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    if abs(hi - lo) < 1e-6:
        pad = max(abs(hi) * 0.05, 1.0)
    else:
        pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _tactile_ylim(series: Iterable[np.ndarray]) -> Tuple[float, float]:
    lo, hi = _shared_ylim(series)
    if lo >= TACTILE_YMIN and hi <= TACTILE_YMAX:
        return TACTILE_YMIN, TACTILE_YMAX
    lo = min(lo, TACTILE_YMIN)
    hi = max(hi, TACTILE_YMAX)
    return float(np.floor(lo / 10.0) * 10.0), float(np.ceil(hi / 10.0) * 10.0)


def _start_aligned_prediction(raw: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, float]:
    raw = np.asarray(raw, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if raw.size == 0 or pred.size == 0:
        return pred, 0.0
    shift = float(raw.reshape(-1)[0] - pred.reshape(-1)[0])
    return pred + shift, shift


def _mean_bias_calibrated_prediction(raw: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, float]:
    raw, pred = _clean_metric_arrays(raw, pred)
    if raw.size == 0 or pred.size == 0:
        return pred, 0.0
    count = min(raw.size, pred.size)
    shift = float(np.mean(raw.reshape(-1)[:count] - pred.reshape(-1)[:count]))
    if not np.isfinite(shift):
        shift = 0.0
    return pred + shift, shift


def set_shared_y_limits_from_lines(axes) -> None:
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        series = [line.get_ydata() for line in ax.get_lines()]
        if series:
            ax.set_ylim(*_tactile_ylim(series))
            if ax.get_ylim() == (TACTILE_YMIN, TACTILE_YMAX):
                ax.set_yticks(TACTILE_YTICKS)


def draw_tactile_prediction_profile(ax, ts, raw, pred, err, color):
    """Draw raw tactile, predicted self-touch, and error band on the shared raw scale."""
    plot_raw = np.asarray(raw, dtype=np.float32)
    plot_pred = np.asarray(pred, dtype=np.float32)
    mark_every = max(1, len(ts) // 18)

    ax.fill_between(
        ts,
        plot_raw,
        plot_pred,
        alpha=0.20,
        color=color,
        label="raw-pred gap",
        zorder=1,
    )
    ax.plot(
        ts,
        plot_raw,
        label="raw tactile",
        color=color,
        linewidth=1.8,
        alpha=0.90,
        zorder=3,
    )
    pred_line, = ax.plot(
        ts,
        plot_pred,
        label="pred self-touch avg taxels",
        color=color,
        linewidth=2.8,
        linestyle=(0, (4, 2)),
        marker="o",
        markevery=mark_every,
        markersize=3.0,
        zorder=10,
    )
    try:
        import matplotlib.patheffects as path_effects
        pred_line.set_path_effects([
            path_effects.Stroke(linewidth=4.8, foreground="white"),
            path_effects.Normal(),
        ])
    except Exception:
        pass


def align_next_step_prediction(
    raw_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    next_step: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align predictions to targets using the model training contract.

    Self-touch models predict the next tactile frame: prediction at timestep t is
    trained against raw tactile at timestep t+1. Plots and CSV exports must use
    the same alignment, otherwise the curves look phase-shifted and the error
    margin is wrong.
    """
    raw_arr = np.asarray(raw_arr, dtype=np.float32)
    pred_arr = np.asarray(pred_arr, dtype=np.float32)

    if raw_arr.ndim >= 3 and pred_arr.ndim >= 3:
        if next_step and raw_arr.shape[1] > 1 and pred_arr.shape[1] > 0:
            steps = min(raw_arr.shape[1] - 1, pred_arr.shape[1])
            raw_cmp = raw_arr[:, 1 : 1 + steps, ...]
            pred_cmp = pred_arr[:, :steps, ...]
            timesteps = np.arange(steps)
        else:
            steps = min(raw_arr.shape[1], pred_arr.shape[1])
            raw_cmp = raw_arr[:, :steps, ...]
            pred_cmp = pred_arr[:, :steps, ...]
            timesteps = np.arange(steps)
        return raw_cmp, pred_cmp, timesteps

    return raw_arr, pred_arr, np.arange(raw_arr.shape[0] if raw_arr.ndim else 1)


def _array_from_mapping(mapping: Mapping, key: str) -> Optional[np.ndarray]:
    if key not in mapping or mapping[key] is None:
        return None
    try:
        arr = tensor_to_numpy(mapping[key])
    except Exception as exc:
        print(f"[warn] failed to convert {key}; plotting zeros instead: {exc}")
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _fallback_raw_shape(
    data: Mapping,
    preds: Mapping[str, object],
    finger_names: Sequence[str],
    finger_keys: Sequence[str],
    *,
    next_step: bool,
) -> Tuple[int, int, int]:
    """Infer a (batch, raw_timesteps, dim) shape for synthetic zero channels."""
    for key in finger_keys:
        arr = _array_from_mapping(data, key)
        if arr is not None and arr.ndim >= 3:
            return int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[-1])

    for name in finger_names:
        arr = _array_from_mapping(preds, name)
        if arr is not None and arr.ndim >= 3:
            raw_steps = int(arr.shape[1] + 1) if next_step else int(arr.shape[1])
            return int(arr.shape[0]), max(raw_steps, 1), int(arr.shape[-1])

    return 1, 1, TAXELS_PER_FINGER


def _zero_raw_array(
    *,
    batch: int,
    raw_steps: int,
    dim: int,
) -> np.ndarray:
    return np.zeros((max(batch, 1), max(raw_steps, 1), max(dim, 1)), dtype=np.float32)


def _zero_pred_for_raw(raw_arr: np.ndarray, *, next_step: bool) -> np.ndarray:
    if raw_arr.ndim >= 3:
        steps = raw_arr.shape[1] if not next_step else max(raw_arr.shape[1] - 1, 1)
        return np.zeros((raw_arr.shape[0], steps, raw_arr.shape[-1]), dtype=np.float32)
    return np.zeros_like(raw_arr, dtype=np.float32)


def _raw_metric_fieldnames(finger_names: Sequence[str]) -> List[str]:
    fieldnames = [
        "epoch",
        "prediction_mae",
        "prediction_accuracy",
        "prediction_profile_accuracy",
        "prediction_raw_accuracy",
        "prediction_rmse",
        "prediction_corr",
        "prediction_r2",
        "prediction_error_p95",
        "prediction_bias",
        "prediction_profile_bias",
        "prediction_abs_bias",
        "prediction_raw_mean",
        "prediction_pred_mean",
        "prediction_uncalibrated_raw_accuracy",
        "prediction_uncalibrated_bias",
        "prediction_bias_calibration_shift_abs",
    ]
    for name in finger_names:
        fieldnames.extend([
            f"{name}_mae",
            f"{name}_accuracy",
            f"{name}_profile_accuracy",
            f"{name}_raw_accuracy",
            f"{name}_rmse",
            f"{name}_corr",
            f"{name}_r2",
            f"{name}_error_p95",
            f"{name}_bias",
            f"{name}_profile_bias",
            f"{name}_abs_bias",
            f"{name}_raw_mean",
            f"{name}_pred_mean",
            f"{name}_uncalibrated_raw_accuracy",
            f"{name}_uncalibrated_bias",
            f"{name}_bias_calibration_shift",
        ])
    return fieldnames


def _ensure_metric_history_schema(path: str, fieldnames: Sequence[str]) -> Sequence[str]:
    if not os.path.isfile(path):
        return fieldnames
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    merged_fieldnames = list(fieldnames) + [name for name in old_fieldnames if name not in fieldnames]

    if old_fieldnames == merged_fieldnames:
        return merged_fieldnames
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in merged_fieldnames})
    return merged_fieldnames


def _append_raw_metric_history(plot_dir: str, epoch: int, metrics: Mapping[str, float], finger_names: Sequence[str]) -> str:
    path = os.path.join(plot_dir, "raw_prediction_metrics.csv")
    fieldnames = _ensure_metric_history_schema(path, _raw_metric_fieldnames(finger_names))
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        row = {
            "epoch": int(epoch),
            "prediction_mae": float(metrics.get("tactile_line_mae", 0.0)),
            "prediction_accuracy": float(metrics.get("tactile_line_raw_accuracy", 0.0)),
            "prediction_profile_accuracy": float(metrics.get("tactile_line_profile_accuracy", 0.0)),
            "prediction_raw_accuracy": float(metrics.get("tactile_line_raw_accuracy", 0.0)),
            "prediction_rmse": float(metrics.get("tactile_line_rmse", 0.0)),
            "prediction_corr": float(metrics.get("tactile_line_corr", 0.0)),
            "prediction_r2": float(metrics.get("tactile_line_r2", 0.0)),
            "prediction_error_p95": float(metrics.get("tactile_line_error_p95", 0.0)),
            "prediction_bias": float(metrics.get("tactile_line_bias", 0.0)),
            "prediction_profile_bias": float(metrics.get("tactile_line_profile_bias", 0.0)),
            "prediction_abs_bias": float(metrics.get("tactile_line_abs_bias", 0.0)),
            "prediction_raw_mean": float(metrics.get("tactile_line_raw_mean", 0.0)),
            "prediction_pred_mean": float(metrics.get("tactile_line_pred_mean", 0.0)),
            "prediction_uncalibrated_raw_accuracy": float(
                metrics.get("tactile_line_uncalibrated_raw_accuracy", 0.0)
            ),
            "prediction_uncalibrated_bias": float(metrics.get("tactile_line_uncalibrated_bias", 0.0)),
            "prediction_bias_calibration_shift_abs": float(
                metrics.get("tactile_line_bias_calibration_shift_abs", 0.0)
            ),
        }
        for name in finger_names:
            row[f"{name}_mae"] = float(metrics.get(f"{name}_mae", 0.0))
            row[f"{name}_accuracy"] = float(
                metrics.get(f"{name}_raw_accuracy", metrics.get(f"{name}_accuracy", 0.0))
            )
            row[f"{name}_profile_accuracy"] = float(metrics.get(f"{name}_profile_accuracy", 0.0))
            row[f"{name}_raw_accuracy"] = float(
                metrics.get(f"{name}_raw_accuracy", metrics.get(f"{name}_accuracy", 0.0))
            )
            row[f"{name}_rmse"] = float(metrics.get(f"{name}_rmse", 0.0))
            row[f"{name}_corr"] = float(metrics.get(f"{name}_corr", 0.0))
            row[f"{name}_r2"] = float(metrics.get(f"{name}_r2", 0.0))
            row[f"{name}_error_p95"] = float(metrics.get(f"{name}_error_p95", 0.0))
            row[f"{name}_bias"] = float(metrics.get(f"{name}_bias", 0.0))
            row[f"{name}_profile_bias"] = float(metrics.get(f"{name}_profile_bias", 0.0))
            row[f"{name}_abs_bias"] = float(metrics.get(f"{name}_abs_bias", 0.0))
            row[f"{name}_raw_mean"] = float(metrics.get(f"{name}_raw_mean", 0.0))
            row[f"{name}_pred_mean"] = float(metrics.get(f"{name}_pred_mean", 0.0))
            row[f"{name}_uncalibrated_raw_accuracy"] = float(
                metrics.get(f"{name}_uncalibrated_raw_accuracy", row[f"{name}_raw_accuracy"])
            )
            row[f"{name}_uncalibrated_bias"] = float(metrics.get(f"{name}_uncalibrated_bias", row[f"{name}_bias"]))
            row[f"{name}_bias_calibration_shift"] = float(metrics.get(f"{name}_bias_calibration_shift", 0.0))
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _save_raw_metric_plot(plot_dir: str, finger_names: Sequence[str], history_path: Optional[str] = None) -> Optional[str]:
    history_path = history_path or os.path.join(plot_dir, "raw_prediction_metrics.csv")
    if not os.path.isfile(history_path):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows = []
    with open(history_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None

    epochs = np.array([int(float(r["epoch"])) for r in rows], dtype=np.int32)
    plt.figure(figsize=(10, 5))
    overall = np.array([float(r.get("prediction_mae", 0.0)) for r in rows], dtype=np.float32)
    y_series = [overall]
    plt.plot(epochs, overall, color="black", linewidth=2.2, marker="o", markersize=4, label="overall MAE")
    if len(epochs) > 0:
        plt.annotate(
            f"{overall[-1]:.1f}",
            xy=(epochs[-1], overall[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9, fontweight="bold", color="black",
            va="center",
        )
    for name in finger_names:
        key = f"{name}_mae"
        if key in rows[0]:
            vals = np.array([float(r.get(key, 0.0)) for r in rows], dtype=np.float32)
            y_series.append(vals)
            color = FINGER_COLORS.get(name)
            plt.plot(
                epochs,
                vals,
                linewidth=1.4,
                alpha=0.8,
                marker="o",
                markersize=3,
                label=name,
                color=color,
            )
            if len(epochs) > 0:
                plt.annotate(
                    f"{vals[-1]:.1f}",
                    xy=(epochs[-1], vals[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=8, color=color,
                    va="center",
                )
    plt.title("Raw tactile MAE vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Raw MAE")
    if len(epochs) > 1:
        x_pad = (epochs[-1] - epochs[0]) * 0.08
        plt.xlim(0, epochs[-1] + max(x_pad, 5))
    elif len(epochs) == 1:
        plt.xlim(0, max(int(epochs[0]), 1))
    if y_series:
        y_min, y_max = _shared_ylim(y_series)
        plt.ylim(max(0.0, y_min), y_max)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(plot_dir, "raw_prediction_mae.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def _epoch_from_profile_filename(filename: str) -> Optional[int]:
    stem, _ = os.path.splitext(filename)
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return None


def _profile_accuracy_rows_from_csvs(plot_dir: str, finger_names: Sequence[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not os.path.isdir(plot_dir):
        return rows
    for filename in sorted(os.listdir(plot_dir)):
        if not filename.startswith("tactile_profile_epoch_") or not filename.endswith(".csv"):
            continue
        epoch_from_name = _epoch_from_profile_filename(filename)
        by_finger = {name: {"raw": [], "pred": [], "active": False} for name in finger_names}
        path = os.path.join(plot_dir, filename)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("finger")
                if name not in by_finger:
                    continue
                active = int(_safe_float(row.get("active"), 1.0)) == 1
                if not active:
                    continue
                by_finger[name]["active"] = True
                by_finger[name]["raw"].append(
                    _safe_float(
                        row.get("raw_tactile_avg"),
                        _safe_float(row.get("raw_tactile"), 0.0),
                    )
                )
                by_finger[name]["pred"].append(
                    _safe_float(
                        row.get("pred_self_touch_avg"),
                        _safe_float(row.get("pred_self_touch_raw"), 0.0),
                    )
                )
                if epoch_from_name is None:
                    epoch_from_name = int(_safe_float(row.get("epoch"), 0.0))
        if epoch_from_name is None:
            continue
        out_row: Dict[str, float] = {"epoch": float(epoch_from_name)}
        acc_values = []
        for name, values in by_finger.items():
            if not values["active"] or not values["raw"]:
                continue
            acc = prediction_accuracy(
                np.asarray(values["raw"], dtype=np.float32),
                np.asarray(values["pred"], dtype=np.float32),
                scale=PROFILE_ACCURACY_SCALE,
            )
            out_row[f"{name}_accuracy"] = acc
            out_row[f"{name}_profile_accuracy"] = acc
            acc_values.append(acc)
        if acc_values:
            out_row["prediction_accuracy"] = float(np.mean(acc_values))
            out_row["prediction_profile_accuracy"] = float(np.mean(acc_values))
            rows.append(out_row)
    return rows


def _save_prediction_accuracy_plot(plot_dir: str, finger_names: Sequence[str], history_path: Optional[str] = None) -> Optional[str]:
    history_path = history_path or os.path.join(plot_dir, "raw_prediction_metrics.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows: List[Dict[str, float]] = []
    if os.path.isfile(history_path):
        with open(history_path, newline="") as f:
            history_rows = list(csv.DictReader(f))
        has_accuracy = any(
            np.isfinite(_safe_float(row.get("prediction_raw_accuracy")))
            or np.isfinite(_safe_float(row.get("prediction_accuracy")))
            or any(np.isfinite(_safe_float(row.get(f"{name}_raw_accuracy"))) for name in finger_names)
            for row in history_rows
        )
        if has_accuracy:
            rows = history_rows
    if not rows:
        return None

    epochs = np.array([int(_safe_float(r.get("epoch"), 0.0)) for r in rows], dtype=np.int32)
    overall_key = "prediction_raw_accuracy" if any(
        np.isfinite(_safe_float(r.get("prediction_raw_accuracy"))) for r in rows
    ) else "prediction_accuracy"
    overall = np.array([_safe_float(r.get(overall_key)) for r in rows], dtype=np.float32)
    if not np.isfinite(overall).any():
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        overall,
        color="black",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="overall raw taxel acc",
    )
    if len(epochs) > 0 and np.isfinite(overall[-1]):
        plt.annotate(
            f"{overall[-1]:.1f}%",
            xy=(epochs[-1], overall[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="black",
            va="center",
        )

    y_series = [overall]
    for name in finger_names:
        raw_key = f"{name}_raw_accuracy"
        key = raw_key if any(np.isfinite(_safe_float(r.get(raw_key))) for r in rows) else f"{name}_accuracy"
        vals = np.array([_safe_float(r.get(key)) for r in rows], dtype=np.float32)
        if not np.isfinite(vals).any():
            continue
        y_series.append(vals)
        color = FINGER_COLORS.get(name)
        plt.plot(
            epochs,
            vals,
            linewidth=1.4,
            alpha=0.8,
            marker="o",
            markersize=3,
            label=name,
            color=color,
        )
        if len(epochs) > 0 and np.isfinite(vals[-1]):
            plt.annotate(
                f"{vals[-1]:.1f}%",
                xy=(epochs[-1], vals[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                va="center",
            )

    plt.title("Raw taxel accuracy vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("raw taxel accuracy (%)")
    if len(epochs) > 1:
        x_pad = (epochs[-1] - epochs[0]) * 0.08
        plt.xlim(0, epochs[-1] + max(x_pad, 5))
    elif len(epochs) == 1:
        plt.xlim(0, max(int(epochs[0]), 1))
    finite_values = np.concatenate([vals[np.isfinite(vals)] for vals in y_series if np.isfinite(vals).any()])
    y_min = max(0.0, float(np.min(finite_values)) - 5.0)
    plt.ylim(y_min, 100.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(plot_dir, "raw_prediction_accuracy.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_tactile_temporal_profiles(
    *,
    data: Mapping,
    preds: Mapping[str, object],
    epoch: int,
    plot_dir: str,
    dataset_param: Mapping,
    combinations: Optional[Sequence[str]],
    finger_names: Sequence[str] = FINGER_ORDER,
    finger_keys: Optional[Sequence[str]] = None,
    next_step: bool = True,
    bias_calibration=False,
    bias_calibration_fingers: Optional[Sequence[str]] = None,
) -> Dict[str, Dict]:
    """Save raw-value tactile profile, raw-value CSV, and raw metric history plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped tactile profile plot.")
        return {"images": {}, "metrics": {}, "files": {}}

    os.makedirs(plot_dir, exist_ok=True)
    active = included_fingers_from_combinations(combinations)
    use_bias_calibration = _as_bool(bias_calibration, default=False)
    calibrated_fingers = _finger_set(bias_calibration_fingers, finger_names)
    scaling_param = load_scaling_param(dataset_param)
    keys = list(finger_keys) if finger_keys is not None else [FINGER_TO_KEY[name] for name in finger_names]
    ref_batch, ref_raw_steps, ref_dim = _fallback_raw_shape(
        data,
        preds,
        finger_names,
        keys,
        next_step=next_step,
    )

    profiles: List[Dict] = []
    for name, key in zip(finger_names, keys):
        raw_arr = _array_from_mapping(data, key)
        pred_arr = _array_from_mapping(preds, name)
        has_raw = raw_arr is not None
        has_pred = pred_arr is not None

        if raw_arr is None:
            if pred_arr is not None and pred_arr.ndim >= 3:
                raw_steps = int(pred_arr.shape[1] + 1) if next_step else int(pred_arr.shape[1])
                raw_arr = _zero_raw_array(
                    batch=int(pred_arr.shape[0]),
                    raw_steps=raw_steps,
                    dim=int(pred_arr.shape[-1]),
                )
            else:
                raw_arr = _zero_raw_array(
                    batch=ref_batch,
                    raw_steps=ref_raw_steps,
                    dim=ref_dim,
                )
        if pred_arr is None:
            pred_arr = _zero_pred_for_raw(raw_arr, next_step=next_step)

        raw_scaled, pred_scaled, timesteps = align_next_step_prediction(
            raw_arr,
            pred_arr,
            next_step=next_step,
        )
        is_active = name in active

        if is_active:
            if has_raw:
                raw_cmp = maybe_unscale(raw_scaled, key, dataset_param, scaling_param)
            else:
                raw_cmp = np.zeros_like(raw_scaled)

            if has_pred:
                pred_scaled = np.nan_to_num(
                    np.asarray(pred_scaled, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                pred_cmp = maybe_unscale(
                    pred_scaled,
                    key,
                    dataset_param,
                    scaling_param,
                )
            else:
                pred_cmp = np.zeros_like(raw_cmp)
        else:
            steps = len(timesteps)
            raw_cmp = np.zeros((raw_scaled.shape[0], steps, raw_scaled.shape[-1]), dtype=np.float32)
            pred_cmp = np.zeros_like(raw_cmp)

        pred_uncalibrated_cmp = np.asarray(pred_cmp, dtype=np.float32).copy()
        uncalibrated_abs_error = np.abs(raw_cmp - pred_uncalibrated_cmp)
        uncalibrated_signed_error = raw_cmp - pred_uncalibrated_cmp
        uncalibrated_accuracy = prediction_accuracy(raw_cmp, pred_uncalibrated_cmp)
        uncalibrated_bias = float(np.mean(uncalibrated_signed_error)) if uncalibrated_signed_error.size else 0.0
        calibration_shift = 0.0
        if (
            use_bias_calibration
            and is_active
            and has_raw
            and has_pred
            and name.lower() in calibrated_fingers
        ):
            pred_cmp, calibration_shift = _mean_bias_calibrated_prediction(raw_cmp, pred_cmp)

        raw_profile = temporal_profile(raw_cmp)
        pred_profile_raw = temporal_profile(pred_cmp)
        pred_uncalibrated_profile = temporal_profile(pred_uncalibrated_cmp)
        timesteps, raw_profile, pred_profile_raw = _truncate_profile_window(
            timesteps, raw_profile, pred_profile_raw
        )
        _, pred_uncalibrated_profile = _truncate_profile_window(
            np.arange(pred_uncalibrated_profile.shape[0]), pred_uncalibrated_profile
        )
        pred_profile_start_aligned, display_start_shift = _start_aligned_prediction(
            raw_profile, pred_profile_raw
        )
        pred_profile = pred_profile_raw
        residual_profile = raw_profile - pred_profile
        err_profile = np.abs(residual_profile)
        abs_error = np.abs(raw_cmp - pred_cmp)
        signed_error = raw_cmp - pred_cmp
        raw_mae = float(np.mean(abs_error))
        raw_rmse = rmse(raw_cmp, pred_cmp)
        display_mae = float(np.mean(np.abs(raw_profile - pred_profile)))
        raw_profile_mae = float(np.mean(np.abs(raw_profile - pred_profile_raw)))
        start_aligned_profile_mae = float(np.mean(np.abs(raw_profile - pred_profile_start_aligned)))
        profile_accuracy = prediction_accuracy(
            raw_profile,
            pred_profile,
            scale=PROFILE_ACCURACY_SCALE,
        )
        raw_accuracy = prediction_accuracy(raw_cmp, pred_cmp)
        corr = safe_corr(raw_cmp, pred_cmp)
        r2 = r2_score(raw_cmp, pred_cmp)
        error_p95 = float(np.percentile(abs_error, 95)) if abs_error.size else 0.0
        bias = float(np.mean(signed_error)) if signed_error.size else 0.0
        profile_bias = float(np.mean(residual_profile)) if residual_profile.size else 0.0
        abs_bias = abs(bias)
        raw_mean = float(np.mean(raw_cmp)) if raw_cmp.size else 0.0
        pred_mean = float(np.mean(pred_cmp)) if pred_cmp.size else 0.0
        profiles.append(
            {
                "name": name,
                "key": key,
                "active": is_active,
                "timesteps": timesteps,
                "raw": raw_profile,
                "pred": pred_profile,
                "pred_raw": pred_profile_raw,
                "pred_uncalibrated": pred_uncalibrated_profile,
                "pred_start_aligned": pred_profile_start_aligned,
                "residual": residual_profile,
                "err": err_profile,
                "mae": raw_mae,
                "profile_mae": raw_profile_mae,
                "display_profile_mae": display_mae,
                "start_aligned_profile_mae": start_aligned_profile_mae,
                "profile_accuracy": profile_accuracy,
                "accuracy": raw_accuracy,
                "rmse": raw_rmse,
                "corr": corr,
                "r2": r2,
                "error_p95": error_p95,
                "bias": bias,
                "profile_bias": profile_bias,
                "abs_bias": abs_bias,
                "raw_mean": raw_mean,
                "pred_mean": pred_mean,
                "uncalibrated_raw_accuracy": uncalibrated_accuracy,
                "uncalibrated_bias": uncalibrated_bias,
                "uncalibrated_abs_error": float(np.mean(uncalibrated_abs_error)) if uncalibrated_abs_error.size else 0.0,
                "bias_calibration_shift": calibration_shift,
                "display_start_shift": display_start_shift,
                "has_raw": has_raw,
                "has_pred": has_pred,
            }
        )

    images: Dict[str, str] = {}
    files: Dict[str, str] = {}
    metrics: Dict[str, float] = {}
    if not profiles:
        return {"images": images, "metrics": metrics, "files": files}

    fig, axes = plt.subplots(
        len(profiles),
        1,
        figsize=(16, 3.25 * len(profiles)),
        sharex=True,
        constrained_layout=True,
    )
    if len(profiles) == 1:
        axes = [axes]
    csv_path = os.path.join(plot_dir, f"tactile_profile_epoch_{epoch:04d}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "finger", "active", "timestep",
            "raw_tactile_avg", "pred_self_touch_avg", "residual_raw_minus_pred",
            "raw_tactile", "pred_self_touch_raw", "pred_self_touch_start_aligned",
            "display_start_shift", "bias_calibration_shift", "pred_self_touch_uncalibrated_avg",
            "error_margin_abs",
        ])
        for ax, profile in zip(axes, profiles):
            name = profile["name"]
            ts = profile["timesteps"]
            raw = profile["raw"]
            pred = profile["pred"]
            pred_raw = profile["pred_raw"]
            pred_uncalibrated = profile["pred_uncalibrated"]
            pred_start_aligned = profile["pred_start_aligned"]
            residual = profile["residual"]
            err = profile["err"]
            color = FINGER_COLORS.get(name, "black")
            draw_tactile_prediction_profile(ax, ts, raw, pred, err, color)
            title_name = name.capitalize()
            if not profile["active"]:
                title_name = f"{title_name} (excluded)"
            shift_text = ""
            if abs(float(profile["bias_calibration_shift"])) > 1e-6:
                shift_text = f" | shift={profile['bias_calibration_shift']:.2f}"
            ax.set_title(
                f"{title_name} | raw_acc={profile['accuracy']:.1f}% | bias={profile['bias']:.2f}{shift_text}",
                fontsize=9,
            )
            ax.set_ylabel("tactile value")
            ylim = _tactile_ylim([raw, pred])
            ax.set_ylim(*ylim)
            if ylim == (TACTILE_YMIN, TACTILE_YMAX):
                ax.set_yticks(TACTILE_YTICKS)
            ax.set_xlabel("Timestep")
            ax.tick_params(axis="x", which="both", labelbottom=True)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            for (
                timestep,
                raw_value,
                pred_raw_value,
                pred_aligned_value,
                pred_uncalibrated_value,
                residual_value,
                err_value,
            ) in zip(
                ts, raw, pred_raw, pred_start_aligned, pred_uncalibrated, residual, err
            ):
                writer.writerow([
                    int(epoch),
                    name,
                    int(profile["active"]),
                    int(timestep),
                    float(raw_value),
                    float(pred_raw_value),
                    float(residual_value),
                    float(raw_value),
                    float(pred_raw_value),
                    float(pred_aligned_value),
                    float(profile["display_start_shift"]),
                    float(profile["bias_calibration_shift"]),
                    float(pred_uncalibrated_value),
                    float(err_value),
                ])
            metrics[f"{name}_mae"] = profile["mae"]
            metrics[f"{name}_profile_mae"] = profile["profile_mae"]
            metrics[f"{name}_display_profile_mae"] = profile["display_profile_mae"]
            metrics[f"{name}_start_aligned_profile_mae"] = profile["start_aligned_profile_mae"]
            metrics[f"{name}_profile_accuracy"] = profile["profile_accuracy"]
            metrics[f"{name}_accuracy"] = profile["accuracy"]
            metrics[f"{name}_raw_accuracy"] = profile["accuracy"]
            metrics[f"{name}_rmse"] = profile["rmse"]
            metrics[f"{name}_corr"] = profile["corr"]
            metrics[f"{name}_r2"] = profile["r2"]
            metrics[f"{name}_error_p95"] = profile["error_p95"]
            metrics[f"{name}_bias"] = profile["bias"]
            metrics[f"{name}_profile_bias"] = profile["profile_bias"]
            metrics[f"{name}_abs_bias"] = profile["abs_bias"]
            metrics[f"{name}_raw_mean"] = profile["raw_mean"]
            metrics[f"{name}_pred_mean"] = profile["pred_mean"]
            metrics[f"{name}_uncalibrated_raw_accuracy"] = profile["uncalibrated_raw_accuracy"]
            metrics[f"{name}_uncalibrated_bias"] = profile["uncalibrated_bias"]
            metrics[f"{name}_bias_calibration_shift"] = profile["bias_calibration_shift"]
            metrics[f"{name}_active"] = float(profile["active"])

    for ax in axes:
        ax.set_xlim(0, TACTILE_XMAX)
    fig.suptitle(TACTILE_PLOT_TITLE, fontsize=12)
    plot_path = os.path.join(plot_dir, f"tactile_profile_epoch_{epoch:04d}.png")
    plt.savefig(plot_path, dpi=160)
    plt.close()
    images["tactile_profile"] = plot_path
    files["tactile_profile_csv"] = csv_path

    residual_fig, residual_axes = plt.subplots(
        len(profiles),
        1,
        figsize=(16, 2.4 * len(profiles)),
        sharex=True,
        constrained_layout=True,
    )
    if len(profiles) == 1:
        residual_axes = [residual_axes]
    for ax, profile in zip(residual_axes, profiles):
        name = profile["name"]
        color = FINGER_COLORS.get(name, "black")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
        ax.plot(
            profile["timesteps"],
            profile["residual"],
            color=color,
            linewidth=1.6,
            label=f"{name} raw-pred",
        )
        ax.set_ylabel("residual")
        ax.set_title(
            f"{name.capitalize()} residual | mean={profile['profile_bias']:.2f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in residual_axes:
        ax.set_xlim(0, TACTILE_XMAX)
        ax.set_xlabel("Timestep")
    residual_path = os.path.join(plot_dir, f"tactile_residual_epoch_{epoch:04d}.png")
    residual_fig.suptitle("Raw tactile minus predicted self-touch", fontsize=12)
    residual_fig.savefig(residual_path, dpi=160)
    plt.close(residual_fig)
    images["tactile_residual"] = residual_path

    metric_profiles = [p for p in profiles if p["active"] and p["has_raw"] and p["has_pred"]]
    if not metric_profiles:
        metric_profiles = [p for p in profiles if p["has_raw"] and p["has_pred"]]
    if not metric_profiles:
        metric_profiles = [p for p in profiles if p["active"]] or profiles
    metrics["tactile_line_mae"] = float(np.mean([p["mae"] for p in metric_profiles]))
    metrics["tactile_line_profile_mae"] = float(np.mean([p["profile_mae"] for p in metric_profiles]))
    metrics["tactile_line_display_profile_mae"] = float(np.mean([p["display_profile_mae"] for p in metric_profiles]))
    metrics["tactile_line_start_aligned_profile_mae"] = float(
        np.mean([p["start_aligned_profile_mae"] for p in metric_profiles])
    )
    metrics["tactile_line_profile_accuracy"] = float(np.mean([p["profile_accuracy"] for p in metric_profiles]))
    metrics["tactile_line_raw_accuracy"] = float(np.mean([p["accuracy"] for p in metric_profiles]))
    metrics["tactile_line_accuracy"] = metrics["tactile_line_raw_accuracy"]
    metrics["tactile_line_rmse"] = float(np.mean([p["rmse"] for p in metric_profiles]))
    metrics["tactile_line_corr"] = float(np.mean([p["corr"] for p in metric_profiles]))
    metrics["tactile_line_r2"] = float(np.mean([p["r2"] for p in metric_profiles]))
    metrics["tactile_line_error_p95"] = float(np.mean([p["error_p95"] for p in metric_profiles]))
    metrics["tactile_line_bias"] = float(np.mean([p["bias"] for p in metric_profiles]))
    metrics["tactile_line_profile_bias"] = float(np.mean([p["profile_bias"] for p in metric_profiles]))
    metrics["tactile_line_abs_bias"] = float(np.mean([p["abs_bias"] for p in metric_profiles]))
    metrics["tactile_line_raw_mean"] = float(np.mean([p["raw_mean"] for p in metric_profiles]))
    metrics["tactile_line_pred_mean"] = float(np.mean([p["pred_mean"] for p in metric_profiles]))
    metrics["tactile_line_uncalibrated_raw_accuracy"] = float(
        np.mean([p["uncalibrated_raw_accuracy"] for p in metric_profiles])
    )
    metrics["tactile_line_uncalibrated_bias"] = float(np.mean([p["uncalibrated_bias"] for p in metric_profiles]))
    metrics["tactile_line_bias_calibration_shift_abs"] = float(
        np.mean([abs(p["bias_calibration_shift"]) for p in metric_profiles])
    )

    history_metrics = dict(metrics)
    history_path = _append_raw_metric_history(plot_dir, epoch, history_metrics, [p["name"] for p in profiles])
    files["raw_prediction_metrics"] = history_path
    metric_plot_path = _save_raw_metric_plot(plot_dir, [p["name"] for p in profiles], history_path)
    if metric_plot_path:
        images["raw_prediction_mae"] = metric_plot_path
    accuracy_plot_path = _save_prediction_accuracy_plot(plot_dir, [p["name"] for p in profiles], history_path)
    if accuracy_plot_path:
        images["raw_prediction_accuracy"] = accuracy_plot_path

    return {"images": images, "metrics": metrics, "files": files}


def split_concatenated_tactile(arr: np.ndarray, finger_names: Sequence[str]) -> Dict[str, np.ndarray]:
    output = {}
    for idx, name in enumerate(finger_names):
        start = idx * TAXELS_PER_FINGER
        end = start + TAXELS_PER_FINGER
        if arr.shape[-1] >= end:
            output[name] = arr[..., start:end]
    return output


def plot_concatenated_tactile_profile(
    *,
    save_dir: str,
    epoch: int,
    tactile_target: np.ndarray,
    tactile_pred: np.ndarray,
    combinations: Optional[Sequence[str]],
    finger_names: Sequence[str],
) -> Dict[str, Dict]:
    target = np.nan_to_num(np.asarray(tactile_target, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.nan_to_num(np.asarray(tactile_pred, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    target_parts = split_concatenated_tactile(target, finger_names)
    pred_parts = split_concatenated_tactile(pred, finger_names)
    if target.ndim == 3 and pred.ndim == 2:
        pred_parts = {name: np.repeat(part[:, None, :], target.shape[1], axis=1) for name, part in pred_parts.items()}
    pseudo_data = {FINGER_TO_KEY[name]: target_parts[name] for name in target_parts}
    pseudo_preds = {name: pred_parts[name] for name in target_parts if name in pred_parts}
    return plot_tactile_temporal_profiles(
        data=pseudo_data,
        preds=pseudo_preds,
        epoch=epoch,
        plot_dir=save_dir,
        dataset_param={"modality": {}},
        combinations=combinations,
        finger_names=list(finger_names),
        finger_keys=[FINGER_TO_KEY[name] for name in finger_names],
        next_step=False,
    )
