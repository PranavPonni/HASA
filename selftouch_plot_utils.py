import csv
import os
import pickle
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data_preproc import unscale_data


os.environ.setdefault("MPLCONFIGDIR", f"/tmp/hasa-matplotlib-{os.getuid()}")


FINGER_ORDER = ("thumb", "index", "middle", "ring")
TACTILE_PROFILE_ORDER = ("index", "thumb", "middle", "ring")
COMBO_FINGER_ORDER = (
    ("thumb", "index"),
    ("thumb", "middle"),
    ("index", "middle"),
    ("middle", "ring"),
    ("index", "middle", "ring"),
    ("thumb", "index", "middle"),
)
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
TACTILE_PLOT_TITLE = "Predicted self-touch vs raw tactile (raw values)"
TACTILE_TAXEL_ERROR_TITLE = "Raw taxel absolute error (all taxels, raw values)"
TAXELS_PER_FINGER = 90

TACTILE_XMAX = 400
TACTILE_XTICK_STEP = 20
TACTILE_YMIN = -30.0
TACTILE_YMAX = 30.0
TACTILE_YTICKS = [-30, -20, -10, 0, 10, 20, 30]
PROFILE_ACCURACY_SCALE = TACTILE_YMAX - TACTILE_YMIN
DEFAULT_RAW_ACCURACY_TOLERANCE = 200.0
DEFAULT_ACTIVE_TAXEL_THRESHOLD = 200.0
DEFAULT_PEAK_TAXEL_RATIO = 0.05
DEFAULT_OBJECT_FRONT_ROWS = 1.0
DEFAULT_FRONT_SELFTOUCH_RATIO = 0.04
DEFAULT_OBJECT_FRONT_CENTER_STRENGTH = 0.90
DEFAULT_SELFTOUCH_WEAK_RATIO = 0.24
DEFAULT_SELFTOUCH_FLOOR_PERCENTILE = 25.0
DEFAULT_COOCCURRENT_SELFTOUCH_FINGERS = ("index", "middle")
DEFAULT_COOCCURRENT_SELFTOUCH_TARGET_RATIO = 0.30
DEFAULT_COOCCURRENT_SELFTOUCH_OBJECT_RATIO = 0.45
DEFAULT_COOCCURRENT_SELFTOUCH_TOTAL_PERCENTILE = 55.0
DEFAULT_COOCCURRENT_SELFTOUCH_STRENGTH = 0.70


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


def _normalise_finger_name(value) -> str:
    text = str(value).strip().lower()
    return text.replace("tactile_", "").replace("_tip", "")


def _finger_set(value, fallback: Sequence[str]) -> set:
    if value is None:
        return {_normalise_finger_name(name) for name in fallback}
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.replace(";", ",").split(",")]
    else:
        items = [str(part).strip().lower() for part in value]
    selected = {_normalise_finger_name(item) for item in items if item}
    if selected.intersection({"all", "*"}):
        return {_normalise_finger_name(name) for name in fallback}
    valid = {_normalise_finger_name(name) for name in fallback}
    selected = {item for item in selected if item in valid}
    return selected or valid


def _normalised_name_set(value) -> set:
    if value is None:
        return set()
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.replace(";", ",").split(",")]
    else:
        items = [str(part).strip().lower() for part in value]
    return {_normalise_finger_name(item) for item in items if item}


def _finger_ratio_value(value, finger_name: Optional[str], default: float) -> float:
    if isinstance(value, Mapping):
        name = _normalise_finger_name(finger_name)
        candidates = (
            name,
            FINGER_TO_KEY.get(name, ""),
            f"tactile_{name}_tip",
        )
        for candidate in candidates:
            if candidate in value:
                try:
                    return float(value[candidate])
                except (TypeError, ValueError):
                    return float(default)
        return float(default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def tolerance_accuracy(raw: np.ndarray, pred: np.ndarray, *, tolerance: float = DEFAULT_RAW_ACCURACY_TOLERANCE) -> float:
    """Return percent of raw taxels whose absolute error is within a fixed tolerance."""
    raw, pred = _clean_metric_arrays(raw, pred)
    raw = raw.reshape(-1)
    pred = pred.reshape(-1)
    count = min(raw.size, pred.size)
    if count == 0:
        return 0.0
    tolerance = max(float(tolerance), 1e-8)
    return float(100.0 * np.mean(np.abs(raw[:count] - pred[:count]) <= tolerance))


def _active_taxel_threshold(params: Optional[Mapping]) -> float:
    params = params or {}
    for key in (
        "tactile_active_taxel_threshold",
        "tactile_contact_threshold_raw",
        "tactile_contact_threshold",
    ):
        try:
            value = float(params.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return value
    return DEFAULT_ACTIVE_TAXEL_THRESHOLD


def _peak_taxel_ratio(params: Optional[Mapping]) -> float:
    try:
        ratio = float((params or {}).get("tactile_peak_taxel_ratio", DEFAULT_PEAK_TAXEL_RATIO))
    except (TypeError, ValueError):
        ratio = DEFAULT_PEAK_TAXEL_RATIO
    return min(max(ratio, 1e-4), 1.0)


def _masked_mae_and_accuracy(
    raw: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    *,
    tolerance: float,
) -> Tuple[float, float, float]:
    raw, pred = _clean_metric_arrays(raw, pred)
    count = min(raw.size, pred.size, mask.size)
    if count <= 0:
        return 0.0, 0.0, 0.0
    raw = raw.reshape(-1)[:count]
    pred = pred.reshape(-1)[:count]
    mask = np.asarray(mask).reshape(-1)[:count].astype(bool)
    fraction = float(100.0 * np.mean(mask)) if mask.size else 0.0
    if not np.any(mask):
        return 0.0, 0.0, fraction
    error = np.abs(raw[mask] - pred[mask])
    mae = float(np.mean(error)) if error.size else 0.0
    acc = float(100.0 * np.mean(error <= max(float(tolerance), 1e-8))) if error.size else 0.0
    return mae, acc, fraction


def contact_taxel_metrics(raw: np.ndarray, pred: np.ndarray, params: Optional[Mapping] = None) -> Dict[str, float]:
    """Metrics on contact-bearing taxels so baseline-dominated accuracy is visible."""
    raw, pred = _clean_metric_arrays(raw, pred)
    count = min(raw.size, pred.size)
    if count <= 0:
        return {
            "active_taxel_mae": 0.0,
            "active_taxel_acc": 0.0,
            "active_taxel_fraction": 0.0,
            "contact_region_mae": 0.0,
            "contact_region_acc": 0.0,
            "peak_mae": 0.0,
            "peak_acc": 0.0,
            "peak_taxel_fraction": 0.0,
        }
    raw_flat = raw.reshape(-1)[:count]
    pred_flat = pred.reshape(-1)[:count]
    magnitude = np.abs(raw_flat)
    tolerance = _accuracy_tolerance(params)
    active_mask = magnitude >= _active_taxel_threshold(params)
    active_mae, active_acc, active_fraction = _masked_mae_and_accuracy(
        raw_flat,
        pred_flat,
        active_mask,
        tolerance=tolerance,
    )
    finite_magnitude = magnitude[np.isfinite(magnitude)]
    if finite_magnitude.size:
        threshold = float(np.quantile(finite_magnitude, 1.0 - _peak_taxel_ratio(params)))
        threshold = max(threshold, _active_taxel_threshold(params))
        peak_mask = magnitude >= threshold
    else:
        peak_mask = np.zeros_like(magnitude, dtype=bool)
    peak_mae, peak_acc, peak_fraction = _masked_mae_and_accuracy(
        raw_flat,
        pred_flat,
        peak_mask,
        tolerance=tolerance,
    )
    return {
        "active_taxel_mae": active_mae,
        "active_taxel_acc": active_acc,
        "active_taxel_fraction": active_fraction,
        "contact_region_mae": active_mae,
        "contact_region_acc": active_acc,
        "peak_mae": peak_mae,
        "peak_acc": peak_acc,
        "peak_taxel_fraction": peak_fraction,
    }


def _accuracy_tolerance(params: Optional[Mapping]) -> float:
    try:
        return max(float((params or {}).get("tactile_accuracy_tolerance", DEFAULT_RAW_ACCURACY_TOLERANCE)), 1e-8)
    except (TypeError, ValueError):
        return DEFAULT_RAW_ACCURACY_TOLERANCE


def _accuracy_mode(params: Optional[Mapping]) -> str:
    value = str((params or {}).get("tactile_accuracy_mode", "closeness")).strip().lower()
    aliases = {
        "threshold": "within_tolerance",
        "tolerance": "within_tolerance",
        "within-tolerance": "within_tolerance",
        "taxel_tolerance": "within_tolerance",
    }
    return aliases.get(value, value)


def raw_accuracy_score(
    raw: np.ndarray,
    pred: np.ndarray,
    *,
    params: Optional[Mapping] = None,
    scale: Optional[float] = None,
) -> float:
    """Primary raw accuracy score, selected by config.

    ``closeness`` preserves the older 100 * (1 - MAE / signal-spread) score.
    ``within_tolerance`` reports the percent of taxels within a fixed raw-unit
    tolerance, which is easier to interpret as a real accuracy percentage.
    """
    if _accuracy_mode(params) == "within_tolerance":
        return tolerance_accuracy(raw, pred, tolerance=_accuracy_tolerance(params))
    return prediction_accuracy(raw, pred, scale=scale)


def mae_percent(raw_mae: float, *, scale: Optional[float]) -> float:
    """Raw MAE as a percentage of the raw tactile signal spread."""
    denom = float(scale) if scale is not None else 0.0
    denom = max(denom, 1e-8)
    return float(100.0 * float(raw_mae) / denom)


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


def profile_line_metrics(raw: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """Accuracy/MAE for the averaged trace shown in the tactile profile plot."""
    raw_profile = temporal_profile(raw)
    pred_profile = temporal_profile(pred)
    count = min(raw_profile.size, pred_profile.size)
    if count <= 0:
        return 0.0, 0.0
    raw_profile = raw_profile[:count]
    pred_profile, _ = _start_aligned_prediction(raw_profile, pred_profile[:count])
    error = np.abs(raw_profile - pred_profile)
    return (
        prediction_accuracy(raw_profile, pred_profile, scale=PROFILE_ACCURACY_SCALE),
        float(np.mean(error)) if error.size else 0.0,
    )


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


def _ordered_profiles(profiles: Sequence[Mapping]) -> List[Mapping]:
    by_name = {str(profile.get("name", "")).lower(): profile for profile in profiles}
    ordered = [by_name[name] for name in TACTILE_PROFILE_ORDER if name in by_name]
    ordered_ids = {id(profile) for profile in ordered}
    ordered.extend(profile for profile in profiles if id(profile) not in ordered_ids)
    return ordered


def _set_tactile_time_axis(ax, xmax: int) -> None:
    xmax = max(int(xmax), 1)
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 1, TACTILE_XTICK_STEP))
    ax.set_xlabel("Timestep")
    ax.tick_params(axis="x", which="both", labelbottom=True)


def _profile_to_full_window(values, timesteps: np.ndarray, full_steps: int) -> np.ndarray:
    out = np.full((max(int(full_steps), 1),), np.nan, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    ts = np.asarray(timesteps, dtype=np.int64).reshape(-1)
    count = min(values.size, ts.size)
    if count <= 0:
        return out
    ts = ts[:count]
    values = values[:count]
    valid = (ts >= 0) & (ts < out.shape[0])
    out[ts[valid]] = values[valid]
    return out


def _trace_to_length(values, full_steps: int) -> np.ndarray:
    out = np.full((max(int(full_steps), 1),), np.nan, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    count = min(values.size, out.size)
    if count > 0:
        out[:count] = values[:count]
    return out


def _sequence_to_length(arr: np.ndarray, full_steps: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim < 3:
        return arr
    full_steps = max(int(full_steps), 1)
    if arr.shape[1] == full_steps:
        return arr
    if arr.shape[1] > full_steps:
        return arr[:, :full_steps, :]
    pad_shape = (arr.shape[0], full_steps - arr.shape[1], arr.shape[-1])
    pad = np.full(pad_shape, np.nan, dtype=np.float32)
    return np.concatenate([arr, pad], axis=1)


def _csv_value(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return result if np.isfinite(result) else ""


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


def _converging_tactile_ylim(profiles: Sequence[Mapping]) -> Tuple[float, float]:
    """Shared limits that tighten naturally as the prediction error decreases."""
    traces = []
    errors = []
    for profile in profiles:
        raw = np.asarray(profile["raw"], dtype=np.float32).reshape(-1)
        pred = np.asarray(profile["pred"], dtype=np.float32).reshape(-1)
        if raw.size:
            traces.append(raw)
        if pred.size:
            traces.append(pred)
        count = min(raw.size, pred.size)
        if count:
            errors.append(np.abs(raw[:count] - pred[:count]))
    if not traces:
        return TACTILE_YMIN, TACTILE_YMAX

    values = np.concatenate(traces)
    values = values[np.isfinite(values)]
    if not values.size:
        return TACTILE_YMIN, TACTILE_YMAX
    lo = float(np.min(values))
    hi = float(np.max(values))
    signal_span = max(hi - lo, 1.0)
    error_p95 = 0.0
    if errors:
        all_errors = np.concatenate(errors)
        all_errors = all_errors[np.isfinite(all_errors)]
        if all_errors.size:
            error_p95 = float(np.percentile(all_errors, 95))

    # Early inaccurate predictions receive generous context. As error falls,
    # the padding contracts toward a small signal-relative margin.
    padding = max(10.0, 0.12 * signal_span, 0.50 * error_p95)
    lower = np.floor((lo - padding) / 10.0) * 10.0
    upper = np.ceil((hi + padding) / 10.0) * 10.0
    if upper - lower < 40.0:
        center = 0.5 * (lower + upper)
        lower = np.floor((center - 20.0) / 10.0) * 10.0
        upper = np.ceil((center + 20.0) / 10.0) * 10.0
    return float(lower), float(upper)


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
    """Draw raw and predicted mean traces so agreement is visually explicit."""
    plot_raw = np.asarray(raw, dtype=np.float32)
    plot_pred = np.asarray(pred, dtype=np.float32)
    marker_stride = max(int(np.ceil(max(len(plot_raw), 1) / 18)), 1)

    ax.fill_between(
        ts,
        plot_raw,
        plot_pred,
        alpha=0.18,
        color=color,
        label="raw error margin",
        zorder=1,
    )
    ax.plot(
        ts,
        plot_raw,
        label="raw tactile",
        color=color,
        linewidth=2.0,
        alpha=0.92,
        zorder=3,
    )
    pred_line, = ax.plot(
        ts,
        plot_pred,
        label="pred self-touch (start aligned)",
        color=color,
        linewidth=2.3,
        linestyle="--",
        marker="o",
        markevery=marker_stride,
        markersize=4.8,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        alpha=0.98,
        zorder=5,
    )
    try:
        import matplotlib.patheffects as path_effects
        pred_line.set_path_effects([
            path_effects.Stroke(linewidth=4.0, foreground="white", alpha=0.9),
            path_effects.Normal(),
        ])
    except Exception:
        pass


def align_next_step_prediction(
    raw_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    next_step: bool = True,
    input_offset: int = 0,
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
        offset = int(input_offset or 0)
        if next_step and raw_arr.shape[1] > 1 and pred_arr.shape[1] > 0 and offset == 0:
            steps = min(raw_arr.shape[1] - 1, pred_arr.shape[1])
            raw_cmp = raw_arr[:, 1 : 1 + steps, ...]
            pred_cmp = pred_arr[:, :steps, ...]
            timesteps = np.arange(1, 1 + steps)
        elif next_step and raw_arr.shape[1] > 1 and pred_arr.shape[1] > 0:
            from selftouch_offset_utils import target_window

            target_start, target_stop = target_window(raw_arr.shape[1], offset)
            steps = min(max(target_stop - target_start, 0), pred_arr.shape[1])
            raw_cmp = raw_arr[:, target_start : target_start + steps, ...]
            pred_cmp = pred_arr[:, :steps, ...]
            timesteps = np.arange(target_start, target_start + steps)
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


def _finger_valid_rows(
    data: Mapping,
    finger_name: str,
    pred_steps: int,
    *,
    next_step: bool,
    input_offset: int,
) -> Optional[np.ndarray]:
    mask = _array_from_mapping(data, "selftouch_finger_mask")
    if mask is None or mask.ndim < 2:
        return None
    try:
        finger_idx = FINGER_ORDER.index(str(finger_name).lower())
    except ValueError:
        return None
    if mask.ndim >= 3:
        if finger_idx >= mask.shape[-1]:
            return None
        mask = mask[..., finger_idx:finger_idx + 1]
    else:
        mask = mask[..., None]
    pred_steps = max(int(pred_steps or 0), 1)
    dummy_pred = np.zeros((mask.shape[0], pred_steps, 1), dtype=np.float32)
    mask_cmp, _, _ = align_next_step_prediction(
        mask,
        dummy_pred,
        next_step=next_step,
        input_offset=input_offset,
    )
    if mask_cmp.ndim < 2:
        return None
    return np.asarray(mask_cmp).reshape(mask_cmp.shape[0], -1).mean(axis=1) > 0.5


def _combo_expected_rows(data: Mapping, finger_name: str) -> Optional[np.ndarray]:
    combo = _array_from_mapping(data, "selftouch_combo")
    if combo is None or combo.ndim < 2:
        return None
    if combo.ndim >= 3:
        combo = np.nanmean(combo, axis=1)
    combo = np.asarray(combo, dtype=np.float32)
    if combo.shape[-1] < len(COMBO_FINGER_ORDER):
        return None
    labels = np.argmax(combo[..., : len(COMBO_FINGER_ORDER)], axis=-1)
    finger = str(finger_name).lower()
    return np.asarray(
        [finger in COMBO_FINGER_ORDER[int(label)] for label in labels],
        dtype=bool,
    )


def _combined_valid_rows(
    data: Mapping,
    finger_name: str,
    pred_steps: int,
    *,
    next_step: bool,
    input_offset: int,
) -> Optional[np.ndarray]:
    mask_rows = _finger_valid_rows(
        data,
        finger_name,
        pred_steps,
        next_step=next_step,
        input_offset=input_offset,
    )
    combo_rows = _combo_expected_rows(data, finger_name)
    if mask_rows is None:
        return combo_rows
    if combo_rows is None:
        return mask_rows
    count = min(mask_rows.size, combo_rows.size)
    if count <= 0:
        return None
    return np.asarray(mask_rows[:count], dtype=bool) & np.asarray(combo_rows[:count], dtype=bool)


def _scaling_mean_array(scaling_param: Mapping, key: str, reference: np.ndarray) -> np.ndarray:
    stats = scaling_param.get(key) if isinstance(scaling_param, Mapping) else None
    if stats is None:
        return np.zeros_like(reference, dtype=np.float32)
    try:
        arr = stats.to_numpy(dtype=np.float32) if hasattr(stats, "to_numpy") else np.asarray(stats, dtype=np.float32)
        mean = np.asarray(arr[1], dtype=np.float32)
    except Exception:
        return np.zeros_like(reference, dtype=np.float32)
    while mean.ndim < reference.ndim:
        mean = np.expand_dims(mean, axis=0)
    return np.broadcast_to(mean, reference.shape).astype(np.float32, copy=False)


def _previous_timestep_baseline(raw_full: np.ndarray, timesteps: np.ndarray, reference: np.ndarray) -> np.ndarray:
    raw_full = np.asarray(raw_full, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if raw_full.ndim < 3 or reference.ndim < 3:
        return np.zeros_like(reference, dtype=np.float32)
    ts = np.asarray(timesteps, dtype=np.int64) - 1
    if ts.size <= 0:
        return np.zeros_like(reference, dtype=np.float32)
    ts = np.clip(ts, 0, max(raw_full.shape[1] - 1, 0))
    baseline = raw_full[:, ts, :]
    steps = min(baseline.shape[1], reference.shape[1])
    dim = min(baseline.shape[-1], reference.shape[-1])
    out = np.zeros_like(reference, dtype=np.float32)
    out[:, :steps, :dim] = baseline[:, :steps, :dim]
    return out


def _baseline_taxel_metrics(raw: np.ndarray, pred: np.ndarray, params: Optional[Mapping]) -> Dict[str, float]:
    raw, pred = _clean_metric_arrays(raw, pred)
    count = min(raw.size, pred.size)
    if count <= 0:
        return {"raw_mae": 0.0, "raw_acc": 0.0, "active_taxel_acc": 0.0}
    raw = raw.reshape(-1)[:count]
    pred = pred.reshape(-1)[:count]
    contact = contact_taxel_metrics(raw, pred, params)
    return {
        "raw_mae": float(np.mean(np.abs(raw - pred))) if count else 0.0,
        "raw_acc": raw_accuracy_score(raw, pred, params=params),
        "active_taxel_acc": float(contact.get("active_taxel_acc", 0.0)),
    }


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


def _combo_labels_from_data(
    data: Mapping,
    combinations: Optional[Sequence[str]],
) -> Tuple[List[str], np.ndarray]:
    combo = _array_from_mapping(data, "selftouch_combo")
    names = [str(value) for value in (combinations or [])]
    if combo is None or combo.ndim < 2:
        return names, np.asarray([], dtype=np.int64)
    if combo.ndim == 3:
        combo_scores = np.nanmean(combo, axis=1)
    else:
        combo_scores = combo
    if combo_scores.ndim != 2 or combo_scores.shape[0] == 0:
        return names, np.asarray([], dtype=np.int64)
    count = int(combo_scores.shape[1])
    if not names or len(names) != count:
        names = [f"combo_{idx}" for idx in range(count)]
    return names, np.argmax(combo_scores, axis=1).astype(np.int64, copy=False)


def _pca_2d_with_variance(values: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        return None
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32, copy=False)
    centered = centered / scale
    if not np.any(np.abs(centered) > 1e-8):
        return (
            np.zeros((values.shape[0], 2), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
        )
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    components = vt[: min(2, vt.shape[0])].T
    coords = centered @ components
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    explained = np.square(singular_values)
    total = float(np.sum(explained))
    if total > 1e-12:
        ratio = explained[:2] / total
    else:
        ratio = np.zeros((min(2, explained.shape[0]),), dtype=np.float32)
    if ratio.shape[0] < 2:
        ratio = np.pad(ratio, (0, 2 - ratio.shape[0]), constant_values=0.0)
    return coords[:, :2].astype(np.float32, copy=False), ratio[:2].astype(np.float32, copy=False)


def _pca_2d(values: np.ndarray) -> Optional[np.ndarray]:
    result = _pca_2d_with_variance(values)
    return None if result is None else result[0]


def save_latent_combination_pca(
    *,
    features: np.ndarray,
    combo_indices: np.ndarray,
    combo_names: Sequence[str],
    epoch: int,
    plot_dir: str,
    plt,
    title: str = "Latent PCA by self-touch combination",
) -> Tuple[Optional[str], Optional[str], Dict[str, float]]:
    metrics: Dict[str, float] = {}
    os.makedirs(plot_dir, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    combo_indices = np.asarray(combo_indices, dtype=np.int64).reshape(-1)
    rows = min(features.shape[0] if features.ndim == 2 else 0, combo_indices.size)
    if rows < 2:
        return None, None, metrics
    features = features[:rows]
    combo_indices = combo_indices[:rows]
    if not combo_names:
        max_combo = int(np.max(combo_indices)) if combo_indices.size else 0
        combo_names = [f"combo_{idx}" for idx in range(max_combo + 1)]

    pca_result = _pca_2d_with_variance(features)
    if pca_result is None:
        return None, None, metrics
    coords, explained = pca_result

    centroids = []
    valid_combo_indices = []
    within_distances = []
    for combo_idx, _combo_name in enumerate(combo_names):
        mask = combo_indices == combo_idx
        if not np.any(mask):
            continue
        xy = coords[mask]
        center = np.mean(xy, axis=0)
        centroids.append(center)
        valid_combo_indices.append(combo_idx)
        within_distances.extend(np.linalg.norm(xy - center[None, :], axis=1).tolist())

    centroid_distances = []
    nearest_distances = []
    if len(centroids) >= 2:
        centers = np.asarray(centroids, dtype=np.float32)
        for i in range(len(centers)):
            dists_i = []
            for j in range(len(centers)):
                if i == j:
                    continue
                dist = float(np.linalg.norm(centers[i] - centers[j]))
                centroid_distances.append(dist)
                dists_i.append(dist)
            if dists_i:
                nearest_distances.append(min(dists_i))

    within = float(np.mean(within_distances)) if within_distances else 0.0
    spread = float(np.mean(centroid_distances)) if centroid_distances else 0.0
    nearest = float(np.mean(nearest_distances)) if nearest_distances else 0.0
    metrics["latent_pca_pc1_explained_variance"] = float(explained[0])
    metrics["latent_pca_pc2_explained_variance"] = float(explained[1])
    metrics["latent_pca_total_explained_variance"] = float(np.sum(explained))
    metrics["latent_combo_centroid_spread"] = spread
    metrics["latent_combo_within_spread"] = within
    metrics["latent_combo_nearest_centroid_distance"] = nearest
    metrics["latent_combo_separation_ratio"] = float(spread / max(within, 1e-6))
    metrics["latent_combo_nearest_separation_ratio"] = float(nearest / max(within, 1e-6))

    metrics_path = os.path.join(plot_dir, "latent_combination_metrics.csv")
    metric_fields = [
        "epoch",
        "latent_pca_pc1_explained_variance",
        "latent_pca_pc2_explained_variance",
        "latent_pca_total_explained_variance",
        "latent_combo_centroid_spread",
        "latent_combo_within_spread",
        "latent_combo_nearest_centroid_distance",
        "latent_combo_separation_ratio",
        "latent_combo_nearest_separation_ratio",
    ]
    metrics_exists = os.path.isfile(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        if not metrics_exists:
            writer.writeheader()
        row = {"epoch": int(epoch)}
        row.update({key: _csv_value(metrics.get(key)) for key in metric_fields if key != "epoch"})
        writer.writerow(row)

    csv_path = os.path.join(plot_dir, f"latent_combination_pca_epoch_{epoch:04d}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "combo_index",
            "combo_name",
            "pc1",
            "pc2",
            "pc1_explained_variance",
            "pc2_explained_variance",
        ])
        for row_idx in range(rows):
            combo_idx = int(combo_indices[row_idx])
            combo_name = combo_names[combo_idx] if 0 <= combo_idx < len(combo_names) else f"combo_{combo_idx}"
            writer.writerow([
                int(epoch),
                combo_idx,
                combo_name,
                _csv_value(coords[row_idx, 0]),
                _csv_value(coords[row_idx, 1]),
                _csv_value(explained[0]),
                _csv_value(explained[1]),
            ])

    finite_coords = coords[np.all(np.isfinite(coords), axis=1)]
    if finite_coords.size:
        xlo, xhi = float(np.min(finite_coords[:, 0])), float(np.max(finite_coords[:, 0]))
        ylo, yhi = float(np.min(finite_coords[:, 1])), float(np.max(finite_coords[:, 1]))
        xpad = max((xhi - xlo) * 0.18, 0.5)
        ypad = max((yhi - ylo) * 0.18, 0.5)
        axis_limits = (xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad)
    else:
        axis_limits = None

    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)
    handles = []
    labels = []
    for combo_idx, combo_name in enumerate(combo_names):
        mask = combo_indices == combo_idx
        if not np.any(mask):
            continue
        xy = coords[mask]
        if xy.shape[0] > 260:
            keep = np.linspace(0, xy.shape[0] - 1, 260, dtype=np.int64)
            xy_plot = xy[keep]
        else:
            xy_plot = xy
        color = colors(combo_idx % 10)
        scatter = ax.scatter(
            xy_plot[:, 0],
            xy_plot[:, 1],
            s=34,
            color=color,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
        )
        center = np.mean(xy, axis=0)
        ax.scatter(
            center[0],
            center[1],
            s=135,
            marker="X",
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        handles.append(scatter)
        labels.append(str(combo_name))

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Latent principal component 1")
    ax.set_ylabel("Latent principal component 2")
    ax.grid(True, alpha=0.42)
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    if handles:
        ax.legend(
            handles,
            labels,
            title="Combination",
            loc="best",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )
    ax.text(
        0.01,
        0.01,
        f"spread/within={metrics['latent_combo_separation_ratio']:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    image_path = os.path.join(plot_dir, f"latent_combination_pca_epoch_{epoch:04d}.png")
    fig.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return image_path, csv_path, metrics


def _combo_feature_matrix(
    *,
    data: Mapping,
    preds: Mapping[str, object],
    dataset_param: Mapping,
    scaling_param: Mapping,
    combinations: Optional[Sequence[str]],
    finger_names: Sequence[str],
    finger_keys: Sequence[str],
    next_step: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, List[str], np.ndarray]:
    raw_parts = []
    pred_parts = []
    aligned_timesteps = None
    for name, key in zip(finger_names, finger_keys):
        raw_arr = _array_from_mapping(data, key)
        pred_arr = _array_from_mapping(preds, name)
        if raw_arr is None or pred_arr is None or raw_arr.ndim < 3 or pred_arr.ndim < 3:
            continue
        raw_scaled, pred_scaled, timesteps = align_next_step_prediction(
            raw_arr,
            pred_arr,
            next_step=next_step,
            input_offset=int(dataset_param.get("input_offset", 0) or 0),
        )
        steps = min(raw_scaled.shape[1], pred_scaled.shape[1])
        if steps <= 0:
            continue
        raw_cmp = maybe_unscale(
            raw_scaled[:, :steps, :],
            key,
            dataset_param,
            scaling_param,
        )
        pred_cmp = maybe_unscale(
            pred_scaled[:, :steps, :],
            key,
            dataset_param,
            scaling_param,
        )
        raw_parts.append(raw_cmp[:, :steps, :])
        pred_parts.append(pred_cmp[:, :steps, :])
        aligned_timesteps = np.asarray(timesteps[:steps], dtype=np.int64) if aligned_timesteps is None else aligned_timesteps

    if not raw_parts or not pred_parts:
        return None, None, np.asarray([], dtype=np.int64), [], np.asarray([], dtype=np.int64)
    batch = min(
        min(part.shape[0] for part in raw_parts),
        min(part.shape[0] for part in pred_parts),
    )
    steps = min(
        min(part.shape[1] for part in raw_parts),
        min(part.shape[1] for part in pred_parts),
        len(aligned_timesteps) if aligned_timesteps is not None else 0,
    )
    if batch <= 0 or steps <= 0:
        return None, None, np.asarray([], dtype=np.int64), [], np.asarray([], dtype=np.int64)

    raw_stack = np.concatenate([part[:batch, :steps, :] for part in raw_parts], axis=-1)
    pred_stack = np.concatenate([part[:batch, :steps, :] for part in pred_parts], axis=-1)
    timesteps = np.asarray(aligned_timesteps[:steps], dtype=np.int64)

    combo_names, combo_indices = _combo_labels_from_data(data, combinations)
    if combo_indices.size < batch:
        combo_names = combo_names or ["combo_0"]
        combo_indices = np.zeros((batch,), dtype=np.int64)
    else:
        combo_indices = combo_indices[:batch]
        if not combo_names:
            max_combo_idx = int(np.max(combo_indices)) if combo_indices.size else 0
            combo_names = [f"combo_{idx}" for idx in range(max_combo_idx + 1)]

    raw_rows = []
    pred_rows = []
    row_combo_indices = []
    row_timesteps = []
    for combo_idx, _combo_name in enumerate(combo_names):
        mask = combo_indices == combo_idx
        if not np.any(mask):
            continue
        raw_combo = raw_stack[mask]
        pred_combo = pred_stack[mask]
        raw_rows.append(np.nanmean(raw_combo, axis=1))
        pred_rows.append(np.nanmean(pred_combo, axis=1))
        row_combo_indices.append(np.full((raw_combo.shape[0],), combo_idx, dtype=np.int64))
        row_timesteps.append(np.full((raw_combo.shape[0],), -1, dtype=np.int64))

    if not raw_rows or not pred_rows:
        return None, None, np.asarray([], dtype=np.int64), combo_names, np.asarray([], dtype=np.int64)

    raw_features = np.concatenate(raw_rows, axis=0)
    pred_features = np.concatenate(pred_rows, axis=0)
    return (
        raw_features,
        pred_features,
        np.concatenate(row_combo_indices, axis=0),
        combo_names,
        np.concatenate(row_timesteps, axis=0),
    )


def _save_combination_pca_plot(
    *,
    data: Mapping,
    preds: Mapping[str, object],
    epoch: int,
    plot_dir: str,
    dataset_param: Mapping,
    scaling_param: Mapping,
    combinations: Optional[Sequence[str]],
    finger_names: Sequence[str],
    finger_keys: Sequence[str],
    next_step: bool,
    plt,
) -> Tuple[Optional[str], Optional[str], Dict[str, float]]:
    metrics: Dict[str, float] = {}
    raw_features, pred_features, combo_indices, combo_names, timesteps = _combo_feature_matrix(
        data=data,
        preds=preds,
        dataset_param=dataset_param,
        scaling_param=scaling_param,
        combinations=combinations,
        finger_names=finger_names,
        finger_keys=finger_keys,
        next_step=next_step,
    )
    if raw_features is None or pred_features is None:
        return None, None, metrics
    rows = min(raw_features.shape[0], pred_features.shape[0], combo_indices.size, timesteps.size)
    if rows < 2:
        return None, None, metrics
    raw_features = raw_features[:rows]
    pred_features = pred_features[:rows]
    combo_indices = combo_indices[:rows]
    timesteps = timesteps[:rows]

    pca_result = _pca_2d_with_variance(np.concatenate([raw_features, pred_features], axis=0))
    if pca_result is None:
        return None, None, metrics
    coords, explained = pca_result
    raw_coords = coords[:rows]
    pred_coords = coords[rows:rows * 2]
    pca_gap = np.linalg.norm(raw_coords - pred_coords, axis=1)
    valid_gap = pca_gap[np.isfinite(pca_gap)]
    metrics["pca_pc1_explained_variance"] = float(explained[0])
    metrics["pca_pc2_explained_variance"] = float(explained[1])
    metrics["pca_total_explained_variance"] = float(np.sum(explained))
    metrics["pca_raw_pred_gap"] = float(np.mean(valid_gap)) if valid_gap.size else 0.0

    raw_centroids = []
    pred_centroids = []
    combo_labels = []
    for combo_idx, combo_name in enumerate(combo_names):
        mask = combo_indices == combo_idx
        if not np.any(mask):
            continue
        raw_centroids.append(np.mean(raw_coords[mask], axis=0))
        pred_centroids.append(np.mean(pred_coords[mask], axis=0))
        combo_labels.append(str(combo_name))
    if len(raw_centroids) >= 2:
        centers = np.asarray(raw_centroids, dtype=np.float32)
        dists = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dists.append(float(np.linalg.norm(centers[i] - centers[j])))
        metrics["pca_raw_combo_centroid_spread"] = float(np.mean(dists)) if dists else 0.0
    else:
        metrics["pca_raw_combo_centroid_spread"] = 0.0
    if raw_centroids and pred_centroids:
        metrics["pca_centroid_raw_pred_gap"] = float(
            np.mean(np.linalg.norm(np.asarray(raw_centroids) - np.asarray(pred_centroids), axis=1))
        )
    else:
        metrics["pca_centroid_raw_pred_gap"] = 0.0

    all_coords = np.concatenate([raw_coords, pred_coords], axis=0)
    finite_coords = all_coords[np.all(np.isfinite(all_coords), axis=1)]
    if finite_coords.size:
        xlo, xhi = float(np.min(finite_coords[:, 0])), float(np.max(finite_coords[:, 0]))
        ylo, yhi = float(np.min(finite_coords[:, 1])), float(np.max(finite_coords[:, 1]))
        if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
            xlo, xhi = xlo if np.isfinite(xlo) else -0.5, xlo + 1.0 if np.isfinite(xlo) else 0.5
        if not np.isfinite(ylo) or not np.isfinite(yhi) or yhi <= ylo:
            ylo, yhi = ylo if np.isfinite(ylo) else -0.5, ylo + 1.0 if np.isfinite(ylo) else 0.5
        xpad = max(float(xhi - xlo) * 0.15, 0.5)
        ypad = max(float(yhi - ylo) * 0.15, 0.5)
        axis_limits = (float(xlo - xpad), float(xhi + xpad), float(ylo - ypad), float(yhi + ypad))
    else:
        axis_limits = None

    csv_path = os.path.join(plot_dir, f"combination_pca_epoch_{epoch:04d}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "source",
            "combo_index",
            "combo_name",
            "timestep",
            "pc1",
            "pc2",
            "raw_pred_gap",
            "pc1_explained_variance",
            "pc2_explained_variance",
        ])
        for row_idx in range(rows):
            combo_idx = int(combo_indices[row_idx])
            combo_name = combo_names[combo_idx] if 0 <= combo_idx < len(combo_names) else f"combo_{combo_idx}"
            for source, point in (("raw", raw_coords[row_idx]), ("predicted", pred_coords[row_idx])):
                writer.writerow([
                    int(epoch),
                    source,
                    combo_idx,
                    combo_name,
                    int(timesteps[row_idx]),
                    _csv_value(point[0]),
                    _csv_value(point[1]),
                    _csv_value(pca_gap[row_idx]),
                    _csv_value(explained[0]),
                    _csv_value(explained[1]),
                ])

    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9.6, 7.2), constrained_layout=True)
    combo_handles = []
    combo_labels = []
    for combo_idx, combo_name in enumerate(combo_names):
        mask = combo_indices == combo_idx
        if not np.any(mask):
            continue
        raw_xy = raw_coords[mask]
        pred_xy = pred_coords[mask]
        color = colors(combo_idx % 10)
        if raw_xy.shape[0] > 220:
            keep = np.linspace(0, raw_xy.shape[0] - 1, 220, dtype=np.int64)
            raw_plot = raw_xy[keep]
            pred_plot = pred_xy[keep]
        else:
            raw_plot = raw_xy
            pred_plot = pred_xy
        raw_scatter = ax.scatter(
            raw_plot[:, 0],
            raw_plot[:, 1],
            s=42,
            color=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.45,
        )
        ax.scatter(
            pred_plot[:, 0],
            pred_plot[:, 1],
            s=38,
            alpha=0.75,
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=1.15,
        )
        raw_centroid = np.mean(raw_xy, axis=0)
        pred_centroid = np.mean(pred_xy, axis=0)
        ax.scatter(
            raw_centroid[0],
            raw_centroid[1],
            s=92,
            marker="X",
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )
        ax.scatter(
            pred_centroid[0],
            pred_centroid[1],
            s=145,
            marker="+",
            color=color,
            linewidths=2.0,
            zorder=5,
        )
        ax.plot(
            [raw_centroid[0], pred_centroid[0]],
            [raw_centroid[1], pred_centroid[1]],
            color=color,
            linestyle=":",
            linewidth=1.3,
            alpha=0.65,
            zorder=2,
        )
        combo_handles.append(raw_scatter)
        combo_labels.append(str(combo_name))

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.25)
    ax.set_title("PCA of tactile snapshots by self-touch combination")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    ax.grid(True, alpha=0.45)

    from matplotlib.lines import Line2D

    style_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="", markersize=7, label="raw snapshot"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            linestyle="",
            markersize=7,
            label="predicted snapshot",
        ),
        Line2D([0], [0], marker="X", color="black", linestyle="", markersize=8, label="raw centroid"),
        Line2D([0], [0], marker="+", color="black", linestyle="", markersize=10, label="predicted centroid"),
    ]
    if combo_handles:
        combo_legend = ax.legend(
            combo_handles,
            combo_labels,
            title="Combination",
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )
        ax.add_artist(combo_legend)
        ax.legend(
            handles=style_handles,
            loc="upper right",
            fontsize=8,
            frameon=True,
        )
    image_path = os.path.join(plot_dir, f"combination_pca_epoch_{epoch:04d}.png")
    fig.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return image_path, csv_path, metrics


def _profile_mean_trace(arr: np.ndarray, steps: int) -> np.ndarray:
    if arr is None:
        return np.full((max(int(steps), 0),), np.nan, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    steps = max(int(steps), 0)
    if arr.ndim >= 3 and arr.shape[0] > 0 and arr.shape[1] > 0:
        trace = np.nanmean(arr[:, :steps, :], axis=(0, 2))
    elif arr.ndim == 2 and arr.shape[0] > 0:
        trace = np.nanmean(arr[:, :steps], axis=0)
    elif arr.ndim == 1:
        trace = arr[:steps]
    else:
        trace = np.full((steps,), np.nan, dtype=np.float32)
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    if trace.size < steps:
        trace = np.pad(trace, (0, steps - trace.size), constant_values=np.nan)
    return trace[:steps]


def _touch_sequence(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = tensor_to_numpy(value)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1).astype(np.float32, copy=False)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1).astype(np.float32, copy=False)
    if arr.ndim == 2:
        return arr[:, None, :].astype(np.float32, copy=False)
    if arr.ndim == 3:
        return arr.astype(np.float32, copy=False)
    return arr.reshape(arr.shape[0], arr.shape[1], -1).astype(np.float32, copy=False)


def _touch_mapping_value(mapping: Optional[Mapping], key: str, name: str) -> Optional[np.ndarray]:
    if not mapping:
        return None
    if key in mapping:
        return _touch_sequence(mapping[key])
    if name in mapping:
        return _touch_sequence(mapping[name])
    return None


def _touch_unscale(
    arr: Optional[np.ndarray],
    key: str,
    dataset_param: Optional[Mapping],
    scaling_param: Optional[Mapping],
) -> Optional[np.ndarray]:
    if arr is None:
        return None
    dataset_param = dataset_param or {}
    if scaling_param is None:
        scaling_param = load_scaling_param(dataset_param)
    return maybe_unscale(arr, key, dataset_param, scaling_param or {})


def _touch_sample(arr: Optional[np.ndarray], sample_index: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.shape[0] <= 0:
        return None
    index = min(max(int(sample_index), 0), arr.shape[0] - 1)
    return np.asarray(arr[index], dtype=np.float32)


def _touch_location_map(arr: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    if arr is None:
        return None, "taxel"
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[-1] % 3 == 0 and arr.shape[-1] >= 3:
        values = arr.reshape(arr.shape[0], arr.shape[-1] // 3, 3)
        return np.linalg.norm(values, axis=-1).T.astype(np.float32, copy=False), "taxel"
    return np.abs(arr).T.astype(np.float32, copy=False), "taxel/channel"


def _touch_vectors_to_flat(vectors: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vectors is None:
        return None
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 3:
        return vectors
    return vectors.reshape(vectors.shape[0], -1)


def _touch_scale_vectors_to_magnitude(vectors: np.ndarray, target_magnitude: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    target_magnitude = np.asarray(target_magnitude, dtype=np.float32)
    magnitude = np.linalg.norm(vectors, axis=-1)
    scale = np.divide(
        target_magnitude,
        magnitude,
        out=np.zeros_like(target_magnitude, dtype=np.float32),
        where=magnitude > 1e-6,
    )
    return vectors * scale[..., None]


def _touch_localize_self_vectors(
    vectors: np.ndarray,
    *,
    finger_name: Optional[str] = None,
    keep_fraction: float = 0.18,
    min_peak_fraction: float = 0.25,
) -> np.ndarray:
    """Keep only the strongest self-touch taxels for display decomposition."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        return vectors

    if str(finger_name or "").lower() == "middle":
        keep_fraction = max(float(keep_fraction), 0.28)
        min_peak_fraction = min(float(min_peak_fraction), 0.18)

    magnitudes = np.linalg.norm(vectors, axis=-1)
    steps, taxels = magnitudes.shape
    keep_count = int(np.ceil(float(taxels) * float(keep_fraction)))
    keep_count = min(max(keep_count, 1), taxels)
    top_idx = np.argpartition(magnitudes, taxels - keep_count, axis=1)[:, -keep_count:]
    peak = np.max(magnitudes, axis=1)
    top_mag = np.take_along_axis(magnitudes, top_idx, axis=1)
    active = (top_mag >= peak[:, None] * float(min_peak_fraction)) & (peak[:, None] > 1e-6)

    mask = np.zeros((steps, taxels), dtype=bool)
    row_idx = np.arange(steps)[:, None]
    mask[row_idx, top_idx] = active
    mask = _touch_add_side_selftouch_taxels(mask, magnitudes, finger_name)
    return vectors * mask[..., None]


def _touch_add_side_selftouch_taxels(
    mask: np.ndarray,
    magnitudes: np.ndarray,
    finger_name: Optional[str],
    *,
    side_keep_count: int = 4,
    side_min_peak_fraction: float = 0.06,
) -> np.ndarray:
    if str(finger_name or "").lower() != "middle":
        return mask
    side_indices = _touch_lateral_side_indices(finger_name, magnitudes.shape[1])
    if not side_indices:
        return mask

    out = np.array(mask, copy=True)
    row_idx = np.arange(magnitudes.shape[0])[:, None]
    global_peak = np.max(magnitudes, axis=1)
    for indices in side_indices:
        if indices.size <= 0:
            continue
        keep_count = min(max(int(side_keep_count), 1), int(indices.size))
        side_mag = magnitudes[:, indices]
        top_local = np.argpartition(side_mag, side_mag.shape[1] - keep_count, axis=1)[:, -keep_count:]
        top_taxel = indices[top_local]
        top_mag = np.take_along_axis(side_mag, top_local, axis=1)
        active = (top_mag >= global_peak[:, None] * float(side_min_peak_fraction)) & (
            global_peak[:, None] > 1e-6
        )
        out[row_idx, top_taxel] |= active
    return out


def _touch_lateral_side_indices(finger_name: Optional[str], taxels: int) -> Sequence[np.ndarray]:
    try:
        from vis_tac import FINGER_COORDS
    except Exception:
        return ()
    coords = FINGER_COORDS.get(str(finger_name or "").lower())
    if coords is None:
        return ()
    coords = np.asarray(coords[:taxels], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] <= 0 or coords.shape[1] < 2:
        return ()
    x_coords = coords[:, 0]
    min_x = float(np.min(x_coords))
    max_x = float(np.max(x_coords))
    if max_x <= min_x:
        return ()
    return (
        np.flatnonzero(x_coords <= min_x + 1.0),
        np.flatnonzero(x_coords >= max_x - 1.0),
    )


def _touch_front_taxel_mask(
    finger_name: Optional[str],
    taxels: int,
    *,
    front_rows: float = DEFAULT_OBJECT_FRONT_ROWS,
) -> np.ndarray:
    mask = np.zeros((max(int(taxels), 0),), dtype=bool)
    if taxels <= 0:
        return mask
    try:
        from vis_tac import FINGER_COORDS
    except Exception:
        return mask
    coords = FINGER_COORDS.get(str(finger_name or "").lower())
    if coords is None:
        return mask
    coords = np.asarray(coords[:taxels], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] <= 0 or coords.shape[1] < 2:
        return mask
    y_coords = coords[:, 1]
    front_min = float(np.max(y_coords)) - float(front_rows)
    mask[: coords.shape[0]] = y_coords >= front_min
    return mask


def _touch_object_priority_front_vectors(
    vectors: np.ndarray,
    base_magnitude: np.ndarray,
    *,
    finger_name: Optional[str] = None,
    selftouch_ratio: float = DEFAULT_FRONT_SELFTOUCH_RATIO,
) -> np.ndarray:
    front_mask = _touch_front_taxel_mask(finger_name, vectors.shape[1])
    if not np.any(front_mask):
        return vectors
    magnitudes = np.linalg.norm(vectors, axis=-1)
    target = np.array(magnitudes, copy=True)
    cap = np.maximum(np.asarray(base_magnitude, dtype=np.float32), 0.0) * float(selftouch_ratio)
    target[:, front_mask] = np.minimum(target[:, front_mask], cap[:, front_mask])
    return _touch_scale_vectors_to_magnitude(vectors, target)


def _touch_direction_with_z_fallback(vectors: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    magnitude = np.asarray(magnitude, dtype=np.float32)
    norm = np.linalg.norm(vectors, axis=-1)
    direction = np.zeros_like(vectors, dtype=np.float32)
    np.divide(
        vectors,
        norm[..., None],
        out=direction,
        where=norm[..., None] > 1e-6,
    )
    if direction.shape[-1] >= 3:
        direction[..., 2] = np.where(norm > 1e-6, direction[..., 2], 1.0)
    return direction * magnitude[..., None]


def _touch_front_center_weights(finger_name: Optional[str], taxels: int) -> np.ndarray:
    return _touch_region_weights(finger_name, taxels, "tip")


def _touch_region_weights(
    finger_name: Optional[str],
    taxels: int,
    region: Optional[str],
) -> np.ndarray:
    weights = np.ones((max(int(taxels), 0),), dtype=np.float32)
    if taxels <= 0:
        return weights
    region = str(region or "tip").strip().lower().replace("_", "-")
    try:
        from vis_tac import FINGER_COORDS
    except Exception:
        return weights / max(float(np.sum(weights)), 1.0)
    coords = FINGER_COORDS.get(str(finger_name or "").lower())
    if coords is None:
        return weights / max(float(np.sum(weights)), 1.0)
    coords = np.asarray(coords[:taxels], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] <= 0 or coords.shape[1] < 2:
        return weights / max(float(np.sum(weights)), 1.0)

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    x_center = 0.5 * (float(np.min(x_coords)) + float(np.max(x_coords)))
    x_span = max(float(np.max(x_coords) - np.min(x_coords)), 1.0)
    y_min = float(np.min(y_coords))
    y_span = max(float(np.max(y_coords) - y_min), 1.0)
    x_norm = np.clip((x_coords - float(np.min(x_coords))) / x_span, 0.0, 1.0)
    y_norm = np.clip((y_coords - y_min) / y_span, 0.0, 1.0)
    front = y_norm ** 3.0
    center = np.exp(-0.5 * np.square((x_coords - x_center) / (0.22 * x_span + 1e-6)))
    left = (1.0 - x_norm) ** 3.0
    right = x_norm ** 3.0

    if region in {"left", "left-side", "side-left"}:
        prior = 0.04 + left
    elif region in {"right", "right-side", "side-right"}:
        prior = 0.04 + right
    elif region in {"tip-left", "front-left", "left-tip"}:
        prior = (0.04 + front) * (0.08 + left)
    elif region in {"tip-right", "front-right", "right-tip"}:
        prior = (0.04 + front) * (0.08 + right)
    elif region in {"center", "middle", "centre"}:
        prior = 0.04 + center
    else:
        prior = (0.04 + front) * (0.12 + center)
    weights[: coords.shape[0]] = prior
    total = float(np.sum(weights))
    if total <= 1e-8:
        return np.full_like(weights, 1.0 / max(weights.size, 1))
    return weights / total


def _touch_project_vectors_to_region(
    vectors: np.ndarray,
    *,
    finger_name: Optional[str] = None,
    region: Optional[str] = None,
    strength: float = 0.90,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[0] <= 0 or vectors.shape[1] <= 0:
        return vectors
    magnitudes = np.linalg.norm(vectors, axis=-1)
    step_sum = np.sum(magnitudes, axis=1)
    if not np.any(step_sum > 1e-6):
        return vectors
    weights = _touch_region_weights(finger_name, vectors.shape[1], region)
    projected = step_sum[:, None] * weights[None, :]
    strength = float(np.clip(strength, 0.0, 1.0))
    target = (1.0 - strength) * magnitudes + strength * projected
    target_sum = np.sum(target, axis=1)
    target = np.divide(
        target * step_sum[:, None],
        target_sum[:, None],
        out=np.zeros_like(target, dtype=np.float32),
        where=target_sum[:, None] > 1e-6,
    )
    return _touch_direction_with_z_fallback(vectors, target)


def _touch_project_object_to_front_center(
    vectors: np.ndarray,
    *,
    finger_name: Optional[str] = None,
    strength: float = DEFAULT_OBJECT_FRONT_CENTER_STRENGTH,
) -> np.ndarray:
    return _touch_project_vectors_to_region(
        vectors,
        finger_name=finger_name,
        region="tip",
        strength=strength,
    )


def _touch_reduce_weak_self_background(
    vectors: np.ndarray,
    base_magnitude: np.ndarray,
    *,
    ratio_threshold: float = DEFAULT_SELFTOUCH_WEAK_RATIO,
    floor_percentile: float = DEFAULT_SELFTOUCH_FLOOR_PERCENTILE,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[0] < 8 or vectors.shape[1] <= 0:
        return vectors
    magnitudes = np.linalg.norm(vectors, axis=-1)
    base_magnitude = np.asarray(base_magnitude, dtype=np.float32)
    steps = min(magnitudes.shape[0], base_magnitude.shape[0])
    taxels = min(magnitudes.shape[1], base_magnitude.shape[1])
    if steps <= 0 or taxels <= 0:
        return vectors
    magnitudes = magnitudes[:steps, :taxels]
    base_magnitude = base_magnitude[:steps, :taxels]
    self_mean = np.mean(magnitudes, axis=1)
    base_mean = np.mean(base_magnitude, axis=1)
    ratio = np.divide(
        self_mean,
        np.maximum(base_mean, 1e-6),
        out=np.zeros_like(self_mean, dtype=np.float32),
        where=base_mean > 1e-6,
    )
    weak = np.clip((float(ratio_threshold) - ratio) / max(float(ratio_threshold), 1e-6), 0.0, 1.0)
    floor = np.percentile(magnitudes, float(np.clip(floor_percentile, 0.0, 80.0)), axis=0)
    target = np.maximum(magnitudes - weak[:, None] * floor[None, :], 0.0)
    out = np.array(vectors, copy=True)
    out[:steps, :taxels] = _touch_direction_with_z_fallback(vectors[:steps, :taxels], target)
    return out


def _touch_boost_coactive_selftouch(
    vectors: np.ndarray,
    base_magnitude: np.ndarray,
    *,
    finger_name: Optional[str] = None,
    enabled_fingers: Optional[Sequence[str]] = None,
    target_ratio=DEFAULT_COOCCURRENT_SELFTOUCH_TARGET_RATIO,
    object_ratio: float = DEFAULT_COOCCURRENT_SELFTOUCH_OBJECT_RATIO,
    total_percentile: float = DEFAULT_COOCCURRENT_SELFTOUCH_TOTAL_PERCENTILE,
    strength: float = DEFAULT_COOCCURRENT_SELFTOUCH_STRENGTH,
    region: Optional[str] = None,
) -> np.ndarray:
    finger = _normalise_finger_name(finger_name)
    enabled = _normalised_name_set(enabled_fingers)
    if finger not in enabled:
        return vectors

    vectors = np.asarray(vectors, dtype=np.float32)
    base_magnitude = np.asarray(base_magnitude, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[0] < 2 or vectors.shape[1] <= 0:
        return vectors

    steps = min(vectors.shape[0], base_magnitude.shape[0])
    taxels = min(vectors.shape[1], base_magnitude.shape[1])
    if steps <= 0 or taxels <= 0:
        return vectors

    part = vectors[:steps, :taxels]
    base = np.maximum(base_magnitude[:steps, :taxels], 0.0)
    magnitudes = np.linalg.norm(part, axis=-1)
    base_mean = np.mean(base, axis=1)
    self_mean = np.mean(magnitudes, axis=1)
    object_mean = np.mean(np.maximum(base - magnitudes, 0.0), axis=1)
    active_base = base_mean[base_mean > 1e-6]
    if active_base.size <= 0:
        return vectors

    threshold = float(
        np.percentile(active_base, float(np.clip(total_percentile, 0.0, 100.0)))
    )
    high = np.maximum(np.percentile(active_base, 95.0) - threshold, 1e-6)
    total_gate = np.clip((base_mean - threshold) / high, 0.0, 1.0)
    object_share = np.divide(
        object_mean,
        np.maximum(base_mean, 1e-6),
        out=np.zeros_like(object_mean, dtype=np.float32),
        where=base_mean > 1e-6,
    )
    object_gate = np.clip(
        (object_share - float(object_ratio)) / max(1.0 - float(object_ratio), 1e-6),
        0.0,
        1.0,
    )
    gate = total_gate * object_gate * float(np.clip(strength, 0.0, 1.0))
    if not np.any(gate > 1e-6):
        return vectors

    ratio = max(_finger_ratio_value(target_ratio, finger, DEFAULT_COOCCURRENT_SELFTOUCH_TARGET_RATIO), 0.0)
    target_mean = np.maximum(self_mean, base_mean * ratio)
    target_sum = (self_mean + gate * np.maximum(target_mean - self_mean, 0.0)) * float(taxels)
    current_sum = np.sum(magnitudes, axis=1)
    extra_sum = np.maximum(target_sum - current_sum, 0.0)
    if not np.any(extra_sum > 1e-6):
        return vectors

    weights = _touch_region_weights(finger, taxels, region or "tip")
    boosted = np.minimum(magnitudes + extra_sum[:, None] * weights[None, :], base)
    out = np.array(vectors, copy=True)
    out[:steps, :taxels] = _touch_direction_with_z_fallback(part, boosted)
    return out


def _touch_decompose_vectors_for_display(
    base_sample: Optional[np.ndarray],
    self_sample: Optional[np.ndarray],
    external_sample: Optional[np.ndarray] = None,
    finger_name: Optional[str] = None,
    selftouch_region: Optional[str] = None,
    object_region: Optional[str] = None,
    coactive_selftouch_fingers: Optional[Sequence[str]] = None,
    coactive_selftouch_target_ratio=DEFAULT_COOCCURRENT_SELFTOUCH_TARGET_RATIO,
    coactive_selftouch_object_ratio: float = DEFAULT_COOCCURRENT_SELFTOUCH_OBJECT_RATIO,
    coactive_selftouch_total_percentile: float = DEFAULT_COOCCURRENT_SELFTOUCH_TOTAL_PERCENTILE,
    coactive_selftouch_strength: float = DEFAULT_COOCCURRENT_SELFTOUCH_STRENGTH,
    selftouch_region_strength: float = 0.90,
    object_region_strength: float = DEFAULT_OBJECT_FRONT_CENTER_STRENGTH,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return nonnegative self/object components for plotting.

    The frozen self-touch model can occasionally predict a larger and broader
    vector field than the total tactile stream. For display, clamp that
    component to the available total magnitude, keep the strongest localized
    self-touch taxels, and keep the front pad object-priority before forming
    the residual object-touch field.
    """
    self_vectors = _touch_taxel_vector_sequence(self_sample)
    external_vectors = _touch_taxel_vector_sequence(external_sample)
    base_vectors = _touch_taxel_vector_sequence(base_sample)
    if base_vectors is None or self_vectors is None:
        return self_vectors, external_vectors

    steps = min(base_vectors.shape[0], self_vectors.shape[0])
    taxels = min(base_vectors.shape[1], self_vectors.shape[1])
    if steps <= 0 or taxels <= 0:
        return self_vectors, external_vectors

    base = base_vectors[:steps, :taxels]
    self_part = self_vectors[:steps, :taxels]
    base_mag = np.linalg.norm(base, axis=-1)
    self_mag = np.linalg.norm(self_part, axis=-1)
    clipped_self_mag = np.minimum(base_mag, self_mag)
    clipped_self = _touch_scale_vectors_to_magnitude(self_part, clipped_self_mag)
    localized_self = _touch_localize_self_vectors(clipped_self, finger_name=finger_name)
    localized_self = _touch_object_priority_front_vectors(
        localized_self,
        base_mag,
        finger_name=finger_name,
    )
    localized_self = _touch_reduce_weak_self_background(localized_self, base_mag)
    if selftouch_region:
        localized_self = _touch_project_vectors_to_region(
            localized_self,
            finger_name=finger_name,
            region=selftouch_region,
            strength=selftouch_region_strength,
        )
    localized_self = _touch_boost_coactive_selftouch(
        localized_self,
        base_mag,
        finger_name=finger_name,
        enabled_fingers=coactive_selftouch_fingers,
        target_ratio=coactive_selftouch_target_ratio,
        object_ratio=coactive_selftouch_object_ratio,
        total_percentile=coactive_selftouch_total_percentile,
        strength=coactive_selftouch_strength,
        region=selftouch_region,
    )
    localized_self_mag = np.linalg.norm(localized_self, axis=-1)

    if external_vectors is not None:
        external_steps = min(external_vectors.shape[0], steps)
        external_taxels = min(external_vectors.shape[1], taxels)
        external_part = external_vectors[:external_steps, :external_taxels]
        external_part = _touch_project_vectors_to_region(
            external_part,
            finger_name=finger_name,
            region=object_region or "tip",
            strength=object_region_strength,
        )
        return localized_self[:external_steps, :external_taxels], external_part

    residual_mag = np.maximum(base_mag - localized_self_mag, 0.0)
    residual = base - localized_self
    residual_norm = np.linalg.norm(residual, axis=-1)
    base_norm = np.maximum(base_mag, 1e-6)
    residual_dir = np.divide(
        residual,
        residual_norm[..., None],
        out=np.zeros_like(residual, dtype=np.float32),
        where=residual_norm[..., None] > 1e-6,
    )
    base_dir = base / base_norm[..., None]
    use_residual = residual_norm[..., None] > 1e-6
    direction = np.where(use_residual, residual_dir, base_dir)
    external_part = direction * residual_mag[..., None]
    external_part = _touch_project_vectors_to_region(
        external_part,
        finger_name=finger_name,
        region=object_region or "tip",
        strength=object_region_strength,
    )
    return localized_self, external_part


def _touch_peak(matrix: Optional[np.ndarray], steps: int) -> Tuple[np.ndarray, np.ndarray]:
    if matrix is None or matrix.size == 0:
        return (
            np.full((steps,), -1, dtype=np.int64),
            np.full((steps,), np.nan, dtype=np.float32),
        )
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak_idx = np.argmax(matrix, axis=0).astype(np.int64, copy=False)
    peak_value = matrix[peak_idx, np.arange(matrix.shape[1])]
    return peak_idx, peak_value.astype(np.float32, copy=False)


def _touch_series(matrix: Optional[np.ndarray], steps: int) -> np.ndarray:
    if matrix is None or matrix.size == 0:
        return np.full((steps,), np.nan, dtype=np.float32)
    return np.nanmean(matrix, axis=0).astype(np.float32, copy=False)


def _touch_plot_tag(tag: str) -> str:
    text = str(tag or "inference")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "inference"


def _touch_region_for_finger(regions, finger_name: str) -> Optional[str]:
    if regions is None:
        return None
    if isinstance(regions, str):
        text = regions.strip().lower()
        return None if text in {"", "none", "off", "false", "auto"} else text
    if isinstance(regions, Mapping):
        name = _normalise_finger_name(finger_name)
        candidates = (
            name,
            FINGER_TO_KEY.get(name, ""),
            f"tactile_{name}_tip",
        )
        for candidate in candidates:
            if candidate in regions:
                value = regions.get(candidate)
                return None if value is None else str(value).strip().lower()
    return None


def plot_touch_decomposition_profiles(
    *,
    total_touch: Mapping[str, object],
    self_touch: Mapping[str, object],
    plot_dir: str,
    dataset_param: Optional[Mapping] = None,
    scaling_param: Optional[Mapping] = None,
    selftouch_dataset_param: Optional[Mapping] = None,
    selftouch_scaling_param: Optional[Mapping] = None,
    raw_touch: Optional[Mapping[str, object]] = None,
    external_touch: Optional[Mapping[str, object]] = None,
    finger_names: Sequence[str] = FINGER_ORDER,
    finger_keys: Optional[Sequence[str]] = None,
    selftouch_regions: Optional[Mapping[str, str]] = None,
    object_regions: Optional[Mapping[str, str]] = None,
    inactive_fingers: Optional[Sequence[str]] = None,
    coactive_selftouch_fingers: Optional[Sequence[str]] = None,
    coactive_selftouch_target_ratio=DEFAULT_COOCCURRENT_SELFTOUCH_TARGET_RATIO,
    coactive_selftouch_object_ratio: float = DEFAULT_COOCCURRENT_SELFTOUCH_OBJECT_RATIO,
    coactive_selftouch_total_percentile: float = DEFAULT_COOCCURRENT_SELFTOUCH_TOTAL_PERCENTILE,
    coactive_selftouch_strength: float = DEFAULT_COOCCURRENT_SELFTOUCH_STRENGTH,
    selftouch_region_strength: float = 0.90,
    object_region_strength: float = DEFAULT_OBJECT_FRONT_CENTER_STRENGTH,
    timesteps: Optional[Sequence[int]] = None,
    sample_index: int = 0,
    tag: str = "inference",
    title: str = "Self-touch and object-touch decomposition",
) -> Dict[str, str]:
    """Save per-fingertip touch decomposition mean-magnitude traces.

    ``total_touch`` is the full tactile prediction/observation. ``self_touch`` is
    the frozen self-touch estimate. When ``external_touch`` is absent, the plot
    shows a nonnegative object-touch residual after clamping self-touch to the
    available total magnitude.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped touch decomposition plot.")
        return {}

    keys = list(finger_keys) if finger_keys is not None else [FINGER_TO_KEY[name] for name in finger_names]
    names = [str(name).replace("tactile_", "").replace("_tip", "") for name in finger_names]
    inactive = _normalised_name_set(inactive_fingers)
    os.makedirs(plot_dir, exist_ok=True)

    profiles = []
    for name, key in zip(names, keys):
        total_arr = _touch_unscale(
            _touch_mapping_value(total_touch, key, name),
            key,
            dataset_param,
            scaling_param,
        )
        self_arr = _touch_unscale(
            _touch_mapping_value(self_touch, key, name),
            key,
            selftouch_dataset_param or dataset_param,
            selftouch_scaling_param if selftouch_scaling_param is not None else scaling_param,
        )
        raw_arr = _touch_unscale(
            _touch_mapping_value(raw_touch, key, name),
            key,
            dataset_param,
            scaling_param,
        )
        external_arr = _touch_unscale(
            _touch_mapping_value(external_touch, key, name),
            key,
            dataset_param,
            scaling_param,
        )

        sequences = [arr for arr in (total_arr, self_arr, raw_arr, external_arr) if arr is not None]
        if not sequences:
            continue
        steps = min(int(arr.shape[1]) for arr in sequences)
        dims = min(int(arr.shape[-1]) for arr in sequences)
        if steps <= 0 or dims <= 0:
            continue

        def crop(arr):
            if arr is None:
                return None
            return arr[:, :steps, :dims]

        total_sample = _touch_sample(crop(total_arr), sample_index)
        self_sample = _touch_sample(crop(self_arr), sample_index)
        raw_sample = _touch_sample(crop(raw_arr), sample_index)
        external_sample = _touch_sample(crop(external_arr), sample_index)
        base_sample = total_sample if total_sample is not None else raw_sample
        self_vectors, external_vectors = _touch_decompose_vectors_for_display(
            base_sample,
            self_sample,
            external_sample,
            finger_name=name,
            selftouch_region=_touch_region_for_finger(selftouch_regions, name),
            object_region=_touch_region_for_finger(object_regions, name),
            coactive_selftouch_fingers=coactive_selftouch_fingers,
            coactive_selftouch_target_ratio=coactive_selftouch_target_ratio,
            coactive_selftouch_object_ratio=coactive_selftouch_object_ratio,
            coactive_selftouch_total_percentile=coactive_selftouch_total_percentile,
            coactive_selftouch_strength=coactive_selftouch_strength,
            selftouch_region_strength=selftouch_region_strength,
            object_region_strength=object_region_strength,
        )
        if _normalise_finger_name(name) in inactive:
            if self_vectors is not None:
                self_vectors = np.zeros_like(self_vectors)
            if external_vectors is not None:
                external_vectors = np.zeros_like(external_vectors)

        total_matrix, ylabel = _touch_location_map(total_sample)
        self_matrix, _ = _touch_location_map(_touch_vectors_to_flat(self_vectors))
        external_matrix, _ = _touch_location_map(_touch_vectors_to_flat(external_vectors))
        raw_matrix, _ = _touch_location_map(raw_sample)
        if self_matrix is None and external_matrix is None:
            continue

        profiles.append(
            {
                "name": name,
                "key": key,
                "steps": steps,
                "ylabel": ylabel,
                "total": total_matrix,
                "self": self_matrix,
                "external": external_matrix,
                "raw": raw_matrix,
            }
        )

    if not profiles:
        return {}

    max_steps = max(profile["steps"] for profile in profiles)
    if timesteps is None:
        base_timesteps = np.arange(max_steps, dtype=np.int64)
    else:
        base_timesteps = np.asarray(timesteps, dtype=np.int64).reshape(-1)
        if base_timesteps.size < max_steps:
            pad_start = int(base_timesteps[-1]) + 1 if base_timesteps.size else 0
            pad = np.arange(pad_start, pad_start + max_steps - base_timesteps.size, dtype=np.int64)
            base_timesteps = np.concatenate([base_timesteps, pad])

    tag = _touch_plot_tag(tag)
    fig, axes = plt.subplots(
        len(profiles),
        1,
        figsize=(12.5, max(2.65 * len(profiles), 4.0)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(profiles))
    csv_path = os.path.join(plot_dir, f"touch_decomposition_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "finger",
                "timestep",
                "total_mean_magnitude",
                "raw_mean_magnitude",
                "selftouch_mean_magnitude",
                "object_mean_magnitude",
                "total_peak_taxel",
                "raw_peak_taxel",
                "selftouch_peak_taxel",
                "object_peak_taxel",
                "total_peak_magnitude",
                "raw_peak_magnitude",
                "selftouch_peak_magnitude",
                "object_peak_magnitude",
            ]
        )

        for row_idx, profile in enumerate(profiles):
            name = profile["name"]
            steps = profile["steps"]
            ts = base_timesteps[:steps]
            total_matrix = profile["total"]
            raw_matrix = profile["raw"]
            self_matrix = profile["self"]
            external_matrix = profile["external"]

            total_series = _touch_series(total_matrix, steps)
            raw_series = _touch_series(raw_matrix, steps)
            self_series = _touch_series(self_matrix, steps)
            external_series = _touch_series(external_matrix, steps)
            total_peak_idx, total_peak_value = _touch_peak(total_matrix, steps)
            raw_peak_idx, raw_peak_value = _touch_peak(raw_matrix, steps)
            self_peak_idx, self_peak_value = _touch_peak(self_matrix, steps)
            external_peak_idx, external_peak_value = _touch_peak(external_matrix, steps)

            ax_trace = axes[row_idx]
            if total_matrix is not None:
                ax_trace.plot(ts, total_series, color="black", linewidth=1.8, label="total")
            if self_matrix is not None:
                ax_trace.plot(ts, self_series, color="#2ca25f", linewidth=1.9, label="self-touch")
            if external_matrix is not None:
                ax_trace.plot(ts, external_series, color="#2b6cb0", linewidth=1.9, label="object touch")
            ax_trace.set_title(f"{name.capitalize()} mean contact magnitude", fontsize=10)
            ax_trace.set_ylabel("mean magnitude")
            ax_trace.grid(True, alpha=0.25)
            ax_trace.legend(
                fontsize=8,
                loc="upper right",
                ncol=3,
                framealpha=0.92,
            )

            for index in range(steps):
                writer.writerow(
                    [
                        name,
                        int(ts[index]),
                        _csv_value(total_series[index]),
                        _csv_value(raw_series[index]),
                        _csv_value(self_series[index]),
                        _csv_value(external_series[index]),
                        int(total_peak_idx[index]),
                        int(raw_peak_idx[index]),
                        int(self_peak_idx[index]),
                        int(external_peak_idx[index]),
                        _csv_value(total_peak_value[index]),
                        _csv_value(raw_peak_value[index]),
                        _csv_value(self_peak_value[index]),
                        _csv_value(external_peak_value[index]),
                    ]
                )

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(title, fontsize=12)
    image_path = os.path.join(plot_dir, f"touch_decomposition_{tag}.png")
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    return {"touch_decomposition": image_path, "touch_decomposition_csv": csv_path}


def _touch_taxel_vector_sequence(sample: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if sample is None:
        return None
    sample = np.nan_to_num(
        np.asarray(sample, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if sample.ndim == 1:
        sample = sample[None, :]
    sample = sample.reshape(sample.shape[0], -1)
    if sample.shape[-1] % 3 == 0 and sample.shape[-1] >= 3:
        return sample.reshape(sample.shape[0], sample.shape[-1] // 3, 3)
    out = np.zeros((sample.shape[0], sample.shape[-1], 3), dtype=np.float32)
    out[..., 2] = sample
    return out


def save_touch_decomposition_visualization(
    *,
    total_touch: Mapping[str, object],
    self_touch: Mapping[str, object],
    plot_dir: str,
    dataset_param: Optional[Mapping] = None,
    scaling_param: Optional[Mapping] = None,
    selftouch_dataset_param: Optional[Mapping] = None,
    selftouch_scaling_param: Optional[Mapping] = None,
    raw_touch: Optional[Mapping[str, object]] = None,
    external_touch: Optional[Mapping[str, object]] = None,
    finger_names: Sequence[str] = FINGER_ORDER,
    finger_keys: Optional[Sequence[str]] = None,
    selftouch_regions: Optional[Mapping[str, str]] = None,
    object_regions: Optional[Mapping[str, str]] = None,
    inactive_fingers: Optional[Sequence[str]] = None,
    coactive_selftouch_fingers: Optional[Sequence[str]] = None,
    coactive_selftouch_target_ratio=DEFAULT_COOCCURRENT_SELFTOUCH_TARGET_RATIO,
    coactive_selftouch_object_ratio: float = DEFAULT_COOCCURRENT_SELFTOUCH_OBJECT_RATIO,
    coactive_selftouch_total_percentile: float = DEFAULT_COOCCURRENT_SELFTOUCH_TOTAL_PERCENTILE,
    coactive_selftouch_strength: float = DEFAULT_COOCCURRENT_SELFTOUCH_STRENGTH,
    selftouch_region_strength: float = 0.90,
    object_region_strength: float = DEFAULT_OBJECT_FRONT_CENTER_STRENGTH,
    sample_index: int = 0,
    tag: str = "inference",
    title: Optional[str] = None,
    fps: int = 10,
    frame_stride: int = 1,
    clim_percentile: float = 99.0,
) -> Dict[str, str]:
    """Save a four-fingertip XELA-style video for self-touch/object-touch state."""
    try:
        from vis_tac import FourFingerTouchStateVisualizer
    except Exception as exc:
        print(f"[warn] failed to import tactile visualizer: {exc}")
        return {}

    keys = list(finger_keys) if finger_keys is not None else [FINGER_TO_KEY[name] for name in finger_names]
    names = [str(name).replace("tactile_", "").replace("_tip", "") for name in finger_names]
    inactive = _normalised_name_set(inactive_fingers)
    os.makedirs(plot_dir, exist_ok=True)

    touch_state = {}
    any_data = False
    for name, key in zip(names, keys):
        total_arr = _touch_unscale(
            _touch_mapping_value(total_touch, key, name),
            key,
            dataset_param,
            scaling_param,
        )
        self_arr = _touch_unscale(
            _touch_mapping_value(self_touch, key, name),
            key,
            selftouch_dataset_param or dataset_param,
            selftouch_scaling_param if selftouch_scaling_param is not None else scaling_param,
        )
        raw_arr = _touch_unscale(
            _touch_mapping_value(raw_touch, key, name),
            key,
            dataset_param,
            scaling_param,
        )
        external_arr = _touch_unscale(
            _touch_mapping_value(external_touch, key, name),
            key,
            dataset_param,
            scaling_param,
        )
        sequences = [arr for arr in (total_arr, self_arr, raw_arr, external_arr) if arr is not None]
        if not sequences:
            touch_state[name] = {}
            continue
        steps = min(int(arr.shape[1]) for arr in sequences)
        dims = min(int(arr.shape[-1]) for arr in sequences)
        if steps <= 0 or dims <= 0:
            touch_state[name] = {}
            continue

        def crop(arr):
            if arr is None:
                return None
            return arr[:, :steps, :dims]

        total_sample = _touch_sample(crop(total_arr), sample_index)
        self_sample = _touch_sample(crop(self_arr), sample_index)
        raw_sample = _touch_sample(crop(raw_arr), sample_index)
        object_sample = _touch_sample(crop(external_arr), sample_index)
        base_sample = total_sample if total_sample is not None else raw_sample
        self_vectors, object_vectors = _touch_decompose_vectors_for_display(
            base_sample,
            self_sample,
            object_sample,
            finger_name=name,
            selftouch_region=_touch_region_for_finger(selftouch_regions, name),
            object_region=_touch_region_for_finger(object_regions, name),
            coactive_selftouch_fingers=coactive_selftouch_fingers,
            coactive_selftouch_target_ratio=coactive_selftouch_target_ratio,
            coactive_selftouch_object_ratio=coactive_selftouch_object_ratio,
            coactive_selftouch_total_percentile=coactive_selftouch_total_percentile,
            coactive_selftouch_strength=coactive_selftouch_strength,
            selftouch_region_strength=selftouch_region_strength,
            object_region_strength=object_region_strength,
        )
        if _normalise_finger_name(name) in inactive:
            if self_vectors is not None:
                self_vectors = np.zeros_like(self_vectors)
            if object_vectors is not None:
                object_vectors = np.zeros_like(object_vectors)
        touch_state[name] = {
            "selftouch": self_vectors,
            "object": object_vectors,
        }
        any_data = any_data or self_vectors is not None or object_vectors is not None

    if not any_data:
        return {}

    tag = _touch_plot_tag(tag)
    path = os.path.join(plot_dir, f"touch_decomposition_{tag}_visualizer.mp4")
    visualizer = FourFingerTouchStateVisualizer(
        fingers=names,
        clim_percentile=clim_percentile,
        title=title or f"Touch states ({tag})",
    )
    video_path = visualizer.export_touch_state_video(
        touch_state,
        path=path,
        fps=max(int(fps or 10), 1),
        frame_stride=max(int(frame_stride or 1), 1),
    )
    return {"touch_decomposition_video": video_path}


def _save_tactile_profile_plot(
    *,
    profiles: Sequence[Mapping],
    epoch: int,
    plot_dir: str,
    dataset_param: Mapping,
    plt,
) -> Optional[str]:
    if not profiles:
        return None
    profiles = _ordered_profiles(profiles)
    sequence_length = int(dataset_param.get("sequence_length", TACTILE_XMAX) or TACTILE_XMAX)
    fig, axes = plt.subplots(
        len(profiles),
        1,
        figsize=(15.5, 2.4 * len(profiles)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    plotted_series = []
    for profile in profiles:
        timesteps = np.asarray(profile.get("timesteps", []), dtype=np.int64).reshape(-1)
        steps = int(timesteps.size)
        raw_trace = _profile_mean_trace(profile.get("raw_cmp"), steps)
        pred_trace = _profile_mean_trace(profile.get("pred_cmp"), steps)
        plotted_series.extend([raw_trace, pred_trace])
        profile["plot_raw_trace"] = raw_trace
        profile["plot_pred_trace"] = pred_trace

    for ax, profile in zip(axes, profiles):
        name = str(profile.get("name", "finger"))
        timesteps = np.asarray(profile.get("timesteps", []), dtype=np.int64).reshape(-1)
        raw_trace = np.asarray(profile.get("plot_raw_trace", []), dtype=np.float32)
        pred_trace = np.asarray(profile.get("plot_pred_trace", []), dtype=np.float32)
        count = min(timesteps.size, raw_trace.size, pred_trace.size)
        color = FINGER_COLORS.get(name, "black")
        title_name = name.capitalize()
        if count > 0:
            ts = timesteps[:count]
            raw = raw_trace[:count]
            pred, _ = _start_aligned_prediction(raw, pred_trace[:count])
            draw_tactile_prediction_profile(ax, ts, raw, pred, np.abs(raw - pred), color)
            pred_acc = prediction_accuracy(raw, pred, scale=PROFILE_ACCURACY_SCALE)
        else:
            pred_acc = float(profile.get("accuracy", 0.0))
        ax.set_ylim(TACTILE_YMIN, TACTILE_YMAX)
        ax.set_yticks(TACTILE_YTICKS)
        ax.set_ylabel("tactile value")
        ax.set_title(
            f"{title_name} | pred_acc={pred_acc:.1f}%",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    for ax in axes:
        _set_tactile_time_axis(ax, max(sequence_length, 1))
    if profiles and np.asarray(profiles[0].get("timesteps", [])).size:
        ts0 = np.asarray(profiles[0].get("timesteps"), dtype=np.int64)
        caption = f"aligned timesteps {int(np.min(ts0))}-{int(np.max(ts0))} from raw sequence length {sequence_length}"
    else:
        caption = f"raw sequence length {sequence_length}"
    fig.suptitle(f"{TACTILE_PLOT_TITLE} ({caption})", fontsize=12)
    image_path = os.path.join(plot_dir, f"tactile_profile_epoch_{epoch:04d}.png")
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    return image_path


def _core_metric_fieldnames(finger_names: Sequence[str]) -> List[str]:
    fields = [
        "epoch",
        "accuracy_mode",
        "accuracy_tolerance",
        "prediction_mae",
        "prediction_raw_mae",
        "prediction_accuracy",
        "prediction_raw_accuracy",
        "prediction_taxel_raw_accuracy",
        "prediction_profile_accuracy",
        "prediction_line_accuracy",
        "prediction_active_taxel_accuracy",
        "prediction_active_taxel_mae",
        "prediction_active_taxel_fraction",
        "prediction_contact_region_accuracy",
        "prediction_contact_region_mae",
        "prediction_peak_accuracy",
        "prediction_peak_mae",
        "prediction_peak_taxel_fraction",
        "prediction_closeness_accuracy",
        "zero_baseline_raw_accuracy",
        "zero_baseline_raw_mae",
        "zero_baseline_active_taxel_accuracy",
        "train_mean_baseline_raw_accuracy",
        "train_mean_baseline_raw_mae",
        "train_mean_baseline_active_taxel_accuracy",
        "previous_timestep_baseline_raw_accuracy",
        "previous_timestep_baseline_raw_mae",
        "previous_timestep_baseline_active_taxel_accuracy",
        "pca_image",
        "pca_csv",
        "pca_pc1_explained_variance",
        "pca_pc2_explained_variance",
        "pca_total_explained_variance",
        "pca_raw_pred_gap",
        "pca_centroid_raw_pred_gap",
        "pca_raw_combo_centroid_spread",
    ]
    for name in finger_names:
        fields.extend([
            f"{name}_mae",
            f"{name}_raw_mae",
            f"{name}_accuracy",
            f"{name}_raw_accuracy",
            f"{name}_taxel_raw_accuracy",
            f"{name}_profile_accuracy",
            f"{name}_line_accuracy",
            f"{name}_active_taxel_accuracy",
            f"{name}_active_taxel_mae",
            f"{name}_active_taxel_fraction",
            f"{name}_contact_region_accuracy",
            f"{name}_contact_region_mae",
            f"{name}_peak_accuracy",
            f"{name}_peak_mae",
            f"{name}_peak_taxel_fraction",
            f"{name}_closeness_accuracy",
            f"{name}_zero_baseline_raw_accuracy",
            f"{name}_zero_baseline_raw_mae",
            f"{name}_zero_baseline_active_taxel_accuracy",
            f"{name}_train_mean_baseline_raw_accuracy",
            f"{name}_train_mean_baseline_raw_mae",
            f"{name}_train_mean_baseline_active_taxel_accuracy",
            f"{name}_previous_timestep_baseline_raw_accuracy",
            f"{name}_previous_timestep_baseline_raw_mae",
            f"{name}_previous_timestep_baseline_active_taxel_accuracy",
        ])
    return fields


def _append_core_metric_history(
    plot_dir: str,
    epoch: int,
    metrics: Mapping[str, float],
    finger_names: Sequence[str],
    *,
    pca_image: Optional[str] = None,
    pca_csv: Optional[str] = None,
) -> str:
    path = os.path.join(plot_dir, "tactile_metrics.csv")
    fieldnames = _core_metric_fieldnames(finger_names)
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        row = {
            "epoch": int(epoch),
            "accuracy_mode": str(metrics.get("accuracy_mode", "")),
            "accuracy_tolerance": float(metrics.get("accuracy_tolerance", 0.0)),
            "prediction_mae": float(metrics.get("tactile_line_mae", metrics.get("prediction_mae", 0.0))),
            "prediction_raw_mae": float(
                metrics.get("taxel_raw_mae", metrics.get("tactile_line_raw_mae", metrics.get("prediction_mae", 0.0)))
            ),
            "prediction_accuracy": float(
                metrics.get("tactile_taxel_raw_accuracy", metrics.get("tactile_line_raw_accuracy", metrics.get("prediction_accuracy", 0.0)))
            ),
            "prediction_raw_accuracy": float(
                metrics.get("tactile_taxel_raw_accuracy", metrics.get("tactile_line_raw_accuracy", 0.0))
            ),
            "prediction_taxel_raw_accuracy": float(
                metrics.get("tactile_taxel_raw_accuracy", metrics.get("tactile_line_raw_accuracy", 0.0))
            ),
            "prediction_profile_accuracy": float(
                metrics.get("tactile_line_profile_accuracy", metrics.get("profile_acc", 0.0))
            ),
            "prediction_line_accuracy": float(
                metrics.get("line_acc", metrics.get("tactile_line_profile_accuracy", 0.0))
            ),
            "prediction_active_taxel_accuracy": float(metrics.get("active_taxel_acc", 0.0)),
            "prediction_active_taxel_mae": float(metrics.get("active_taxel_mae", 0.0)),
            "prediction_active_taxel_fraction": float(metrics.get("active_taxel_fraction", 0.0)),
            "prediction_contact_region_accuracy": float(metrics.get("contact_region_acc", 0.0)),
            "prediction_contact_region_mae": float(metrics.get("contact_region_mae", 0.0)),
            "prediction_peak_accuracy": float(metrics.get("peak_acc", 0.0)),
            "prediction_peak_mae": float(metrics.get("peak_mae", 0.0)),
            "prediction_peak_taxel_fraction": float(metrics.get("peak_taxel_fraction", 0.0)),
            "prediction_closeness_accuracy": float(
                metrics.get("tactile_line_closeness_accuracy", metrics.get("prediction_closeness_accuracy", 0.0))
            ),
            "zero_baseline_raw_accuracy": float(metrics.get("zero_baseline_raw_acc", 0.0)),
            "zero_baseline_raw_mae": float(metrics.get("zero_baseline_raw_mae", 0.0)),
            "zero_baseline_active_taxel_accuracy": float(metrics.get("zero_baseline_active_taxel_acc", 0.0)),
            "train_mean_baseline_raw_accuracy": float(metrics.get("train_mean_baseline_raw_acc", 0.0)),
            "train_mean_baseline_raw_mae": float(metrics.get("train_mean_baseline_raw_mae", 0.0)),
            "train_mean_baseline_active_taxel_accuracy": float(
                metrics.get("train_mean_baseline_active_taxel_acc", 0.0)
            ),
            "previous_timestep_baseline_raw_accuracy": float(
                metrics.get("previous_timestep_baseline_raw_acc", 0.0)
            ),
            "previous_timestep_baseline_raw_mae": float(metrics.get("previous_timestep_baseline_raw_mae", 0.0)),
            "previous_timestep_baseline_active_taxel_accuracy": float(
                metrics.get("previous_timestep_baseline_active_taxel_acc", 0.0)
            ),
            "pca_image": pca_image or "",
            "pca_csv": pca_csv or "",
            "pca_pc1_explained_variance": float(metrics.get("pca_pc1_explained_variance", 0.0)),
            "pca_pc2_explained_variance": float(metrics.get("pca_pc2_explained_variance", 0.0)),
            "pca_total_explained_variance": float(metrics.get("pca_total_explained_variance", 0.0)),
            "pca_raw_pred_gap": float(metrics.get("pca_raw_pred_gap", 0.0)),
            "pca_centroid_raw_pred_gap": float(metrics.get("pca_centroid_raw_pred_gap", 0.0)),
            "pca_raw_combo_centroid_spread": float(metrics.get("pca_raw_combo_centroid_spread", 0.0)),
        }
        for name in finger_names:
            row[f"{name}_mae"] = float(metrics.get(f"{name}_raw_mae", metrics.get(f"{name}_mae", 0.0)))
            row[f"{name}_raw_mae"] = float(metrics.get(f"{name}_raw_mae", metrics.get(f"{name}_mae", 0.0)))
            row[f"{name}_accuracy"] = float(
                metrics.get(f"{name}_taxel_raw_accuracy", metrics.get(f"{name}_raw_accuracy", metrics.get(f"{name}_accuracy", 0.0)))
            )
            row[f"{name}_raw_accuracy"] = float(
                metrics.get(f"{name}_taxel_raw_accuracy", metrics.get(f"{name}_raw_accuracy", metrics.get(f"{name}_accuracy", 0.0)))
            )
            row[f"{name}_taxel_raw_accuracy"] = float(
                metrics.get(f"{name}_taxel_raw_accuracy", metrics.get(f"{name}_raw_accuracy", 0.0))
            )
            row[f"{name}_profile_accuracy"] = float(metrics.get(f"{name}_profile_accuracy", 0.0))
            row[f"{name}_line_accuracy"] = float(
                metrics.get(f"{name}_line_accuracy", metrics.get(f"{name}_profile_accuracy", 0.0))
            )
            row[f"{name}_active_taxel_accuracy"] = float(metrics.get(f"{name}_active_taxel_acc", 0.0))
            row[f"{name}_active_taxel_mae"] = float(metrics.get(f"{name}_active_taxel_mae", 0.0))
            row[f"{name}_active_taxel_fraction"] = float(metrics.get(f"{name}_active_taxel_fraction", 0.0))
            row[f"{name}_contact_region_accuracy"] = float(metrics.get(f"{name}_contact_region_acc", 0.0))
            row[f"{name}_contact_region_mae"] = float(metrics.get(f"{name}_contact_region_mae", 0.0))
            row[f"{name}_peak_accuracy"] = float(metrics.get(f"{name}_peak_acc", 0.0))
            row[f"{name}_peak_mae"] = float(metrics.get(f"{name}_peak_mae", 0.0))
            row[f"{name}_peak_taxel_fraction"] = float(metrics.get(f"{name}_peak_taxel_fraction", 0.0))
            row[f"{name}_closeness_accuracy"] = float(metrics.get(f"{name}_closeness_accuracy", 0.0))
            for baseline_name in ("zero_baseline", "train_mean_baseline", "previous_timestep_baseline"):
                row[f"{name}_{baseline_name}_raw_accuracy"] = float(
                    metrics.get(f"{name}_{baseline_name}_raw_acc", 0.0)
                )
                row[f"{name}_{baseline_name}_raw_mae"] = float(
                    metrics.get(f"{name}_{baseline_name}_raw_mae", 0.0)
                )
                row[f"{name}_{baseline_name}_active_taxel_accuracy"] = float(
                    metrics.get(f"{name}_{baseline_name}_active_taxel_acc", 0.0)
                )
        writer.writerow(row)
    return path


def _raw_metric_fieldnames(finger_names: Sequence[str]) -> List[str]:
    fieldnames = [
        "epoch",
        "optimizer_step",
        "seed",
        "use_tactile_history",
        "accuracy_mode",
        "accuracy_tolerance",
        "prediction_mae",
        "prediction_raw_mae",
        "prediction_mae_percent",
        "prediction_raw_mae_percent",
        "prediction_accuracy",
        "prediction_closeness_accuracy",
        "prediction_profile_accuracy",
        "prediction_raw_accuracy",
        "prediction_taxel_raw_accuracy",
        "prediction_line_accuracy",
        "prediction_active_taxel_accuracy",
        "prediction_active_taxel_mae",
        "prediction_active_taxel_fraction",
        "prediction_contact_region_accuracy",
        "prediction_contact_region_mae",
        "prediction_peak_accuracy",
        "prediction_peak_mae",
        "prediction_peak_taxel_fraction",
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
        "zero_baseline_raw_accuracy",
        "zero_baseline_raw_mae",
        "zero_baseline_active_taxel_accuracy",
        "train_mean_baseline_raw_accuracy",
        "train_mean_baseline_raw_mae",
        "train_mean_baseline_active_taxel_accuracy",
        "previous_timestep_baseline_raw_accuracy",
        "previous_timestep_baseline_raw_mae",
        "previous_timestep_baseline_active_taxel_accuracy",
    ]
    for name in finger_names:
        fieldnames.extend([
            f"{name}_mae",
            f"{name}_raw_mae",
            f"{name}_mae_percent",
            f"{name}_raw_mae_percent",
            f"{name}_accuracy",
            f"{name}_closeness_accuracy",
            f"{name}_profile_accuracy",
            f"{name}_raw_accuracy",
            f"{name}_taxel_raw_accuracy",
            f"{name}_line_accuracy",
            f"{name}_active_taxel_accuracy",
            f"{name}_active_taxel_mae",
            f"{name}_active_taxel_fraction",
            f"{name}_contact_region_accuracy",
            f"{name}_contact_region_mae",
            f"{name}_peak_accuracy",
            f"{name}_peak_mae",
            f"{name}_peak_taxel_fraction",
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
            f"{name}_zero_baseline_raw_accuracy",
            f"{name}_zero_baseline_raw_mae",
            f"{name}_zero_baseline_active_taxel_accuracy",
            f"{name}_train_mean_baseline_raw_accuracy",
            f"{name}_train_mean_baseline_raw_mae",
            f"{name}_train_mean_baseline_active_taxel_accuracy",
            f"{name}_previous_timestep_baseline_raw_accuracy",
            f"{name}_previous_timestep_baseline_raw_mae",
            f"{name}_previous_timestep_baseline_active_taxel_accuracy",
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
            "optimizer_step": int(metrics.get("optimizer_step", 0)),
            "seed": int(metrics.get("seed", 0)),
            "use_tactile_history": int(bool(metrics.get("use_tactile_history", False))),
            "accuracy_mode": str(metrics.get("accuracy_mode", "")),
            "accuracy_tolerance": float(metrics.get("accuracy_tolerance", 0.0)),
            "prediction_mae": float(metrics.get("tactile_line_mae", 0.0)),
            "prediction_raw_mae": float(metrics.get("tactile_line_raw_mae", metrics.get("tactile_line_mae", 0.0))),
            "prediction_mae_percent": float(metrics.get("tactile_line_mae_percent", 0.0)),
            "prediction_raw_mae_percent": float(
                metrics.get("tactile_line_raw_mae_percent", metrics.get("tactile_line_mae_percent", 0.0))
            ),
            "prediction_accuracy": float(
                metrics.get("tactile_taxel_raw_accuracy", metrics.get("tactile_line_raw_accuracy", 0.0))
            ),
            "prediction_closeness_accuracy": float(metrics.get("tactile_line_closeness_accuracy", 0.0)),
            "prediction_profile_accuracy": float(metrics.get("tactile_line_profile_accuracy", 0.0)),
            "prediction_raw_accuracy": float(
                metrics.get("tactile_taxel_raw_accuracy", metrics.get("tactile_line_raw_accuracy", 0.0))
            ),
            "prediction_taxel_raw_accuracy": float(
                metrics.get("tactile_taxel_raw_accuracy", metrics.get("tactile_line_raw_accuracy", 0.0))
            ),
            "prediction_line_accuracy": float(
                metrics.get("line_acc", metrics.get("tactile_line_profile_accuracy", 0.0))
            ),
            "prediction_active_taxel_accuracy": float(metrics.get("active_taxel_acc", 0.0)),
            "prediction_active_taxel_mae": float(metrics.get("active_taxel_mae", 0.0)),
            "prediction_active_taxel_fraction": float(metrics.get("active_taxel_fraction", 0.0)),
            "prediction_contact_region_accuracy": float(metrics.get("contact_region_acc", 0.0)),
            "prediction_contact_region_mae": float(metrics.get("contact_region_mae", 0.0)),
            "prediction_peak_accuracy": float(metrics.get("peak_acc", 0.0)),
            "prediction_peak_mae": float(metrics.get("peak_mae", 0.0)),
            "prediction_peak_taxel_fraction": float(metrics.get("peak_taxel_fraction", 0.0)),
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
            "zero_baseline_raw_accuracy": float(metrics.get("zero_baseline_raw_acc", 0.0)),
            "zero_baseline_raw_mae": float(metrics.get("zero_baseline_raw_mae", 0.0)),
            "zero_baseline_active_taxel_accuracy": float(metrics.get("zero_baseline_active_taxel_acc", 0.0)),
            "train_mean_baseline_raw_accuracy": float(metrics.get("train_mean_baseline_raw_acc", 0.0)),
            "train_mean_baseline_raw_mae": float(metrics.get("train_mean_baseline_raw_mae", 0.0)),
            "train_mean_baseline_active_taxel_accuracy": float(
                metrics.get("train_mean_baseline_active_taxel_acc", 0.0)
            ),
            "previous_timestep_baseline_raw_accuracy": float(
                metrics.get("previous_timestep_baseline_raw_acc", 0.0)
            ),
            "previous_timestep_baseline_raw_mae": float(metrics.get("previous_timestep_baseline_raw_mae", 0.0)),
            "previous_timestep_baseline_active_taxel_accuracy": float(
                metrics.get("previous_timestep_baseline_active_taxel_acc", 0.0)
            ),
        }
        for name in finger_names:
            row[f"{name}_mae"] = float(metrics.get(f"{name}_mae", 0.0))
            row[f"{name}_raw_mae"] = float(metrics.get(f"{name}_raw_mae", metrics.get(f"{name}_mae", 0.0)))
            row[f"{name}_mae_percent"] = float(metrics.get(f"{name}_mae_percent", 0.0))
            row[f"{name}_raw_mae_percent"] = float(
                metrics.get(f"{name}_raw_mae_percent", metrics.get(f"{name}_mae_percent", 0.0))
            )
            row[f"{name}_accuracy"] = float(
                metrics.get(f"{name}_taxel_raw_accuracy", metrics.get(f"{name}_raw_accuracy", metrics.get(f"{name}_accuracy", 0.0)))
            )
            row[f"{name}_closeness_accuracy"] = float(metrics.get(f"{name}_closeness_accuracy", 0.0))
            row[f"{name}_profile_accuracy"] = float(metrics.get(f"{name}_profile_accuracy", 0.0))
            row[f"{name}_raw_accuracy"] = float(
                metrics.get(f"{name}_taxel_raw_accuracy", metrics.get(f"{name}_raw_accuracy", metrics.get(f"{name}_accuracy", 0.0)))
            )
            row[f"{name}_taxel_raw_accuracy"] = float(
                metrics.get(f"{name}_taxel_raw_accuracy", metrics.get(f"{name}_raw_accuracy", 0.0))
            )
            row[f"{name}_line_accuracy"] = float(
                metrics.get(f"{name}_line_accuracy", metrics.get(f"{name}_profile_accuracy", 0.0))
            )
            row[f"{name}_active_taxel_accuracy"] = float(metrics.get(f"{name}_active_taxel_acc", 0.0))
            row[f"{name}_active_taxel_mae"] = float(metrics.get(f"{name}_active_taxel_mae", 0.0))
            row[f"{name}_active_taxel_fraction"] = float(metrics.get(f"{name}_active_taxel_fraction", 0.0))
            row[f"{name}_contact_region_accuracy"] = float(metrics.get(f"{name}_contact_region_acc", 0.0))
            row[f"{name}_contact_region_mae"] = float(metrics.get(f"{name}_contact_region_mae", 0.0))
            row[f"{name}_peak_accuracy"] = float(metrics.get(f"{name}_peak_acc", 0.0))
            row[f"{name}_peak_mae"] = float(metrics.get(f"{name}_peak_mae", 0.0))
            row[f"{name}_peak_taxel_fraction"] = float(metrics.get(f"{name}_peak_taxel_fraction", 0.0))
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
            for baseline_name in ("zero_baseline", "train_mean_baseline", "previous_timestep_baseline"):
                row[f"{name}_{baseline_name}_raw_accuracy"] = float(
                    metrics.get(f"{name}_{baseline_name}_raw_acc", 0.0)
                )
                row[f"{name}_{baseline_name}_raw_mae"] = float(
                    metrics.get(f"{name}_{baseline_name}_raw_mae", 0.0)
                )
                row[f"{name}_{baseline_name}_active_taxel_accuracy"] = float(
                    metrics.get(f"{name}_{baseline_name}_active_taxel_acc", 0.0)
                )
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
                raw_value = _safe_float(
                    row.get("raw_tactile_avg"),
                    _safe_float(row.get("raw_tactile")),
                )
                pred_value = _safe_float(
                    row.get("pred_self_touch_avg"),
                    _safe_float(row.get("pred_self_touch_raw")),
                )
                if np.isfinite(raw_value) and np.isfinite(pred_value):
                    by_finger[name]["raw"].append(raw_value)
                    by_finger[name]["pred"].append(pred_value)
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
            np.isfinite(_safe_float(row.get("prediction_taxel_raw_accuracy")))
            or np.isfinite(_safe_float(row.get("prediction_raw_accuracy")))
            or np.isfinite(_safe_float(row.get("prediction_accuracy")))
            or any(
                np.isfinite(_safe_float(row.get(f"{name}_taxel_raw_accuracy")))
                or np.isfinite(_safe_float(row.get(f"{name}_raw_accuracy")))
                for name in finger_names
            )
            for row in history_rows
        )
        if has_accuracy:
            rows = history_rows
    if not rows:
        return None

    epochs = np.array([int(_safe_float(r.get("epoch"), 0.0)) for r in rows], dtype=np.int32)
    overall_key = "prediction_taxel_raw_accuracy" if any(
        np.isfinite(_safe_float(r.get("prediction_taxel_raw_accuracy"))) for r in rows
    ) else "prediction_raw_accuracy" if any(
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
        taxel_key = f"{name}_taxel_raw_accuracy"
        raw_key = f"{name}_raw_accuracy"
        if any(np.isfinite(_safe_float(r.get(taxel_key))) for r in rows):
            key = taxel_key
        elif any(np.isfinite(_safe_float(r.get(raw_key))) for r in rows):
            key = raw_key
        else:
            key = f"{name}_accuracy"
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

    plt.title("Raw taxel threshold accuracy vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("taxels within configured tolerance (%)")
    if len(epochs) > 1:
        x_pad = (epochs[-1] - epochs[0]) * 0.08
        plt.xlim(0, epochs[-1] + max(x_pad, 5))
    elif len(epochs) == 1:
        plt.xlim(0, max(int(epochs[0]), 1))
    plt.ylim(0.0, 100.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(plot_dir, "raw_prediction_accuracy.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def summarize_tactile_metrics_and_pca(
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
    save_combination_pca: bool = True,
    save_tactile_profile: bool = False,
) -> Dict[str, Dict]:
    """Return lightweight tactile metrics and optional profile/PCA images."""
    os.makedirs(plot_dir, exist_ok=True)
    active = included_fingers_from_combinations(combinations)
    scaling_param = load_scaling_param(dataset_param)
    keys = list(finger_keys) if finger_keys is not None else [FINGER_TO_KEY[name] for name in finger_names]
    ref_batch, ref_raw_steps, ref_dim = _fallback_raw_shape(
        data,
        preds,
        finger_names,
        keys,
        next_step=next_step,
    )

    profiles = []
    for name, key in zip(finger_names, keys):
        raw_arr = _array_from_mapping(data, key)
        pred_arr = _array_from_mapping(preds, name)
        has_raw = raw_arr is not None
        has_pred = pred_arr is not None
        if raw_arr is None:
            raw_arr = _zero_raw_array(
                batch=int(pred_arr.shape[0]) if pred_arr is not None and pred_arr.ndim >= 3 else ref_batch,
                raw_steps=int(pred_arr.shape[1] + 1) if pred_arr is not None and pred_arr.ndim >= 3 and next_step else ref_raw_steps,
                dim=int(pred_arr.shape[-1]) if pred_arr is not None and pred_arr.ndim >= 3 else ref_dim,
            )
        if pred_arr is None:
            pred_arr = _zero_pred_for_raw(raw_arr, next_step=next_step)

        raw_scaled, pred_scaled, timesteps = align_next_step_prediction(
            raw_arr,
            pred_arr,
            next_step=next_step,
            input_offset=int(dataset_param.get("input_offset", 0) or 0),
        )
        steps = min(raw_scaled.shape[1], pred_scaled.shape[1])
        raw_scaled = raw_scaled[:, :steps, :]
        pred_scaled = pred_scaled[:, :steps, :]
        timesteps = np.asarray(timesteps[:steps], dtype=np.int64)
        valid_rows = _combined_valid_rows(
            data,
            name,
            int(pred_arr.shape[1]) if pred_arr is not None and pred_arr.ndim >= 3 else steps,
            next_step=next_step,
            input_offset=int(dataset_param.get("input_offset", 0) or 0),
        )

        if name in active and has_raw and has_pred:
            raw_cmp = maybe_unscale(raw_scaled, key, dataset_param, scaling_param)
            raw_full_cmp = maybe_unscale(raw_arr, key, dataset_param, scaling_param)
            pred_cmp = maybe_unscale(
                np.nan_to_num(pred_scaled, nan=0.0, posinf=0.0, neginf=0.0),
                key,
                dataset_param,
                scaling_param,
            )
        else:
            raw_cmp = np.zeros_like(raw_scaled, dtype=np.float32)
            pred_cmp = np.zeros_like(raw_cmp, dtype=np.float32)
            raw_full_cmp = np.zeros(
                (
                    raw_cmp.shape[0],
                    int(raw_arr.shape[1]) if raw_arr is not None and raw_arr.ndim >= 3 else raw_cmp.shape[1] + 1,
                    raw_cmp.shape[-1],
                ),
                dtype=np.float32,
            )

        valid_labels = True
        if valid_rows is not None:
            valid_labels = bool(np.any(valid_rows))
            if valid_labels:
                raw_cmp = raw_cmp[valid_rows]
                pred_cmp = pred_cmp[valid_rows]
                raw_full_cmp = raw_full_cmp[valid_rows]
            else:
                raw_cmp = raw_cmp[:0]
                pred_cmp = pred_cmp[:0]
                raw_full_cmp = raw_full_cmp[:0]

        abs_error = np.abs(raw_cmp - pred_cmp)
        zero_baseline = np.zeros_like(raw_cmp, dtype=np.float32)
        train_mean_baseline = _scaling_mean_array(scaling_param, key, raw_cmp)
        previous_timestep_baseline = _previous_timestep_baseline(raw_full_cmp, timesteps, raw_cmp)
        profiles.append(
            {
                "name": name,
                "active": name in active,
                "has_raw": has_raw,
                "has_pred": has_pred,
                "valid_labels": valid_labels,
                "timesteps": timesteps,
                "raw_cmp": raw_cmp,
                "pred_cmp": pred_cmp,
                "mae": float(np.mean(abs_error)) if abs_error.size else 0.0,
                "zero_baseline": _baseline_taxel_metrics(raw_cmp, zero_baseline, dataset_param),
                "train_mean_baseline": _baseline_taxel_metrics(raw_cmp, train_mean_baseline, dataset_param),
                "previous_timestep_baseline": _baseline_taxel_metrics(
                    raw_cmp,
                    previous_timestep_baseline,
                    dataset_param,
                ),
            }
        )

    metrics: Dict[str, float] = {}
    images: Dict[str, str] = {}
    files: Dict[str, str] = {}
    scale_values = [
        np.asarray(profile["raw_cmp"], dtype=np.float32).reshape(-1)
        for profile in profiles
        if profile["active"] and profile["has_raw"] and profile["valid_labels"] and np.asarray(profile["raw_cmp"]).size
    ]
    shared_raw_accuracy_scale = _robust_signal_spread(np.concatenate(scale_values)) if scale_values else None
    for profile in profiles:
        closeness_accuracy = prediction_accuracy(
            profile["raw_cmp"],
            profile["pred_cmp"],
            scale=shared_raw_accuracy_scale,
        )
        taxel_accuracy = raw_accuracy_score(
            profile["raw_cmp"],
            profile["pred_cmp"],
            params=dataset_param,
            scale=shared_raw_accuracy_scale,
        )
        line_accuracy, line_mae = profile_line_metrics(profile["raw_cmp"], profile["pred_cmp"])
        contact_metrics = contact_taxel_metrics(profile["raw_cmp"], profile["pred_cmp"], dataset_param)
        name = profile["name"]
        metrics[f"{name}_raw_mae"] = profile["mae"]
        metrics[f"{name}_mae"] = profile["mae"]
        metrics[f"{name}_profile_mae"] = line_mae
        metrics[f"{name}_raw_accuracy"] = taxel_accuracy
        metrics[f"{name}_taxel_raw_accuracy"] = taxel_accuracy
        metrics[f"{name}_profile_accuracy"] = line_accuracy
        metrics[f"{name}_line_accuracy"] = line_accuracy
        metrics[f"{name}_closeness_accuracy"] = closeness_accuracy
        metrics[f"{name}_active_taxel_acc"] = contact_metrics["active_taxel_acc"]
        metrics[f"{name}_active_taxel_mae"] = contact_metrics["active_taxel_mae"]
        metrics[f"{name}_active_taxel_fraction"] = contact_metrics["active_taxel_fraction"]
        metrics[f"{name}_contact_region_acc"] = contact_metrics["contact_region_acc"]
        metrics[f"{name}_contact_region_mae"] = contact_metrics["contact_region_mae"]
        metrics[f"{name}_peak_acc"] = contact_metrics["peak_acc"]
        metrics[f"{name}_peak_mae"] = contact_metrics["peak_mae"]
        metrics[f"{name}_peak_taxel_fraction"] = contact_metrics["peak_taxel_fraction"]
        for baseline_name in ("zero_baseline", "train_mean_baseline", "previous_timestep_baseline"):
            baseline = profile.get(baseline_name, {})
            metrics[f"{name}_{baseline_name}_raw_acc"] = float(baseline.get("raw_acc", 0.0))
            metrics[f"{name}_{baseline_name}_raw_mae"] = float(baseline.get("raw_mae", 0.0))
            metrics[f"{name}_{baseline_name}_active_taxel_acc"] = float(
                baseline.get("active_taxel_acc", 0.0)
            )
        profile["accuracy"] = taxel_accuracy
        profile["taxel_accuracy"] = taxel_accuracy
        profile["profile_accuracy"] = line_accuracy
        profile["profile_mae"] = line_mae
        profile["closeness_accuracy"] = closeness_accuracy
        profile.update(contact_metrics)

    metric_profiles = [
        profile
        for profile in profiles
        if profile["active"] and profile["has_raw"] and profile["has_pred"] and profile["valid_labels"]
    ]
    if not metric_profiles:
        metric_profiles = [
            profile
            for profile in profiles
            if profile["has_raw"] and profile["has_pred"] and profile["valid_labels"]
        ]
    if not metric_profiles:
        metric_profiles = [profile for profile in profiles if profile["active"]] or profiles
    metrics["tactile_line_mae"] = float(np.mean([profile["mae"] for profile in metric_profiles]))
    metrics["tactile_line_raw_mae"] = metrics["tactile_line_mae"]
    metrics["taxel_raw_mae"] = metrics["tactile_line_mae"]
    metrics["tactile_line_profile_mae"] = float(np.mean([profile["profile_mae"] for profile in metric_profiles]))
    metrics["tactile_taxel_raw_accuracy"] = float(
        np.mean([profile["taxel_accuracy"] for profile in metric_profiles])
    )
    metrics["tactile_line_profile_accuracy"] = float(
        np.mean([profile["profile_accuracy"] for profile in metric_profiles])
    )
    metrics["tactile_line_raw_accuracy"] = metrics["tactile_taxel_raw_accuracy"]
    metrics["tactile_line_accuracy"] = metrics["tactile_line_profile_accuracy"]
    metrics["tactile_line_closeness_accuracy"] = float(
        np.mean([profile["closeness_accuracy"] for profile in metric_profiles])
    )
    metrics["raw_mae"] = metrics["tactile_line_mae"]
    metrics["raw_acc"] = metrics["tactile_taxel_raw_accuracy"]
    metrics["taxel_raw_acc"] = metrics["tactile_taxel_raw_accuracy"]
    metrics["profile_acc"] = metrics["tactile_line_profile_accuracy"]
    metrics["line_acc"] = metrics["tactile_line_profile_accuracy"]
    metrics["closeness_acc"] = metrics["tactile_line_closeness_accuracy"]
    metrics["active_taxel_acc"] = float(np.mean([profile["active_taxel_acc"] for profile in metric_profiles]))
    metrics["active_taxel_mae"] = float(np.mean([profile["active_taxel_mae"] for profile in metric_profiles]))
    metrics["active_taxel_fraction"] = float(
        np.mean([profile["active_taxel_fraction"] for profile in metric_profiles])
    )
    metrics["contact_region_acc"] = float(np.mean([profile["contact_region_acc"] for profile in metric_profiles]))
    metrics["contact_region_mae"] = float(np.mean([profile["contact_region_mae"] for profile in metric_profiles]))
    metrics["peak_acc"] = float(np.mean([profile["peak_acc"] for profile in metric_profiles]))
    metrics["peak_mae"] = float(np.mean([profile["peak_mae"] for profile in metric_profiles]))
    metrics["peak_taxel_fraction"] = float(
        np.mean([profile["peak_taxel_fraction"] for profile in metric_profiles])
    )
    for baseline_name in ("zero_baseline", "train_mean_baseline", "previous_timestep_baseline"):
        metrics[f"{baseline_name}_raw_acc"] = float(
            np.mean([profile.get(baseline_name, {}).get("raw_acc", 0.0) for profile in metric_profiles])
        )
        metrics[f"{baseline_name}_raw_mae"] = float(
            np.mean([profile.get(baseline_name, {}).get("raw_mae", 0.0) for profile in metric_profiles])
        )
        metrics[f"{baseline_name}_active_taxel_acc"] = float(
            np.mean(
                [
                    profile.get(baseline_name, {}).get("active_taxel_acc", 0.0)
                    for profile in metric_profiles
                ]
            )
        )
    metrics["accuracy_mode"] = _accuracy_mode(dataset_param)
    metrics["accuracy_tolerance"] = _accuracy_tolerance(dataset_param)

    if save_tactile_profile:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            tactile_profile_path = _save_tactile_profile_plot(
                profiles=profiles,
                epoch=epoch,
                plot_dir=plot_dir,
                dataset_param=dataset_param,
                plt=plt,
            )
        except Exception as exc:
            print(f"[warn] failed to save lightweight self-touch tactile profile: {exc}")
            tactile_profile_path = None
        if tactile_profile_path:
            images["tactile_profile"] = tactile_profile_path

    if save_combination_pca:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            combo_pca_path, combo_pca_csv_path, combo_pca_metrics = _save_combination_pca_plot(
                data=data,
                preds=preds,
                epoch=epoch,
                plot_dir=plot_dir,
                dataset_param=dataset_param,
                scaling_param=scaling_param,
                combinations=combinations,
                finger_names=finger_names,
                finger_keys=keys,
                next_step=next_step,
                plt=plt,
            )
        except Exception as exc:
            print(f"[warn] failed to save lightweight self-touch combination PCA: {exc}")
            combo_pca_path = None
            combo_pca_csv_path = None
            combo_pca_metrics = {}
        if combo_pca_path:
            images["combination_pca"] = combo_pca_path
        if combo_pca_csv_path:
            files["combination_pca_csv"] = combo_pca_csv_path
        if combo_pca_metrics:
            metrics.update(combo_pca_metrics)

    core_metrics_path = _append_core_metric_history(
        plot_dir,
        epoch,
        metrics,
        [str(profile.get("name", "")) for profile in _ordered_profiles(profiles)],
        pca_image=images.get("combination_pca"),
        pca_csv=files.get("combination_pca_csv"),
    )
    files["tactile_metrics_csv"] = core_metrics_path

    return {"images": images, "metrics": metrics, "files": files}


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
    optimizer_step: Optional[int] = None,
    seed: Optional[int] = None,
    use_tactile_history: Optional[bool] = None,
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
            input_offset=int(dataset_param.get("input_offset", 0) or 0),
        )
        valid_rows = _combined_valid_rows(
            data,
            name,
            int(pred_arr.shape[1]) if pred_arr is not None and pred_arr.ndim >= 3 else int(pred_scaled.shape[1]),
            next_step=next_step,
            input_offset=int(dataset_param.get("input_offset", 0) or 0),
        )
        is_active = name in active
        full_steps = max(
            int(dataset_param.get("sequence_length", 0) or 0),
            int(raw_arr.shape[1]) if raw_arr.ndim >= 3 else 0,
            int(timesteps[-1]) + 1 if len(timesteps) else 0,
            1,
        )
        full_dim = int(raw_arr.shape[-1]) if raw_arr.ndim >= 3 else TAXELS_PER_FINGER

        if is_active:
            if has_raw:
                raw_cmp = maybe_unscale(raw_scaled, key, dataset_param, scaling_param)
                raw_full_cmp = maybe_unscale(raw_arr, key, dataset_param, scaling_param)
            else:
                raw_cmp = np.zeros_like(raw_scaled)
                raw_full_cmp = np.zeros((raw_scaled.shape[0], full_steps, full_dim), dtype=np.float32)

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
            raw_full_cmp = np.zeros((raw_scaled.shape[0], full_steps, full_dim), dtype=np.float32)

        raw_full_cmp = _sequence_to_length(raw_full_cmp, full_steps)

        valid_labels = True
        if valid_rows is not None:
            valid_labels = bool(np.any(valid_rows))
            if valid_labels:
                raw_cmp = raw_cmp[valid_rows]
                pred_cmp = pred_cmp[valid_rows]
                if raw_full_cmp.ndim >= 3:
                    raw_full_cmp = raw_full_cmp[valid_rows]
            else:
                steps = len(timesteps)
                dim = int(raw_scaled.shape[-1]) if raw_scaled.ndim >= 3 else TAXELS_PER_FINGER
                raw_cmp = np.zeros((1, steps, dim), dtype=np.float32)
                pred_cmp = np.zeros_like(raw_cmp)
                raw_full_cmp = np.zeros((1, full_steps, full_dim), dtype=np.float32)

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

        zero_baseline = np.zeros_like(raw_cmp, dtype=np.float32)
        train_mean_baseline = _scaling_mean_array(scaling_param, key, raw_cmp)
        previous_timestep_baseline = _previous_timestep_baseline(raw_full_cmp, timesteps, raw_cmp)
        zero_baseline_metrics = _baseline_taxel_metrics(raw_cmp, zero_baseline, dataset_param)
        train_mean_baseline_metrics = _baseline_taxel_metrics(
            raw_cmp,
            train_mean_baseline,
            dataset_param,
        )
        previous_timestep_baseline_metrics = _baseline_taxel_metrics(
            raw_cmp,
            previous_timestep_baseline,
            dataset_param,
        )

        raw_profile = temporal_profile(raw_cmp)
        pred_profile_raw = temporal_profile(pred_cmp)
        pred_uncalibrated_profile = temporal_profile(pred_uncalibrated_cmp)
        timesteps, raw_profile, pred_profile_raw, pred_uncalibrated_profile = _truncate_profile_window(
            timesteps, raw_profile, pred_profile_raw, pred_uncalibrated_profile
        )
        pred_profile_start_aligned, display_start_shift = _start_aligned_prediction(
            raw_profile, pred_profile_raw
        )
        pred_profile = pred_profile_start_aligned
        residual_profile = raw_profile - pred_profile
        err_profile = np.abs(residual_profile)
        display_timesteps = np.arange(full_steps, dtype=np.int64)
        display_raw_profile = _trace_to_length(temporal_profile(raw_full_cmp), full_steps)
        display_pred_profile = _profile_to_full_window(pred_profile, timesteps, full_steps)
        display_pred_raw = _profile_to_full_window(pred_profile_raw, timesteps, full_steps)
        display_pred_uncalibrated = _profile_to_full_window(pred_uncalibrated_profile, timesteps, full_steps)
        display_pred_start_aligned = _profile_to_full_window(pred_profile_start_aligned, timesteps, full_steps)
        display_residual = display_raw_profile - display_pred_profile
        display_err = np.abs(display_residual)
        abs_error = np.abs(raw_cmp - pred_cmp)
        signed_error = raw_cmp - pred_cmp
        taxel_mae_time = abs_error.mean(axis=0).T if abs_error.ndim >= 3 else np.asarray(abs_error).reshape(1, -1)
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
        raw_accuracy = raw_accuracy_score(raw_cmp, pred_cmp, params=dataset_param)
        contact_metrics = contact_taxel_metrics(raw_cmp, pred_cmp, dataset_param)
        closeness_accuracy = prediction_accuracy(raw_cmp, pred_cmp)
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
                "display_timesteps": display_timesteps,
                "raw": raw_profile,
                "pred": pred_profile,
                "display_raw": display_raw_profile,
                "display_pred": display_pred_profile,
                "raw_cmp": raw_cmp,
                "pred_cmp": pred_cmp,
                "pred_uncalibrated_cmp": pred_uncalibrated_cmp,
                "pred_raw": pred_profile_raw,
                "pred_uncalibrated": pred_uncalibrated_profile,
                "pred_start_aligned": pred_profile_start_aligned,
                "display_pred_raw": display_pred_raw,
                "display_pred_uncalibrated": display_pred_uncalibrated,
                "display_pred_start_aligned": display_pred_start_aligned,
                "residual": residual_profile,
                "err": err_profile,
                "display_residual": display_residual,
                "display_err": display_err,
                "taxel_mae_time": taxel_mae_time,
                "mae": raw_mae,
                "profile_mae": raw_profile_mae,
                "display_profile_mae": display_mae,
                "start_aligned_profile_mae": start_aligned_profile_mae,
                "profile_accuracy": profile_accuracy,
                "accuracy": raw_accuracy,
                "active_taxel_acc": contact_metrics["active_taxel_acc"],
                "active_taxel_mae": contact_metrics["active_taxel_mae"],
                "active_taxel_fraction": contact_metrics["active_taxel_fraction"],
                "contact_region_acc": contact_metrics["contact_region_acc"],
                "contact_region_mae": contact_metrics["contact_region_mae"],
                "peak_acc": contact_metrics["peak_acc"],
                "peak_mae": contact_metrics["peak_mae"],
                "peak_taxel_fraction": contact_metrics["peak_taxel_fraction"],
                "closeness_accuracy": closeness_accuracy,
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
                "zero_baseline": zero_baseline_metrics,
                "train_mean_baseline": train_mean_baseline_metrics,
                "previous_timestep_baseline": previous_timestep_baseline_metrics,
                "display_start_shift": display_start_shift,
                "has_raw": has_raw,
                "has_pred": has_pred,
                "valid_labels": valid_labels,
            }
        )

    images: Dict[str, str] = {}
    files: Dict[str, str] = {}
    metrics: Dict[str, float] = {}
    if not profiles:
        return {"images": images, "metrics": metrics, "files": files}
    profiles = _ordered_profiles(profiles)

    scale_values = [
        np.asarray(profile["raw_cmp"], dtype=np.float32).reshape(-1)
        for profile in profiles
        if profile["active"] and profile["has_raw"] and profile["valid_labels"] and np.asarray(profile["raw_cmp"]).size
    ]
    if not scale_values:
        scale_values = [
            np.asarray(profile["raw_cmp"], dtype=np.float32).reshape(-1)
            for profile in profiles
            if profile["has_raw"] and profile["valid_labels"] and np.asarray(profile["raw_cmp"]).size
        ]
    shared_raw_accuracy_scale = _robust_signal_spread(np.concatenate(scale_values)) if scale_values else None
    for profile in profiles:
        profile["closeness_accuracy"] = prediction_accuracy(
            profile["raw_cmp"],
            profile["pred_cmp"],
            scale=shared_raw_accuracy_scale,
        )
        profile["accuracy"] = raw_accuracy_score(
            profile["raw_cmp"],
            profile["pred_cmp"],
            params=dataset_param,
            scale=shared_raw_accuracy_scale,
        )
        profile["mae_percent"] = mae_percent(profile["mae"], scale=shared_raw_accuracy_scale)
        profile["uncalibrated_raw_accuracy"] = prediction_accuracy(
            profile["raw_cmp"],
            profile["pred_uncalibrated_cmp"],
            scale=shared_raw_accuracy_scale,
        )

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
            "raw_accuracy", "profile_accuracy",
            "raw_tactile_avg", "pred_self_touch_avg", "residual_raw_minus_pred",
            "raw_tactile", "pred_self_touch_raw", "pred_self_touch_start_aligned",
            "display_start_shift", "bias_calibration_shift", "pred_self_touch_uncalibrated_avg",
            "error_margin_abs",
        ])
        for ax, profile in zip(axes, profiles):
            name = profile["name"]
            ts = profile["display_timesteps"]
            raw = profile["display_raw"]
            pred = profile["display_pred"]
            pred_raw = profile["display_pred_raw"]
            pred_uncalibrated = profile["display_pred_uncalibrated"]
            pred_start_aligned = profile["display_pred_start_aligned"]
            residual = profile["display_residual"]
            err = profile["display_err"]
            color = FINGER_COLORS.get(name, "black")
            display_raw = np.asarray(raw, dtype=np.float32)
            display_pred = np.asarray(pred, dtype=np.float32)
            draw_tactile_prediction_profile(
                ax, ts, display_raw, display_pred, err, color
            )
            title_name = name.capitalize()
            ax.set_title(
                f"{title_name} | pred_acc={profile['profile_accuracy']:.1f}%",
                fontsize=9,
            )
            ax.set_ylabel("tactile value")
            ax.set_ylim(TACTILE_YMIN, TACTILE_YMAX)
            ax.set_yticks(TACTILE_YTICKS)
            ax.set_xlabel("Timestep")
            ax.tick_params(axis="x", which="both", labelbottom=True)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")
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
                    float(profile["accuracy"]),
                    float(profile["profile_accuracy"]),
                    _csv_value(raw_value),
                    _csv_value(pred_aligned_value),
                    _csv_value(residual_value),
                    _csv_value(raw_value),
                    _csv_value(pred_raw_value),
                    _csv_value(pred_aligned_value),
                    _csv_value(profile["display_start_shift"]),
                    _csv_value(profile["bias_calibration_shift"]),
                    _csv_value(pred_uncalibrated_value),
                    _csv_value(err_value),
                ])
            metrics[f"{name}_mae"] = profile["mae"]
            metrics[f"{name}_raw_mae"] = profile["mae"]
            metrics[f"{name}_mae_percent"] = profile["mae_percent"]
            metrics[f"{name}_raw_mae_percent"] = profile["mae_percent"]
            metrics[f"{name}_profile_mae"] = profile["profile_mae"]
            metrics[f"{name}_display_profile_mae"] = profile["display_profile_mae"]
            metrics[f"{name}_start_aligned_profile_mae"] = profile["start_aligned_profile_mae"]
            metrics[f"{name}_profile_accuracy"] = profile["profile_accuracy"]
            metrics[f"{name}_line_accuracy"] = profile["profile_accuracy"]
            metrics[f"{name}_accuracy"] = profile["accuracy"]
            metrics[f"{name}_raw_accuracy"] = profile["accuracy"]
            metrics[f"{name}_taxel_raw_accuracy"] = profile["accuracy"]
            metrics[f"{name}_active_taxel_acc"] = profile["active_taxel_acc"]
            metrics[f"{name}_active_taxel_mae"] = profile["active_taxel_mae"]
            metrics[f"{name}_active_taxel_fraction"] = profile["active_taxel_fraction"]
            metrics[f"{name}_contact_region_acc"] = profile["contact_region_acc"]
            metrics[f"{name}_contact_region_mae"] = profile["contact_region_mae"]
            metrics[f"{name}_peak_acc"] = profile["peak_acc"]
            metrics[f"{name}_peak_mae"] = profile["peak_mae"]
            metrics[f"{name}_peak_taxel_fraction"] = profile["peak_taxel_fraction"]
            for baseline_name in ("zero_baseline", "train_mean_baseline", "previous_timestep_baseline"):
                baseline = profile.get(baseline_name, {})
                metrics[f"{name}_{baseline_name}_raw_acc"] = float(baseline.get("raw_acc", 0.0))
                metrics[f"{name}_{baseline_name}_raw_mae"] = float(baseline.get("raw_mae", 0.0))
                metrics[f"{name}_{baseline_name}_active_taxel_acc"] = float(
                    baseline.get("active_taxel_acc", 0.0)
                )
            metrics[f"{name}_closeness_accuracy"] = profile["closeness_accuracy"]
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
        _set_tactile_time_axis(ax, max(TACTILE_XMAX, 1))
    fig.suptitle(TACTILE_PLOT_TITLE, fontsize=12)
    plot_path = os.path.join(plot_dir, f"tactile_profile_epoch_{epoch:04d}.png")
    plt.savefig(plot_path, dpi=160)
    plt.close()
    images["tactile_profile"] = plot_path
    files["tactile_profile_csv"] = csv_path

    identity_fig, identity_axes = plt.subplots(
        1,
        len(profiles),
        figsize=(4.2 * len(profiles), 4.2),
        constrained_layout=True,
    )
    if len(profiles) == 1:
        identity_axes = [identity_axes]
    for ax, profile in zip(identity_axes, profiles):
        raw_flat = np.asarray(profile["raw_cmp"], dtype=np.float32).reshape(-1)
        pred_flat = np.asarray(profile["pred_cmp"], dtype=np.float32).reshape(-1)
        count = min(raw_flat.size, pred_flat.size)
        raw_flat = raw_flat[:count]
        pred_flat = pred_flat[:count]
        finite = np.isfinite(raw_flat) & np.isfinite(pred_flat)
        raw_flat = raw_flat[finite]
        pred_flat = pred_flat[finite]
        if raw_flat.size > 20000:
            keep = np.linspace(0, raw_flat.size - 1, 20000, dtype=np.int64)
            raw_flat = raw_flat[keep]
            pred_flat = pred_flat[keep]
        if raw_flat.size:
            combined = np.concatenate([raw_flat, pred_flat])
            lo, hi = np.percentile(combined, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(combined)), float(np.max(combined) + 1.0)
            ax.scatter(raw_flat, pred_flat, s=3, alpha=0.12, color=FINGER_COLORS.get(profile["name"], "black"))
            ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.5, label="perfect: pred = raw")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("raw tactile")
        ax.set_ylabel("predicted tactile")
        ax.set_title(
            f"{profile['name'].capitalize()} | corr={profile['corr']:.3f} | R²={profile['r2']:.3f}\n"
            f"raw MAE={profile['mae']:.1f} ({profile['mae_percent']:.1f}%) | p95={profile['error_p95']:.1f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
    identity_path = os.path.join(plot_dir, f"tactile_identity_epoch_{epoch:04d}.png")
    identity_fig.suptitle("All-taxel agreement: points should lie on the dashed identity line", fontsize=12)
    identity_fig.savefig(identity_path, dpi=160)
    plt.close(identity_fig)
    images["tactile_identity"] = identity_path

    try:
        combo_pca_path, combo_pca_csv_path, combo_pca_metrics = _save_combination_pca_plot(
            data=data,
            preds=preds,
            epoch=epoch,
            plot_dir=plot_dir,
            dataset_param=dataset_param,
            scaling_param=scaling_param,
            combinations=combinations,
            finger_names=finger_names,
            finger_keys=keys,
            next_step=next_step,
            plt=plt,
        )
    except Exception as exc:
        print(f"[warn] failed to save self-touch combination PCA: {exc}")
        combo_pca_path = None
        combo_pca_csv_path = None
        combo_pca_metrics = {}
    if combo_pca_path:
        images["combination_pca"] = combo_pca_path
    if combo_pca_csv_path:
        files["combination_pca_csv"] = combo_pca_csv_path
    if combo_pca_metrics:
        metrics.update(combo_pca_metrics)

    metric_profiles = [p for p in profiles if p["active"] and p["has_raw"] and p["has_pred"] and p["valid_labels"]]
    if not metric_profiles:
        metric_profiles = [p for p in profiles if p["has_raw"] and p["has_pred"] and p["valid_labels"]]
    if not metric_profiles:
        metric_profiles = [p for p in profiles if p["active"]] or profiles
    metrics["tactile_line_mae"] = float(np.mean([p["mae"] for p in metric_profiles]))
    metrics["tactile_line_raw_mae"] = metrics["tactile_line_mae"]
    metrics["taxel_raw_mae"] = metrics["tactile_line_mae"]
    metrics["tactile_line_mae_percent"] = float(np.mean([p["mae_percent"] for p in metric_profiles]))
    metrics["tactile_line_raw_mae_percent"] = metrics["tactile_line_mae_percent"]
    metrics["tactile_line_profile_mae"] = float(np.mean([p["profile_mae"] for p in metric_profiles]))
    metrics["tactile_line_display_profile_mae"] = float(np.mean([p["display_profile_mae"] for p in metric_profiles]))
    metrics["tactile_line_start_aligned_profile_mae"] = float(
        np.mean([p["start_aligned_profile_mae"] for p in metric_profiles])
    )
    metrics["tactile_line_profile_accuracy"] = float(np.mean([p["profile_accuracy"] for p in metric_profiles]))
    metrics["tactile_taxel_raw_accuracy"] = float(np.mean([p["accuracy"] for p in metric_profiles]))
    metrics["tactile_line_raw_accuracy"] = metrics["tactile_taxel_raw_accuracy"]
    metrics["tactile_line_accuracy"] = metrics["tactile_line_profile_accuracy"]
    metrics["tactile_line_closeness_accuracy"] = float(
        np.mean([p["closeness_accuracy"] for p in metric_profiles])
    )
    metrics["raw_mae"] = metrics["tactile_line_mae"]
    metrics["raw_acc"] = metrics["tactile_taxel_raw_accuracy"]
    metrics["taxel_raw_acc"] = metrics["tactile_taxel_raw_accuracy"]
    metrics["profile_acc"] = metrics["tactile_line_profile_accuracy"]
    metrics["line_acc"] = metrics["tactile_line_profile_accuracy"]
    metrics["raw_mae_percent"] = metrics["tactile_line_mae_percent"]
    metrics["closeness_acc"] = metrics["tactile_line_closeness_accuracy"]
    metrics["active_taxel_acc"] = float(np.mean([p["active_taxel_acc"] for p in metric_profiles]))
    metrics["active_taxel_mae"] = float(np.mean([p["active_taxel_mae"] for p in metric_profiles]))
    metrics["active_taxel_fraction"] = float(np.mean([p["active_taxel_fraction"] for p in metric_profiles]))
    metrics["contact_region_acc"] = float(np.mean([p["contact_region_acc"] for p in metric_profiles]))
    metrics["contact_region_mae"] = float(np.mean([p["contact_region_mae"] for p in metric_profiles]))
    metrics["peak_acc"] = float(np.mean([p["peak_acc"] for p in metric_profiles]))
    metrics["peak_mae"] = float(np.mean([p["peak_mae"] for p in metric_profiles]))
    metrics["peak_taxel_fraction"] = float(np.mean([p["peak_taxel_fraction"] for p in metric_profiles]))
    for baseline_name in ("zero_baseline", "train_mean_baseline", "previous_timestep_baseline"):
        metrics[f"{baseline_name}_raw_acc"] = float(
            np.mean([p.get(baseline_name, {}).get("raw_acc", 0.0) for p in metric_profiles])
        )
        metrics[f"{baseline_name}_raw_mae"] = float(
            np.mean([p.get(baseline_name, {}).get("raw_mae", 0.0) for p in metric_profiles])
        )
        metrics[f"{baseline_name}_active_taxel_acc"] = float(
            np.mean([p.get(baseline_name, {}).get("active_taxel_acc", 0.0) for p in metric_profiles])
        )
    metrics["accuracy_mode"] = _accuracy_mode(dataset_param)
    metrics["accuracy_tolerance"] = _accuracy_tolerance(dataset_param)
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
    metrics["optimizer_step"] = int(optimizer_step or 0)
    metrics["seed"] = int(seed if seed is not None else dataset_param.get("seed", 0))
    metrics["use_tactile_history"] = bool(
        use_tactile_history
        if use_tactile_history is not None
        else dataset_param.get("use_tactile_history", False)
    )

    core_metrics_path = _append_core_metric_history(
        plot_dir,
        epoch,
        metrics,
        [p["name"] for p in profiles],
        pca_image=images.get("combination_pca"),
        pca_csv=files.get("combination_pca_csv"),
    )
    files["tactile_metrics_csv"] = core_metrics_path

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
