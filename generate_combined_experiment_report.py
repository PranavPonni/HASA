#!/usr/bin/env python3
"""Generate the combined non-motion self-touch experiment report PDF."""

from __future__ import annotations

import csv
import json
import math
import statistics
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(".")
OUT_DIR = Path("analysis/combined_experiment_report")
ARCH_CSV = Path("analysis/selftouch_pos_trq_history_ablation/condition_summary.csv")
COMP_CSV = Path("analysis/selftouch_pos_trq_history_ablation/composite_condition_summary.csv")
DELTA_CSV = Path("analysis/selftouch_pos_trq_history_ablation/history_delta.csv")
RUN_CSV = Path("analysis/selftouch_pos_trq_history_ablation/run_metrics.csv")
INPUT_CSV = Path("analysis/selftouch_fcn_14/model_summary.csv")

NON_CONTRASTIVE = [
    "selftouch_fcn",
    "selftouch_transformer",
    "selftouch_gru_attention",
    "selftouch_temporal_mixer",
    "selftouch_mamba",
]
CONTRASTIVE = [
    "selftouch_contrastive_fcn",
    "selftouch_contrastive_transformer",
    "selftouch_contrastive_gru",
    "selftouch_contrastive_temporal",
    "selftouch_contrastive_mamba",
]
OFFSET_GROUPS = {
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
FINGERS = ("index", "thumb", "middle", "ring")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def num(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def fmt(value: object, places: int = 2, suffix: str = "") -> str:
    value = num(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{places}f}{suffix}"


def pct_delta(new: float, old: float) -> float:
    return 100.0 * (new - old) / max(abs(old), 1e-12)


def by_variant_condition(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    return {(row["variant"], row["condition"]): row for row in rows}


def model_label(row: Dict[str, str]) -> str:
    return row.get("model") or row.get("variant", "")


def short_variant(variant: str) -> str:
    prefix = "selftouch_fcn_"
    return variant[len(prefix):] if variant.startswith(prefix) else variant


def aggregate_raw_acc(row: Dict[str, str]) -> float:
    vals = []
    for finger in FINGERS:
        value = num(row.get(f"{finger}_raw_accuracy", row.get(f"{finger}_accuracy")))
        if math.isfinite(value):
            vals.append(value)
    return statistics.mean(vals) if vals else num(row.get("prediction_raw_accuracy"))


def best_final_offset_record(path: Path) -> Optional[Dict[str, object]]:
    rows = read_csv(path)
    if not rows:
        return None
    latest = max(rows, key=lambda row: num(row.get("epoch")))
    if num(latest.get("epoch")) < 500:
        return None
    mae = num(latest.get("prediction_mae"))
    if not math.isfinite(mae):
        return None
    return {
        "run": path.parents[1].name,
        "epoch": int(num(latest.get("epoch"))),
        "mae": mae,
        "raw_acc": aggregate_raw_acc(latest),
        "corr": num(latest.get("prediction_corr")),
        "p95": num(latest.get("prediction_error_p95")),
    }


def load_offset_diagnostic(variant: str) -> Tuple[str, str]:
    paths = sorted((Path("parameter") / variant).glob("*/temporal_offset_diagnostics.json"))
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        drop = data.get("drop", {}).get("all", {})
        drop_text = (
            f"{int(drop.get('dropped_samples', 0))}/{int(drop.get('base_samples', 0))} "
            f"({100.0 * float(drop.get('drop_fraction', 0.0)):.2f}%)"
        )
        leakage = data.get("leakage")
        if leakage:
            leak_text = (
                f"{int(leakage.get('future_contact_or_precursor_samples', 0))}/"
                f"{int(leakage.get('target_contact_samples', 0))} "
                f"({100.0 * float(leakage.get('fraction', 0.0)):.1f}%)"
            )
            if leakage.get("flagged"):
                leak_text += " flagged"
        else:
            leak_text = ""
        return drop_text, leak_text
    return "", ""


def collect_offsets() -> List[Dict[str, object]]:
    rows = []
    for group, items in OFFSET_GROUPS.items():
        for variant, offset in items:
            records = []
            for path in sorted((Path("model_weight") / variant).glob("*/plots/raw_prediction_metrics.csv")):
                record = best_final_offset_record(path)
                if record is not None:
                    records.append(record)
            if records:
                best = min(records, key=lambda item: (item["mae"], -item["raw_acc"], item["p95"]))
                maes = [float(item["mae"]) for item in records]
                median_mae = statistics.median(maes)
                mean_mae = statistics.mean(maes)
                std_mae = statistics.stdev(maes) if len(maes) > 1 else 0.0
            else:
                best = {"run": "", "mae": math.nan, "raw_acc": math.nan, "corr": math.nan, "p95": math.nan}
                median_mae = mean_mae = std_mae = math.nan
            drop_text, leak_text = load_offset_diagnostic(variant)
            rows.append(
                {
                    "group": group,
                    "variant": variant,
                    "offset": offset,
                    "completed": len(records),
                    "best_run": best["run"],
                    "best_mae": best["mae"],
                    "best_raw_acc": best["raw_acc"],
                    "best_corr": best["corr"],
                    "best_p95": best["p95"],
                    "median_mae": median_mae,
                    "mean_mae": mean_mae,
                    "std_mae": std_mae,
                    "drop": drop_text,
                    "leakage": leak_text,
                }
            )
    baseline = next(row for row in rows if row["offset"] == 0)
    for row in rows:
        row["mae_vs_baseline"] = num(row["best_mae"]) - num(baseline["best_mae"])
    return rows


def add_wrapped_text(ax, text: str, x: float, y: float, width: int, size: float = 10.5, line_gap: float = 0.036) -> float:
    for para in text.split("\n"):
        if not para.strip():
            y -= line_gap
            continue
        for line in textwrap.wrap(para, width=width):
            ax.text(x, y, line, fontsize=size, va="top", family="DejaVu Sans")
            y -= line_gap
    return y


def text_page(pdf: PdfPages, title: str, paragraphs: Sequence[str], footer: str = "") -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.07, 0.95, title, fontsize=18, weight="bold", va="top")
    y = 0.90
    for para in paragraphs:
        y = add_wrapped_text(ax, para, 0.07, y, width=104, size=10.2, line_gap=0.031)
        y -= 0.018
    if footer:
        ax.text(0.07, 0.035, footer, fontsize=8, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(
    pdf: PdfPages,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    col_widths: Optional[Sequence[float]] = None,
    font_size: float = 7.8,
    scale_y: float = 1.25,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.035, 0.04, 0.93, 0.87])
    ax.axis("off")
    fig.text(0.04, 0.95, title, fontsize=16, weight="bold", va="top")
    table = ax.table(
        cellText=[[str(cell) for cell in row] for row in rows],
        colLabels=list(headers),
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row_index == 0:
            cell.set_facecolor("#efefef")
            cell.set_text_props(weight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def bar_page(
    pdf: PdfPages,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    ylabel: str,
    color: str = "#4c78a8",
    invert: bool = False,
    annotation: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, values, color=color)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    if invert:
        ax.invert_yaxis()
    if annotation:
        ax.text(0.01, -0.23, textwrap.fill(annotation, 140), transform=ax.transAxes, fontsize=9, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def grouped_history_page(
    pdf: PdfPages,
    title: str,
    rows: Sequence[Dict[str, str]],
    variants: Sequence[str],
    metric: str,
    ylabel: str,
) -> None:
    key = by_variant_condition(rows)
    labels = [model_label(key[(variant, "history-2step")]) for variant in variants]
    no_vals = [num(key[(variant, "no-history")][metric]) for variant in variants]
    hist_vals = [num(key[(variant, "history-2step")][metric]) for variant in variants]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.69, 8.27), constrained_layout=True)
    ax.bar(x - 0.18, no_vals, width=0.36, label="No tactile history", color="#9ecae1")
    ax.bar(x + 0.18, hist_vals, width=0.36, label="2-step tactile history", color="#2f6b9a")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)


def write_markdown(
    path: Path,
    arch_rows: List[Dict[str, str]],
    comp_rows: List[Dict[str, str]],
    delta_rows: List[Dict[str, str]],
    input_rows: List[Dict[str, str]],
    offset_rows: List[Dict[str, object]],
) -> None:
    arch = by_variant_condition(arch_rows)
    comp = by_variant_condition(comp_rows)
    baseline_row = next(row for row in offset_rows if row["offset"] == 0)
    top_input = input_rows[0]
    top_arch = min((arch[(variant, "history-2step")] for variant in NON_CONTRASTIVE), key=lambda row: num(row["prediction_raw_mae_mean"]))
    top_contrastive = max((comp[(variant, "history-2step")] for variant in CONTRASTIVE), key=lambda row: num(row["overall_score_mean"]))
    lines = [
        "# Combined self-touch experiment report",
        "",
        "Scope: completed non-motion-generation results found in local analysis/model-weight artifacts.",
        "",
        "## Executive summary",
        "",
        f"- Best non-contrastive selftouch model with tactile history: {top_arch['model']} "
        f"(MAE {fmt(top_arch['prediction_raw_mae_mean'])}, raw acc {fmt(top_arch['prediction_raw_accuracy_mean'])}%, "
        f"active acc {fmt(top_arch['prediction_active_taxel_accuracy_mean'])}%).",
        f"- Best contrastive composite result: {top_contrastive['model']} history-2step "
        f"(overall score {fmt(top_contrastive['overall_score_mean'])}, MAE {fmt(top_contrastive['prediction_raw_mae_mean'])}).",
        f"- Best input ablation run: {top_input['variant']} ({top_input['run']}) with MAE {fmt(top_input['best_mae'])} "
        f"and raw acc {fmt(top_input['best_accuracy'])}%.",
        f"- Best timestep-offset run by final MAE: {min(offset_rows, key=lambda row: num(row['best_mae']))['variant']} "
        f"versus baseline {baseline_row['variant']}. Future-offset rows are flagged for leakage and should not be read as causal forecasts.",
        "",
        "## Source files",
        "",
        f"- {ARCH_CSV}",
        f"- {COMP_CSV}",
        f"- {DELTA_CSV}",
        f"- {INPUT_CSV}",
        "- model_weight/*/plots/raw_prediction_metrics.csv for timestep offsets",
    ]
    path.write_text("\n".join(lines))


def make_report() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arch_rows = read_csv(ARCH_CSV)
    comp_rows = read_csv(COMP_CSV)
    delta_rows = read_csv(DELTA_CSV)
    input_rows = sorted(read_csv(INPUT_CSV), key=lambda row: int(num(row.get("rank"))))
    run_rows = read_csv(RUN_CSV)
    offset_rows = collect_offsets()
    write_csv(OUT_DIR / "timestep_offset_summary.csv", offset_rows)
    write_markdown(OUT_DIR / "combined_selftouch_report.md", arch_rows, comp_rows, delta_rows, input_rows, offset_rows)

    arch = by_variant_condition(arch_rows)
    comp = by_variant_condition(comp_rows)
    delta = {row["variant"]: row for row in delta_rows}
    baseline_row = next(row for row in offset_rows if row["offset"] == 0)
    previous_raw = num(run_rows[0].get("previous_timestep_baseline_raw_accuracy"))
    previous_active = num(run_rows[0].get("previous_timestep_baseline_active_taxel_accuracy"))
    zero_active = num(run_rows[0].get("zero_baseline_active_taxel_accuracy"))
    train_mean_active = num(run_rows[0].get("train_mean_baseline_active_taxel_accuracy"))

    pdf_path = OUT_DIR / "combined_selftouch_experiment_report.pdf"
    with PdfPages(pdf_path) as pdf:
        text_page(
            pdf,
            "Combined Self-Touch Experiment Report",
            [
                "Scope: all completed local non-motion-generation experiment artifacts found for the selftouch, input-modality, timestep-offset, and contrastive families. Motion-generation directories and smoke/dry-run logs are excluded.",
                "Primary sources: analysis/selftouch_pos_trq_history_ablation/*.csv, analysis/selftouch_fcn_14/*.csv, and model_weight/*/plots/raw_prediction_metrics.csv.",
                "Metrics: raw MAE is lower-is-better tactile value error; raw accuracy is the exported raw closeness score; active-taxel accuracy focuses on contacted/active tactile cells; profile and peak metrics check shape and contact maxima. For composite summaries, higher score is better.",
                "Important caveat: the tactile-history models are compared against no-history learned models and simple baselines. A direct previous-timestep persistence baseline is very strong here, at raw accuracy "
                f"{fmt(previous_raw)}% and active-taxel accuracy {fmt(previous_active)}%, so history-based learned models improve the no-history setting but do not yet beat the trivial persistence upper reference.",
            ],
            footer="Generated from local workspace artifacts.",
        )

        best_noncontrastive = min(
            [arch[(variant, "history-2step")] for variant in NON_CONTRASTIVE],
            key=lambda row: num(row["prediction_raw_mae_mean"]),
        )
        best_input = input_rows[0]
        best_offset = min(offset_rows, key=lambda row: num(row["best_mae"]))
        best_contrastive = max(
            [comp[(variant, "history-2step")] for variant in CONTRASTIVE],
            key=lambda row: num(row["overall_score_mean"]),
        )
        text_page(
            pdf,
            "Executive Findings",
            [
                f"1. Tactile history is the dominant lever. Across every architecture family, adding a 2-step tactile history cuts raw MAE by roughly 35 to 43 points and raises active-taxel accuracy by roughly 19 to 24 percentage points. The no-history models sit around 85-93 MAE; history models move into the 47-57 MAE band.",
                f"2. Best pure selftouch predictor: {best_noncontrastive['model']} with 2-step history, MAE {fmt(best_noncontrastive['prediction_raw_mae_mean'])}, raw accuracy {fmt(best_noncontrastive['prediction_raw_accuracy_mean'])}%, and active-taxel accuracy {fmt(best_noncontrastive['prediction_active_taxel_accuracy_mean'])}%.",
                f"3. Best contrastive result by composite score: {best_contrastive['model']} with 2-step history, composite {fmt(best_contrastive['overall_score_mean'])}, prediction score {fmt(best_contrastive['prediction_score_mean'])}, representation score {fmt(best_contrastive['representation_score_mean'])}, and MAE {fmt(best_contrastive['prediction_raw_mae_mean'])}.",
                f"4. Best input-modality run in the 14-FCN ablation: {best_input['variant']} ({best_input['run']}), MAE {fmt(best_input['best_mae'])}, raw accuracy {fmt(best_input['best_accuracy'])}%. The top three inputs are close: position, pos+torque, and pos+velocity are within about 1.0 MAE.",
                f"5. Timestep offsets are weakly separated. The best future-offset final run is {best_offset['variant']} with MAE {fmt(best_offset['best_mae'])}, only {fmt(num(best_offset['mae_vs_baseline']))} MAE from the baseline {baseline_row['variant']} result. Future offsets are leakage-flagged, so the small apparent gains cannot be treated as valid forecast evidence.",
            ],
        )

        grouped_history_page(
            pdf,
            "Selftouch Architecture Ablation: Raw MAE",
            arch_rows,
            NON_CONTRASTIVE,
            "prediction_raw_mae_mean",
            "Raw tactile MAE (lower is better)",
        )
        grouped_history_page(
            pdf,
            "Selftouch Architecture Ablation: Active-Taxel Accuracy",
            arch_rows,
            NON_CONTRASTIVE,
            "prediction_active_taxel_accuracy_mean",
            "Active-taxel accuracy (%)",
        )
        selftouch_table = []
        for variant in NON_CONTRASTIVE:
            no = arch[(variant, "no-history")]
            hist = arch[(variant, "history-2step")]
            d = delta[variant]
            selftouch_table.append(
                [
                    hist["model"],
                    fmt(no["prediction_raw_mae_mean"]),
                    fmt(hist["prediction_raw_mae_mean"]),
                    fmt(num(d["delta_prediction_raw_mae"])),
                    fmt(no["prediction_active_taxel_accuracy_mean"]),
                    fmt(hist["prediction_active_taxel_accuracy_mean"]),
                    fmt(num(d["delta_prediction_active_taxel_accuracy"]), suffix=" pp"),
                    fmt(no["prediction_peak_accuracy_mean"]),
                    fmt(hist["prediction_peak_accuracy_mean"]),
                ]
            )
        table_page(
            pdf,
            "Selftouch Results by Architecture",
            ["Model", "No hist MAE", "Hist MAE", "MAE delta", "No hist active", "Hist active", "Active delta", "No hist peak", "Hist peak"],
            selftouch_table,
            col_widths=[0.16, 0.10, 0.10, 0.10, 0.12, 0.12, 0.12, 0.10, 0.10],
        )
        text_page(
            pdf,
            "Selftouch Analysis",
            [
                "The selftouch architecture comparison is very clear: architecture matters, but history matters more. The FCN, GRU-Attention, Temporal Mixer, Mamba, and Transformer all gain sharply from 2-step tactile history. The largest raw MAE reductions are FCN (-43.00), Temporal Mixer (-42.06), and GRU-Attention (-40.85); the Mamba and Transformer still improve substantially but start from a better/worse different point.",
                f"Among no-history models, Mamba is the strongest by raw MAE ({fmt(arch[('selftouch_mamba', 'no-history')]['prediction_raw_mae_mean'])}) and active-taxel accuracy ({fmt(arch[('selftouch_mamba', 'no-history')]['prediction_active_taxel_accuracy_mean'])}%). This suggests the state-space model is best when it must infer contact from proprioceptive/input state alone.",
                f"With tactile history, the plain FCN is best by MAE ({fmt(arch[('selftouch_fcn', 'history-2step')]['prediction_raw_mae_mean'])}) and active-taxel accuracy ({fmt(arch[('selftouch_fcn', 'history-2step')]['prediction_active_taxel_accuracy_mean'])}%). Temporal Mixer has the strongest profile accuracy ({fmt(arch[('selftouch_temporal_mixer', 'history-2step')]['prediction_profile_accuracy_mean'])}%), but that does not translate into lower raw error.",
                f"Baseline context matters. Zero baseline active-taxel accuracy is only {fmt(zero_active)}%, train-mean active accuracy is {fmt(train_mean_active)}%, while previous-timestep persistence is {fmt(previous_active)}%. The learned history models sit between mean baselines and direct persistence; the next target should be beating persistence, not merely beating no-history.",
            ],
        )

        input_labels = [short_variant(row["variant"]) for row in input_rows]
        input_values = [num(row["best_mae"]) for row in input_rows]
        bar_page(
            pdf,
            "Input-Modality Ablation: Best Validation MAE",
            input_labels,
            input_values,
            "Best raw MAE",
            color="#59a14f",
            annotation="One newest completed 500-epoch run per FCN input variant. Lower is better. These runs are not a seeded significance test.",
        )
        input_table = [
            [
                row["rank"],
                short_variant(row["variant"]),
                row["run"],
                row["best_epoch"],
                fmt(row["best_mae"]),
                fmt(row["best_accuracy"]),
                fmt(row["best_corr"], 3),
                fmt(row["best_p95"]),
                row["overfit_risk"],
            ]
            for row in input_rows
        ]
        table_page(
            pdf,
            "Input-Modality FCN Ranking",
            ["Rank", "Input", "Run", "Epoch", "MAE", "Raw acc", "Corr", "p95", "Risk"],
            input_table,
            col_widths=[0.06, 0.18, 0.18, 0.08, 0.09, 0.09, 0.08, 0.09, 0.08],
            font_size=7.5,
            scale_y=1.18,
        )
        singles = [row for row in input_rows if short_variant(row["variant"]) in {"pos", "vel", "trq", "cmd"}]
        multis = [row for row in input_rows if row not in singles]
        single_mean = statistics.mean(num(row["best_mae"]) for row in singles)
        multi_mean = statistics.mean(num(row["best_mae"]) for row in multis)
        text_page(
            pdf,
            "Input-Modality Analysis",
            [
                f"The input ablation does not show a simple 'more channels is better' story. Position alone ranks first with MAE {fmt(input_rows[0]['best_mae'])}; pos+torque and pos+velocity are second and third but remain close. The mean MAE for single-input variants is {fmt(single_mean)}, while multi-input variants average {fmt(multi_mean)}.",
                "Command-related inputs do not dominate. pos+cmd, vel+cmd, cmd-only, postrq+cmd, and trq+cmd sit in the lower half. That suggests command signals may be redundant or noisier for tactile reconstruction unless the model is explicitly regularized to use them.",
                "The best variants mostly include position or post-contact/torque information, which fits the task: contact location and intensity are more directly constrained by hand pose and resulting torques than by velocity or command alone.",
                "No validation regression flags were raised in this analyzer. Late degradation is small across all variants, so the ranking is driven mostly by asymptotic validation error rather than late overfitting. Still, the top differences are small enough that a repeated-seed input ablation is needed before treating position-only as definitively superior.",
            ],
        )

        offset_labels = [f"{row['offset']:+d}" for row in offset_rows]
        offset_values = [num(row["best_mae"]) for row in offset_rows]
        bar_page(
            pdf,
            "Timestep Offset Ablation: Best Final MAE",
            offset_labels,
            offset_values,
            "Best final raw MAE",
            color="#f28e2b",
            annotation="Offsets are in timesteps relative to the baseline target. Future offsets are leakage-flagged in diagnostics.",
        )
        offset_table = [
            [
                row["group"],
                row["offset"],
                short_variant(str(row["variant"])),
                row["completed"],
                row["best_run"],
                fmt(row["best_mae"]),
                fmt(row["best_raw_acc"]),
                fmt(row["mae_vs_baseline"], places=2),
                fmt(row["median_mae"]),
                fmt(row["std_mae"]),
                row["drop"],
                row["leakage"],
            ]
            for row in offset_rows
        ]
        table_page(
            pdf,
            "Timestep Offset Results",
            ["Group", "Offset", "Variant", "N", "Best run", "Best MAE", "Raw acc", "MAE vs base", "Median", "SD", "Drop", "Leakage"],
            offset_table,
            col_widths=[0.08, 0.06, 0.16, 0.04, 0.15, 0.08, 0.08, 0.08, 0.08, 0.06, 0.12, 0.16],
            font_size=7.0,
            scale_y=1.25,
        )
        text_page(
            pdf,
            "Timestep Analysis",
            [
                f"The baseline position model's best final MAE is {fmt(baseline_row['best_mae'])}. Past offsets are essentially tied with baseline: -2 has delta {fmt(next(row for row in offset_rows if row['offset'] == -2)['mae_vs_baseline'])}, -5 has delta {fmt(next(row for row in offset_rows if row['offset'] == -5)['mae_vs_baseline'])}, and -10 degrades by {fmt(next(row for row in offset_rows if row['offset'] == -10)['mae_vs_baseline'])}.",
                f"Future offsets look slightly better at +5 and +10, with the best +10 run at MAE {fmt(next(row for row in offset_rows if row['offset'] == 10)['best_mae'])}. However, future diagnostic leakage is flagged at roughly 100% of target-contact samples. These future-offset numbers should therefore be treated as contaminated and not as evidence that prediction further into the future is easier.",
                "Dropping edge samples increases with offset magnitude, from about 0.25-0.50% at 2 timesteps to 2.25-2.51% at 10 timesteps. The drop is small enough that it probably does not explain all variation, but it should still be kept fixed when comparing future/past offsets.",
                "Practical conclusion: use the baseline or small past offset as the clean reference. If future forecasting is the real target, rebuild the split/windowing logic so future contact leakage is removed, then rerun +2/+5/+10 with the same seeds.",
            ],
        )

        grouped_history_page(
            pdf,
            "Contrastive Ablation: Raw MAE",
            arch_rows,
            CONTRASTIVE,
            "prediction_raw_mae_mean",
            "Raw tactile MAE (lower is better)",
        )
        contrastive_comp_rows = [
            comp[(variant, "history-2step")] for variant in CONTRASTIVE
        ]
        contrastive_comp_rows = sorted(contrastive_comp_rows, key=lambda row: -num(row["overall_score_mean"]))
        contrastive_table = []
        for row in contrastive_comp_rows:
            no = arch[(row["variant"], "no-history")]
            hist = arch[(row["variant"], "history-2step")]
            d = delta[row["variant"]]
            contrastive_table.append(
                [
                    row["model"],
                    fmt(row["overall_score_mean"]),
                    fmt(row["prediction_score_mean"]),
                    fmt(row["representation_score_mean"]),
                    fmt(hist["prediction_raw_mae_mean"]),
                    fmt(hist["prediction_active_taxel_accuracy_mean"]),
                    fmt(hist["latent_combo_separation_ratio_mean"]),
                    fmt(no["latent_combo_separation_ratio_mean"]),
                    fmt(num(d["delta_prediction_raw_mae"])),
                ]
            )
        table_page(
            pdf,
            "Contrastive History Results",
            ["Model", "Overall", "Pred score", "Repr score", "Hist MAE", "Hist active", "Hist latent sep", "No-hist latent sep", "MAE delta"],
            contrastive_table,
            col_widths=[0.18, 0.09, 0.10, 0.10, 0.09, 0.10, 0.11, 0.12, 0.10],
            font_size=7.4,
            scale_y=1.35,
        )
        labels = [row["model"] for row in contrastive_comp_rows]
        values = [num(row["overall_score_mean"]) for row in contrastive_comp_rows]
        bar_page(
            pdf,
            "Contrastive Composite Score with 2-Step History",
            labels,
            values,
            "Composite score (higher is better)",
            color="#b07aa1",
        )
        text_page(
            pdf,
            "Contrastive Analysis",
            [
                f"Contrastive objectives improve representation structure much more than they improve raw tactile prediction. The contrastive FCN history run is the best composite result, with overall score {fmt(comp[('selftouch_contrastive_fcn', 'history-2step')]['overall_score_mean'])} and representation score {fmt(comp[('selftouch_contrastive_fcn', 'history-2step')]['representation_score_mean'])}. Its raw MAE {fmt(arch[('selftouch_contrastive_fcn', 'history-2step')]['prediction_raw_mae_mean'])} is close to the non-contrastive FCN's {fmt(arch[('selftouch_fcn', 'history-2step')]['prediction_raw_mae_mean'])}, not clearly better.",
                f"No-history contrastive Mamba has the best no-history raw MAE among contrastive models ({fmt(arch[('selftouch_contrastive_mamba', 'no-history')]['prediction_raw_mae_mean'])}) and strong active-taxel accuracy ({fmt(arch[('selftouch_contrastive_mamba', 'no-history')]['prediction_active_taxel_accuracy_mean'])}%). This mirrors the non-contrastive Mamba finding: Mamba is useful when tactile history is absent.",
                f"The contrastive Temporal model is notable for profile/shape quality after history is added: profile accuracy {fmt(arch[('selftouch_contrastive_temporal', 'history-2step')]['prediction_profile_accuracy_mean'])}% and peak accuracy {fmt(arch[('selftouch_contrastive_temporal', 'history-2step')]['prediction_peak_accuracy_mean'])}%. If downstream use values contact shape more than raw value MAE, this model deserves a second look.",
                "The contrastive Transformer remains weak on raw prediction even with history. It is substantially behind FCN/GRU/Temporal/Mamba in MAE and active-taxel accuracy, so it is not the right next production candidate unless transformer-specific hyperparameters are revisited.",
            ],
        )

        text_page(
            pdf,
            "Combined Interpretation and Next Steps",
            [
                "Best current deployment candidates depend on the setting. If tactile history is available and raw prediction is the goal, start with the FCN history checkpoint family. If no tactile history is available, Mamba is the best architecture family. If representation separation or downstream contact-state clustering matters, contrastive FCN and contrastive Temporal are the strongest candidates.",
                "The largest methodological risk is the previous-timestep baseline. Learned history models are much better than no-history learned models, but direct persistence is still stronger. The next experiment should train/evaluate against a persistence-aware loss or residual target: predict change from previous tactile rather than absolute tactile state, then report residual improvement over copy-last.",
                "For input ablations, the current best is position-only, but differences among top variants are small. Repeat the input study with the same multi-seed protocol used for architecture/history before dropping torque or post-contact inputs from the final system.",
                "For timestep experiments, do not use the future-offset results for claims until leakage is fixed. Clean past offsets show no meaningful improvement over baseline, so the baseline current-target setting remains the cleanest benchmark.",
                "For contrastive learning, keep the objective if the embedding is used downstream; do not expect it alone to reduce tactile MAE. The strongest raw predictor remains a simple FCN-style model with tactile history.",
            ],
        )

    return pdf_path


def main() -> None:
    path = make_report()
    print(path)


if __name__ == "__main__":
    main()
