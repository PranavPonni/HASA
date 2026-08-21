#!/usr/bin/env python3
"""Generate architecture diagrams for the self-touch ablation studies.

The diagrams are intentionally simple and close to the reference image:
input boxes on the left, a model/backbone block in the center, and tactile
prediction heads on the right.  SVG is the source of truth; PNG copies and
vertical contact sheets are emitted when Pillow is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import argparse
import math
import textwrap

import yaml

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - SVG generation still works without PIL.
    Image = None
    ImageDraw = None
    ImageFont = None


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "analysis" / "selftouch_ablation_diagrams"

FCN_INPUT_VARIANTS = [
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

POSITION_OFFSET_VARIANTS = [
    ("selftouch_fcn_pos", 0),
    ("selftouch_fcn_pos_tplus2", 2),
    ("selftouch_fcn_pos_tplus5", 5),
    ("selftouch_fcn_pos_tplus10", 10),
    ("selftouch_fcn_pos_tminus2", -2),
    ("selftouch_fcn_pos_tminus5", -5),
    ("selftouch_fcn_pos_tminus10", -10),
]

POS_TRQ_BACKBONE_VARIANTS = [
    "selftouch_gru_attention",
    "selftouch_temporal_mixer",
    "selftouch_fcn",
    "selftouch_transformer",
    "selftouch_mamba",
    "selftouch_contrastive_gru",
    "selftouch_contrastive_temporal",
    "selftouch_contrastive_fcn",
    "selftouch_contrastive_transformer",
    "selftouch_contrastive_mamba",
]

FINGER_NAMES = ("index", "thumb", "middle", "ring")

MODALITY_LABELS = {
    "hand_jnt_pos": ("Joint Positions", "hand_jnt_pos"),
    "hand_jnt_vel": ("Joint Velocities", "hand_jnt_vel"),
    "hand_jnt_trq": ("Joint Torques", "hand_jnt_trq"),
    "hand_jnt_cmd_pos": ("Commanded Joint Positions", "hand_jnt_cmd_pos"),
}

SHORT_MODALITY_NAMES = {
    "hand_jnt_pos": "position",
    "hand_jnt_vel": "velocity",
    "hand_jnt_trq": "torque",
    "hand_jnt_cmd_pos": "command",
}

BACKBONE_LABELS = {
    "gru_attention": ("GRU + Attn", "causal recurrent attention"),
    "tsmixer": ("TSMixer", "temporal/channel mixing"),
    "tcn": ("TCN / FCN", "dilated temporal conv"),
    "patchtst": ("PatchTST", "causal patch transformer"),
    "mamba": ("Mamba", "state-space sequence model"),
}

COLORS = {
    "page": "#ffffff",
    "panel": "#f2f2f2",
    "panel_stroke": "#7b7b7b",
    "text": "#24272a",
    "muted": "#5f6368",
    "black": "#111111",
    "model_fill": "#d9d9d9",
    "joint_fill": "#fff5ef",
    "joint_stroke": "#ff7043",
    "context_fill": "#eef6ff",
    "context_stroke": "#3778bf",
    "history_fill": "#f5f0ff",
    "history_stroke": "#7e57c2",
    "output_fill": "#eef9ec",
    "output_stroke": "#39a935",
    "aux_fill": "#effaf9",
    "aux_stroke": "#279c97",
    "note_fill": "#fffdf7",
    "note_stroke": "#d1b46b",
}


@dataclass(frozen=True)
class BoxSpec:
    title: str
    lines: tuple[str, ...]
    fill: str
    stroke: str
    title_color: str = COLORS["text"]


@dataclass(frozen=True)
class DiagramSpec:
    slug: str
    family: str
    title: str
    subtitle: str
    inputs: tuple[BoxSpec, ...]
    processor_title: str
    processor_lines: tuple[str, ...]
    outputs: tuple[BoxSpec, ...]
    notes: tuple[str, ...]


def load_params(variant: str) -> dict:
    path = ROOT / "parameter" / variant / "parameter_base" / "parameter_base.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def modality_names(modalities: list[str] | tuple[str, ...]) -> str:
    return " + ".join(SHORT_MODALITY_NAMES.get(name, name) for name in modalities)


def offset_label(offset: int) -> str:
    if offset == 0:
        return "t"
    sign = "+" if offset > 0 else ""
    return f"t{sign}{offset}"


def offset_slug(offset: int) -> str:
    if offset == 0:
        return "baseline"
    sign = "plus" if offset > 0 else "minus"
    return f"t{sign}{abs(offset)}"


def target_window(num_timesteps: int, input_offset: int, first_target_index: int = 1) -> tuple[int, int]:
    start = max(first_target_index, -int(input_offset))
    stop = int(num_timesteps)
    if input_offset > 0:
        stop = min(stop, int(num_timesteps) - int(input_offset))
    start = min(max(start, 0), int(num_timesteps))
    stop = min(max(stop, 0), int(num_timesteps))
    if stop < start:
        stop = start
    return start, stop


def dropped_target_count(num_timesteps: int, input_offset: int, first_target_index: int = 1) -> int:
    base = max(0, int(num_timesteps) - int(first_target_index))
    start, stop = target_window(num_timesteps, input_offset, first_target_index)
    return max(0, base - max(0, stop - start))


def tactile_output_lines(model_cfg: dict) -> tuple[str, str]:
    tactile_dim = int(model_cfg.get("tactile_dim", 90))
    per_finger = f"{tactile_dim} dims per fingertip"
    if tactile_dim % 3 == 0:
        per_finger = f"{tactile_dim // 3} x 3 per fingertip"
    total = tactile_dim * len(FINGER_NAMES)
    return (
        "Index / thumb / middle / ring",
        f"4 x {tactile_dim} = {total} dims ({per_finger})",
    )


def joint_input_boxes(model_cfg: dict, *, frame: str = "t", window_suffix: str | None = None) -> list[BoxSpec]:
    hand_dim = int(model_cfg.get("hand_dim", 16))
    boxes: list[BoxSpec] = []
    for modality in model_cfg.get("input_modalities", ("hand_jnt_pos",)):
        label, key = MODALITY_LABELS.get(modality, (modality, modality))
        lines = [f"{key}({frame})", f"{hand_dim} dims/frame"]
        if window_suffix:
            lines.append(window_suffix)
        boxes.append(
            BoxSpec(
                title=label,
                lines=tuple(lines),
                fill=COLORS["joint_fill"],
                stroke=COLORS["joint_stroke"],
            )
        )
    return boxes


def common_context_boxes(model_cfg: dict) -> list[BoxSpec]:
    boxes: list[BoxSpec] = []
    if bool(model_cfg.get("use_combo_condition", False)):
        boxes.append(
            BoxSpec(
                title="Self-touch Combo",
                lines=(f"{int(model_cfg.get('combo_dim', 6))} contact classes", "conditioning input"),
                fill=COLORS["context_fill"],
                stroke=COLORS["context_stroke"],
            )
        )
    if bool(model_cfg.get("use_phase_condition", False)):
        boxes.append(
            BoxSpec(
                title="Self-touch Phase",
                lines=(f"{int(model_cfg.get('phase_dim', 10))} dims", "phase conditioning"),
                fill=COLORS["context_fill"],
                stroke=COLORS["context_stroke"],
            )
        )
    return boxes


def tactile_history_box(model_cfg: dict, *, target_frame: str = "t") -> BoxSpec | None:
    if not bool(model_cfg.get("use_tactile_history", False)):
        return None
    steps = int(model_cfg.get("tactile_history_steps", 1))
    tactile_dim = int(model_cfg.get("tactile_dim", 90))
    fingers = tuple(model_cfg.get("tactile_history_fingers", FINGER_NAMES))
    if target_frame == "t+1":
        history_range = "t" if steps == 1 else f"t ... t-{steps - 1}"
    else:
        history_range = f"{target_frame}-1" if steps == 1 else f"{target_frame}-1 ... {target_frame}-{steps}"
    return BoxSpec(
        title="Tactile History",
        lines=(
            history_range,
            f"{len(fingers)} fingers x {tactile_dim} x {steps}",
        ),
        fill=COLORS["history_fill"],
        stroke=COLORS["history_stroke"],
    )


def fcn_input_diagram(index: int, variant: str) -> DiagramSpec:
    params = load_params(variant)
    model = params["Model"]
    modalities = tuple(model.get("input_modalities", ("hand_jnt_pos",)))
    temporal_steps = int(model.get("temporal_window_steps", 1))
    use_deltas = bool(model.get("use_pos_temporal_deltas", model.get("use_input_temporal_deltas", True)))
    suffix = f"{temporal_steps}-step causal window"
    if use_deltas:
        suffix += " + deltas"
    inputs = joint_input_boxes(model, frame="t", window_suffix=suffix)
    inputs.extend(common_context_boxes(model))
    history = tactile_history_box(model, target_frame="t")
    if history:
        inputs.append(history)
    output_lines = tactile_output_lines(model)
    return DiagramSpec(
        slug=f"{index:02d}_{variant}",
        family="fcn_input_ablation",
        title=f"FCN Input Ablation {index:02d}/14: {modality_names(modalities).title()}",
        subtitle=f"{variant} | target tactile frame: t",
        inputs=tuple(inputs),
        processor_title="FCN",
        processor_lines=(
            str(model.get("architecture", "temporal_mean_residual")).replace("_", " "),
            "encoder -> temporal blocks -> decoder",
            "mean trace + taxel residual heads",
        ),
        outputs=(
            BoxSpec(
                title="Predicted Self-touch Tactile(t)",
                lines=output_lines,
                fill=COLORS["output_fill"],
                stroke=COLORS["output_stroke"],
                title_color=COLORS["output_stroke"],
            ),
        ),
        notes=(
            "Ablated variable: joint input stream set.",
            f"Shared context: combo={bool(model.get('use_combo_condition'))}, phase={bool(model.get('use_phase_condition'))}, tactile history={bool(model.get('use_tactile_history'))}.",
        ),
    )


def position_offset_diagram(index: int, variant: str, offset: int) -> DiagramSpec:
    params = load_params(variant)
    model = params["Model"]
    dataset = params.get("Dataset", {})
    sequence_length = int(dataset.get("sequence_length", 400))
    start, stop = target_window(sequence_length, offset)
    dropped = dropped_target_count(sequence_length, offset)
    target_count = max(0, sequence_length - 1)
    frame = offset_label(offset)
    if offset > 0:
        condition = f"future input {frame}"
    elif offset < 0:
        condition = f"past input {frame}"
    else:
        condition = "baseline input t"
    temporal_steps = int(model.get("temporal_window_steps", 1))
    suffix = f"{temporal_steps}-step causal window + deltas"
    inputs = joint_input_boxes(model, frame=frame, window_suffix=suffix)
    inputs.extend(common_context_boxes(model))
    history = tactile_history_box(model, target_frame="t")
    if history:
        inputs.append(history)
    output_lines = tactile_output_lines(model)
    target_range = "none"
    if stop > start:
        target_range = f"{start}..{stop - 1}"
    return DiagramSpec(
        slug=f"{index:02d}_{variant}_{offset_slug(offset)}",
        family="position_offset_ablation",
        title=f"Position Offset Ablation {index:02d}/07: {condition}",
        subtitle=f"{variant} | input_offset={offset:+d} | predict tactile frame t",
        inputs=tuple(inputs),
        processor_title="FCN",
        processor_lines=(
            "position temporal mean-residual",
            f"input frame: {frame}",
            f"valid targets: {target_range}",
        ),
        outputs=(
            BoxSpec(
                title="Predicted Self-touch Tactile(t)",
                lines=output_lines,
                fill=COLORS["output_fill"],
                stroke=COLORS["output_stroke"],
                title_color=COLORS["output_stroke"],
            ),
        ),
        notes=(
            f"Offset ablation changes only the position input frame: input = target t {offset:+d}.",
            f"Drops {dropped}/{target_count} targets per {sequence_length}-step episode.",
        ),
    )


def backbone_diagram(index: int, variant: str) -> DiagramSpec:
    params = load_params(variant)
    model = params["Model"]
    backbone = str(model.get("backbone", model.get("architecture", "backbone"))).lower()
    backbone_title, backbone_detail = BACKBONE_LABELS.get(backbone, (backbone.upper(), "sequence backbone"))
    contrastive = bool(model.get("contrastive_encoder", False))
    inputs = joint_input_boxes(model, frame="t", window_suffix="next-step causal input")
    history = tactile_history_box(model, target_frame="t+1")
    if history:
        inputs.append(history)
    inputs.append(
        BoxSpec(
            title="Self-touch Combo",
            lines=("6 contact classes", "decoder routing"),
            fill=COLORS["context_fill"],
            stroke=COLORS["context_stroke"],
        )
    )
    processor_lines = [
        "finger-aware pos+trq encoder",
        backbone_detail,
        "combo-specific tactile decoders",
    ]
    if contrastive:
        processor_lines.append("contrastive projection/class heads")
    outputs: list[BoxSpec] = [
        BoxSpec(
            title="Predicted Self-touch Tactile(t+1)",
            lines=tactile_output_lines(model),
            fill=COLORS["output_fill"],
            stroke=COLORS["output_stroke"],
            title_color=COLORS["output_stroke"],
        )
    ]
    if contrastive:
        outputs.append(
            BoxSpec(
                title="Contrastive Outputs",
                lines=(
                    f"{int(model.get('projection_dim', 64))}-dim projection",
                    f"{int(model.get('num_classes', 6))}-class combo logits",
                ),
                fill=COLORS["aux_fill"],
                stroke=COLORS["aux_stroke"],
                title_color=COLORS["aux_stroke"],
            )
        )
    history_note = "2-step tactile history" if bool(model.get("use_tactile_history", False)) else "no tactile history"
    return DiagramSpec(
        slug=f"{index:02d}_{variant}",
        family="pos_trq_backbone_ablation",
        title=f"Pos+Trq Backbone Ablation {index:02d}/10: {backbone_title}",
        subtitle=f"{variant} | {model.get('experiment_condition', history_note)}",
        inputs=tuple(inputs),
        processor_title=backbone_title,
        processor_lines=tuple(processor_lines),
        outputs=tuple(outputs),
        notes=(
            "Ablated variable: sequence backbone and contrastive objective.",
            f"Input streams fixed to position + torque; {history_note}.",
        ),
    )


def general_contrastive_fcn_pos_diagram() -> DiagramSpec:
    params = load_params("selftouch_fcn_pos")
    model = params["Model"]
    hand_dim = int(model.get("hand_dim", 16))
    temporal_steps = int(model.get("temporal_window_steps", 20))
    return DiagramSpec(
        slug="01_general_contrastive_fcn_pos",
        family="general_contrastive_fcn_pos",
        title="General Contrastive Learning: FCN Position Encoder",
        subtitle="Example setup using hand_jnt_pos windows and self-touch combo labels",
        inputs=(
            BoxSpec(
                title="Position View A",
                lines=(
                    "hand_jnt_pos(t)",
                    f"{hand_dim} dims/frame, {temporal_steps}-step window",
                ),
                fill=COLORS["joint_fill"],
                stroke=COLORS["joint_stroke"],
            ),
            BoxSpec(
                title="Positive View B",
                lines=(
                    "same combo / nearby window",
                    "paired or augmented sample",
                ),
                fill=COLORS["joint_fill"],
                stroke=COLORS["joint_stroke"],
            ),
            BoxSpec(
                title="Batch Negatives",
                lines=(
                    "other windows in batch",
                    "different combo/contact class",
                ),
                fill=COLORS["context_fill"],
                stroke=COLORS["context_stroke"],
            ),
            BoxSpec(
                title="Combo Label",
                lines=(
                    "6 self-touch classes",
                    "supervised contrastive target",
                ),
                fill=COLORS["history_fill"],
                stroke=COLORS["history_stroke"],
            ),
        ),
        processor_title="FCN",
        processor_lines=(
            "shared weights for all views",
            "position temporal encoder",
            "projection head",
        ),
        outputs=(
            BoxSpec(
                title="Projected Embeddings",
                lines=(
                    "normalize z_a and z_b",
                    "representation space",
                ),
                fill=COLORS["aux_fill"],
                stroke=COLORS["aux_stroke"],
                title_color=COLORS["aux_stroke"],
            ),
            BoxSpec(
                title="Contrastive Loss",
                lines=(
                    "pull positives together",
                    "push negatives apart",
                ),
                fill=COLORS["output_fill"],
                stroke=COLORS["output_stroke"],
                title_color=COLORS["output_stroke"],
            ),
        ),
        notes=(
            "Conceptual setup: learn FCN position embeddings with positives, negatives, and combo labels.",
            "The encoder can feed a tactile head later.",
        ),
    )


def wrap_chars(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    for part in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(part, width=max_chars, break_long_words=False)
        lines.extend(wrapped or [""])
    return lines


class SvgCanvas:
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8.5" refY="5" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L10,5 L0,10 Z" fill="#111111" />',
            "</marker>",
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{COLORS["page"]}" />',
        ]

    def rounded_rect(self, x: float, y: float, w: float, h: float, r: float, fill: str, stroke: str, sw: float = 3):
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}" ry="{r:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, sw: float = 4, arrow: bool = False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{COLORS["black"]}" stroke-width="{sw}" stroke-linecap="round"{marker} />'
        )

    def text(self, x: float, y: float, text: str, size: int, weight: int = 400, color: str = COLORS["text"], anchor: str = "start"):
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    def text_block(self, x: float, y: float, lines: list[str], size: int, color: str, anchor: str = "start", weight: int = 400, line_gap: int = 22):
        for idx, line in enumerate(lines):
            self.text(x, y + idx * line_gap, line, size, weight=weight, color=color, anchor=anchor)

    def save(self, path: Path):
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def find_font(bold: bool = False, size: int = 24):
    if ImageFont is None:
        return None
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_width(draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def wrap_pixels(draw, text: str, font, max_width: int) -> list[str]:
    result: list[str] = []
    for raw in str(text).splitlines() or [""]:
        words = raw.split()
        if not words:
            result.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if text_width(draw, candidate, font) <= max_width:
                line = candidate
            else:
                result.append(line)
                line = word
        result.append(line)
    return result


def draw_arrow(draw, start: tuple[float, float], end: tuple[float, float], width: int = 4, fill: str = COLORS["black"]):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=fill, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 13
    spread = math.radians(28)
    p1 = (x2, y2)
    p2 = (x2 - head * math.cos(angle - spread), y2 - head * math.sin(angle - spread))
    p3 = (x2 - head * math.cos(angle + spread), y2 - head * math.sin(angle + spread))
    draw.polygon((p1, p2, p3), fill=fill)


def diagram_geometry(spec: DiagramSpec) -> dict[str, int]:
    width = 1180
    top = 122
    box_h = 82
    gap = 14
    output_h = 94
    input_stack = len(spec.inputs) * box_h + max(0, len(spec.inputs) - 1) * gap
    output_stack = len(spec.outputs) * output_h + max(0, len(spec.outputs) - 1) * 18
    content = max(input_stack, output_stack, 250)
    height = max(470, top + content + 104)
    return {
        "width": width,
        "height": height,
        "top": top,
        "box_h": box_h,
        "gap": gap,
        "output_h": output_h,
        "content": content,
    }


def box_centers(y: float, count: int, height: float, gap: float) -> list[float]:
    return [y + idx * (height + gap) + height / 2 for idx in range(count)]


def render_svg(spec: DiagramSpec, path: Path):
    g = diagram_geometry(spec)
    w = g["width"]
    h = g["height"]
    top = g["top"]
    box_h = g["box_h"]
    gap = g["gap"]
    output_h = g["output_h"]
    content = g["content"]

    left_x, left_w = 108, 292
    processor_x, processor_w, processor_h = 500, 200, 250
    output_x, output_w = 750, 360
    bus_x = 452

    input_stack = len(spec.inputs) * box_h + max(0, len(spec.inputs) - 1) * gap
    output_stack = len(spec.outputs) * output_h + max(0, len(spec.outputs) - 1) * 18
    input_y = top + (content - input_stack) / 2
    output_y = top + (content - output_stack) / 2
    processor_y = top + (content - processor_h) / 2
    center_y = top + content / 2

    svg = SvgCanvas(w, h)
    svg.rounded_rect(42, 32, w - 84, h - 64, 86, COLORS["panel"], COLORS["panel_stroke"], 3)
    svg.text(86, 72, spec.title, 27, weight=700, color=COLORS["text"])
    svg.text(86, 101, spec.subtitle, 17, weight=400, color=COLORS["muted"])

    input_centers = box_centers(input_y, len(spec.inputs), box_h, gap)
    for box, cy in zip(spec.inputs, input_centers):
        y = cy - box_h / 2
        svg.rounded_rect(left_x, y, left_w, box_h, 18, box.fill, box.stroke, 3)
        title_lines = wrap_chars(box.title, 24)
        line_y = y + 25
        svg.text(left_x + left_w / 2, line_y, title_lines[0], 17, weight=700, color=box.title_color, anchor="middle")
        body_lines = []
        for line in box.lines:
            body_lines.extend(wrap_chars(line, 28))
        body_lines = body_lines[:2]
        svg.text_block(left_x + left_w / 2, line_y + 25, body_lines, 14, COLORS["text"], anchor="middle", line_gap=19)
        svg.line(left_x + left_w, cy, bus_x, cy, sw=4)

    if input_centers:
        svg.line(bus_x, min(input_centers), bus_x, max(input_centers), sw=4)
        svg.line(bus_x, center_y, processor_x - 4, center_y, sw=4, arrow=True)

    svg.rounded_rect(processor_x, processor_y, processor_w, processor_h, 24, COLORS["model_fill"], COLORS["black"], 4)
    processor_title_size = 29 if len(spec.processor_title) > 7 else 31
    svg.text(processor_x + processor_w / 2, processor_y + 78, spec.processor_title, processor_title_size, weight=700, anchor="middle")
    proc_lines: list[str] = []
    for line in spec.processor_lines:
        proc_lines.extend(wrap_chars(line, 18))
    svg.text_block(
        processor_x + processor_w / 2,
        processor_y + 116,
        proc_lines[:5],
        14,
        COLORS["text"],
        anchor="middle",
        line_gap=18,
    )

    output_centers = box_centers(output_y, len(spec.outputs), output_h, 18)
    if output_centers:
        if len(output_centers) == 1:
            svg.line(processor_x + processor_w, center_y, output_x - 5, output_centers[0], sw=4, arrow=True)
        else:
            out_bus_x = output_x - 46
            svg.line(processor_x + processor_w, center_y, out_bus_x, center_y, sw=4)
            svg.line(out_bus_x, min(output_centers), out_bus_x, max(output_centers), sw=4)
            for cy in output_centers:
                svg.line(out_bus_x, cy, output_x - 5, cy, sw=4, arrow=True)

    for box, cy in zip(spec.outputs, output_centers):
        y = cy - output_h / 2
        svg.rounded_rect(output_x, y, output_w, output_h, 18, box.fill, box.stroke, 3)
        svg.text(output_x + output_w / 2, y + 29, box.title, 17, weight=700, color=box.title_color, anchor="middle")
        body_lines = []
        for line in box.lines:
            body_lines.extend(wrap_chars(line, 32))
        svg.text_block(output_x + output_w / 2, y + 54, body_lines[:2], 14, COLORS["text"], anchor="middle", line_gap=18)

    note_y = h - 72
    note_text = "  |  ".join(spec.notes)
    for idx, line in enumerate(wrap_chars(note_text, 118)[:2]):
        svg.text(86, note_y + idx * 20, line, 14, color=COLORS["muted"])

    svg.save(path)


def render_png(spec: DiagramSpec, path: Path):
    if Image is None or ImageDraw is None:
        return
    scale = 2
    g = diagram_geometry(spec)
    w = g["width"] * scale
    h = g["height"] * scale
    img = Image.new("RGB", (w, h), COLORS["page"])
    draw = ImageDraw.Draw(img)

    def s(value: float) -> int:
        return int(round(value * scale))

    title_font = find_font(True, 27 * scale)
    subtitle_font = find_font(False, 17 * scale)
    box_title_font = find_font(True, 17 * scale)
    body_font = find_font(False, 14 * scale)
    processor_font = find_font(True, 31 * scale)

    top = g["top"]
    box_h = g["box_h"]
    gap = g["gap"]
    output_h = g["output_h"]
    content = g["content"]
    left_x, left_w = 108, 292
    processor_x, processor_w, processor_h = 500, 200, 250
    output_x, output_w = 750, 360
    bus_x = 452

    input_stack = len(spec.inputs) * box_h + max(0, len(spec.inputs) - 1) * gap
    output_stack = len(spec.outputs) * output_h + max(0, len(spec.outputs) - 1) * 18
    input_y = top + (content - input_stack) / 2
    output_y = top + (content - output_stack) / 2
    processor_y = top + (content - processor_h) / 2
    center_y = top + content / 2

    draw.rounded_rectangle(
        (s(42), s(32), s(g["width"] - 42), s(g["height"] - 32)),
        radius=s(86),
        fill=COLORS["panel"],
        outline=COLORS["panel_stroke"],
        width=s(3),
    )
    draw.text((s(86), s(48)), spec.title, fill=COLORS["text"], font=title_font)
    draw.text((s(86), s(82)), spec.subtitle, fill=COLORS["muted"], font=subtitle_font)

    input_centers = box_centers(input_y, len(spec.inputs), box_h, gap)
    for box, cy in zip(spec.inputs, input_centers):
        y = cy - box_h / 2
        draw.rounded_rectangle(
            (s(left_x), s(y), s(left_x + left_w), s(y + box_h)),
            radius=s(18),
            fill=box.fill,
            outline=box.stroke,
            width=s(3),
        )
        title_lines = wrap_pixels(draw, box.title, box_title_font, s(left_w - 28))
        title = title_lines[0]
        tw = text_width(draw, title, box_title_font)
        draw.text((s(left_x + left_w / 2) - tw // 2, s(y + 12)), title, fill=box.title_color, font=box_title_font)
        body_lines: list[str] = []
        for line in box.lines:
            body_lines.extend(wrap_pixels(draw, line, body_font, s(left_w - 30)))
        for idx, line in enumerate(body_lines[:2]):
            tw = text_width(draw, line, body_font)
            draw.text(
                (s(left_x + left_w / 2) - tw // 2, s(y + 42 + idx * 20)),
                line,
                fill=COLORS["text"],
                font=body_font,
            )
        draw.line((s(left_x + left_w), s(cy), s(bus_x), s(cy)), fill=COLORS["black"], width=s(4))

    if input_centers:
        draw.line((s(bus_x), s(min(input_centers)), s(bus_x), s(max(input_centers))), fill=COLORS["black"], width=s(4))
        draw_arrow(draw, (s(bus_x), s(center_y)), (s(processor_x - 5), s(center_y)), width=s(4))

    draw.rounded_rectangle(
        (s(processor_x), s(processor_y), s(processor_x + processor_w), s(processor_y + processor_h)),
        radius=s(24),
        fill=COLORS["model_fill"],
        outline=COLORS["black"],
        width=s(4),
    )
    if len(spec.processor_title) > 7:
        processor_font = find_font(True, 29 * scale)
    tw = text_width(draw, spec.processor_title, processor_font)
    draw.text(
        (s(processor_x + processor_w / 2) - tw // 2, s(processor_y + 48)),
        spec.processor_title,
        fill=COLORS["text"],
        font=processor_font,
    )
    proc_lines: list[str] = []
    for line in spec.processor_lines:
        proc_lines.extend(wrap_pixels(draw, line, body_font, s(processor_w - 22)))
    for idx, line in enumerate(proc_lines[:5]):
        tw = text_width(draw, line, body_font)
        draw.text(
            (s(processor_x + processor_w / 2) - tw // 2, s(processor_y + 104 + idx * 18)),
            line,
            fill=COLORS["text"],
            font=body_font,
        )

    output_centers = box_centers(output_y, len(spec.outputs), output_h, 18)
    if output_centers:
        if len(output_centers) == 1:
            draw_arrow(
                draw,
                (s(processor_x + processor_w), s(center_y)),
                (s(output_x - 7), s(output_centers[0])),
                width=s(4),
            )
        else:
            out_bus_x = output_x - 46
            draw.line((s(processor_x + processor_w), s(center_y), s(out_bus_x), s(center_y)), fill=COLORS["black"], width=s(4))
            draw.line((s(out_bus_x), s(min(output_centers)), s(out_bus_x), s(max(output_centers))), fill=COLORS["black"], width=s(4))
            for cy in output_centers:
                draw_arrow(draw, (s(out_bus_x), s(cy)), (s(output_x - 7), s(cy)), width=s(4))

    for box, cy in zip(spec.outputs, output_centers):
        y = cy - output_h / 2
        draw.rounded_rectangle(
            (s(output_x), s(y), s(output_x + output_w), s(y + output_h)),
            radius=s(18),
            fill=box.fill,
            outline=box.stroke,
            width=s(3),
        )
        title_lines = wrap_pixels(draw, box.title, box_title_font, s(output_w - 30))
        title = title_lines[0]
        tw = text_width(draw, title, box_title_font)
        draw.text((s(output_x + output_w / 2) - tw // 2, s(y + 14)), title, fill=box.title_color, font=box_title_font)
        body_lines: list[str] = []
        for line in box.lines:
            body_lines.extend(wrap_pixels(draw, line, body_font, s(output_w - 32)))
        for idx, line in enumerate(body_lines[:2]):
            tw = text_width(draw, line, body_font)
            draw.text(
                (s(output_x + output_w / 2) - tw // 2, s(y + 48 + idx * 18)),
                line,
                fill=COLORS["text"],
                font=body_font,
            )

    note_text = "  |  ".join(spec.notes)
    note_lines = wrap_pixels(draw, note_text, body_font, s(g["width"] - 170))
    for idx, line in enumerate(note_lines[:2]):
        draw.text((s(86), s(g["height"] - 76 + idx * 20)), line, fill=COLORS["muted"], font=body_font)

    img = img.resize((g["width"], g["height"]), Image.Resampling.LANCZOS)
    img.save(path)


def build_specs() -> dict[str, list[DiagramSpec]]:
    return {
        "general_contrastive_fcn_pos": [
            general_contrastive_fcn_pos_diagram(),
        ],
        "fcn_input_ablation": [
            fcn_input_diagram(index, variant)
            for index, variant in enumerate(FCN_INPUT_VARIANTS, start=1)
        ],
        "position_offset_ablation": [
            position_offset_diagram(index, variant, offset)
            for index, (variant, offset) in enumerate(POSITION_OFFSET_VARIANTS, start=1)
        ],
        "pos_trq_backbone_ablation": [
            backbone_diagram(index, variant)
            for index, variant in enumerate(POS_TRQ_BACKBONE_VARIANTS, start=1)
        ],
    }


def make_contact_sheet(paths: list[Path], out_path: Path, title: str):
    if Image is None or not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(img.width for img in images)
    padding = 28
    title_h = 70
    height = title_h + sum(img.height for img in images) + padding * (len(images) + 1)
    sheet = Image.new("RGB", (width + padding * 2, height), COLORS["page"])
    draw = ImageDraw.Draw(sheet)
    title_font = find_font(True, 28)
    draw.text((padding, 22), title, fill=COLORS["text"], font=title_font)
    y = title_h + padding
    for img in images:
        x = padding + (width - img.width) // 2
        sheet.paste(img, (x, y))
        y += img.height + padding
    sheet.save(out_path)


def write_index(out_dir: Path, specs_by_family: dict[str, list[DiagramSpec]]):
    labels = {
        "general_contrastive_fcn_pos": "General Contrastive FCN Position",
        "fcn_input_ablation": "FCN Input Ablation",
        "position_offset_ablation": "Position Offset Ablation",
        "pos_trq_backbone_ablation": "Pos+Trq Backbone Ablation",
    }
    lines = [
        "# Self-touch Ablation Diagrams",
        "",
        "Generated by `generate_selftouch_ablation_diagrams.py` from the local parameter YAML files.",
        "",
    ]
    for family, specs in specs_by_family.items():
        lines.extend([f"## {labels.get(family, family)}", ""])
        sheet = f"{family}_sheet.png"
        if (out_dir / sheet).exists():
            lines.append(f"- Sheet: [{sheet}]({sheet})")
        for spec in specs:
            rel_png = f"{family}/{spec.slug}.png"
            rel_svg = f"{family}/{spec.slug}.svg"
            lines.append(f"- [{spec.title}]({rel_png}) ([SVG]({rel_svg}))")
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    specs_by_family = build_specs()

    for family, specs in specs_by_family.items():
        family_dir = out_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        png_paths: list[Path] = []
        for spec in specs:
            svg_path = family_dir / f"{spec.slug}.svg"
            png_path = family_dir / f"{spec.slug}.png"
            render_svg(spec, svg_path)
            render_png(spec, png_path)
            if png_path.exists():
                png_paths.append(png_path)
        make_contact_sheet(
            png_paths,
            out_dir / f"{family}_sheet.png",
            family.replace("_", " ").title(),
        )

    write_index(out_dir, specs_by_family)
    total = sum(len(specs) for specs in specs_by_family.values())
    print(f"Wrote {total} self-touch ablation diagrams to {out_dir}")


if __name__ == "__main__":
    main()
