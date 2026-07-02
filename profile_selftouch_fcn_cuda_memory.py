"""Profile one synthetic train step for each selftouch FCN input variant.

The report is a quick CUDA-memory estimate for comparing the 14 FCN variants
before launching full training. It uses the configured model dimensions,
sequence length, batch size, AMP setting, and loss coefficients.
"""

import argparse
import csv
import importlib
from pathlib import Path

import torch
import yaml


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

TACTILE_KEYS = (
    "tactile_index_tip",
    "tactile_thumb_tip",
    "tactile_middle_tip",
)
JOINT_KEYS = (
    "hand_jnt_pos",
    "hand_jnt_vel",
    "hand_jnt_trq",
    "hand_jnt_cmd_pos",
)


def _load_params(variant):
    path = Path("parameter") / variant / "parameter_base" / "parameter_base.yaml"
    with path.open("r") as f:
        return yaml.safe_load(f), path


def _bytes_to_mib(value):
    return float(value) / (1024.0 * 1024.0)


def _autocast(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _make_batch(params, batch_size, sequence_length, device):
    model = params["Model"]
    hand_dim = int(model["hand_dim"])
    tactile_dim = int(model["tactile_dim"])
    data = {}
    for key in TACTILE_KEYS:
        data[key] = torch.randn(batch_size, sequence_length, tactile_dim, device=device)
    for key in JOINT_KEYS:
        data[key] = torch.randn(batch_size, sequence_length, hand_dim, device=device)
    return data


def _loss_coef(params, device):
    coef = dict(params["Train"].get("loss_coef", {}))
    tactile_dim = int(params["Model"]["tactile_dim"])
    scale = torch.ones(tactile_dim, device=device)
    slope = torch.ones(tactile_dim, device=device)
    coef["tactile_raw_scale_by_key"] = {key: scale for key in TACTILE_KEYS}
    coef["tactile_raw_slope_by_key"] = {key: slope for key in TACTILE_KEYS}
    return coef


def profile_variant(variant, args, device):
    params, path = _load_params(variant)
    model_param = params["Model"]
    train_param = params["Train"]
    batch_size = int(args.batch_size or train_param.get("batch_size", 8))
    sequence_length = int(args.sequence_length or params["Dataset"].get("sequence_length", 400))
    use_amp = bool(train_param.get("use_amp", False)) and not args.no_amp

    module = importlib.import_module(f"model.{variant}.selftouch")
    model = module.SelfTouch(model_param).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_param.get("lr", 8e-5)), foreach=False)
    data = _make_batch(params, batch_size, sequence_length, device)
    loss_coef = _loss_coef(params, device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before_alloc = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)

    optimizer.zero_grad(set_to_none=True)
    with _autocast(use_amp):
        losses, _preds = model.forward_loss(**data, loss_coef=loss_coef)
        total_loss = losses[0]
    total_loss.backward()
    grad_clip = float(train_param.get("grad_clip_norm", 0.0) or 0.0)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    torch.cuda.synchronize(device)

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    current_alloc = torch.cuda.memory_allocated(device)
    current_reserved = torch.cuda.memory_reserved(device)
    params_count = sum(p.numel() for p in model.parameters())

    del optimizer, model, data, loss_coef, total_loss, losses, _preds
    torch.cuda.empty_cache()

    return {
        "variant": variant,
        "param_file": str(path),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "input_modalities": "+".join(model_param.get("input_modalities", [])),
        "parameters": params_count,
        "amp": int(use_amp),
        "before_alloc_mib": _bytes_to_mib(before_alloc),
        "before_reserved_mib": _bytes_to_mib(before_reserved),
        "peak_alloc_mib": _bytes_to_mib(peak_alloc),
        "peak_reserved_mib": _bytes_to_mib(peak_reserved),
        "current_alloc_mib": _bytes_to_mib(current_alloc),
        "current_reserved_mib": _bytes_to_mib(current_reserved),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="logs/selftouch_fcn_cuda_memory.csv")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-warmup-profile", action="store_true")
    parser.add_argument("variants", nargs="*", default=VARIANTS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this on the training GPU.")

    device = torch.device("cuda:0")
    warmup = torch.empty((1024, 1024), device=device)
    del warmup
    torch.cuda.empty_cache()
    if not args.no_warmup_profile and args.variants:
        profile_variant(args.variants[0], args, device)

    rows = []
    for variant in args.variants:
        row = profile_variant(variant, args, device)
        rows.append(row)
        print(
            f"{variant}: peak_alloc={row['peak_alloc_mib']:.1f} MiB "
            f"peak_reserved={row['peak_reserved_mib']:.1f} MiB "
            f"params={row['parameters']:,}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
