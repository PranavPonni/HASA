"""Evaluate an averaged selftouch ensemble on the shared test split."""

import argparse
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

import torch
from ruamel.yaml import YAML

from selftouch_plot_utils import plot_tactile_temporal_profiles


DEFAULT_VARIANTS = (
    "selftouch_fcn_postrqcmd",
    "selftouch_fcn_poscmdvel",
    "selftouch_fcn_poscmd",
)

MODEL_ENTRYPOINTS = {
    "selftouch_fcn": ("selftouch.py", "SelfTouch"),
    "selftouch_tcn": ("selftouch.py", "SelfTouchTCN"),
    "selftouch_transformer": ("selftouch.py", "SelfTouchTransformer"),
    "selftouch_gru_attention": ("selftouch.py", "SelfTouchGRUAttention"),
    "selftouch_temporal_mixer": ("selftouch.py", "SelfTouchTemporalMixer"),
    "selftouch_contrastive_fcn": ("selftouch_contrastive_fcn.py", "SelfTouchContrastiveFCN"),
    "selftouch_contrastive_gru": ("selftouch_contrastive_gru.py", "SelfTouchContrastiveGRU"),
    "selftouch_contrastive_temporal": (
        "selftouch_contrastive_temporal.py",
        "SelfTouchContrastiveTemporal",
    ),
    "selftouch_contrastive_transformer": (
        "selftouch_contrastive_transformer.py",
        "SelfTouchContrastiveTransformer",
    ),
}


def import_attr(path, attr):
    path = Path(path)
    module_name = f"_selftouch_ensemble_{path.parent.name}_{path.stem}_{attr}"
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, attr)
    finally:
        sys.path[:] = old_path


def read_config(variant):
    yaml = YAML(typ="safe")
    path = Path("parameter") / variant / "parameter_base" / "parameter_base.yaml"
    with path.open("r") as handle:
        return yaml.load(handle)


def model_entrypoint(variant):
    if variant.startswith("selftouch_fcn_"):
        return Path("model") / variant / "selftouch.py", "SelfTouch"
    filename, class_name = MODEL_ENTRYPOINTS[variant]
    return Path("model") / variant / filename, class_name


def checkpoint_epoch(path):
    match = re.search(r"epoch(\d+)\.pth$", str(path))
    return int(match.group(1)) if match else -1


def latest_checkpoint(variant, run_name=None):
    if run_name:
        candidates = glob.glob(f"model_weight/{variant}/{run_name}/epoch*.pth")
    else:
        candidates = glob.glob(f"model_weight/{variant}/*/epoch*.pth")
    if not candidates:
        return None
    return max((Path(path) for path in candidates), key=lambda path: (checkpoint_epoch(path), path.stat().st_mtime))


def parse_key_value(items):
    output = {}
    for item in items or []:
        key, value = item.split("=", 1)
        output[key] = value
    return output


def load_model(variant, checkpoint, device):
    cfg = read_config(variant)
    entrypoint, class_name = model_entrypoint(variant)
    model_cls = import_attr(entrypoint, class_name)
    model = model_cls(cfg["Model"]).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return cfg, model


def tensor_to_device(data, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def predict(model, data, loss_coef):
    with torch.no_grad():
        _, preds = model.forward_loss(**data, loss_coef=loss_coef)
    if isinstance(preds, dict):
        return {
            "index": preds["index"].detach().cpu(),
            "thumb": preds["thumb"].detach().cpu(),
            "middle": preds["middle"].detach().cpu(),
        }
    return {
        "index": preds[0].detach().cpu(),
        "thumb": preds[1].detach().cpu(),
        "middle": preds[2].detach().cpu(),
    }


def average_predictions(predictions):
    min_len = min(pred[key].shape[1] for pred in predictions for key in ("index", "thumb", "middle"))
    averaged = {}
    for key in ("index", "thumb", "middle"):
        stacked = torch.stack([pred[key][:, :min_len, :] for pred in predictions], dim=0)
        averaged[key] = stacked.mean(dim=0)
    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", default=None, help="Variant to include; repeatable.")
    parser.add_argument("--run", action="append", default=None, help="variant=run_name checkpoint shortcut.")
    parser.add_argument("--checkpoint", action="append", default=None, help="variant=/path/to/epoch.pth.")
    parser.add_argument("--output-dir", default="model_weight/selftouch_ensemble_top3/ensemble/plots")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    variants = tuple(args.variant or DEFAULT_VARIANTS)
    run_by_variant = parse_key_value(args.run)
    ckpt_by_variant = parse_key_value(args.checkpoint)
    device = torch.device(args.device)

    first_cfg = read_config(variants[0])
    loader_path = Path("model") / variants[0] / "data_loader.py"
    data_loader_cls = import_attr(loader_path, "CustomDataLoader")
    dataset = data_loader_cls(first_cfg["Dataset"])
    test_data_cpu = dataset.get_test_data()
    test_data_device = tensor_to_device(test_data_cpu, device)

    predictions = []
    checkpoints = {}
    for variant in variants:
        checkpoint = ckpt_by_variant.get(variant)
        if checkpoint is None:
            checkpoint = latest_checkpoint(variant, run_by_variant.get(variant))
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(f"No checkpoint found for {variant}; pass --checkpoint {variant}=...")
        checkpoints[variant] = str(checkpoint)
        cfg, model = load_model(variant, checkpoint, device)
        predictions.append(predict(model, test_data_device, cfg["Train"]["loss_coef"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = plot_tactile_temporal_profiles(
        data=test_data_cpu,
        preds=average_predictions(predictions),
        epoch=0,
        plot_dir=str(output_dir),
        dataset_param=first_cfg["Dataset"],
        combinations=first_cfg["Dataset"].get("combinations", []),
        finger_names=["index", "thumb", "middle"],
        finger_keys=["tactile_index_tip", "tactile_thumb_tip", "tactile_middle_tip"],
        next_step=True,
    )

    print("Ensemble checkpoints:")
    for variant, checkpoint in checkpoints.items():
        print(f"  {variant}: {checkpoint}")
    print(f"Metrics CSV: {bundle['files']['raw_prediction_metrics']}")
    metrics = bundle.get("metrics", {})
    for key in ("tactile_line_mae", "tactile_line_raw_accuracy", "tactile_line_error_p95"):
        if key in metrics:
            print(f"{key}: {float(metrics[key]):.4f}")


if __name__ == "__main__":
    main()
