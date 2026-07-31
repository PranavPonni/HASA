"""Export trained self-touch predictions as per-combination PyTorch path files."""

from __future__ import annotations

import os
import json
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from selftouch_combo_utils import COMBO_TO_TACTILE_KEYS, FINGER_KEYS, FINGER_NAMES, normalize_combo_name
from selftouch_plot_utils import align_next_step_prediction, load_scaling_param, maybe_unscale, tensor_to_numpy


def _combo_labels(data: Mapping, combinations: Sequence[str]) -> tuple[list[str], np.ndarray]:
    combo_names = [normalize_combo_name(value) for value in combinations]
    combo = data.get("selftouch_combo")
    if combo is None:
        return combo_names, np.zeros((0,), dtype=np.int64)
    combo_np = tensor_to_numpy(combo)
    if combo_np.ndim >= 3:
        combo_np = np.nanmean(combo_np, axis=1)
    if combo_np.ndim != 2 or combo_np.shape[0] == 0:
        return combo_names, np.zeros((0,), dtype=np.int64)
    if not combo_names or len(combo_names) != combo_np.shape[1]:
        combo_names = [f"combo_{idx}" for idx in range(combo_np.shape[1])]
    labels = np.argmax(combo_np[:, : len(combo_names)], axis=1).astype(np.int64, copy=False)
    return combo_names, labels


def _combo_active_fingers(combo_name: str) -> list[str]:
    keys = COMBO_TO_TACTILE_KEYS.get(normalize_combo_name(combo_name), ())
    active = []
    for key in keys:
        if key in FINGER_KEYS:
            active.append(FINGER_NAMES[FINGER_KEYS.index(key)])
    return active


def _to_cpu_tensor(value: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float32).copy())


def _finger_mask_for_rows(data: Mapping, rows: np.ndarray, timesteps: np.ndarray) -> Optional[torch.Tensor]:
    mask = data.get("selftouch_finger_mask")
    if mask is None:
        return None
    mask_np = tensor_to_numpy(mask)
    if mask_np.ndim < 3 or mask_np.shape[0] == 0:
        return None
    if rows.size:
        mask_np = mask_np[rows]
    if timesteps.size:
        valid_steps = np.asarray(timesteps, dtype=np.int64)
        valid_steps = valid_steps[(valid_steps >= 0) & (valid_steps < mask_np.shape[1])]
        mask_np = mask_np[:, valid_steps, :]
    return _to_cpu_tensor(mask_np)


def _write_manifest(
    *,
    output_dir: str,
    exported: Mapping[str, str],
    combo_names: Sequence[str],
    epoch: int,
    checkpoint_path: Optional[str],
    model_name: Optional[str],
    dataset_param: Mapping,
    next_step: bool,
) -> str:
    entries = {}
    for combo_idx, combo_name in enumerate(combo_names):
        path = exported.get(combo_name)
        if not path:
            continue
        entries[combo_name] = {
            "file": os.path.basename(path),
            "combo_index": int(combo_idx),
            "active_fingers": _combo_active_fingers(combo_name),
        }
    manifest = {
        "kind": "selftouch_combo_path_manifest",
        "version": 1,
        "model_name": str(model_name or ""),
        "checkpoint_path": str(checkpoint_path or ""),
        "epoch": int(epoch),
        "split": "test",
        "next_step_prediction": bool(next_step),
        "input_offset": int(dataset_param.get("input_offset", 0) or 0),
        "sequence_length": int(dataset_param.get("sequence_length", 0) or 0),
        "combo_count": int(len(entries)),
        "finger_order": list(FINGER_NAMES),
        "tactile_units": {
            "normalized": "dataset training scale",
            "raw": "inverse-scaled tactile raw units",
        },
        "paths": entries,
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def export_combo_path_artifacts(
    *,
    data: Mapping,
    preds: Mapping[str, torch.Tensor],
    dataset_param: Mapping,
    combinations: Sequence[str],
    output_dir: str,
    epoch: int,
    checkpoint_path: Optional[str] = None,
    model_name: Optional[str] = None,
    next_step: bool = True,
) -> Dict[str, str]:
    """Write one `.pth` file per configured self-touch combination.

    Each file stores aligned normalized predictions/targets plus raw-unit
    predictions/targets.  The files are data artifacts for downstream motion
    generation; they are not separate model checkpoints.
    """

    os.makedirs(output_dir, exist_ok=True)
    combo_names, labels = _combo_labels(data, combinations)
    scaling_param = load_scaling_param(dataset_param)
    input_offset = int(dataset_param.get("input_offset", 0) or 0)
    exported = {}

    for combo_idx, combo_name in enumerate(combo_names):
        rows = np.where(labels == combo_idx)[0] if labels.size else np.asarray([], dtype=np.int64)
        payload = {
            "kind": "selftouch_combo_path",
            "combo_name": combo_name,
            "combo_index": int(combo_idx),
            "active_fingers": _combo_active_fingers(combo_name),
            "finger_order": list(FINGER_NAMES),
            "finger_keys": list(FINGER_KEYS),
            "epoch": int(epoch),
            "model_name": str(model_name or ""),
            "checkpoint_path": str(checkpoint_path or ""),
            "split": "test",
            "next_step_prediction": bool(next_step),
            "input_offset": input_offset,
            "sequence_length": int(dataset_param.get("sequence_length", 0) or 0),
            "tactile_units": {
                "normalized": "dataset training scale",
                "raw": "inverse-scaled tactile raw units",
            },
            "timesteps": torch.empty(0, dtype=torch.long),
            "finger_mask": None,
            "pred_normalized": {},
            "target_normalized": {},
            "pred_raw": {},
            "target_raw": {},
        }

        shared_timesteps = None
        for finger, key in zip(FINGER_NAMES, FINGER_KEYS):
            raw_arr = tensor_to_numpy(data[key])
            pred_arr = tensor_to_numpy(preds[finger])
            if rows.size:
                raw_arr = raw_arr[rows]
                pred_arr = pred_arr[rows]
            else:
                raw_arr = raw_arr[:0]
                pred_arr = pred_arr[:0]
            raw_norm, pred_norm, timesteps = align_next_step_prediction(
                raw_arr,
                pred_arr,
                next_step=next_step,
                input_offset=input_offset,
            )
            raw_units = maybe_unscale(raw_norm, key, dataset_param, scaling_param)
            pred_units = maybe_unscale(pred_norm, key, dataset_param, scaling_param)

            payload["target_normalized"][finger] = _to_cpu_tensor(raw_norm)
            payload["pred_normalized"][finger] = _to_cpu_tensor(pred_norm)
            payload["target_raw"][finger] = _to_cpu_tensor(raw_units)
            payload["pred_raw"][finger] = _to_cpu_tensor(pred_units)
            if shared_timesteps is None:
                shared_timesteps = np.asarray(timesteps, dtype=np.int64)

        if shared_timesteps is None:
            shared_timesteps = np.zeros((0,), dtype=np.int64)
        payload["timesteps"] = torch.as_tensor(shared_timesteps.copy(), dtype=torch.long)
        payload["finger_mask"] = _finger_mask_for_rows(data, rows, shared_timesteps)
        payload["num_sequences"] = int(rows.size)

        path = os.path.join(output_dir, f"{combo_name}.pth")
        torch.save(payload, path)
        exported[combo_name] = path

    _write_manifest(
        output_dir=output_dir,
        exported=exported,
        combo_names=combo_names,
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        dataset_param=dataset_param,
        next_step=next_step,
    )
    return exported
