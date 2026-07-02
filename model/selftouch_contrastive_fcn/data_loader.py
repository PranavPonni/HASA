"""
data_loader.py for selftouch_contrastive_fcn
Loads sequential episodes with combination labels for contrastive training.
Uses joint position inputs for pure joint-state tactile prediction.
Provides per-finger tactile tensors and combination labels.
"""
import os
import sys
import re
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import data_preproc
from dataloader_base import BaseDataLoader

DEFAULT_COMBINATIONS = [
    "thumb-index",
    "thumb-middle",
    "index-middle",
]

JOINT_IDX = list(range(16))
JOINT_FEATURE_KEYS = [
    "hand_jnt_pos",
    "hand_jnt_vel",
    "hand_jnt_trq",
    "hand_jnt_cmd_pos",
]

FINGER_KEYS = [
    "tactile_index_tip",
    "tactile_thumb_tip",
]


def _canonical(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _infer_label(ep_name: str, combinations: Sequence[str]) -> int:
    ep_lower = ep_name.lower()
    for idx, combo in enumerate(combinations):
        if combo.replace("-", "_") in ep_lower:
            return idx
    return -1


def _fallback_dim(key: str) -> int:
    return 90 if key.startswith("tactile_") else 16


class CustomDataLoader(BaseDataLoader):
    """Sequential data loader with combination labels for contrastive_fcn training."""

    def __init__(self, dataset_param: dict):
        super().__init__()
        self.dataset_param = dataset_param

        data_dir         = dataset_param["data_dir"]
        modality         = dataset_param["modality"]
        sequence_length  = dataset_param["sequence_length"]
        self.shift_data  = dataset_param.get("shift_data", 1)
        self.combinations = dataset_param.get("combinations", DEFAULT_COMBINATIONS)

        # Build per-episode dicts with all modalities + labels
        train_data: Dict[str, list] = {m: [] for m in modality}
        train_data["label"] = []
        test_data:  Dict[str, list] = {m: [] for m in modality}
        test_data["label"]  = []
        self.test_episode_names: List[str] = []

        all_ep_names = self._get_all_episode_names(data_dir)
        test_ep_names_cfg = data_preproc.resolve_test_episode_names(all_ep_names, dataset_param, "test_data")
        for ep_name in all_ep_names:
            ep_path = os.path.join(data_dir, ep_name)
            if not os.path.isdir(ep_path):
                continue
            ep_data = self._load_episode(ep_path, modality, sequence_length)
            if ep_data is None:
                continue
            label = _infer_label(ep_name, self.combinations)
            if ep_name in test_ep_names_cfg:
                for m in modality:
                    test_data[m].append(ep_data[m])
                test_data["label"].append(label)
                self.test_episode_names.append(ep_name)
            else:
                for m in modality:
                    train_data[m].append(ep_data[m])
                train_data["label"].append(label)

        self.scaling_param = self._load_or_create_scaling_param(train_data, modality)

        # Stack into tensors via base helper
        self.train_tensors = self._stack_data(train_data, modality)
        self.test_tensors  = self._stack_data(test_data,  modality)
        self.batch_size   = 8
        self.shuffle      = True
        self._train_idx   = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_all_episode_names(self, data_dir: str) -> List[str]:
        names = []
        for root, dirs, files in os.walk(data_dir):
            pkl_files = [f for f in files if re.match(r"timestep\d+\.pkl$", f)]
            if pkl_files:
                rel = os.path.relpath(root, data_dir)
                ep_name = rel if rel != "." else os.path.basename(root)
                names.append(ep_name)
        return sorted(names)

    def _load_episode(
        self, ep_path: str, modality: dict, sequence_length: int
    ) -> Optional[Dict[str, np.ndarray]]:
        files = self._episode_timestep_files(ep_path, sequence_length)

        raw: Dict[str, list] = {k: [] for k in modality}
        for fp in files:
            if not os.path.isfile(fp):
                if self.dataset_param.get("allow_missing_timesteps", False):
                    return None
                rel = os.path.relpath(fp, self.dataset_param["data_dir"])
                raise FileNotFoundError(f"No data available for {rel}.")
            with open(fp, "rb") as f:
                ts = pickle.load(f)
            for key in modality:
                val = ts.get(key, None)
                if val is None:
                    if key != "tactile_ring_tip" and not self.dataset_param.get("allow_missing_modalities", False):
                        rel = os.path.relpath(fp, self.dataset_param["data_dir"])
                        raise KeyError(f"Missing required modality {key} in {rel}.")
                    # ring_tip is absent in this 3-finger dataset, so keep an
                    # explicit zero channel for model interfaces that expect it.
                    prev = raw[key]
                    fallback_dim = prev[0].shape[0] if prev else _fallback_dim(key)
                    arr = np.zeros(fallback_dim, dtype=np.float32)
                else:
                    arr = np.asarray(val, dtype=np.float32).reshape(-1)
                raw[key].append(arr)

        return {k: np.stack(raw[k], axis=0) for k in modality}  # (T, dim)

    def _episode_timestep_files(self, ep_path: str, sequence_length: int) -> List[str]:
        has_zero = os.path.isfile(os.path.join(ep_path, "timestep0.pkl"))
        has_last_zero = os.path.isfile(os.path.join(ep_path, f"timestep{sequence_length - 1}.pkl"))
        has_one = os.path.isfile(os.path.join(ep_path, "timestep1.pkl"))
        has_last_one = os.path.isfile(os.path.join(ep_path, f"timestep{sequence_length}.pkl"))
        if has_zero and has_last_zero:
            start = 0
        elif has_one and has_last_one:
            start = 1
        elif has_zero:
            start = 0
        else:
            start = 1
        return [os.path.join(ep_path, f"timestep{i}.pkl") for i in range(start, start + sequence_length)]

    def _load_or_create_scaling_param(
        self, train_data: Dict[str, list], modality: dict
    ) -> Dict:
        scaling_path = os.path.join(
            self.dataset_param["param_file_dir"], "scaling_param.pkl"
        )
        if os.path.isfile(scaling_path):
            with open(scaling_path, "rb") as f:
                return pickle.load(f)

        scaling_param = {}
        for key in modality:
            values = train_data.get(key, [])
            if not values:
                continue
            arr = np.concatenate(values, axis=0)
            scaling_param[key] = data_preproc.scaling_param(arr, modality[key])

        os.makedirs(os.path.dirname(scaling_path), exist_ok=True)
        with open(scaling_path, "wb") as f:
            pickle.dump(scaling_param, f, protocol=-1)
        return scaling_param

    def _stack_data(
        self, data: Dict[str, list], modality: dict
    ) -> Dict[str, torch.Tensor]:
        """Stack episode lists into (N, T, dim) tensors and normalise via scaling_param."""
        if not data.get("label"):
            return {}

        scaling_param = getattr(self, "scaling_param", {})

        tensors: Dict[str, torch.Tensor] = {}
        for key in modality:
            arr = np.stack(data[key], axis=0)  # (N, T, dim)
            # normalize using scaling_param if available
            if scaling_param and key in scaling_param:
                sp = scaling_param[key]
                norm_cfg = modality[key]
                try:
                    arr = data_preproc.scaling_data(arr, sp, norm_cfg)
                except Exception as exc:
                    print(f"[warn] failed to scale {key}; using raw values: {exc}")
            tensors[key] = torch.tensor(arr, dtype=torch.float32)

        tensors["label"] = torch.tensor(data["label"], dtype=torch.long)

        # Store pos and cmd_pos separately (model builds shifted input internally)
        for key in JOINT_FEATURE_KEYS:
            if key in tensors:
                tensors[key] = tensors[key][:, :, JOINT_IDX]

        return tensors

    # ── iterator interface ────────────────────────────────────────────────────

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def set_shuffle(self, shuffle: bool):
        self.shuffle = shuffle

    def __iter__(self):
        N = self.train_tensors.get("label", torch.tensor([])).shape[0]
        if N == 0:
            return
        indices = torch.randperm(N) if self.shuffle else torch.arange(N)
        bs = self.batch_size
        for start in range(0, N, bs):
            idx = indices[start: start + bs]
            yield {k: v[idx] for k, v in self.train_tensors.items()}

    def __len__(self) -> int:
        return self.train_tensors.get("label", torch.tensor([])).shape[0]

    def get_test_data(self) -> Dict[str, torch.Tensor]:
        return self.test_tensors
