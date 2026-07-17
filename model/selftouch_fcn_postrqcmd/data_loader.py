import torch
from torch.utils.data import Dataset, DataLoader
import pdb
import data_preproc
import json
import os
import numpy as np
import visualizer as vis
from dataloader_base import BaseDataLoader
import einops
import gc
import pdb
from selftouch_offset_utils import base_target_count, target_window, valid_target_count

from selftouch_combo_utils import (
    COMBO_KEYS,
    combo_vector_for_episode,
    tactile_keys_for_episode,
    validate_selftouch_combinations,
    validate_selftouch_data_dir,
)

TACTILE_KEYS = [
    "tactile_index_tip",
    "tactile_thumb_tip",
    "tactile_middle_tip",
    "tactile_ring_tip",
]
JOINT_KEYS = [
    "hand_jnt_pos",
    "hand_jnt_vel",
    "hand_jnt_trq",
    "hand_jnt_cmd_pos",
]
def _episode_timesteps_from_modalities(value_map):
    for key in (*JOINT_KEYS, *TACTILE_KEYS):
        value = value_map.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim > 0:
            return max(int(arr.shape[0]), 1)
    return 1


def _missing_tactile(value):
    if value is None:
        return True
    arr = np.asarray(value)
    if arr.dtype == object:
        flat = arr.reshape(-1)
        return flat.size == 0 or all(item is None for item in flat)
    return False


def _flatten_tactile(value, timesteps=1):
    timesteps = max(int(timesteps or 1), 1)
    if _missing_tactile(value):
        return np.zeros((timesteps, 90), dtype=np.float32)
    arr = np.asarray(value)
    if arr.ndim == 3:
        return einops.rearrange(arr, "t a d -> t (a d)").astype(np.float32, copy=False)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    t = int(arr.shape[0]) if arr.ndim > 0 else timesteps
    return np.zeros((max(t, timesteps, 1), 90), dtype=np.float32)


def _finger_valid_mask(value_map, timesteps):
    valid = [0.0 if _missing_tactile(value_map.get(key)) else 1.0 for key in TACTILE_KEYS]
    valid = np.asarray(valid, dtype=np.float32)
    return np.repeat(valid[None, :], max(int(timesteps or 1), 1), axis=0)


class CustomDataLoader(BaseDataLoader):
    def __init__(self, dataset_param):
        super().__init__()
        self.dataset_param = dataset_param
        self.mem="ram"
        validate_selftouch_data_dir(self.dataset_param)
        dir_data, data_found = data_preproc.get_sequence_dict(self.dataset_param, mem=self.mem)
        validate_selftouch_combinations(dir_data, self.dataset_param)
        # Change tactile shape
        dir_data=self.change_shape(dir_data)
        dir_data=self.add_context_features(dir_data)
        train_dir_data, test_dir_data = data_preproc.split_train_test(dir_data, self.dataset_param,"test_data")
        del dir_data
        gc.collect()
        print("splitted train and test")
        self.write_temporal_offset_diagnostics(train_dir_data, test_dir_data)
        
        scaling_path = os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl")
        if os.path.isfile(scaling_path):
            scaling_param = data_preproc.load_pkl_file(scaling_path)
            scaling_param_exists = True
        else:
            scaling_param = data_preproc.get_scaling_param(train_dir_data, self.dataset_param, mem=self.mem)
            scaling_param_exists = False
        print("got scaled param")
        if not scaling_param_exists:
            data_preproc.save_pkl_file(scaling_param, scaling_path)
        print("scaling param ready")

        try:
            train_dir_data = data_preproc.scale_dir_data(train_dir_data, scaling_param, self.dataset_param, mem=self.mem)
        except (KeyError, ValueError) as exc:
            if not scaling_param_exists:
                raise
            print(f"[warn] stale scaling_param.pkl detected; recomputing scaler: {exc}")
            scaling_param = data_preproc.get_scaling_param(train_dir_data, self.dataset_param, mem=self.mem)
            data_preproc.save_pkl_file(scaling_param, scaling_path)
            train_dir_data = data_preproc.scale_dir_data(train_dir_data, scaling_param, self.dataset_param, mem=self.mem)
        print("scaled train data")

        test_dir_data = data_preproc.scale_dir_data(test_dir_data, scaling_param, self.dataset_param, mem=self.mem)
        print("scaled test data")


        train_dir_data = data_preproc.rearrange(train_dir_data, self.dataset_param, mem=self.mem)
        print("rearranged train data")
        train_dir_data=self.merge_modalities([train_dir_data,data_found])
        train_dir_data=self.data_to_tensor(train_dir_data)

        test_dir_data = data_preproc.rearrange(test_dir_data, self.dataset_param, mem=self.mem)
        print("rearranged test data")
        test_dir_data=self.merge_modalities([test_dir_data,data_found])
        self.test_dir_data=self.data_to_tensor(test_dir_data)
            
        self.set_train_data(train_dir_data)
        self.set_memory_location(self.mem)
    def __len__(self):
        if self._is_tensor_table(self.train_dir_data):
            return self._tensor_table_len(self.train_dir_data)
        return len(self.train_dir_data.keys())
    
    def change_shape(self,data):
        for ep, val in data.items():
            timesteps = _episode_timesteps_from_modalities(val)
            data[ep]["selftouch_finger_mask"] = _finger_valid_mask(val, timesteps)
            for key in TACTILE_KEYS:
                if key in val:
                    data[ep][key] = _flatten_tactile(val[key], timesteps)
            for key in JOINT_KEYS:
                if key in val:
                    data[ep][key] = np.asarray(val[key], dtype=np.float32)

        return data

    def add_context_features(self, data):
        add_combo = bool(self.dataset_param.get("add_selftouch_combo_condition", True))
        add_phase = bool(self.dataset_param.get("add_selftouch_phase_condition", True))
        if not add_combo and not add_phase:
            return data
        for ep, val in data.items():
            steps = 1
            for key in (*TACTILE_KEYS, *JOINT_KEYS):
                if key in val and np.asarray(val[key]).ndim > 0:
                    steps = int(np.asarray(val[key]).shape[0])
                    break

            if add_combo:
                text = str(ep).lower().replace("-", "_")
                combo = combo_vector_for_episode(ep)
                val["selftouch_combo"] = np.repeat(combo[None, :], steps, axis=0)

            if add_phase:
                if steps <= 1:
                    phase = np.zeros((steps,), dtype=np.float32)
                else:
                    phase = np.linspace(0.0, 1.0, steps, dtype=np.float32)
                features = [phase, phase * 2.0 - 1.0]
                for freq in (1.0, 2.0, 4.0, 8.0):
                    angle = (2.0 * np.pi * freq * phase).astype(np.float32)
                    features.extend([np.sin(angle), np.cos(angle)])
                val["selftouch_phase"] = np.stack(features, axis=-1).astype(np.float32, copy=False)
        return data

    def write_temporal_offset_diagnostics(self, train_dir_data, test_dir_data):
        offset = int(self.dataset_param.get("input_offset", 0) or 0)
        diagnostics = {
            "input_offset": offset,
            "sequence_length": int(self.dataset_param.get("sequence_length", 0) or 0),
            "drop": {
                "train": self._drop_stats(train_dir_data, offset),
                "test": self._drop_stats(test_dir_data, offset),
                "all": self._drop_stats({**train_dir_data, **test_dir_data}, offset),
            },
        }
        if offset > 0:
            diagnostics["leakage"] = self._leakage_stats({**train_dir_data, **test_dir_data}, offset)

        self._print_temporal_offset_diagnostics(diagnostics)
        param_dir = self.dataset_param.get("param_file_dir")
        if param_dir:
            os.makedirs(param_dir, exist_ok=True)
            path = os.path.join(param_dir, "temporal_offset_diagnostics.json")
            with open(path, "w") as f:
                json.dump(diagnostics, f, indent=2, sort_keys=True)
            print(f"[offset] diagnostics written: {path}")

    def _episode_timesteps(self, episode):
        for key in (*TACTILE_KEYS, *JOINT_KEYS, "selftouch_combo", "selftouch_phase"):
            value = episode.get(key)
            if value is not None:
                arr = np.asarray(value)
                if arr.ndim > 0:
                    return int(arr.shape[0])
        return int(self.dataset_param.get("sequence_length", 0) or 0)

    def _drop_stats(self, split_data, offset):
        base = 0
        valid = 0
        episodes = 0
        target_start = None
        target_stop = None
        for episode in split_data.values():
            timesteps = self._episode_timesteps(episode)
            start, stop = target_window(timesteps, offset)
            if target_start is None:
                target_start = start
                target_stop = stop
            base += base_target_count(timesteps)
            valid += valid_target_count(timesteps, offset)
            episodes += 1
        dropped = max(0, base - valid)
        return {
            "episodes": episodes,
            "base_samples": base,
            "valid_samples": valid,
            "dropped_samples": dropped,
            "drop_fraction": float(dropped / base) if base else 0.0,
            "target_start": int(target_start or 0),
            "target_stop": int(target_stop or 0),
        }

    def _episode_tactile_keys(self, episode_name, episode):
        return tactile_keys_for_episode(episode_name, episode)

    def _max_abs_tactile(self, episode, keys, indices):
        values = []
        for key in keys:
            arr = np.asarray(episode[key], dtype=np.float32)
            if arr.ndim < 2 or len(indices) == 0:
                continue
            selected = arr[indices]
            values.append(np.nanmax(np.abs(selected.reshape(selected.shape[0], -1)), axis=1))
        if not values:
            return np.zeros((len(indices),), dtype=np.float32)
        return np.maximum.reduce(values)

    def _leakage_stats(self, split_data, offset):
        contact_threshold = float(self.dataset_param.get("leakage_contact_abs_threshold", 300.0))
        precursor_threshold = float(
            self.dataset_param.get("leakage_precursor_abs_threshold", contact_threshold * 0.5)
        )
        flag_threshold = float(self.dataset_param.get("leakage_flag_fraction_threshold", 0.5))
        target_contact_samples = 0
        future_contact_or_precursor_samples = 0
        considered_samples = 0
        for episode_name, episode in split_data.items():
            timesteps = self._episode_timesteps(episode)
            target_start, target_stop = target_window(timesteps, offset)
            if target_stop <= target_start:
                continue
            target_indices = np.arange(target_start, target_stop, dtype=np.int64)
            future_indices = target_indices + int(offset)
            keys = self._episode_tactile_keys(episode_name, episode)
            target_max = self._max_abs_tactile(episode, keys, target_indices)
            future_max = self._max_abs_tactile(episode, keys, future_indices)
            target_contact = target_max >= contact_threshold
            considered_samples += int(target_indices.size)
            target_contact_samples += int(target_contact.sum())
            future_contact_or_precursor_samples += int(
                (target_contact & (future_max >= precursor_threshold)).sum()
            )
        fraction = (
            float(future_contact_or_precursor_samples / target_contact_samples)
            if target_contact_samples
            else 0.0
        )
        return {
            "contact_abs_threshold": contact_threshold,
            "precursor_abs_threshold": precursor_threshold,
            "target_contact_samples": target_contact_samples,
            "future_contact_or_precursor_samples": future_contact_or_precursor_samples,
            "fraction": fraction,
            "flag_fraction_threshold": flag_threshold,
            "flagged": bool(fraction >= flag_threshold and target_contact_samples > 0),
            "considered_samples": considered_samples,
        }

    def _print_temporal_offset_diagnostics(self, diagnostics):
        drop = diagnostics["drop"]["all"]
        print(
            "[offset] input_offset={offset:+d} target=[{start},{stop}) "
            "dropped={dropped}/{base} ({frac:.3%})".format(
                offset=int(diagnostics["input_offset"]),
                start=int(drop["target_start"]),
                stop=int(drop["target_stop"]),
                dropped=int(drop["dropped_samples"]),
                base=int(drop["base_samples"]),
                frac=float(drop["drop_fraction"]),
            )
        )
        leakage = diagnostics.get("leakage")
        if leakage:
            msg = (
                "[offset] future-label leakage check: "
                f"{leakage['future_contact_or_precursor_samples']}/"
                f"{leakage['target_contact_samples']} target-contact samples "
                f"({leakage['fraction']:.1%}) already show contact/precursor "
                f"at t+{int(diagnostics['input_offset'])}"
            )
            if leakage.get("flagged"):
                msg += " [FLAGGED]"
            print(msg)


    def get_test_data(self):
        return self.test_dir_data
