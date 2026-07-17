import torch
from torch.utils.data import Dataset, DataLoader
import pdb
import data_preproc
import os
import numpy as np
import visualizer as vis
from dataloader_base import BaseDataLoader
import einops
import gc
import pdb

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
        train_dir_data, test_dir_data = data_preproc.split_train_test(dir_data, self.dataset_param,"test_data")
        del dir_data
        gc.collect()
        print("splitted train and test")
        
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

        test_dir_data = data_preproc.rearrange(test_dir_data, self.dataset_param, mem=self.mem)
        print("rearranged test data")
        test_dir_data=self.merge_modalities([test_dir_data,data_found])
        self.test_dir_data=self.data_to_tensor(test_dir_data)
            
        self.set_train_data(train_dir_data)
        self.set_memory_location(self.mem)
    def __len__(self):
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


    def get_test_data(self):
        return self.test_dir_data
