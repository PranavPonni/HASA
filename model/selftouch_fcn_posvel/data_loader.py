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
COMBO_KEYS = ("thumb_index", "thumb_middle", "index_middle")


def _flatten_tactile(value):
    arr = np.asarray(value)
    if arr.ndim == 3:
        return einops.rearrange(arr, "t a d -> t (a d)")
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    t = arr.shape[0] if arr.ndim > 0 else 1
    return np.zeros((t, 90), dtype=np.float32)


class CustomDataLoader(BaseDataLoader):
    def __init__(self, dataset_param):
        super().__init__()
        self.dataset_param = dataset_param
        self.mem="ram"
        dir_data, data_found = data_preproc.get_sequence_dict(self.dataset_param, mem=self.mem)
        # Change tactile shape
        dir_data=self.change_shape(dir_data)
        dir_data=self.add_context_features(dir_data)
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
        except ValueError as exc:
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
            for key in TACTILE_KEYS:
                if key in val:
                    data[ep][key] = _flatten_tactile(val[key])
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
                combo = np.zeros((3,), dtype=np.float32)
                for idx, name in enumerate(COMBO_KEYS):
                    if name in text:
                        combo[idx] = 1.0
                        break
                if combo.sum() == 0.0:
                    # Unknown names get a neutral condition rather than a hard zero.
                    combo[:] = 1.0 / float(combo.size)
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


    def get_test_data(self):
        return self.test_dir_data
