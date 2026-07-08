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
        JOINT_IDX = [0, 1, 2, 3, 12, 13, 14, 15]
        for ep, val in data.items():
            for key in TACTILE_KEYS:
                if key in val:
                    data[ep][key] = _flatten_tactile(val[key])
            for key in JOINT_KEYS:
                if key in val:
                    data[ep][key] = np.asarray(val[key], dtype=np.float32)[:, JOINT_IDX]

        return data


    def get_test_data(self):
        return self.test_dir_data
