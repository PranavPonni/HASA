import os

import einops
import numpy as np

import data_preproc
from dataloader_base import BaseDataLoader


JOINT_IDX = [0, 1, 2, 3, 12, 13, 14, 15]
TACTILE_KEYS = ["tactile_index_tip", "tactile_thumb_tip"]
SELFTOUCH_KEY_MAP = {
    "selftouch_hand_jnt_pos": "hand_jnt_pos",
    "selftouch_hand_jnt_vel": "hand_jnt_vel",
    "selftouch_hand_jnt_trq": "hand_jnt_trq",
    "selftouch_hand_jnt_cmd_pos": "hand_jnt_cmd_pos",
}


def _flatten_tactile(value):
    arr = np.asarray(value)
    if arr.ndim == 3:
        return einops.rearrange(arr, "t a d -> t (a d)").astype(np.float32, copy=False)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported tactile shape: {arr.shape}")


class CustomDataLoader(BaseDataLoader):
    def __init__(self, dataset_param, model_param):
        super().__init__()
        self.dataset_param = dataset_param
        self.model_param = model_param
        self.mem = "ram"

        dir_data, data_found = data_preproc.get_sequence_dict(
            self.dataset_param, mem=self.mem
        )
        dir_data = self.change_shape(dir_data)
        train_dir_data, test_dir_data = data_preproc.split_train_test(
            dir_data, self.dataset_param, "test_data"
        )
        print("splitted train and test")

        scaling_param = data_preproc.get_scaling_param(
            train_dir_data, self.dataset_param, mem=self.mem
        )
        data_preproc.save_pkl_file(
            scaling_param,
            os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"),
        )
        print("saved pkl data")

        train_dir_data = data_preproc.scale_dir_data(
            train_dir_data, scaling_param, self.dataset_param, mem=self.mem
        )
        test_dir_data = data_preproc.scale_dir_data(
            test_dir_data, scaling_param, self.dataset_param, mem=self.mem
        )

        if self.model_param.get("use_selftouch", False):
            train_dir_data = self._scale_selftouch_inputs(train_dir_data)
            test_dir_data = self._scale_selftouch_inputs(test_dir_data)

        train_dir_data = data_preproc.rearrange(
            train_dir_data, self.dataset_param, mem=self.mem
        )
        train_dir_data = self.merge_modalities([train_dir_data, data_found])

        test_dir_data = data_preproc.rearrange(
            test_dir_data, self.dataset_param, mem=self.mem
        )
        test_dir_data = self.merge_modalities([test_dir_data, data_found])

        self.test_episode_names = list(test_dir_data.keys())
        self.test_dir_data = self.data_to_tensor(test_dir_data)
        self.set_train_data(train_dir_data)
        self.set_memory_location(self.mem)

    def __len__(self):
        return len(self.train_dir_data.keys())

    def change_shape(self, data):
        for _, val in data.items():
            for key in TACTILE_KEYS:
                if key in val:
                    val[key] = _flatten_tactile(val[key])

            full_pos = np.asarray(val["hand_jnt_pos"]).astype(np.float32, copy=False)
            full_vel = np.asarray(val["hand_jnt_vel"]).astype(np.float32, copy=False)
            full_trq = np.asarray(val.get("hand_jnt_trq", np.zeros_like(full_pos))).astype(
                np.float32, copy=False
            )
            full_cmd = np.asarray(val.get("hand_jnt_cmd_pos", full_pos)).astype(
                np.float32, copy=False
            )

            if self.model_param.get("use_selftouch", False):
                val["selftouch_hand_jnt_pos"] = full_pos
                val["selftouch_hand_jnt_vel"] = full_vel
                val["selftouch_hand_jnt_trq"] = full_trq
                val["selftouch_hand_jnt_cmd_pos"] = full_cmd

            val["hand_jnt_pos"] = full_pos[:, JOINT_IDX]
            val["hand_jnt_vel"] = full_vel[:, JOINT_IDX]
        return data

    def _scale_selftouch_inputs(self, data):
        if not self.model_param.get("scale_selftouch_with_pretrained", True):
            return data

        st_param_path = self.model_param.get("st_param")
        if not st_param_path or not os.path.isfile(st_param_path):
            print(f"[sat] Missing st_param for selftouch scaling: {st_param_path}")
            return data

        st_params = data_preproc.read_yaml(st_param_path)
        st_modality = st_params.get("Dataset", {}).get("modality", {})
        st_scaling_path = self.model_param.get("st_scaling_param") or os.path.join(
            os.path.dirname(st_param_path), "scaling_param.pkl"
        )
        if not os.path.isfile(st_scaling_path):
            print(f"[sat] Missing selftouch scaling_param.pkl: {st_scaling_path}")
            return data

        st_scaling = data_preproc.load_pkl_file(st_scaling_path)
        for _, item in data.items():
            for extra_key, source_key in SELFTOUCH_KEY_MAP.items():
                if (
                    extra_key in item
                    and source_key in st_scaling
                    and source_key in st_modality
                ):
                    item[extra_key] = data_preproc.scaling_data(
                        item[extra_key], st_scaling[source_key], st_modality[source_key]
                    )
        return data

    def get_test_data(self):
        return self.test_dir_data

