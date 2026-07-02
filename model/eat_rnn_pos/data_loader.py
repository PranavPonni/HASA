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

class CustomDataLoader(BaseDataLoader):
    def __init__(self, dataset_param,model_param):
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
        

        print("got scaled param")
        scaling_param=data_preproc.load_pkl_file(os.path.join(os.path.dirname(model_param["st_param"]), "scaling_param.pkl"))
        data_preproc.save_pkl_file(scaling_param, os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"))
        print("saved pkl data")

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
            data[ep]["tactile_index_tip"]=einops.rearrange(val["tactile_index_tip"],"t a d -> t (a d)")
            data[ep]["tactile_thumb_tip"]=einops.rearrange(val["tactile_thumb_tip"],"t a d -> t (a d)")
            data[ep]["hand_jnt_pos"]=val["hand_jnt_pos"][:,[0, 1, 2, 3, 12, 13, 14, 15]]
            data[ep]["hand_jnt_cmd_pos"]=val["hand_jnt_cmd_pos"][:,[0, 1, 2, 3, 12, 13, 14, 15]]

        return data


    def get_test_data(self):
        return self.test_dir_data