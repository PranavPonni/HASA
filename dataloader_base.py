import torch
import random
import data_preproc
import sys
import numpy as np
from abc import ABC, abstractmethod
import pdb

class BaseDataLoader(ABC):
    def __init__(self):
        self.batch_size = None
        self.shuffle = None
        self.indices = None
        self.train_dir_data = None
        self.mem="ram"

    def set_memory_location(self,mem):
        if mem=="ram":
            self.mem=mem
        elif mem=="rom":
            self.mem=mem
        else:
            print("You have to select rom or ram as a location to handle the data")
            sys.exit(1)
        
        print(f"Data will be handled in {self.mem}")

    def _shuffle_indices(self):
        random.shuffle(self.indices)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def set_train_data(self, data):
        self.train_dir_data = data

    def _is_tensor_table(self, data):
        return (
            isinstance(data, dict)
            and bool(data)
            and all(torch.is_tensor(value) for value in data.values())
        )

    def _tensor_table_len(self, data):
        first = next(iter(data.values()))
        return int(first.shape[0]) if first.ndim > 0 else 0

    def merge_modalities(self,data_list):
        episode_sets = [set(d.keys()) for d in data_list]
        common_episodes = set.intersection(*episode_sets)

        if not common_episodes:
            sys.exit("Error: No common episodes found among dictionaries.")

        merged_dict = {}
        for episode in common_episodes:
            merged_modalities = {}
            for data_dict in data_list:
                if episode in data_dict:
                    for modality, array in data_dict[episode].items():
                        if modality in merged_modalities:
                            print("Same modality is in several data")
                            sys.exit(1)
                        else:
                            merged_modalities[modality] = array
                else:
                    print("episode is not in data dict")
                    sys.exit(1)
            merged_dict[episode] = merged_modalities

        return merged_dict
    
    def data_to_tensor(self,data):
        all_modalities = list(next(iter(data.values())).keys())

        grouped_data = {modality: [] for modality in all_modalities}
        for episode in data.values():
            for modality, array in episode.items():
                if not isinstance(array, torch.Tensor):
                    array = torch.tensor(array)
                grouped_data[modality].append(array)

        stacked_tensors = {modality: torch.stack(tensors, dim=0).to(torch.float32) for modality, tensors in grouped_data.items()}
        return stacked_tensors

    
    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        if self.batch_size is None:
            self.batch_size = 1
        if self.shuffle is None:
            self.shuffle = False
        if self._is_tensor_table(self.train_dir_data):
            self.indices = list(range(self._tensor_table_len(self.train_dir_data)))
        elif self.indices is None:
            self.indices = list(self.train_dir_data.keys())
        if self.shuffle:
            self._shuffle_indices()
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        end_index = min(self.current_index + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_index:end_index]
        batch = self._get_batch_indices(self.train_dir_data,batch_indices)
        self.current_index = end_index
        return batch

    def _get_batch_indices(self, dictionary, batch_list):
        if self._is_tensor_table(dictionary):
            return {key: value[batch_list] for key, value in dictionary.items()}

        combined_arrays = {}
        reference_keys = None
        reference_shapes = None
        for batch in batch_list:
            if batch not in dictionary:
                raise ValueError(f"Batch '{batch}' is not found in the dictionary.")
            if self.mem == "ram":
                current_dict = dictionary[batch]
            elif self.mem == "rom":
                current_dict = data_preproc.load_pkl_file(dictionary[batch])
            else:
                print("Wrong location to load batch")
                sys.exit(1)
            current_keys = set(current_dict.keys())
            current_shapes = {key: current_dict[key].shape for key in current_keys}
            if reference_keys is None:
                reference_keys = current_keys
                reference_shapes = current_shapes
                combined_arrays = {key: [] for key in reference_keys}
            else:
                if reference_keys != current_keys:
                    raise ValueError(f"Key mismatch found in batch '{batch}'. Expected keys: {reference_keys}, found: {current_keys}.")
                shape_mismatches = {key: (reference_shapes[key], current_shapes[key]) for key in reference_keys if reference_shapes[key] != current_shapes[key]}
                if shape_mismatches:
                    raise ValueError(f"Shape mismatches found in batch '{batch}': {shape_mismatches}.")
            
            for key, array in current_dict.items():
                combined_arrays[key].append(array.astype(np.float32))
        result = {modality: torch.from_numpy(np.stack(arrays, axis=0)) for modality, arrays in combined_arrays.items()}
        return result
