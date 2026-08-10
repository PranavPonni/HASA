import os
import cv2
import pdb
import numpy as np
import pandas as pd
import pickle
import einops
import copy
from ruamel.yaml import YAML
import torch
from tqdm import tqdm
import gc
import sys
from multiprocessing import Pool
import re

LEGACY_PROJECT_ROOT = "/root/motionlearning"
PROJECT_ROOT = os.environ.get("HASA_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))


def _localize_legacy_paths(value):
    if isinstance(value, str):
        return value.replace(LEGACY_PROJECT_ROOT, PROJECT_ROOT)
    if isinstance(value, dict):
        for key in list(value.keys()):
            value[key] = _localize_legacy_paths(value[key])
        return value
    if isinstance(value, list):
        for idx, item in enumerate(value):
            value[idx] = _localize_legacy_paths(item)
        return value
    return value


def localize_legacy_paths(value):
    return _localize_legacy_paths(value)

def read_yaml(file_path):
    yaml = YAML()
    with open(file_path, 'r') as infile:
        data = yaml.load(infile)
    return _localize_legacy_paths(data)

def write_yaml(data, file_path):
    yaml = YAML()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as outfile:
        yaml.dump(data, outfile)

def get_file_names(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names

def load_images(directory, file_list):
    images = []
    for filename in file_list:
        image_path = os.path.join(directory, filename)
        
        image = cv2.imdecode(\
            np.fromfile(image_path, dtype=np.uint8),\
            cv2.IMREAD_UNCHANGED)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCVはBGR形式で読み込まれるため、RGBに変換
            images.append(image)
    return images


def get_dir_name(path):
    original_path = path
    path = _localize_legacy_paths(path)
    if path != original_path:
        print(f"[path] localized data path: {original_path} -> {path}")
    return sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])


def _sequence_file_list(data_param, dir_list):
    """Infer whether a dataset is zero- or one-indexed and return exact timestep files."""
    sequence_length = int(data_param["sequence_length"])
    data_dir = data_param["data_dir"]
    first_dir = next((d for d in dir_list if os.path.isdir(os.path.join(data_dir, d))), None)
    if first_dir is None:
        return [f"timestep{i}.pkl" for i in range(sequence_length)]

    first_path = os.path.join(data_dir, first_dir)
    has_zero = os.path.isfile(os.path.join(first_path, "timestep0.pkl"))
    has_last_zero = os.path.isfile(os.path.join(first_path, f"timestep{sequence_length - 1}.pkl"))
    has_one = os.path.isfile(os.path.join(first_path, "timestep1.pkl"))
    has_last_one = os.path.isfile(os.path.join(first_path, f"timestep{sequence_length}.pkl"))

    if has_zero and has_last_zero:
        start = 0
    elif has_one and has_last_one:
        start = 1
    elif has_zero:
        start = 0
    else:
        start = 1
    return [f"timestep{i}.pkl" for i in range(start, start + sequence_length)]


def _missing_data_allowed(data_param):
    return bool(data_param.get("allow_missing_timesteps", False))


def _fallback_or_raise(exc, prev_data, data_param):
    if _missing_data_allowed(data_param) and prev_data is not None:
        return prev_data
    raise exc


def _episode_group_key(name):
    match = re.match(r"(.+)_episode(\d+)$", str(name))
    if not match:
        return str(name), None
    return match.group(1), int(match.group(2))


def _resolve_episode_name_set(episode_names, value):
    if value is None:
        return set()
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return set(episode_names)
        return {value}
    return set(value)


def resolve_test_episode_names(episode_names, dataset_param, key="test_data"):
    """Resolve explicit or policy-driven test episode names."""
    policy = dataset_param.get("test_split_policy")
    if not policy:
        return _resolve_episode_name_set(episode_names, dataset_param.get(key))

    grouped = {}
    for name in sorted(episode_names):
        group, number = _episode_group_key(name)
        grouped.setdefault(group, []).append((number, name))

    policy = str(policy).lower()
    test_names = set()
    if policy in {"blocked_tail", "tail_block", "holdout_tail"}:
        fraction = float(dataset_param.get("test_split_fraction", 0.2))
        explicit_count = dataset_param.get("test_episodes_per_combo")
        for items in grouped.values():
            items = sorted(items, key=lambda item: (-1 if item[0] is None else item[0], item[1]))
            count = int(explicit_count) if explicit_count is not None else int(np.ceil(len(items) * fraction))
            count = max(1, min(len(items), count))
            test_names.update(name for _, name in items[-count:])
        return test_names

    if policy in {"blocked_head", "head_block", "holdout_head"}:
        fraction = float(dataset_param.get("test_split_fraction", 0.2))
        explicit_count = dataset_param.get("test_episodes_per_combo")
        for items in grouped.values():
            items = sorted(items, key=lambda item: (-1 if item[0] is None else item[0], item[1]))
            count = int(explicit_count) if explicit_count is not None else int(np.ceil(len(items) * fraction))
            count = max(1, min(len(items), count))
            test_names.update(name for _, name in items[:count])
        return test_names

    raise ValueError(f"Unsupported test_split_policy: {dataset_param.get('test_split_policy')}")


def filtered_dir_names(data_param):
    dir_list = get_dir_name(data_param["data_dir"])

    include_names = data_param.get("include_episodes")
    if include_names is not None:
        include_names = _resolve_episode_name_set(dir_list, include_names)
        missing_include = sorted(include_names - set(dir_list))
        if missing_include:
            raise ValueError(
                "Dataset.include_episodes contains episodes not found in data_dir: "
                + ", ".join(missing_include[:10])
            )
        dir_list = [name for name in dir_list if name in include_names]

    exclude_names = data_param.get("exclude_episodes")
    if exclude_names is not None:
        exclude_names = _resolve_episode_name_set(dir_list, exclude_names)
        dir_list = [name for name in dir_list if name not in exclude_names]

    if not dir_list:
        raise ValueError(f"No episode directories selected in {data_param['data_dir']}")

    return dir_list

def cached_dir(data_param):
    dir_list=filtered_dir_names(data_param)
    total_data={}
    for dir in dir_list:
        total_data[dir]="./.cache/data_{}.pkl".format(dir)
    return total_data


def process_dir(args):
    import os
    import pickle
    import numpy as np
    import copy

    dir, data_param, file_list, mem = args
    data_exists = np.zeros(data_param["sequence_length"], dtype=bool)
    prev_data = None
    cache_data_file = f"./.cache/data_{dir}.pkl"
    cache_exists_file = f"./.cache/exist_{dir}.pkl"

    os.makedirs("./.cache", exist_ok=True)

    if mem == "rom":
        dir_data = []
        with open(cache_data_file, 'wb') as cache_data_f, open(cache_exists_file, 'wb') as cache_exists_f:
            for i, file in enumerate(file_list):
                try:
                    with open(os.path.join(data_param["data_dir"], dir, file), 'rb') as f:
                        try:
                            data = pickle.load(f)
                            for key, value in data.items():
                                if isinstance(value, np.ndarray) and value.dtype == np.float64:
                                    data[key] = value.astype(np.float32)
                            prev_data = copy.deepcopy(data)
                            data_exists[i] = True
                        except pickle.UnpicklingError as exc:
                            data = _fallback_or_raise(exc, prev_data, data_param)
                except FileNotFoundError:
                    data = _fallback_or_raise(
                        FileNotFoundError(f"No data available for {dir}/{file}."),
                        prev_data,
                        data_param,
                    )
                dir_data.append(data)
            dir_data = sequence_to_numpy({dir: dir_data}, data_param)[dir]
            pickle.dump(dir_data, cache_data_f)
            pickle.dump({"data_found":data_exists}, cache_exists_f)

        return dir, cache_data_file, cache_exists_file

    else:
        dir_data = []
        for i, file in enumerate(file_list):
            try:
                with open(os.path.join(data_param["data_dir"], dir, file), 'rb') as f:
                    try:
                        data = pickle.load(f)
                        for key, value in data.items():
                            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                                data[key] = value.astype(np.float32)
                        prev_data = copy.deepcopy(data)
                        data_exists[i] = True
                    except pickle.UnpicklingError as exc:
                        data = _fallback_or_raise(exc, prev_data, data_param)
            except FileNotFoundError:
                data = _fallback_or_raise(
                    FileNotFoundError(f"No data available for {dir}/{file}."),
                    prev_data,
                    data_param,
                )

            dir_data.append(data)
        data_exists={"data_found":data_exists}
        dir_data = sequence_to_numpy({dir: dir_data}, data_param)[dir]
        return dir, dir_data, data_exists


def process_dir_single(dir, data_param, file_list, mem):
    import os
    import pickle
    import numpy as np
    import copy

    data_exists = np.zeros(data_param["sequence_length"], dtype=bool)
    prev_data = None
    cache_data_file = f"./.cache/data_{dir}.pkl"
    cache_exists_file = f"./.cache/exist_{dir}.pkl"

    os.makedirs("./.cache", exist_ok=True)

    if mem == "rom":
        dir_data = []
        with open(cache_data_file, 'wb') as cache_data_f, open(cache_exists_file, 'wb') as cache_exists_f:
            for i, file in enumerate(file_list):
                try:
                    with open(os.path.join(data_param["data_dir"], dir, file), 'rb') as f:
                        try:
                            data = pickle.load(f)
                            for key, value in data.items():
                                if isinstance(value, np.ndarray) and value.dtype == np.float64:
                                    data[key] = value.astype(np.float32)
                            prev_data = copy.deepcopy(data)
                            data_exists[i] = True
                        except pickle.UnpicklingError as exc:
                            data = _fallback_or_raise(exc, prev_data, data_param)
                except FileNotFoundError:
                    data = _fallback_or_raise(
                        FileNotFoundError(f"No data available for {dir}/{file}."),
                        prev_data,
                        data_param,
                    )
                dir_data.append(data)
            dir_data = sequence_to_numpy({dir: dir_data}, data_param)[dir]
            pickle.dump(dir_data, cache_data_f)
            pickle.dump({"data_found":data_exists}, cache_exists_f)

        return dir, cache_data_file, cache_exists_file

    else:
        dir_data = []
        for i, file in enumerate(file_list):
            try:
                with open(os.path.join(data_param["data_dir"], dir, file), 'rb') as f:
                    try:
                        data = pickle.load(f)
                        for key, value in data.items():
                            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                                data[key] = value.astype(np.float32)
                        prev_data = copy.deepcopy(data)
                        data_exists[i] = True
                    except pickle.UnpicklingError as exc:
                        data = _fallback_or_raise(exc, prev_data, data_param)
            except FileNotFoundError:
                data = _fallback_or_raise(
                    FileNotFoundError(f"No data available for {dir}/{file}."),
                    prev_data,
                    data_param,
                )

            dir_data.append(data)
        data_exists={"data_found":data_exists}
        dir_data = sequence_to_numpy({dir: dir_data}, data_param)[dir]

        return dir, dir_data, data_exists
    
def get_sequence_dict_single(data_param, mem="rom"):
    data_param = _localize_legacy_paths(copy.deepcopy(data_param))
    dir_list = filtered_dir_names(data_param)
    file_list = _sequence_file_list(data_param, dir_list)

    total_data = {}
    data_found = {}

    for dir in tqdm(dir_list, desc="Processing directories", unit="dir"):
        print(dir)
        dir, dir_data, data_exists = process_dir_single(dir, data_param, file_list, mem)
        total_data[dir] = dir_data
        data_found[dir] = data_exists
    return total_data, data_found


def get_sequence_dict(data_param,mem="rom",proc_num=4):
    data_param = _localize_legacy_paths(copy.deepcopy(data_param))
    dir_list=filtered_dir_names(data_param)
    file_list=_sequence_file_list(data_param, dir_list)
    
    total_data={}
    data_found = {}

    with Pool(proc_num) as p:
        results = list(tqdm(p.imap(process_dir, [(dir, data_param, file_list,mem) for dir in dir_list]), total=len(dir_list), desc="Processing directories", unit="dir"))

    for dir, dir_data, data_exists in results:
        total_data[dir]=dir_data
        data_found[dir]=data_exists

    return total_data, data_found


def sequence_to_numpy(dir_data, dataset_param):
    total_data = {}
    for key, item in dir_data.items():
        modality_data = {}
        for modality in dataset_param["modality"].keys():
            data_list = [array[modality] for array in item]
            modality_data[modality] = np.stack(data_list, 0)
        total_data[key] = modality_data
        dir_data[key]=None
    return total_data

def scaling_param(modality_item,scaling_rule):
    if scaling_rule[1]=="each":
        series = pd.DataFrame(modality_item)
        stats = series.describe(percentiles=[0.02, 0.98])
        return stats
    elif scaling_rule[1]=="all":
        series = pd.DataFrame(modality_item.reshape(-1,1))
        stats = series.describe(percentiles=[0.02, 0.98])
        return stats
    else:
        return None
    

def combine_pandas_describe(describe_list):
    total_count = sum(desc.loc['count'] for desc in describe_list)
    total_sum = sum(desc.loc['mean'] * desc.loc['count'] for desc in describe_list)
    mean = total_sum / total_count

    sum_of_squares = sum(
        ((desc.loc['std'] ** 2) * (desc.loc['count'] - 1) + 
         desc.loc['mean'] ** 2 * desc.loc['count']) for desc in describe_list
    )
    variance = (sum_of_squares - total_count * mean**2) / (total_count - 1)
    std = np.sqrt(variance)

    min_values = pd.concat([desc.loc['min'] for desc in describe_list]).groupby(level=0).min()
    max_values = pd.concat([desc.loc['max'] for desc in describe_list]).groupby(level=0).max()

    result = pd.DataFrame({
        'count': total_count,
        'mean': mean,
        'std': std,
        'min': min_values,
        '25%': None,
        '50%': None,
        '75%': None,
        'max': max_values
    })
    
    return result.T

def change_img_size(dir_data, dataset_param,mem="ram"):
    img_dict = dataset_param["img_size"]
    for folder_nm in dir_data.keys():
        for key, img_size in img_dict.items():
            if mem=="ram":
                video_before = dir_data[folder_nm][key]
                video_after = np.zeros((video_before.shape[0], img_size[0], img_size[1], video_before.shape[3]))
                for i in range(video_before.shape[0]):
                    video_after[i] = cv2.resize(video_before[i], (img_size[1], img_size[0]))
                dir_data[folder_nm][key] = video_after
            elif mem=="rom":
                data=load_pkl_file(dir_data[folder_nm])
                video_before = data[key]
                video_after = np.zeros((video_before.shape[0], img_size[0], img_size[1], video_before.shape[3]))
                for i in range(video_before.shape[0]):
                    video_after[i] = cv2.resize(video_before[i], (img_size[1], img_size[0]))
                data[key] = video_after
                save_pkl_file(data,dir_data[folder_nm])
            else:
                print("Select ram or rom as mem")

    return dir_data

def crop_img(dir_data, dataset_param, mem="ram"):
    crop_dict = dataset_param.get("crop_coords", {})
    for folder_nm in dir_data.keys():
        for key, coords in crop_dict.items():
            if mem == "ram":
                video_before = dir_data[folder_nm][key]
                x1, y1, x2, y2 = coords
                cropped_shape = (video_before.shape[0], y2 - y1, x2 - x1, video_before.shape[3])
                video_after = np.empty(cropped_shape, dtype=video_before.dtype)
                for i in range(video_before.shape[0]):
                    video_after[i] = video_before[i, y1:y2, x1:x2]
                dir_data[folder_nm][key] = video_after
            elif mem == "rom":
                data=load_pkl_file(dir_data[folder_nm])
                video_before = data[key]
                x1, y1, x2, y2 = coords
                cropped_shape = (video_before.shape[0], y2 - y1, x2 - x1, video_before.shape[3])
                video_after = np.empty(cropped_shape, dtype=video_before.dtype)
                for i in range(video_before.shape[0]):
                    video_after[i] = video_before[i, y1:y2, x1:x2]
                data[key] = video_after
                save_pkl_file(data,dir_data[folder_nm])
            else:
                print("Select ram or rom as mem")
    return dir_data


# dataframeを返す
# しかし、標準偏差stdの値だけ合わないの注意
def get_scaling_param(dir_data,dataset_param,mem="ram"):
    
    if mem=="ram":
        modality_numpy={}
        for modality in dataset_param["modality"].keys():
            each_m_data=[]
            for key,item in dir_data.items():
                each_m_data+=[item[modality]]
            modality_numpy[modality]=np.concatenate(each_m_data,0)

        modality_scale_param={}
        for modality in dataset_param["modality"].keys():
            
            stats=scaling_param(modality_numpy[modality],dataset_param["modality"][modality])
            modality_scale_param[modality]=copy.deepcopy(stats)

    elif mem=="rom":

        modality_scale_param={}
        for i,episode_path in enumerate(dir_data):
            data=load_pkl_file(dir_data[episode_path])
            for modality in dataset_param["modality"].keys():
                if i==0:
                    modality_scale_param[modality]=[]
                prm=scaling_param(data[modality],dataset_param["modality"][modality])
                if prm is None:
                    modality_scale_param[modality]=None
                    continue
                else:
                    modality_scale_param[modality]+=[prm]

        for modality in dataset_param["modality"].keys():
            if modality_scale_param[modality] is not None:
                modality_scale_param[modality]=combine_pandas_describe(modality_scale_param[modality])
    else:
        print("The selection for the mem is wrong")
        sys.exit(1)
    return modality_scale_param


def save_pkl_file(data, file_path,log=False):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f,protocol=-1)
    if log:
        print("saving file at "+file_path)

def load_pkl_file(file_path, log=False):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if log:
        print("loading file from "+file_path)
    return data


def standardize_data(md, sc, sp):
    mean = _scaling_stat(sc, "mean", 1)
    std_dev = _scaling_stat(sc, "std", 2)
    safe_std = np.where(np.asarray(std_dev) == 0, 1.0, std_dev)
    md = (md - mean) / safe_std
    return md


def _scaling_stat(sc, label, fallback_idx):
    if isinstance(sc, pd.core.frame.DataFrame):
        if label in sc.index:
            return sc.loc[label].to_numpy()
        sc = sc.to_numpy()
    elif isinstance(sc, pd.core.series.Series):
        if label in sc.index:
            return sc.loc[label]
        sc = sc.to_numpy()

    sc = np.asarray(sc)
    if fallback_idx < sc.shape[0]:
        return sc[fallback_idx]
    if label == "max" and sc.shape[0] > 0:
        return sc[-1]
    raise IndexError(
        f"scaling statistics missing '{label}' row; "
        f"available shape is {sc.shape}"
    )

def normalize_data(md, sc, sp):
    min_value = _scaling_stat(sc, "min", 3)
    max_value = _scaling_stat(sc, "max", 7)
    range_ = max_value - min_value
    # Avoid division by zero for constant features (e.g. all-zero ring_tip).
    # When range is 0 all values equal min_value, so (md - min_value) = 0 and the
    # safe denominator of 1.0 maps every sample to sp[2][0] (a constant).
    safe_range = np.where(np.asarray(range_) == 0, 1.0, range_)
    md = (md - min_value) * (sp[2][1] - sp[2][0]) / safe_range + sp[2][0]
    return md

def robust_normalize_data(md, sc, sp):
    low_value = _scaling_stat(sc, "2%", 4)
    high_value = _scaling_stat(sc, "98%", 6)
    range_ = high_value - low_value
    safe_range = np.where(np.asarray(range_) == 0, 1.0, range_)
    md = (md - low_value) * (sp[2][1] - sp[2][0]) / safe_range + sp[2][0]
    max_idx = md > sp[2][1]
    min_idx = md < sp[2][0]
    md[max_idx] = sp[2][1]
    md[min_idx] = sp[2][0]
    return md

def limit_data(md,sc,sp):

    md=(md-sp[1][0])*(sp[2][1]-sp[2][0])/(sp[1][1]-sp[1][0])+sp[2][0]
    md = np.clip(md, a_min=sp[2][0], a_max=sp[2][1])
    return md

def scaling_data(modality_data,scaling_rule,scaling_param):

    if scaling_param[0]=="n":
        modality_data=normalize_data(modality_data,scaling_rule,scaling_param)
    elif scaling_param[0]=="rn":
        modality_data=robust_normalize_data(modality_data,scaling_rule,scaling_param)
    elif scaling_param[0]=="s":
        modality_data = standardize_data(modality_data, scaling_rule, scaling_param)

    elif scaling_param[0]=="l":
        modality_data=limit_data(modality_data,scaling_rule,scaling_param)
    else:
        pass
    return modality_data
    


def scale_dir_data(dir_data, scaling_param,dataset_param,mem="ram"):
    if mem=="ram":
        copied_dir_data=copy.deepcopy(dir_data)
        for modality in dataset_param["modality"].keys():
            for key,item in dir_data.items():
                if modality in item.keys():
                    modality_file_data=item[modality]
                    copied_dir_data[key][modality]=\
                        scaling_data(modality_file_data,scaling_param[modality],dataset_param["modality"][modality])  
        return copied_dir_data
    elif mem=="rom":
        for i,episode_path in enumerate(dir_data):
            data=load_pkl_file(dir_data[episode_path])
            scaled_data={}
            for modality in dataset_param["modality"].keys():
                scaled_data[modality]=scaling_data(data[modality],scaling_param[modality],dataset_param["modality"][modality])
            save_pkl_file(scaled_data,dir_data[episode_path])
        return dir_data

    else:  
        print("The selection for the mem is wrong")
        sys.exit(1)


def unnormalize_data(md, sc, sp):
    min_value = _scaling_stat(sc, "min", 3)
    max_value = _scaling_stat(sc, "max", 7)
    md = (md - sp[2][0]) * (max_value - min_value) / (sp[2][1] - sp[2][0]) + min_value
    return md

def robust_unnormalize_data(md, sc, sp):
    low_value = _scaling_stat(sc, "2%", 4)
    high_value = _scaling_stat(sc, "98%", 6)
    md = (md - sp[2][0]) * (high_value - low_value) / (sp[2][1] - sp[2][0]) + low_value
    return md

def unlimit_data(md, sc, sp):
    md = (md - sp[2][0]) * (sp[1][1] - sp[1][0]) / (sp[2][1] - sp[2][0]) + sp[1][0]
    return md

def unstandardize_data(md, sc, sp):
    mean = sc[1]
    std_dev = sc[2]
    md = md * std_dev + mean
    return md

def unscale_data(modality_data, scaling_rule, scaling_param):
    if isinstance(scaling_rule, pd.core.frame.DataFrame):
        scaling_rule = scaling_rule.to_numpy()
    if scaling_param[0] == "n":
        modality_data = unnormalize_data(modality_data, scaling_rule, scaling_param)
    elif scaling_param[0] == "rn":
        modality_data = robust_unnormalize_data(modality_data, scaling_rule, scaling_param)
    elif scaling_param[0] == "s":
        modality_data = unstandardize_data(modality_data, scaling_rule, scaling_param)
    elif scaling_param[0] == "l":
        modality_data = unlimit_data(modality_data, scaling_rule, scaling_param)
    else:
        pass
    return modality_data

def unscale_dir_data(dir_data, scaling_param, dataset_param, mem="ram"):
    if mem == "ram":
        copied_dir_data = copy.deepcopy(dir_data)
        for modality in dataset_param["modality"].keys():
            for key, item in dir_data.items():
                if modality in item:
                    modality_file_data = item[modality]
                    copied_dir_data[key][modality] = \
                        unscale_data(modality_file_data, scaling_param[modality], dataset_param["modality"][modality])
        return copied_dir_data
    elif mem == "rom":
        for i, episode_path in enumerate(dir_data):
            data = load_pkl_file(dir_data[episode_path])
            unscaled_data = {}
            for modality in dataset_param["modality"].keys():
                if modality in data:
                    unscaled_data[modality] = unscale_data(data[modality], scaling_param[modality], dataset_param["modality"][modality])
            save_pkl_file(unscaled_data, dir_data[episode_path])
        return dir_data
    else:
        print("The selection for the mem is wrong")
        sys.exit(1)


def modality_numpy(dir_data,dataset_param,data_found=None):
    modality_data=dataset_param["modality"].keys()
    modality_numpy={}
    for modality in modality_data:
        modality_list=[]
        for key,item in dir_data.items():
            modality_list+=[item[modality]]
        modality_numpy[modality]=np.stack(modality_list,0)
    if data_found is not None:
        modality_list=[]
        for key,_ in dir_data.items():
            modality_list+=[data_found[key]]
        modality_numpy["data_found"]=np.stack(modality_list,0)
    return modality_numpy


def rearrange(dir_data,dataset_param,mem="ram"):
    if mem=="ram":
        copied_dir_data=copy.deepcopy(dir_data)
        for modality in dataset_param["modality"].keys():
            for key,item in dir_data.items():
                if modality in item.keys():
                    if len(dir_data[key][modality].shape) == 1:
                        dir_data[key][modality] = dir_data[key][modality][:,None]
                    copied_dir_data[key][modality]=\
                        einops.rearrange(dir_data[key][modality],dataset_param["modality"][modality][3])  
        return copied_dir_data
    elif mem=="rom":
        for i,episode_path in enumerate(dir_data):
            data=load_pkl_file(dir_data[episode_path])
            arranged_data={}
            for modality in dataset_param["modality"].keys():
                if len(data[modality].shape) == 1:
                    data[modality] = data[modality][:,None]
                arranged_data[modality]=einops.rearrange(data[modality],dataset_param["modality"][modality][3]) 
            save_pkl_file(arranged_data,dir_data[episode_path])
        return dir_data
    else:  
        print("The selection for the mem is wrong")
        sys.exit(1)

def get_inverse_transform(transform):
    parts = transform.split('->')
    if len(parts) != 2:
        raise ValueError("Transform format is incorrect")
    left, right = parts
    left = left.strip()
    right = right.strip()
    inverse_transform = f"{right} -> {left}"
    return inverse_transform

def restore(dir_data, dataset_param, mem="ram"):
    if mem == "ram":
        restored_dir_data = copy.deepcopy(dir_data)
        for modality in dataset_param["modality"].keys():
            inverse_transform = get_inverse_transform(dataset_param["modality"][modality][3])
            for key, item in dir_data.items():
                if modality in item:
                    restored_dir_data[key][modality] = einops.rearrange(dir_data[key][modality], inverse_transform)
        return restored_dir_data
    elif mem == "rom":
        for i, episode_path in enumerate(dir_data):
            data = load_pkl_file(dir_data[episode_path])
            restored_data = {}
            for modality in dataset_param["modality"].keys():
                if modality in data:
                    inverse_transform = get_inverse_transform(dataset_param["modality"][modality][3])
                    restored_data[modality] = einops.rearrange(data[modality], inverse_transform)
            save_pkl_file(restored_data, dir_data[episode_path])
        return dir_data
    else:
        print("The selection for the mem is wrong")
        sys.exit(1)



def split_train_test(dir_data,dataset_param,key="test_data"):
    train_dir_data={}
    test_dir_data={}
    test_names = resolve_test_episode_names(dir_data.keys(), dataset_param, key)
    train_names = dataset_param.get("train_data")
    if train_names is not None:
        train_names = _resolve_episode_name_set(dir_data.keys(), train_names)
        if not test_names:
            test_names = set(train_names)

        missing_train = sorted(train_names - set(dir_data.keys()))
        if missing_train:
            raise ValueError(
                "Dataset.train_data contains episodes not found in data_dir: "
                + ", ".join(missing_train[:10])
            )

        for file_name,data in dir_data.items():
            if file_name in train_names:
                train_dir_data[file_name]=data
            if file_name in test_names:
                test_dir_data[file_name]=data
        return train_dir_data,test_dir_data

    for file_name,data in dir_data.items():
        if file_name in test_names:
            test_dir_data[file_name]=data

        else:
            train_dir_data[file_name]=data
    return train_dir_data,test_dir_data

def shift_data(data,dataset_param):
    train_data={}
    test_data={}
    for modality,value in data.items():
        train_data[modality]=value[:,:(-dataset_param["shift_data"])]
        test_data[modality]=value[:,dataset_param["shift_data"]:]
    return train_data,test_data


def expand_data_found(expand_length: int, data_found: dict, mem="ram"):
    expanded_data_found = {}
    for key, array_or_path in data_found.items():
        if mem == "ram":
            tensor_list = array_or_path["data_found"].tolist()
        elif mem == "rom":
            array = load_pkl_file(array_or_path)["data_found"]
            tensor_list = array.tolist()
        else:
            raise ValueError("Invalid memory mode. Please use 'ram' or 'rom'.")

        try:
            first_false_index = tensor_list.index(False)
        except ValueError:
            first_false_index = len(tensor_list)
        
        for i in range(expand_length):
            if first_false_index + i < len(tensor_list):
                tensor_list[first_false_index + i] = True
        
        expanded_array = np.array(tensor_list, dtype=bool)

        if mem == "ram":
            expanded_data_found[key] = {"data_found":expanded_array}
        elif mem == "rom":
            save_pkl_file({"data_found":expanded_array},array_or_path)

    return expanded_data_found if mem == "ram" else data_found
