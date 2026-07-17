import os
from typing import Mapping, Sequence

import numpy as np


FINGER_NAMES = ["index", "thumb", "middle", "ring"]
FINGER_KEYS = ["tactile_index_tip", "tactile_thumb_tip", "tactile_middle_tip", "tactile_ring_tip"]
COMBO_KEYS = (
    "thumb_index",
    "thumb_middle",
    "index_middle",
    "middle_ring",
    "thumb_index_middle",
    "index_middle_ring",
)
COMBO_TO_TACTILE_KEYS = {
    "thumb_index": ("tactile_thumb_tip", "tactile_index_tip"),
    "thumb_middle": ("tactile_thumb_tip", "tactile_middle_tip"),
    "index_middle": ("tactile_index_tip", "tactile_middle_tip"),
    "middle_ring": ("tactile_middle_tip", "tactile_ring_tip"),
    "thumb_index_middle": ("tactile_thumb_tip", "tactile_index_tip", "tactile_middle_tip"),
    "index_middle_ring": ("tactile_index_tip", "tactile_middle_tip", "tactile_ring_tip"),
}


def normalize_combo_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def combo_vector_for_episode(episode_name: str, combo_keys: Sequence[str] = COMBO_KEYS) -> np.ndarray:
    text = normalize_combo_name(episode_name)
    combo = np.zeros((len(combo_keys),), dtype=np.float32)
    for idx, name in enumerate(combo_keys):
        if text == name or text.startswith(f"{name}_episode"):
            combo[idx] = 1.0
            break
    if combo.sum() == 0.0:
        combo[:] = 1.0 / float(combo.size)
    return combo


def expected_combo_keys(dataset_param: Mapping) -> Sequence[str]:
    combinations = dataset_param.get("combinations") or []
    return tuple(normalize_combo_name(combo) for combo in combinations)


def present_combo_keys(episodes: Sequence[str], combo_keys: Sequence[str] = COMBO_KEYS) -> set:
    present = set()
    for episode_name in episodes:
        text = normalize_combo_name(episode_name)
        for combo in combo_keys:
            if text == combo or text.startswith(f"{combo}_episode"):
                present.add(combo)
    return present


def validate_selftouch_combinations(dir_data: Mapping, dataset_param: Mapping) -> None:
    if not bool(dataset_param.get("require_all_selftouch_combinations", True)):
        return
    expected = expected_combo_keys(dataset_param)
    if not expected:
        return
    present = present_combo_keys(dir_data.keys(), expected)
    missing = [combo for combo in expected if combo not in present]
    if missing:
        data_dir = dataset_param.get("data_dir", "<unknown>")
        available = sorted(present_combo_keys(dir_data.keys(), COMBO_KEYS))
        raise FileNotFoundError(
            "Dataset is configured for self-touch combinations "
            f"{list(expected)}, but {data_dir} is missing episode folders for "
            f"{missing}. Available combinations: {available}."
        )

    modality = dataset_param.get("modality", {})
    if any("ring" in combo for combo in expected) and "tactile_ring_tip" not in modality:
        raise KeyError(
            "Dataset.combinations includes ring combinations, but Dataset.modality "
            "does not include tactile_ring_tip."
        )


def validate_selftouch_data_dir(dataset_param: Mapping) -> None:
    data_dir = dataset_param.get("data_dir")
    if not data_dir or not os.path.isdir(data_dir):
        return
    episodes = {
        name: None
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    }
    validate_selftouch_combinations(episodes, dataset_param)


def tactile_keys_for_episode(episode_name: str, episode: Mapping) -> Sequence[str]:
    text = normalize_combo_name(episode_name)
    preferred = None
    for combo, keys in COMBO_TO_TACTILE_KEYS.items():
        if text == combo or text.startswith(f"{combo}_episode"):
            preferred = keys
            break
    if preferred is None:
        preferred = tuple(FINGER_KEYS)
    return [key for key in preferred if key in episode]
