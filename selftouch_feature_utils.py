import torch
from torch import nn


DEFAULT_TACTILE_HISTORY_FINGERS = ("index", "thumb", "middle", "ring")
JOINT_MODALITIES = (
    "hand_jnt_pos",
    "hand_jnt_vel",
    "hand_jnt_trq",
    "hand_jnt_cmd_pos",
)


def _positive_int(value, default=1):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(1, value)


def _finger_names(fingers=None):
    if fingers is None:
        return DEFAULT_TACTILE_HISTORY_FINGERS
    if isinstance(fingers, str):
        return tuple(part.strip() for part in fingers.split(",") if part.strip())
    return tuple(fingers)


def _joint_modalities(input_modalities=None):
    if input_modalities is None:
        return None
    if isinstance(input_modalities, str):
        input_modalities = (
            part.strip() for part in input_modalities.split(",") if part.strip()
        )
    modalities = tuple(input_modalities)
    if not modalities:
        raise ValueError("input_modalities must contain at least one joint stream")
    unknown = [name for name in modalities if name not in JOINT_MODALITIES]
    if unknown:
        raise ValueError(
            "Unsupported joint input modality/modalities: " + ", ".join(unknown)
        )
    return modalities


def labels_from_selftouch_combo(selftouch_combo):
    """Convert a one-hot self-touch combination stream into class labels."""
    if selftouch_combo is None:
        return None
    if not torch.is_tensor(selftouch_combo):
        selftouch_combo = torch.as_tensor(selftouch_combo)
    if selftouch_combo.ndim < 2 or selftouch_combo.shape[-1] < 1:
        return None
    combo = selftouch_combo.float()
    if combo.ndim >= 3:
        combo = combo.mean(dim=1)
    elif combo.ndim > 2:
        combo = combo.reshape(combo.shape[0], -1, combo.shape[-1]).mean(dim=1)
    return combo.argmax(dim=-1).long()


def joint_feature_dim(
    hand_dim,
    use_derived_features=False,
    use_joint_pos_only=False,
    *,
    input_modalities=None,
    temporal_window_steps=1,
    tactile_dim=90,
    use_tactile_history=False,
    tactile_history_steps=1,
    tactile_history_fingers=None,
):
    """Return input width for the joint-state feature vector."""
    modalities = _joint_modalities(input_modalities)
    if modalities is not None:
        base_dim = int(hand_dim) * len(modalities)
        if use_derived_features:
            base_dim += int(hand_dim) * 4
    elif use_joint_pos_only:
        base_dim = int(hand_dim)
    else:
        base_dim = int(hand_dim) * (8 if use_derived_features else 4)
    base_dim *= _positive_int(temporal_window_steps)

    if use_tactile_history:
        base_dim += (
            int(tactile_dim)
            * _positive_int(tactile_history_steps)
            * len(_finger_names(tactile_history_fingers))
        )
    return base_dim


def _causal_delta(x):
    delta = torch.zeros_like(x)
    if x.shape[1] > 1:
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return delta


def causal_temporal_window(x, start=0, stop=None, steps=1):
    """Flatten the current and previous ``steps - 1`` frames for each timestep."""
    steps = _positive_int(steps)
    stop = x.shape[1] if stop is None else int(stop)
    start = int(start)
    length = max(0, stop - start)
    if steps <= 1:
        return x[:, start:stop, :]

    chunks = []
    for lag in range(steps):
        src_start = start - lag
        src_stop = stop - lag
        chunk = x.new_zeros(x.shape[0], length, x.shape[2])
        valid_start = max(src_start, 0)
        valid_stop = min(src_stop, x.shape[1])
        if valid_stop > valid_start:
            dst_start = valid_start - src_start
            dst_stop = dst_start + (valid_stop - valid_start)
            chunk[:, dst_start:dst_stop, :] = x[:, valid_start:valid_stop, :]
        chunks.append(chunk)
    return torch.cat(chunks, dim=-1)


def build_tactile_history(
    *,
    tactile_index_tip=None,
    tactile_thumb_tip=None,
    tactile_middle_tip=None,
    tactile_ring_tip=None,
    target_start=1,
    target_stop=None,
    history_steps=1,
    tactile_dim=90,
    fingers=None,
    reference=None,
):
    """Return lagged tactile ground-truth features aligned to target timesteps.

    For each target timestep ``t``, the output contains tactile values from
    ``t-1, t-2, ...``. Missing early history is zero padded.
    """
    fingers = _finger_names(fingers)
    values = {
        "index": tactile_index_tip,
        "thumb": tactile_thumb_tip,
        "middle": tactile_middle_tip,
        "ring": tactile_ring_tip,
    }
    first_value = next((values[name] for name in fingers if values.get(name) is not None), None)
    if reference is None:
        reference = first_value
    if reference is None:
        raise ValueError("reference or at least one tactile tensor is required for tactile history")

    history_steps = _positive_int(history_steps)
    target_start = int(target_start)
    if target_stop is None:
        if first_value is not None:
            target_stop = int(first_value.shape[1])
        else:
            target_stop = int(reference.shape[1])
    target_stop = int(target_stop)
    length = max(0, target_stop - target_start)
    batch = int(reference.shape[0])
    width = int(tactile_dim)
    chunks = []

    for lag in range(1, history_steps + 1):
        for name in fingers:
            value = values.get(name)
            if value is None:
                chunks.append(reference.new_zeros(batch, length, width))
                continue
            src_start = target_start - lag
            src_stop = target_stop - lag
            chunk = value.new_zeros(batch, length, value.shape[-1])
            valid_start = max(src_start, 0)
            valid_stop = min(src_stop, value.shape[1])
            if valid_stop > valid_start:
                dst_start = valid_start - src_start
                dst_stop = dst_start + (valid_stop - valid_start)
                chunk[:, dst_start:dst_stop, :] = value[:, valid_start:valid_stop, :]
            chunks.append(chunk)

    if not chunks:
        return reference.new_zeros(batch, length, 0)
    return torch.cat(chunks, dim=-1)


def build_joint_features(
    hand_jnt_pos,
    hand_jnt_vel=None,
    hand_jnt_trq=None,
    hand_jnt_cmd_pos=None,
    *,
    next_step=False,
    use_derived_features=False,
    use_joint_pos_only=False,
    temporal_window_steps=1,
    use_tactile_history=False,
    tactile_history_steps=1,
    tactile_history_fingers=None,
    tactile_dim=90,
    input_modalities=None,
    tactile_index_tip=None,
    tactile_thumb_tip=None,
    tactile_middle_tip=None,
    tactile_ring_tip=None,
):
    """Build causal joint-state features for self-touch prediction.

    The derived features use only joint state at or before each timestep:
    command error, torque delta, velocity delta, and absolute torque.
    """
    modalities = _joint_modalities(input_modalities)
    if modalities is not None:
        streams = {
            "hand_jnt_pos": hand_jnt_pos,
            "hand_jnt_vel": hand_jnt_vel,
            "hand_jnt_trq": hand_jnt_trq,
            "hand_jnt_cmd_pos": hand_jnt_cmd_pos,
        }
        missing = [name for name in modalities if streams.get(name) is None]
        if missing:
            raise KeyError(
                "Missing required joint input modality/modalities: "
                + ", ".join(missing)
            )
        features = torch.cat([streams[name] for name in modalities], dim=-1)
        if use_derived_features:
            zeros = torch.zeros_like(hand_jnt_pos)
            hand_jnt_vel = zeros if hand_jnt_vel is None else hand_jnt_vel
            hand_jnt_trq = zeros if hand_jnt_trq is None else hand_jnt_trq
            hand_jnt_cmd_pos = zeros if hand_jnt_cmd_pos is None else hand_jnt_cmd_pos
            features = torch.cat(
                [
                    features,
                    hand_jnt_cmd_pos - hand_jnt_pos,
                    _causal_delta(hand_jnt_trq),
                    _causal_delta(hand_jnt_vel),
                    hand_jnt_trq.abs(),
                ],
                dim=-1,
            )
    elif use_joint_pos_only:
        features = hand_jnt_pos
    else:
        zeros = torch.zeros_like(hand_jnt_pos)
        hand_jnt_vel = zeros if hand_jnt_vel is None else hand_jnt_vel
        hand_jnt_trq = zeros if hand_jnt_trq is None else hand_jnt_trq
        hand_jnt_cmd_pos = zeros if hand_jnt_cmd_pos is None else hand_jnt_cmd_pos

        chunks = [
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
        ]
        if use_derived_features:
            chunks.extend(
                [
                    hand_jnt_cmd_pos - hand_jnt_pos,
                    _causal_delta(hand_jnt_trq),
                    _causal_delta(hand_jnt_vel),
                    hand_jnt_trq.abs(),
                ]
            )
        features = torch.cat(chunks, dim=-1)

    input_start = 0
    input_stop = features.shape[1]
    target_start = 0
    target_stop = features.shape[1]
    if next_step:
        input_stop = max(0, features.shape[1] - 1)
        target_start = 1
        target_stop = features.shape[1]

    out = causal_temporal_window(
        features,
        start=input_start,
        stop=input_stop,
        steps=temporal_window_steps,
    )
    if use_tactile_history:
        tactile_history = build_tactile_history(
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
            target_start=target_start,
            target_stop=target_stop,
            history_steps=tactile_history_steps,
            tactile_dim=tactile_dim,
            fingers=tactile_history_fingers,
            reference=features,
        )
        out = torch.cat([out, tactile_history.to(device=out.device, dtype=out.dtype)], dim=-1)
    return out


class TactileHead(nn.Module):
    """Predict tactile values directly from model hidden state."""

    def __init__(self, hidden_dim, input_dim, tactile_dim):
        super().__init__()
        self.hidden = nn.Linear(int(hidden_dim), int(tactile_dim))

    def forward(self, hidden, joint_features=None):
        return self.hidden(hidden)
