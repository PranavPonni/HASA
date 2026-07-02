import torch
from torch import nn


def joint_feature_dim(hand_dim, use_derived_features=False, use_joint_pos_only=False):
    """Return input width for the joint-state feature vector."""
    if use_joint_pos_only:
        return int(hand_dim)
    return int(hand_dim) * (8 if use_derived_features else 4)


def _causal_delta(x):
    delta = torch.zeros_like(x)
    if x.shape[1] > 1:
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return delta


def build_joint_features(
    hand_jnt_pos,
    hand_jnt_vel=None,
    hand_jnt_trq=None,
    hand_jnt_cmd_pos=None,
    *,
    next_step=False,
    use_derived_features=False,
    use_joint_pos_only=False,
):
    """Build causal joint-state features for self-touch prediction.

    The derived features use only joint state at or before each timestep:
    command error, torque delta, velocity delta, and absolute torque.
    """
    if use_joint_pos_only:
        return hand_jnt_pos[:, :-1, :] if next_step else hand_jnt_pos

    zeros = torch.zeros_like(hand_jnt_pos)
    hand_jnt_vel = zeros if hand_jnt_vel is None else hand_jnt_vel
    hand_jnt_trq = zeros if hand_jnt_trq is None else hand_jnt_trq
    hand_jnt_cmd_pos = zeros if hand_jnt_cmd_pos is None else hand_jnt_cmd_pos

    features = [
        hand_jnt_pos,
        hand_jnt_vel,
        hand_jnt_trq,
        hand_jnt_cmd_pos,
    ]
    if use_derived_features:
        features.extend(
            [
                hand_jnt_cmd_pos - hand_jnt_pos,
                _causal_delta(hand_jnt_trq),
                _causal_delta(hand_jnt_vel),
                hand_jnt_trq.abs(),
            ]
        )

    if next_step:
        features = [feat[:, :-1, :] for feat in features]
    return torch.cat(features, dim=-1)


class TactileHead(nn.Module):
    """Predict tactile values directly from model hidden state."""

    def __init__(self, hidden_dim, input_dim, tactile_dim):
        super().__init__()
        self.hidden = nn.Linear(int(hidden_dim), int(tactile_dim))

    def forward(self, hidden, joint_features=None):
        return self.hidden(hidden)
