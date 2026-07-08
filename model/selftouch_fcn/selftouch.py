import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
from selftouch_feature_utils import build_joint_features, joint_feature_dim
from selftouch_loss_utils import active_tactile_loss, finger_loss_config


class SelfTouch(nn.Module):
    """Pure FCN (no temporal) baseline for selftouch prediction.

    Input:  Joint positions(t) [hand_dim]
    Output: Predicted Self-touch Tactile(t) for index_tip, thumb_tip, and middle_tip [tactile_dim each]
    """

    def __init__(self, param):
        super().__init__()
        self.hand_dim = int(param["hand_dim"])
        self.tactile_dim = int(param["tactile_dim"])
        self.use_joint_pos_only = bool(param.get("use_joint_pos_only", True))
        self.use_derived_features = bool(param.get("use_derived_features", False))
        self.temporal_window_steps = max(1, int(param.get("temporal_window_steps", 1)))
        self.use_tactile_history = bool(param.get("use_tactile_history", False))
        self.tactile_history_steps = max(1, int(param.get("tactile_history_steps", 1)))
        self.tactile_history_fingers = tuple(
            param.get("tactile_history_fingers", ("index", "thumb", "middle"))
        )
        rec_dim = param["rec_dim"]
        dropout = param["dropout"]
        activation = nn.GELU()

        rec_in = joint_feature_dim(
            self.hand_dim,
            self.use_derived_features,
            self.use_joint_pos_only,
            temporal_window_steps=self.temporal_window_steps,
            tactile_dim=self.tactile_dim,
            use_tactile_history=self.use_tactile_history,
            tactile_history_steps=self.tactile_history_steps,
            tactile_history_fingers=self.tactile_history_fingers,
        )
        rec_out = self.tactile_dim * 3  # index_tip + thumb_tip + middle_tip

        self.net = nn.Sequential(
            nn.Linear(rec_in, rec_dim // 2),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(rec_dim // 2, rec_dim),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(rec_dim, rec_out),
        )
        self._init_tactile_bias()

    def _init_tactile_bias(self):
        return

    def _activate_output(self, x):
        return x

    def forward(
        self,
        hand_jnt_pos,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        hand_jnt_cmd_pos=None,
        tactile_index_tip=None,
        tactile_thumb_tip=None,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
        **_unused,
    ):
        x = build_joint_features(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            next_step=True,
            use_derived_features=self.use_derived_features,
            use_joint_pos_only=self.use_joint_pos_only,
            temporal_window_steps=self.temporal_window_steps,
            use_tactile_history=self.use_tactile_history,
            tactile_history_steps=self.tactile_history_steps,
            tactile_history_fingers=self.tactile_history_fingers,
            tactile_dim=self.tactile_dim,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
        )
        out = self._activate_output(self.net(x))
        idx_pred = out[..., :self.tactile_dim]
        thumb_pred = out[..., self.tactile_dim:self.tactile_dim * 2]
        middle_pred = out[..., self.tactile_dim * 2:]
        return idx_pred, thumb_pred, middle_pred

    def forward_loss(self, tactile_index_tip, tactile_thumb_tip,
                     hand_jnt_pos=None, hand_jnt_cmd_pos=None, hand_jnt_vel=None, hand_jnt_trq=None,
                     data_found=None, loss_coef=None,
                     tactile_middle_tip=None, tactile_ring_tip=None, **_unused):
        loss_coef = loss_coef or {}
        idx_pred, thumb_pred, middle_pred = self.forward(
            hand_jnt_pos=hand_jnt_pos,
            hand_jnt_vel=hand_jnt_vel,
            hand_jnt_trq=hand_jnt_trq,
            hand_jnt_cmd_pos=hand_jnt_cmd_pos,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
        )

        # Targets: tactile at t=1..T-1
        loss_index = active_tactile_loss(
            idx_pred,
            tactile_index_tip[:, 1:, :],
            finger_loss_config(loss_coef, "tactile_index_tip"),
        )
        loss_thumb = active_tactile_loss(
            thumb_pred,
            tactile_thumb_tip[:, 1:, :],
            finger_loss_config(loss_coef, "tactile_thumb_tip"),
        )
        loss_middle = active_tactile_loss(
            middle_pred,
            tactile_middle_tip[:, 1:, :],
            finger_loss_config(loss_coef, "tactile_middle_tip"),
        )

        total_loss = (
            loss_coef.get("tactile_index_tip", 1.0) * loss_index
            + loss_coef.get("tactile_thumb_tip", 1.0) * loss_thumb
            + loss_coef.get("tactile_middle_tip", 1.0) * loss_middle
        )
        return (total_loss, loss_index, loss_thumb, loss_middle), (idx_pred, thumb_pred, middle_pred)


if __name__ == "__main__":
    print("Model for predicting self touch")
