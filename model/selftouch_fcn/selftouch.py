import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
from selftouch_feature_utils import build_joint_features, joint_feature_dim
from selftouch_loss_utils import active_tactile_loss, finger_loss_config, select_valid_finger_targets


class SelfTouch(nn.Module):
    """Pure FCN (no temporal) baseline for selftouch prediction.

    Input:  Joint positions(t) [hand_dim]
        Output: Predicted Self-touch Tactile(t) for index_tip, thumb_tip, middle_tip, and ring_tip [tactile_dim each]
    """

    def __init__(self, param):
        super().__init__()
        self.hand_dim = int(param["hand_dim"])
        self.tactile_dim = int(param["tactile_dim"])
        self.output_min = float(param.get("output_min", 0.1))
        self.output_max = float(param.get("output_max", 0.9))
        if not self.output_min < self.output_max:
            raise ValueError("output_min must be smaller than output_max")
        self.use_joint_pos_only = bool(param.get("use_joint_pos_only", True))
        self.use_derived_features = bool(param.get("use_derived_features", False))
        self.temporal_window_steps = max(1, int(param.get("temporal_window_steps", 1)))
        self.use_tactile_history = bool(param.get("use_tactile_history", False))
        self.tactile_history_steps = max(1, int(param.get("tactile_history_steps", 1)))
        self.tactile_history_fingers = tuple(
            param.get("tactile_history_fingers", ("index", "thumb", "middle", "ring"))
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
        rec_out = self.tactile_dim * 4  # index_tip + thumb_tip + middle_tip + ring_tip

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
        return self.output_min + (self.output_max - self.output_min) * torch.sigmoid(x)

    def output_bias_from_normalized_target(self, target):
        unit = (target - self.output_min) / (self.output_max - self.output_min)
        eps = torch.finfo(target.dtype).eps
        return torch.logit(unit.clamp(min=eps, max=1.0 - eps))

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
        middle_pred = out[..., self.tactile_dim * 2:self.tactile_dim * 3]
        ring_pred = out[..., self.tactile_dim * 3:]
        return idx_pred, thumb_pred, middle_pred, ring_pred

    def forward_loss(self, tactile_index_tip, tactile_thumb_tip,
                     hand_jnt_pos=None, hand_jnt_cmd_pos=None, hand_jnt_vel=None, hand_jnt_trq=None,
                     data_found=None, selftouch_finger_mask=None, loss_coef=None,
                     tactile_middle_tip=None, tactile_ring_tip=None, **_unused):
        loss_coef = loss_coef or {}
        idx_pred, thumb_pred, middle_pred, ring_pred = self.forward(
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
        target_start = 1
        target_stop = tactile_index_tip.shape[1]
        index_target = tactile_index_tip[:, target_start:target_stop, :]
        thumb_target = tactile_thumb_tip[:, target_start:target_stop, :]
        middle_target = tactile_middle_tip[:, target_start:target_stop, :]
        ring_target = tactile_ring_tip[:, target_start:target_stop, :]
        idx_loss_pred, index_target = select_valid_finger_targets(
            idx_pred, index_target, selftouch_finger_mask, "index", target_start, target_stop
        )
        thumb_loss_pred, thumb_target = select_valid_finger_targets(
            thumb_pred, thumb_target, selftouch_finger_mask, "thumb", target_start, target_stop
        )
        middle_loss_pred, middle_target = select_valid_finger_targets(
            middle_pred, middle_target, selftouch_finger_mask, "middle", target_start, target_stop
        )
        ring_loss_pred, ring_target = select_valid_finger_targets(
            ring_pred, ring_target, selftouch_finger_mask, "ring", target_start, target_stop
        )
        loss_index = active_tactile_loss(
            idx_loss_pred,
            index_target,
            finger_loss_config(loss_coef, "tactile_index_tip"),
        )
        loss_thumb = active_tactile_loss(
            thumb_loss_pred,
            thumb_target,
            finger_loss_config(loss_coef, "tactile_thumb_tip"),
        )
        loss_middle = active_tactile_loss(
            middle_loss_pred,
            middle_target,
            finger_loss_config(loss_coef, "tactile_middle_tip"),
        )
        loss_ring = active_tactile_loss(
            ring_loss_pred,
            ring_target,
            finger_loss_config(loss_coef, "tactile_ring_tip"),
        )

        total_loss = (
            loss_coef.get("tactile_index_tip", 1.0) * loss_index
            + loss_coef.get("tactile_thumb_tip", 1.0) * loss_thumb
            + loss_coef.get("tactile_middle_tip", 1.0) * loss_middle
            + loss_coef.get("tactile_ring_tip", 1.0) * loss_ring
        )
        return (total_loss, loss_index, loss_thumb, loss_middle, loss_ring), (
            idx_pred,
            thumb_pred,
            middle_pred,
            ring_pred,
        )


if __name__ == "__main__":
    print("Model for predicting self touch")
