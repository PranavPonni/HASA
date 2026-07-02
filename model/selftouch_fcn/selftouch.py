import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
from selftouch_loss_utils import active_tactile_loss


class SelfTouch(nn.Module):
    """Pure FCN (no temporal) baseline for selftouch prediction.

    Input:  Joint positions(t) [hand_dim]
    Output: Predicted Self-touch Tactile(t) for index_tip, thumb_tip, and middle_tip [tactile_dim each]
    """

    def __init__(self, param):
        super().__init__()
        self.hand_dim = int(param["hand_dim"])
        self.tactile_dim = param["tactile_dim"]
        rec_dim = param["rec_dim"]
        dropout = param["dropout"]
        activation = nn.GELU()

        rec_in = self.hand_dim
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

    def forward(self, hand_jnt_pos, **_unused):
        x = hand_jnt_pos[:, 1:, :]
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
        idx_pred, thumb_pred, middle_pred = self.forward(hand_jnt_pos=hand_jnt_pos)

        # Targets: tactile at t=1..T-1
        loss_index = active_tactile_loss(idx_pred, tactile_index_tip[:, 1:, :], loss_coef)
        loss_thumb = active_tactile_loss(thumb_pred, tactile_thumb_tip[:, 1:, :], loss_coef)
        loss_middle = active_tactile_loss(middle_pred, tactile_middle_tip[:, 1:, :], loss_coef)

        total_loss = (
            loss_coef.get("tactile_index_tip", 1.0) * loss_index
            + loss_coef.get("tactile_thumb_tip", 1.0) * loss_thumb
            + loss_coef.get("tactile_middle_tip", 1.0) * loss_middle
        )
        return (total_loss, loss_index, loss_thumb, loss_middle), (idx_pred, thumb_pred, middle_pred)


if __name__ == "__main__":
    print("Model for predicting self touch")
