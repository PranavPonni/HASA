import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
import torch.nn.functional as F


class SelfTouch(nn.Module):
    def __init__(self, param):
        super().__init__()

        rec_dim = param["rec_dim"]
        self.hand_dim = param["hand_dim"]
        self.tactile_dim = param["tactile_dim"]
        dropout = param["dropout"]
        activation = nn.GELU()

        rec_in = self.hand_dim * 2
        rec_out = self.tactile_dim * 2 + self.hand_dim

        self.enc = nn.Sequential(
            nn.Linear(rec_in, rec_dim // 2),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(rec_dim // 2, rec_dim),
            activation,
        )

        self.dec = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(rec_dim, rec_dim // 2),
            activation,
            nn.Linear(rec_dim // 2, rec_out),
            activation,
        )

        # Split sizes for decoding: [index_tip, thumb_tip, joint_pos]
        self.split_sizes = [
            self.tactile_dim,  # index_tip
            self.tactile_dim,  # thumb_tip
            self.hand_dim,     # jnt_pos
        ]

    def forward_loss(
        self,
        tactile_index_tip,
        tactile_thumb_tip,
        hand_jnt_pos,
        hand_jnt_cmd_pos,
        data_found,
        loss_coef
    ):
        # tactile_index_tip (b,t,d)
        # tactile_thumb_tip (b,t,d)
        # hand_jnt_pos (b,t,d)
        # hand_jnt_cmd_pos (b,t,d)

        idx_pred, thumb_pred, pos_pred = self.forward(
            hand_jnt_pos,
            hand_jnt_cmd_pos
        )

        loss_index = F.mse_loss(
            idx_pred[:, :-1, :],
            tactile_index_tip[:, 1:, :],
        )
        loss_thumb = F.mse_loss(
            thumb_pred[:, :-1, :],
            tactile_thumb_tip[:, 1:, :],
        )
        loss_pos = F.mse_loss(
            pos_pred[:, :-1, :],
            hand_jnt_pos[:, 1:, :],
        )

        total_loss = (
            loss_index * loss_coef["tactile_index_tip"] +
            loss_thumb * loss_coef["tactile_thumb_tip"] +
            loss_pos * loss_coef["hand_jnt_pos"]
        )

        return (total_loss, loss_index, loss_thumb, loss_pos), (idx_pred, thumb_pred, pos_pred)

    def forward(self, hand_jnt_pos, hand_jnt_cmd_pos):
        # hand_jnt_pos (b,t,d)
        # hand_jnt_cmd_pos (b,t,d)

        x = torch.cat([
            hand_jnt_pos,
            hand_jnt_cmd_pos
        ], dim=-1)

        h = self.enc(x)
        idx_pred, thumb_pred, pos_pred = torch.split(
            self.dec(h), self.split_sizes, dim=-1
        )
        return idx_pred, thumb_pred, pos_pred


if __name__ == "__main__":
    print("Model for predicting self touch without velocity and torque")
