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
        rec_out = self.tactile_dim * 4 + self.hand_dim  # index, thumb, middle, ring, pos

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

        # Split sizes: [index_tip, thumb_tip, middle_tip, ring_tip, joint_pos]
        self.split_sizes = [
            self.tactile_dim,  # index_tip
            self.tactile_dim,  # thumb_tip
            self.tactile_dim,  # middle_tip
            self.tactile_dim,  # ring_tip
            self.hand_dim,     # jnt_pos
        ]

    def encode(self, hand_jnt_pos, hand_jnt_vel):
        """Return per-timestep encoder representation (B, T, rec_dim)."""
        x = torch.cat([hand_jnt_pos, hand_jnt_vel], dim=-1)
        return self.enc(x)

    def forward(self, hand_jnt_pos, hand_jnt_vel):
        h = self.encode(hand_jnt_pos, hand_jnt_vel)
        idx_pred, thumb_pred, middle_pred, ring_pred, pos_pred = torch.split(
            self.dec(h), self.split_sizes, dim=-1
        )
        return idx_pred, thumb_pred, middle_pred, ring_pred, pos_pred


if __name__ == "__main__":
    print("Model for predicting self touch without velocity and torque")
