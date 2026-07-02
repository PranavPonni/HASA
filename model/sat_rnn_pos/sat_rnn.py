import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
import torch.nn.functional as F
from util import mask_mse_loss, add_gausian_noise, closed_loop
import pdb

class ExternalTouchRNN(nn.Module):
    def __init__(self, selftouch: nn.Module, param: dict):
        super().__init__()
        hid_dim = param["hid_dim"]
        self.hand_dim = param["hand_dim"]
        self.tactile_dim = param["tactile_dim"]
        self.selftouch = selftouch.eval()
        activation = nn.LeakyReLU()

        rec_in = 2 * self.tactile_dim + 2 * self.tactile_dim + self.hand_dim  # ext_idx, ext_thumb, idx_self, thumb_self, jnt_pos
        rec_out = 2 * self.tactile_dim + 2 * self.hand_dim
        self.split_sizes = [
            self.tactile_dim,  # idx_ext
            self.tactile_dim,  # thumb_ext
            self.hand_dim,     # jnt_pos
            self.hand_dim,     # jnt_vel
        ]

        self.lstm_cell = nn.LSTMCell(rec_in, hid_dim)
        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            activation,
            nn.Linear(hid_dim // 2, rec_out),
        )

        self.freeze_selftouch()

    def freeze_selftouch(self):
        for p in self.selftouch.parameters():
            p.requires_grad = False

    def forward(
        self,
        tactile_index_tip_t,
        tactile_thumb_tip_t,
        hand_jnt_pos_t,
        hand_jnt_vel_t,
        prev_self_touch=None,
        hidden=None,
    ):
        # Get self-touch prediction for current step
        if prev_self_touch is None:
            idx_self_t, thumb_self_t, *_ = self.selftouch(hand_jnt_pos_t, hand_jnt_vel_t)
        else:
            idx_self_t, thumb_self_t = prev_self_touch

        # Build LSTM input
        x_t = torch.cat([
            tactile_index_tip_t,
            tactile_thumb_tip_t,
            idx_self_t,
            thumb_self_t,
            hand_jnt_pos_t
        ], dim=-1)

        # RNN step
        h_next, c_next = self.lstm_cell(x_t, hidden) if hidden is not None else self.lstm_cell(x_t)
        out = self.dec(h_next)

        # Split outputs
        total_idx_pred, total_thumb_pred, jnt_pos_pred, jnt_vel_pred = torch.split(out, self.split_sizes, dim=-1)

        # Predict next self-touch using predicted pos and vel
        idx_self_next, thumb_self_next, *_ = self.selftouch(jnt_pos_pred, jnt_vel_pred)

        return (
            jnt_pos_pred,
            jnt_vel_pred,
            total_idx_pred,
            total_thumb_pred,
        ), (idx_self_next, thumb_self_next), (h_next, c_next)

    def forward_loss(
        self,
        tactile_index_tip: torch.Tensor,
        tactile_thumb_tip: torch.Tensor,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_vel: torch.Tensor,
        data_found: torch.Tensor,
        loss_coef: dict,
        cls_rate=None,
        noise=None
    ):
        B, T, _ = tactile_index_tip.shape
        prev_self_touch = None
        hidden = None

        # Collect predictions
        idx_preds, thumb_preds, pos_preds, vel_preds = [], [], [], []

        for t in range(T-1):
            # Add noise
            tin_t = add_gausian_noise(tactile_index_tip[:, t], noise["tactile_noise"])
            tth_t = add_gausian_noise(tactile_thumb_tip[:, t], noise["tactile_noise"])
            pos_t = add_gausian_noise(hand_jnt_pos[:, t], noise["joint_noise"])
            vel_t = add_gausian_noise(hand_jnt_vel[:, t], noise["joint_noise"])

            # Closed-loop correction
            if t != 0:
                pos_t = closed_loop(pos_t, pos_pred, cls_rate["joint_rate"])
                tin_t = closed_loop(tin_t, total_idx_pred, cls_rate["tactile_rate"])
                tth_t = closed_loop(tth_t, total_thumb_pred, cls_rate["tactile_rate"])

            # Forward step
            preds, (idx_self_next, thumb_self_next), hidden = self.forward(
                tin_t, tth_t, pos_t, vel_t, prev_self_touch, hidden
            )
            pos_pred, vel_pred, total_idx_pred, total_thumb_pred = preds
            prev_self_touch = (idx_self_next, thumb_self_next)

            # Store
            idx_preds.append(total_idx_pred)
            thumb_preds.append(total_thumb_pred)
            pos_preds.append(pos_pred)
            vel_preds.append(vel_pred)

        # Stack
        idx_preds = torch.stack(idx_preds, dim=1)
        thumb_preds = torch.stack(thumb_preds, dim=1)
        pos_preds = torch.stack(pos_preds, dim=1)
        vel_preds = torch.stack(vel_preds, dim=1)

        # Losses
        loss_idx = mask_mse_loss(idx_preds, tactile_index_tip[:, 1:], data_found[:, 1:])
        loss_thumb = mask_mse_loss(thumb_preds, tactile_thumb_tip[:, 1:], data_found[:, 1:])
        loss_pos = mask_mse_loss(pos_preds, hand_jnt_pos[:, 1:], data_found[:, 1:])
        loss_vel = mask_mse_loss(vel_preds, hand_jnt_vel[:, 1:], data_found[:, 1:])

        total_loss = (
            loss_idx * loss_coef['tactile_index_tip'] +
            loss_thumb * loss_coef['tactile_thumb_tip'] +
            loss_pos * loss_coef['hand_jnt_pos'] +
            loss_vel * loss_coef['hand_jnt_vel']
        )

        return (total_loss, loss_idx, loss_thumb, loss_pos, loss_vel), (idx_preds, thumb_preds, pos_preds, vel_preds)

if __name__ == "__main__":
    print("ExternalTouchRNN with selftouch_fcn_pos pretrained selftouch module")
