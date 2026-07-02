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

        rec_in = 2 * self.tactile_dim + self.hand_dim  # ext_idx, ext_thumb, idx_self, thumb_self, jnt_pos
        rec_out = 2 * self.hand_dim

        self.split_sizes = [
            self.hand_dim,     # jnt_pos
            self.hand_dim,     # jnt_cmd_pos
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
        hand_jnt_pos_t,
        prev_self_touch=None,
        hidden=None,
    ):
        # Get self-touch prediction for current step
        if prev_self_touch is None:
            seq_args = [hand_jnt_pos_t, hand_jnt_pos_t]
            idx_self_t, thumb_self_t, _ = self.selftouch(*seq_args)
        else:
            idx_self_t, thumb_self_t = prev_self_touch

        # Build LSTM input
        x_t = torch.cat([
            idx_self_t,
            thumb_self_t,
            hand_jnt_pos_t
        ], dim=-1)

        # RNN step
        h_next, c_next = self.lstm_cell(x_t, hidden) if hidden is not None else self.lstm_cell(x_t)
        out = self.dec(h_next)

        # Split outputs
        jnt_pos_pred, jnt_cmd_pos_pred = torch.split(out, self.split_sizes, dim=-1)

        # Predict next self-touch from own model
        seq_args_next = [jnt_pos_pred, jnt_cmd_pos_pred]
        idx_self_next, thumb_self_next, _ = self.selftouch(*seq_args_next)


        return (
            jnt_pos_pred,
            jnt_cmd_pos_pred,
        ), (idx_self_next, thumb_self_next), (h_next, c_next)

    def forward_loss(
        self,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_cmd_pos: torch.Tensor,
        data_found: torch.Tensor,
        loss_coef: dict,
        cls_rate=None,
        noise=None
    ):
        B, T, _ = hand_jnt_pos.shape
        prev_self_touch = None
        hidden = None

        # Collect predictions
        pos_preds, cmd_preds = [], []

        for t in range(T-1):
            # Add noise
            pos_t = add_gausian_noise(hand_jnt_pos[:, t], noise)

            # Closed-loop correction
            if t != 0:
                pos_t = closed_loop(pos_t, pos_pred, cls_rate)

            # Forward step
            preds, (idx_self_next, thumb_self_next), hidden = self.forward(
                pos_t, prev_self_touch, hidden
            )
            pos_pred, cmd_pred = preds
            prev_self_touch = (idx_self_next, thumb_self_next)

            # Store
            pos_preds.append(pos_pred)
            cmd_preds.append(cmd_pred)

        # Stack
        pos_preds = torch.stack(pos_preds, dim=1)
        cmd_preds = torch.stack(cmd_preds, dim=1)

        # Losses
        loss_pos = mask_mse_loss(pos_preds, hand_jnt_pos[:, 1:], data_found[:, 1:])
        loss_cmd = mask_mse_loss(cmd_preds, hand_jnt_cmd_pos[:, 1:], data_found[:, 1:])

        total_loss = (
            loss_pos * loss_coef['hand_jnt_pos'] +
            loss_cmd * loss_coef['hand_jnt_cmd_pos']
        )

        return (total_loss, loss_pos, loss_cmd), (pos_preds, cmd_preds)

if __name__ == "__main__":
    print("ExternalTouchRNN without velocity and torque inputs")
