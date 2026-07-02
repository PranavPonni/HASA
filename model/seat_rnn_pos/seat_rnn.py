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

        rec_in = self.tactile_dim * 6 + self.hand_dim  # 6 tactile vectors + joint
        rec_out = self.tactile_dim * 6 + self.hand_dim * 2  # 6 tactile + jnt pos/cmd
        self.split_sizes = [
            self.tactile_dim,  # index_self
            self.tactile_dim,  # thumb_self
            self.tactile_dim,  # index_ext
            self.tactile_dim,  # thumb_ext
            self.tactile_dim,  # index_total
            self.tactile_dim,  # thumb_total
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
        tactile_index_self_t,
        tactile_thumb_self_t,
        tactile_index_ext_t,
        tactile_thumb_ext_t,
        tactile_index_total_t,
        tactile_thumb_total_t,
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
            tactile_index_self_t,
            tactile_thumb_self_t,
            tactile_index_ext_t,
            tactile_thumb_ext_t,
            tactile_index_total_t,
            tactile_thumb_total_t,
            hand_jnt_pos_t
        ], dim=-1)

        # RNN step
        h_next, c_next = self.lstm_cell(x_t, hidden) if hidden is not None else self.lstm_cell(x_t)
        out = self.dec(h_next)

        # Split outputs
        idx_self_pred, thumb_self_pred, idx_ext_pred, thumb_ext_pred, idx_total_pred, thumb_total_pred, \
        jnt_pos_pred, jnt_cmd_pos_pred = torch.split(out, self.split_sizes, dim=-1)

        # Predict next self-touch from own model
        seq_args_next = [jnt_pos_pred, jnt_cmd_pos_pred]
        idx_self_next, thumb_self_next, _ = self.selftouch(*seq_args_next)

        return (
            jnt_pos_pred,
            jnt_cmd_pos_pred,
            idx_self_pred,
            thumb_self_pred,
            idx_ext_pred,
            thumb_ext_pred,
            idx_total_pred,
            thumb_total_pred,
        ), (h_next, c_next)

    def forward_loss(
        self,
        tactile_index_self: torch.Tensor,
        tactile_thumb_self: torch.Tensor,
        tactile_index_ext: torch.Tensor,
        tactile_thumb_ext: torch.Tensor,
        tactile_index_total: torch.Tensor,
        tactile_thumb_total: torch.Tensor,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_cmd_pos: torch.Tensor,
        data_found: torch.Tensor,
        loss_coef: dict,
        cls_rate=None,
        noise=None
    ):
        B, T, _ = tactile_index_self.shape
        hidden = None

        # Collect predictions
        idx_self_preds, thumb_self_preds = [], []
        idx_ext_preds, thumb_ext_preds = [], []
        idx_total_preds, thumb_total_preds = [], []
        pos_preds, cmd_preds = [], []

        for t in range(T-1):
            # Noise added
            idx_self_t = add_gausian_noise(tactile_index_self[:, t], noise["tactile_noise"])
            thumb_self_t = add_gausian_noise(tactile_thumb_self[:, t], noise["tactile_noise"])
            idx_ext_t = add_gausian_noise(tactile_index_ext[:, t], noise["tactile_noise"])
            thumb_ext_t = add_gausian_noise(tactile_thumb_ext[:, t], noise["tactile_noise"])
            idx_total_t = add_gausian_noise(tactile_index_total[:, t], noise["tactile_noise"])
            thumb_total_t = add_gausian_noise(tactile_thumb_total[:, t], noise["tactile_noise"])
            pos_t = add_gausian_noise(hand_jnt_pos[:, t], noise["joint_noise"])

            # Closed-loop correction
            if t != 0:
                pos_t = closed_loop(pos_t, pos_pred, cls_rate["joint_rate"])
                idx_self_t = closed_loop(idx_self_t, idx_self_pred, cls_rate["tactile_rate"])
                thumb_self_t = closed_loop(thumb_self_t, thumb_self_pred, cls_rate["tactile_rate"])
                idx_ext_t = closed_loop(idx_ext_t, idx_ext_pred, cls_rate["tactile_rate"])

            # Forward step
            preds, hidden = self.forward(
                idx_self_t, thumb_self_t,
                idx_ext_t, thumb_ext_t,
                idx_total_t, thumb_total_t,
                pos_t,
                hidden
            )
            pos_pred, cmd_pred, idx_self_pred, thumb_self_pred, \
            idx_ext_pred, thumb_ext_pred, idx_total_pred, thumb_total_pred = preds

            # Store predictions
            idx_self_preds.append(idx_self_pred)
            thumb_self_preds.append(thumb_self_pred)
            idx_ext_preds.append(idx_ext_pred)
            thumb_ext_preds.append(thumb_ext_pred)
            idx_total_preds.append(idx_total_pred)
            thumb_total_preds.append(thumb_total_pred)
            pos_preds.append(pos_pred)
            cmd_preds.append(cmd_pred)

        # Stack over time
        idx_self_preds = torch.stack(idx_self_preds, dim=1)
        thumb_self_preds = torch.stack(thumb_self_preds, dim=1)
        idx_ext_preds = torch.stack(idx_ext_preds, dim=1)
        thumb_ext_preds = torch.stack(thumb_ext_preds, dim=1)
        idx_total_preds = torch.stack(idx_total_preds, dim=1)
        thumb_total_preds = torch.stack(thumb_total_preds, dim=1)
        pos_preds = torch.stack(pos_preds, dim=1)
        cmd_preds = torch.stack(cmd_preds, dim=1)

        # Compute losses
        loss_idx_self = mask_mse_loss(idx_self_preds, tactile_index_self[:, 1:], data_found[:, 1:])
        loss_thumb_self = mask_mse_loss(thumb_self_preds, tactile_thumb_self[:, 1:], data_found[:, 1:])
        loss_idx_ext = mask_mse_loss(idx_ext_preds, tactile_index_ext[:, 1:], data_found[:, 1:])
        loss_thumb_ext = mask_mse_loss(thumb_ext_preds, tactile_thumb_ext[:, 1:], data_found[:, 1:])
        loss_idx_total = mask_mse_loss(idx_total_preds, tactile_index_total[:, 1:], data_found[:, 1:])
        loss_thumb_total = mask_mse_loss(thumb_total_preds, tactile_thumb_total[:, 1:], data_found[:, 1:])
        loss_pos = mask_mse_loss(pos_preds, hand_jnt_pos[:, 1:], data_found[:, 1:])
        loss_cmd = mask_mse_loss(cmd_preds, hand_jnt_cmd_pos[:, 1:], data_found[:, 1:])

        # Total loss
        total_loss = (
            loss_idx_self * loss_coef['tactile_index_self'] +
            loss_thumb_self * loss_coef['tactile_thumb_self'] +
            loss_idx_ext * loss_coef['tactile_index_ext'] +
            loss_thumb_ext * loss_coef['tactile_thumb_ext'] +
            loss_idx_total * loss_coef['tactile_index_total'] +
            loss_thumb_total * loss_coef['tactile_thumb_total'] +
            loss_pos * loss_coef['hand_jnt_pos'] +
            loss_cmd * loss_coef['hand_jnt_cmd_pos']
        )

        return (
            total_loss,
            loss_idx_self,
            loss_thumb_self,
            loss_idx_ext,
            loss_thumb_ext,
            loss_idx_total,
            loss_thumb_total,
            loss_pos,
            loss_cmd
        ), (
            idx_self_preds,
            thumb_self_preds,
            idx_ext_preds,
            thumb_ext_preds,
            idx_total_preds,
            thumb_total_preds,
            pos_preds,
            cmd_preds
        )

if __name__ == "__main__":
    print("ExternalTouchRNN model with self, external, and total tactile as input/output.")
