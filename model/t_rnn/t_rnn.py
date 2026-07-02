import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
import torch.nn.functional as F
from util import mask_mse_loss,add_gausian_noise,closed_loop
import pdb

class ExternalTouchRNN(nn.Module):
    def __init__(self, param: dict):
        super().__init__()
        hid_dim = param["hid_dim"]
        self.hand_dim = param["hand_dim"]
        self.tactile_dim = param["tactile_dim"]
        activation = nn.LeakyReLU()
        rec_in = 2 * self.tactile_dim + 3 * self.hand_dim
        rec_out = 2 * self.tactile_dim + 4 * self.hand_dim
        self.split_sizes = [
            self.tactile_dim,
            self.tactile_dim,
            self.hand_dim,
            self.hand_dim,
            self.hand_dim,
            self.hand_dim,
        ]
        self.lstm_cell = nn.LSTMCell(rec_in, hid_dim)
        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            activation,
            nn.Linear(hid_dim // 2, rec_out),
        )


    def forward(
        self,
        tactile_index_tip_t,
        tactile_thumb_tip_t,
        hand_jnt_pos_t,
        hand_jnt_vel_t,
        hand_jnt_trq_t,
        hidden=None,
    ):
        x_t = torch.cat([
            tactile_index_tip_t,
            tactile_thumb_tip_t,
            hand_jnt_pos_t,
            hand_jnt_vel_t,
            hand_jnt_trq_t,
        ], dim=-1)
        h_next, c_next = self.lstm_cell(x_t, hidden)
        out = self.dec(h_next)
        (
            idx_tactile_pred,
            thumb_tactile_pred,
            jnt_pos_pred,
            jnt_vel_pred,
            jnt_trq_pred,
            jnt_cmd_pos_pred
        ) = torch.split(out, self.split_sizes, dim=-1)
        return (
            idx_tactile_pred,
            thumb_tactile_pred,
            jnt_pos_pred,
            jnt_vel_pred,
            jnt_trq_pred,
            jnt_cmd_pos_pred,
        ), (h_next, c_next)

    def forward_loss(
        self,
        tactile_index_tip: torch.Tensor,
        tactile_thumb_tip: torch.Tensor,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_vel: torch.Tensor,
        hand_jnt_trq: torch.Tensor,
        hand_jnt_cmd_pos: torch.Tensor,
        data_found: torch.Tensor,
        loss_coef: dict,
        cls_rate=0.5,
        noise=0.1
    ):
        B, T, _ = tactile_index_tip.shape
        hidden = None
        # collect predictions per time step
        idx_preds = []
        thumb_preds = []
        pos_preds = []
        vel_preds = []
        trq_preds = []
        cmd_preds = []

        for t in range(T-1):
            tin_t = add_gausian_noise(tactile_index_tip[:, t],noise)
            tth_t = add_gausian_noise(tactile_thumb_tip[:, t],noise)
            pos_t = add_gausian_noise(hand_jnt_pos[:, t],noise)
            vel_t = add_gausian_noise(hand_jnt_vel[:, t],noise)
            trq_t = add_gausian_noise(hand_jnt_trq[:, t],noise)

            if t!=0:
                pos_t=closed_loop(pos_t,pos_pred,cls_rate)
                vel_t=closed_loop(vel_t,vel_pred,cls_rate)
                trq_t=closed_loop(trq_t,trq_pred,cls_rate)
                tin_t=closed_loop(tin_t,total_idx_pred,cls_rate)
                tth_t=closed_loop(tth_t,total_thumb_pred,cls_rate)


            preds, hidden = self.forward(
                tin_t, tth_t, pos_t, vel_t, trq_t, hidden
            )
            (total_idx_pred, total_thumb_pred,pos_pred, vel_pred, trq_pred, cmd_pred) = preds

            idx_preds.append(total_idx_pred)
            thumb_preds.append(total_thumb_pred)
            pos_preds.append(pos_pred)
            vel_preds.append(vel_pred)
            trq_preds.append(trq_pred)
            cmd_preds.append(cmd_pred)

        # stack predictions: shape (B, steps, D)
        idx_preds = torch.stack(idx_preds, dim=1)
        thumb_preds = torch.stack(thumb_preds, dim=1)
        pos_preds = torch.stack(pos_preds, dim=1)
        vel_preds = torch.stack(vel_preds, dim=1)
        trq_preds = torch.stack(trq_preds, dim=1)
        cmd_preds = torch.stack(cmd_preds, dim=1)

        # compute losses directly with slicing
        loss_tactile_idx = mask_mse_loss(idx_preds, tactile_index_tip[:, 1:],data_found[:,1:])
        loss_tactile_thumb = mask_mse_loss(thumb_preds, tactile_thumb_tip[:, 1:],data_found[:,1:])
        loss_pos = mask_mse_loss(pos_preds, hand_jnt_pos[:, 1:],data_found[:,1:])
        loss_vel = mask_mse_loss(vel_preds, hand_jnt_vel[:, 1:],data_found[:,1:])
        loss_trq = mask_mse_loss(trq_preds, hand_jnt_trq[:, 1:],data_found[:,1:])
        loss_cmd = mask_mse_loss(cmd_preds, hand_jnt_cmd_pos[:, 1:],data_found[:,1:])

        total_loss = (
            loss_tactile_idx * loss_coef['tactile_index_tip'] +
            loss_tactile_thumb * loss_coef['tactile_thumb_tip'] +
            loss_pos * loss_coef['hand_jnt_pos'] +
            loss_vel * loss_coef['hand_jnt_vel'] +
            loss_trq * loss_coef['hand_jnt_trq'] +
            loss_cmd * loss_coef['hand_jnt_cmd_pos']
        )
        return (total_loss,loss_tactile_idx,loss_tactile_thumb,loss_pos,loss_vel,loss_trq,loss_cmd),\
               (idx_preds,thumb_preds,pos_preds,vel_preds,trq_preds,cmd_preds,)
                
            

if __name__ == "__main__":
    print("ExternalTouchRNN with forward_loss initialized")
