import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn
import torch.nn.functional as F
import einops


class SelfTouch(nn.Module):
    def __init__(self,param):
        super().__init__()

        rec_dim=param["rec_dim"]
        self.hand_dim=param["hand_dim"]
        self.tactile_dim=param["tactile_dim"]
        dropout=param["dropout"]
        activation = nn.LeakyReLU()

        rec_in = self.hand_dim*4
        rec_out = self.hand_dim*3+self.tactile_dim*2


        self.enc=nn.Sequential(nn.Linear(rec_in,rec_dim//2),\
                               activation,\
                               nn.Dropout(p=dropout), 
                               nn.Linear(rec_dim//2,rec_dim),\
                               activation,\
                               )
        
        self.dec=nn.Sequential(nn.Dropout(p=dropout),
                               nn.Linear(rec_dim,rec_dim//2),\
                               activation,
                               nn.Linear(rec_dim//2,rec_out),\
                               activation)
        
        self.split_sizes = [
            self.tactile_dim,  # index_tip
            self.tactile_dim,  # thumb_tip
            self.hand_dim,     # jnt_pos
            self.hand_dim,     # jnt_vel
            self.hand_dim,     # jnt_trq
        ]

    

    def forward_loss(self,tactile_index_tip,\
                    tactile_thumb_tip,\
                    hand_jnt_pos,\
                    hand_jnt_vel,\
                    hand_jnt_trq,\
                    hand_jnt_cmd_pos,\
                    data_found,\
                    loss_coef):
        #  tactile_index_tip (b,t,d)
        #  tactile_thumb_tip (b,t,d)
        #  hand_jnt_pos (b,t,d)
        #  hand_jnt_vel (b,t,d)
        #  hand_jnt_trq (b,t,d)
        #  hand_jnt_cmd_pos (b,t,d)
        #  loss_coef {"modality": float}

        idx_pred, thumb_pred, pos_pred, vel_pred, trq_pred=\
        self.forward(hand_jnt_pos,\
                     hand_jnt_vel,\
                     hand_jnt_trq,\
                     hand_jnt_cmd_pos)
        
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
        loss_vel = F.mse_loss(
            vel_pred[:, :-1, :],
            hand_jnt_vel[:, 1:, :],
        )
        loss_trq = F.mse_loss(
            trq_pred[:, :-1, :],
            hand_jnt_trq[:, 1:, :],
        )

        total_loss=loss_index*loss_coef["tactile_index_tip"]+\
                    loss_thumb*loss_coef["tactile_thumb_tip"]+\
                     loss_pos*loss_coef["hand_jnt_pos"]+\
                      loss_vel*loss_coef["hand_jnt_vel"]+\
                       loss_trq*loss_coef["hand_jnt_trq"]

        return (total_loss,loss_index,loss_thumb,loss_pos,loss_vel,loss_trq), (idx_pred,thumb_pred,pos_pred,vel_pred,trq_pred)
    

    def forward(self,hand_jnt_pos,\
                     hand_jnt_vel,\
                     hand_jnt_trq,\
                     hand_jnt_cmd_pos):
        #  hand_jnt_pos (b,t,d)
        #  hand_jnt_vel (b,t,d)
        #  hand_jnt_trq (b,t,d)
        #  hand_jnt_cmd_pos (b,t,d)

        x = torch.cat([
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos
        ], dim=-1)
        
        h=self.enc(x)
        idx_pred, thumb_pred, pos_pred, vel_pred, trq_pred\
            =torch.split(self.dec(h), self.split_sizes, dim=-1)
        return idx_pred, thumb_pred, pos_pred, vel_pred, trq_pred


if __name__ == "__main__":
    print("Model for predicting self touch")