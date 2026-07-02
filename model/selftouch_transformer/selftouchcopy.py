import math
import torch
from torch import nn
from selftouch_loss_utils import active_tactile_loss, reconstruction_loss


class SelfTouchTransformer(nn.Module):

    def __init__(self, param):
        super().__init__()
        d_model      = param["d_model"]
        hand_dim     = param["hand_dim"]
        nhead        = param["nhead"]
        enc_layers   = param.get("encoder_layers", 4)
        dropout      = param.get("dropout", 0.1)
        max_len      = param.get("max_seq_len", 2048)
        self.causal  = param.get("causal", True)

        self.hand_dim = hand_dim
        self.d_model  = d_model

        # Modality embeddings (project each modality to d_model, then fuse by sum)
        self.embed_pos = nn.Linear(hand_dim, d_model)
        self.embed_vel = nn.Linear(hand_dim, d_model)
        self.embed_trq = nn.Linear(hand_dim, d_model)
        self.embed_cmd = nn.Linear(hand_dim, d_model)

        self.input_dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding for time dimension
        pe = self._build_sinusoidal_positional_encoding(max_len, d_model)  # (max_len, d_model)
        self.register_buffer("pos_encoding", pe, persistent=False)

        # Encoder over time (S=T, N=B, E=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=False, 
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        # Regression heads (linear; no activation for MSE)
        self.to_idx   = nn.Linear(d_model, 90)        # 30*3
        self.to_thumb = nn.Linear(d_model, 90)        # 30*3
        self.to_pos   = nn.Linear(d_model, hand_dim)
        self.to_vel   = nn.Linear(d_model, hand_dim)
        self.to_trq   = nn.Linear(d_model, hand_dim)

        self._reset_parameters()

    @staticmethod
    def _build_sinusoidal_positional_encoding(n_position, d_model):
        position = torch.arange(0, n_position, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(n_position, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (L, d_model)

    @staticmethod
    def _generate_causal_mask(T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hand_jnt_pos,      # (B,T,hand_dim)
        hand_jnt_vel,      # (B,T,hand_dim)
        hand_jnt_trq,      # (B,T,hand_dim)
        hand_jnt_cmd_pos,  # (B,T,hand_dim)
        src_key_padding_mask: torch.Tensor = None,  # (B,T) True for padding
    ):
        
        B, T, hd = hand_jnt_pos.shape
        assert hd == self.hand_dim, f"hand_dim mismatch: got {hd}, expected {self.hand_dim}"

        # Modality projection then fuse by summation
        x = (
            self.embed_pos(hand_jnt_pos)
            + self.embed_vel(hand_jnt_vel)
            + self.embed_trq(hand_jnt_trq)
            + self.embed_cmd(hand_jnt_cmd_pos)
        ) 

        # Add temporal positional encoding
        pe = self.pos_encoding[:T, :].unsqueeze(0)  
        x = x + pe  
        x = self.input_dropout(x)
        x = x.transpose(0, 1) 
        attn_mask = self._generate_causal_mask(T, x.device) if self.causal else None
        memory = self.encoder(
            x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask
        ) 

        y = memory.transpose(0, 1)  

        idx_pred   = self.to_idx(y)        # (B,T,90)
        thumb_pred = self.to_thumb(y)      # (B,T,90)
        pos_pred   = self.to_pos(y)        # (B,T,hand_dim)
        vel_pred   = self.to_vel(y)        # (B,T,hand_dim)
        trq_pred   = self.to_trq(y)        # (B,T,hand_dim)

        return idx_pred, thumb_pred, pos_pred, vel_pred, trq_pred


    def forward_loss(
        self,
        tactile_index_tip,   # (B,T,90)
        tactile_thumb_tip,   # (B,T,90)
        hand_jnt_pos,        # (B,T,hand_dim)
        hand_jnt_vel,        # (B,T,hand_dim)
        hand_jnt_trq,        # (B,T,hand_dim)
        hand_jnt_cmd_pos,    # (B,T,hand_dim)
        data_found=None,
        loss_coef=None,
        src_key_padding_mask: torch.Tensor = None,  # (B,T) True for padding
    ):
        idx_pred, thumb_pred, pos_pred, vel_pred, trq_pred = self.forward(
            hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos,
            src_key_padding_mask=src_key_padding_mask
        )
        loss_index = active_tactile_loss(idx_pred[:, :-1, :],   tactile_index_tip[:, 1:, :], loss_coef)
        loss_thumb = active_tactile_loss(thumb_pred[:, :-1, :], tactile_thumb_tip[:, 1:, :], loss_coef)
        loss_pos   = reconstruction_loss(pos_pred[:, :-1, :],   hand_jnt_pos[:, 1:, :], loss_coef)
        loss_vel   = reconstruction_loss(vel_pred[:, :-1, :],   hand_jnt_vel[:, 1:, :], loss_coef)
        loss_trq   = reconstruction_loss(trq_pred[:, :-1, :],   hand_jnt_trq[:, 1:, :], loss_coef)

        if loss_coef is None:
            total_loss = loss_index + loss_thumb + loss_pos + loss_vel + loss_trq
        else:
            total_loss = (
                loss_coef.get("tactile_index_tip", 1.0) * loss_index
                + loss_coef.get("tactile_thumb_tip", 1.0) * loss_thumb
                + loss_coef.get("hand_jnt_pos", 1.0)     * loss_pos
                + loss_coef.get("hand_jnt_vel", 1.0)     * loss_vel
                + loss_coef.get("hand_jnt_trq", 1.0)     * loss_trq
            )

        return (
            total_loss, loss_index, loss_thumb, loss_pos, loss_vel, loss_trq
        ), (
            idx_pred, thumb_pred, pos_pred, vel_pred, trq_pred
        )

if __name__ == "__main__":
    param = {
        "hand_dim": 21,
        "d_model": 256,
        "nhead": 8,
        "encoder_layers": 3,
        "dropout": 0.1,
        "max_seq_len": 512,
        "causal": True,
    }
    model = SelfTouchTransformer(param)

    B, T, hand_dim = 2, 5, param["hand_dim"]
    tactile_index_tip = torch.randn(B, T, 90)
    tactile_thumb_tip = torch.randn(B, T, 90)
    hand_jnt_pos      = torch.randn(B, T, hand_dim)
    hand_jnt_vel      = torch.randn(B, T, hand_dim)
    hand_jnt_trq      = torch.randn(B, T, hand_dim)
    hand_jnt_cmd_pos  = torch.randn(B, T, hand_dim)

    loss_vals, preds = model.forward_loss(
        tactile_index_tip,
        tactile_thumb_tip,
        hand_jnt_pos,
        hand_jnt_vel,
        hand_jnt_trq,
        hand_jnt_cmd_pos,
        loss_coef={
            "tactile_index_tip": 1.0,
            "tactile_thumb_tip": 1.0,
            "hand_jnt_pos": 1.0,
            "hand_jnt_vel": 1.0,
            "hand_jnt_trq": 1.0,
        }
    )
    print("Total loss:", loss_vals[0].item())
