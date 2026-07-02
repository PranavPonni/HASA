import math
import torch
from torch import nn
from selftouch_loss_utils import active_tactile_loss
from selftouch_feature_utils import TactileHead, build_joint_features, joint_feature_dim


class SelfTouchTransformer(nn.Module):
    """Causal Transformer for selftouch prediction.

    Input:  Joint position(t)  [hand_dim]
    Output: Predicted Self-touch Tactile(t) for index_tip, thumb_tip, and middle_tip [tactile_dim each]
    """

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
        self.tactile_dim = param.get("tactile_dim", 90)
        self.use_joint_pos_only = bool(param.get("use_joint_pos_only", True))
        self.use_derived_features = bool(param.get("use_derived_features", False))

        self.input_dim = (
            joint_feature_dim(hand_dim, self.use_derived_features, self.use_joint_pos_only)
            if self.use_joint_pos_only or self.use_derived_features
            else int(param.get("input_dim", hand_dim * 4))
        )
        self.embed_input = nn.Linear(self.input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        pe = self._build_sinusoidal_positional_encoding(max_len, d_model)
        self.register_buffer("pos_encoding", pe, persistent=False)

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

        # Index, thumb, and middle tactile outputs. Ring is intentionally plot-only
        # because the current dataset does not include ring tactile data.
        self.to_idx = TactileHead(d_model, self.input_dim, self.tactile_dim)
        self.to_thumb = TactileHead(d_model, self.input_dim, self.tactile_dim)
        self.to_middle = TactileHead(d_model, self.input_dim, self.tactile_dim)

        self._reset_parameters()
        self._init_tactile_bias()

    @staticmethod
    def _build_sinusoidal_positional_encoding(n_position, d_model):
        position = torch.arange(0, n_position, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(n_position, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def _generate_causal_mask(T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_tactile_bias(self):
        return

    def _activate_output(self, x):
        return x

    def _build_input(self, hand_jnt_pos, hand_jnt_vel=None, hand_jnt_trq=None, hand_jnt_cmd_pos=None):
        return build_joint_features(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            next_step=True,
            use_derived_features=self.use_derived_features,
            use_joint_pos_only=self.use_joint_pos_only,
        )

    def encode(self, hand_jnt_pos, hand_jnt_vel=None, hand_jnt_trq=None, hand_jnt_cmd_pos=None, src_key_padding_mask=None):
        """Encode joint state at t, returning (B, T-1, d_model)."""
        x_raw = self._build_input(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)
        B, T, _ = x_raw.shape
        x = self.embed_input(x_raw)
        pe = self.pos_encoding[:T, :].unsqueeze(0)
        x = self.input_dropout(x + pe)
        x = x.transpose(0, 1)  # (T, B, d_model)
        attn_mask = self._generate_causal_mask(T, x.device) if self.causal else None
        memory = self.encoder(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        return memory.transpose(0, 1)  # (B, T-1, d_model)

    def forward(self, hand_jnt_pos, hand_jnt_vel=None, hand_jnt_trq=None, hand_jnt_cmd_pos=None, src_key_padding_mask=None):
        x_raw = self._build_input(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)
        B, T, _ = x_raw.shape
        x = self.embed_input(x_raw)
        pe = self.pos_encoding[:T, :].unsqueeze(0)
        x = self.input_dropout(x + pe)
        x = x.transpose(0, 1)
        attn_mask = self._generate_causal_mask(T, x.device) if self.causal else None
        y = self.encoder(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask).transpose(0, 1)
        idx_pred = self._activate_output(self.to_idx(y, x_raw))
        thumb_pred = self._activate_output(self.to_thumb(y, x_raw))
        middle_pred = self._activate_output(self.to_middle(y, x_raw))
        return idx_pred, thumb_pred, middle_pred

    def forward_loss(self, tactile_index_tip, tactile_thumb_tip,
                     hand_jnt_pos, hand_jnt_cmd_pos=None, hand_jnt_vel=None, hand_jnt_trq=None,
                     data_found=None, loss_coef=None,
                     src_key_padding_mask=None, tactile_middle_tip=None,
                     tactile_ring_tip=None, **_unused):
        idx_pred, thumb_pred, middle_pred = self.forward(
            hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos, src_key_padding_mask
        )

        # Targets: tactile at t+1, predicted from joint state at t.
        loss_index = active_tactile_loss(idx_pred, tactile_index_tip[:, 1:, :], loss_coef)
        loss_thumb = active_tactile_loss(thumb_pred, tactile_thumb_tip[:, 1:, :], loss_coef)
        loss_middle = active_tactile_loss(middle_pred, tactile_middle_tip[:, 1:, :], loss_coef)

        total_loss = (
            (loss_coef or {}).get("tactile_index_tip", 1.0) * loss_index +
            (loss_coef or {}).get("tactile_thumb_tip", 1.0) * loss_thumb +
            (loss_coef or {}).get("tactile_middle_tip", 1.0) * loss_middle
        )
        return (total_loss, loss_index, loss_thumb, loss_middle), (idx_pred, thumb_pred, middle_pred)
