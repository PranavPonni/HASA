import torch
from torch import nn

from selftouch_loss_utils import active_tactile_loss, finger_loss_config
from selftouch_feature_utils import TactileHead, build_joint_features, joint_feature_dim


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.left_pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.functional.pad(x, (self.left_pad, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class TemporalMixerBlock(nn.Module):
    def __init__(self, d_model, kernel_size, dilation, dropout):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal = CausalDepthwiseConv1d(d_model, kernel_size, dilation)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x):
        temporal = self.temporal(self.temporal_norm(x))
        x = x + self.gate(x) * temporal
        x = x + self.channel(self.channel_norm(x))
        return x


class SelfTouchTemporalMixer(nn.Module):
    """Causal temporal convolution mixer with a lightweight attention refinement."""

    def __init__(self, param):
        super().__init__()
        hand_dim = int(param["hand_dim"])
        tactile_dim = int(param.get("tactile_dim", 90))
        d_model = int(param.get("d_model", 256))
        depth = int(param.get("mixer_layers", 6))
        kernel_size = int(param.get("kernel_size", 7))
        nhead = int(param.get("nhead", 8))
        dropout = float(param.get("dropout", 0.1))
        self.causal = bool(param.get("causal", True))

        self.hand_dim = hand_dim
        self.tactile_dim = tactile_dim
        self.d_model = d_model
        self.input_dim = int(param.get("input_dim", hand_dim * 4))
        self.use_joint_pos_only = bool(param.get("use_joint_pos_only", True))
        self.use_derived_features = bool(param.get("use_derived_features", False))
        self.temporal_window_steps = max(1, int(param.get("temporal_window_steps", 1)))
        self.use_tactile_history = bool(param.get("use_tactile_history", False))
        self.tactile_history_steps = max(1, int(param.get("tactile_history_steps", 1)))
        self.tactile_history_fingers = tuple(
            param.get("tactile_history_fingers", ("index", "thumb", "middle"))
        )
        self.input_dim = (
            joint_feature_dim(
                hand_dim,
                self.use_derived_features,
                self.use_joint_pos_only,
                temporal_window_steps=self.temporal_window_steps,
                tactile_dim=self.tactile_dim,
                use_tactile_history=self.use_tactile_history,
                tactile_history_steps=self.tactile_history_steps,
                tactile_history_fingers=self.tactile_history_fingers,
            )
            if self.use_joint_pos_only
            or self.use_derived_features
            or self.use_tactile_history
            or self.temporal_window_steps > 1
            else self.input_dim
        )

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                TemporalMixerBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                    dilation=2 ** (idx % 4),
                    dropout=dropout,
                )
                for idx in range(depth)
            ]
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d_model)

        # Index, thumb, and middle tactile outputs. Ring is intentionally plot-only
        # because the current dataset does not include ring tactile data.
        self.to_idx = TactileHead(d_model, self.input_dim, tactile_dim)
        self.to_thumb = TactileHead(d_model, self.input_dim, tactile_dim)
        self.to_middle = TactileHead(d_model, self.input_dim, tactile_dim)

        self._reset_parameters()
        self._init_tactile_bias()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_tactile_bias(self):
        return

    def _activate_output(self, x):
        return x

    @staticmethod
    def _causal_mask(timesteps, device):
        return torch.triu(
            torch.ones(timesteps, timesteps, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _build_input(
        self,
        hand_jnt_pos,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        hand_jnt_cmd_pos=None,
        tactile_index_tip=None,
        tactile_thumb_tip=None,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
    ):
        return build_joint_features(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            next_step=True,
            use_derived_features=self.use_derived_features,
            use_joint_pos_only=self.use_joint_pos_only,
            temporal_window_steps=self.temporal_window_steps,
            use_tactile_history=self.use_tactile_history,
            tactile_history_steps=self.tactile_history_steps,
            tactile_history_fingers=self.tactile_history_fingers,
            tactile_dim=self.tactile_dim,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
        )

    def encode(self, hand_jnt_pos, hand_jnt_vel=None, hand_jnt_trq=None, hand_jnt_cmd_pos=None, src_key_padding_mask=None):
        """Encode joint state at t, returning (B, T-1, d_model)."""
        x = self._build_input(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        attn_in = self.attn_norm(h)
        mask = self._causal_mask(h.shape[1], h.device) if self.causal else None
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            attn_mask=mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        return self.out_norm(h + attn_out)

    def forward(
        self,
        hand_jnt_pos,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        hand_jnt_cmd_pos=None,
        src_key_padding_mask=None,
        tactile_index_tip=None,
        tactile_thumb_tip=None,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
    ):
        x = self._build_input(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
        )
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        attn_in = self.attn_norm(h)
        mask = self._causal_mask(h.shape[1], h.device) if self.causal else None
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            attn_mask=mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        y = self.out_norm(h + attn_out)
        return (
            self._activate_output(self.to_idx(y, x)),
            self._activate_output(self.to_thumb(y, x)),
            self._activate_output(self.to_middle(y, x)),
        )

    def forward_loss(self, tactile_index_tip, tactile_thumb_tip,
                     hand_jnt_pos, hand_jnt_cmd_pos=None, hand_jnt_vel=None, hand_jnt_trq=None,
                     data_found=None, loss_coef=None,
                     src_key_padding_mask=None, tactile_middle_tip=None,
                     tactile_ring_tip=None, **_unused):
        idx_pred, thumb_pred, middle_pred = self.forward(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            src_key_padding_mask,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
        )

        loss_index = active_tactile_loss(
            idx_pred,
            tactile_index_tip[:, 1:, :],
            finger_loss_config(loss_coef, "tactile_index_tip"),
        )
        loss_thumb = active_tactile_loss(
            thumb_pred,
            tactile_thumb_tip[:, 1:, :],
            finger_loss_config(loss_coef, "tactile_thumb_tip"),
        )
        loss_middle = active_tactile_loss(
            middle_pred,
            tactile_middle_tip[:, 1:, :],
            finger_loss_config(loss_coef, "tactile_middle_tip"),
        )

        coef = loss_coef or {}
        total_loss = (
            coef.get("tactile_index_tip", 1.0) * loss_index
            + coef.get("tactile_thumb_tip", 1.0) * loss_thumb
            + coef.get("tactile_middle_tip", 1.0) * loss_middle
        )
        return (total_loss, loss_index, loss_thumb, loss_middle), (idx_pred, thumb_pred, middle_pred)


SelfTouchTransformer = SelfTouchTemporalMixer
