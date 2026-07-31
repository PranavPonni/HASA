import math

import torch
from torch import nn
import torch.nn.functional as F

from selftouch_feature_utils import (
    build_tactile_history,
    build_joint_features,
    joint_feature_dim,
    labels_from_selftouch_combo,
)
from selftouch_loss_utils import (
    active_tactile_loss,
    finger_loss_config,
    select_valid_finger_targets,
    supervised_contrastive_loss,
)


FINGER_LOSS_KEYS = (
    "tactile_index_tip",
    "tactile_thumb_tip",
    "tactile_middle_tip",
    "tactile_ring_tip",
)
FINGER_NAMES = ("index", "thumb", "middle", "ring")
JOINT_FINGER_NAMES = ("index", "thumb", "middle", "ring")
COMBO_CODES = ("TI", "TM", "IM", "MR", "IMR", "TIM")
COMBO_FINGER_ORDER = (
    ("thumb", "index"),
    ("thumb", "middle"),
    ("index", "middle"),
    ("middle", "ring"),
    ("index", "middle", "ring"),
    ("thumb", "index", "middle"),
)
FINGER_TO_OUTPUT_INDEX = {name: idx for idx, name in enumerate(FINGER_NAMES)}


def _as_tuple(value, default=()):
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _positive_int(value, default=1):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = int(default)
    return max(1, value)


def _sinusoidal_encoding(length, dim):
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


def _causal_mask(timesteps, device):
    return torch.triu(
        torch.ones(timesteps, timesteps, device=device, dtype=torch.bool),
        diagonal=1,
    )


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(d_model)))

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class FingerAwareInputEncoder(nn.Module):
    """Encode 16 position + 16 torque values as four finger-local feature slots."""

    def __init__(self, hand_dim=16, finger_feature_dim=32):
        super().__init__()
        self.hand_dim = int(hand_dim)
        if self.hand_dim % 4 != 0:
            raise ValueError("finger-aware self-touch input expects hand_dim divisible by 4")
        self.finger_joint_dim = self.hand_dim // 4
        self.finger_feature_dim = int(finger_feature_dim)
        pos_dim = max(1, self.finger_feature_dim // 2)
        trq_dim = max(1, self.finger_feature_dim - pos_dim)
        self.pos_encoder = nn.Linear(self.finger_joint_dim, pos_dim)
        self.trq_encoder = nn.Linear(self.finger_joint_dim, trq_dim)
        self.finger_embedding = nn.Parameter(torch.zeros(4, self.finger_feature_dim))
        self.output_dim = 4 * self.finger_feature_dim

    def _fit_hand_width(self, x, reference):
        if x is None:
            return reference.new_zeros(reference.shape[0], reference.shape[1], self.hand_dim)
        if x.shape[-1] == self.hand_dim:
            return x
        if x.shape[-1] > self.hand_dim:
            return x[..., : self.hand_dim]
        return F.pad(x, (0, self.hand_dim - x.shape[-1]))

    def _fingerwise(self, x):
        x = x.contiguous().reshape(
            x.shape[0],
            x.shape[1],
            4,
            self.finger_joint_dim,
        )
        return x

    def forward(self, hand_jnt_pos, hand_jnt_trq=None, next_step=True):
        if hand_jnt_pos is None:
            raise KeyError("hand_jnt_pos is required for finger-aware self-touch input")
        stop = max(0, hand_jnt_pos.shape[1] - 1) if next_step else hand_jnt_pos.shape[1]
        pos = hand_jnt_pos[:, :stop, :]
        trq = hand_jnt_trq[:, :stop, :] if hand_jnt_trq is not None else None
        pos = self._fit_hand_width(pos, pos)
        trq = self._fit_hand_width(trq, pos)
        steps = min(pos.shape[1], trq.shape[1])
        pos = self._fingerwise(pos[:, :steps, :])
        trq = self._fingerwise(trq[:, :steps, :])
        encoded = torch.cat([self.pos_encoder(pos), self.trq_encoder(trq)], dim=-1)
        encoded = encoded + self.finger_embedding.view(1, 1, 4, -1)
        return encoded.reshape(encoded.shape[0], encoded.shape[1], self.output_dim)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_pad = int(dilation) * (int(kernel_size) - 1)
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=int(kernel_size),
                dilation=int(dilation),
            )
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.left_pad, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class TCNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TCNBackbone(nn.Module):
    def __init__(self, d_model, param):
        super().__init__()
        depth = _positive_int(param.get("temporal_blocks", 6), 6)
        kernel_size = _positive_int(param.get("temporal_kernel_size", 9), 9)
        dilations = param.get("temporal_dilations")
        if not dilations:
            dilations = [2 ** (idx % 5) for idx in range(depth)]
        dilations = list(dilations)
        while len(dilations) < depth:
            dilations.extend(dilations)
        dropout = float(param.get("dropout", 0.1))
        self.blocks = nn.Sequential(
            *[
                TCNResidualBlock(d_model, kernel_size, int(dilations[idx]), dropout)
                for idx in range(depth)
            ]
        )

    def forward(self, x, src_key_padding_mask=None):
        return self.blocks(x)


class PatchTransformerBackbone(nn.Module):
    """PatchTST-style temporal patch encoder with per-timestep reconstruction."""

    def __init__(self, d_model, param):
        super().__init__()
        self.patch_len = _positive_int(param.get("patch_len", 16), 16)
        self.patch_stride = _positive_int(param.get("patch_stride", 4), 4)
        nhead = int(param.get("nhead", 4))
        layers = _positive_int(param.get("encoder_layers", 3), 3)
        dropout = float(param.get("dropout", 0.1))
        max_seq_len = _positive_int(param.get("max_seq_len", 512), 512)
        ffn_dim = int(param.get("dim_feedforward", param.get("ffn_dim", 4 * d_model)))
        self.causal = _as_bool(param.get("causal", True))
        max_tokens = (max_seq_len + self.patch_stride - 1) // self.patch_stride + 2

        self.patch_embed = nn.Sequential(
            nn.Linear(d_model * self.patch_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "pos_encoding", _sinusoidal_encoding(max_tokens, d_model), persistent=False
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def _patches(self, x):
        padded = F.pad(x.transpose(1, 2), (self.patch_len - 1, 0)).transpose(1, 2)
        patches = padded.unfold(1, self.patch_len, self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.reshape(x.shape[0], patches.shape[1], self.patch_len * x.shape[-1])

    def forward(self, x, src_key_padding_mask=None):
        timesteps = x.shape[1]
        if timesteps == 0:
            return x
        tokens = self.patch_embed(self._patches(x))
        tokens = tokens + self.pos_encoding[: tokens.shape[1]].unsqueeze(0)
        mask = _causal_mask(tokens.shape[1], tokens.device) if self.causal else None
        tokens = self.encoder(tokens, mask=mask)
        tokens = self.norm(tokens)
        expanded = tokens.repeat_interleave(self.patch_stride, dim=1)
        if expanded.shape[1] < timesteps:
            pad = expanded[:, -1:, :].expand(-1, timesteps - expanded.shape[1], -1)
            expanded = torch.cat([expanded, pad], dim=1)
        return expanded[:, :timesteps, :]


class GRUAttentionBackbone(nn.Module):
    def __init__(self, d_model, param):
        super().__init__()
        layers = _positive_int(param.get("gru_layers", param.get("encoder_layers", 2)), 2)
        dropout = float(param.get("dropout", 0.1))
        self.bidirectional = _as_bool(param.get("bidirectional_gru", True))
        hidden = d_model // 2 if self.bidirectional else d_model
        gru_out = hidden * (2 if self.bidirectional else 1)
        nhead = int(param.get("nhead", param.get("attention_heads", 4)))
        self.causal_attention = _as_bool(param.get("causal", False)) and not self.bidirectional

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(gru_out, d_model) if gru_out != d_model else nn.Identity()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        ffn_dim = int(
            param.get(
                "ffn_dim",
                round(d_model * float(param.get("ffn_expansion", 4.0))),
            )
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.pooler = AttentionPool(d_model)

    def forward(self, x, src_key_padding_mask=None):
        h, _ = self.gru(x)
        h = self.out_proj(h)
        attn_in = self.attn_norm(h)
        mask = _causal_mask(h.shape[1], h.device) if self.causal_attention else None
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            attn_mask=mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        h = h + attn_out
        h = h + self.ffn(self.ffn_norm(h))
        return h

    def pool(self, x):
        return self.pooler(x)


class TSMixerBlock(nn.Module):
    def __init__(self, seq_len, d_model, dropout, time_expansion, channel_expansion):
        super().__init__()
        seq_hidden = max(seq_len, int(seq_len * time_expansion))
        feature_hidden = max(d_model, int(d_model * channel_expansion))
        self.seq_len = int(seq_len)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.seq_len, seq_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_hidden, self.seq_len),
            nn.Dropout(dropout),
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, feature_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden, d_model),
            nn.Dropout(dropout),
        )

    def _fit_time(self, x):
        if x.shape[-1] == self.seq_len:
            return x
        if x.shape[-1] < self.seq_len:
            return F.pad(x, (0, self.seq_len - x.shape[-1]))
        return x[..., : self.seq_len]

    def forward(self, x):
        timesteps = x.shape[1]
        y = self.time_norm(x).transpose(1, 2)
        y = self.time_mlp(self._fit_time(y))[..., :timesteps].transpose(1, 2)
        x = x + y
        x = x + self.feature_mlp(self.feature_norm(x))
        return x


class TSMixerBackbone(nn.Module):
    def __init__(self, d_model, param):
        super().__init__()
        sequence_length = _positive_int(param.get("sequence_length", 400), 400)
        seq_len = _positive_int(param.get("mixer_seq_len", sequence_length - 1), sequence_length - 1)
        depth = _positive_int(param.get("mixer_layers", param.get("encoder_layers", 6)), 6)
        dropout = float(param.get("dropout", 0.1))
        time_expansion = float(param.get("time_mixing_expansion", 2.0))
        channel_expansion = float(param.get("channel_mixing_expansion", 2.0))
        self.use_patches = _as_bool(param.get("mixer_use_patches", True))
        self.patch_len = _positive_int(param.get("patch_len", 4), 4)
        self.patch_stride = _positive_int(param.get("patch_stride", 2), 2)
        token_seq_len = (
            (seq_len + self.patch_stride - 1) // self.patch_stride + 2
            if self.use_patches
            else seq_len
        )
        if self.use_patches:
            self.patch_embed = nn.Sequential(
                nn.Linear(d_model * self.patch_len, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.blocks = nn.Sequential(
            *[
                TSMixerBlock(
                    token_seq_len,
                    d_model,
                    dropout,
                    time_expansion,
                    channel_expansion,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def _patches(self, x):
        padded = F.pad(x.transpose(1, 2), (self.patch_len - 1, 0)).transpose(1, 2)
        patches = padded.unfold(1, self.patch_len, self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.reshape(x.shape[0], patches.shape[1], self.patch_len * x.shape[-1])

    def forward(self, x, src_key_padding_mask=None):
        timesteps = x.shape[1]
        if timesteps == 0:
            return x
        if not self.use_patches:
            return self.norm(self.blocks(x))
        tokens = self.patch_embed(self._patches(x))
        tokens = self.norm(self.blocks(tokens))
        expanded = tokens.repeat_interleave(self.patch_stride, dim=1)
        if expanded.shape[1] < timesteps:
            pad = expanded[:, -1:, :].expand(-1, timesteps - expanded.shape[1], -1)
            expanded = torch.cat([expanded, pad], dim=1)
        return expanded[:, :timesteps, :]


class SelectiveSSMBlock(nn.Module):
    """Small Mamba-inspired selective state-space block implemented in PyTorch."""

    def __init__(self, d_model, param):
        super().__init__()
        expand = _positive_int(param.get("mamba_expand", 2), 2)
        self.inner = int(d_model) * expand
        self.d_state = _positive_int(param.get("mamba_d_state", 16), 16)
        kernel = _positive_int(param.get("mamba_conv_kernel", 4), 4)
        dropout = float(param.get("dropout", 0.1))
        self.fast_scan = _as_bool(param.get("mamba_fast_scan", True))

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.inner * 2)
        self.conv = nn.Conv1d(
            self.inner,
            self.inner,
            kernel_size=kernel,
            groups=self.inner,
        )
        self.left_pad = kernel - 1
        if not self.fast_scan:
            self.dt_proj = nn.Linear(self.inner, self.inner)
            self.b_proj = nn.Linear(self.inner, self.inner * self.d_state)
            self.c_proj = nn.Linear(self.inner, self.inner * self.d_state)
            base = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.inner, 1)
            self.a_log = nn.Parameter(torch.log(base))
            self.d_skip = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        xz = self.in_proj(self.norm(x))
        u, gate = xz.chunk(2, dim=-1)
        u = F.pad(u.transpose(1, 2), (self.left_pad, 0))
        u = F.silu(self.conv(u).transpose(1, 2))
        if self.fast_scan:
            y = u * F.silu(gate)
            return residual + self.dropout(self.out_proj(y))

        delta = F.softplus(self.dt_proj(u)) + 1e-4
        b = self.b_proj(u).view(u.shape[0], u.shape[1], self.inner, self.d_state)
        c = self.c_proj(u).view(u.shape[0], u.shape[1], self.inner, self.d_state)
        a = -torch.exp(self.a_log.float()).to(device=u.device, dtype=u.dtype)
        state = u.new_zeros(u.shape[0], self.inner, self.d_state)

        outputs = []
        for step in range(u.shape[1]):
            dt = delta[:, step].unsqueeze(-1)
            state = torch.exp(dt * a.unsqueeze(0)) * state
            state = state + dt * b[:, step] * u[:, step].unsqueeze(-1)
            y = (state * c[:, step]).sum(dim=-1) + self.d_skip * u[:, step]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(gate)
        return residual + self.dropout(self.out_proj(y))


class MambaBackbone(nn.Module):
    def __init__(self, d_model, param):
        super().__init__()
        layers = _positive_int(param.get("mamba_layers", param.get("encoder_layers", 6)), 6)
        self.blocks = nn.Sequential(*[SelectiveSSMBlock(d_model, param) for _ in range(layers)])
        self.norm = RMSNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        return self.norm(self.blocks(x))


class AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


def _make_backbone(backbone, d_model, param):
    key = str(backbone).strip().lower()
    if key in {"fcn", "tcn"}:
        return TCNBackbone(d_model, param)
    if key in {"transformer", "patchtst", "patch_transformer"}:
        return PatchTransformerBackbone(d_model, param)
    if key in {"gru", "gru_attention", "bigru"}:
        return GRUAttentionBackbone(d_model, param)
    if key in {"temporal", "temporal_mixer", "tsmixer", "mixer"}:
        return TSMixerBackbone(d_model, param)
    if key in {"mamba", "ssm"}:
        return MambaBackbone(d_model, param)
    raise ValueError(f"Unsupported self-touch backbone: {backbone}")


class SelfTouchSequenceModel(nn.Module):
    """Shared four-finger self-touch model for backbone and contrastive sweeps."""

    def __init__(self, param, backbone, contrastive=False):
        super().__init__()
        param = dict(param or {})
        self.backbone_name = str(backbone)
        self.contrastive = bool(contrastive)
        self.hand_dim = int(param.get("hand_dim", 16))
        self.tactile_dim = int(param.get("tactile_dim", 90))
        self.use_joint_pos_only = _as_bool(param.get("use_joint_pos_only", True))
        self.use_derived_features = _as_bool(param.get("use_derived_features", False))
        self.input_modalities = _as_tuple(param.get("input_modalities")) or None
        self.temporal_window_steps = _positive_int(param.get("temporal_window_steps", 1), 1)
        self.use_finger_aware_input = _as_bool(param.get("use_finger_aware_input", False))
        self.exclude_combo_from_encoder = _as_bool(
            param.get("exclude_combo_from_encoder", self.use_finger_aware_input)
        )
        self.use_tactile_history = _as_bool(param.get("use_tactile_history", False))
        self.tactile_history_steps = _positive_int(param.get("tactile_history_steps", 1), 1)
        self.tactile_history_fingers = _as_tuple(
            param.get("tactile_history_fingers"),
            ("index", "thumb", "middle", "ring"),
        )
        self.use_combo_condition = _as_bool(param.get("use_combo_condition", False))
        self.combo_dim = int(param.get("combo_dim", 6))
        self.use_phase_condition = _as_bool(param.get("use_phase_condition", False))
        self.phase_dim = int(param.get("phase_dim", 10))
        self.use_mean_residual_head = False
        self.output_activation = str(param.get("output_activation", "identity")).lower()
        self.output_min = float(param.get("output_min", 0.1))
        self.output_max = float(param.get("output_max", 0.9))

        if self.use_finger_aware_input:
            self.finger_input = FingerAwareInputEncoder(
                self.hand_dim,
                int(param.get("finger_feature_dim", 32)),
            )
            base_input_dim = self.finger_input.output_dim
            if self.use_tactile_history:
                base_input_dim += (
                    self.tactile_dim
                    * self.tactile_history_steps
                    * len(self.tactile_history_fingers)
                )
        else:
            self.finger_input = None
            base_input_dim = joint_feature_dim(
                self.hand_dim,
                self.use_derived_features,
                self.use_joint_pos_only,
                input_modalities=self.input_modalities,
                temporal_window_steps=self.temporal_window_steps,
                tactile_dim=self.tactile_dim,
                use_tactile_history=self.use_tactile_history,
                tactile_history_steps=self.tactile_history_steps,
                tactile_history_fingers=self.tactile_history_fingers,
            )
        self.encoder_use_combo_condition = (
            self.use_combo_condition and not self.exclude_combo_from_encoder
        )
        self.encoder_use_phase_condition = (
            self.use_phase_condition and not self.exclude_combo_from_encoder
        )
        self.context_dim = (
            (self.combo_dim if self.encoder_use_combo_condition else 0)
            + (self.phase_dim if self.encoder_use_phase_condition else 0)
        )
        self.input_dim = int(base_input_dim + self.context_dim)
        d_model = int(param.get("d_model", param.get("encoder_dim", param.get("hidden_dim", 256))))
        combo_decoder_dim = int(param.get("combo_decoder_dim", 256))
        dropout = float(param.get("dropout", 0.1))

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.backbone = _make_backbone(backbone, d_model, param)
        self.decoder = nn.LayerNorm(d_model)
        self.combo_finger_order = COMBO_FINGER_ORDER
        self.combo_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, combo_decoder_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(combo_decoder_dim, self.tactile_dim * len(fingers)),
                )
                for fingers in self.combo_finger_order
            ]
        )
        contact_sets = torch.zeros(len(self.combo_finger_order), len(FINGER_NAMES))
        for combo_idx, fingers in enumerate(self.combo_finger_order):
            for finger in fingers:
                contact_sets[combo_idx, FINGER_TO_OUTPUT_INDEX[finger]] = 1.0
        self.register_buffer(
            "combo_contact_sets",
            contact_sets,
            persistent=False,
        )
        self.mean_residual_net = None

        if self.contrastive:
            proj_hidden_dim = int(param.get("projection_hidden_dim", 128))
            proj_dim = int(param.get("projection_dim", 64))
            num_classes = int(param.get("num_classes", self.combo_dim))
            self.pooler = (
                self.backbone.pool
                if hasattr(self.backbone, "pool")
                else AttentionPool(d_model)
            )
            self.projector = nn.Sequential(
                nn.Linear(d_model, proj_hidden_dim),
                nn.GELU(),
                nn.Linear(proj_hidden_dim, proj_dim),
            )
            self.classifier = nn.Linear(d_model, num_classes)

        self._reset_parameters()
        self._init_tactile_bias()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.finger_input is not None:
            nn.init.normal_(self.finger_input.finger_embedding, mean=0.0, std=0.02)
        if self.mean_residual_net is not None:
            nn.init.zeros_(self.mean_residual_net.weight)
            nn.init.zeros_(self.mean_residual_net.bias)

    def _init_tactile_bias(self):
        return

    def output_bias_from_normalized_target(self, target):
        if self.output_activation != "sigmoid":
            return target
        unit = (target - self.output_min) / (self.output_max - self.output_min)
        eps = torch.finfo(target.dtype).eps
        return torch.logit(unit.clamp(min=eps, max=1.0 - eps))

    def _activate_output(self, x):
        if self.output_activation == "sigmoid":
            return self.output_min + (self.output_max - self.output_min) * torch.sigmoid(x)
        if self.output_activation == "tanh":
            midpoint = (self.output_min + self.output_max) * 0.5
            half_range = (self.output_max - self.output_min) * 0.5
            return midpoint + half_range * torch.tanh(x)
        return x

    def _align_context(self, value, x, width):
        batch, timesteps = x.shape[:2]
        if value is None:
            return x.new_zeros(batch, timesteps, width)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        else:
            value = value.to(device=x.device, dtype=x.dtype)
        if value.ndim == 2:
            value = value.unsqueeze(1).expand(-1, timesteps, -1)
        elif value.ndim >= 3:
            value = value[:, :timesteps, :]
            if value.shape[1] < timesteps:
                pad = value.new_zeros(value.shape[0], timesteps - value.shape[1], value.shape[-1])
                value = torch.cat([value, pad], dim=1)
        else:
            return x.new_zeros(batch, timesteps, width)
        if value.shape[-1] < width:
            value = F.pad(value, (0, width - value.shape[-1]))
        return value[..., :width]

    def _append_context(self, x, selftouch_combo=None, selftouch_phase=None):
        chunks = [x]
        if self.encoder_use_combo_condition:
            chunks.append(self._align_context(selftouch_combo, x, self.combo_dim))
        if self.encoder_use_phase_condition:
            chunks.append(self._align_context(selftouch_phase, x, self.phase_dim))
        return torch.cat(chunks, dim=-1) if len(chunks) > 1 else x

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
        selftouch_combo=None,
        selftouch_phase=None,
    ):
        if self.finger_input is not None:
            x = self.finger_input(hand_jnt_pos, hand_jnt_trq, next_step=True)
            if self.use_tactile_history:
                target_stop = int(hand_jnt_pos.shape[1])
                target_start = max(0, target_stop - x.shape[1])
                tactile_history = build_tactile_history(
                    tactile_index_tip=tactile_index_tip,
                    tactile_thumb_tip=tactile_thumb_tip,
                    tactile_middle_tip=tactile_middle_tip,
                    tactile_ring_tip=tactile_ring_tip,
                    target_start=target_start,
                    target_stop=target_stop,
                    history_steps=self.tactile_history_steps,
                    tactile_dim=self.tactile_dim,
                    fingers=self.tactile_history_fingers,
                    reference=x,
                )
                x = torch.cat(
                    [x, tactile_history.to(device=x.device, dtype=x.dtype)],
                    dim=-1,
                )
            return self._append_context(x, selftouch_combo, selftouch_phase)
        x = build_joint_features(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            next_step=True,
            use_derived_features=self.use_derived_features,
            use_joint_pos_only=self.use_joint_pos_only,
            input_modalities=self.input_modalities,
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
        return self._append_context(x, selftouch_combo, selftouch_phase)

    def encode_sequence(
        self,
        hand_jnt_pos,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        hand_jnt_cmd_pos=None,
        tactile_index_tip=None,
        tactile_thumb_tip=None,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
        selftouch_combo=None,
        selftouch_phase=None,
        src_key_padding_mask=None,
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
            selftouch_combo=selftouch_combo,
            selftouch_phase=selftouch_phase,
        )
        h = self.input_proj(x)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask[:, : h.shape[1]]
        h = self.backbone(h, src_key_padding_mask=src_key_padding_mask)
        return h, x

    def encode(
        self,
        hand_jnt_pos,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        hand_jnt_cmd_pos=None,
        selftouch_combo=None,
        selftouch_phase=None,
    ):
        h, _ = self.encode_sequence(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            selftouch_combo=selftouch_combo,
            selftouch_phase=selftouch_phase,
        )
        if self.contrastive:
            return self.pooler(h)
        return h.mean(dim=1)

    def _combo_labels(self, selftouch_combo, batch, device):
        labels = labels_from_selftouch_combo(selftouch_combo)
        if labels is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        labels = labels.to(device=device).long().reshape(-1)
        if labels.shape[0] < batch:
            labels = F.pad(labels, (0, batch - labels.shape[0]))
        return labels[:batch].clamp(min=0, max=len(self.combo_decoders) - 1)

    def _contrastive_pair_weights(self, labels):
        if labels is None or self.combo_contact_sets.numel() == 0:
            return None
        labels = labels.to(device=self.combo_contact_sets.device).long().reshape(-1)
        labels = labels.clamp(min=0, max=self.combo_contact_sets.shape[0] - 1)
        contacts = self.combo_contact_sets[labels]
        intersection = torch.minimum(contacts[:, None, :], contacts[None, :, :]).sum(dim=-1)
        union = torch.maximum(contacts[:, None, :], contacts[None, :, :]).sum(dim=-1)
        similarity = intersection / union.clamp_min(1.0)
        weights = 1.0 - similarity
        same = labels[:, None].eq(labels[None, :])
        weights = torch.where(same, torch.ones_like(weights), weights)
        eye = torch.eye(labels.shape[0], device=weights.device, dtype=torch.bool)
        return weights.masked_fill(eye, 0.0)

    def _predict_from_hidden(self, hidden, selftouch_combo=None):
        latent = self.decoder(hidden)
        batch, timesteps = latent.shape[:2]
        labels = self._combo_labels(selftouch_combo, batch, latent.device)
        outputs = [
            latent.new_zeros(batch, timesteps, self.tactile_dim)
            for _ in FINGER_NAMES
        ]

        for combo_idx, (decoder, fingers) in enumerate(
            zip(self.combo_decoders, self.combo_finger_order)
        ):
            mask = labels.eq(combo_idx).to(dtype=latent.dtype).view(batch, 1, 1)
            if not bool(mask.any()):
                continue
            combo_pred = self._activate_output(decoder(latent)) * mask
            offset = 0
            for finger in fingers:
                finger_idx = FINGER_TO_OUTPUT_INDEX[finger]
                outputs[finger_idx] = (
                    outputs[finger_idx]
                    + combo_pred[..., offset : offset + self.tactile_dim]
                )
                offset += self.tactile_dim
        return tuple(outputs)

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
        selftouch_combo=None,
        selftouch_phase=None,
        **_unused,
    ):
        h, _ = self.encode_sequence(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
            selftouch_combo=selftouch_combo,
            selftouch_phase=selftouch_phase,
            src_key_padding_mask=src_key_padding_mask,
        )
        preds = self._predict_from_hidden(h, selftouch_combo=selftouch_combo)
        if not self.contrastive:
            return preds

        embedding = self.pooler(h)
        projection = F.normalize(self.projector(embedding), dim=-1)
        logits = self.classifier(embedding)
        return {
            "embedding": embedding,
            "projection": projection,
            "logits": logits,
            "idx_pred": preds[0],
            "thumb_pred": preds[1],
            "middle_pred": preds[2],
            "ring_pred": preds[3],
        }

    def _tactile_losses(
        self,
        preds,
        tactile_index_tip,
        tactile_thumb_tip,
        tactile_middle_tip,
        tactile_ring_tip,
        selftouch_finger_mask,
        loss_coef,
    ):
        target_start = 1
        target_stop = tactile_index_tip.shape[1]
        targets = (
            tactile_index_tip[:, target_start:target_stop, :],
            tactile_thumb_tip[:, target_start:target_stop, :],
            tactile_middle_tip[:, target_start:target_stop, :],
            tactile_ring_tip[:, target_start:target_stop, :],
        )
        losses = []
        for pred, target, finger, key in zip(preds, targets, FINGER_NAMES, FINGER_LOSS_KEYS):
            loss_pred, loss_target = select_valid_finger_targets(
                pred,
                target,
                selftouch_finger_mask,
                finger,
                target_start,
                target_stop,
            )
            losses.append(
                active_tactile_loss(
                    loss_pred,
                    loss_target,
                    finger_loss_config(loss_coef, key),
                )
            )
        return tuple(losses)

    def forward_loss(
        self,
        tactile_index_tip,
        tactile_thumb_tip,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
        hand_jnt_pos=None,
        hand_jnt_cmd_pos=None,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        labels=None,
        data_found=None,
        selftouch_combo=None,
        selftouch_phase=None,
        selftouch_finger_mask=None,
        loss_coef=None,
        src_key_padding_mask=None,
        **_unused,
    ):
        coef = loss_coef or {}
        out = self.forward(
            hand_jnt_pos=hand_jnt_pos,
            hand_jnt_vel=hand_jnt_vel,
            hand_jnt_trq=hand_jnt_trq,
            hand_jnt_cmd_pos=hand_jnt_cmd_pos,
            src_key_padding_mask=src_key_padding_mask,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
            selftouch_combo=selftouch_combo,
            selftouch_phase=selftouch_phase,
        )
        if self.contrastive:
            preds = (out["idx_pred"], out["thumb_pred"], out["middle_pred"], out["ring_pred"])
        else:
            preds = out

        loss_index, loss_thumb, loss_middle, loss_ring = self._tactile_losses(
            preds,
            tactile_index_tip,
            tactile_thumb_tip,
            tactile_middle_tip,
            tactile_ring_tip,
            selftouch_finger_mask,
            coef,
        )
        total = (
            coef.get("tactile_index_tip", 1.0) * loss_index
            + coef.get("tactile_thumb_tip", 1.0) * loss_thumb
            + coef.get("tactile_middle_tip", 1.0) * loss_middle
            + coef.get("tactile_ring_tip", 1.0) * loss_ring
        )

        if self.contrastive:
            if labels is None:
                labels = labels_from_selftouch_combo(selftouch_combo)
            if labels is not None:
                if not torch.is_tensor(labels):
                    labels = torch.as_tensor(labels, device=out["logits"].device)
                else:
                    labels = labels.to(device=out["logits"].device)
                labels = labels.long()
            pair_weights = (
                self._contrastive_pair_weights(labels)
                if _as_bool(coef.get("contact_overlap_contrastive", True))
                else None
            )
            contrastive_loss = supervised_contrastive_loss(
                out["projection"],
                labels,
                coef.get("contrastive_temperature", 0.07),
                pair_weights=pair_weights,
            )
            if labels is None:
                cls_loss = out["logits"].sum() * 0.0
            else:
                cls_loss = F.cross_entropy(out["logits"], labels)
            total = (
                total
                + coef.get("contrastive", 0.1) * contrastive_loss
                + coef.get("classification", 0.2) * cls_loss
            )

        return (total, loss_index, loss_thumb, loss_middle, loss_ring), preds
