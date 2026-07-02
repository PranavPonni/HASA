import torch
from torch import nn

from selftouch_loss_utils import active_tactile_loss


class TemporalBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, dropout=0.0):
        super().__init__()
        padding = (int(kernel_size) // 2) * int(dilation)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=int(dilation)),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=int(dilation)),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        if y.shape[1] != x.shape[1]:
            y = y[:, : x.shape[1], :]
        return self.norm(x + y)


class ControlledTemporalSelfTouch(nn.Module):
    """Shared encoder/decoder FCN used by all selftouch_fcn input variants."""

    INPUT_MODALITIES = ("hand_jnt_pos",)

    def __init__(self, param):
        super().__init__()
        self.hand_dim = int(param["hand_dim"])
        self.tactile_dim = int(param["tactile_dim"])
        self.input_modalities = tuple(param.get("input_modalities", self.INPUT_MODALITIES))
        if not self.input_modalities:
            raise ValueError("Model.input_modalities must contain at least one stream")

        hidden_dim = int(param.get("hidden_dim", param["rec_dim"]))
        encoder_dim = int(param.get("encoder_dim", 64))
        decoder_dim = int(param.get("decoder_dim", 64))
        decoder_out_dim = int(param.get("decoder_out_dim", 188))
        temporal_blocks = int(param.get("temporal_blocks", 0))
        temporal_kernel_size = int(param.get("temporal_kernel_size", 5))
        dropout = float(param["dropout"])
        self.use_combo = bool(param.get("use_combo_condition", True))
        self.combo_dim = int(param.get("combo_dim", 3))
        self.use_phase = bool(param.get("use_phase_condition", True))
        self.phase_dim = int(param.get("phase_dim", 10))
        self.use_mean_residual_head = bool(param.get("use_mean_residual_head", True))

        rec_in = self.hand_dim * len(self.input_modalities)
        if self.use_combo:
            rec_in += self.combo_dim
        if self.use_phase:
            rec_in += self.phase_dim
        rec_out = self.tactile_dim * 3

        self.encoder = nn.Sequential(
            nn.Linear(rec_in, encoder_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_out_dim),
            nn.GELU(),
        )
        temporal_dilations = list(param.get("temporal_dilations", []))
        if not temporal_dilations:
            temporal_dilations = [1] * max(0, temporal_blocks)
        self.temporal = nn.Sequential(*[
            TemporalBlock(
                hidden_dim,
                kernel_size=temporal_kernel_size,
                dilation=temporal_dilations[idx % len(temporal_dilations)],
                dropout=dropout,
            )
            for idx in range(max(0, temporal_blocks))
        ])
        self.output_net = nn.Linear(decoder_out_dim, rec_out)
        self.mean_net = nn.Linear(decoder_out_dim, 3)
        self._init_mean_head()
        self._init_tactile_bias()

    def _init_mean_head(self):
        nn.init.zeros_(self.mean_net.weight)
        nn.init.zeros_(self.mean_net.bias)

    def _init_tactile_bias(self):
        return

    def _activate_output(self, x):
        return x

    def _finger_loss_coef(self, loss_coef, tactile_key):
        cfg = dict(loss_coef or {})
        scales = cfg.get("tactile_raw_scale_by_key")
        if isinstance(scales, dict) and tactile_key in scales:
            cfg["tactile_raw_scale"] = scales[tactile_key]
        slopes = cfg.get("tactile_raw_slope_by_key")
        if isinstance(slopes, dict) and tactile_key in slopes:
            cfg["tactile_raw_slope"] = slopes[tactile_key]
        return cfg

    def _build_input(
        self,
        hand_jnt_pos=None,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        hand_jnt_cmd_pos=None,
        selftouch_combo=None,
        selftouch_phase=None,
    ):
        streams = {
            "hand_jnt_pos": hand_jnt_pos,
            "hand_jnt_vel": hand_jnt_vel,
            "hand_jnt_trq": hand_jnt_trq,
            "hand_jnt_cmd_pos": hand_jnt_cmd_pos,
        }
        chunks = []
        for name in self.input_modalities:
            value = streams.get(name)
            if value is None:
                raise KeyError(f"Missing required selftouch_fcn input: {name}")
            chunks.append(value[:, 1:, :])
        reference = chunks[0]
        if self.use_combo:
            if selftouch_combo is None:
                combo = reference.new_zeros(reference.shape[0], reference.shape[1], self.combo_dim)
            else:
                combo = selftouch_combo[:, 1:, :].to(device=reference.device, dtype=reference.dtype)
            chunks.append(combo)
        if self.use_phase:
            if selftouch_phase is None:
                phase = reference.new_zeros(reference.shape[0], reference.shape[1], self.phase_dim)
            else:
                phase = selftouch_phase[:, 1:, :].to(device=reference.device, dtype=reference.dtype)
            chunks.append(phase)
        return torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]

    def forward(
        self,
        hand_jnt_pos=None,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        hand_jnt_cmd_pos=None,
        selftouch_combo=None,
        selftouch_phase=None,
    ):
        x = self._build_input(
            hand_jnt_pos=hand_jnt_pos,
            hand_jnt_vel=hand_jnt_vel,
            hand_jnt_trq=hand_jnt_trq,
            hand_jnt_cmd_pos=hand_jnt_cmd_pos,
            selftouch_combo=selftouch_combo,
            selftouch_phase=selftouch_phase,
        )
        hidden = self.encoder(x)
        hidden = self.temporal(hidden)
        decoded = self.decoder(hidden)
        out = self._activate_output(self.output_net(decoded))
        if self.use_mean_residual_head:
            out = out + self.mean_net(decoded).repeat_interleave(self.tactile_dim, dim=-1)
        idx_pred = out[..., : self.tactile_dim]
        thumb_pred = out[..., self.tactile_dim : self.tactile_dim * 2]
        middle_pred = out[..., self.tactile_dim * 2 :]
        return idx_pred, thumb_pred, middle_pred

    def forward_loss(
        self,
        tactile_index_tip,
        tactile_thumb_tip,
        hand_jnt_pos=None,
        hand_jnt_cmd_pos=None,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        data_found=None,
        selftouch_combo=None,
        selftouch_phase=None,
        loss_coef=None,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
        **_unused,
    ):
        loss_coef = loss_coef or {}
        idx_pred, thumb_pred, middle_pred = self.forward(
            hand_jnt_pos=hand_jnt_pos,
            hand_jnt_vel=hand_jnt_vel,
            hand_jnt_trq=hand_jnt_trq,
            hand_jnt_cmd_pos=hand_jnt_cmd_pos,
            selftouch_combo=selftouch_combo,
            selftouch_phase=selftouch_phase,
        )

        loss_index = active_tactile_loss(
            idx_pred,
            tactile_index_tip[:, 1:, :],
            self._finger_loss_coef(loss_coef, "tactile_index_tip"),
        )
        loss_thumb = active_tactile_loss(
            thumb_pred,
            tactile_thumb_tip[:, 1:, :],
            self._finger_loss_coef(loss_coef, "tactile_thumb_tip"),
        )
        loss_middle = active_tactile_loss(
            middle_pred,
            tactile_middle_tip[:, 1:, :],
            self._finger_loss_coef(loss_coef, "tactile_middle_tip"),
        )

        total_loss = (
            loss_coef.get("tactile_index_tip", 1.0) * loss_index
            + loss_coef.get("tactile_thumb_tip", 1.0) * loss_thumb
            + loss_coef.get("tactile_middle_tip", 1.0) * loss_middle
        )
        return (total_loss, loss_index, loss_thumb, loss_middle), (
            idx_pred,
            thumb_pred,
            middle_pred,
        )
