import torch
from torch import nn

from selftouch_feature_utils import build_tactile_history, causal_temporal_window
from selftouch_loss_utils import active_tactile_loss, finger_loss_config, select_valid_finger_targets
from selftouch_offset_utils import target_window


class DilatedTemporalBlock(nn.Module):
    def __init__(self, dim, kernel_size=9, dilation=1, dropout=0.0):
        super().__init__()
        padding = (int(kernel_size) // 2) * int(dilation)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=int(kernel_size),
            padding=padding,
            dilation=int(dilation),
            groups=dim,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Conv1d(dim * 2, dim, kernel_size=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        y = self.depthwise(x.transpose(1, 2))
        y = self.pointwise(y).transpose(1, 2)
        if y.shape[1] != x.shape[1]:
            y = y[:, : x.shape[1], :]
        return self.norm(x + y)


class SelfTouch(nn.Module):
    """Position-only temporal FCN with explicit mean-trace and taxel residual heads."""

    INPUT_MODALITIES = ("hand_jnt_pos",)

    def __init__(self, param):
        super().__init__()
        self.hand_dim = int(param["hand_dim"])
        self.tactile_dim = int(param["tactile_dim"])
        self.input_offset = int(param.get("input_offset", 0))
        self.input_modalities = tuple(param.get("input_modalities", self.INPUT_MODALITIES))
        if self.input_modalities != self.INPUT_MODALITIES:
            raise ValueError("selftouch_fcn_pos requires hand_jnt_pos only")

        hidden_dim = int(param.get("hidden_dim", param.get("rec_dim", 256)))
        encoder_dim = int(param.get("encoder_dim", hidden_dim))
        decoder_dim = int(param.get("decoder_dim", hidden_dim))
        kernel_size = int(param.get("temporal_kernel_size", 9))
        dropout = float(param.get("dropout", 0.08))
        use_deltas = bool(param.get("use_pos_temporal_deltas", True))
        use_combo = bool(param.get("use_combo_condition", True))
        combo_dim = int(param.get("combo_dim", 3))
        use_phase = bool(param.get("use_phase_condition", True))
        phase_dim = int(param.get("phase_dim", 10))
        self.use_deltas = use_deltas
        self.use_combo = use_combo
        self.combo_dim = combo_dim
        self.use_phase = use_phase
        self.phase_dim = phase_dim
        self.temporal_window_steps = max(1, int(param.get("temporal_window_steps", 1)))
        self.use_tactile_history = bool(param.get("use_tactile_history", False))
        self.tactile_history_steps = max(1, int(param.get("tactile_history_steps", 1)))
        self.tactile_history_fingers = tuple(
            param.get("tactile_history_fingers", ("index", "thumb", "middle", "ring"))
        )

        feature_dim = self.hand_dim * (3 if use_deltas else 1)
        if use_combo:
            feature_dim += combo_dim
        if use_phase:
            feature_dim += phase_dim
        feature_dim *= self.temporal_window_steps
        if self.use_tactile_history:
            feature_dim += self.tactile_dim * self.tactile_history_steps * len(self.tactile_history_fingers)
        self.input_net = nn.Sequential(
            nn.Linear(feature_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        dilations = param.get("temporal_dilations", [1, 2, 4, 8, 16, 1])
        self.temporal = nn.Sequential(*[
            DilatedTemporalBlock(
                hidden_dim,
                kernel_size=kernel_size,
                dilation=int(dilation),
                dropout=dropout,
            )
            for dilation in dilations
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
        )
        rec_out = self.tactile_dim * 4
        self.output_net = nn.Linear(decoder_dim, rec_out)
        self.mean_net = nn.Linear(decoder_dim, 4)
        self._init_mean_head()

    def _init_mean_head(self):
        nn.init.zeros_(self.mean_net.weight)
        nn.init.zeros_(self.mean_net.bias)

    def _finger_loss_coef(self, loss_coef, tactile_key):
        cfg = dict(loss_coef or {})
        scales = cfg.get("tactile_raw_scale_by_key")
        if isinstance(scales, dict) and tactile_key in scales:
            cfg["tactile_raw_scale"] = scales[tactile_key]
        slopes = cfg.get("tactile_raw_slope_by_key")
        if isinstance(slopes, dict) and tactile_key in slopes:
            cfg["tactile_raw_slope"] = slopes[tactile_key]
        return cfg

    def _target_window(self, num_timesteps):
        start, stop = target_window(num_timesteps, self.input_offset)
        if stop <= start:
            raise ValueError(
                f"input_offset={self.input_offset} leaves no valid target timesteps "
                f"for sequence length {num_timesteps}"
            )
        return start, stop

    @staticmethod
    def _causal_delta(x):
        delta = torch.zeros_like(x)
        if x.shape[1] > 1:
            delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return delta

    def _pos_features(
        self,
        hand_jnt_pos,
        selftouch_combo=None,
        selftouch_phase=None,
        tactile_index_tip=None,
        tactile_thumb_tip=None,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
    ):
        target_start, target_stop = self._target_window(hand_jnt_pos.shape[1])
        input_start = target_start + self.input_offset
        input_stop = target_stop + self.input_offset
        pos = hand_jnt_pos
        features = [pos]

        if self.use_deltas:
            delta = self._causal_delta(pos)
            accel = self._causal_delta(delta)
            features.extend([delta, accel])

        if self.use_combo:
            if selftouch_combo is None:
                combo = pos.new_zeros(pos.shape[0], pos.shape[1], self.combo_dim)
            else:
                combo = selftouch_combo.to(device=pos.device, dtype=pos.dtype)
            features.append(combo)

        if self.use_phase:
            if selftouch_phase is None:
                phase = pos.new_zeros(pos.shape[0], pos.shape[1], self.phase_dim)
            else:
                phase = selftouch_phase.to(device=pos.device, dtype=pos.dtype)
            features.append(phase)

        features = causal_temporal_window(
            torch.cat(features, dim=-1),
            start=input_start,
            stop=input_stop,
            steps=self.temporal_window_steps,
        )
        if self.use_tactile_history:
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
                reference=features,
            )
            features = torch.cat(
                [features, tactile_history.to(device=features.device, dtype=features.dtype)],
                dim=-1,
            )
        return features

    def forward(
        self,
        hand_jnt_pos=None,
        hand_jnt_vel=None,
        hand_jnt_trq=None,
        hand_jnt_cmd_pos=None,
        selftouch_combo=None,
        selftouch_phase=None,
        tactile_index_tip=None,
        tactile_thumb_tip=None,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
    ):
        if hand_jnt_pos is None:
            raise KeyError("Missing required selftouch_fcn_pos input: hand_jnt_pos")
        x = self._pos_features(
            hand_jnt_pos,
            selftouch_combo=selftouch_combo,
            selftouch_phase=selftouch_phase,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
        )
        hidden = self.input_net(x)
        hidden = self.temporal(hidden)
        decoded = self.decoder(hidden)

        taxel = self.output_net(decoded)
        mean_adjust = self.mean_net(decoded).repeat_interleave(self.tactile_dim, dim=-1)
        out = taxel + mean_adjust

        idx_pred = out[..., : self.tactile_dim]
        thumb_pred = out[..., self.tactile_dim : self.tactile_dim * 2]
        middle_pred = out[..., self.tactile_dim * 2 : self.tactile_dim * 3]
        ring_pred = out[..., self.tactile_dim * 3 :]
        return idx_pred, thumb_pred, middle_pred, ring_pred

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
        selftouch_finger_mask=None,
        loss_coef=None,
        tactile_middle_tip=None,
        tactile_ring_tip=None,
        **_unused,
    ):
        loss_coef = loss_coef or {}
        target_start, target_stop = self._target_window(tactile_index_tip.shape[1])
        idx_pred, thumb_pred, middle_pred, ring_pred = self.forward(
            hand_jnt_pos=hand_jnt_pos,
            selftouch_combo=selftouch_combo,
            selftouch_phase=selftouch_phase,
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            tactile_middle_tip=tactile_middle_tip,
            tactile_ring_tip=tactile_ring_tip,
        )

        index_target = tactile_index_tip[:, target_start:target_stop, :]
        thumb_target = tactile_thumb_tip[:, target_start:target_stop, :]
        middle_target = tactile_middle_tip[:, target_start:target_stop, :]
        ring_target = tactile_ring_tip[:, target_start:target_stop, :]
        idx_loss_pred, index_target = select_valid_finger_targets(
            idx_pred, index_target, selftouch_finger_mask, "index", target_start, target_stop
        )
        thumb_loss_pred, thumb_target = select_valid_finger_targets(
            thumb_pred, thumb_target, selftouch_finger_mask, "thumb", target_start, target_stop
        )
        middle_loss_pred, middle_target = select_valid_finger_targets(
            middle_pred, middle_target, selftouch_finger_mask, "middle", target_start, target_stop
        )
        ring_loss_pred, ring_target = select_valid_finger_targets(
            ring_pred, ring_target, selftouch_finger_mask, "ring", target_start, target_stop
        )

        loss_index = active_tactile_loss(
            idx_loss_pred,
            index_target,
            finger_loss_config(loss_coef, "tactile_index_tip"),
        )
        loss_thumb = active_tactile_loss(
            thumb_loss_pred,
            thumb_target,
            finger_loss_config(loss_coef, "tactile_thumb_tip"),
        )
        loss_middle = active_tactile_loss(
            middle_loss_pred,
            middle_target,
            finger_loss_config(loss_coef, "tactile_middle_tip"),
        )
        loss_ring = active_tactile_loss(
            ring_loss_pred,
            ring_target,
            finger_loss_config(loss_coef, "tactile_ring_tip"),
        )

        total_loss = (
            loss_coef.get("tactile_index_tip", 1.0) * loss_index
            + loss_coef.get("tactile_thumb_tip", 1.0) * loss_thumb
            + loss_coef.get("tactile_middle_tip", 1.0) * loss_middle
            + loss_coef.get("tactile_ring_tip", 1.0) * loss_ring
        )
        return (total_loss, loss_index, loss_thumb, loss_middle, loss_ring), (
            idx_pred,
            thumb_pred,
            middle_pred,
            ring_pred,
        )


if __name__ == "__main__":
    print("Model for predicting self touch")
