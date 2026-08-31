import math

import torch
from torch import nn
import torch.nn.functional as F


def _causal_mask(length, device):
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )


def _noise_value(noise, key, fallback=0.0):
    if noise is None:
        return fallback
    if isinstance(noise, dict):
        return float(noise.get(key, fallback))
    return float(noise)


def _add_noise(x, std):
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def _positive_int(value, default=1):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = int(default)
    return max(1, value)


def _masked_mse(pred, target, mask):
    loss = F.mse_loss(pred, target, reduction="none")
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(loss).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _fit_timesteps(value, target_len):
    """Crop or zero-pad temporal features to match the policy observation length."""
    if value.shape[1] == target_len:
        return value
    if value.shape[1] > target_len:
        return value[:, :target_len]
    pad = value.new_zeros(value.shape[0], target_len - value.shape[1], value.shape[-1])
    return torch.cat([value, pad], dim=1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None].float() * emb[None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class MotionACT(nn.Module):
    """ACT-style encoder/decoder adapted from DexWild for local motion tensors."""

    def __init__(self, obs_dim, action_dim, param):
        super().__init__()
        d_model = int(param.get("d_model", param.get("token_dim", 128)))
        nhead = int(param.get("nhead", 4))
        dropout = float(param.get("dropout", 0.1))
        ff_dim = int(param.get("dim_feedforward", d_model * 4))
        max_seq_len = int(param.get("max_seq_len", 512))
        enc_layers = int(param.get("encoder_layers", 3))
        dec_layers = int(param.get("decoder_layers", 3))

        self.causal = bool(param.get("causal", True))
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.query = nn.Embedding(max_seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.out = nn.Linear(d_model, action_dim)

    def forward(self, obs, target_actions=None, output_steps=None):
        batch_size, src_steps, _ = obs.shape
        if target_actions is not None:
            query_steps = int(target_actions.shape[1])
        elif output_steps is not None:
            query_steps = int(output_steps)
        else:
            query_steps = int(src_steps)
        if src_steps > self.pos.shape[1] or query_steps > self.query.num_embeddings:
            raise ValueError(
                f"Sequence length src={src_steps}, query={query_steps} exceeds max_seq_len"
            )
        src = self.obs_proj(obs) + self.pos[:, :src_steps]
        src_mask = _causal_mask(src_steps, obs.device) if self.causal else None
        tgt_mask = _causal_mask(query_steps, obs.device) if self.causal else None
        memory = self.encoder(src, mask=src_mask)
        query_idx = torch.arange(query_steps, device=obs.device)
        tgt = self.query(query_idx)[None].expand(batch_size, -1, -1)
        pred = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.out(pred), {}


class MotionViT(nn.Module):
    """ViT-style token mixer where each motion timestep is treated as a patch."""

    def __init__(self, obs_dim, action_dim, param):
        super().__init__()
        d_model = int(param.get("d_model", param.get("token_dim", 128)))
        nhead = int(param.get("nhead", 4))
        dropout = float(param.get("dropout", 0.1))
        ff_dim = int(param.get("dim_feedforward", d_model * 4))
        max_seq_len = int(param.get("max_seq_len", 512))
        layers = int(param.get("encoder_layers", 6))

        self.causal = bool(param.get("causal", True))
        self.patch = nn.Linear(obs_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, action_dim)

    def forward(self, obs, target_actions=None):
        steps = obs.shape[1]
        if steps > self.pos.shape[1]:
            raise ValueError(f"Sequence length {steps} exceeds max_seq_len")
        tokens = self.patch(obs) + self.pos[:, :steps]
        mask = _causal_mask(steps, obs.device) if self.causal else None
        tokens = self.encoder(tokens, mask=mask)
        return self.out(self.norm(tokens)), {}


class MotionDiffusion(nn.Module):
    """Lightweight diffusion-transformer head for next-step motion targets."""

    def __init__(self, obs_dim, action_dim, param):
        super().__init__()
        d_model = int(param.get("d_model", param.get("token_dim", 128)))
        nhead = int(param.get("nhead", 4))
        dropout = float(param.get("dropout", 0.1))
        ff_dim = int(param.get("dim_feedforward", d_model * 4))
        max_seq_len = int(param.get("max_seq_len", 512))
        layers = int(param.get("encoder_layers", 4))
        self.train_steps = int(param.get("diffusion_steps", 64))
        self.causal = bool(param.get("causal", True))

        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.noisy_proj = nn.Linear(action_dim, d_model)
        self.time = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        denoise_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.cond_encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.denoiser = nn.TransformerEncoder(denoise_layer, num_layers=layers)
        self.base_head = nn.Linear(d_model, action_dim)
        self.noise_head = nn.Linear(d_model, action_dim)

        betas = torch.linspace(1e-4, 2e-2, self.train_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )

    def _condition(self, obs):
        steps = obs.shape[1]
        if steps > self.pos.shape[1]:
            raise ValueError(f"Sequence length {steps} exceeds max_seq_len")
        tokens = self.obs_proj(obs) + self.pos[:, :steps]
        mask = _causal_mask(steps, obs.device) if self.causal else None
        return self.cond_encoder(tokens, mask=mask)

    def forward(self, obs, target_actions=None):
        cond = self._condition(obs)
        base_pred = self.base_head(cond)
        aux = {}
        if target_actions is not None:
            batch_size = target_actions.shape[0]
            timesteps = torch.randint(
                0, self.train_steps, (batch_size,), device=target_actions.device
            )
            noise = torch.randn_like(target_actions)
            a = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
            b = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
            noisy_actions = a * target_actions + b * noise
            time_tokens = self.time(timesteps)[:, None]
            tokens = self.noisy_proj(noisy_actions) + cond + time_tokens
            mask = (
                _causal_mask(tokens.shape[1], tokens.device) if self.causal else None
            )
            noise_pred = self.noise_head(self.denoiser(tokens, mask=mask))
            aux = {"noise_pred": noise_pred, "noise": noise}
        return base_pred, aux


class DexWildMotionModel(nn.Module):
    TACTILE_OUTPUT_MAP = {
        "tactile_index_tip": ("idx_pred", 0),
        "tactile_thumb_tip": ("thumb_pred", 1),
        "tactile_middle_tip": ("middle_pred", 2),
        "tactile_ring_tip": ("ring_pred", 3),
    }

    def __init__(self, param, selftouch=None):
        super().__init__()
        self.param = param
        self.hand_dim = int(param["hand_dim"])
        self.tactile_dim = int(param["tactile_dim"])
        self.use_selftouch = bool(param.get("use_selftouch", False))
        self.selftouch = selftouch
        self.selftouch_input_dim = int(param.get("selftouch_input_dim", 16))
        self.selftouch_joint_indices = list(
            param.get("selftouch_joint_indices", list(range(16)))
        )
        self.tactile_keys = tuple(
            param.get(
                "tactile_keys",
                [
                    "tactile_index_tip",
                    "tactile_thumb_tip",
                    "tactile_middle_tip",
                    "tactile_ring_tip",
                ],
            )
        )
        if "joint_state_keys" in param:
            self.joint_state_keys = tuple(param["joint_state_keys"])
        elif bool(param.get("use_velocity_input", False)):
            self.joint_state_keys = ("hand_jnt_pos", "hand_jnt_vel")
        else:
            self.joint_state_keys = ("hand_jnt_pos",)
        self.selftouch_feature_keys = tuple(
            param.get("selftouch_feature_keys", self.tactile_keys)
        )
        teacher_loss_keys = param.get(
            "selftouch_teacher_loss_keys",
            self.selftouch_feature_keys,
        )
        if teacher_loss_keys is None:
            teacher_loss_keys = ()
        self.selftouch_teacher_loss_keys = tuple(
            key
            for key in teacher_loss_keys
            if key in self.tactile_keys
        )
        self.selftouch_feature_scale = float(param.get("selftouch_feature_scale", 1.0))
        self.output_keys = self.tactile_keys + self.joint_state_keys
        self.temporal_delta_keys = tuple(
            key
            for key in param.get("temporal_delta_keys", [])
            if key in self.output_keys
        )
        self.action_chunk_size = max(
            0,
            int(param.get("action_chunk_size", param.get("chunk_size", 0)) or 0),
        )
        self.action_chunk_stride = _positive_int(
            param.get("action_chunk_stride", param.get("chunk_stride", 1)),
            default=1,
        )
        self.act_obs_steps = _positive_int(
            param.get("act_obs_steps", param.get("trajectory_condition_steps", 1)),
            default=1,
        )
        loss_names = (
            ("total_loss",)
            + self.output_keys
            + tuple(f"{key}_delta" for key in self.temporal_delta_keys)
        )
        if self.use_selftouch and self.selftouch_teacher_loss_keys:
            loss_names = loss_names + ("selftouch_teacher",)
        self.loss_names = loss_names

        obs_dim = len(self.tactile_keys) * self.tactile_dim
        obs_dim += len(self.joint_state_keys) * self.hand_dim
        if self.use_selftouch:
            obs_dim += len(self.selftouch_feature_keys) * self.tactile_dim
        self.action_dim = len(self.tactile_keys) * self.tactile_dim
        self.action_dim += len(self.joint_state_keys) * self.hand_dim

        arch = str(param.get("dexwild_arch", "act")).lower()
        if arch == "act":
            self.policy = MotionACT(obs_dim, self.action_dim, param)
        elif arch in {"diff", "diffusion"}:
            self.policy = MotionDiffusion(obs_dim, self.action_dim, param)
        elif arch == "vit":
            self.policy = MotionViT(obs_dim, self.action_dim, param)
        else:
            raise ValueError(f"Unknown dexwild_arch: {arch}")
        self.arch = arch

        if self.selftouch is not None:
            self.selftouch.eval()
            for p in self.selftouch.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.selftouch is not None:
            self.selftouch.eval()
        return self

    def uses_action_chunks(self):
        return self.arch == "act" and self.action_chunk_size > 0

    def _expand_selftouch_joint(self, x):
        if x.shape[-1] == self.selftouch_input_dim:
            return x
        out = x.new_zeros(*x.shape[:-1], self.selftouch_input_dim)
        indices = self.selftouch_joint_indices
        if len(indices) != x.shape[-1] or max(indices) >= self.selftouch_input_dim:
            indices = list(range(min(x.shape[-1], self.selftouch_input_dim)))
        index = torch.tensor(indices, dtype=torch.long, device=x.device)
        return out.index_copy(-1, index, x[..., : len(indices)])

    def _as_sequence(self, value, reference):
        if value is None:
            return None
        if value.dim() == 2 and reference.dim() == 3:
            return value[:, None]
        return value

    def _stream_sequence(self, streams, key, reference=None):
        value = streams.get(key)
        if value is None:
            if reference is None:
                raise KeyError(f"Missing required motion stream: {key}")
            value = torch.zeros_like(reference)
        if reference is not None:
            value = self._as_sequence(value, reference)
        return value

    def _motion_streams_from_args(
        self,
        tactile_index_tip=None,
        tactile_thumb_tip=None,
        hand_jnt_pos=None,
        hand_jnt_vel=None,
        **streams,
    ):
        if tactile_index_tip is not None:
            streams["tactile_index_tip"] = tactile_index_tip
        if tactile_thumb_tip is not None:
            streams["tactile_thumb_tip"] = tactile_thumb_tip
        if hand_jnt_pos is not None:
            streams["hand_jnt_pos"] = hand_jnt_pos
        if hand_jnt_vel is not None:
            streams["hand_jnt_vel"] = hand_jnt_vel
        reference = next((streams.get(k) for k in self.output_keys if streams.get(k) is not None), None)
        if reference is not None:
            for key, value in list(streams.items()):
                if torch.is_tensor(value):
                    streams[key] = self._as_sequence(value, reference)
        return streams

    def _precomputed_selftouch_features(self, streams, reference):
        features = {}
        for key in self.selftouch_feature_keys:
            value = streams.get(f"selftouch_feature_{key}")
            if value is None:
                return None
            features[key] = self._as_sequence(value, reference)
        return features

    def _precomputed_selftouch_teacher_features(self, streams, reference):
        features = {}
        for key in self.selftouch_teacher_loss_keys:
            value = streams.get(f"selftouch_teacher_feature_{key}")
            if value is None:
                return None
            features[key] = self._as_sequence(value, reference)
        return features

    def _selftouch_features(
        self,
        streams,
        selftouch_hand_jnt_pos=None,
        selftouch_hand_jnt_vel=None,
        selftouch_hand_jnt_trq=None,
        selftouch_hand_jnt_cmd_pos=None,
        selftouch_tactile_index_tip=None,
        selftouch_tactile_thumb_tip=None,
        selftouch_tactile_middle_tip=None,
        selftouch_tactile_ring_tip=None,
        selftouch_combo=None,
        selftouch_phase=None,
    ):
        if not self.use_selftouch:
            return None
        hand_jnt_pos = self._stream_sequence(streams, "hand_jnt_pos")
        precomputed = self._precomputed_selftouch_features(streams, hand_jnt_pos)
        if precomputed is not None:
            return precomputed
        if self.selftouch is None:
            raise RuntimeError("use_selftouch=True but no selftouch model was loaded")

        fallback_joint = hand_jnt_pos
        selftouch_hand_jnt_pos = self._as_sequence(selftouch_hand_jnt_pos, fallback_joint)
        selftouch_hand_jnt_vel = self._as_sequence(selftouch_hand_jnt_vel, fallback_joint)
        selftouch_hand_jnt_trq = self._as_sequence(selftouch_hand_jnt_trq, fallback_joint)
        selftouch_hand_jnt_cmd_pos = self._as_sequence(
            selftouch_hand_jnt_cmd_pos,
            fallback_joint,
        )
        selftouch_tactile_index_tip = self._as_sequence(
            selftouch_tactile_index_tip,
            fallback_joint,
        )
        selftouch_tactile_thumb_tip = self._as_sequence(
            selftouch_tactile_thumb_tip,
            fallback_joint,
        )
        selftouch_tactile_middle_tip = self._as_sequence(
            selftouch_tactile_middle_tip,
            fallback_joint,
        )
        selftouch_tactile_ring_tip = self._as_sequence(
            selftouch_tactile_ring_tip,
            fallback_joint,
        )
        selftouch_combo = self._as_sequence(selftouch_combo, fallback_joint)
        selftouch_phase = self._as_sequence(selftouch_phase, fallback_joint)

        pos = (
            selftouch_hand_jnt_pos
            if selftouch_hand_jnt_pos is not None
            else self._expand_selftouch_joint(hand_jnt_pos)
        )
        vel_source = streams.get("hand_jnt_vel", torch.zeros_like(hand_jnt_pos))
        vel = (
            selftouch_hand_jnt_vel
            if selftouch_hand_jnt_vel is not None
            else self._expand_selftouch_joint(vel_source)
        )
        trq = (
            selftouch_hand_jnt_trq
            if selftouch_hand_jnt_trq is not None
            else torch.zeros_like(pos)
        )
        cmd = (
            selftouch_hand_jnt_cmd_pos
            if selftouch_hand_jnt_cmd_pos is not None
            else pos
        )
        if pos.shape[1] < 2:
            return {
                key: pos.new_zeros(pos.shape[0], pos.shape[1], self.tactile_dim)
                for key in self.selftouch_feature_keys
            }
        with torch.no_grad():
            out = self.selftouch(
                pos,
                vel,
                trq,
                cmd,
                tactile_index_tip=selftouch_tactile_index_tip,
                tactile_thumb_tip=selftouch_tactile_thumb_tip,
                tactile_middle_tip=selftouch_tactile_middle_tip,
                tactile_ring_tip=selftouch_tactile_ring_tip,
                selftouch_combo=selftouch_combo,
                selftouch_phase=selftouch_phase,
            )
        features = {}
        for key in self.selftouch_feature_keys:
            dict_key, tuple_idx = self.TACTILE_OUTPUT_MAP[key]
            if isinstance(out, dict):
                features[key] = out[dict_key]
            else:
                features[key] = out[tuple_idx]
        return features

    def _selftouch_teacher_features(self, streams, **selftouch_kwargs):
        hand_jnt_pos = self._stream_sequence(streams, "hand_jnt_pos")
        precomputed = self._precomputed_selftouch_teacher_features(
            streams,
            hand_jnt_pos,
        )
        if precomputed is not None:
            return precomputed
        raw_streams = {
            key: value
            for key, value in streams.items()
            if not key.startswith("selftouch_feature_")
            and not key.startswith("selftouch_teacher_feature_")
        }
        return self._selftouch_features(raw_streams, **selftouch_kwargs)

    def _is_temporal_tensor(self, value, reference_steps=None):
        if not torch.is_tensor(value):
            return False
        if value.dim() >= 3:
            return True
        return (
            value.dim() == 2
            and reference_steps is not None
            and int(value.shape[1]) == int(reference_steps)
        )

    def _slice_time(self, value, end, reference_steps=None):
        if self._is_temporal_tensor(value, reference_steps=reference_steps):
            return value[:, :end]
        return value

    def _slice_temporal_mapping(self, mapping, end, reference_steps=None):
        return {
            key: self._slice_time(value, end, reference_steps=reference_steps)
            for key, value in mapping.items()
        }

    def _slice_time_range(self, value, start, end, reference_steps=None):
        if self._is_temporal_tensor(value, reference_steps=reference_steps):
            return value[:, start:end]
        return value

    def _slice_temporal_range_mapping(self, mapping, start, end, reference_steps=None):
        return {
            key: self._slice_time_range(
                value,
                start,
                end,
                reference_steps=reference_steps,
            )
            for key, value in mapping.items()
        }

    def _tail_time(self, value, steps, reference_steps=None):
        if self._is_temporal_tensor(value, reference_steps=reference_steps):
            return value[:, -steps:]
        return value

    def _tail_temporal_mapping(self, mapping, steps, reference_steps=None):
        return {
            key: self._tail_time(value, steps, reference_steps=reference_steps)
            for key, value in mapping.items()
        }

    def _trajectory_condition_steps(self, streams):
        steps = int(self.param.get("trajectory_condition_steps", 0) or 0)
        if steps <= 0:
            return 0
        reference = self._stream_sequence(streams, self.output_keys[0])
        return max(1, min(steps, int(reference.shape[1])))

    def _chunk_obs_steps(self, streams):
        reference = self._stream_sequence(streams, self.output_keys[0])
        return max(1, min(self.act_obs_steps, int(reference.shape[1])))

    def _build_obs(self, streams, noise=None, drop_last=True, **selftouch_kwargs):
        tactile_std = _noise_value(noise, "tactile_noise", 0.0)
        joint_std = _noise_value(noise, "joint_noise", 0.0)

        parts = []
        for key in self.tactile_keys:
            seq = self._stream_sequence(streams, key)
            if drop_last:
                seq = seq[:, :-1]
            parts.append(_add_noise(seq, tactile_std))
        for key in self.joint_state_keys:
            seq = self._stream_sequence(streams, key)
            if drop_last:
                seq = seq[:, :-1]
            parts.append(_add_noise(seq, joint_std))

        st_features = self._selftouch_features(streams, **selftouch_kwargs)
        if st_features is not None:
            target_len = parts[0].shape[1]
            for key in self.selftouch_feature_keys:
                parts.append(
                    _fit_timesteps(st_features[key], target_len)
                    * self.selftouch_feature_scale
                )
        return torch.cat(parts, dim=-1)

    def _target_actions(self, streams):
        return torch.cat(
            [self._stream_sequence(streams, key)[:, 1:] for key in self.output_keys],
            dim=-1,
        )

    def _target_actions_from_range(self, streams, start, end):
        return torch.cat(
            [
                self._stream_sequence(streams, key)[:, start:end]
                for key in self.output_keys
            ],
            dim=-1,
        )

    def _split(self, actions):
        widths = [
            self.tactile_dim if key in self.tactile_keys else self.hand_dim
            for key in self.output_keys
        ]
        return dict(zip(self.output_keys, torch.split(actions, widths, dim=-1)))

    def _forward_chunk_loss(
        self,
        streams,
        data_found,
        loss_coef=None,
        compute_selftouch_teacher_loss=None,
        noise=None,
        **selftouch_kwargs,
    ):
        obs_steps = self._chunk_obs_steps(streams)
        reference = self._stream_sequence(streams, self.output_keys[0])
        total_steps = int(reference.shape[1])
        target_start = obs_steps
        target_end = min(total_steps, target_start + self.action_chunk_size)
        if target_end <= target_start:
            raise ValueError(
                "Chunked ACT needs at least one target timestep after "
                f"act_obs_steps={obs_steps}; got sequence length {total_steps}."
            )

        obs_streams = self._slice_temporal_mapping(
            streams,
            obs_steps,
            reference_steps=total_steps,
        )
        obs_selftouch_kwargs = self._slice_temporal_mapping(
            selftouch_kwargs,
            obs_steps,
            reference_steps=total_steps,
        )
        obs = self._build_obs(
            obs_streams,
            noise=noise,
            drop_last=False,
            **obs_selftouch_kwargs,
        )
        target_actions = self._target_actions_from_range(
            streams,
            target_start,
            target_end,
        )
        pred_actions, aux = self.policy(obs, target_actions)
        preds = self._split(pred_actions)

        mask = data_found[:, target_start:target_end]
        loss_coef = loss_coef or {}
        selftouch_teacher_coef = float(
            loss_coef.get(
                "selftouch_teacher",
                self.param.get("selftouch_teacher_loss_coef", 0.0),
            )
            or 0.0
        )
        component_losses = []
        total_loss = pred_actions.new_tensor(0.0)
        for key in self.output_keys:
            target = self._stream_sequence(streams, key)[:, target_start:target_end]
            loss = _masked_mse(preds[key], target, mask)
            component_losses.append(loss)
            total_loss = total_loss + loss * float(loss_coef.get(key, 1.0))
        for key in self.temporal_delta_keys:
            target = self._stream_sequence(streams, key)[:, target_start:target_end]
            anchor = self._stream_sequence(streams, key)[:, target_start - 1 : target_start]
            anchored_pred = torch.cat([anchor, preds[key]], dim=1)
            target_full = torch.cat([anchor, target], dim=1)
            pred_delta = anchored_pred[:, 1:] - anchored_pred[:, :-1]
            target_delta = target_full[:, 1:] - target_full[:, :-1]
            loss = _masked_mse(pred_delta, target_delta, mask)
            component_losses.append(loss)
            total_loss = total_loss + loss * float(
                loss_coef.get(f"{key}_delta", 1.0)
            )
        if self.use_selftouch and self.selftouch_teacher_loss_keys:
            if compute_selftouch_teacher_loss is None:
                compute_selftouch_teacher_loss = selftouch_teacher_coef != 0.0
            if compute_selftouch_teacher_loss:
                teacher_features = self._selftouch_teacher_features(
                    streams,
                    **selftouch_kwargs,
                )
                teacher_losses = []
                for key in self.selftouch_teacher_loss_keys:
                    if key not in preds or key not in teacher_features:
                        continue
                    teacher_seq = teacher_features[key].detach()
                    if teacher_seq.shape[1] < target_end:
                        teacher_seq = _fit_timesteps(teacher_seq, target_end)
                    teacher_target = teacher_seq[:, target_start:target_end]
                    teacher_losses.append(_masked_mse(preds[key], teacher_target, mask))
                if teacher_losses:
                    teacher_loss = torch.stack(teacher_losses).mean()
                else:
                    teacher_loss = total_loss * 0.0
            else:
                teacher_loss = total_loss * 0.0
            component_losses.append(teacher_loss)
            total_loss = total_loss + teacher_loss * selftouch_teacher_coef
        if "noise_pred" in aux:
            diffusion_loss = _masked_mse(aux["noise_pred"], aux["noise"], mask)
            total_loss = total_loss + diffusion_loss * float(
                self.param.get("diffusion_loss_coef", 0.1)
            )

        return (total_loss, *component_losses), preds

    def forward_loss(
        self,
        tactile_index_tip=None,
        tactile_thumb_tip=None,
        hand_jnt_pos=None,
        hand_jnt_vel=None,
        data_found=None,
        loss_coef=None,
        compute_selftouch_teacher_loss=None,
        cls_rate=None,
        noise=None,
        selftouch_hand_jnt_pos=None,
        selftouch_hand_jnt_vel=None,
        selftouch_hand_jnt_trq=None,
        selftouch_hand_jnt_cmd_pos=None,
        selftouch_tactile_index_tip=None,
        selftouch_tactile_thumb_tip=None,
        selftouch_tactile_middle_tip=None,
        selftouch_tactile_ring_tip=None,
        selftouch_combo=None,
        selftouch_phase=None,
        **streams,
    ):
        del cls_rate
        streams = self._motion_streams_from_args(
            tactile_index_tip=tactile_index_tip,
            tactile_thumb_tip=tactile_thumb_tip,
            hand_jnt_pos=hand_jnt_pos,
            hand_jnt_vel=hand_jnt_vel,
            **streams,
        )
        selftouch_kwargs = {
            "selftouch_hand_jnt_pos": selftouch_hand_jnt_pos,
            "selftouch_hand_jnt_vel": selftouch_hand_jnt_vel,
            "selftouch_hand_jnt_trq": selftouch_hand_jnt_trq,
            "selftouch_hand_jnt_cmd_pos": selftouch_hand_jnt_cmd_pos,
            "selftouch_tactile_index_tip": selftouch_tactile_index_tip,
            "selftouch_tactile_thumb_tip": selftouch_tactile_thumb_tip,
            "selftouch_tactile_middle_tip": selftouch_tactile_middle_tip,
            "selftouch_tactile_ring_tip": selftouch_tactile_ring_tip,
            "selftouch_combo": selftouch_combo,
            "selftouch_phase": selftouch_phase,
        }
        if self.uses_action_chunks():
            return self._forward_chunk_loss(
                streams,
                data_found,
                loss_coef=loss_coef,
                compute_selftouch_teacher_loss=compute_selftouch_teacher_loss,
                noise=noise,
                **selftouch_kwargs,
            )
        condition_steps = self._trajectory_condition_steps(streams)
        if condition_steps > 0:
            if self.arch != "act":
                raise ValueError("trajectory_condition_steps currently requires dexwild_arch: act")
            reference_steps = int(
                self._stream_sequence(streams, self.output_keys[0]).shape[1]
            )
            obs_streams = self._slice_temporal_mapping(
                streams,
                condition_steps,
                reference_steps=reference_steps,
            )
            obs_selftouch_kwargs = self._slice_temporal_mapping(
                selftouch_kwargs,
                condition_steps,
                reference_steps=reference_steps,
            )
            obs = self._build_obs(
                obs_streams,
                noise=noise,
                drop_last=False,
                **obs_selftouch_kwargs,
            )
        else:
            obs = self._build_obs(
                streams,
                noise=noise,
                drop_last=True,
                **selftouch_kwargs,
            )
        target_actions = self._target_actions(streams)
        pred_actions, aux = self.policy(obs, target_actions)
        preds = self._split(pred_actions)

        mask = data_found[:, 1:]
        loss_coef = loss_coef or {}
        selftouch_teacher_coef = float(
            loss_coef.get(
                "selftouch_teacher",
                self.param.get("selftouch_teacher_loss_coef", 0.0),
            )
            or 0.0
        )
        component_losses = []
        total_loss = pred_actions.new_tensor(0.0)
        for key in self.output_keys:
            target = self._stream_sequence(streams, key)[:, 1:]
            loss = _masked_mse(preds[key], target, mask)
            component_losses.append(loss)
            total_loss = total_loss + loss * float(loss_coef.get(key, 1.0))
        for key in self.temporal_delta_keys:
            target_full = self._stream_sequence(streams, key)
            anchored_pred = torch.cat([target_full[:, :1], preds[key]], dim=1)
            pred_delta = anchored_pred[:, 1:] - anchored_pred[:, :-1]
            target_delta = target_full[:, 1:] - target_full[:, :-1]
            loss = _masked_mse(pred_delta, target_delta, mask)
            component_losses.append(loss)
            total_loss = total_loss + loss * float(
                loss_coef.get(f"{key}_delta", 1.0)
            )
        if self.use_selftouch and self.selftouch_teacher_loss_keys:
            if compute_selftouch_teacher_loss is None:
                compute_selftouch_teacher_loss = selftouch_teacher_coef != 0.0
            if compute_selftouch_teacher_loss:
                teacher_features = self._selftouch_teacher_features(
                    streams,
                    **selftouch_kwargs,
                )
                teacher_losses = []
                for key in self.selftouch_teacher_loss_keys:
                    if key not in preds or key not in teacher_features:
                        continue
                    teacher_target = _fit_timesteps(
                        teacher_features[key].detach(),
                        preds[key].shape[1],
                    )
                    teacher_losses.append(_masked_mse(preds[key], teacher_target, mask))
                if teacher_losses:
                    teacher_loss = torch.stack(teacher_losses).mean()
                else:
                    teacher_loss = total_loss * 0.0
            else:
                teacher_loss = total_loss * 0.0
            component_losses.append(teacher_loss)
            total_loss = total_loss + teacher_loss * selftouch_teacher_coef
        if "noise_pred" in aux:
            diffusion_loss = _masked_mse(aux["noise_pred"], aux["noise"], mask)
            total_loss = total_loss + diffusion_loss * float(
                self.param.get("diffusion_loss_coef", 0.1)
            )

        return (total_loss, *component_losses), preds

    def forward_action_chunk(self, output_steps=None, **kwargs):
        selftouch_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key.startswith("selftouch_")
        }
        streams = self._motion_streams_from_args(**kwargs)
        obs_steps = self._chunk_obs_steps(streams)
        reference_steps = int(self._stream_sequence(streams, self.output_keys[0]).shape[1])
        streams = self._tail_temporal_mapping(
            streams,
            obs_steps,
            reference_steps=reference_steps,
        )
        selftouch_kwargs = self._tail_temporal_mapping(
            selftouch_kwargs,
            obs_steps,
            reference_steps=reference_steps,
        )
        obs = self._build_obs(streams, drop_last=False, **selftouch_kwargs)
        output_steps = int(output_steps or self.action_chunk_size or obs.shape[1])
        pred_actions, _ = self.policy(obs, None, output_steps=output_steps)
        return self._split(pred_actions)

    def forward_sequence(self, output_steps=None, **kwargs):
        selftouch_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key.startswith("selftouch_")
        }
        streams = self._motion_streams_from_args(**kwargs)
        condition_steps = self._trajectory_condition_steps(streams)
        if condition_steps > 0:
            if self.arch != "act":
                raise ValueError("trajectory_condition_steps currently requires dexwild_arch: act")
            reference_steps = int(
                self._stream_sequence(streams, self.output_keys[0]).shape[1]
            )
            streams = self._slice_temporal_mapping(
                streams,
                condition_steps,
                reference_steps=reference_steps,
            )
            selftouch_kwargs = self._slice_temporal_mapping(
                selftouch_kwargs,
                condition_steps,
                reference_steps=reference_steps,
            )
            obs = self._build_obs(streams, drop_last=False, **selftouch_kwargs)
            if output_steps is None:
                output_steps = max(1, int(self.param.get("sequence_length", obs.shape[1])) - 1)
        else:
            obs = self._build_obs(streams, drop_last=True, **selftouch_kwargs)
        pred_actions, _ = self.policy(obs, None, output_steps=output_steps)
        return self._split(pred_actions)

    def forward(
        self,
        tactile_index_tip_t=None,
        tactile_thumb_tip_t=None,
        hand_jnt_pos_t=None,
        hand_jnt_vel_t=None,
        **kwargs,
    ):
        selftouch_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key.startswith("selftouch_")
        }
        legacy_streams = {
            "tactile_index_tip": tactile_index_tip_t,
            "tactile_thumb_tip": tactile_thumb_tip_t,
            "hand_jnt_pos": hand_jnt_pos_t,
            "hand_jnt_vel": hand_jnt_vel_t,
        }
        for key, value in legacy_streams.items():
            if value is not None:
                if key in kwargs and kwargs[key] is not None:
                    raise ValueError(f"Motion stream '{key}' was provided twice")
                kwargs[key] = value

        streams = self._motion_streams_from_args(**kwargs)
        if self.uses_action_chunks():
            chunk_preds = self.forward_action_chunk(
                output_steps=1,
                **streams,
                **selftouch_kwargs,
            )
            return {key: value[:, 0] for key, value in chunk_preds.items()}
        parts = []
        for key in self.tactile_keys:
            parts.append(self._stream_sequence(streams, key))
        for key in self.joint_state_keys:
            parts.append(self._stream_sequence(streams, key))
        obs = torch.cat(parts, dim=-1)

        st_features = self._selftouch_features(streams, **selftouch_kwargs)
        if st_features is not None:
            target_len = obs.shape[1]
            obs = torch.cat(
                [obs]
                + [
                    _fit_timesteps(st_features[key], target_len)
                    for key in self.selftouch_feature_keys
                ],
                dim=-1,
            )
        pred_actions, _ = self.policy(obs, None)
        return {key: value[:, -1] for key, value in self._split(pred_actions).items()}
