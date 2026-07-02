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


def _masked_mse(pred, target, mask):
    loss = F.mse_loss(pred, target, reduction="none")
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(loss).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


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

    def forward(self, obs, target_actions=None):
        batch_size, steps, _ = obs.shape
        if steps > self.pos.shape[1]:
            raise ValueError(f"Sequence length {steps} exceeds max_seq_len")
        src = self.obs_proj(obs) + self.pos[:, :steps]
        mask = _causal_mask(steps, obs.device) if self.causal else None
        memory = self.encoder(src, mask=mask)
        query_idx = torch.arange(steps, device=obs.device)
        tgt = self.query(query_idx)[None].expand(batch_size, -1, -1)
        pred = self.decoder(tgt, memory, tgt_mask=mask)
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
    def __init__(self, param, selftouch=None):
        super().__init__()
        self.param = param
        self.hand_dim = int(param["hand_dim"])
        self.tactile_dim = int(param["tactile_dim"])
        self.use_selftouch = bool(param.get("use_selftouch", False))
        self.use_velocity_input = bool(param.get("use_velocity_input", False))
        self.selftouch = selftouch
        self.selftouch_input_dim = int(param.get("selftouch_input_dim", 16))
        self.selftouch_joint_indices = list(
            param.get("selftouch_joint_indices", [0, 1, 2, 3, 12, 13, 14, 15])
        )

        obs_dim = 2 * self.tactile_dim + self.hand_dim
        if self.use_velocity_input:
            obs_dim += self.hand_dim
        if self.use_selftouch:
            obs_dim += 2 * self.tactile_dim
        self.action_dim = 2 * self.tactile_dim + 2 * self.hand_dim

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

    def _expand_selftouch_joint(self, x):
        if x.shape[-1] == self.selftouch_input_dim:
            return x
        out = x.new_zeros(*x.shape[:-1], self.selftouch_input_dim)
        indices = self.selftouch_joint_indices
        if len(indices) != x.shape[-1] or max(indices) >= self.selftouch_input_dim:
            indices = list(range(min(x.shape[-1], self.selftouch_input_dim)))
        index = torch.tensor(indices, dtype=torch.long, device=x.device)
        return out.index_copy(-1, index, x[..., : len(indices)])

    def _selftouch_features(
        self,
        hand_jnt_pos,
        hand_jnt_vel,
        selftouch_hand_jnt_pos=None,
        selftouch_hand_jnt_vel=None,
        selftouch_hand_jnt_trq=None,
        selftouch_hand_jnt_cmd_pos=None,
    ):
        if not self.use_selftouch:
            return None
        if self.selftouch is None:
            raise RuntimeError("use_selftouch=True but no selftouch model was loaded")

        pos = (
            selftouch_hand_jnt_pos
            if selftouch_hand_jnt_pos is not None
            else self._expand_selftouch_joint(hand_jnt_pos)
        )
        vel = (
            selftouch_hand_jnt_vel
            if selftouch_hand_jnt_vel is not None
            else self._expand_selftouch_joint(hand_jnt_vel)
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
        with torch.no_grad():
            idx_self, thumb_self, *_ = self.selftouch(pos, vel, trq, cmd)
        return idx_self, thumb_self

    def _build_obs(
        self,
        tactile_index_tip,
        tactile_thumb_tip,
        hand_jnt_pos,
        hand_jnt_vel,
        noise=None,
        selftouch_hand_jnt_pos=None,
        selftouch_hand_jnt_vel=None,
        selftouch_hand_jnt_trq=None,
        selftouch_hand_jnt_cmd_pos=None,
    ):
        tactile_std = _noise_value(noise, "tactile_noise", 0.0)
        joint_std = _noise_value(noise, "joint_noise", 0.0)

        idx = _add_noise(tactile_index_tip[:, :-1], tactile_std)
        thumb = _add_noise(tactile_thumb_tip[:, :-1], tactile_std)
        pos = _add_noise(hand_jnt_pos[:, :-1], joint_std)
        vel = _add_noise(hand_jnt_vel[:, :-1], joint_std)

        parts = [idx, thumb, pos]
        if self.use_velocity_input:
            parts.append(vel)
        st_features = self._selftouch_features(
            hand_jnt_pos,
            hand_jnt_vel,
            selftouch_hand_jnt_pos,
            selftouch_hand_jnt_vel,
            selftouch_hand_jnt_trq,
            selftouch_hand_jnt_cmd_pos,
        )
        if st_features is not None:
            parts.extend([st_features[0][:, :-1], st_features[1][:, :-1]])
        return torch.cat(parts, dim=-1)

    def _target_actions(self, tactile_index_tip, tactile_thumb_tip, hand_jnt_pos, hand_jnt_vel):
        return torch.cat(
            [
                tactile_index_tip[:, 1:],
                tactile_thumb_tip[:, 1:],
                hand_jnt_pos[:, 1:],
                hand_jnt_vel[:, 1:],
            ],
            dim=-1,
        )

    def _split(self, actions):
        return torch.split(
            actions,
            [self.tactile_dim, self.tactile_dim, self.hand_dim, self.hand_dim],
            dim=-1,
        )

    def forward_loss(
        self,
        tactile_index_tip,
        tactile_thumb_tip,
        hand_jnt_pos,
        hand_jnt_vel,
        data_found,
        loss_coef,
        cls_rate=None,
        noise=None,
        selftouch_hand_jnt_pos=None,
        selftouch_hand_jnt_vel=None,
        selftouch_hand_jnt_trq=None,
        selftouch_hand_jnt_cmd_pos=None,
        **_unused,
    ):
        obs = self._build_obs(
            tactile_index_tip,
            tactile_thumb_tip,
            hand_jnt_pos,
            hand_jnt_vel,
            noise=noise,
            selftouch_hand_jnt_pos=selftouch_hand_jnt_pos,
            selftouch_hand_jnt_vel=selftouch_hand_jnt_vel,
            selftouch_hand_jnt_trq=selftouch_hand_jnt_trq,
            selftouch_hand_jnt_cmd_pos=selftouch_hand_jnt_cmd_pos,
        )
        target_actions = self._target_actions(
            tactile_index_tip, tactile_thumb_tip, hand_jnt_pos, hand_jnt_vel
        )
        pred_actions, aux = self.policy(obs, target_actions)

        idx_preds, thumb_preds, pos_preds, vel_preds = self._split(pred_actions)
        mask = data_found[:, 1:]
        loss_idx = _masked_mse(idx_preds, tactile_index_tip[:, 1:], mask)
        loss_thumb = _masked_mse(thumb_preds, tactile_thumb_tip[:, 1:], mask)
        loss_pos = _masked_mse(pos_preds, hand_jnt_pos[:, 1:], mask)
        loss_vel = _masked_mse(vel_preds, hand_jnt_vel[:, 1:], mask)

        total_loss = (
            loss_idx * loss_coef.get("tactile_index_tip", 1.0)
            + loss_thumb * loss_coef.get("tactile_thumb_tip", 1.0)
            + loss_pos * loss_coef.get("hand_jnt_pos", 1.0)
            + loss_vel * loss_coef.get("hand_jnt_vel", 1.0)
        )
        if "noise_pred" in aux:
            diffusion_loss = _masked_mse(aux["noise_pred"], aux["noise"], mask)
            total_loss = total_loss + diffusion_loss * float(
                self.param.get("diffusion_loss_coef", 0.1)
            )

        return (
            total_loss,
            loss_idx,
            loss_thumb,
            loss_pos,
            loss_vel,
        ), (idx_preds, thumb_preds, pos_preds, vel_preds)

    def forward(
        self,
        tactile_index_tip_t,
        tactile_thumb_tip_t,
        hand_jnt_pos_t,
        hand_jnt_vel_t=None,
        **kwargs,
    ):
        if tactile_index_tip_t.dim() == 2:
            tactile_index_tip_t = tactile_index_tip_t[:, None]
            tactile_thumb_tip_t = tactile_thumb_tip_t[:, None]
            hand_jnt_pos_t = hand_jnt_pos_t[:, None]
            if hand_jnt_vel_t is not None:
                hand_jnt_vel_t = hand_jnt_vel_t[:, None]
        if hand_jnt_vel_t is None:
            hand_jnt_vel_t = torch.zeros_like(hand_jnt_pos_t)
        obs = torch.cat(
            [tactile_index_tip_t, tactile_thumb_tip_t, hand_jnt_pos_t],
            dim=-1,
        )
        if self.use_velocity_input:
            obs = torch.cat([obs, hand_jnt_vel_t], dim=-1)
        pred_actions, _ = self.policy(obs, None)
        return self._split(pred_actions[:, -1])

