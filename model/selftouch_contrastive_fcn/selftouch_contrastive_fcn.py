"""
selftouch_contrastive_fcn.py
Contrastive transformer encoder + per-timestep FCN tactile head.

Input:  Joint position(t) [hand_dim]
Output: Predicted Self-touch Tactile(t+1) for index_tip + thumb_tip + middle_tip [tactile_dim * 3]
        + Contrastive / classification objectives on CLS embedding

The encoder is causal so each tactile prediction only sees joint state up to that step.
The FCN tactile head is applied per-timestep on encoder outputs.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from selftouch_loss_utils import active_tactile_loss, supervised_contrastive_loss
from selftouch_feature_utils import TactileHead, build_joint_features, joint_feature_dim


class SelfTouchContrastiveFCN(nn.Module):
    """Contrastive encoder + FCN tactile predictor."""

    def __init__(self, param: dict):
        super().__init__()
        hand_dim      = param.get("hand_dim",       16)
        use_joint_pos_only = bool(param.get("use_joint_pos_only", True))
        use_derived = bool(param.get("use_derived_features", False))
        input_dim = (
            joint_feature_dim(hand_dim, use_derived, use_joint_pos_only)
            if use_joint_pos_only or use_derived
            else param.get("input_dim", hand_dim * 4)
        )
        num_classes   = param.get("num_classes",    3)
        tactile_dim   = param.get("tactile_dim",    90)      # per-finger
        d_model       = param.get("d_model",        128)
        nhead         = param.get("nhead",          4)
        enc_layers    = param.get("encoder_layers", 2)
        proj_dim      = param.get("projection_dim", 64)
        dropout       = param.get("dropout",        0.1)
        max_seq_len   = param.get("max_seq_len",    512)
        rec_dim       = param.get("rec_dim",        256)     # FCN hidden
        self.tactile_dim = tactile_dim
        self.input_dim = int(input_dim)
        self.use_joint_pos_only = use_joint_pos_only
        self.use_derived_features = use_derived

        # ── Encoder (causal) ──────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "pos_encoding", self._positional_encoding(max_seq_len, d_model), persistent=False
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.norm = nn.LayerNorm(d_model)

        # ── Contrastive heads (operate on CLS = mean pool) ───────────────────
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.classifier = nn.Linear(d_model, num_classes)

        # ── FCN tactile trunk + per-finger heads ─────────────────────────────
        self.tactile_head = nn.Sequential(
            nn.Linear(d_model, rec_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_idx = TactileHead(rec_dim, self.input_dim, tactile_dim)
        self.to_thumb = TactileHead(rec_dim, self.input_dim, tactile_dim)
        self.to_middle = TactileHead(rec_dim, self.input_dim, tactile_dim)
        self._init_tactile_bias()

    @staticmethod
    def _positional_encoding(length: int, dim: int):
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _build_input(
        self,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_vel: torch.Tensor = None,
        hand_jnt_trq: torch.Tensor = None,
        hand_jnt_cmd_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """Build joint state at t for predicting tactile at t+1."""
        return build_joint_features(
            hand_jnt_pos,
            hand_jnt_vel,
            hand_jnt_trq,
            hand_jnt_cmd_pos,
            next_step=True,
            use_derived_features=self.use_derived_features,
            use_joint_pos_only=self.use_joint_pos_only,
        )

    def encode_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full sequence, returning (B, T, d_model)."""
        _, t, _ = x.shape
        h = self.input_proj(x) + self.pos_encoding[:t].unsqueeze(0)
        h = self.encoder(h, mask=self._causal_mask(t, x.device))
        return self.norm(h)

    def encode(
        self,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_vel: torch.Tensor = None,
        hand_jnt_trq: torch.Tensor = None,
        hand_jnt_cmd_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return CLS embedding (mean pool) of shape (B, d_model)."""
        x = self._build_input(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)
        return self.encode_seq(x).mean(dim=1)

    def _activate_tactile(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _init_tactile_bias(self):
        return

    def forward(
        self,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_vel: torch.Tensor = None,
        hand_jnt_trq: torch.Tensor = None,
        hand_jnt_cmd_pos: torch.Tensor = None,
    ):
        x = self._build_input(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)  # (B, T-1, input_dim)
        h_seq = self.encode_seq(x)                              # (B, T-1, d_model)
        cls   = h_seq.mean(dim=1)                              # (B, d_model)
        z     = F.normalize(self.projector(cls), dim=-1)
        logits = self.classifier(cls)
        tactile_h = self.tactile_head(h_seq)
        return {
            "embedding":   cls,
            "projection":  z,
            "logits":      logits,
            "idx_pred":    self._activate_tactile(self.to_idx(tactile_h, x)),
            "thumb_pred":  self._activate_tactile(self.to_thumb(tactile_h, x)),
            "middle_pred": self._activate_tactile(self.to_middle(tactile_h, x)),
        }

    def forward_loss(
        self,
        tactile_index_tip: torch.Tensor,
        tactile_thumb_tip: torch.Tensor,
        tactile_middle_tip: torch.Tensor,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_cmd_pos: torch.Tensor = None,
        labels: torch.Tensor = None,
        hand_jnt_vel: torch.Tensor = None,
        hand_jnt_trq: torch.Tensor = None,
        tactile_ring_tip=None,
        data_found=None,
        loss_coef: dict = None,
    ):
        """
        tactile_*: (B, T, 90)
        hand_jnt_*: (B, T, hand_dim)
        labels: (B,)
        """
        out = self.forward(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)

        contrastive_loss = supervised_contrastive_loss(out["projection"], labels, (loss_coef or {}).get("contrastive_temperature", 0.1))
        cls_loss = F.cross_entropy(out["logits"], labels)

        # Targets at t+1, predicted from joint state at t.
        coef = loss_coef or {}
        loss_idx = active_tactile_loss(out["idx_pred"], tactile_index_tip[:, 1:], coef)
        loss_thb = active_tactile_loss(out["thumb_pred"], tactile_thumb_tip[:, 1:], coef)
        loss_mid = active_tactile_loss(out["middle_pred"], tactile_middle_tip[:, 1:], coef)

        total = (
            coef.get("contrastive", 1.0) * contrastive_loss
            + coef.get("classification", 1.0) * cls_loss
            + coef.get("tactile_index_tip", 1.0) * loss_idx
            + coef.get("tactile_thumb_tip", 1.0) * loss_thb
            + coef.get("tactile_middle_tip", 1.0) * loss_mid
        )

        losses = {
            "total": total,
            "contrastive": contrastive_loss,
            "classification": cls_loss,
            "tactile_index": loss_idx,
            "tactile_thumb": loss_thb,
            "tactile_middle": loss_mid,
        }
        preds = {
            "index": out["idx_pred"],
            "thumb": out["thumb_pred"],
            "middle": out["middle_pred"],
        }
        return losses, preds

import math
import torch
from torch import nn
import torch.nn.functional as F
from selftouch_loss_utils import active_tactile_loss, supervised_contrastive_loss
