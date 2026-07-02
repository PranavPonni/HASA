"""
selftouch_contrastive_transformer.py
Contrastive transformer encoder + causal transformer decoder tactile head.

Input:  Joint position(t) [hand_dim]
Output: Predicted Self-touch Tactile(t+1) for index_tip + thumb_tip + middle_tip [tactile_dim * 3]
        + Contrastive / classification objectives on CLS embedding

Encoder: causal transformer → CLS embedding for contrastive/classification.
Decoder: causal transformer applied to per-timestep encoder output → tactile prediction.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from selftouch_loss_utils import active_tactile_loss, supervised_contrastive_loss
from selftouch_feature_utils import TactileHead, build_joint_features, joint_feature_dim


class SelfTouchContrastiveTransformer(nn.Module):
    """Contrastive encoder + causal transformer decoder for tactile prediction."""

    def __init__(self, param: dict):
        super().__init__()
        hand_dim     = param.get("hand_dim",       16)
        use_joint_pos_only = bool(param.get("use_joint_pos_only", True))
        use_derived = bool(param.get("use_derived_features", False))
        input_dim = (
            joint_feature_dim(hand_dim, use_derived, use_joint_pos_only)
            if use_joint_pos_only or use_derived
            else param.get("input_dim", hand_dim * 4)
        )
        num_classes  = param.get("num_classes",    3)
        tactile_dim  = param.get("tactile_dim",    90)      # per-finger
        d_model      = param.get("d_model",        128)
        nhead        = param.get("nhead",           4)
        enc_layers   = param.get("encoder_layers", 2)
        dec_layers   = param.get("decoder_layers", 2)
        proj_dim     = param.get("projection_dim", 64)
        dropout      = param.get("dropout",        0.1)
        max_seq_len  = param.get("max_seq_len",    512)
        self.tactile_loss_name = param.get("tactile_loss", "smooth_l1")
        self.tactile_dim = tactile_dim
        self.input_dim = int(input_dim)
        self.use_joint_pos_only = use_joint_pos_only
        self.use_derived_features = use_derived

        # ── Encoder (causal) ───────────────────────────────────────────────────
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
        self.norm    = nn.LayerNorm(d_model)

        # ── Contrastive heads ─────────────────────────────────────────────────
        self.projector  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.classifier = nn.Linear(d_model, num_classes)

        # ── Causal decoder: index + thumb + middle ───────────────────────────
        dec_layer = nn.TransformerEncoderLayer(    # auto-reg: encoder with causal mask
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_layers)
        self.dec_norm = nn.LayerNorm(d_model)
        self.to_idx = TactileHead(d_model, self.input_dim, tactile_dim)
        self.to_thumb = TactileHead(d_model, self.input_dim, tactile_dim)
        self.to_middle = TactileHead(d_model, self.input_dim, tactile_dim)
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
        """Upper-triangular mask (True = ignore) of shape (T, T)."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def encode_seq(self, x: torch.Tensor) -> torch.Tensor:
        """Causal encoder output (B, T, d_model)."""
        _, t, _ = x.shape
        h = self.input_proj(x) + self.pos_encoding[:t].unsqueeze(0)
        h = self.encoder(h, mask=self._causal_mask(t, x.device))
        return self.norm(h)

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

    def encode(
        self,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_vel: torch.Tensor = None,
        hand_jnt_trq: torch.Tensor = None,
        hand_jnt_cmd_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """CLS embedding (B, d_model) for PCA/contrastive."""
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
        cls   = h_seq.mean(dim=1)
        z     = F.normalize(self.projector(cls), dim=-1)
        logits = self.classifier(cls)

        # Causal decoder on encoder output
        T = x.shape[1]
        causal_mask = self._causal_mask(T, x.device)
        dec_out     = self.decoder(h_seq, mask=causal_mask)
        dec_out     = self.dec_norm(dec_out)                       # (B, T-1, d_model)
        return {
            "embedding":   cls,
            "projection":  z,
            "logits":      logits,
            "idx_pred":    self._activate_tactile(self.to_idx(dec_out, x)),
            "thumb_pred":  self._activate_tactile(self.to_thumb(dec_out, x)),
            "middle_pred": self._activate_tactile(self.to_middle(dec_out, x)),
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
        out = self.forward(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)

        contrastive_loss = supervised_contrastive_loss(out["projection"], labels, (loss_coef or {}).get("contrastive_temperature", 0.1))
        cls_loss = F.cross_entropy(out["logits"], labels)

        # Targets at t+1, predicted from joint state at t.
        coef = loss_coef or {}
        cfg = {**coef, "tactile_loss": coef.get("tactile_loss", self.tactile_loss_name)}
        loss_idx = active_tactile_loss(out["idx_pred"], tactile_index_tip[:, 1:], cfg)
        loss_thb = active_tactile_loss(out["thumb_pred"], tactile_thumb_tip[:, 1:], cfg)
        loss_mid = active_tactile_loss(out["middle_pred"], tactile_middle_tip[:, 1:], cfg)

        total = (
            coef.get("contrastive", 1.0) * contrastive_loss
            + coef.get("classification", 1.0) * cls_loss
            + coef.get("tactile_index_tip", 1.0) * loss_idx
            + coef.get("tactile_thumb_tip", 1.0) * loss_thb
            + coef.get("tactile_middle_tip", 1.0) * loss_mid
        )

        losses = {
            "total":          total,
            "contrastive":    contrastive_loss,
            "classification": cls_loss,
            "tactile_index":  loss_idx,
            "tactile_thumb":  loss_thb,
            "tactile_middle": loss_mid,
        }
        preds = {
            "index": out["idx_pred"],
            "thumb": out["thumb_pred"],
            "middle": out["middle_pred"],
        }
        return losses, preds
