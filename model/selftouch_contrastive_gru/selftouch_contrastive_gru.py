"""
selftouch_contrastive_gru.py
Contrastive causal-GRU encoder + causal GRU decoder for tactile prediction.

Input:  Joint position(t) [hand_dim]
Output: Predicted Self-touch Tactile(t+1) for index_tip + thumb_tip + middle_tip [tactile_dim * 3]
        + Contrastive / classification objectives on sequence-level embedding

Encoder: unidirectional GRU over past/current state → mean pooled hidden → CLS for contrastive.
Decoder: Causal single-direction GRU applied step-by-step → per-timestep tactile prediction.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from selftouch_loss_utils import active_tactile_loss, supervised_contrastive_loss
from selftouch_feature_utils import TactileHead, build_joint_features, joint_feature_dim


class SelfTouchContrastiveGRU(nn.Module):
    """Contrastive BiGRU encoder + causal GRU decoder for tactile prediction."""

    def __init__(self, param: dict):
        super().__init__()
        hand_dim    = param.get("hand_dim",       16)
        use_joint_pos_only = bool(param.get("use_joint_pos_only", True))
        use_derived = bool(param.get("use_derived_features", False))
        input_dim = (
            joint_feature_dim(hand_dim, use_derived, use_joint_pos_only)
            if use_joint_pos_only or use_derived
            else param.get("input_dim", hand_dim * 4)
        )
        num_classes = param.get("num_classes",    3)
        tactile_dim = param.get("tactile_dim",    90)     # per-finger
        d_model     = param.get("d_model",        256)
        enc_layers  = param.get("encoder_layers", 2)
        dec_layers  = param.get("decoder_layers", 2)
        proj_dim    = param.get("projection_dim", 128)
        dropout     = param.get("dropout",        0.1)

        self.tactile_loss_name = param.get("tactile_loss", "smooth_l1")
        self.tactile_dim  = tactile_dim
        self.input_dim = int(input_dim)
        self.use_joint_pos_only = use_joint_pos_only
        self.use_derived_features = use_derived

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Linear(input_dim, d_model)

        # ── Causal GRU encoder ───────────────────────────────────────────────
        self.encoder = nn.GRU(
            input_size=d_model, hidden_size=d_model,
            num_layers=enc_layers, batch_first=True,
            bidirectional=False, dropout=dropout if enc_layers > 1 else 0.0,
        )
        self.enc_norm = nn.LayerNorm(d_model)

        # ── Contrastive projection + classifier ──────────────────────────────
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.classifier = nn.Linear(d_model, num_classes)

        # ── Causal GRU decoder ───────────────────────────────────────────────
        self.decoder = nn.GRU(
            input_size=d_model, hidden_size=d_model,
            num_layers=dec_layers, batch_first=True,
            bidirectional=False, dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.dec_norm    = nn.LayerNorm(d_model)
        self.to_idx = TactileHead(d_model, self.input_dim, tactile_dim)
        self.to_thumb = TactileHead(d_model, self.input_dim, tactile_dim)
        self.to_middle = TactileHead(d_model, self.input_dim, tactile_dim)
        self._init_tactile_bias()

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

    def _activate_tactile(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _init_tactile_bias(self):
        return

    def encode(
        self,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_vel: torch.Tensor = None,
        hand_jnt_trq: torch.Tensor = None,
        hand_jnt_cmd_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """CLS embedding (B, d_model) for PCA/contrastive."""
        x = self._build_input(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)
        h = self.input_proj(x)
        enc_out, _ = self.encoder(h)           # (B, T-1, d_model)
        enc_out = self.enc_norm(enc_out)
        return enc_out.mean(dim=1)             # mean-pool → (B, d_model)

    def forward(
        self,
        hand_jnt_pos: torch.Tensor,
        hand_jnt_vel: torch.Tensor = None,
        hand_jnt_trq: torch.Tensor = None,
        hand_jnt_cmd_pos: torch.Tensor = None,
    ):
        x = self._build_input(hand_jnt_pos, hand_jnt_vel, hand_jnt_trq, hand_jnt_cmd_pos)   # (B, T-1, input_dim)
        h = self.input_proj(x)                                   # (B, T-1, d_model)

        # Causal encoder
        enc_out, _ = self.encoder(h)     # (B, T-1, d_model)
        enc_out = self.enc_norm(enc_out)

        # Contrastive heads (sequence-level)
        cls    = enc_out.mean(dim=1)              # (B, d_model)
        z      = F.normalize(self.projector(cls), dim=-1)
        logits = self.classifier(cls)

        # Causal decoder (on encoder output, unidirectional GRU)
        dec_out, _ = self.decoder(enc_out)        # (B, T-1, d_model)
        dec_out = self.dec_norm(dec_out)

        idx_pred = self._activate_tactile(self.to_idx(dec_out, x))
        thumb_pred = self._activate_tactile(self.to_thumb(dec_out, x))
        middle_pred = self._activate_tactile(self.to_middle(dec_out, x))

        return {
            "embedding":  cls,
            "projection": z,
            "logits":     logits,
            "idx_pred":   idx_pred,
            "thumb_pred": thumb_pred,
            "middle_pred": middle_pred,
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
