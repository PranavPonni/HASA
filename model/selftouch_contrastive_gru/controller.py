"""
controller.py for selftouch_contrastive_gru
Training + testing controller.  Training uses:
  - Supervised contrastive loss (SupCon) on CLS embeddings
  - Cross-entropy classification loss
  - Per-finger MSE tactile prediction loss (next-timestep)
"""
import os
import sys
import csv
import glob
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from controller_base import AbstractController

from selftouch_contrastive_gru import SelfTouchContrastiveGRU
from data_loader import CustomDataLoader
from selftouch_plot_utils import (
    active_loss_coef,
    draw_tactile_prediction_profile,
    included_fingers_from_combinations,
    plot_tactile_temporal_profiles,
    set_shared_y_limits_from_lines,
    TACTILE_PLOT_TITLE,
)
from training_speed_utils import (
    EarlyStopper,
    apply_lr_schedule,
    amp_enabled,
    autocast,
    make_grad_scaler,
    maybe_watch_model,
    should_run_period,
    wandb_init_kwargs,
    wandb_log_heartbeat,
)

FINGER_NAMES  = ["index", "thumb", "middle", "ring"]
FINGER_KEYS   = ["tactile_index_tip", "tactile_thumb_tip", "tactile_middle_tip", "tactile_ring_tip"]
FINGER_COLORS = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]
TAXELS = 90


def _pca_2d(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2 or x.shape[1] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    centered = x - x.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return (centered @ vh[:2].T).astype(np.float32)
    except np.linalg.LinAlgError:
        return centered[:, :2].astype(np.float32) if centered.shape[1] >= 2 \
            else np.zeros((x.shape[0], 2), dtype=np.float32)


def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (SupCon, Khosla et al. 2020)."""
    device = features.device
    bs = features.shape[0]
    # Force fp32: torch.matmul inside autocast downcasts fp32 inputs to fp16,
    # so masked_fill(-1e9) overflows fp16 range even after features.float().
    with autocast(False):
        features = nn.functional.normalize(features.float(), dim=-1)
        # compute similarity matrix
        sim = torch.matmul(features, features.T) / temperature  # (B, B)
        mask_self = torch.eye(bs, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask_self, -1e9)

        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = pos_mask * (~mask_self).float()

        exp_sim = torch.exp(sim)
        exp_sim = exp_sim * (~mask_self).float()
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        positives = pos_mask.sum(dim=1)
        valid = positives > 0
        if not torch.any(valid):
            return torch.tensor(0.0, device=device)
        loss = -(pos_mask * log_prob).sum(dim=1)[valid] / positives[valid]
        return loss.mean()


class RNN_controller(AbstractController):
    """Training + testing controller for selftouch_contrastive_gru."""

    def __init__(self, model_param, mode_param, dataset_param, config_param):
        super().__init__(model_param, mode_param, dataset_param, config_param)

    # ── train ─────────────────────────────────────────────────────────────────
    def train_controller(self, sweep: bool = False):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("Device:", self.device)

        batch_size      = self.mode_param["batch_size"]
        total_epoch     = self.mode_param["num_epochs"]
        lr              = self.mode_param["lr"]
        weight_decay    = self.mode_param.get("weight_decay", 1e-4)
        model_save_iter = self.mode_param["model_save_iter"]
        eval_every      = self.mode_param.get("eval_every", 1)
        plot_every      = self.mode_param.get("plot_every", eval_every)
        self.grad_clip_norm = self.mode_param.get("grad_clip_norm", 1.0)
        combinations    = self.dataset_param.get("combinations", [])
        if combinations:
            self.model_param["num_classes"] = len(combinations)
        self.loss_coef  = active_loss_coef(self.mode_param["loss_coef"], combinations)
        temperature     = self.mode_param.get("temperature", 0.1)

        self.model = SelfTouchContrastiveGRU(self.model_param).to(self.device)
        os.makedirs(self.model_param["model_save_path"], exist_ok=True)
        plot_dir = os.path.join(self.model_param["model_save_path"], "plots")
        os.makedirs(plot_dir, exist_ok=True)

        if not sweep:
            config = {**self.model_param, **self.mode_param.get("loss_coef", {})}
            wandb.init(**wandb_init_kwargs(self.mode_param), config=config)
            wandb_log_heartbeat(wandb)
        maybe_watch_model(wandb, self.model, self.mode_param)

        dataset = CustomDataLoader(self.dataset_param)
        dataset.set_batch_size(batch_size)
        dataset.set_shuffle(True)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.use_amp = amp_enabled(self.mode_param, self.device)
        self.scaler = make_grad_scaler(self.use_amp)
        if self.use_amp:
            print("[speed] AMP enabled")

        metrics_csv = os.path.join(self.model_param["model_save_path"], "epoch_metrics.csv")
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "lr", "total_loss", "contrastive_loss", "cls_loss",
                "loss_index", "loss_thumb", "loss_middle",
                "best_total_loss",
            ])

        best_total_loss = 1e10
        early_stopper = EarlyStopper(self.mode_param)

        for epoch in range(total_epoch):
            current_lr = apply_lr_schedule(self.optimizer, lr, epoch, self.mode_param)
            self.model.train()
            for data in dataset:
                self.optimizer.zero_grad(set_to_none=True)
                data = {k: v.to(self.device) for k, v in data.items()}
                pos    = data["hand_jnt_pos"]
                vel    = data["hand_jnt_vel"]
                trq    = data["hand_jnt_trq"]
                cmd    = data["hand_jnt_cmd_pos"]
                labels = data["label"]

                with autocast(self.use_amp):
                    sup_con = _supcon_loss(
                        nn.functional.normalize(self.model.encode(pos, vel, trq, cmd), dim=-1),
                        labels, temperature,
                    ) * self.loss_coef.get("contrastive", 1.0)

                    losses, preds = self.model.forward_loss(
                        tactile_index_tip=data["tactile_index_tip"],
                        tactile_thumb_tip=data["tactile_thumb_tip"],
                        tactile_middle_tip=data["tactile_middle_tip"],
                        hand_jnt_pos=pos,
                        hand_jnt_vel=vel,
                        hand_jnt_trq=trq,
                        hand_jnt_cmd_pos=cmd,
                        labels=labels,
                        loss_coef=self.loss_coef,
                    )
                    total = (
                        sup_con
                        + self.loss_coef.get("classification", 1.0) * losses["classification"]
                        + self.loss_coef.get("tactile_index_tip", 1.0) * losses["tactile_index"]
                        + self.loss_coef.get("tactile_thumb_tip", 1.0) * losses["tactile_thumb"]
                        + self.loss_coef.get("tactile_middle_tip", 1.0) * losses["tactile_middle"]
                    )
                if self.use_amp:
                    self.scaler.scale(total).backward()
                    if self.grad_clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total.backward()
                    if self.grad_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                if (epoch + 1) % model_save_iter == 0 or epoch == 0:
                    ckpt = os.path.join(self.model_param["model_save_path"], f"epoch{epoch}.pth")
                    torch.save(self.model.state_dict(), ckpt)
                    wandb.save(ckpt, policy="now")
                    print("Model saved at:", ckpt)

                if not should_run_period(epoch, total_epoch, eval_every):
                    print(f"Epoch {epoch + 1}/{total_epoch} | eval skipped")
                    continue

                test_data = dataset.get_test_data()
                test_d = {k: v.to(self.device) for k, v in test_data.items()}
                pos_t  = test_d["hand_jnt_pos"]
                vel_t  = test_d["hand_jnt_vel"]
                trq_t  = test_d["hand_jnt_trq"]
                cmd_t  = test_d["hand_jnt_cmd_pos"]
                lbl_t  = test_d["label"]

                with autocast(self.use_amp):
                    sup_con_t = _supcon_loss(
                        nn.functional.normalize(self.model.encode(pos_t, vel_t, trq_t, cmd_t), dim=-1),
                        lbl_t, temperature,
                    )
                    test_losses, test_preds = self.model.forward_loss(
                        tactile_index_tip=test_d["tactile_index_tip"],
                        tactile_thumb_tip=test_d["tactile_thumb_tip"],
                        tactile_middle_tip=test_d["tactile_middle_tip"],
                        hand_jnt_pos=pos_t,
                        hand_jnt_vel=vel_t,
                        hand_jnt_trq=trq_t,
                        hand_jnt_cmd_pos=cmd_t,
                        labels=lbl_t,
                        loss_coef=self.loss_coef,
                    )

                tl = (
                    self.loss_coef.get("contrastive", 1.0) * float(sup_con_t.detach().cpu())
                    + self.loss_coef.get("classification", 1.0) * float(test_losses["classification"].detach().cpu())
                    + self.loss_coef.get("tactile_index_tip", 1.0) * float(test_losses["tactile_index"].detach().cpu())
                    + self.loss_coef.get("tactile_thumb_tip", 1.0) * float(test_losses["tactile_thumb"].detach().cpu())
                    + self.loss_coef.get("tactile_middle_tip", 1.0) * float(test_losses["tactile_middle"].detach().cpu())
                )
                if tl < best_total_loss:
                    best_total_loss = tl

                log = {
                    "epoch":            epoch + 1,
                    "lr":               current_lr,
                    "total_loss":       tl,
                    "contrastive_loss": float(sup_con_t.detach().cpu()),
                    "cls_loss":         float(test_losses["classification"].detach().cpu()),
                    "loss_index":       float(test_losses["tactile_index"].detach().cpu()),
                    "loss_thumb":       float(test_losses["tactile_thumb"].detach().cpu()),
                    "loss_middle":      float(test_losses["tactile_middle"].detach().cpu()),
                    "best_total_loss":  float(best_total_loss),
                }

                plot_paths = {"metrics": {}, "images": {}}
                if should_run_period(epoch, total_epoch, plot_every):
                    plot_paths = self._save_epoch_plots(
                        test_d, test_preds, epoch, plot_dir, combinations,
                        dataset.test_episode_names, pos_t, vel_t, trq_t, cmd_t, lbl_t
                    )
                log.update(plot_paths.get("metrics", {}))
                for wandb_key, path in plot_paths.get("images", {}).items():
                    log[wandb_key] = wandb.Image(path)

                with open(metrics_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch + 1, current_lr, log["total_loss"], log["contrastive_loss"], log["cls_loss"],
                        log["loss_index"], log["loss_thumb"], log["loss_middle"],
                        log["best_total_loss"],
                    ])
                wandb.log(log)
                print(
                    f"Epoch {epoch + 1}/{total_epoch} | lr={current_lr:.2e} | "
                    f"total={log['total_loss']:.4f} | "
                    f"contrastive={log['contrastive_loss']:.4f} | "
                    f"cls={log['cls_loss']:.4f} | "
                    f"idx={log['loss_index']:.4f} | thb={log['loss_thumb']:.4f} | "
                    f"mid={log['loss_middle']:.4f} | "
                    f"best={log['best_total_loss']:.4f}"
                )

                early_metric = log.get(early_stopper.monitor, tl)
                if early_stopper.step(early_metric, epoch):
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"{early_stopper.monitor} plateaued; best={early_stopper.best:.6f}"
                    )
                    break

        if not sweep:
            wandb.finish()

    # ── epoch plots ───────────────────────────────────────────────────────────
    def _save_epoch_plots(
        self, data, preds, epoch, plot_dir, combinations, episode_names,
        pos_seq, vel_seq, trq_seq, cmd_seq, labels
    ):
        plot_bundle = plot_tactile_temporal_profiles(
            data=data,
            preds=preds,
            epoch=epoch + 1,
            plot_dir=plot_dir,
            dataset_param=self.dataset_param,
            combinations=combinations,
            finger_names=FINGER_NAMES,
            finger_keys=FINGER_KEYS,
            next_step=True,
        )
        images = dict(plot_bundle.get("images", {}))
        metrics = dict(plot_bundle.get("metrics", {}))

        # — combination PCA using encoder embeddings —
        combo_path = self._save_combination_pca(
            pos_seq, vel_seq, trq_seq, cmd_seq, labels, epoch, plot_dir, combinations, episode_names
        )
        if combo_path:
            images["combination_pca"] = combo_path

        return {"images": images, "metrics": metrics}

    def _save_combination_pca(
        self, pos_seq, vel_seq, trq_seq, cmd_seq, labels, epoch, plot_dir, combinations, episode_names
    ):
        if pos_seq.shape[0] < 2:
            return None
        with torch.no_grad():
            embs = self.model.encode(pos_seq, vel_seq, trq_seq, cmd_seq).detach().cpu().numpy()
        embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
        lbl_np = labels.detach().cpu().numpy()
        pca_pts = _pca_2d(embs)
        cmap = plt.cm.get_cmap("tab10")

        plt.figure(figsize=(8, 6))
        for ci, combo in enumerate(combinations):
            idx = np.where(lbl_np == ci)[0]
            if idx.size:
                plt.scatter(pca_pts[idx, 0], pca_pts[idx, 1],
                            color=cmap(ci % 10), s=40, alpha=0.8, label=combo)
        # unknown labels
        uk = np.where(lbl_np == -1)[0]
        if uk.size:
            plt.scatter(pca_pts[uk, 0], pca_pts[uk, 1], color="gray", s=30, alpha=0.5, label="unknown")

        plt.title(f"Combination PCA — encoder embeddings (epoch {epoch + 1})")
        plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
        path = os.path.join(plot_dir, f"combo_pca_epoch_{epoch + 1:04d}.png")
        plt.savefig(path, dpi=160); plt.close()
        return path

    # ── test ──────────────────────────────────────────────────────────────────
    def test_controller(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        combinations = self.dataset_param.get("combinations", [])
        if combinations:
            self.model_param["num_classes"] = len(combinations)
        self.model = SelfTouchContrastiveGRU(self.model_param).to(self.device)

        save_dir = self.model_param.get("model_save_path", "")
        model_path = self.mode_param.get("model_load_path", "")
        if not model_path or not os.path.isfile(str(model_path)):
            cands = sorted(glob.glob(os.path.join(save_dir, "*.pth")))
            model_path = cands[-1] if cands else None
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError("No checkpoint found.")
        print(f"[ckpt] Loading: {model_path}")
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.loss_coef = active_loss_coef(self.mode_param["loss_coef"], self.dataset_param.get("combinations", []))
        dataset   = CustomDataLoader(self.dataset_param)
        test_data = dataset.get_test_data()
        data = {k: v.to(self.device) for k, v in test_data.items()}

        scaling_path = os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl")
        if not os.path.isfile(scaling_path):
            raise FileNotFoundError(f"Missing scaling_param.pkl: {scaling_path}")
        with open(scaling_path, "rb") as f:
            scaling_param = pickle.load(f)

        os.makedirs("./log/graphs", exist_ok=True)
        BATCH, T = 0, self.dataset_param["sequence_length"]
        comps = ["X", "Y", "Z"]
        active_fingers = included_fingers_from_combinations(combinations)

        with torch.no_grad():
            _, preds = self.model.forward_loss(
                tactile_index_tip=data["tactile_index_tip"],
                tactile_thumb_tip=data["tactile_thumb_tip"],
                tactile_middle_tip=data["tactile_middle_tip"],
                hand_jnt_pos=data["hand_jnt_pos"],
                hand_jnt_vel=data["hand_jnt_vel"],
                hand_jnt_trq=data["hand_jnt_trq"],
                hand_jnt_cmd_pos=data["hand_jnt_cmd_pos"],
                labels=data["label"],
                loss_coef=self.loss_coef,
            )

            from data_preproc import unscale_data

            def unscale(x_np, mod_key):
                return unscale_data(
                    x_np, scaling_param[mod_key], self.dataset_param["modality"][mod_key]
                )

            def unscale_pred(x_np, mod_key):
                return unscale(x_np, mod_key)

            finger_items = [
                (fname, fkey)
                for fname, fkey in zip(FINGER_NAMES, FINGER_KEYS)
                if fname in active_fingers and fkey in data and fname in preds
            ]
            fig, axes = plt.subplots(
                len(finger_items), 3,
                figsize=(18, 3.5 * len(finger_items)),
                sharex=True,
            )
            axes = np.atleast_2d(axes)
            for row, (fname, fkey) in enumerate(finger_items):
                raw_t  = data[fkey].cpu()
                pred_t = preds[fname].cpu()
                pred_t_steps = pred_t.shape[1]
                raw_np  = unscale(raw_t.numpy(),  fkey).reshape(-1, T, 30, 3)
                pred_np = unscale_pred(pred_t.numpy(), fkey).reshape(-1, pred_t_steps, 30, 3)
                raw_bs  = raw_np[:, 1:1 + pred_t_steps, :, :]
                pred_bs = pred_np
                ts = np.arange(1, 1 + pred_t_steps)

                print(f"\n── {fname.upper()} (raw values, batch {BATCH}) ──")
                for ci, cname in enumerate(comps):
                    avg_raw  = raw_bs[BATCH, :, :, ci].mean(axis=1)
                    avg_pred = pred_bs[BATCH, :, :, ci].mean(axis=1)
                    avg_err  = avg_raw - avg_pred
                    mae = float(np.abs(avg_err).mean())
                    print(f"  {cname}: raw_range=[{avg_raw.min():.1f}, {avg_raw.max():.1f}]  "
                          f"pred_range=[{avg_pred.min():.1f}, {avg_pred.max():.1f}]  MAE={mae:.2f}")
                    ax = axes[row, ci]
                    color = FINGER_COLORS[row % len(FINGER_COLORS)]
                    draw_tactile_prediction_profile(ax, ts, avg_raw, avg_pred, avg_err, color)
                    ax.set_title(f"{fname.capitalize()} – {cname}", fontsize=9)
                    ax.set_ylabel("Raw tactile [a.u.]", fontsize=7)
                    ax.grid(True, alpha=0.3)
                    if row == 0:
                        ax.legend(fontsize=6)
            set_shared_y_limits_from_lines(axes)
            for ax in axes[-1]:
                ax.set_xlabel("Timestep")
            fig.suptitle(TACTILE_PLOT_TITLE, fontsize=12)
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            plot_path = "./log/graphs/all_fingers_tactile_test_cfcn.png"
            plt.savefig(plot_path, dpi=160); plt.close()
            print(f"\n[OK] Per-finger test plot saved: {plot_path}")

    # ── unused modes ──────────────────────────────────────────────────────────
    def pretrain_controller(self, sweep: bool = False):
        self.train_controller(sweep=sweep)

    def motion_controller(self):
        pass

    def sweep_controller(self):
        self.train_controller(sweep=True)
