import sys
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from controller_base import AbstractController
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from selftouch import SelfTouchTransformer
from data_loader import CustomDataLoader
import einops
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_preproc import unscale_data
import pickle
import copy
import csv
import time
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
    as_float,
    apply_lr_schedule,
    amp_enabled,
    autocast,
    configure_cuda_memory_fraction,
    configure_torch_threads,
    data_batch_size,
    empty_cuda_cache,
    make_grad_scaler,
    maybe_watch_model,
    selftouch_loss_step,
    should_run_period,
    slice_data_batch,
    wandb_init_kwargs,
    wandb_log_heartbeat,
)

FINGER_NAMES  = ["index", "thumb", "middle", "ring"]
FINGER_KEYS   = ["tactile_index_tip", "tactile_thumb_tip", "tactile_middle_tip", "tactile_ring_tip"]
FINGER_COLORS = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]


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


def _infer_combination(ep_name: str, combinations: list) -> str:
    ep_lower = ep_name.lower()
    for combo in combinations:
        if combo.replace("-", "_") in ep_lower:
            return combo
    return "unknown"


class RNN_controller(AbstractController):
    def __init__(self, model_param, mode_param, dataset_param, config_param):
        super().__init__(model_param, mode_param, dataset_param, config_param)
        if torch.cuda.is_available() and str(mode_param.get("device", os.environ.get("SELFTOUCH_DEVICE", "auto"))).lower() != "cpu":
            torch.cuda.init()

    # ── train ─────────────────────────────────────────────────────────────────
    def train_controller(self, sweep=False):
        configure_torch_threads(self.mode_param)
        device_setting = str(self.mode_param.get("device", os.environ.get("SELFTOUCH_DEVICE", "auto"))).lower()
        if device_setting == "cpu":
            self.device = torch.device("cpu")
        elif device_setting.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif device_setting in {"auto", ""}:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        else:
            print(f"[warn] requested device '{device_setting}' is unavailable; falling back to CPU")
            self.device = torch.device("cpu")
        print("Device:", self.device)

        batch_size      = int(os.environ.get("SELFTOUCH_BATCH_SIZE", self.mode_param["batch_size"]))
        eval_batch_size = int(os.environ.get("SELFTOUCH_EVAL_BATCH_SIZE", self.mode_param.get("eval_batch_size", 0)) or 0)
        total_epoch     = self.mode_param["num_epochs"]
        lr              = self.mode_param["lr"]
        weight_decay    = self.mode_param.get("weight_decay", 1e-4)
        model_save_iter = self.mode_param["model_save_iter"]
        eval_every      = self.mode_param.get("eval_every", 1)
        plot_every      = self.mode_param.get("plot_every", eval_every)
        self.grad_clip_norm = self.mode_param.get("grad_clip_norm", 1.0)
        empty_cache_after_save = bool(self.mode_param.get("empty_cache_after_save", True))
        empty_cache_after_eval = bool(self.mode_param.get("empty_cache_after_eval", False))
        train_step_sleep_seconds = float(os.environ.get("SELFTOUCH_TRAIN_STEP_SLEEP", self.mode_param.get("train_step_sleep_seconds", 0.0)) or 0.0)
        max_train_batches_per_epoch = int(os.environ.get("SELFTOUCH_MAX_TRAIN_BATCHES", self.mode_param.get("max_train_batches_per_epoch", 0)) or 0)

        self.sequence_length = self.dataset_param["sequence_length"]
        self.shift_data      = self.dataset_param["shift_data"]
        combinations         = self.dataset_param.get("combinations", [])
        self.loss_coef       = active_loss_coef(self.mode_param["loss_coef"], combinations)

        configure_cuda_memory_fraction(self.mode_param, self.device)
        self.model = SelfTouchTransformer(self.model_param).to(self.device)
        os.makedirs(self.model_param["model_save_path"], exist_ok=True)
        plot_dir = os.path.join(self.model_param["model_save_path"], "plots")
        os.makedirs(plot_dir, exist_ok=True)

        metrics_csv = os.path.join(self.model_param["model_save_path"], "epoch_metrics.csv")
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr", "total_loss",
                "loss_index", "loss_thumb", "loss_middle", "best_total_loss",
            ])

        if not sweep:
            config = {**self.model_param, **self.mode_param["loss_coef"]}
            wandb.init(**wandb_init_kwargs(self.mode_param), config=config)
            wandb_log_heartbeat(wandb)
        maybe_watch_model(wandb, self.model, self.mode_param)

        dataset = CustomDataLoader(self.dataset_param)
        dataset.set_batch_size(batch_size)
        dataset.set_shuffle(True)
        wandb.save(
            os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"),
            policy="now",
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.use_amp = amp_enabled(self.mode_param, self.device)
        self.scaler = make_grad_scaler(self.use_amp)
        if self.use_amp:
            print("[speed] AMP enabled")
        if eval_batch_size > 0:
            print(f"[cuda] Eval runs in CPU-sourced chunks of {eval_batch_size}")

        best_total_loss = 1e10
        early_stopper = EarlyStopper(self.mode_param)
        for epoch in range(total_epoch):
            current_lr = apply_lr_schedule(self.optimizer, lr, epoch, self.mode_param)
            train_steps = 0
            for data in dataset:
                self.model.train()
                self.calc_loss_func(data, "train")
                if train_step_sleep_seconds > 0.0:
                    time.sleep(train_step_sleep_seconds)
                train_steps += 1
                if max_train_batches_per_epoch > 0 and train_steps >= max_train_batches_per_epoch:
                    break

            self.model.eval()
            with torch.no_grad():
                if (epoch + 1) % model_save_iter == 0 or epoch == 0:
                    ckpt_path = os.path.join(
                        self.model_param["model_save_path"], f"epoch{epoch}.pth"
                    )
                    torch.save(self.model.state_dict(), ckpt_path)
                    wandb.save(ckpt_path, policy="now")
                    print("Model saved at:", ckpt_path)
                    if empty_cache_after_save:
                        empty_cuda_cache(self.device)

                if not should_run_period(epoch, total_epoch, eval_every):
                    print(f"Epoch {epoch + 1}/{total_epoch} | eval skipped")
                    continue

                data = dataset.get_test_data()
                (total_loss, loss_idx, loss_thb, loss_mid), \
                    (idx_pred, thumb_pred, middle_pred) = \
                    self.eval_loss_func(data, eval_batch_size)

                if best_total_loss > total_loss:
                    best_total_loss = total_loss

                tl  = as_float(total_loss)
                li  = as_float(loss_idx)
                lt  = as_float(loss_thb)
                lm  = as_float(loss_mid)
                bl  = as_float(best_total_loss)

                with open(metrics_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, current_lr, tl, li, lt, lm, bl])

                logger_dict = {
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "total_loss": tl, "loss_index": li, "loss_thumb": lt,
                    "loss_middle": lm,
                    "best_total_loss": bl,
                }

                preds = {
                    "index": idx_pred, "thumb": thumb_pred, "middle": middle_pred,
                }
                plot_paths = {"metrics": {}, "images": {}}
                if should_run_period(epoch, total_epoch, plot_every):
                    plot_paths = self._save_epoch_plots(
                        data, preds, epoch, plot_dir,
                        combinations, dataset.test_episode_names,
                    )
                logger_dict.update(plot_paths.get("metrics", {}))
                for wandb_key, path in plot_paths.get("images", {}).items():
                    logger_dict[wandb_key] = wandb.Image(path)

                wandb.log(logger_dict)
                print(
                    f"Epoch {epoch + 1}/{total_epoch} | lr={current_lr:.2e} | "
                    f"total={tl:.4f} | idx={li:.4f} | thb={lt:.4f} | "
                    f"mid={lm:.4f} | best={bl:.4f}"
                )

                early_metric = logger_dict.get(early_stopper.monitor, tl)
                if early_stopper.step(early_metric, epoch):
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"{early_stopper.monitor} plateaued; best={early_stopper.best:.6f}"
                    )
                    break
                if empty_cache_after_eval:
                    empty_cuda_cache(self.device)

        if not sweep:
            wandb.finish()

    # ── epoch plots ───────────────────────────────────────────────────────────
    def _save_epoch_plots(
        self, data, preds, epoch, plot_dir, combinations, episode_names
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

        # — combination PCA of encoder embeddings —
        combo_path = self._save_combination_pca_plot(
            data, epoch, plot_dir, combinations, episode_names
        )
        if combo_path:
            images["combination_pca"] = combo_path

        return {"images": images, "metrics": metrics}

    def _save_combination_pca_plot(
        self, data, epoch, plot_dir, combinations, episode_names
    ):
        if not combinations or not episode_names:
            return None
        N = data["hand_jnt_pos"].shape[0]
        embeddings, labels = [], []

        for i in range(N):
            pos_i = data["hand_jnt_pos"][i:i+1].to(self.device)
            vel_i = data.get("hand_jnt_vel", torch.zeros_like(data["hand_jnt_pos"]))[i:i+1].to(self.device)
            trq_i = data.get("hand_jnt_trq", torch.zeros_like(data["hand_jnt_pos"]))[i:i+1].to(self.device)
            cmd_i = data["hand_jnt_cmd_pos"][i:i+1].to(self.device)
            h = self.model.encode(pos_i, vel_i, trq_i, cmd_i)   # (1, T-1, d_model)
            emb = h.mean(dim=1).detach().cpu().numpy().flatten()
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(emb)
            ep_name = episode_names[i] if i < len(episode_names) else ""
            labels.append(_infer_combination(ep_name, combinations))

        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.shape[0] < 2:
            return None

        pca_pts    = _pca_2d(embeddings)
        combo_list = list(dict.fromkeys(combinations + [l for l in labels if l not in combinations]))
        cmap       = plt.cm.get_cmap("tab10")

        plt.figure(figsize=(8, 6))
        for ci, combo in enumerate(combo_list):
            idx = [j for j, l in enumerate(labels) if l == combo]
            if idx:
                plt.scatter(
                    pca_pts[idx, 0], pca_pts[idx, 1],
                    color=cmap(ci % 10), s=40, alpha=0.8, label=combo,
                )
        plt.title(f"Combination PCA — encoder embeddings (epoch {epoch + 1})")
        plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
        combo_pca_path = os.path.join(plot_dir, f"combo_pca_epoch_{epoch + 1:04d}.png")
        plt.savefig(combo_pca_path, dpi=160); plt.close()
        return combo_pca_path

    # ── test ──────────────────────────────────────────────────────────────────
    def test_controller(self):
        import glob, pickle
        import numpy as np, torch, einops
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from data_preproc import unscale_data
        from data_loader import CustomDataLoader
        from selftouch import SelfTouchTransformer

        BATCH = 0
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("Device:", self.device)

        self.model = SelfTouchTransformer(self.model_param).to(self.device)
        model_path = (self.mode_param or {}).get("model_load_path", None)
        if not model_path or not os.path.isfile(str(model_path)):
            save_dir = (self.model_param or {}).get("model_save_path", "")
            cands = sorted(glob.glob(os.path.join(save_dir, "*.pth")))
            model_path = cands[-1] if cands else None
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                "[ckpt] No valid checkpoint. Set Test.model_load_path or Model.model_save_path."
            )
        print(f"[ckpt] Loading: {model_path}")
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Remap legacy keys
        remapped = {}
        for k, v in state.items():
            nk = k
            if k == "embed_cmd_pos.weight": nk = "embed_cmd.weight"
            if k == "embed_cmd_pos.bias":   nk = "embed_cmd.bias"
            remapped[nk] = v
        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        self.model.eval()
        if missing:
            print("[ckpt] Missing keys (re-init):", missing)
        if unexpected:
            print("[ckpt] Unexpected keys (ignored):", unexpected)

        self.loss_coef       = self.mode_param["loss_coef"]
        self.sequence_length = self.dataset_param["sequence_length"]
        dataset   = CustomDataLoader(self.dataset_param)
        test_data = dataset.get_test_data()
        data = {k: v.to(self.device) for k, v in test_data.items()}

        os.makedirs("./log/graphs", exist_ok=True)

        with torch.no_grad():
            (_, li, lt, lm), (idx_pred, thumb_pred, middle_pred) = \
                self.model.forward_loss(**data, loss_coef=self.loss_coef)

            print(f"Test losses — index:{float(li):.4f}  thumb:{float(lt):.4f}  middle:{float(lm):.4f}")

            scaling_path = os.path.join(
                self.dataset_param["param_file_dir"], "scaling_param.pkl"
            )
            if not os.path.isfile(scaling_path):
                raise FileNotFoundError(f"Missing scaling_param.pkl: {scaling_path}")
            with open(scaling_path, "rb") as f:
                scaling_param = pickle.load(f)

            def unscale(x_np, mod_key):
                return unscale_data(
                    x_np, scaling_param[mod_key], self.dataset_param["modality"][mod_key]
                )

            def unscale_pred(x_np, mod_key):
                return unscale(x_np, mod_key)

            T = self.sequence_length
            finger_data = {
                "index":  (data["tactile_index_tip"].cpu(),  idx_pred.cpu(),    "tactile_index_tip"),
                "thumb":  (data["tactile_thumb_tip"].cpu(),  thumb_pred.cpu(),  "tactile_thumb_tip"),
                "middle": (data["tactile_middle_tip"].cpu(), middle_pred.cpu(), "tactile_middle_tip"),
            }
            active_fingers = included_fingers_from_combinations(self.dataset_param.get("combinations", []))
            finger_items = [
                (fname, values)
                for fname, values in finger_data.items()
                if fname in active_fingers
            ]

            comps = ["X", "Y", "Z"]
            fig, axes = plt.subplots(
                len(finger_items), 3,
                figsize=(18, 3.5 * len(finger_items)),
                sharex=True,
            )
            axes = np.atleast_2d(axes)

            for row, (fname, (raw_t, pred_t, mod_key)) in enumerate(finger_items):
                pred_t_steps = pred_t.shape[1]
                raw_np  = unscale(raw_t.numpy(),  mod_key).reshape(-1, T, 30, 3)
                pred_np = unscale_pred(pred_t.numpy(), mod_key).reshape(-1, pred_t_steps, 30, 3)
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
            plot_path = "./log/graphs/all_fingers_tactile_test.png"
            plt.savefig(plot_path, dpi=160); plt.close()
            print(f"\n[OK] Per-finger test plot saved: {plot_path}")

            rows = {}
            for fname, (raw_t, pred_t, mod_key) in finger_items:
                pred_t_steps = pred_t.shape[1]
                if not rows:
                    rows["Timestep"] = np.arange(1, 1 + pred_t_steps)
                raw_np  = unscale(raw_t.numpy(),  mod_key).reshape(-1, T, 30, 3)
                pred_np = unscale_pred(pred_t.numpy(), mod_key).reshape(-1, pred_t_steps, 30, 3)
                raw_bs  = raw_np[:, 1:1 + pred_t_steps, :, :]
                pred_bs = pred_np
                for ci, cname in enumerate(comps):
                    avg_raw = raw_bs[BATCH, :, :, ci].mean(axis=1)
                    avg_pred = pred_bs[BATCH, :, :, ci].mean(axis=1)
                    rows[f"Raw_{fname.capitalize()}_{cname}"]  = np.round(avg_raw, 1)
                    rows[f"Pred_{fname.capitalize()}_{cname}"] = np.round(avg_pred, 1)
                    rows[f"Err_{fname.capitalize()}_{cname}"]  = np.round(avg_raw - avg_pred, 1)

            import pandas as pd
            df = pd.DataFrame(rows)
            csv_path = "./log/graphs/self_tactile_avg_force_data.csv"
            df.to_csv(csv_path, index=False)
            print(f"[OK] CSV saved: {csv_path}")

    def pretrain_controller(self, sweep=False):
        pass

    def motion_controller(self):
        pass

    def sweep_controller(self):
        if "train" in self.config_param:
            print("START TRAIN")
            self.train_controller(sweep=True)
        elif "pretrain" in self.config_param:
            print("START PRETRAIN")
            self.pretrain_controller(sweep=True)
        else:
            print("not setted")
            sys.exit(1)

    def calc_loss_func(self, data, mode="train"):
        return selftouch_loss_step(
            model=self.model,
            optimizer=self.optimizer,
            scaler=getattr(self, "scaler", None),
            loss_coef=self.loss_coef,
            data=data,
            device=self.device,
            mode=mode,
            use_amp=getattr(self, "use_amp", False),
            mode_param=self.mode_param,
        )

    def eval_loss_func(self, data, eval_batch_size=0):
        eval_batch_size = int(eval_batch_size or 0)
        num_samples = data_batch_size(data)
        if eval_batch_size <= 0 or num_samples <= eval_batch_size:
            return self.calc_loss_func(data, "test")

        weighted_losses = None
        pred_chunks = None
        for start in range(0, num_samples, eval_batch_size):
            end = min(start + eval_batch_size, num_samples)
            weight = float(end - start) / float(max(num_samples, 1))
            losses, preds = self.calc_loss_func(slice_data_batch(data, start, end), "test")
            losses = tuple(losses)
            preds = tuple(preds)
            if weighted_losses is None:
                weighted_losses = [0.0 for _ in losses]
                pred_chunks = [[] for _ in preds]
            for idx, loss_value in enumerate(losses):
                weighted_losses[idx] += as_float(loss_value) * weight
            for idx, pred in enumerate(preds):
                pred_chunks[idx].append(pred.detach().cpu() if torch.is_tensor(pred) else torch.as_tensor(pred))
        return tuple(weighted_losses), tuple(torch.cat(chunks, dim=0) for chunks in pred_chunks)
