import sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
from controller_base import AbstractController
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from selftouch import SelfTouch
from data_loader import CustomDataLoader
import einops
import wandb
import numpy as np
import visualizer as vis
from schedulefree import RAdamScheduleFree
from augument import EIPLSARNNAugment
import data_preproc as dp
from util import log_memory_usage,LossScheduler,search_wandb_search,restore_data
from vis_tac import DualXelaVisualizer
import copy
from training_speed_utils import (
    EarlyStopper,
    apply_lr_schedule,
    amp_enabled,
    autocast,
    configure_torch_threads,
    configure_cuda_memory_fraction,
    empty_cuda_cache,
    make_grad_scaler,
    maybe_watch_model,
    should_run_period,
    wandb_init_kwargs,
)
from selftouch_plot_utils import active_loss_coef, load_scaling_param, plot_tactile_temporal_profiles


FINGER_NAMES = ["index", "thumb", "middle"]
FINGER_KEYS = ["tactile_index_tip", "tactile_thumb_tip", "tactile_middle_tip"]
WANDB_TACTILE_METRIC_KEYS = {
    "tactile_line_mae",
    "tactile_line_raw_accuracy",
    "tactile_line_rmse",
    "tactile_line_error_p95",
    "tactile_line_bias",
    "index_mae",
    "index_raw_accuracy",
    "thumb_mae",
    "thumb_raw_accuracy",
    "middle_mae",
    "middle_raw_accuracy",
}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _wandb_metric_subset(metrics):
    return {
        key: float(value)
        for key, value in (metrics or {}).items()
        if key in WANDB_TACTILE_METRIC_KEYS
    }


def _scaling_array(stats):
    if stats is None:
        return None
    if hasattr(stats, "to_numpy"):
        return stats.to_numpy(dtype=np.float32)
    return np.asarray(stats, dtype=np.float32)


class RNN_controller(AbstractController):
    def __init__(self, model_param, mode_param, dataset_param,config_param):
        super().__init__(model_param, mode_param, dataset_param,config_param)

    def train_controller(self, sweep=False):
        configure_torch_threads(self.mode_param)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print("Device: ", self.device)

        batch_size = int(self.mode_param["batch_size"])
        eval_batch_size = int(self.mode_param.get("eval_batch_size", 0) or 0)
        total_epoch = int(self.mode_param["num_epochs"])
        lr = float(self.mode_param["lr"])
        model_save_iter = int(self.mode_param["model_save_iter"])
        eval_every = int(self.mode_param.get("eval_every", 1))
        plot_every = int(self.mode_param.get("plot_every", eval_every))
        wandb_log_images = _as_bool(self.mode_param.get("wandb_log_images", False))
        wandb_save_model = _as_bool(self.mode_param.get("wandb_save_model", False))
        wandb_save_scaling = _as_bool(self.mode_param.get("wandb_save_scaling", False))
        empty_cache_after_save = _as_bool(self.mode_param.get("empty_cache_after_save", True))
        empty_cache_after_eval = _as_bool(self.mode_param.get("empty_cache_after_eval", False))

        self.sequence_length = int(self.dataset_param["sequence_length"])
        self.shift_data = int(self.dataset_param["shift_data"])
        combinations = self.dataset_param.get("combinations", [])
        self.loss_coef = active_loss_coef(self.mode_param["loss_coef"], combinations)

        os.makedirs(self.model_param["model_save_path"], exist_ok=True)
        plot_dir = os.path.join(self.model_param["model_save_path"], "plots")
        if os.path.isdir(plot_dir):
            for filename in os.listdir(plot_dir):
                if filename.startswith(("raw_prediction_", "tactile_profile_epoch_", "tactile_residual_epoch_")):
                    path = os.path.join(plot_dir, filename)
                    if os.path.isfile(path):
                        os.remove(path)
        os.makedirs(plot_dir, exist_ok=True)

        if not sweep:
            config={**self.model_param,**self.mode_param["loss_coef"]}
            wandb.init(**wandb_init_kwargs(self.mode_param, config=config))
            
        setup_start = time.perf_counter()
        dataset = CustomDataLoader(self.dataset_param)
        setup_seconds = time.perf_counter() - setup_start
        configure_cuda_memory_fraction(self.mode_param, self.device)
        self.model = SelfTouch(self.model_param).to(self.device)
        self._initialize_output_bias_from_train_targets(dataset)
        maybe_watch_model(wandb, self.model, self.mode_param)
        self._attach_raw_scale_loss_weights()
        dataset.set_batch_size(batch_size)
        dataset.set_shuffle(True)
        train_batches_per_epoch = (len(dataset) + batch_size - 1) // batch_size
        print(
            f"[speed] dataset setup={setup_seconds:.1f}s"
            f" | train_samples={len(dataset)}"
            f" | batch_size={batch_size}"
            f" | train_steps_per_epoch={train_batches_per_epoch}"
        )
        scaling_path = os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl")
        if wandb_save_scaling and os.path.isfile(scaling_path):
            wandb.save(scaling_path, policy="now")
        adamw_kwargs = {"lr": lr}
        if "adamw_foreach" in self.mode_param:
            adamw_kwargs["foreach"] = _as_bool(self.mode_param.get("adamw_foreach"))
        else:
            adamw_kwargs["foreach"] = False
        try:
            self.optimizer = optim.AdamW(self.model.parameters(), **adamw_kwargs)
        except TypeError:
            adamw_kwargs.pop("foreach", None)
            self.optimizer = optim.AdamW(self.model.parameters(), **adamw_kwargs)
        self.use_amp = amp_enabled(self.mode_param, self.device)
        self.scaler = make_grad_scaler(self.use_amp)
        if self.use_amp:
            print("[speed] AMP enabled")
        if eval_batch_size > 0:
            print(f"[cuda] Eval runs in CPU-sourced chunks of {eval_batch_size}")

        best_total_loss=1e10
        early_stopper = EarlyStopper(self.mode_param)

        for epoch in range(total_epoch):
            epoch_start = time.perf_counter()
            current_lr = apply_lr_schedule(self.optimizer, lr, epoch, self.mode_param)
            train_start = time.perf_counter()
            train_steps = 0
            for data in dataset:
                self.model.train()
                self.calc_loss_func(data,"train")
                train_steps += 1
            train_seconds = time.perf_counter() - train_start

            self.model.eval()
            with torch.no_grad():
                logger_dict={}
                save_seconds = 0.0
                if (epoch + 1) % model_save_iter == 0 or epoch==0:
                    save_start = time.perf_counter()
                    model_save_path = os.path.join(self.model_param["model_save_path"],f"epoch{epoch}.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    if wandb_save_model:
                        wandb.save(model_save_path, policy="now")
                    print("Model saved at:", model_save_path) 
                    if empty_cache_after_save:
                        empty_cuda_cache(self.device)
                    save_seconds = time.perf_counter() - save_start

                if not should_run_period(epoch, total_epoch, eval_every):
                    epoch_seconds = time.perf_counter() - epoch_start
                    print(
                        f"Epoch {epoch + 1}/{total_epoch} | eval skipped"
                        f" | steps={train_steps}"
                        f" | sec train={train_seconds:.1f}"
                        f" save={save_seconds:.1f}"
                        f" total={epoch_seconds:.1f}"
                    )
                    continue

                data = dataset.get_test_data()
                eval_start = time.perf_counter()
                (total_loss, loss_index, loss_thumb, loss_middle), \
                        (tactile_index_tip, tactile_thumb_tip, tactile_middle_tip) = \
                        self.eval_loss_func(data, eval_batch_size)
                eval_seconds = time.perf_counter() - eval_start
                
                if best_total_loss>total_loss:
                    best_total_loss=total_loss

                logger_dict={"lr": float(current_lr),
                                "epoch": epoch + 1,
                                "total_loss": float(total_loss),
                                "loss_index": float(loss_index),
                                "loss_thumb": float(loss_thumb),
                                "loss_middle": float(loss_middle),
                                "best_total_loss": float(best_total_loss)}

                if should_run_period(epoch, total_epoch, plot_every):
                    plot_start = time.perf_counter()
                    preds = {
                        "index": tactile_index_tip,
                        "thumb": tactile_thumb_tip,
                        "middle": tactile_middle_tip,
                    }
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
                    plot_metrics = plot_bundle.get("metrics", {})
                    logger_dict.update(_wandb_metric_subset(plot_metrics))
                    if wandb_log_images:
                        for wandb_key, path in plot_bundle.get("images", {}).items():
                            logger_dict[wandb_key] = wandb.Image(path)
                    plot_seconds = time.perf_counter() - plot_start
                else:
                    plot_metrics = {}
                    plot_seconds = 0.0

                wandb_start = time.perf_counter()
                wandb.log(logger_dict)
                wandb_seconds = time.perf_counter() - wandb_start
                metric_msg = ""
                if "tactile_line_mae" in plot_metrics:
                    metric_msg = (
                        f" | raw_mae={float(plot_metrics['tactile_line_mae']):.3f}"
                        f" | raw_acc={float(plot_metrics['tactile_line_raw_accuracy']):.1f}%"
                        f" | raw_avg={float(plot_metrics['tactile_line_raw_mean']):.3f}"
                        f" | pred_avg={float(plot_metrics['tactile_line_pred_mean']):.3f}"
                        f" | bias(raw-pred)={float(plot_metrics['tactile_line_bias']):.3f}"
                    )
                epoch_seconds = time.perf_counter() - epoch_start
                print(f"Epoch {epoch + 1}/{total_epoch} | lr={current_lr:.2e} | "
                      f"total={float(total_loss):.4f} | idx={float(loss_index):.4f} | "
                      f"thb={float(loss_thumb):.4f} | mid={float(loss_middle):.4f} | "
                      f"best={float(best_total_loss):.4f}{metric_msg}"
                      f" | steps={train_steps}"
                      f" | sec train={train_seconds:.1f}"
                      f" eval={eval_seconds:.1f}"
                      f" plot={plot_seconds:.1f}"
                      f" wandb={wandb_seconds:.1f}"
                      f" save={save_seconds:.1f}"
                      f" total={epoch_seconds:.1f}")

                early_metric = logger_dict.get(early_stopper.monitor, total_loss)
                if early_stopper.step(float(early_metric), epoch):
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"{early_stopper.monitor} plateaued; best={early_stopper.best:.6f}"
                    )
                    break

                if empty_cache_after_eval:
                    empty_cuda_cache(self.device)

        if not sweep:
            wandb.finish()                   

    def _raw_slope_array_for_key(self, scaling_param, key):
        stats = _scaling_array(scaling_param.get(key))
        if stats is None or stats.ndim < 2:
            return None
        rule = self.dataset_param.get("modality", {}).get(key)
        if not rule:
            return None

        mode = str(rule[0]).lower()
        output_range = float(rule[2][1] - rule[2][0]) if len(rule) > 2 else 1.0
        output_range = max(output_range, 1e-6)
        if mode == "n" and stats.shape[0] > 7:
            raw_scale = (stats[7] - stats[3]) / output_range
        elif mode == "rn" and stats.shape[0] > 6:
            raw_scale = (stats[6] - stats[4]) / output_range
        elif mode == "s" and stats.shape[0] > 2:
            raw_scale = stats[2]
        else:
            return None

        raw_scale = np.asarray(raw_scale, dtype=np.float32)
        positive = raw_scale[raw_scale > 1e-6]
        if positive.size == 0:
            return None
        fallback = float(np.median(positive))
        raw_scale = np.where(np.isfinite(raw_scale) & (raw_scale > 1e-6), raw_scale, fallback)
        clip_quantile = float(self.loss_coef.get("tactile_raw_slope_clip_quantile", 1.0))
        if 0.0 < clip_quantile < 1.0:
            upper = float(np.quantile(raw_scale, clip_quantile))
            raw_scale = np.clip(raw_scale, 0.0, max(upper, fallback))
        return raw_scale.astype(np.float32, copy=False)

    def _raw_scale_for_key(self, scaling_param, key):
        raw_scale = self._raw_slope_array_for_key(scaling_param, key)
        if raw_scale is None:
            return None
        positive = raw_scale[raw_scale > 1e-6]
        if positive.size == 0:
            return None
        raw_scale = raw_scale / float(np.mean(positive))
        clip = float(self.loss_coef.get("tactile_raw_scale_clip", 5.0))
        raw_scale = np.clip(raw_scale, 1.0 / max(clip, 1.0), clip)
        return torch.as_tensor(raw_scale, dtype=torch.float32, device=self.device)

    def _raw_slope_for_key(self, scaling_param, key):
        raw_slope = self._raw_slope_array_for_key(scaling_param, key)
        if raw_slope is None:
            return None
        return torch.as_tensor(raw_slope, dtype=torch.float32, device=self.device)

    def _initialize_output_bias_from_train_targets(self, dataset):
        if not _as_bool(self.mode_param.get("init_output_bias_from_tactile_mean", True)):
            return
        output_net = getattr(self.model, "output_net", None)
        if output_net is None or getattr(output_net, "bias", None) is None:
            return
        train_data = getattr(dataset, "train_dir_data", None)
        if not isinstance(train_data, dict):
            return

        means = []
        for key in FINGER_KEYS:
            value = train_data.get(key)
            if value is None or not torch.is_tensor(value) or value.ndim < 3:
                return
            target = value[:, 1:, :] if value.shape[1] > 1 else value
            means.append(target.mean(dim=(0, 1)))
        bias = torch.cat(means, dim=0).to(device=self.device, dtype=output_net.bias.dtype)
        if bias.numel() != output_net.bias.numel():
            print(
                f"[bias] skipped output bias initialization: "
                f"target mean has {bias.numel()} values, output bias has {output_net.bias.numel()}"
            )
            return
        with torch.no_grad():
            if _as_bool(self.mode_param.get("zero_output_head_on_bias_init", True)):
                init_std = float(self.mode_param.get("output_head_init_std", 1e-4))
                if init_std > 0.0:
                    output_net.weight.normal_(mean=0.0, std=init_std)
                else:
                    output_net.weight.zero_()
            output_net.bias.copy_(bias)
        print("[bias] initialized output head from training tactile mean")

    def _attach_raw_scale_loss_weights(self):
        if not (
            float(self.loss_coef.get("tactile_raw_scale_weight", 0.0))
            or float(self.loss_coef.get("tactile_raw_topk_weight", 0.0))
            or float(self.loss_coef.get("tactile_raw_bias_weight", 0.0))
            or float(self.loss_coef.get("tactile_raw_timestep_mean_weight", 0.0))
            or float(self.loss_coef.get("tactile_raw_taxel_mean_weight", 0.0))
        ):
            return
        scaling_param = load_scaling_param(self.dataset_param)
        raw_scale_by_key = {}
        raw_slope_by_key = {}
        for key in FINGER_KEYS:
            raw_scale = self._raw_scale_for_key(scaling_param, key)
            if raw_scale is not None:
                raw_scale_by_key[key] = raw_scale
            raw_slope = self._raw_slope_for_key(scaling_param, key)
            if raw_slope is not None:
                raw_slope_by_key[key] = raw_slope
        if raw_scale_by_key:
            self.loss_coef["tactile_raw_scale_by_key"] = raw_scale_by_key
            print(
                "[loss] raw-scale tactile weighting enabled for "
                + ", ".join(sorted(raw_scale_by_key))
            )
        if raw_slope_by_key:
            self.loss_coef["tactile_raw_slope_by_key"] = raw_slope_by_key
            print(
                "[loss] raw-unit tactile mean/bias loss enabled for "
                + ", ".join(sorted(raw_slope_by_key))
            )

    def test_controller(self):
        pass

    def pretrain_controller(self,sweep=False):
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

    def _data_batch_size(self, data):
        for value in data.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
        return 0

    def _slice_data_batch(self, data, start, end):
        output = {}
        for key, value in data.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= end:
                output[key] = value[start:end]
            else:
                output[key] = value
        return output

    def _assert_cpu_data(self, data, mode):
        if not _as_bool(self.mode_param.get("assert_cpu_batches", True)):
            return
        for key, value in data.items():
            if torch.is_tensor(value) and value.is_cuda:
                raise RuntimeError(
                    f"{mode} batch key '{key}' is already on CUDA before calc_loss_func; "
                    "data must stay in CPU RAM until the batch transfer step."
                )

    def eval_loss_func(self, data, eval_batch_size=0):
        eval_batch_size = int(eval_batch_size or 0)
        num_samples = self._data_batch_size(data)
        if eval_batch_size <= 0 or num_samples <= eval_batch_size:
            return self.calc_loss_func(data, "test")

        weighted_losses = np.zeros(4, dtype=np.float64)
        pred_chunks = [[], [], []]
        total_count = 0
        for start in range(0, num_samples, eval_batch_size):
            end = min(start + eval_batch_size, num_samples)
            chunk = self._slice_data_batch(data, start, end)
            (total_loss, loss_index, loss_thumb, loss_middle), preds = self.calc_loss_func(chunk, "test")
            chunk_count = end - start
            for idx, loss_value in enumerate((total_loss, loss_index, loss_thumb, loss_middle)):
                if torch.is_tensor(loss_value):
                    loss_value = loss_value.detach()
                weighted_losses[idx] += float(loss_value) * chunk_count
            for idx, pred in enumerate(preds):
                pred_chunks[idx].append(pred.detach().cpu())
            total_count += chunk_count
            del chunk, preds, total_loss, loss_index, loss_thumb, loss_middle

        averaged = tuple(float(value / max(total_count, 1)) for value in weighted_losses)
        preds = tuple(torch.cat(chunks, dim=0) for chunks in pred_chunks)
        return averaged, preds

    def calc_loss_func(self, data, mode="train"):
        if mode == "train":
            self.optimizer.zero_grad(set_to_none=True)

        self._assert_cpu_data(data, mode)
        data = {key: val.to(self.device) for key, val in data.items()}

        with autocast(getattr(self, "use_amp", False)):
            (total_loss, loss_index, loss_thumb, loss_middle), \
                (tactile_index_tip, tactile_thumb_tip, tactile_middle_tip) = \
                self.model.forward_loss(**data, loss_coef=self.loss_coef)

        if mode == "train":
            if getattr(self, "use_amp", False):
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

        return (
            total_loss, loss_index, loss_thumb, loss_middle
        ), (
            tactile_index_tip, tactile_thumb_tip, tactile_middle_tip
        )
