import sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
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
import time
import copy
from training_speed_utils import (
    EarlyStopper,
    apply_lr_schedule,
    amp_enabled,
    autocast,
    configure_cuda_memory_fraction,
    configure_torch_threads,
    make_grad_scaler,
    maybe_watch_model,
    selftouch_loss_step,
    should_run_period,
)
from selftouch_plot_utils import active_loss_coef, plot_tactile_temporal_profiles


FINGER_NAMES = ["index", "thumb", "middle"]
FINGER_KEYS = ["tactile_index_tip", "tactile_thumb_tip", "tactile_middle_tip"]


class RNN_controller(AbstractController):
    def __init__(self, model_param, mode_param, dataset_param,config_param):
        super().__init__(model_param, mode_param, dataset_param,config_param)
        if torch.cuda.is_available() and str(mode_param.get("device", os.environ.get("SELFTOUCH_DEVICE", "auto"))).lower() != "cpu":
            torch.cuda.init()

    def train_controller(self, sweep=False):
        configure_torch_threads(self.mode_param)
        device_setting = str(self.mode_param.get('device', os.environ.get('SELFTOUCH_DEVICE', 'auto'))).lower()
        if device_setting == 'cpu':
            self.device = torch.device('cpu')
        elif device_setting.startswith('cuda') and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif device_setting in {'auto', ''}:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        else:
            print(f"[warn] requested device '{device_setting}' is unavailable; falling back to CPU")
            self.device = torch.device('cpu')
        print("Device: ", self.device)

        batch_size = int(os.environ.get("SELFTOUCH_BATCH_SIZE", self.mode_param["batch_size"]))
        total_epoch = self.mode_param["num_epochs"]
        lr = self.mode_param["lr"]
        model_save_iter = self.mode_param["model_save_iter"]
        eval_every = self.mode_param.get("eval_every", 1)
        plot_every = self.mode_param.get("plot_every", eval_every)
        train_step_sleep_seconds = float(os.environ.get("SELFTOUCH_TRAIN_STEP_SLEEP", self.mode_param.get("train_step_sleep_seconds", 0.0)) or 0.0)
        max_train_batches_per_epoch = int(os.environ.get("SELFTOUCH_MAX_TRAIN_BATCHES", self.mode_param.get("max_train_batches_per_epoch", 0)) or 0)

        self.sequence_length = self.dataset_param["sequence_length"]
        self.shift_data = self.dataset_param["shift_data"]
        combinations = self.dataset_param.get("combinations", [])
        self.loss_coef = active_loss_coef(self.mode_param["loss_coef"], combinations)

        
        configure_cuda_memory_fraction(self.mode_param, self.device)
        self.model = SelfTouch(self.model_param).to(self.device)
        os.makedirs(self.model_param["model_save_path"], exist_ok=True)
        plot_dir = os.path.join(self.model_param["model_save_path"], "plots")
        os.makedirs(plot_dir, exist_ok=True)

        if not sweep:
            config={**self.model_param,**self.mode_param["loss_coef"]}
            wandb.init(project=self.mode_param["project"],config=config)
        maybe_watch_model(wandb, self.model, self.mode_param)
            
        dataset = CustomDataLoader(self.dataset_param)
        dataset.set_batch_size(batch_size)
        dataset.set_shuffle(True)
        wandb.save(os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"),policy="now")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.use_amp = amp_enabled(self.mode_param, self.device)
        self.scaler = make_grad_scaler(self.use_amp)
        if self.use_amp:
            print("[speed] AMP enabled")

        best_total_loss=1e10
        early_stopper = EarlyStopper(self.mode_param)

        for epoch in range(total_epoch):
            current_lr = apply_lr_schedule(self.optimizer, lr, epoch, self.mode_param)
            train_steps = 0
            for data in dataset:
                self.model.train()
                self.calc_loss_func(data,"train")
                if train_step_sleep_seconds > 0.0:
                    time.sleep(train_step_sleep_seconds)
                train_steps += 1
                if max_train_batches_per_epoch > 0 and train_steps >= max_train_batches_per_epoch:
                    break

            self.model.eval()
            with torch.no_grad():
                logger_dict={}
                if (epoch + 1) % model_save_iter == 0 or epoch==0:
                    model_save_path = os.path.join(self.model_param["model_save_path"],f"epoch{epoch}.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    wandb.save(model_save_path,policy="now")
                    print("Model saved at:", model_save_path) 

                if not should_run_period(epoch, total_epoch, eval_every):
                    print(f"Epoch {epoch + 1}/{total_epoch} | eval skipped")
                    continue

                data = dataset.get_test_data()
                (total_loss, loss_index, loss_thumb, loss_middle), \
                        (tactile_index_tip, tactile_thumb_tip, tactile_middle_tip) = self.calc_loss_func(data, "test")
                
                if best_total_loss>total_loss:
                    best_total_loss=total_loss

                logger_dict={"lr": current_lr,
                                "epoch": epoch + 1,
                                "total_loss": total_loss,
                                "loss_index": loss_index,
                                "loss_thumb": loss_thumb,
                                "loss_middle": loss_middle,
                                "best_total_loss": best_total_loss}

                if should_run_period(epoch, total_epoch, plot_every):
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
                    logger_dict.update(plot_bundle.get("metrics", {}))
                    for wandb_key, path in plot_bundle.get("images", {}).items():
                        logger_dict[wandb_key] = wandb.Image(path)

                wandb.log(logger_dict)
                print(f"Epoch {epoch + 1}/{total_epoch} | lr={current_lr:.2e} | "
                      f"total={float(total_loss):.4f} | idx={float(loss_index):.4f} | "
                      f"thb={float(loss_thumb):.4f} | mid={float(loss_middle):.4f} | "
                      f"best={float(best_total_loss):.4f}")

                early_metric = logger_dict.get(early_stopper.monitor, total_loss)
                if early_stopper.step(float(early_metric), epoch):
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"{early_stopper.monitor} plateaued; best={early_stopper.best:.6f}"
                    )
                    break

        if not sweep:
            wandb.finish()                   

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
