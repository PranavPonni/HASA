import os
import sys

import torch
from torch import optim
import wandb

from controller_base import AbstractController
import data_preproc as dp
from model.dexwild_motion_common.data_loader import CustomDataLoader
from model.dexwild_motion_common.models import DexWildMotionModel


class RNN_controller(AbstractController):
    def __init__(self, model_param, mode_param, dataset_param, config_param):
        super().__init__(model_param, mode_param, dataset_param, config_param)
        if torch.cuda.is_available():
            torch.cuda.init()

    def _load_selftouch(self):
        if not self.model_param.get("use_selftouch", False):
            return None

        st_param_path = self.model_param.get("st_param")
        st_model_path = self.model_param.get("st_model")
        if not st_param_path or not os.path.isfile(st_param_path):
            raise FileNotFoundError(f"Missing selftouch parameter file: {st_param_path}")
        if not st_model_path or not os.path.isfile(st_model_path):
            raise FileNotFoundError(f"Missing selftouch checkpoint: {st_model_path}")

        from model.selftouch_fcn_pos.selftouch import SelfTouch

        st_params = dp.read_yaml(st_param_path)["Model"]
        selftouch = SelfTouch(st_params).to(self.device)
        state = torch.load(st_model_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        selftouch.load_state_dict(state, strict=True)
        selftouch.eval()
        for p in selftouch.parameters():
            p.requires_grad = False
        return selftouch

    def _build_model(self):
        selftouch = self._load_selftouch()
        return DexWildMotionModel(self.model_param, selftouch=selftouch).to(self.device)

    def _wandb_init(self, sweep):
        if sweep:
            wandb.watch(self.model, log="gradients", log_freq=100)
            return
        init_kwargs = {"project": self.mode_param["project"]}
        entity = self.mode_param.get("wandb_entity")
        if entity:
            init_kwargs["entity"] = entity
        wandb.init(**init_kwargs, config={**self.model_param, **self.mode_param})
        if self.mode_param.get("wandb_watch", False):
            wandb.watch(
                self.model,
                log=self.mode_param.get("wandb_watch_log", "gradients"),
                log_freq=self.mode_param.get("wandb_watch_freq", 100),
            )

    def train_controller(self, sweep=False):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("Device:", self.device)

        batch_size = self.mode_param["batch_size"]
        total_epoch = self.mode_param["num_epochs"]
        lr = self.mode_param["lr"]
        weight_decay = self.mode_param.get("weight_decay", 0.0)
        model_save_iter = self.mode_param["model_save_iter"]
        eval_every = self.mode_param.get("eval_every", 1)
        grad_clip_norm = self.mode_param.get("grad_clip_norm")
        self.loss_coef = self.mode_param["loss_coef"]

        os.makedirs(self.model_param["model_save_path"], exist_ok=True)
        self.model = self._build_model()
        self._wandb_init(sweep)

        dataset = CustomDataLoader(self.dataset_param, self.model_param)
        dataset.set_batch_size(batch_size)
        dataset.set_shuffle(True)
        scaling_path = os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl")
        if os.path.isfile(scaling_path):
            wandb.save(scaling_path, policy="now")

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_total_loss = float("inf")
        for epoch in range(total_epoch):
            for data in dataset:
                self.model.train()
                self.calc_loss_func(data, "train", grad_clip_norm=grad_clip_norm)

            if eval_every and (epoch + 1) % eval_every != 0:
                continue

            self.model.eval()
            with torch.no_grad():
                if (epoch + 1) % model_save_iter == 0 or epoch == 0:
                    model_save_path = os.path.join(
                        self.model_param["model_save_path"], f"epoch{epoch}.pth"
                    )
                    torch.save(self.model.state_dict(), model_save_path)
                    wandb.save(model_save_path, policy="now")
                    print("Model saved at:", model_save_path)

                data = dataset.get_test_data()
                (total_loss, loss_idx, loss_thumb, loss_pos, loss_vel), _ = (
                    self.calc_loss_func(data, "test")
                )
                best_total_loss = min(best_total_loss, float(total_loss.detach().cpu()))
                logger_dict = {
                    "epoch": epoch + 1,
                    "total_loss": float(total_loss.detach().cpu()),
                    "loss_index": float(loss_idx.detach().cpu()),
                    "loss_thumb": float(loss_thumb.detach().cpu()),
                    "loss_pos": float(loss_pos.detach().cpu()),
                    "loss_vel": float(loss_vel.detach().cpu()),
                    "best_total_loss": best_total_loss,
                }
                wandb.log(logger_dict)
                print(
                    f"Epoch {epoch + 1}/{total_epoch} | "
                    f"total={logger_dict['total_loss']:.5f} | "
                    f"idx={logger_dict['loss_index']:.5f} | "
                    f"thumb={logger_dict['loss_thumb']:.5f} | "
                    f"pos={logger_dict['loss_pos']:.5f} | "
                    f"vel={logger_dict['loss_vel']:.5f}"
                )

        if not sweep:
            wandb.finish()

    def calc_loss_func(self, data, mode="train", grad_clip_norm=None):
        if mode == "train":
            self.optimizer.zero_grad()

        data = {
            key: val.to(self.device) if hasattr(val, "to") else val
            for key, val in data.items()
        }
        losses, preds = self.model.forward_loss(
            **data,
            loss_coef=self.loss_coef,
            cls_rate=self.mode_param.get("cls_rate"),
            noise=self.mode_param.get("noise"),
        )
        if mode == "train":
            losses[0].backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            self.optimizer.step()
        return losses, preds

    def test_controller(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("Device:", self.device)
        self.loss_coef = self.mode_param["loss_coef"]
        self.model = self._build_model()

        model_load_path = self.mode_param.get("model_load_path")
        if model_load_path:
            if not os.path.isfile(model_load_path):
                raise FileNotFoundError(f"Missing model checkpoint: {model_load_path}")
            self.model.load_state_dict(
                torch.load(model_load_path, map_location=self.device, weights_only=False)
            )
        self.model.eval()

        dataset = CustomDataLoader(self.dataset_param, self.model_param)
        with torch.no_grad():
            losses, _ = self.calc_loss_func(dataset.get_test_data(), "test")
        print(
            "test finished:",
            {
                "total_loss": float(losses[0].detach().cpu()),
                "loss_index": float(losses[1].detach().cpu()),
                "loss_thumb": float(losses[2].detach().cpu()),
                "loss_pos": float(losses[3].detach().cpu()),
                "loss_vel": float(losses[4].detach().cpu()),
            },
        )

    def pretrain_controller(self, sweep=False):
        pass

    def motion_controller(self):
        raise NotImplementedError(
            "DexWild motion-policy variants currently implement train/test only. "
            "Add a ROS bridge adapter before using motion mode."
        )

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

