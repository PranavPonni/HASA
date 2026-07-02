import sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from controller_base import AbstractController
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from t_rnn import ExternalTouchRNN
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
import pdb


class RNN_controller(AbstractController):
    def __init__(self, model_param, mode_param, dataset_param,config_param):
        super().__init__(model_param, mode_param, dataset_param,config_param)
        torch.cuda.init()

    def train_controller(self, sweep=False):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print("Device: ", self.device)

        batch_size = self.mode_param["batch_size"]
        total_epoch = self.mode_param["num_epochs"]
        lr = self.mode_param["lr"]
        model_save_iter = self.mode_param["model_save_iter"]

        self.sequence_length = self.dataset_param["sequence_length"]
        self.shift_data = self.dataset_param["shift_data"]
        self.loss_coef = self.mode_param["loss_coef"]
        
        self.model=ExternalTouchRNN(self.model_param).to(self.device)

        if not sweep:
            config={**self.model_param,**self.mode_param["loss_coef"]}
            wandb.init(project=self.mode_param["project"],config=config)
            wandb.watch(self.model,log="all",log_freq=10)
        else:
            wandb.watch(self.model,log="all",log_freq=10)
            
        dataset = CustomDataLoader(self.dataset_param,self.model_param)
        dataset.set_batch_size(batch_size)
        dataset.set_shuffle(True)
        wandb.save(os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"),policy="now")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        best_total_loss=1e10

        for epoch in range(total_epoch):
            for data in dataset:
                self.model.train()
                self.calc_loss_func(data,"train")

            self.model.eval()
            with torch.no_grad():
                logger_dict={}
                if (epoch + 1) % model_save_iter == 0 or epoch==0:
                    model_save_path = os.path.join(self.model_param["model_save_path"],f"epoch{epoch}.pth")
                    torch.save(self.model.state_dict(), model_save_path)
                    wandb.save(model_save_path,policy="now")
                    print("Model saved at:", model_save_path) 

                data = dataset.get_test_data()

                (total_loss,loss_tactile_idx,loss_tactile_thumb,loss_pos,loss_cmd),\
                _ = self.calc_loss_func(data,"test")
                
                if best_total_loss>total_loss:
                    best_total_loss=total_loss

                logger_dict={"total_loss": total_loss,
                                "loss_index": loss_tactile_idx,
                                    "loss_thumb": loss_tactile_thumb,
                                            "loss_pos": loss_pos,
                                                    "loss_vel": loss_cmd,\
                                                        "best_total_loss": best_total_loss}

                wandb.log(logger_dict)

            if not sweep:
                wandb.finish()                   

    def test_controller(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import torch

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Load test episode
        from util import get_episode
        DIR = "/root/motionlearning/data_server/stiff_pd/motion_all/2.0_0_episode30"
        data = get_episode(DIR)
        EPISODE = "motion"
        dir_data = {EPISODE: data}
        dir_data = self.reshape_data(dir_data)

        # Unscale the tactile observations
        import data_preproc as dp
        scaling_param = dp.load_pkl_file(os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"))
        unscale = lambda x, name: dp.unscale_data(x, scaling_param[name], self.dataset_param["modality"][name])
        index_raw = unscale(dir_data[EPISODE]["tactile_index_tip"], "tactile_index_tip").reshape(-1, 30, 3)
        thumb_raw = unscale(dir_data[EPISODE]["tactile_thumb_tip"], "tactile_thumb_tip").reshape(-1, 30, 3)

        timesteps = np.arange(index_raw.shape[0])
        os.makedirs("./log/graphs/t_rnn_xyz_combined", exist_ok=True)

        def plot_component(data, label, tag):
            comp_names = ['X', 'Y', 'Z']
            for i in range(3):
                comp = data[:, :, i]  # shape (T, 30)
                avg = comp.mean(axis=1)  # shape (T,)
                plt.figure(figsize=(15, 4))
                
                # Plot all 30 raw force curves (one per taxel)
                for j in range(30):
                    plt.plot(timesteps, comp[:, j], alpha=0.5, linewidth=0.8)

                # Overlay average line
                plt.plot(timesteps, avg, label=f"Avg {comp_names[i]}-Force", color="black", linewidth=2.5)

                plt.title(f"T-RNN: {comp_names[i]}-Force ({label} Tip)")
                plt.xlabel("Time Step")
                plt.ylabel(f"{comp_names[i]} Force")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"./log/trnn_xyz_combined{tag.lower()}_{comp_names[i].lower()}.png")
                plt.close()

        # Plot all components
        plot_component(index_raw, "Index", "index")
        plot_component(thumb_raw, "Thumb", "thumb")

        # Save CSV
        df = pd.DataFrame({
            "Timestep": timesteps,
            "Index_X_Avg": index_raw[:, :, 0].mean(axis=1),
            "Index_Y_Avg": index_raw[:, :, 1].mean(axis=1),
            "Index_Z_Avg": index_raw[:, :, 2].mean(axis=1),
            "Thumb_X_Avg": thumb_raw[:, :, 0].mean(axis=1),
            "Thumb_Y_Avg": thumb_raw[:, :, 1].mean(axis=1),
            "Thumb_Z_Avg": thumb_raw[:, :, 2].mean(axis=1),
        })
        df.to_csv("./log/graphs/t_rnn_xyz_combined/trnn_component_timeseries.csv", index=False)

        print("[✓] T-RNN: Component-wise tactile plots + CSV saved in ./log/")

    def pretrain_controller(self,sweep=False):
        pass

    def motion_controller(self):
        from ros_bridge import XelAllegro
        from py_node_exec import NodeExec
        import re
        # Parameter for Pranav
        CTRL_FREQ=10.
        EPOCH=5999
        node_exec = NodeExec(node_name="data_collection", freq=CTRL_FREQ)
        node_exec.spin_thread_start()

        robot = XelAllegro(
            ctrl_freq=CTRL_FREQ,
            hand_topic_prefix="allegroHand_0",
            tactile_topic_prefix=["index_tip", "middle_tip", "ring_tip", "thumb_tip"]
        )
        robot.torque_on()
        connected = robot.wait_for_cmd_connection(timeout_sec=5.0)
        conn_count = robot.get_cmd_connection_count()
        if not connected:
            print(f"[WARN] No subscribers on joint_cmd topic (connections={conn_count}). Commands may not move the hand.")
        else:
            print(f"[INFO] joint_cmd connections: {conn_count}")

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print("Device: ", self.device)

        # loading model
        self.model=ExternalTouchRNN(self.model_param).to(self.device)

        model_load_path = None
        if isinstance(self.mode_param, dict):
            model_load_path = self.mode_param.get("model_load_path")

        if model_load_path and os.path.isfile(model_load_path):
            resolved_model_path = model_load_path
        else:
            model_dir = self.model_param["model_save_path"]
            preferred_path = os.path.join(model_dir, f"epoch{EPOCH}.pth")
            if os.path.isfile(preferred_path):
                resolved_model_path = preferred_path
            else:
                if not os.path.isdir(model_dir):
                    raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
                epoch_files = []
                for name in os.listdir(model_dir):
                    match = re.fullmatch(r"epoch(\d+)\.pth", name)
                    if match:
                        epoch_files.append((int(match.group(1)), name))
                if not epoch_files:
                    raise FileNotFoundError(
                        f"No checkpoint files found in {model_dir}. Expected files like epoch<N>.pth"
                    )
                last_epoch, last_name = max(epoch_files, key=lambda x: x[0])
                resolved_model_path = os.path.join(model_dir, last_name)
                print(
                    f"[WARN] Missing {preferred_path}. Using latest available checkpoint: "
                    f"{resolved_model_path} (epoch {last_epoch})"
                )

        print(f"Loading model checkpoint: {resolved_model_path}")
        model_state=torch.load(resolved_model_path,map_location=self.device,weights_only=False)
        self.model.eval()
        self.model.load_state_dict(model_state)

        # loading parameter
        scaling_param=dp.load_pkl_file(os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"))
        # make the robot go to initial position
        robot.move_to_initial()
        EPISODE="motion"
        state=None
        n_steps = 400
        debug_every = 20
        if isinstance(self.mode_param, dict):
            n_steps = int(self.mode_param.get("n_steps", n_steps))
            debug_every = int(self.mode_param.get("debug_every", debug_every))
        with torch.no_grad():
            for ts in range(n_steps):
                if node_exec.ok():
                    obs = robot.get_obs()
                    dir_data={EPISODE: {key: val[None,:] for key,val in obs.items()}}
                    dir_data=self.reshape_data(dir_data)
                    dir_data = dp.scale_dir_data(dir_data, scaling_param, self.dataset_param, mem="ram")
                    dir_data = dp.rearrange(dir_data, self.dataset_param, mem="ram")
                    data = {key: torch.tensor(val).to(self.device).float() for key,val in dir_data[EPISODE].items()}
                    (_,jnt_cmd_pos_pred,_,_),\
                        state \
                        = self.model.forward(data["tactile_index_tip"],data["tactile_thumb_tip"],\
                                                data["hand_jnt_pos"],state)

                    cmd_pos_norm = jnt_cmd_pos_pred.to("cpu").detach().numpy()
                    cmd_pos_raw = dp.unscale_data(
                        cmd_pos_norm,
                        scaling_param["hand_jnt_pos"],
                        self.dataset_param["modality"]["hand_jnt_pos"],
                    )

                    follower_cmd = np.asarray(obs["hand_jnt_pos"], dtype=float).copy()
                    finger_indices = [0, 1, 2, 3, 12, 13, 14, 15]
                    for i, idx in enumerate(finger_indices):
                        follower_cmd[idx] = cmd_pos_raw[0][i]

                    if debug_every > 0 and (ts % debug_every == 0 or ts == n_steps - 1):
                        current_8 = np.asarray(obs["hand_jnt_pos"], dtype=float)[finger_indices]
                        cmd_8 = cmd_pos_raw[0]
                        delta_8 = cmd_8 - current_8
                        print(
                            f"[motion-debug] step={ts} "
                            f"cmd_norm[min,max]=({cmd_pos_norm.min():.4f},{cmd_pos_norm.max():.4f}) "
                            f"cmd_raw[min,max]=({cmd_8.min():.4f},{cmd_8.max():.4f}) "
                            f"|delta|[mean,max]=({np.abs(delta_8).mean():.4f},{np.abs(delta_8).max():.4f}) "
                            f"conn={robot.get_cmd_connection_count()}"
                        )

                    robot.set_jnt_cmd(follower_cmd)
                    node_exec.sleep()
                else:
                    print("Node execution stopped.")
                    break
                print("step",ts)



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

    
        
    def calc_loss_func(self,data,mode="train"):

        if mode=="train":
            self.optimizer.zero_grad()

        data={key: val.to(self.device) for key, val in data.items()}
        (total_loss,loss_tactile_idx,loss_tactile_thumb,loss_pos,loss_cmd),\
               (idx_preds,thumb_preds,pos_preds,cmd_preds)\
                =self.model.forward_loss(**data,loss_coef=self.loss_coef,cls_rate=self.mode_param["cls_rate"],noise=self.mode_param["noise"])
        if mode=="train":
            total_loss.backward()
            self.optimizer.step()  

        return (total_loss,loss_tactile_idx,loss_tactile_thumb,loss_pos,loss_cmd),\
               (idx_preds,thumb_preds,pos_preds,cmd_preds)

    def reshape_data(self,dir_data):
        for ep, val in dir_data.items():
            dir_data[ep]["tactile_index_tip"]=einops.rearrange(val["tactile_index_tip"],"t a d -> t (a d)")
            dir_data[ep]["tactile_thumb_tip"]=einops.rearrange(val["tactile_thumb_tip"],"t a d -> t (a d)")
            dir_data[ep]["hand_jnt_pos"]=val["hand_jnt_pos"][:,[0, 1, 2, 3, 12, 13, 14, 15]]
            dir_data[ep]["hand_jnt_vel"]=val["hand_jnt_vel"][:,[0, 1, 2, 3, 12, 13, 14, 15]]
        return dir_data
