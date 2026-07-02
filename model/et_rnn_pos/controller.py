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
from model.et_rnn_pos.et_rnn import ExternalTouchRNN
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
        
        self_touch_param=dp.read_yaml(self.model_param["st_param"])["Model"]
        self.model_param={**self_touch_param,**self.model_param}
        self.selftouch = SelfTouch(self.model_param).to(self.device)
        model_state=torch.load(self.model_param["st_model"],map_location=self.device,weights_only=False)
        self.selftouch.load_state_dict(model_state)
        self.model=ExternalTouchRNN(self.selftouch,self.model_param).to(self.device)

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
                                                    "loss_cmd": loss_cmd,\
                                                        "best_total_loss": best_total_loss}

                wandb.log(logger_dict)

            if not sweep:
                wandb.finish()                   

    def test_controller(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        from util import get_episode
        os.makedirs("./log", exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Load config
        EPOCH = 4999
        EPISODE = "motion"
        DIR = "/root/motionlearning/data_server/stiff_pd/motion_all/2.0_0_episode30"

        # Load data
        data = get_episode(DIR)
        dir_data = {EPISODE: data}
        dir_data = self.reshape_data(dir_data)

        # Load scale params
        scaling_param = dp.load_pkl_file(os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"))
        proc_data = dp.scale_dir_data(dir_data, scaling_param, self.dataset_param, mem="ram")
        proc_data = dp.rearrange(proc_data, self.dataset_param, mem="ram")

        # Load model
        self_touch_param = dp.read_yaml(self.model_param["st_param"])["Model"]
        self.model_param = {**self_touch_param, **self.model_param}
        self.selftouch = SelfTouch(self.model_param).to(self.device)
        self.model = ExternalTouchRNN(self.selftouch, self.model_param).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.model_param["model_save_path"], f"epoch{EPOCH}.pth"), map_location=self.device))
        self.model.eval()

        # Inference
        pred_ext_idx, pred_ext_thumb = [], []
        for ts in range(proc_data[EPISODE]["hand_jnt_pos"].shape[0]):
            sample = {k: torch.tensor(v[ts][None]).float().to(self.device) for k, v in proc_data[EPISODE].items()}
            idx, thumb = self.model(
                sample["tactile_index_tip"],
                sample["tactile_thumb_tip"],
                sample["hand_jnt_pos"]
            )[0]
            pred_ext_idx.append(idx.squeeze(0).detach().cpu().numpy())
            pred_ext_thumb.append(thumb.squeeze(0).detach().cpu().numpy())

        # Unscale
        def unscale(x, name): return dp.unscale_data(x, scaling_param[name], self.dataset_param["modality"][name])
        gt_idx = unscale(dir_data[EPISODE]["tactile_index_tip"], "tactile_index_tip")
        gt_thumb = unscale(dir_data[EPISODE]["tactile_thumb_tip"], "tactile_thumb_tip")
        pred_ext_idx = unscale(np.stack(pred_ext_idx), "tactile_index_tip")
        pred_ext_thumb = unscale(np.stack(pred_ext_thumb), "tactile_thumb_tip")

        def norm(x): return np.linalg.norm(x.reshape(x.shape[0], 30, 3), axis=2).mean(axis=1)
        t = np.arange(len(gt_idx))

        def plot(y1, y2, label1, label2, title, fname):
            plt.figure(figsize=(15, 4))
            plt.plot(t, y1, label=label1)
            plt.plot(t, y2, label=label2)
            plt.title(title)
            plt.xlabel("Timestep")
            plt.ylabel("Avg Force")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

        plot(norm(gt_idx), norm(pred_ext_idx), "GT", "External", "ET-RNN: Index - External", "./log/et_index_ext.png")
        plot(norm(gt_thumb), norm(pred_ext_thumb), "GT", "External", "ET-RNN: Thumb - External", "./log/et_thumb_ext.png")

        pd.DataFrame({
            "Timestep": t,
            "GT_Index": norm(gt_idx),
            "External_Index": norm(pred_ext_idx),
            "GT_Thumb": norm(gt_thumb),
            "External_Thumb": norm(pred_ext_thumb),
        }).to_csv("./log/et_timeseries.csv", index=False)

        print("[✓] ET-RNN: saved plots and CSV in ./log/")

    def pretrain_controller(self,sweep=False):
        pass

    def motion_controller(self):
        from ros_bridge import XelAllegro
        from py_node_exec import NodeExec
        # Parameter for Pranav
        CTRL_FREQ=10.
        EPOCH=999
        node_exec = NodeExec(node_name="data_collection", freq=CTRL_FREQ)
        node_exec.spin_thread_start()

        robot = XelAllegro(
            ctrl_freq=CTRL_FREQ,
            hand_topic_prefix="allegroHand_0",
            tactile_topic_prefix=["index_tip","thumb_tip"]
        )

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print("Device: ", self.device)

        # loading model
        self_touch_param=dp.read_yaml(self.model_param["st_param"])["Model"]
        self.model_param={**self_touch_param,**self.model_param}
        self.selftouch = SelfTouch(self.model_param).to(self.device)
        self.model=ExternalTouchRNN(self.selftouch,self.model_param).to(self.device)
        model_state=torch.load(os.path.join(self.model_param["model_save_path"],f"epoch{EPOCH}.pth"),map_location=self.device,weights_only=False)
        self.model.eval()
        self.model.load_state_dict(model_state)

        # loading parameter
        scaling_param=dp.load_pkl_file(os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"))
        # make the robot go to initial position
        robot.move_to_initial()
        EPISODE="motion"
        state=None
        self_touch=None
        with torch.no_grad():
            for ts in range(200):
                if node_exec.ok():
                    obs = robot.get_obs()
                    dir_data={EPISODE: {key: val[None,:] for key,val in obs.items()}}
                    dir_data=self.reshape_data(dir_data)
                    dir_data = dp.scale_dir_data(dir_data, scaling_param, self.dataset_param, mem="ram")
                    dir_data = dp.rearrange(dir_data, self.dataset_param, mem="ram")
                    data = {key: torch.tensor(val).to(self.device).float() for key,val in dir_data[EPISODE].items()}
                    (_,_,_,jnt_cmd_pos_pred,_,_),\
                        self_touch, state \
                        = self.model.forward(data["tactile_index_tip"],data["tactile_thumb_tip"],\
                                                data["hand_jnt_pos"],self_touch,state)
                    pred = {}
                    pred[EPISODE] = {}
                    pred[EPISODE]["hand_jnt_cmd_pos"] = jnt_cmd_pos_pred.to("cpu").detach().numpy()
                    
                    pred = dp.restore(pred, self.dataset_param)
                    pred = dp.unscale_dir_data(pred, scaling_param, self.dataset_param)

                    follower_cmd = np.zeros(16, dtype=float)
                    finger_indices = [0, 1, 2, 3, 12, 13, 14, 15]
                    for i, idx in enumerate(finger_indices):
                        follower_cmd[idx] = pred[EPISODE]["hand_jnt_cmd_pos"][0][i]
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
            dir_data[ep]["hand_jnt_cmd_pos"]=val["hand_jnt_cmd_pos"][:,[0, 1, 2, 3, 12, 13, 14, 15]]
        return dir_data
