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
from model.sat_rnn_pos.sat_rnn import ExternalTouchRNN
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
                                                    "loss_vel": loss_cmd,\
                                                        "best_total_loss": best_total_loss}

                wandb.log(logger_dict)

            if not sweep:
                wandb.finish()                   

    # ===========================================
    # FOR LEAD  — test_controller()
    # ===========================================
    def test_controller(self):
        import os, re, glob, shutil
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        import data_preproc as dp
        from util import get_episode
        from selftouch import SelfTouch
        from model.sat_rnn_pos.sat_rnn import ExternalTouchRNN

        # -----------------------
        # Device
        # -----------------------
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # -----------------------
        # Config (EDIT THESE)
        # -----------------------
        EPOCH       = 2699
        EPISODE_KEY = "motion"

        # 1) One episode to analyze (plots/CSV)
        #    Examples: 0.7_-20_episode0 / 1.3_0_episode5 / 2.0_20_episode10
        DIR_EPISODE = "/root/motionlearning/data_server/new/motion_all/2.0_0_episode8"

        # 2) Episodes to aggregate ONLY for PCA (parent folder with many episode folders)
        DIR_PCA     = "/root/motionlearning/data_server/new/motion_all"

        # PCA / features
        PCA_NCOMP        = 2
        RANDOM_SEED      = 42
        PCA_EQUAL_ASPECT = False  # True => same scale on x/y

        # Use the SAME initial-window length for both cases (RAW, no baseline)
        K_INIT      = 1   # 1 = exactly first frame

        # -----------------------
        # Build + load the model
        # -----------------------
        self_touch_param = dp.read_yaml(self.model_param["st_param"])["Model"]
        self.model_param = {**self_touch_param, **self.model_param}
        self.selftouch   = SelfTouch(self.model_param).to(self.device)
        self.model       = ExternalTouchRNN(self.selftouch, self.model_param).to(self.device)

        ckpt = os.path.join(self.model_param["model_save_path"], f"epoch{EPOCH}.pth")
        print(f"[ckpt] Loading: {ckpt}")
        state = torch.load(ckpt, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = {k.replace("module.", ""): v for k, v in state["model_state_dict"].items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:    print("[ckpt] Missing keys (truncated):", missing[:10])
        if unexpected: print("[ckpt] Unexpected keys (truncated):", unexpected[:10])
        self.model.eval()

        # -----------------------
        # Helpers
        # -----------------------
        def _natural_key(s: str):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

        def _try_sort_files(files):
            pats = [
                re.compile(r"timestep[_\-]?(\d+)", re.I),
                re.compile(r"step[_\-]?(\d+)", re.I),
                re.compile(r"frame[_\-]?(\d+)", re.I),
                re.compile(r"(\d+)", re.I),
            ]
            bases = [os.path.basename(f) for f in files]
            for pat in pats:
                pairs = []
                for f, b in zip(files, bases):
                    m = pat.search(b)
                    if m:
                        pairs.append((int(m.group(1)), b, f))
                if pairs:
                    pairs.sort(key=lambda x: (x[0], x[1].lower()))
                    return [p[2] for p in pairs]
            return sorted(files, key=lambda f: _natural_key(os.path.basename(f)))

        def _build_shadow_dir(src_dir):
            assert os.path.isdir(src_dir), f"[test] Not a directory: {src_dir}"
            candidates = glob.glob(os.path.join(src_dir, "*.pkl"))
            if not candidates:
                raise FileNotFoundError(f"[test] No .pkl files in: {src_dir}")
            sorted_files = _try_sort_files(candidates)
            shadow = os.path.join(src_dir, "__shadow_for_util")
            if os.path.exists(shadow):
                for g in glob.glob(os.path.join(shadow, "*")):
                    try: os.remove(g)
                    except IsADirectoryError: shutil.rmtree(g, ignore_errors=True)
            else:
                os.makedirs(shadow, exist_ok=True)
            for i, src in enumerate(sorted_files):
                dst = os.path.join(shadow, f"timestep{i:06d}.pkl")
                try:
                    if hasattr(os, "symlink"):
                        if os.path.exists(dst): os.remove(dst)
                        os.symlink(src, dst)
                    else:
                        shutil.copyfile(src, dst)
                except Exception:
                    shutil.copyfile(src, dst)
            print(f"[test] Shadow episode prepared: {shadow} ({len(sorted_files)} files)")
            return shadow

        def load_episode_compat(src_dir):
            return get_episode(_build_shadow_dir(src_dir))

        def _as_T_taxel_3(arr, expect_taxels=None):
            a = np.asarray(arr)
            if a.ndim == 3:
                T, n_taxel, three = a.shape
                assert three == 3, f"Expected last dim=3; got {a.shape}"
                if expect_taxels is not None and n_taxel != expect_taxels:
                    raise ValueError(f"Expected {expect_taxels} taxels, got {n_taxel}")
                return a
            if a.ndim == 2:
                T, D = a.shape
                if D % 3 != 0: raise ValueError(f"Cannot infer taxels from shape {a.shape} (D % 3 != 0)")
                n_taxel = D // 3
                return a.reshape(T, n_taxel, 3)
            if a.ndim == 1:
                D = a.shape[0]
                if D % 3 != 0: raise ValueError(f"Cannot infer taxels from shape {a.shape} (len % 3 != 0)")
                n_taxel = D // 3
                return a.reshape(1, n_taxel, 3)
            raise ValueError(f"Unsupported ndim={a.ndim} for tactile array")

        def _vec(t):
            return t.reshape(-1).astype(np.float32)

        def _make_output_dir(taskname):
            od = os.path.join("./log/graphs", taskname)
            os.makedirs(od, exist_ok=True)
            return od

        def moving_average(x, k=3):
            if k <= 1: return x
            c = np.convolve(x, np.ones(k)/k, mode="same")
            c[:k//2] = x[:k//2]; c[-(k//2):] = x[-(k//2):]
            return c

        def _common_ylim(*arrays):
            vals = np.concatenate([np.asarray(a).reshape(-1) for a in arrays if a is not None and len(a) > 0])
            return float(vals.min()), float(vals.max())

        # -----------------------
        # ONE episode: plots/CSV (baseline used only for these viz)
        # -----------------------
        task_name = os.path.basename(DIR_EPISODE.rstrip("/"))
        print(f"\n[test] === Episode: {task_name} ===")
        out_dir = _make_output_dir(task_name)

        data = load_episode_compat(DIR_EPISODE)
        dir_data = {EPISODE_KEY: data}
        dir_data = self.reshape_data(dir_data)

        scaling_param = dp.load_pkl_file(os.path.join(self.dataset_param["param_file_dir"], "scaling_param.pkl"))
        proc_data = dp.scale_dir_data(dir_data, scaling_param, self.dataset_param, mem="ram")
        proc_data = dp.rearrange(proc_data, self.dataset_param, mem="ram")

        # Forward pass (per-timestep)
        pred_idx_list, pred_thumb_list = [], []
        Tseq = proc_data[EPISODE_KEY]["hand_jnt_pos"].shape[0]
        for ts in range(Tseq):
            sample = {k: torch.tensor(v[ts][None]).to(self.device).float()
                    for k, v in proc_data[EPISODE_KEY].items()}
            _, _, total_idx_pred, total_thumb_pred = self.model.forward(
                sample["tactile_index_tip"],
                sample["tactile_thumb_tip"],
                sample["hand_jnt_pos"],
                None, None
            )[0]
            pred_idx_list.append(total_idx_pred.squeeze(0).detach().cpu().numpy())
            pred_thumb_list.append(total_thumb_pred.squeeze(0).detach().cpu().numpy())

        pred_idx_scaled   = np.stack(pred_idx_list, axis=0)
        pred_thumb_scaled = np.stack(pred_thumb_list, axis=0)

        # Unscale RAW (measured) and PRED (to raw units)
        raw_idx   = _as_T_taxel_3(dir_data[EPISODE_KEY]["tactile_index_tip"], expect_taxels=30)
        raw_thumb = _as_T_taxel_3(dir_data[EPISODE_KEY]["tactile_thumb_tip"], expect_taxels=30)

        def unscale(x, name):
            return dp.unscale_data(x, scaling_param[name], self.dataset_param["modality"][name])

        pred_idx_raw   = _as_T_taxel_3(unscale(pred_idx_scaled,   "tactile_index_tip"),  expect_taxels=30)
        pred_thumb_raw = _as_T_taxel_3(unscale(pred_thumb_scaled, "tactile_thumb_tip"), expect_taxels=30)

        # Baseline-subtracted versions for plots only
        initial_idx   = raw_idx[:1, :, :]
        initial_thumb = raw_thumb[:1, :, :]

        raw_idx_bs     = raw_idx   - initial_idx
        raw_thumb_bs   = raw_thumb - initial_thumb
        pred_idx_bs    = pred_idx_raw - initial_idx
        pred_thumb_bs  = pred_thumb_raw - initial_thumb
        ext_idx        = raw_idx_bs   - pred_idx_bs
        ext_thumb      = raw_thumb_bs - pred_thumb_bs

        # Per-component shared limits across both fingers
        comp_labels = ['X', 'Y', 'Z']
        per_comp_limits = {}
        for i, _ in enumerate(comp_labels):
            idx_raw_c = raw_idx_bs[:, :, i].mean(axis=1)
            idx_pr_c  = pred_idx_bs[:, :, i].mean(axis=1)
            idx_ex_c  = ext_idx[:, :, i].mean(axis=1)
            thb_raw_c = raw_thumb_bs[:, :, i].mean(axis=1)
            thb_pr_c  = pred_thumb_bs[:, :, i].mean(axis=1)
            thb_ex_c  = ext_thumb[:, :, i].mean(axis=1)
            for arr in (idx_raw_c, idx_pr_c, idx_ex_c, thb_raw_c, thb_pr_c, thb_ex_c):
                arr -= arr[0]
            per_comp_limits[i] = _common_ylim(idx_raw_c, idx_pr_c, idx_ex_c, thb_raw_c, thb_pr_c, thb_ex_c)

        os.makedirs(out_dir, exist_ok=True)

        def plot_components(raw, pred, ext, finger_label, save_prefix, per_comp_limits):
            timesteps = np.arange(raw.shape[0])
            for i, comp in enumerate(comp_labels):
                plt.figure(figsize=(15, 5))
                raw_avg  = raw[:, :, i].mean(axis=1)
                pred_avg = pred[:, :, i].mean(axis=1)
                ext_avg  = ext[:, :, i].mean(axis=1)
                raw_avg  -= raw_avg[0]; pred_avg -= pred_avg[0]; ext_avg -= ext_avg[0]
                plt.plot(timesteps, raw_avg,  label="Raw Tactile Avg", linewidth=2)
                plt.plot(timesteps, pred_avg, label="Predicted Self-Touch Avg", linestyle="--", linewidth=2)
                plt.plot(timesteps, ext_avg,  label="External Touch Avg",  linestyle="-.", linewidth=2)
                plt.title(f"{finger_label} - {comp} Component Force")
                plt.xlabel("Timestep"); plt.ylabel("Force [a.u.]")
                plt.ylim(*per_comp_limits[i])
                plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{save_prefix}_{comp.lower()}.png"))
                plt.close()

        plot_components(raw_idx_bs,   pred_idx_bs,   ext_idx,   "Index Tip", "index_components", per_comp_limits)
        plot_components(raw_thumb_bs, pred_thumb_bs, ext_thumb, "Thumb Tip", "thumb_components", per_comp_limits)

        # ----------------
        # CSV export (avg)
        # ----------------
        def avg_components_dict(t_T_30_3, label):
            compX = t_T_30_3[:, :, 0].mean(axis=1)
            compY = t_T_30_3[:, :, 1].mean(axis=1)
            compZ = t_T_30_3[:, :, 2].mean(axis=1)
            compX -= compX[0]; compY -= compY[0]; compZ -= compZ[0]
            return {f"{label}_X": np.round(compX, 3),
                    f"{label}_Y": np.round(compY, 3),
                    f"{label}_Z": np.round(compZ, 3)}

        Tloc = raw_idx_bs.shape[0]
        data_dict = {"Timestep": np.arange(Tloc)}
        data_dict.update(avg_components_dict(raw_idx_bs,     "RAW_Index"))
        data_dict.update(avg_components_dict(pred_idx_bs,    "Pred_Index"))
        data_dict.update(avg_components_dict(ext_idx,        "Ext_Index"))
        data_dict.update(avg_components_dict(raw_thumb_bs,   "RAW_Thumb"))
        data_dict.update(avg_components_dict(pred_thumb_bs,  "Pred_Thumb"))
        data_dict.update(avg_components_dict(ext_thumb,      "Ext_Thumb"))
        df_avg = pd.DataFrame(data_dict)
        csv_path = os.path.join(out_dir, "sat_tactile_avg_force_data.csv")
        df_avg.to_csv(csv_path, index=False)
        print(f"[✓] Averaged tactile CSV saved to {csv_path}")

        # =========================
        # PCA FEATURES (RAW initial window; no baseline)
        # =========================
        def _pca_caseA_feat_initial_raw(raw_idx, raw_thumb, K):
            T = raw_idx.shape[0]
            K = min(max(1, K), T)
            raw_avg = np.concatenate([raw_idx[:K], raw_thumb[:K]], axis=1).mean(axis=0)  # (60,3)
            return _vec(raw_avg)

        def _pca_caseB_feat_initial_raw(gt_i_raw, gt_t_raw, pr_i_raw, pr_t_raw, K):
            # RAW tactile ⊕ Pred self-touch
            T = gt_i_raw.shape[0]
            K = min(max(1, K), T)
            RAW0 = np.concatenate([gt_i_raw[:K], gt_t_raw[:K]], axis=1).mean(axis=0)  # (60,3)
            PR0  = np.concatenate([pr_i_raw[:K], pr_t_raw[:K]], axis=1).mean(axis=0)  # (60,3)
            return _vec(np.concatenate([RAW0, PR0], axis=0))

        # -----------------------
        # PCA collection (DIR_PCA)
        # -----------------------
        DIAMS  = ("0.7", "1.3", "2.0")
        ANGLES = ("-20", "0", "20")

        NAME_RE = re.compile(r"^(?P<dia>\d+(?:\.\d+)?)_(?P<ang>-?\d+)(?:_.*)?$", re.I)

        def _parse_labels(name):
            n = os.path.basename(name.rstrip("/")).lower()
            m = NAME_RE.match(n)
            if not m:
                return "unknown", "unknown"
            dia = m.group("dia")
            ang = m.group("ang")
            return dia, ang

        def _episode_dirs_for_pca(root_dir):
            """Return [(label, path), ...] including ONLY (dia in DIAMS) and (angle in ANGLES)."""
            if glob.glob(os.path.join(root_dir, "*.pkl")):
                name = os.path.basename(root_dir.rstrip("/"))
                dia, ang = _parse_labels(name)
                if dia in DIAMS and ang in ANGLES:
                    return [(name, root_dir)]
                return []
            pairs = []
            for sub in sorted(os.listdir(root_dir), key=_natural_key):
                p = os.path.join(root_dir, sub)
                if os.path.isdir(p) and glob.glob(os.path.join(p, "*.pkl")):
                    dia, ang = _parse_labels(sub)
                    if dia in DIAMS and ang in ANGLES:
                        pairs.append((sub, p))
            if not pairs:
                raise FileNotFoundError(f"[pca] No matching lead episodes under: {root_dir}")
            return pairs

        pca_caseA, pca_caseB, pca_labels = [], [], []
        eps_for_pca = _episode_dirs_for_pca(DIR_PCA)
        print(f"[pca] Found {len(eps_for_pca)} episode(s) in DIR_PCA (lead: all diameters & angles).")

        rng = np.random.default_rng(RANDOM_SEED)

        for lab, ep_dir in eps_for_pca:
            data_p = load_episode_compat(ep_dir)
            dd     = {EPISODE_KEY: data_p}
            dd     = self.reshape_data(dd)
            proc   = dp.scale_dir_data(dd, scaling_param, self.dataset_param, mem="ram")
            proc   = dp.rearrange(proc, self.dataset_param, mem="ram")

            pred_i, pred_t = [], []
            Tseq_p = proc[EPISODE_KEY]["hand_jnt_pos"].shape[0]
            for ts in range(Tseq_p):
                s = {k: torch.tensor(v[ts][None]).to(self.device).float()
                    for k, v in proc[EPISODE_KEY].items()}
                _, _, ti, tt = self.model.forward(
                    s["tactile_index_tip"], s["tactile_thumb_tip"], s["hand_jnt_pos"], None, None
                )[0]
                pred_i.append(ti.squeeze(0).detach().cpu().numpy())
                pred_t.append(tt.squeeze(0).detach().cpu().numpy())
            pred_i = np.stack(pred_i); pred_t = np.stack(pred_t)

            raw_i = _as_T_taxel_3(dd[EPISODE_KEY]["tactile_index_tip"],  expect_taxels=30)
            raw_t = _as_T_taxel_3(dd[EPISODE_KEY]["tactile_thumb_tip"],  expect_taxels=30)
            pr_i  = _as_T_taxel_3(unscale(pred_i, "tactile_index_tip"),  expect_taxels=30)
            pr_t  = _as_T_taxel_3(unscale(pred_t, "tactile_thumb_tip"),  expect_taxels=30)

            pca_caseA.append(_pca_caseA_feat_initial_raw(raw_i, raw_t, K_INIT))
            pca_caseB.append(_pca_caseB_feat_initial_raw(raw_i, raw_t, pr_i, pr_t, K_INIT))
            pca_labels.append(lab)

        # -----------------------
        # PCA: compute embeddings first (NO plotting yet)
        # -----------------------
        def _pca_embed(X_list, ncomp=2):
            X = np.stack(X_list, axis=0)
            Xs = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
            if Xs.shape[0] < 2:
                return None, None
            U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
            Z = Xs @ Vt[:ncomp].T
            evr = (S**2) / (Xs.shape[0] - 1)
            evr = evr / evr.sum()
            return Z, evr

        Z_A, evr_A = _pca_embed(pca_caseA, ncomp=PCA_NCOMP)
        Z_B, evr_B = _pca_embed(pca_caseB, ncomp=PCA_NCOMP)

        if Z_A is None or Z_B is None:
            print("[pca] Not enough samples for PCA; skipping.")
        else:
            x_all = np.concatenate([Z_A[:,0], Z_B[:,0]])
            y_all = np.concatenate([Z_A[:,1], Z_B[:,1]])
            pad_x = 0.05 * (x_all.max() - x_all.min() + 1e-8)
            pad_y = 0.05 * (y_all.max() - y_all.min() + 1e-8)
            X_LIM = (float(x_all.min() - pad_x), float(x_all.max() + pad_x))
            Y_LIM = (float(y_all.min() - pad_y), float(y_all.max() + pad_y))

            def _pca_plot(Z, labels, title, out_path, subtitle_extra="", xlim=None, ylim=None):
                color_map  = {"0.7": "#2A9D8F", "1.3": "#E9C46A", "2.0": "#E76F51", "unknown": "#808080"}
                marker_map = {"-20": "o", "0": "s", "20": "^", "unknown": "x"}
                plt.figure(figsize=(8,7))
                groups = {}
                for i, lab in enumerate(labels):
                    dia, ang = _parse_labels(lab)
                    key = (dia, ang)
                    groups.setdefault(key, []).append(Z[i])
                seen = set()
                for (dia, ang), pts in groups.items():
                    pts = np.asarray(pts)
                    plt.scatter(
                        pts[:,0], pts[:,1],
                        c=color_map.get(dia, "#808080"),
                        marker=marker_map.get(ang, "x"),
                        s=80, edgecolors="white", linewidths=0.6,
                        label=None if (dia, ang) in seen else f"{dia}mm, {ang}°",
                        alpha=0.95
                    )
                    seen.add((dia, ang))
                    cx, cy = pts.mean(axis=0)
                    dists = np.sqrt(((pts - np.array([cx, cy]))**2).sum(axis=1))
                    radius = float(np.percentile(dists, 65)) + 1e-6
                    circ = plt.Circle(
                        (cx, cy), radius,
                        facecolor=color_map.get(dia, "#808080"),
                        edgecolor="none",
                        alpha=0.15, zorder=0
                    )
                    plt.gca().add_patch(circ)

                plt.legend(loc="best", title="Lead (diameter, angle)")
                plt.title(f"{title}  {subtitle_extra}")
                plt.xlabel("PC1"); plt.ylabel("PC2")
                if xlim: plt.xlim(*xlim)
                if ylim: plt.ylim(*ylim)
                if PCA_EQUAL_ASPECT:
                    plt.gca().set_aspect('equal', adjustable='box')
                plt.grid(True, alpha=0.35)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                plt.tight_layout(); plt.savefig(out_path); plt.close()
                print("[✓] Saved PCA:", out_path)

            os.makedirs("./log/graphs/_pca_all", exist_ok=True)
            _pca_plot(
                Z_A, pca_labels,
                title="Lead Insertion: PCA Case A (Initial-step RAW tactile; index ⊕ thumb)",
                out_path="./log/graphs/_pca_all/lead_pca_caseA_initial_raw.png",
                xlim=X_LIM, ylim=Y_LIM
            )
            _pca_plot(
                Z_B, pca_labels,
                title="Lead Insertion: PCA Case B (Initial-step RAW tactile ⊕ Pred self-touch)",
                out_path="./log/graphs/_pca_all/lead_pca_caseB_initial_raw.png",
                xlim=X_LIM, ylim=Y_LIM
            )

        print("\n[✓] Test complete. Plots & CSV under ./log/graphs/<task>/; PCA under ./log/graphs/_pca_all/")

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
        self_touch_param=dp.read_yaml(self.model_param["st_param"])["Model"]
        self.model_param={**self_touch_param,**self.model_param}
        self.selftouch = SelfTouch(self.model_param).to(self.device)
        self.model=ExternalTouchRNN(self.selftouch,self.model_param).to(self.device)

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
        self_touch=None
        n_steps = 400
        debug_every = 20
        vel_gain = 1.0
        pos_blend = 0.35
        max_step_rad = 0.12
        if isinstance(self.mode_param, dict):
            n_steps = int(self.mode_param.get("n_steps", n_steps))
            debug_every = int(self.mode_param.get("debug_every", debug_every))
            vel_gain = float(self.mode_param.get("vel_gain", vel_gain))
            pos_blend = float(self.mode_param.get("pos_blend", pos_blend))
            max_step_rad = float(self.mode_param.get("max_step_rad", max_step_rad))
        with torch.no_grad():
            for ts in range(n_steps):
                if node_exec.ok():
                    obs = robot.get_obs()
                    dir_data={EPISODE: {key: val[None,:] for key,val in obs.items()}}
                    dir_data=self.reshape_data(dir_data)
                    dir_data = dp.scale_dir_data(dir_data, scaling_param, self.dataset_param, mem="ram")
                    dir_data = dp.rearrange(dir_data, self.dataset_param, mem="ram")
                    data = {key: torch.tensor(val).to(self.device).float() for key,val in dir_data[EPISODE].items()}
                    (jnt_pos_pred,jnt_vel_pred,_,_),\
                        self_touch, state \
                        = self.model.forward(data["tactile_index_tip"],data["tactile_thumb_tip"],\
                                                data["hand_jnt_pos"],data["hand_jnt_vel"],self_touch,state)

                    cmd_pos_norm = jnt_pos_pred.to("cpu").detach().numpy()
                    cmd_vel_norm = jnt_vel_pred.to("cpu").detach().numpy()

                    pos_norm_min, pos_norm_max = self.dataset_param["modality"]["hand_jnt_pos"][2]
                    vel_norm_min, vel_norm_max = self.dataset_param["modality"]["hand_jnt_vel"][2]
                    cmd_pos_norm = np.clip(cmd_pos_norm, pos_norm_min, pos_norm_max)
                    cmd_vel_norm = np.clip(cmd_vel_norm, vel_norm_min, vel_norm_max)

                    cmd_pos_raw = dp.unscale_data(
                        cmd_pos_norm,
                        scaling_param["hand_jnt_pos"],
                        self.dataset_param["modality"]["hand_jnt_pos"],
                    )
                    cmd_vel_raw = dp.unscale_data(
                        cmd_vel_norm,
                        scaling_param["hand_jnt_vel"],
                        self.dataset_param["modality"]["hand_jnt_vel"],
                    )

                    follower_cmd = np.asarray(obs["hand_jnt_pos"], dtype=float).copy()
                    finger_indices = [0, 1, 2, 3, 12, 13, 14, 15]
                    current_8 = np.asarray(obs["hand_jnt_pos"], dtype=float)[finger_indices]
                    dt = 1.0 / CTRL_FREQ
                    cmd_from_vel = current_8 + vel_gain * cmd_vel_raw[0] * dt
                    desired_8 = (1.0 - pos_blend) * cmd_from_vel + pos_blend * cmd_pos_raw[0]
                    delta_8 = desired_8 - current_8
                    delta_8 = np.clip(delta_8, -max_step_rad, max_step_rad)
                    final_8 = current_8 + delta_8
                    for i, idx in enumerate(finger_indices):
                        follower_cmd[idx] = final_8[i]

                    if debug_every > 0 and (ts % debug_every == 0 or ts == n_steps - 1):
                        print(
                            f"[motion-debug] step={ts} "
                            f"pos_norm[min,max]=({cmd_pos_norm.min():.4f},{cmd_pos_norm.max():.4f}) "
                            f"vel_norm[min,max]=({cmd_vel_norm.min():.4f},{cmd_vel_norm.max():.4f}) "
                            f"final_cmd[min,max]=({final_8.min():.4f},{final_8.max():.4f}) "
                            f"|step_delta|[mean,max]=({np.abs(delta_8).mean():.4f},{np.abs(delta_8).max():.4f}) "
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
