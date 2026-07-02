# FOR CLIP
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
    EPOCH       = 1899
    EPISODE_KEY = "motion"

    # 1) Episode to analyze (plots/CSV)
    DIR_EPISODE = "/root/motionlearning/data_server/clip/clip_all/small_back_episode0"

    # 2) Episodes to aggregate ONLY for PCA (parent folder with many episode folders)
    DIR_PCA     = "/root/motionlearning/data_server/clip/clip_all"

    # PCA / features
    PCA_NCOMP         = 2
    PCA_EQUAL_ASPECT  = False   # set True if you want 1 unit on x == 1 unit on y
    K_INIT            = 1       # RAW initial window (same for Case A & B)

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

    # Unscale GT and PRED (to raw units)
    gt_idx_raw   = _as_T_taxel_3(dir_data[EPISODE_KEY]["tactile_index_tip"], expect_taxels=30)
    gt_thumb_raw = _as_T_taxel_3(dir_data[EPISODE_KEY]["tactile_thumb_tip"], expect_taxels=30)

    def unscale(x, name):
        return dp.unscale_data(x, scaling_param[name], self.dataset_param["modality"][name])

    pred_idx_raw   = _as_T_taxel_3(unscale(pred_idx_scaled,   "tactile_index_tip"),  expect_taxels=30)
    pred_thumb_raw = _as_T_taxel_3(unscale(pred_thumb_scaled, "tactile_thumb_tip"), expect_taxels=30)

    # Baseline-subtracted versions for plots only (start from 0)
    initial_idx   = gt_idx_raw[:1, :, :]
    initial_thumb = gt_thumb_raw[:1, :, :]

    gt_idx_bs     = gt_idx_raw   - initial_idx
    gt_thumb_bs   = gt_thumb_raw - initial_thumb
    pred_idx_bs   = pred_idx_raw - initial_idx
    pred_thumb_bs = pred_thumb_raw - initial_thumb
    ext_idx       = gt_idx_bs   - pred_idx_bs
    ext_thumb     = gt_thumb_bs - pred_thumb_bs

    # (2) Per-component shared limits across both fingers
    comp_labels = ['X', 'Y', 'Z']
    per_comp_limits = {}
    for i, _ in enumerate(comp_labels):
        idx_gt_c = gt_idx_bs[:, :, i].mean(axis=1)
        idx_pr_c = pred_idx_bs[:, :, i].mean(axis=1)
        idx_ex_c = ext_idx[:, :, i].mean(axis=1)
        thb_gt_c = gt_thumb_bs[:, :, i].mean(axis=1)
        thb_pr_c = pred_thumb_bs[:, :, i].mean(axis=1)
        thb_ex_c = ext_thumb[:, :, i].mean(axis=1)
        for arr in (idx_gt_c, idx_pr_c, idx_ex_c, thb_gt_c, thb_pr_c, thb_ex_c):
            arr -= arr[0]
        per_comp_limits[i] = _common_ylim(idx_gt_c, idx_pr_c, idx_ex_c, thb_gt_c, thb_pr_c, thb_ex_c)

    os.makedirs(out_dir, exist_ok=True)

    def plot_components(gt, pred, ext, finger_label, save_prefix, per_comp_limits):
        timesteps = np.arange(gt.shape[0])
        for i, comp in enumerate(comp_labels):
            plt.figure(figsize=(15, 5))
            gt_avg   = gt[:, :, i].mean(axis=1)
            pred_avg = pred[:, :, i].mean(axis=1)
            ext_avg  = ext[:, :, i].mean(axis=1)
            gt_avg   -= gt_avg[0]; pred_avg -= pred_avg[0]; ext_avg -= ext_avg[0]
            plt.plot(timesteps, gt_avg,   label="Total Tactile Avg", linewidth=2)
            plt.plot(timesteps, pred_avg, label="Predicted Self-Touch Avg", linestyle="--", linewidth=2)
            plt.plot(timesteps, ext_avg,  label="External Touch Avg",  linestyle="-.", linewidth=2)
            plt.title(f"{finger_label} - {comp} Component Force")
            plt.xlabel("Timestep"); plt.ylabel("Force [a.u.]")
            plt.ylim(*per_comp_limits[i])
            plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{save_prefix}_{comp.lower()}.png"))
            plt.close()

    plot_components(gt_idx_bs,   pred_idx_bs,   ext_idx,   "Index Tip", "index_components", per_comp_limits)
    plot_components(gt_thumb_bs, pred_thumb_bs, ext_thumb, "Thumb Tip", "thumb_components", per_comp_limits)

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

    Tloc = gt_idx_bs.shape[0]
    data_dict = {"Timestep": np.arange(Tloc)}
    data_dict.update(avg_components_dict(gt_idx_bs,     "GT_Index"))
    data_dict.update(avg_components_dict(pred_idx_bs,   "Pred_Index"))
    data_dict.update(avg_components_dict(ext_idx,       "Ext_Index"))
    data_dict.update(avg_components_dict(gt_thumb_bs,   "GT_Thumb"))
    data_dict.update(avg_components_dict(pred_thumb_bs, "Pred_Thumb"))
    data_dict.update(avg_components_dict(ext_thumb,     "Ext_Thumb"))
    df_avg = pd.DataFrame(data_dict)
    csv_path = os.path.join(out_dir, "sat_tactile_avg_force_data.csv")
    df_avg.to_csv(csv_path, index=False)
    print(f"[✓] Averaged tactile CSV saved to {csv_path}")

    # =========================
    # PCA FEATURES (RAW initial window; no baseline)
    # =========================
    def _pca_caseA_feat_initial_raw(gt_idx_raw, gt_thumb_raw, K):
        T = gt_idx_raw.shape[0]
        K = min(max(1, K), T)
        raw_avg = np.concatenate([gt_idx_raw[:K], gt_thumb_raw[:K]], axis=1).mean(axis=0)  # (60,3)
        return _vec(raw_avg)

    def _pca_caseB_feat_initial_raw(gt_i_raw, gt_t_raw, pr_i_raw, pr_t_raw, K):
        T = gt_i_raw.shape[0]
        K = min(max(1, K), T)
        GT0 = np.concatenate([gt_i_raw[:K], gt_t_raw[:K]], axis=1).mean(axis=0)            # (60,3)
        PR0 = np.concatenate([pr_i_raw[:K], pr_t_raw[:K]], axis=1).mean(axis=0)            # (60,3)
        return _vec(np.concatenate([GT0, PR0], axis=0))                                     # (120,3)->flat

    # -----------------------
    # PCA collection (DIR_PCA)
    # -----------------------
    def _parse_labels(name):
        size = "big" if name.startswith("big") else ("small" if name.startswith("small") else "unknown")
        if   "back"   in name: pos = "back"
        elif "middle" in name: pos = "middle"
        elif "front"  in name: pos = "front"
        else: pos = "unknown"
        return size, pos

    def _episode_dirs_for_pca(root_dir):
        """Return [(label, path), ...] but EXCLUDE 'middle' episodes (treat as test)."""
        if glob.glob(os.path.join(root_dir, "*.pkl")):
            name = os.path.basename(root_dir.rstrip("/"))
            size, pos = _parse_labels(name)
            if pos in {"back", "front"}:
                return [(name, root_dir)]
            return []
        pairs = []
        for sub in sorted(os.listdir(root_dir), key=_natural_key):
            p = os.path.join(root_dir, sub)
            if os.path.isdir(p) and glob.glob(os.path.join(p, "*.pkl")):
                size, pos = _parse_labels(sub)
                if pos in {"back", "front"}:
                    pairs.append((sub, p))
        if not pairs:
            raise FileNotFoundError(f"[pca] No BACK/FRONT episode subfolders with .pkl under: {root_dir}")
        return pairs

    pca_caseA, pca_caseB, pca_labels = [], [], []
    eps_for_pca = _episode_dirs_for_pca(DIR_PCA)
    print(f"[pca] Found {len(eps_for_pca)} episode(s) in DIR_PCA (train-only: back/front).")

    for lab, ep_dir in eps_for_pca:
        data_p = load_episode_compat(ep_dir)
        dd     = {EPISODE_KEY: data_p}
        dd     = self.reshape_data(dd)
        proc   = dp.scale_dir_data(dd, scaling_param, self.dataset_param, mem="ram")
        proc   = dp.rearrange(proc, self.dataset_param, mem="ram")

        # forward to get predictions (scaled -> unscale)
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

        gt_i_raw = _as_T_taxel_3(dd[EPISODE_KEY]["tactile_index_tip"],  expect_taxels=30)
        gt_t_raw = _as_T_taxel_3(dd[EPISODE_KEY]["tactile_thumb_tip"],  expect_taxels=30)
        pr_i_raw = _as_T_taxel_3(unscale(pred_i, "tactile_index_tip"),  expect_taxels=30)
        pr_t_raw = _as_T_taxel_3(unscale(pred_t, "tactile_thumb_tip"), expect_taxels=30)

        # --- RAW initial-window features (no baseline), SAME K for A & B ---
        pca_caseA.append(_pca_caseA_feat_initial_raw(gt_i_raw, gt_t_raw, K_INIT))
        pca_caseB.append(_pca_caseB_feat_initial_raw(gt_i_raw, gt_t_raw, pr_i_raw, pr_t_raw, K_INIT))
        pca_labels.append(lab)

    # -----------------------
    # PCA: compute embeddings (NO plotting yet)
    # -----------------------
    def _pca_embed(X_list, ncomp=2):
        X = np.stack(X_list, axis=0)
        Xs = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
        if Xs.shape[0] < 2:
            return None, None
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        Z = Xs @ Vt[:ncomp].T     # (N, ncomp)
        evr = (S**2) / (Xs.shape[0] - 1)
        evr = evr / evr.sum()
        return Z, evr

    # --- Align B to A (fix upside-down) ---
    def _align_Z_to_ref(Z_src, Z_ref):
        Zs = np.asarray(Z_src, dtype=float)
        Zr = np.asarray(Z_ref, dtype=float)
        assert Zs.shape == Zr.shape and Zs.shape[1] == 2, f"bad shapes {Zs.shape} vs {Zr.shape}"
        cs = Zs.mean(axis=0, keepdims=True)
        cr = Zr.mean(axis=0, keepdims=True)
        X = Zs - cs
        Y = Zr - cr
        M = X.T @ Y
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt                    # allows reflection (desired)
        Zs_aligned = (Zs - cs) @ R + cr
        return Zs_aligned

    Z_A, evr_A = _pca_embed(pca_caseA, ncomp=PCA_NCOMP)
    Z_B, evr_B = _pca_embed(pca_caseB, ncomp=PCA_NCOMP)

    if Z_A is None or Z_B is None:
        print("[pca] Not enough samples for PCA; skipping.")
    else:
        Z_B_aligned = _align_Z_to_ref(Z_B, Z_A)  # <--- orientation fix

        # Shared axis limits derived from A and aligned B
        x_all = np.concatenate([Z_A[:, 0], Z_B_aligned[:, 0]])
        y_all = np.concatenate([Z_A[:, 1], Z_B_aligned[:, 1]])
        pad_x = 0.05 * (x_all.max() - x_all.min() + 1e-8)
        pad_y = 0.05 * (y_all.max() - y_all.min() + 1e-8)
        X_LIM = (float(x_all.min() - pad_x), float(x_all.max() + pad_x))
        Y_LIM = (float(y_all.min() - pad_y), float(y_all.max() + pad_y))

        def _pca_plot(Z, labels, title, out_path, subtitle_extra="", xlim=None, ylim=None):
            color_map  = {"small": "#2A9D8F", "big": "#E76F51", "unknown": "#808080"}
            marker_map = {"back": "o", "middle": "s", "front": "^", "unknown": "x"}
            plt.figure(figsize=(8, 7))

            def _parse_labels_local(name):
                return _parse_labels(name)

            groups = {}
            for i, lab in enumerate(labels):
                size, pos = _parse_labels_local(lab)
                key = (size, pos)
                groups.setdefault(key, []).append(Z[i])

            seen = set()
            ax = plt.gca()
            for (size, pos), pts in groups.items():
                pts = np.asarray(pts)
                plt.scatter(
                    pts[:, 0], pts[:, 1],
                    c=color_map.get(size, "#808080"),
                    marker=marker_map.get(pos, "x"),
                    s=80, edgecolors="white", linewidths=0.6,
                    label=None if (size, pos) in seen else f"{size}-{pos}",
                    alpha=0.95
                )
                seen.add((size, pos))
                cx, cy = pts.mean(axis=0)
                dists = np.sqrt(((pts - np.array([cx, cy]))**2).sum(axis=1))
                radius = float(np.percentile(dists, 85)) + 1e-6
                circ = plt.Circle((cx, cy), radius,
                                  facecolor=color_map.get(size, "#808080"),
                                  edgecolor="none", alpha=0.15, zorder=0)
                ax.add_patch(circ)

            plt.legend(loc="best", title="Legend")
            plt.title(f"{title}  {subtitle_extra}")
            plt.xlabel("PC1"); plt.ylabel("PC2")
            if xlim: plt.xlim(*xlim)
            if ylim: plt.ylim(*ylim)
            if PCA_EQUAL_ASPECT:
                ax.set_aspect('equal', adjustable='box')
            plt.grid(True, alpha=0.35)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.tight_layout(); plt.savefig(out_path); plt.close()
            print("[✓] Saved PCA:", out_path)

        os.makedirs("./log/graphs/_pca_all", exist_ok=True)
        _pca_plot(
            Z_A, pca_labels,
            title="Paper Clip: PCA Case A (Initial-step RAW GT; index⊕thumb)",
            out_path="./log/graphs/_pca_all/pca_caseA_initial_raw.png",
            xlim=X_LIM, ylim=Y_LIM
        )
        _pca_plot(
            Z_B_aligned, pca_labels,
            title="Paper Clip: PCA Case B (Initial-step RAW GT ⊕ Pred self-touch) — aligned to Case A",
            out_path="./log/graphs/_pca_all/pca_caseB_initial_raw.png",
            xlim=X_LIM, ylim=Y_LIM
        )

    print("\n[✓] Test complete. Plots & CSV under ./log/graphs/<task>/; PCA under ./log/graphs/_pca_all/")

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

#FOR COINS
# FOR COINS
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
    EPOCH       = 5499
    EPISODE_KEY = "motion"

    # 1) Episode to analyze (plots/CSV)
    DIR_EPISODE = "/root/motionlearning/data_server/coins/coins_all/1yen_left_episode0"

    # 2) Episodes to aggregate ONLY for PCA (parent with many episode folders)
    DIR_PCA     = "/root/motionlearning/data_server/coins/coins_all"

    # PCA / features
    PCA_NCOMP        = 2
    RANDOM_SEED      = 42
    PCA_EQUAL_ASPECT = False   # True -> square axes

    # SAME initial-window length for both cases (RAW, no baseline)
    K_INIT           = 40      # K=1 -> exactly first frame

    # --- PCA alignment & zoom controls ---
    PCA_ALIGN_B_TO_A = True    # align Case B orientation to Case A
    PCA_ZOOM_Q       = (0.10, 0.95)  # zoom to middle 70% (per-axis quantiles)
    PCA_PAD_FRAC     = 0.06    # add 6% padding after zoom crop

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

    # Unscale GT and PRED (to raw units)
    gt_idx_raw   = _as_T_taxel_3(dir_data[EPISODE_KEY]["tactile_index_tip"], expect_taxels=30)
    gt_thumb_raw = _as_T_taxel_3(dir_data[EPISODE_KEY]["tactile_thumb_tip"], expect_taxels=30)

    def unscale(x, name):
        return dp.unscale_data(x, scaling_param[name], self.dataset_param["modality"][name])

    pred_idx_raw   = _as_T_taxel_3(unscale(pred_idx_scaled,   "tactile_index_tip"),  expect_taxels=30)
    pred_thumb_raw = _as_T_taxel_3(unscale(pred_thumb_scaled, "tactile_thumb_tip"), expect_taxels=30)

    # Baseline-subtracted versions for plots only
    initial_idx   = gt_idx_raw[:1, :, :]
    initial_thumb = gt_thumb_raw[:1, :, :]

    gt_idx_bs     = gt_idx_raw   - initial_idx
    gt_thumb_bs   = gt_thumb_raw - initial_thumb
    pred_idx_bs   = pred_idx_raw - initial_idx
    pred_thumb_bs = pred_thumb_raw - initial_thumb
    ext_idx       = gt_idx_bs   - pred_idx_bs
    ext_thumb     = gt_thumb_bs - pred_thumb_bs

    # (2) Per-component shared limits across both fingers
    comp_labels = ['X', 'Y', 'Z']
    per_comp_limits = {}
    for i, _ in enumerate(comp_labels):
        idx_gt_c   = gt_idx_bs[:, :, i].mean(axis=1)
        idx_pr_c   = pred_idx_bs[:, :, i].mean(axis=1)
        idx_ex_c   = ext_idx[:, :, i].mean(axis=1)
        thb_gt_c   = gt_thumb_bs[:, :, i].mean(axis=1)
        thb_pr_c   = pred_thumb_bs[:, :, i].mean(axis=1)
        thb_ex_c   = ext_thumb[:, :, i].mean(axis=1)
        for arr in (idx_gt_c, idx_pr_c, idx_ex_c, thb_gt_c, thb_pr_c, thb_ex_c):
            arr -= arr[0]
        per_comp_limits[i] = _common_ylim(idx_gt_c, idx_pr_c, idx_ex_c, thb_gt_c, thb_pr_c, thb_ex_c)

    os.makedirs(out_dir, exist_ok=True)

    def plot_components(gt, pred, ext, finger_label, save_prefix, per_comp_limits):
        timesteps = np.arange(gt.shape[0])
        for i, comp in enumerate(comp_labels):
            plt.figure(figsize=(15, 5))
            gt_avg   = gt[:, :, i].mean(axis=1)
            pred_avg = pred[:, :, i].mean(axis=1)
            ext_avg  = ext[:, :, i].mean(axis=1)
            gt_avg   -= gt_avg[0]; pred_avg -= pred_avg[0]; ext_avg -= ext_avg[0]
            plt.plot(timesteps, gt_avg,   label="Total Tactile Avg", linewidth=2)
            plt.plot(timesteps, pred_avg, label="Predicted Self-Touch Avg", linestyle="--", linewidth=2)
            plt.plot(timesteps, ext_avg,  label="External Touch Avg",  linestyle="-.", linewidth=2)
            plt.title(f"{finger_label} - {comp} Component Force")
            plt.xlabel("Timestep"); plt.ylabel("Force [a.u.]")
            plt.ylim(*per_comp_limits[i])
            plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{save_prefix}_{comp.lower()}.png"))
            plt.close()

    plot_components(gt_idx_bs,   pred_idx_bs,   ext_idx,   "Index Tip", "index_components", per_comp_limits)
    plot_components(gt_thumb_bs, pred_thumb_bs, ext_thumb, "Thumb Tip", "thumb_components", per_comp_limits)

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

    Tloc = gt_idx_bs.shape[0]
    data_dict = {"Timestep": np.arange(Tloc)}
    data_dict.update(avg_components_dict(gt_idx_bs,     "GT_Index"))
    data_dict.update(avg_components_dict(pred_idx_bs,   "Pred_Index"))
    data_dict.update(avg_components_dict(ext_idx,       "Ext_Index"))
    data_dict.update(avg_components_dict(gt_thumb_bs,   "GT_Thumb"))
    data_dict.update(avg_components_dict(pred_thumb_bs, "Pred_Thumb"))
    data_dict.update(avg_components_dict(ext_thumb,     "Ext_Thumb"))
    df_avg = pd.DataFrame(data_dict)
    csv_path = os.path.join(out_dir, "sat_tactile_avg_force_data.csv")
    df_avg.to_csv(csv_path, index=False)
    print(f"[✓] Averaged tactile CSV saved to {csv_path}")

    # =========================
    # PCA FEATURES (RAW initial window; no baseline)
    # =========================
    def _pca_caseA_feat_initial_raw(gt_idx_raw, gt_thumb_raw, K):
        T = gt_idx_raw.shape[0]
        K = min(max(1, K), T)
        raw_avg = np.concatenate([gt_idx_raw[:K], gt_thumb_raw[:K]], axis=1).mean(axis=0)  # (60,3)
        return _vec(raw_avg)

    def _pca_caseB_feat_initial_raw(gt_i_raw, gt_t_raw, pr_i_raw, pr_t_raw, K):
        T = gt_i_raw.shape[0]
        K = min(max(1, K), T)
        GT0 = np.concatenate([gt_i_raw[:K], gt_t_raw[:K]], axis=1).mean(axis=0)            # (60,3)
        PR0 = np.concatenate([pr_i_raw[:K], pr_t_raw[:K]], axis=1).mean(axis=0)            # (60,3)
        return _vec(np.concatenate([GT0, PR0], axis=0))                                     # (120,3)->flat

    # -----------------------
    # PCA collection (DIR_PCA)
    # -----------------------
    COIN_DENOMS = ("1yen", "100yen", "500yen")
    VALID_POS   = ("left", "right")   # exclude 'middle'

    def _parse_labels(name):
        n = name.lower()
        denom = "unknown"
        for d in COIN_DENOMS:
            if n.startswith(d):
                denom = d
                break
        if   "_left"  in n or "-left"  in n: pos = "left"
        elif "_right" in n or "-right" in n: pos = "right"
        elif "_middle" in n or "-middle" in n: pos = "middle"
        else: pos = "unknown"
        return denom, pos

    def _episode_dirs_for_pca(root_dir):
        """Return [(label, path), ...] including ONLY left/right for 1/100/500yen."""
        if glob.glob(os.path.join(root_dir, "*.pkl")):
            name = os.path.basename(root_dir.rstrip("/"))
            denom, pos = _parse_labels(name)
            if denom in COIN_DENOMS and pos in VALID_POS:
                return [(name, root_dir)]
            return []
        pairs = []
        for sub in sorted(os.listdir(root_dir), key=_natural_key):
            p = os.path.join(root_dir, sub)
            if os.path.isdir(p) and glob.glob(os.path.join(p, "*.pkl")):
                denom, pos = _parse_labels(sub)
                if denom in COIN_DENOMS and pos in VALID_POS:
                    pairs.append((sub, p))
        if not pairs:
            raise FileNotFoundError(f"[pca] No LEFT/RIGHT 1/100/500yen episode subfolders with .pkl under: {root_dir}")
        return pairs

    pca_caseA, pca_caseB, pca_labels = [], [], []
    eps_for_pca = _episode_dirs_for_pca(DIR_PCA)
    print(f"[pca] Found {len(eps_for_pca)} episode(s) in DIR_PCA (coins: left/right only).")

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

        gt_i_raw = _as_T_taxel_3(dd[EPISODE_KEY]["tactile_index_tip"],  expect_taxels=30)
        gt_t_raw = _as_T_taxel_3(dd[EPISODE_KEY]["tactile_thumb_tip"],  expect_taxels=30)
        pr_i_raw = _as_T_taxel_3(unscale(pred_i, "tactile_index_tip"),  expect_taxels=30)
        pr_t_raw = _as_T_taxel_3(unscale(pred_t, "tactile_thumb_tip"), expect_taxels=30)

        pca_caseA.append(_pca_caseA_feat_initial_raw(gt_i_raw, gt_t_raw, K_INIT))
        pca_caseB.append(_pca_caseB_feat_initial_raw(gt_i_raw, gt_t_raw, pr_i_raw, pr_t_raw, K_INIT))
        pca_labels.append(lab)

    # -----------------------
    # PCA: compute embeddings (NO plotting yet)
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
        allX = np.concatenate([Z_A[:, 0], Z_B[:, 0]])
        allY = np.concatenate([Z_A[:, 1], Z_B[:, 1]])
        ql, qh = PCA_ZOOM_Q
        xlo, xhi = np.quantile(allX, [ql, qh])
        ylo, yhi = np.quantile(allY, [ql, qh])
        xpad = (xhi - xlo) * PCA_PAD_FRAC + 1e-8
        ypad = (yhi - ylo) * PCA_PAD_FRAC + 1e-8
        X_LIM = (float(xlo - xpad), float(xhi + xpad))
        Y_LIM = (float(ylo - ypad), float(yhi + ypad))

        def _pca_plot(Z, labels, title, out_path, xlim=None, ylim=None):
            color_map  = {"1yen": "#2A9D8F", "100yen": "#E9C46A", "500yen": "#E76F51", "unknown": "#808080"}
            marker_map = {"left": "o", "right": "^", "middle": "s", "unknown": "x"}
            plt.figure(figsize=(8, 7))
            groups = {}
            for i, lab in enumerate(labels):
                denom, pos = _parse_labels(lab)
                groups.setdefault((denom, pos), []).append(Z[i])
            seen = set()
            for (denom, pos), pts in groups.items():
                pts = np.asarray(pts)
                plt.scatter(
                    pts[:, 0], pts[:, 1],
                    c=color_map.get(denom, "#808080"),
                    marker=marker_map.get(pos, "x"),
                    s=80, edgecolors="white", linewidths=0.6,
                    label=None if (denom, pos) in seen else f"{denom}-{pos}",
                    alpha=0.95
                )
                seen.add((denom, pos))
                cx, cy = pts.mean(axis=0)
                dists = np.sqrt(((pts - np.array([cx, cy]))**2).sum(axis=1))
                radius = float(np.percentile(dists, 65)) + 1e-6
                circ = plt.Circle((cx, cy), radius,
                                facecolor=color_map.get(denom, "#808080"),
                                edgecolor="none", alpha=0.15, zorder=0)
                plt.gca().add_patch(circ)
            plt.legend(loc="best", title="Legend")
            plt.title(title)
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
            title="Coins: PCA Case A (Initial-step RAW GT; index ⊕ thumb) — left/right only",
            out_path="./log/graphs/_pca_all/coins_pca_caseA_initial_raw.png",
            xlim=X_LIM, ylim=Y_LIM
        )
        _pca_plot(
            Z_B, pca_labels,
            title="Coins: PCA Case B (Initial-step RAW GT ⊕ Pred self-touch) — left/right only (aligned & zoomed)",
            out_path="./log/graphs/_pca_all/coins_pca_caseB_initial_raw.png",
            xlim=X_LIM, ylim=Y_LIM
        )

    print("\n[✓] Test complete. Plots & CSV under ./log/graphs/<task>/; PCA under ./log/graphs/_pca_all/")

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

#FOR LEAD

# FOR LEAD
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
    DIR_EPISODE = "/root/motionlearning/data_server/new/motion_all/0.7_-20_episode0"

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

    # Unscale GT and PRED (to raw units)
    gt_idx_raw   = _as_T_taxel_3(dir_data[EPISODE_KEY]["tactile_index_tip"], expect_taxels=30)
    gt_thumb_raw = _as_T_taxel_3(dir_data[EPISODE_KEY]["tactile_thumb_tip"], expect_taxels=30)

    def unscale(x, name):
        return dp.unscale_data(x, scaling_param[name], self.dataset_param["modality"][name])

    pred_idx_raw   = _as_T_taxel_3(unscale(pred_idx_scaled,   "tactile_index_tip"),  expect_taxels=30)
    pred_thumb_raw = _as_T_taxel_3(unscale(pred_thumb_scaled, "tactile_thumb_tip"), expect_taxels=30)

    # Baseline-subtracted versions for plots only
    initial_idx   = gt_idx_raw[:1, :, :]
    initial_thumb = gt_thumb_raw[:1, :, :]

    gt_idx_bs     = gt_idx_raw   - initial_idx
    gt_thumb_bs   = gt_thumb_raw - initial_thumb
    pred_idx_bs   = pred_idx_raw - initial_idx
    pred_thumb_bs = pred_thumb_raw - initial_thumb
    ext_idx       = gt_idx_bs   - pred_idx_bs
    ext_thumb     = gt_thumb_bs - pred_thumb_bs

    # (2) Per-component shared limits across both fingers
    comp_labels = ['X', 'Y', 'Z']
    per_comp_limits = {}
    for i, _ in enumerate(comp_labels):
        idx_gt_c   = gt_idx_bs[:, :, i].mean(axis=1)
        idx_pr_c   = pred_idx_bs[:, :, i].mean(axis=1)
        idx_ex_c   = ext_idx[:, :, i].mean(axis=1)
        thb_gt_c   = gt_thumb_bs[:, :, i].mean(axis=1)
        thb_pr_c   = pred_thumb_bs[:, :, i].mean(axis=1)
        thb_ex_c   = ext_thumb[:, :, i].mean(axis=1)
        for arr in (idx_gt_c, idx_pr_c, idx_ex_c, thb_gt_c, thb_pr_c, thb_ex_c):
            arr -= arr[0]
        per_comp_limits[i] = _common_ylim(idx_gt_c, idx_pr_c, idx_ex_c, thb_gt_c, thb_pr_c, thb_ex_c)

    os.makedirs(out_dir, exist_ok=True)

    def plot_components(gt, pred, ext, finger_label, save_prefix, per_comp_limits):
        timesteps = np.arange(gt.shape[0])
        for i, comp in enumerate(comp_labels):
            plt.figure(figsize=(15, 5))
            gt_avg   = gt[:, :, i].mean(axis=1)
            pred_avg = pred[:, :, i].mean(axis=1)
            ext_avg  = ext[:, :, i].mean(axis=1)
            gt_avg   -= gt_avg[0]; pred_avg -= pred_avg[0]; ext_avg -= ext_avg[0]
            plt.plot(timesteps, gt_avg,   label="Total Tactile Avg", linewidth=2)
            plt.plot(timesteps, pred_avg, label="Predicted Self-Touch Avg", linestyle="--", linewidth=2)
            plt.plot(timesteps, ext_avg,  label="External Touch Avg",  linestyle="-.", linewidth=2)
            plt.title(f"{finger_label} - {comp} Component Force")
            plt.xlabel("Timestep"); plt.ylabel("Force [a.u.]")
            plt.ylim(*per_comp_limits[i])
            plt.grid(True); plt.legend(ncol=3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{save_prefix}_{comp.lower()}.png"))
            plt.close()

    plot_components(gt_idx_bs,   pred_idx_bs,   ext_idx,   "Index Tip", "index_components", per_comp_limits)
    plot_components(gt_thumb_bs, pred_thumb_bs, ext_thumb, "Thumb Tip", "thumb_components", per_comp_limits)

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

    Tloc = gt_idx_bs.shape[0]
    data_dict = {"Timestep": np.arange(Tloc)}
    data_dict.update(avg_components_dict(gt_idx_bs,     "GT_Index"))
    data_dict.update(avg_components_dict(pred_idx_bs,   "Pred_Index"))
    data_dict.update(avg_components_dict(ext_idx,       "Ext_Index"))
    data_dict.update(avg_components_dict(gt_thumb_bs,   "GT_Thumb"))
    data_dict.update(avg_components_dict(pred_thumb_bs, "Pred_Thumb"))
    data_dict.update(avg_components_dict(ext_thumb,     "Ext_Thumb"))
    df_avg = pd.DataFrame(data_dict)
    csv_path = os.path.join(out_dir, "sat_tactile_avg_force_data.csv")
    df_avg.to_csv(csv_path, index=False)
    print(f"[✓] Averaged tactile CSV saved to {csv_path}")

    # =========================
    # PCA FEATURES (RAW initial window; no baseline)
    # =========================
    def _pca_caseA_feat_initial_raw(gt_idx_raw, gt_thumb_raw, K):
        T = gt_idx_raw.shape[0]
        K = min(max(1, K), T)
        raw_avg = np.concatenate([gt_idx_raw[:K], gt_thumb_raw[:K]], axis=1).mean(axis=0)  # (60,3)
        return _vec(raw_avg)

    def _pca_caseB_feat_initial_raw(gt_i_raw, gt_t_raw, pr_i_raw, pr_t_raw, K):
        T = gt_i_raw.shape[0]
        K = min(max(1, K), T)
        GT0 = np.concatenate([gt_i_raw[:K], gt_t_raw[:K]], axis=1).mean(axis=0)            # (60,3)
        PR0 = np.concatenate([pr_i_raw[:K], pr_t_raw[:K]], axis=1).mean(axis=0)            # (60,3)
        return _vec(np.concatenate([GT0, PR0], axis=0))                                     # (120,3)->flat

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

        gt_i_raw = _as_T_taxel_3(dd[EPISODE_KEY]["tactile_index_tip"],  expect_taxels=30)
        gt_t_raw = _as_T_taxel_3(dd[EPISODE_KEY]["tactile_thumb_tip"],  expect_taxels=30)
        pr_i_raw = _as_T_taxel_3(unscale(pred_i, "tactile_index_tip"),  expect_taxels=30)
        pr_t_raw = _as_T_taxel_3(unscale(pred_t, "tactile_thumb_tip"), expect_taxels=30)

        pca_caseA.append(_pca_caseA_feat_initial_raw(gt_i_raw, gt_t_raw, K_INIT))
        pca_caseB.append(_pca_caseB_feat_initial_raw(gt_i_raw, gt_t_raw, pr_i_raw, pr_t_raw, K_INIT))
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
            title="Lead Insertion: PCA Case A (Initial-step RAW GT; index ⊕ thumb)",
            out_path="./log/graphs/_pca_all/lead_pca_caseA_initial_raw.png",
            xlim=X_LIM, ylim=Y_LIM
        )
        _pca_plot(
            Z_B, pca_labels,
            title="Lead Insertion: PCA Case B (Initial-step RAW GT ⊕ Pred self-touch)",
            out_path="./log/graphs/_pca_all/lead_pca_caseB_initial_raw.png",
            xlim=X_LIM, ylim=Y_LIM
        )

    print("\n[✓] Test complete. Plots & CSV under ./log/graphs/<task>/; PCA under ./log/graphs/_pca_all/")
