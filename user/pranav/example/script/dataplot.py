#!/usr/bin/env python3
import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Tuple

# ====== set your episode dir + freq ======
EPISODE_DIR = "/home/handlingteam2/HASA/user/pranav/example/data/new/2.0_-20/episode0/"
CTRL_FREQ = 10.0
# ========================================

# ---------- utils ----------
def list_timesteps(episode_dir: str) -> List[str]:
    """Find timestep*.pkl and sort by the integer index; fallback=mtime."""
    pkls = glob.glob(os.path.join(episode_dir, "*.pkl"))
    if not pkls:
        return []
    def key(p: str) -> Tuple[int, float]:
        m = re.search(r"timestep(\d+)\.pkl$", os.path.basename(p))
        return (int(m.group(1)) if m else 10**9, os.path.getmtime(p))
    return [p for p in sorted(pkls, key=key)]

def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def iter_arrays(obj: Any, prefix: str = "") -> Iterator[Tuple[str, np.ndarray]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            newp = f"{prefix}.{k}" if prefix else str(k)
            yield from iter_arrays(v, newp)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            newp = f"{prefix}[{i}]"
            yield from iter_arrays(v, newp)
    elif isinstance(obj, np.ndarray):
        yield prefix or "__array__", obj

def score_name(name: str, want: str) -> int:
    n = name.lower()
    s = 0
    if "tactile" in n or "taxel" in n or "xela" in n: s += 2
    if "raw" in n: s += 1
    if "pos" in n or "position" in n or "q" in n: s += 2
    if want == "index":
        if "index" in n or "idx" in n: s += 5
        if "thumb" in n: s -= 2
        if "tip" in n: s += 1
    elif want == "thumb":
        if "thumb" in n: s += 5
        if "index" in n or "idx" in n: s -= 2
        if "tip" in n: s += 1
    elif want == "joint":
        if any(k in n for k in ["joint","jnt","hand","allegro"]): s += 5
        if any(k in n for k in ["tactile","taxel","tip","xela"]): s -= 2
    return s

def best_in_this_pickle(obj: Any, want: str) -> np.ndarray:
    """Pick ONE array from this pickle for the requested signal."""
    candidates: Dict[str, np.ndarray] = {}
    for name, arr in iter_arrays(obj):
        if isinstance(arr, np.ndarray) and arr.size > 0:
            sc = score_name(name, want)
            if want == "joint" and arr.ndim == 1 and 8 <= arr.size <= 32:
                sc += 2
            if want in ("index","thumb") and arr.ndim in (1,2,3):
                sc += 1
            if sc >= 3:
                candidates[name] = arr
    if not candidates:
        raise KeyError(f"No '{want}' candidate found in this pickle")
    # choose the “most informative” array (prefer bigger, last-dim==3 for tactile)
    def cand_key(k: str) -> Tuple[int,int,int]:
        a = candidates[k]
        last3 = 1 if (a.ndim >= 1 and a.shape[-1] == 3) else 0
        return (a.ndim, a.size, last3)
    best_name = max(candidates.keys(), key=cand_key)
    return candidates[best_name]

def coerce_tactile_step(a: np.ndarray) -> np.ndarray:
    """
    Convert a single-step tactile to shape (N_taxels, 3) or (1,3) as best effort.
    Common: (N,3) already; (3,) -> (1,3); (N,) -> (N,1).
    """
    A = np.asarray(a)
    if A.ndim == 2 and A.shape[-1] == 3:
        return A  # (N,3)
    if A.ndim == 1 and A.size == 3:
        return A[None, :]  # (1,3)
    if A.ndim == 3 and A.shape[-1] == 3 and A.shape[0] == 1:
        return A[0]  # (1,3) or (N,3)
    if A.ndim == 1:
        return A[:, None]  # (N,1)
    if A.ndim == 2:
        return A  # (N,?) … will average later
    # fallback: flatten sensors to (N,1)
    return A.reshape(-1, 1)

def collapse_tactile_to_xyz_mean(A_stack: List[np.ndarray]) -> np.ndarray:
    """
    Given list of per-step tactile arrays (N_taxels,3) or (N,1/?) -> return (T,3) or (T,1)
    by averaging across taxels per axis.
    """
    outs = []
    is_xyz = None
    for A in A_stack:
        if A.ndim == 2 and A.shape[-1] == 3:
            outs.append(np.nanmean(A, axis=0))   # (3,)
            is_xyz = True if is_xyz is None else is_xyz
        else:
            outs.append(np.nanmean(A))           # scalar
            is_xyz = False if is_xyz is None else is_xyz
    Y = np.stack(outs, axis=0)
    if not is_xyz:
        Y = Y[:, None]  # (T,1)
    return Y  # (T,3) or (T,1)

def coerce_joint_step(a: np.ndarray) -> np.ndarray:
    """Single step joints -> (D,)"""
    A = np.asarray(a)
    if A.ndim == 1:
        return A
    if A.ndim >= 2:
        return A.reshape(-1)  # flatten just in case
    return A

# ---------- main ----------
def main():
    files = list_timesteps(EPISODE_DIR)
    if not files:
        print(f"[Error] No timestep*.pkl in: {EPISODE_DIR}")
        return
    print(f"[Info] Found {len(files)} timesteps in {EPISODE_DIR}")

    idx_steps: List[np.ndarray] = []
    th_steps:  List[np.ndarray] = []
    jp_steps:  List[np.ndarray] = []

    missing = {"index":0, "thumb":0, "joint":0}

    for p in files:
        try:
            obj = load_pickle(p)
        except Exception as e:
            print(f"[Warn] skip {os.path.basename(p)}: {e}")
            continue

        # index
        try:
            a = best_in_this_pickle(obj, "index")
            idx_steps.append(coerce_tactile_step(a))
        except Exception:
            missing["index"] += 1

        # thumb
        try:
            a = best_in_this_pickle(obj, "thumb")
            th_steps.append(coerce_tactile_step(a))
        except Exception:
            missing["thumb"] += 1

        # joints
        try:
            a = best_in_this_pickle(obj, "joint")
            jp_steps.append(coerce_joint_step(a))
        except Exception:
            missing["joint"] += 1

    T = len(files)
    print(f"[Info] collected steps — index:{len(idx_steps)}/{T}, thumb:{len(th_steps)}/{T}, joints:{len(jp_steps)}/{T}")
    if any(missing.values()):
        print(f"[Note] missing frames -> {missing}")

    # Build time axis
    t = np.arange(T) / CTRL_FREQ

    # Tactile: collapse to mean X/Y/Z (or scalar) per step -> (T,3) or (T,1)
    idx_plot = collapse_tactile_to_xyz_mean(idx_steps) if idx_steps else None
    th_plot  = collapse_tactile_to_xyz_mean(th_steps)  if th_steps  else None

    # Joints: stack each step vector -> (T,D)
    jp_plot = np.stack(jp_steps, axis=0) if jp_steps else None

    # Truncate all to same length for plotting
    lengths = [arr.shape[0] for arr in [idx_plot, th_plot, jp_plot] if arr is not None]
    if not lengths:
        print("[Error] Nothing to plot.")
        return
    Tmin = min(lengths)
    if idx_plot is not None: idx_plot = idx_plot[:Tmin]
    if th_plot  is not None: th_plot  = th_plot[:Tmin]
    if jp_plot  is not None: jp_plot  = jp_plot[:Tmin]
    t = t[:Tmin]

    # ---- Plot (single figure, 3 stacked axes) ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax1, ax2, ax3 = axes

    if idx_plot is not None:
        if idx_plot.shape[1] == 3:
            ax1.plot(t, idx_plot[:,0], label="Index X")
            ax1.plot(t, idx_plot[:,1], label="Index Y")
            ax1.plot(t, idx_plot[:,2], label="Index Z")
        else:
            ax1.plot(t, idx_plot[:,0], label="Index mean")
        ax1.set_title("Index Tip Tactile (per-step mean)")
        ax1.set_ylabel("Tactile"); ax1.grid(True); ax1.legend()

    if th_plot is not None:
        if th_plot.shape[1] == 3:
            ax2.plot(t, th_plot[:,0], label="Thumb X")
            ax2.plot(t, th_plot[:,1], label="Thumb Y")
            ax2.plot(t, th_plot[:,2], label="Thumb Z")
        else:
            ax2.plot(t, th_plot[:,0], label="Thumb mean")
        ax2.set_title("Thumb Tip Tactile (per-step mean)")
        ax2.set_ylabel("Tactile"); ax2.grid(True); ax2.legend()

    if jp_plot is not None:
        for j in range(jp_plot.shape[1]):
            ax3.plot(t, jp_plot[:, j], alpha=0.7 if jp_plot.shape[1] <= 16 else 0.3)
        ax3.set_title(f"Joint Positions (D={jp_plot.shape[1]})")
        ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Joint pos"); ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
