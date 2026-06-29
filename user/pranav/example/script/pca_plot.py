import os
import sys
import glob
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
from collections import Counter

# Usage: python3 pca_plot_paperclip.py [DATA_DIR]
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/handlingteam2/HASA/user/pranav/example/data/pca_data/clip"

SIZES = ["small", "big"]
POSITIONS = ["front", "middle", "back"]
VALID_LABELS = {f"{s}_{p}" for s in SIZES for p in POSITIONS}

def _to_vector(snapshot):
    """Coerce a loaded pickle object into a 1D float vector."""
    if isinstance(snapshot, dict):
        for k in ("features", "feat", "vector", "x", "data", "tactile", "tactile_vec"):
            if k in snapshot:
                snapshot = snapshot[k]
                break
    snapshot = np.asarray(snapshot)
    if snapshot.ndim == 2 and 1 in snapshot.shape:
        snapshot = snapshot.reshape(-1)
    elif snapshot.ndim > 1:
        snapshot = snapshot.ravel()
    return snapshot.astype(float)

def _iter_class_folders(data_dir):
    """Yield (label, folder) pairs for supported structures."""
    # 1) Flat pattern: "<size>_<pos>"
    for name in os.listdir(data_dir):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        m = re.fullmatch(r"(small|big)_(front|middle|back)", name)
        if m:
            yield name, folder

    # 2) Nested pattern: "<size>/<pos>"
    for size in SIZES:
        for pos in POSITIONS:
            folder = os.path.join(data_dir, size, pos)
            if os.path.isdir(folder):
                yield f"{size}_{pos}", folder

def load_all_snapshots(data_dir):
    X, labels = [], []
    discovered = list(_iter_class_folders(data_dir))

    if not discovered:
        print("[INFO] No matching class folders found. Expected either:")
        print("       - Flat:   <DATA_DIR>/<size>_<pos>/  e.g., small_front")
        print("       - Nested: <DATA_DIR>/<size>/<pos>/  e.g., small/front/")
        print("       Sizes:", SIZES)
        print("       Positions:", POSITIONS)
        return np.array([]), []

    print("[INFO] Discovered class folders:")
    for lbl, folder in sorted(discovered):
        print(f"  {lbl:>12} -> {folder}")

    for lbl, folder in discovered:
        pkl_paths = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        if not pkl_paths:
            print(f"[WARN] No .pkl files in {folder}")
        for fpath in pkl_paths:
            try:
                with open(fpath, "rb") as f:
                    snap = pickle.load(f)
                vec = _to_vector(snap)
                if vec.ndim != 1:
                    vec = vec.ravel()
                X.append(vec)
                labels.append(lbl)
            except Exception as e:
                print(f"[ERROR] Failed to load {fpath}: {e}")

    if len(X) == 0:
        return np.array([]), []

    # Ensure same length
    lengths = {x.shape[0] for x in X}
    if len(lengths) > 1:
        print(f"[ERROR] Inconsistent feature lengths across snapshots: {sorted(lengths)}")
        print("        Make sure all snapshots have the same dimensionality.")
        return None, None

    return np.vstack(X), labels  # (N, D), list of N labels

def main():
    print(f"[INFO] DATA_DIR = {DATA_DIR}")
    X, labels = load_all_snapshots(DATA_DIR)
    if X is None:
        print("[ABORT] Inconsistent feature lengths.")
        return

    print(f"[INFO] Loaded {len(X)} samples.")
    if len(X) == 0:
        print("[HINT] No data found. Check folder names like 'small_front' or nested 'small/front',")
        print("       and ensure .pkl files are present inside.")
        return

    # Per-class counts
    cnt = Counter(labels)
    print("[INFO] Per-class counts:")
    for k in sorted(cnt):
        print(f"  {k:>12}: {cnt[k]}")

    # Need at least 2 samples for PCA
    if X.shape[0] < 2:
        print(f"[ABORT] Need at least 2 samples for PCA. Current shape: {X.shape}")
        return

    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Style: marker by size, color by position
    cmap = get_cmap("tab10")
    pos_to_color = {p: cmap(i % cmap.N) for i, p in enumerate(POSITIONS)}
    size_to_marker = {"small": "o", "big": "s"}

    plt.figure(figsize=(9, 7))
    for lbl in sorted(set(labels)):
        size, pos = lbl.split("_", 1)
        idx = [i for i, l in enumerate(labels) if l == lbl]
        plt.scatter(
            X_pca[idx, 0], X_pca[idx, 1],
            label=f"{size}-{pos}",
            color=pos_to_color.get(pos, "C0"),
            marker=size_to_marker.get(size, "o"),
            alpha=0.85, edgecolors="none"
        )

    plt.title("PCA of Paperclip Tactile Snapshots (small/big × front/middle/back)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(title="size-position", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
