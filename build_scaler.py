#!/usr/bin/env python3
import os, sys, glob, pickle, argparse, signal
import numpy as np
from typing import Any, Iterable

# -------- streaming stats (Welford) over vectors --------
class RunningStats:
    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.vmin = np.full(dim, np.inf, dtype=np.float64)
        self.vmax = np.full(dim, -np.inf, dtype=np.float64)

    def update_batch(self, x: np.ndarray):  # x: (N, dim)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got {x.shape}")
        N = x.shape[0]
        if N == 0:
            return
        # batch stats
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)  # population var
        # combine
        n_prev = self.n
        self.n += N
        delta = batch_mean - self.mean
        self.mean += delta * (N / self.n)
        self.M2 += (batch_var * N) + (delta * delta) * (n_prev * N / self.n)
        # min/max
        self.vmin = np.minimum(self.vmin, x.min(axis=0))
        self.vmax = np.maximum(self.vmax, x.max(axis=0))

    def finalize(self):
        if self.n > 1:
            std = np.sqrt(self.M2 / (self.n - 1))
        else:
            std = np.zeros_like(self.mean)
        std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "count": int(self.n),
            "mean": self.mean.astype(np.float64).tolist(),
            "std":  std.astype(np.float64).tolist(),
            "min":  self.vmin.astype(np.float64).tolist(),
            "max":  self.vmax.astype(np.float64).tolist(),
        }

# -------- robust shape normalization --------
def to_Tx90(obj: Any) -> np.ndarray:
    """
    Normalize various tactile encodings to (T, 90).
    Accepts:
      - np.ndarray (T, 90)
      - np.ndarray (T, 30, 3)
      - np.ndarray (30, 3)         -> (1, 90)
      - np.ndarray (90,)           -> (1, 90)
      - list/tuple of (30,3) frames
      - list/tuple of (90,) frames
    """
    a = obj
    # If it's a list/tuple of frames, stack them
    if isinstance(a, (list, tuple)):
        if len(a) == 0:
            return np.zeros((0, 90), dtype=np.float64)
        a0 = np.asarray(a[0])
        if a0.ndim == 1 and a0.shape[0] == 90:
            return np.asarray(a, dtype=np.float64).reshape(len(a), 90)
        if a0.ndim == 2 and a0.shape == (30, 3):
            arr = np.asarray(a, dtype=np.float64)  # (T,30,3)
            return arr.reshape(arr.shape[0], 90)
        # try to coerce each element to (90,)
        coerced = [frame.reshape(90,) if np.asarray(frame).size == 90 else None for frame in a]
        if any(v is None for v in coerced):
            raise ValueError(f"Unsupported list element shape(s) {[np.asarray(x).shape for x in a]}")
        return np.asarray(coerced, dtype=np.float64)

    # Numpy array path
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 90:      # (T,90)
        return a.astype(np.float64, copy=False)
    if a.ndim == 3 and a.shape[-2:] == (30, 3):  # (T,30,3)
        T = a.shape[0]
        return a.reshape(T, 90).astype(np.float64, copy=False)
    if a.ndim == 2 and a.shape == (30, 3):    # single frame
        return a.reshape(1, 90).astype(np.float64, copy=False)
    if a.ndim == 1 and a.shape[0] == 90:      # single frame
        return a.reshape(1, 90).astype(np.float64, copy=False)

    # try squeezing leading dims
    squeezed = np.squeeze(a)
    if squeezed.ndim == 2 and squeezed.shape == (30, 3):
        return squeezed.reshape(1, 90).astype(np.float64, copy=False)
    if squeezed.ndim == 1 and squeezed.shape[0] == 90:
        return squeezed.reshape(1, 90).astype(np.float64, copy=False)

    raise ValueError(f"Unsupported tactile shape {a.shape}; expected (T,90), (T,30,3), (30,3), or (90,)")

def iter_episode_pkls(data_root: str) -> Iterable[str]:
    # episode*/**/*.pkl
    patterns = [
        os.path.join(data_root, "episode*", "*.pkl"),
        os.path.join(data_root, "episode*", "**", "*.pkl"),
    ]
    seen = set()
    for pat in patterns:
        for fn in glob.iglob(pat, recursive=True):
            if fn in seen: continue
            seen.add(fn)
            yield fn

# -------- main build logic --------
class WarnLimiter:
    def __init__(self, limit_per_key=10):
        self.limit_per_key = limit_per_key
        self.count = {}

    def warn(self, key: str, msg: str):
        c = self.count.get(key, 0)
        if c < self.limit_per_key:
            print(f"[WARN] {key}: {msg}", file=sys.stderr)
        elif c == self.limit_per_key:
            print(f"[WARN] {key}: (further warnings suppressed)", file=sys.stderr)
        self.count[key] = c + 1

def build_scaler(data_root: str, modalities):
    stats = {m: RunningStats(90) for m in modalities}
    limiter = WarnLimiter(limit_per_key=10)
    missing = {m: 0 for m in modalities}
    total_files = 0

    for pkl_path in iter_episode_pkls(data_root):
        total_files += 1
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            limiter.warn("read", f"{pkl_path}: {e}")
            continue

        for m in modalities:
            if m not in obj:
                missing[m] += 1
                continue
            try:
                Tx90 = to_Tx90(obj[m])
                stats[m].update_batch(Tx90)
            except Exception as e:
                limiter.warn(m, f"{pkl_path}: {e}")

    scaler = {}
    for m, rs in stats.items():
        out = rs.finalize()
        scaler[m] = out
        cnt = out["count"]
        mean_all = np.mean(out["mean"]) if cnt > 0 else float("nan")
        std_max  = np.max(out["std"]) if cnt > 0 else float("nan")
        rng_max  = (np.max(out["max"]) - np.min(out["min"])) if cnt > 0 else float("nan")
        print(f"[{m}] frames={cnt}  mean≈{mean_all:.4g}  std_max≈{std_max:.4g}  range_max≈{rng_max:.4g}")

    print(f"[done] processed_files={total_files}; missing per modality: {missing}")
    return scaler

# -------- script entry --------
def main():
    ap = argparse.ArgumentParser(description="Build scaling_param.pkl from episode PKLs (robust shapes)")
    ap.add_argument("--data-root", default="/root/motionlearning/data_server/new/selftouch",
                    help="Root folder with episode0..episode100 containing PKLs")
    ap.add_argument("--out", default="/root/motionlearning/parameter/selftouch_fcn_pos/wobbly-sweep-1/scaling_param.pkl",
                    help="Destination scaling_param.pkl path")
    ap.add_argument("--modalities", nargs="+",
                    default=["tactile_index_tip", "tactile_thumb_tip"],
                    help="Modalities to compute stats for")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Graceful Ctrl-C: write partial to .partial.pkl
    partial_path = args.out + ".partial.pkl"
    interrupted = {"flag": False}
    def _sigint(_sig, _frm):
        interrupted["flag"] = True
        print("\n[INTERRUPTED] Writing partial scaler...", file=sys.stderr)
        try:
            with open(partial_path, "wb") as f:
                pickle.dump(scaler, f)  # type: ignore[name-defined]
            print(f"[✓] wrote partial: {partial_path}", file=sys.stderr)
        except Exception as e:
            print(f"[ERR] failed to write partial: {e}", file=sys.stderr)
        sys.exit(130)
    signal.signal(signal.SIGINT, _sigint)

    scaler = build_scaler(args.data_root, args.modalities)

    with open(args.out, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[✓] wrote {args.out}")

if __name__ == "__main__":
    main()
