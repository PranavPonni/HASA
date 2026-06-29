# collect_pca_data.py

import os
import time
import numpy as np
import pickle
from py_node_exec import NodeExec
from xela_py import TactileSubscriber

DATA_DIR = "/home/handlingteam2/HASA/user/pranav/example/data/pca_data/small_front"
ANGLE = "-40"  # ← Change this for each run: e.g., "10", "-20", etc.

def main():
    save_path = os.path.join(DATA_DIR, ANGLE)
    os.makedirs(save_path, exist_ok=True)

    node = NodeExec(freq=0.1)
    node.spin_thread_start()
    index = TactileSubscriber(topic_prefix="index_tip")
    thumb = TactileSubscriber(topic_prefix="thumb_tip")

    time.sleep(2.0)
    print(f"[INFO] Recording tactile data at angle {ANGLE}. Make sure hand is holding the pencil lead.")
    time.sleep(2.0)

    index_data = index.get_obs()
    thumb_data = thumb.get_obs()

    # Flatten and concatenate tactile readings
    snapshot = np.concatenate([index_data.flatten(), thumb_data.flatten()])
    snapshot_path = os.path.join(save_path, f"tactile_snapshot_{int(time.time())}.pkl")

    with open(snapshot_path, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"[INFO] Snapshot saved to {snapshot_path}")

    node.spin_thread_finish()

if __name__ == "__main__":
    main()
