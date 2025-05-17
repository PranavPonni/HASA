from py_node_exec import NodeExec
from xela_py import TactileSubscriber
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    node = NodeExec(freq=10)  # Faster update for better visualization
    node.spin_thread_start()

    # Create a subscriber for each fingertip
    fingertips = {
        "thumb": TactileSubscriber(topic_prefix="thumb_tip"),
        "index": TactileSubscriber(topic_prefix="index_tip"),
        "middle": TactileSubscriber(topic_prefix="middle_tip"),
        "ring": TactileSubscriber(topic_prefix="ring_tip")
    }

    # Create the figure and axes for visualization
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle("Real-Time Tactile Sensor Readings")

    # Initialize plots
    images = {}
    for i, finger in enumerate(fingertips.keys()):
        axs[i].set_title(finger.capitalize())
        axs[i].axis('off')  # Hide axis
        # Initialize with a 4x4 grid of zeros
        images[finger] = axs[i].imshow(np.zeros((4, 4)), cmap='viridis', vmin=0, vmax=1)

    plt.tight_layout()

    try:
        while node.ok():
            # Get latest tactile data
            obs = {finger: sub.get_obs() for finger, sub in fingertips.items()}

            # Update each finger's plot
            for finger, data in obs.items():
                if data is not None and len(data) == 16:
                    grid = np.array(data).reshape(4, 4)  # Reshape to 4x4 grid
                    images[finger].set_data(grid)
                    images[finger].set_clim(vmin=np.min(grid), vmax=np.max(grid))  # Adjust color scale

            plt.pause(0.01)
            node.sleep()

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        node.spin_thread_finish()
        plt.close()
