import rospy
import numpy as np
import matplotlib.pyplot as plt

from py_node_exec import NodeExec
from allegro_package import AllegroHand
from xela_py import TactileSubscriber

if __name__ == "__main__":
    # Start ROS node
    node = NodeExec(freq=10)
    node.spin_thread_start()

    # Initialize Allegro Hand
    robot = AllegroHand(hand_topic_prefix="allegroHand_0", ctrl_freq=10)

    # Initialize tactile sensor subscribers (thumb and index)
    fingertips = {
        "thumb": TactileSubscriber(topic_prefix="thumb_tip"),
        "index": TactileSubscriber(topic_prefix="index_tip"),
    }

    # Setup tactile data plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Real-Time Tactile Sensor Readings (Thumb & Index)")
    quivers = {}
    for i, finger in enumerate(fingertips.keys()):
        axs[i].set_title(finger.capitalize())
        axs[i].set_xlim(0, 6)
        axs[i].set_ylim(0, 5)
        axs[i].invert_yaxis()
        axs[i].set_aspect('equal')
        axs[i].grid(True)
        quivers[finger] = axs[i].quiver([], [], [], [], angles='xy', scale_units='xy', scale=1)

    plt.tight_layout()

    print("Start real-time joint and tactile data acquisition. Move the hand freely.")

    try:
        while node.ok():
            # Get Allegro Hand joint positions
            obs_joint = robot.get_obs()
            joint_positions = obs_joint.get("jnt_pos", None)
            if joint_positions is not None:
                print("\nJoint Positions:", np.round(joint_positions, 4))

            # Get tactile sensor data and update plots
            for finger, sub in fingertips.items():
                tactile_data = sub.get_obs()  
                if tactile_data is not None and len(tactile_data) == 90:
                    # Reshape to (30, 3): 30 taxels with (x, y, z)
                    taxel_vectors = tactile_data.reshape(30, 3)

                    # Compute positions for each taxel in a 5x6 grid
                    x_positions = np.tile(np.arange(6), 5)
                    y_positions = np.repeat(np.arange(5), 6)

                    # Extract x and y components for quiver
                    u = taxel_vectors[:, 0]
                    v = taxel_vectors[:, 1]

                    # Update quiver plot
                    quivers[finger].set_UVC(u, v)
                    quivers[finger].set_offsets(np.c_[x_positions, y_positions])

                    # Compute and print magnitudes
                    magnitudes = np.linalg.norm(taxel_vectors, axis=1)
                    tactile_img = magnitudes.reshape(5, 6)
                    print(f"{finger.capitalize()} Tactile Force Magnitudes (mean={np.mean(tactile_img):.3f}):")
                    print(np.round(tactile_img, 2))

            plt.pause(0.01)
            node.sleep()

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        node.spin_thread_finish()
        plt.close()
