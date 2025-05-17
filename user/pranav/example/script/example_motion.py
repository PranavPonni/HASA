from py_node_exec import NodeExec
from allegro_package import AllegroHand
import numpy as np

if __name__ == "__main__":
    # Start ROS node with given frequency
    node = NodeExec(freq=10)  # Increase frequency for smoother feedback
    node.spin_thread_start()
    
    # Initialize Allegro Hand interface
    robot = AllegroHand(hand_topic_prefix="allegroHand_0", ctrl_freq=10)

    print("Starting real-time joint state acquisition. Move the hand freely.")

    try:
        while node.ok():
            obs = robot.get_obs()

            # Print joint positions
            joint_positions = obs.get("jnt_pos", None)
            if joint_positions is not None:
                print("Joint Positions:", np.round(joint_positions, 4))
            
            node.sleep()

    except KeyboardInterrupt:
        print("Exit")

    # Cleanup
    node.spin_thread_finish()
