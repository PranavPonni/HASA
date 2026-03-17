import sys
from pathlib import Path


def _bootstrap_local_ros_python_path():
    """Allow running this script without sourcing ros_ws/devel/setup.bash."""
    repo_root = Path(__file__).resolve().parents[2]
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        repo_root / "ros_ws" / "devel" / "lib" / py_ver / "dist-packages",
        repo_root / "ros_ws" / "src" / "py_node_exec" / "src",
        repo_root / "ros_ws" / "src" / "allegro_package" / "src",
    ]

    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_bootstrap_local_ros_python_path()

from py_node_exec import NodeExec
from allegro_package import AllegroHand
import numpy as np
from allegro_leader import AllegroPrecesionGrasp



if __name__ == "__main__":
    node = NodeExec(freq=5)
    node.spin_thread_start()
    robot = AllegroHand(hand_topic_prefix="allegroHand_0",\
                        ctrl_freq=20)

    teleop = AllegroPrecesionGrasp(wall_kp=0.)
    finger_idx = [0, 1, 2, 3, 12, 13, 14, 15]
    
    follower_cmd=np.zeros(16)

    # teleop_loop
    while node.ok():
        leader_cmd = teleop.update() 
        for i, idx in enumerate(finger_idx):
            follower_cmd[idx] = leader_cmd[i]
        print(leader_cmd)
        obs=robot.get_obs()
        robot.set_joint_cmd(follower_cmd)
        node.sleep()
        
    node.spin_thread_finish()
    teleop.driver.set_torque_mode(False)
    teleop.driver.close()
