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