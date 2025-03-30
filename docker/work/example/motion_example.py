from py_node_exec import NodeExec
from allegro_package import AllegroHand
import time
import numpy as np



if __name__ == "__main__":
    node = NodeExec(freq=5)
    node.spin_thread_start()
    robot = AllegroHand(hand_topic_prefix="allegroHand_0",\
                        ctrl_freq=5)

    step=0
    while node.ok() and step<50:
        obs=robot.get_obs()
        robot.set_joint_cmd(obs["jnt_pos"])
        node.sleep()
        step+=1
        
    node.spin_thread_finish()
