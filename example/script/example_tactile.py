from py_node_exec import NodeExec
from xela_py import TactileSubscriber
import time
import numpy as np



if __name__ == "__main__":
    node = NodeExec(freq=5)
    node.spin_thread_start()
    index = TactileSubscriber(topic_prefix="index_tip")

    while node.ok():
        print(index.get_obs())
        node.sleep()
        
    node.spin_thread_finish()
