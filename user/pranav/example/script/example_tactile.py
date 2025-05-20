from py_node_exec import NodeExec
from xela_py import TactileSubscriber
import time
import numpy as np

if __name__ == "__main__":
    node = NodeExec(freq=0.1)
    node.spin_thread_start()
    index = TactileSubscriber(topic_prefix="index_tip")
    middle = TactileSubscriber(topic_prefix="middle_tip")
    ring = TactileSubscriber(topic_prefix="ring_tip")
    thumb = TactileSubscriber(topic_prefix="thumb_tip")

    print("Start real-time tactile data acquisition. Move the hand freely.")
    time.sleep(1)

    # print("Index tip tactile data:")
    # while node.ok():
    #     print(index.get_obs())
    #     node.sleep()

    print("Thumb tip tactile data:")
    while node.ok():
        print(thumb.get_obs())
        node.sleep()

    # print("Ring tip tactile data:")
    # while node.ok():
    #     print(ring.get_obs())
    #     node.sleep()

    # print("Middle tip tactile data:")
    # while node.ok():
    #     print(middle.get_obs())
    #     node.sleep()
        
    node.spin_thread_finish()
