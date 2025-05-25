#!/usr/bin/env python3
from py_node_exec import NodeExec
from xela_py import TactileSubscriber
from std_msgs.msg import Float64MultiArray
import time
import numpy as np

if __name__ == "__main__":
    node = NodeExec(freq=30.0)  # Run at 30 Hz
    node.spin_thread_start()

    # Initialize Tactile Subscribers
    index = TactileSubscriber(topic_prefix="index_tip")
    thumb = TactileSubscriber(topic_prefix="thumb_tip")

    # print("Publishing Index Finger Tactile Data for PlotJuggler...")

    # time.sleep(1)

    # while node.ok():
    #     # Get tactile observation from index finger
    #     data = index.get_obs()  # This should be a numpy array
    #     print(data)
    #     node.sleep()

    print("Publishing Thumb Finger Tactile Data for PlotJuggler...")

    time.sleep(1)

    while node.ok():
        # Get tactile observation from thumb finger
        data = thumb.get_obs()  # This should be a numpy array
        print(data)
        node.sleep()




    node.spin_thread_finish()
