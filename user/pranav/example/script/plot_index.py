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

    # Initialize ROS publisher
    pub = rospy.Publisher("/tactile/index_tip", Float64MultiArray, queue_size=10)

    print("Publishing Thumb Finger Tactile Data for PlotJuggler...")

    time.sleep(1)

    while node.ok():
        # Get tactile observation from index finger
        data = index.get_obs()  # This should be a numpy array
        if data is not None:
            msg = Float64MultiArray()
            msg.data = data.flatten().tolist()  # Flatten in case it's 2D
            pub.publish(msg)
        node.sleep()




    node.spin_thread_finish()
