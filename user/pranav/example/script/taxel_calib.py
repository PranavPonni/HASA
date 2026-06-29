#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray

index_data = []
thumb_data = []

def index_callback(msg):
    global index_data
    vals = np.array(msg.data).reshape(-1, 3)
    index_data.append(vals)

def thumb_callback(msg):
    global thumb_data
    vals = np.array(msg.data).reshape(-1, 3)
    thumb_data.append(vals)

def main():
    rospy.init_node('resting_data_recorder')
    rospy.Subscriber("/xela/0/values", Float32MultiArray, index_callback)
    rospy.Subscriber("/xela/1/values", Float32MultiArray, thumb_callback)

    duration = rospy.get_param("~duration", 5.0)  # seconds
    rate = rospy.Rate(30)  # 30 Hz
    print(f"Collecting tactile data for {duration} seconds...")

    start_time = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        now = rospy.Time.now().to_sec()
        if now - start_time > duration:
            break
        rate.sleep()

    # Ensure both have same length
    n = min(len(index_data), len(thumb_data))
    index_arr = np.array(index_data[:n])  # (N, 30, 3)
    thumb_arr = np.array(thumb_data[:n])  # (N, 30, 3)

    # Combine to shape (N, 2, 30, 3)
    result = np.stack([index_arr, thumb_arr], axis=1)
    np.save("resting_data.npy", result)
    print(f"✅ Saved {n} frames to resting_data.npy")

if __name__ == "__main__":
    main()
