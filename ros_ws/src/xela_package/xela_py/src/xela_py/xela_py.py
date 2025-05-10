import rospy
from xela_server_ros.msg import SensStream
import numpy as np
import threading

class TactileSubscriber:
    def __init__(self, topic_prefix):
        self.topic_prefix = topic_prefix
        self.msg = None 
        self.lock = threading.Lock()
        topic_name = f"{self.topic_prefix}/xServTopic"
        self.sub = rospy.Subscriber(topic_name, SensStream, self.callback)
        self._init_flg=False
        rospy.loginfo(f"Trying to subscribe topic: {topic_name}")
        while not self._init_flg and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo(f"Succeed in subscribing topic: {topic_name}")


    def callback(self, msg):
        with self.lock:
            self.msg = msg
            self._init_flg = True

    def check_half_or_more_below_or_equal_one(self,arr):
        if arr.shape[1] != 3:
            raise ValueError("Shape should be (n, 3)")
        
        total_elements = arr.size  
        count_leq_one = np.sum(arr <= 1)
        
        return False if count_leq_one >= total_elements / 2 else True

    def get_obs(self):
        with self.lock:
            if self.msg is not None:
                sn_ls = self.msg.sensors[0].taxels
                val = np.array([[t.x, t.y, t.z] for t in sn_ls])
                if self.check_half_or_more_below_or_equal_one(val):
                    return val
                else:
                    rospy.logwarn(f"Sensor {topic_name} is not valid")
            else:
                rospy.logwarn("No tactile data received yet.")
                return None

if __name__ == "__main__":
    print("Code for subscribing tactile sensor data")
