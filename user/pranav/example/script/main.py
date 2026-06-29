import rospy
from xela_py import TactileSubscriber
import numpy as np
from vis import DualXelaVisualizer
import threading

class ROSBridge:
    def __init__(self, visualizer):
        self.index_sub = TactileSubscriber(topic_prefix="index_tip")
        self.thumb_sub = TactileSubscriber(topic_prefix="thumb_tip")
        self.visualizer = visualizer
        self.running = True

        # Baseline values for zero-offset correction
        self.index_baseline = np.zeros((30, 3))
        self.thumb_baseline = np.zeros((30, 3))
        self.calibrated = False

    def calibrate_baseline(self):
        print("Calibrating baseline. Please keep sensors untouched.")
        rospy.sleep(2.0)
        self.index_baseline = np.array(self.index_sub.get_obs())
        self.thumb_baseline = np.array(self.thumb_sub.get_obs())
        print("Baseline calibrated.")
        self.calibrated = True

    def update_loop(self, freq=10):
        self.calibrate_baseline()  

        rate = rospy.Rate(freq)
        while not rospy.is_shutdown() and self.running:
            index_obs = np.array(self.index_sub.get_obs())
            thumb_obs = np.array(self.thumb_sub.get_obs())

            if self.calibrated:
                index_obs -= self.index_baseline
                thumb_obs -= self.thumb_baseline

            if index_obs.shape == (30, 3):
                self.visualizer.set_index_obs(index_obs)
            if thumb_obs.shape == (30, 3):
                self.visualizer.set_thumb_obs(thumb_obs)

            rate.sleep()

    def stop(self):
        self.running = False

if __name__ == "__main__":
    rospy.init_node("xela_dual_main_node", anonymous=True)
    vis = DualXelaVisualizer()
    bridge = ROSBridge(vis)

    thread = threading.Thread(target=bridge.update_loop)
    thread.daemon = True
    thread.start()

    vis.start_animation()  # replaces plt.show()
    bridge.stop()
