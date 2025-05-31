import rospy
import threading
import time
class NodeExec:
    def __init__(self, node_name="NodeExec", freq=1000):
        rospy.init_node(node_name)
        self.rate = rospy.Rate(freq)
    def spin_thread_start(self):
        self.executor_thread = threading.Thread(target=self.spin)
        self.executor_thread.start()
        

    def spin(self):
        rospy.spin()

    def spin_thread_finish(self):
        rospy.signal_shutdown('Node finished')
        self.executor_thread.join()

    def ok(self):
        return not rospy.is_shutdown()

    def finish(self):
        rospy.signal_shutdown('Node finished')

    def sleep(self):
        self.rate.sleep()

    def log(self, message):
        rospy.loginfo(message)

def main():
    node_exec = NodeExec(node_name="HelloNode", freq=10)
    node_exec.spin_thread_start()

    try:
        while node_exec.ok():
            node_exec.log('Hello')
            node_exec.sleep()
    except rospy.ROSInterruptException:
        pass

    node_exec.spin_thread_finish()

if __name__ == "__main__":
    main()