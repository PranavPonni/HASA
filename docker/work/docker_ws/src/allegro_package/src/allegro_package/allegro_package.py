#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import numpy as np
import copy
import time
from threading import Lock
from allegro_package.interpolater import Interpolation

class AllegroHand:
    def __init__(self, hand_topic_prefix="allegroHand", ctrl_freq=10, low_level_freq=100, use_fake_hardware=False):
        self.hand_topic_prefix = hand_topic_prefix.rstrip("/")
        self.lock = Lock()
        self.use_fake_hardware = use_fake_hardware
        self.ctrl_freq = ctrl_freq
        self.low_level_freq = low_level_freq

        self.joint_state_topic = f"/{self.hand_topic_prefix}/joint_states"
        self.joint_cmd_topic = f"/{self.hand_topic_prefix}/joint_cmd"
        self.grasp_cmd_topic = f"/{self.hand_topic_prefix}/lib_cmd"

        self.pub_joint_cmd = rospy.Publisher(self.joint_cmd_topic, JointState, queue_size=1)
        self.pub_grasp_cmd = rospy.Publisher(self.grasp_cmd_topic, String, queue_size=1)
        rospy.Subscriber(self.joint_state_topic, JointState, self._joint_state_callback)

        self.intrp = Interpolation("linear", "normal")
        self.intrp.set_interpolation_config(ctrl_freq, self.low_level_freq, 0.95)

        self._init_value()
        self.last_cmd_time = rospy.Time.now()

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.low_level_freq), self._publish_joint_cmd)

        print("Waiting for subscribing allegro hand state")
        while self._init_flag:
            rospy.sleep(0.1)
        print("Succeed subscribing allegro hand state")


    def _init_value(self):
        self._jnt_pos = None
        self._jnt_vel = None
        self._jnt_trq = None
        self._jnt_cmd_pos = None   
        self.interpolated_jnt_cmds = None 
        self._init_flag = True

    def _joint_state_callback(self, data):
        with self.lock:
            self._jnt_pos = np.array(data.position)
            self._jnt_vel = np.array(data.velocity)
            self._jnt_trq = np.array(data.effort)
            # TODO add joint force
            if self._init_flag:
                self._jnt_cmd_pos = self._jnt_pos
                self._init_flag = False

    def _publish_joint_cmd(self, event):
        with self.lock:
            if self.interpolated_jnt_cmds is not None:
                if self.use_fake_hardware:
                    self._jnt_pos = copy.deepcopy(self.interpolated_jnt_cmds[0])
                else:
                    msg = JointState()
                    msg.position = list(self.interpolated_jnt_cmds[0])
                    self.pub_joint_cmd.publish(msg)
                if self.interpolated_jnt_cmds.shape[0] > 1:
                    self.interpolated_jnt_cmds = copy.deepcopy(self.interpolated_jnt_cmds[1:])

    def set_joint_cmd(self,jnt_cmd):
        with self.lock:
            current_time = rospy.Time.now()
            elapsed_time = (current_time - self.last_cmd_time).to_sec()
            if elapsed_time > 0:
                self.ctrl_freq = 1.0 / elapsed_time
            self.intrp.set_interpolation_config(self.ctrl_freq, self.low_level_freq, 0.95)
            initial_position = self.interpolated_jnt_cmds[-1] if self.interpolated_jnt_cmds is not None else self._jnt_pos
            self._jnt_cmd_pos = np.array(jnt_cmd)
            self.interpolated_jnt_cmds = self.intrp.get_interpolate(initial_position, jnt_cmd)
            self.last_cmd_time = current_time

    def get_obs(self):
        while self._jnt_pos is None:
            rospy.sleep(0.01)
        return {
            'jnt_pos': self._jnt_pos,
            'jnt_vel': self._jnt_vel,
            'jnt_trq': self._jnt_trq,
            'jnt_cmd_pos': self._jnt_cmd_pos
        }

    def torque_off(self):
        rospy.loginfo("AllegroHand: Torque off triggered")
        msg = String("off")
        self.pub_grasp_cmd.publish(msg)

    def torque_on(self):
        rospy.loginfo("AllegroHand: Torque on triggered")
        msg = String("on")
        self.pub_grasp_cmd.publish(msg)

if __name__ == "__main__":
    rospy.init_node('allegro_hand_node')
    hand = AllegroHand()
    rospy.spin()
