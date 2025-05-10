#!/usr/bin/env python3

import rostest
import unittest
import rospy
import numpy as np

from allegro_hand.liballegro import AllegroClient

def wait_topics():
    _, _, topic_types = rospy.get_master().getTopicTypes()
    topic_dict = dict(topic_types)
    while '/allegroHand_test/joint_states' not in topic_dict and '/allegroHand_test/tf' not in topic_dict:
        rospy.sleep(0.02)


class AllegroSelfCheck(unittest.TestCase):
    """
    Very basic self-test capabilities.

    Currently this only ensures the allegro client can publish desired
    joint angles.

    """

    def callback(self, msg):
        pass

    def setUp(self):
        wait_topics()

    def test_allegro_client(self):
        client = AllegroClient(hand_topic_prefix='/allegroHand_test')

        cmd = np.zeros((16,1))
        ret = client.command_joint_position(cmd)

        self.assertTrue(ret)

if __name__ == '__main__':
    import rostest
    rospy.init_node('allegro_hand_test')
    rostest.rosrun('allegro_hand', 'allegro_self_check', AllegroSelfCheck)
