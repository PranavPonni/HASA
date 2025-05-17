#!/usr/bin/env python3
import rospy
import numpy as np

from allegro_package import AllegroHand
from allegro_hand.liballegro import AllegroClient

rospy.init_node('custom_hand_control', anonymous=True)

#initialize client
hand = AllegroClient(hand_topic_prefix="allegroHand_0")
rospy.sleep(1)  # Wait for the hand to initialize

#command joint positions (0.0 - 1.5)
joint_positions = np.array([-0.4,1.2,0.8,1.2, #index
                            0.0,0.0,0.0,0.0, #middle
                            0.0,0.0,0.0,0.0, #ring
                            1.0,0.0,0.4,1.5]) #thumb
hand.command_joint_position(joint_positions)
