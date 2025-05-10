import unittest
from allegro_hand.liballegro import AllegroClient  # Import the client for commanding the Allegro hand.
from sensor_msgs.msg import JointState  # Import the ROS message type for joint states.


class MockPublisher(object):
    """
    A simple mock publisher class that simulates a ROS publisher.
    It counts the number of times a message is "published" and stores
    the last published message.
    """

    def __init__(self):
        # Initialize the counter and the placeholder for the last published message.
        self._pub_count = 0
        self._last_published = None

    def publish(self, args):
        """
        Simulate publishing a message by incrementing the count and saving the message.
        :param args: The message/data that is "published".
        """
        self._pub_count += 1
        self._last_published = args

    # The 'pass' below is not needed because the class is already implemented.
    pass


class TestAllegro(unittest.TestCase):
    """
    Unit tests for the AllegroClient class.
    This class tests various commands such as sending hand configurations,
    joint positions, torques, and other functionalities.
    """

    def setUp(self):
        """
        Set up the test fixture before each test method is run.
        Instantiate an AllegroClient and replace its publishers with our MockPublisher.
        """
        self.client = AllegroClient()
        # Replace actual publishers with mocks to capture published messages.
        self.client.pub_grasp = MockPublisher()
        self.client.pub_joint = MockPublisher()
        self.client.pub_envelop_torque = MockPublisher()

    def test_instantiate(self):
        """
        Test that the AllegroClient is correctly instantiated
        and that no messages have been published initially.
        """
        self.assertIsNotNone(self.client)
        self.assertEqual(0, self.client.pub_grasp._pub_count)
        self.assertEqual(0, self.client.pub_joint._pub_count)
        self.assertEqual(0, self.client.pub_envelop_torque._pub_count)

    def test_send_hand_configuration(self):
        """
        Test sending a valid hand configuration command ("envelop").
        Check that the publish method is called once and the correct message data is sent.
        """
        ret = self.client.command_hand_configuration("envelop")
        self.assertEqual(True, ret)
        self.assertEqual(1, self.client.pub_grasp._pub_count)
        self.assertEqual("envelop", self.client.pub_grasp._last_published.data)

    def test_send_invalid_hand_config(self):
        """
        Test sending an invalid hand configuration command.
        The client should return False and not publish any message.
        """
        ret = self.client.command_hand_configuration("garbage")
        self.assertEqual(False, ret)
        self.assertEqual(0, self.client.pub_grasp._pub_count)

    def test_list_hand_configs(self):
        """
        Test that the method for listing hand configurations returns a valid list.
        Verify that a known hand configuration ("three_finger_grasp") is included.
        """
        ret = self.client.list_hand_configurations()
        self.assertTrue(ret)  # Ensure that the returned list is not empty.
        self.assertIn("three_finger_grasp", ret)  # Check for a known configuration.

    def test_command_envelop_torque(self):
        """
        Test sending a valid envelop torque command.
        Confirm that the value is published and that the command returns True.
        """
        ret = self.client.set_envelop_torque(0.5)
        self.assertTrue(ret)
        self.assertEqual(1, self.client.pub_envelop_torque._pub_count)
        published_value = self.client.pub_envelop_torque._last_published.data
        self.assertEqual(0.5, published_value)

    def test_command_large_envelop_torque(self):
        """
        Test sending a torque command larger than the maximum allowed value.
        The client should cap the torque to 1.0.
        """
        ret = self.client.set_envelop_torque(1.5)
        self.assertTrue(ret)
        self.assertEqual(1, self.client.pub_envelop_torque._pub_count)
        published_value = self.client.pub_envelop_torque._last_published.data
        self.assertEqual(1.0, published_value)

    def test_command_small_envelop_torque(self):
        """
        Test sending a torque command smaller than the minimum allowed value.
        The client should floor the torque to 0.0.
        """
        ret = self.client.set_envelop_torque(-0.5)
        self.assertTrue(ret)
        self.assertEqual(1, self.client.pub_envelop_torque._pub_count)
        published_value = self.client.pub_envelop_torque._last_published.data
        self.assertEqual(0.0, published_value)

    def test_command_pose(self):
        """
        Test sending a valid joint position command.
        The command should publish a JointState message with the correct position.
        """
        des_pose = [0.123] * 16  # Desired pose: list of 16 joint positions.
        ret = self.client.command_joint_position(des_pose)
        self.assertTrue(ret)
        self.assertEqual(1, self.client.pub_joint._pub_count)
        published_state = self.client.pub_joint._last_published

        # Create a reference JointState with the desired positions.
        ref_state = JointState()
        ref_state.position = des_pose
        self.assertEqual(ref_state, published_state)

    def test_command_pose_wrong_dimensions(self):
        """
        Test sending a joint position command with an incorrect number of dimensions.
        The command should fail (return False) and not publish any message.
        """
        des_pose = [0.123] * 2  # Incorrect dimension: should be 16 elements.
        ret = self.client.command_joint_position(des_pose)
        self.assertFalse(ret)
        self.assertEqual(0, self.client.pub_joint._pub_count)
        published_state = self.client.pub_joint._last_published
        self.assertIsNone(published_state)

    def test_command_pose_int_not_array(self):
        """
        Test sending a joint position command that is not an iterable array.
        The command should fail and no message should be published.
        """
        des_pose = 0.123  # Not an array: should be a list of 16 numbers.
        ret = self.client.command_joint_position(des_pose)
        self.assertFalse(ret)
        self.assertEqual(0, self.client.pub_joint._pub_count)
        published_state = self.client.pub_joint._last_published
        self.assertIsNone(published_state)

    def test_poll_position(self):
        """
        Test polling the joint position.
        When wait is set to False, the poll should return None.
        """
        joints = self.client.poll_joint_position(wait=False)
        self.assertIsNone(joints)

    def test_topic_prefix(self):
        """
        Test that providing a custom hand_topic_prefix sets the correct topic name
        for the grasp publisher.
        """
        client = AllegroClient(hand_topic_prefix="/Prefix")
        self.assertEqual("/Prefix/lib_cmd", client.pub_grasp.name)

    def test_topic_prefix_trailing_slash(self):
        """
        Test that the hand_topic_prefix works correctly even if a trailing slash is provided.
        The resulting topic name should not have a double slash.
        """
        client = AllegroClient(hand_topic_prefix="/Prefix/")
        self.assertEqual("/Prefix/lib_cmd", client.pub_grasp.name)

    def test_command_torques(self):
        """
        Test sending valid joint torques.
        Verify that a JointState message is published with the correct effort values.
        """
        des_torques = [0.123] * 16  # Desired torques: list of 16 values.
        ret = self.client.command_joint_torques(des_torques)
        self.assertTrue(ret)
        self.assertEqual(1, self.client.pub_joint._pub_count)
        published_state = self.client.pub_joint._last_published

        # Create a reference JointState with the desired efforts.
        ref_state = JointState()
        ref_state.effort = des_torques
        self.assertEqual(ref_state, published_state)

    def test_command_torques_wrong_dimensions(self):
        """
        Test sending joint torques with the wrong dimensions.
        The command should return False and no message should be published.
        """
        des_torques = [0.123] * 2  # Incorrect dimension: should be 16 elements.
        ret = self.client.command_joint_torques(des_torques)
        self.assertFalse(ret)
        self.assertEqual(0, self.client.pub_joint._pub_count)
        published_state = self.client.pub_joint._last_published
        self.assertIsNone(published_state)

    def test_command_torques_int_not_array(self):
        """
        Test sending a joint torques command that is not an iterable.
        The command should fail and no message should be published.
        """
        des_torques = 0.123  # Not an array: should be a list of 16 numbers.
        ret = self.client.command_joint_torques(des_torques)
        self.assertFalse(ret)
        self.assertEqual(0, self.client.pub_joint._pub_count)
        published_state = self.client.pub_joint._last_published
        self.assertIsNone(published_state)

    def test_disconnect(self):
        """
        Test the disconnect method.
        When disconnecting, the client should publish an "off" command to the grasp topic.
        """
        self.client.disconnect()
        self.assertEqual(1, self.client.pub_grasp._pub_count)
        self.assertEqual("off", self.client.pub_grasp._last_published.data)


if __name__ == '__main__':
    unittest.main()
