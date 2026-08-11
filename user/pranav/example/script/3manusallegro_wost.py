#!/usr/bin/env python3
"""Thumb/index/middle Manus control without commanded finger self-touch."""

from manusallegro import HOME_POS_RAD_16, ManusToAllegroPublisher
import rclpy


class ThreeFingerManusToAllegroWithoutSelfTouch(ManusToAllegroPublisher):
    """Track thumb/index/middle independently and hold ring at home."""

    SELF_TOUCH_HAPTIC_FINGERS = ()
    RING_COMMAND_SLICE = slice(8, 12)

    def __init__(self):
        super().__init__("three_finger_manus_to_allegro_without_self_touch")
        self.get_logger().info(
            "Three-finger no-self-touch mode: thumb, index, and middle remain "
            "independent; ring is held at home."
        )

    def _apply_three_finger_self_touch(self, mqd_20, q_deg):
        # Deliberately bypass index-middle convergence and flex synchronization.
        return 0.0

    def _publish_joint_cmd(self, joint_pos_16):
        command = list(joint_pos_16)
        command[self.RING_COMMAND_SLICE] = HOME_POS_RAD_16[self.RING_COMMAND_SLICE]
        super()._publish_joint_cmd(command)


def main(args=None):
    rclpy.init(args=args)
    node = ThreeFingerManusToAllegroWithoutSelfTouch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_home(repeats=20, sleep_sec=0.02, reason="shutdown")
        node._publish_haptics([0.0, 0.0, 0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
