#!/usr/bin/env python3
"""Manus control for only the Allegro thumb, index, and middle fingers.

This variant reuses the filtering, calibration, interpolation, and haptic
handling from manusallegro.py while keeping the ring-finger joints at home.
"""

from manusallegro import (
    HOME_POS_RAD_16,
    ManusToAllegroPublisher,
    SELF_TOUCH_FULL_DEG,
    SELF_TOUCH_START_DEG,
    SELF_TOUCH_SYNC_FLEX,
)
import rclpy


# Stronger two-finger convergence.  These remain within the URDF spread limits
# of -0.47..0.47 rad.
INDEX_MIDDLE_TOUCH_INDEX_SPREAD_RAD = 0.50
INDEX_MIDDLE_TOUCH_MIDDLE_SPREAD_RAD = -0.20


class ThreeFingerManusToAllegroPublisher(ManusToAllegroPublisher):
    """Control thumb/index/middle and hold the ring finger at home."""

    SELF_TOUCH_HAPTIC_FINGERS = (1, 2)  # index and middle only

    # Allegro command order is index, middle, ring, thumb.  Therefore the ring
    # occupies positions 8..11 in every 16-joint command.
    RING_COMMAND_SLICE = slice(8, 12)

    def __init__(self):
        super().__init__("three_finger_manus_to_allegro_joint_cmd_publisher")
        self.get_logger().info(
            "Three-finger mode: controlling thumb, index, and middle; "
            "holding ring at home."
        )

    def _apply_three_finger_self_touch(self, mqd_20, q_deg):
        """Bring index and middle together as those two fingers curl."""
        index_flex = (q_deg[5] + q_deg[6]) * 0.5
        middle_flex = (q_deg[9] + q_deg[10]) * 0.5
        x = (min(index_flex, middle_flex) - SELF_TOUCH_START_DEG) / max(
            SELF_TOUCH_FULL_DEG - SELF_TOUCH_START_DEG, 1.0
        )
        x = min(1.0, max(0.0, x))
        amount = x * x * (3.0 - 2.0 * x)
        if amount <= 0.0:
            return 0.0

        # mqd blocks are thumb[0:4], index[4:8], middle[8:12], ring[12:16].
        targets = {
            4: INDEX_MIDDLE_TOUCH_INDEX_SPREAD_RAD,
            8: INDEX_MIDDLE_TOUCH_MIDDLE_SPREAD_RAD,
        }
        for joint, target in targets.items():
            mqd_20[joint] += amount * (target - mqd_20[joint])

        sync_amount = amount * SELF_TOUCH_SYNC_FLEX
        for index_joint, middle_joint in ((5, 9), (6, 10), (7, 11)):
            mean = (mqd_20[index_joint] + mqd_20[middle_joint]) * 0.5
            mqd_20[index_joint] += sync_amount * (mean - mqd_20[index_joint])
            mqd_20[middle_joint] += sync_amount * (mean - mqd_20[middle_joint])

        return amount

    def _publish_joint_cmd(self, joint_pos_16):
        command = list(joint_pos_16)
        command[self.RING_COMMAND_SLICE] = HOME_POS_RAD_16[self.RING_COMMAND_SLICE]
        super()._publish_joint_cmd(command)


def main(args=None):
    rclpy.init(args=args)
    node = ThreeFingerManusToAllegroPublisher()
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
