#!/usr/bin/env python3
"""Manus control for an Allegro index-middle or middle-ring finger pair."""

import argparse

import rclpy

from manusallegro import (
    HOME_POS_RAD_16,
    ManusToAllegroPublisher,
    SELF_TOUCH_FULL_DEG,
    SELF_TOUCH_START_DEG,
    SELF_TOUCH_SYNC_FLEX,
)


PAIR_CONFIG = {
    "index-middle": {
        "source_starts": (4, 8),
        "command_slices": (slice(0, 4), slice(4, 8)),
        "spread_targets": (0.50, -0.20),
        "haptic_fingers": (1, 2),
    },
    "middle-ring": {
        "source_starts": (8, 12),
        "command_slices": (slice(4, 8), slice(8, 12)),
        "spread_targets": (0.20, -0.50),
        "haptic_fingers": (2, 3),
    },
}


class TwoFingerManusToAllegroPublisher(ManusToAllegroPublisher):
    """Control one adjacent two-finger pair and hold all other fingers home."""

    def __init__(self, pair):
        self.pair = pair
        self.pair_config = PAIR_CONFIG[pair]
        self.SELF_TOUCH_HAPTIC_FINGERS = self.pair_config["haptic_fingers"]
        node_pair = pair.replace("-", "_")
        super().__init__(f"two_finger_{node_pair}_manus_to_allegro_publisher")
        self.get_logger().info(
            f"Two-finger mode: controlling {pair}; holding all other fingers at home."
        )

    def _apply_three_finger_self_touch(self, mqd_20, q_deg):
        """Bring the selected adjacent pair together as both fingers curl."""
        first_start, second_start = self.pair_config["source_starts"]
        first_flex = (q_deg[first_start + 1] + q_deg[first_start + 2]) * 0.5
        second_flex = (q_deg[second_start + 1] + q_deg[second_start + 2]) * 0.5
        x = (min(first_flex, second_flex) - SELF_TOUCH_START_DEG) / max(
            SELF_TOUCH_FULL_DEG - SELF_TOUCH_START_DEG, 1.0
        )
        x = min(1.0, max(0.0, x))
        amount = x * x * (3.0 - 2.0 * x)
        if amount <= 0.0:
            return 0.0

        for joint, target in zip(
            (first_start, second_start), self.pair_config["spread_targets"]
        ):
            mqd_20[joint] += amount * (target - mqd_20[joint])

        sync_amount = amount * SELF_TOUCH_SYNC_FLEX
        for offset in (1, 2, 3):
            first_joint = first_start + offset
            second_joint = second_start + offset
            mean = (mqd_20[first_joint] + mqd_20[second_joint]) * 0.5
            mqd_20[first_joint] += sync_amount * (mean - mqd_20[first_joint])
            mqd_20[second_joint] += sync_amount * (mean - mqd_20[second_joint])

        return amount

    def _publish_joint_cmd(self, joint_pos_16):
        command = list(HOME_POS_RAD_16)
        for active_slice in self.pair_config["command_slices"]:
            command[active_slice] = joint_pos_16[active_slice]
        super()._publish_joint_cmd(command)


def _parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Control either the Allegro index-middle or middle-ring pair."
    )
    parser.add_argument(
        "pair",
        nargs="?",
        choices=tuple(PAIR_CONFIG),
        default="index-middle",
        help="finger pair to control (default: index-middle)",
    )
    return parser.parse_known_args(args)


def main(args=None):
    parsed, ros_args = _parse_args(args)
    rclpy.init(args=ros_args)
    node = TwoFingerManusToAllegroPublisher(parsed.pair)
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
