#!/usr/bin/env python3
"""Full Manus-to-Allegro control without commanded finger self-touch."""

from manusallegro import ManusToAllegroPublisher
import rclpy


class ManusToAllegroWithoutSelfTouch(ManusToAllegroPublisher):
    """Preserve independent finger tracking without convergence or syncing."""

    SELF_TOUCH_HAPTIC_FINGERS = ()

    def __init__(self):
        super().__init__("manus_to_allegro_without_self_touch")
        self.get_logger().info(
            "No-self-touch mode: index, middle, and ring remain independent."
        )

    def _apply_three_finger_self_touch(self, mqd_20, q_deg):
        # Deliberately bypass all spread convergence and flex synchronization.
        return 0.0


def main(args=None):
    rclpy.init(args=args)
    node = ManusToAllegroWithoutSelfTouch()
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
