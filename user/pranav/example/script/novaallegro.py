#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
import time
from pathlib import Path


PI = math.pi
MOTOR_COUNT = 20
ALLEGRO_JOINT_COUNT = 16

JOINT_NAMES_16 = [
    "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7",
    "joint_8", "joint_9", "joint_10", "joint_11",
    "joint_12", "joint_13", "joint_14", "joint_15",
]

HOME_POS_RAD_16 = [0.0] * ALLEGRO_JOINT_COUNT

EMA_ALPHA = 0.15
INTERP_SPEED = 10.0


def _prepend_env_path(var_name, path_value):
    if not path_value:
        return
    current = os.environ.get(var_name, "")
    parts = [p for p in current.split(":") if p]
    if path_value in parts:
        return
    os.environ[var_name] = f"{path_value}:{current}" if current else path_value


def _bootstrap_senseglove_msgs_path():
    """Allow direct python execution without manually sourcing senseglove setup."""
    script_path = Path(__file__).resolve()
    candidate_roots = []

    for parent in script_path.parents:
        candidate_roots.extend([
            parent / "senseglove_ros",
            parent / "senseglove_ros_foxy",
            parent / "manus_ros2_ws",
        ])

    candidate_roots.extend([
        Path.home() / "senseglove_ros",
        Path.home() / "senseglove_ros_foxy",
        Path.home() / "HASA" / "senseglove_ros2_ws",
        Path.home() / "HASA" / "manus_ros2_ws",
    ])

    seen = set()
    for ws_root in candidate_roots:
        ws_root = ws_root.resolve() if ws_root.exists() else ws_root
        key = str(ws_root)
        if key in seen:
            continue
        seen.add(key)

        install_root = ws_root / "install"
        lib_root = install_root / "lib"
        if not install_root.exists() or not lib_root.exists():
            continue

        for candidate in sorted(lib_root.glob("python*/site-packages")):
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

        _prepend_env_path("LD_LIBRARY_PATH", str(lib_root))
        _prepend_env_path("LD_LIBRARY_PATH", str(install_root / "lib"))
        _prepend_env_path("AMENT_PREFIX_PATH", str(install_root))
        _prepend_env_path("CMAKE_PREFIX_PATH", str(install_root))

    ros_site = "/opt/ros/foxy/lib/python3.8/site-packages"
    if os.path.isdir(ros_site) and ros_site not in sys.path:
        sys.path.append(ros_site)


_bootstrap_senseglove_msgs_path()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from senseglove_msgs.msg import SenseGloveState


class NovaToAllegroPublisher(Node):
    def __init__(self, in_topic, out_topic, interp_rate_hz, unit_mode):
        super().__init__("nova_to_allegro_joint_cmd_publisher")
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.unit_mode = unit_mode
        self.recv_count = 0
        self.pub_count = 0
        self.drop_count = 0

        self._target_pos = list(HOME_POS_RAD_16)
        self._current_pos = list(HOME_POS_RAD_16)
        self._ema_deg = None
        self._have_target = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(SenseGloveState, self.in_topic, self.cb, qos)
        self.pub = self.create_publisher(JointState, self.out_topic, 10)

        self.diag_timer = self.create_timer(2.0, self._diag_cb)
        self.interp_rate_hz = float(interp_rate_hz)
        self._interp_timer = self.create_timer(1.0 / self.interp_rate_hz, self._interp_cb)

        self.get_logger().info(
            f"Subscribing: {self.in_topic} -> Publishing: {self.out_topic}\\n"
            f"Input units: {self.unit_mode}\\n"
            f"Smoothing: EMA alpha={EMA_ALPHA}, interp {self.interp_rate_hz}Hz, speed={INTERP_SPEED}"
        )

        self.publish_home(repeats=20, sleep_sec=0.02, reason="startup")

    def _diag_cb(self):
        self.get_logger().info(
            f"diag recv={self.recv_count} pub={self.pub_count} drop={self.drop_count}"
        )

    def _publish_joint_cmd(self, joint_pos_16):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES_16
        msg.position = joint_pos_16
        self.pub.publish(msg)

    def publish_home(self, repeats=20, sleep_sec=0.02, reason=""):
        for _ in range(repeats):
            self._publish_joint_cmd(HOME_POS_RAD_16)
            time.sleep(sleep_sec)
        if reason:
            self.get_logger().info(f"Home pose command sent ({reason}).")

    def _interp_cb(self):
        if not self._have_target:
            return
        dt = 1.0 / self.interp_rate_hz
        alpha = min(1.0, INTERP_SPEED * dt)
        for i in range(ALLEGRO_JOINT_COUNT):
            self._current_pos[i] += alpha * (self._target_pos[i] - self._current_pos[i])
        self._publish_joint_cmd(list(self._current_pos))
        self.pub_count += 1

    def _to_degree_vector(self, values):
        if self.unit_mode == "deg":
            return list(values)
        if self.unit_mode == "rad":
            return [v * (180.0 / PI) for v in values]

        # auto mode: if all values are small, treat as radians
        max_abs = max(abs(v) for v in values) if values else 0.0
        if max_abs <= 7.0:
            return [v * (180.0 / PI) for v in values]
        return list(values)

    def set_target_left(self, q_deg):
        dir_ = [1, 0.1, 1, 1,
                -1.0, 1, 1, 1,
                -1.0, 1, 1, 1,
                -1.0, 1, 1, 1,
                0.5, -0.5, 1, 1.2]

        calib = [1, 1.2, 1.5, 1.2,
                 1.4, 1, 1.2, 1.2,
                 1.5, 1, 1.2, 1.2,
                 1.7, 1, 1, 1,
                 0.5, 0.5, 1, 1]

        qd = [0.0] * MOTOR_COUNT
        mqd = [0.0] * MOTOR_COUNT

        qd[0] = (58.5 - q_deg[1]) * (PI / 180.0)
        qd[1] = (q_deg[0] + 20.0) * (PI / 180.0)
        for i in range(2, 16):
            qd[i] = q_deg[i] * (PI / 180.0)

        qd[2] += 30.0 * (PI / 180.0)
        qd[0] += 18.0 * (PI / 180.0)
        qd[5] += 15.0 * (PI / 180.0)
        qd[8] += 20.0 * (PI / 180.0)
        qd[9] += 20.0 * (PI / 180.0)
        qd[12] += 36.0 * (PI / 180.0)
        qd[13] += 20.0 * (PI / 180.0)

        for i in range(MOTOR_COUNT):
            mqd[i] = qd[i] * calib[i] * dir_[i]
            if i in (2, 4, 8, 12, 16, 17):
                continue
            if mqd[i] <= 0.0:
                mqd[i] = 0.0

        return mqd

    def cb(self, msg):
        self.recv_count += 1

        if len(msg.position) < MOTOR_COUNT:
            self.drop_count += 1
            if self.drop_count % 50 == 1:
                self.get_logger().warn(
                    f"Received SenseGloveState with {len(msg.position)} joints (need {MOTOR_COUNT})."
                )
            return

        q_raw = [float(v) for v in msg.position[:MOTOR_COUNT]]
        q_deg_raw = self._to_degree_vector(q_raw)

        if self._ema_deg is None:
            self._ema_deg = list(q_deg_raw)
        else:
            for i, value in enumerate(q_deg_raw):
                self._ema_deg[i] += EMA_ALPHA * (value - self._ema_deg[i])

        mqd_20 = self.set_target_left(list(self._ema_deg))
        self._target_pos = mqd_20[4:8] + mqd_20[8:12] + mqd_20[12:16] + mqd_20[0:4]

        if not self._have_target:
            self._current_pos = list(self._target_pos)
        self._have_target = True


def _default_topic(serial, side):
    side_norm = side.strip().lower()
    if side_norm not in ("lh", "rh"):
        raise ValueError("side must be 'lh' or 'rh'")
    return f"/senseglove/glove{serial}/{side_norm}/senseglove_states"


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="01001", help="SenseGlove serial used in gloves.yaml")
    parser.add_argument("--side", default="lh", choices=["lh", "rh"], help="Glove side namespace")
    parser.add_argument("--in-topic", default="", help="Override input topic directly")
    parser.add_argument("--out-topic", default="/allegroHand_0/joint_cmd", help="Allegro joint_cmd topic")
    parser.add_argument("--interp-rate", type=float, default=40.0, help="Output interpolation rate in Hz")
    parser.add_argument(
        "--unit-mode",
        default="auto",
        choices=["auto", "deg", "rad"],
        help="SenseGloveState.position units",
    )
    parsed, ros_args = parser.parse_known_args(args)

    in_topic = parsed.in_topic.strip() or _default_topic(parsed.serial, parsed.side)

    rclpy.init(args=ros_args)
    node = NovaToAllegroPublisher(
        in_topic=in_topic,
        out_topic=parsed.out_topic,
        interp_rate_hz=parsed.interp_rate,
        unit_mode=parsed.unit_mode,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_home(repeats=20, sleep_sec=0.02, reason="shutdown")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
