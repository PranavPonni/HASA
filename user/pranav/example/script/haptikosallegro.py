#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
import time
from pathlib import Path


HAPTIKOS_PY_ROOT = Path("/home/handlingteam2/HaptikosAPI/Haptikos_py")
if HAPTIKOS_PY_ROOT.exists():
    sys.path.insert(0, str(HAPTIKOS_PY_ROOT))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from HaptikosAPI import HaptikosClient, HaptikosDataFormat


PI = math.pi
MOTOR_COUNT = 20
ALLEGRO_JOINT_COUNT = 16

OUT_TOPIC = "/allegroHand_0/joint_cmd"
JOINT_NAMES_16 = [
    "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7",
    "joint_8", "joint_9", "joint_10", "joint_11",
    "joint_12", "joint_13", "joint_14", "joint_15",
]

HOME_POS_RAD_16 = [0.0] * ALLEGRO_JOINT_COUNT

EMA_ALPHA = 0.15
INTERP_RATE_HZ = 40.0
INTERP_SPEED = 10.0


class HaptikosToAllegroPublisher(Node):
    def __init__(self, right=False, poll_rate_hz=40.0):
        super().__init__("haptikos_to_allegro_joint_cmd_publisher")
        self.right = right
        self.poll_count = 0
        self.pub_count = 0
        self.invalid_count = 0

        self.client = HaptikosClient()
        self.pub = self.create_publisher(JointState, OUT_TOPIC, 10)

        self._target_pos = list(HOME_POS_RAD_16)
        self._current_pos = list(HOME_POS_RAD_16)
        self._ema_deg = None
        self._have_target = False

        self.poll_timer = self.create_timer(1.0 / poll_rate_hz, self._poll_haptikos_cb)
        self.interp_timer = self.create_timer(1.0 / INTERP_RATE_HZ, self._interp_cb)
        self.diag_timer = self.create_timer(2.0, self._diag_cb)

        side = "right" if self.right else "left"
        self.get_logger().info(
            f"Polling Haptikos {side} glove -> Publishing: {OUT_TOPIC}\n"
            f"Poll rate: {poll_rate_hz}Hz, output: 16 Allegro joint_cmd values\n"
            f"Smoothing: EMA alpha={EMA_ALPHA}, interp {INTERP_RATE_HZ}Hz, speed={INTERP_SPEED}"
        )

        if not self.client.check_connection(right=self.right):
            self.get_logger().warn(
                f"Haptikos {side} glove is not connected yet. "
                "Start the Haptikos Core App and connect/calibrate the exoskeleton."
            )

        self.publish_home(repeats=20, sleep_sec=0.02, reason="startup")

    def _poll_haptikos_cb(self):
        self.poll_count += 1
        data = self.client.get_hand_data(
            right=self.right,
            format=HaptikosDataFormat.GLOBAL_TO_WRIST,
            positions=False,
            rotations=False,
            angles=True,
        )
        if data.is_valid() != 1:
            self.invalid_count += 1
            return

        q_deg_raw = data.get_angles()
        if not q_deg_raw or len(q_deg_raw) < MOTOR_COUNT:
            self.invalid_count += 1
            return

        q_deg_raw = [float(v) for v in q_deg_raw[:MOTOR_COUNT]]

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

    def _interp_cb(self):
        if not self._have_target:
            return
        dt = 1.0 / INTERP_RATE_HZ
        alpha = min(1.0, INTERP_SPEED * dt)
        for i in range(ALLEGRO_JOINT_COUNT):
            self._current_pos[i] += alpha * (self._target_pos[i] - self._current_pos[i])
        self._publish_joint_cmd(list(self._current_pos))
        self.pub_count += 1

    def _diag_cb(self):
        self.get_logger().info(
            f"diag poll={self.poll_count} pub={self.pub_count} invalid={self.invalid_count} "
            f"connected={self.client.check_connection(right=self.right)}"
        )

    def _publish_joint_cmd(self, joint_pos_16):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = JOINT_NAMES_16
        js.position = joint_pos_16
        self.pub.publish(js)

    def publish_home(self, repeats=20, sleep_sec=0.02, reason=""):
        for _ in range(repeats):
            self._publish_joint_cmd(HOME_POS_RAD_16)
            time.sleep(sleep_sec)
        if reason:
            self.get_logger().info(f"Home pose command sent ({reason}).")

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


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--right", action="store_true", help="Use the right Haptikos glove. Default is left.")
    parser.add_argument("--poll-rate", type=float, default=40.0, help="Haptikos polling rate in Hz.")
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = HaptikosToAllegroPublisher(right=parsed.right, poll_rate_hz=parsed.poll_rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_home(repeats=20, sleep_sec=0.02, reason="shutdown")
        node.client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
