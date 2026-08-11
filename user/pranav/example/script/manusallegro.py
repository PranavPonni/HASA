#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys
import ctypes
import time
from pathlib import Path


def _prepend_env_path(var_name, path_value):
    """Prepend a path segment to an env var if it is not already present."""
    if not path_value:
        return
    current = os.environ.get(var_name, "")
    parts = [p for p in current.split(":") if p]
    if path_value in parts:
        return
    os.environ[var_name] = f"{path_value}:{current}" if current else path_value


def _bootstrap_local_manus_msgs_path():
    """Allow direct python execution without sourcing manus_ros2_ws setup."""
    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        ws_root = parent
        install_root = ws_root / "manus_ros2_ws" / "install"
        pkg_root = install_root / "manus_ros2_msgs"
        install_base = pkg_root / "lib"
        if not install_base.exists() or not install_root.exists() or not pkg_root.exists():
            continue

        # Python message modules
        for candidate in sorted(install_base.glob("python*/site-packages")):
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

        # Shared libraries for rosidl type support.
        _prepend_env_path("LD_LIBRARY_PATH", str(install_base))
        _prepend_env_path("LD_LIBRARY_PATH", str(install_root / "lib"))

        # LD_LIBRARY_PATH updates during runtime can be too late for dlopen on some setups.
        # Preload Manus-generated libraries so rosidl type support imports succeed.
        preload_libs = [
            install_base / "libmanus_ros2_msgs__rosidl_generator_c.so",
            install_base / "libmanus_ros2_msgs__rosidl_typesupport_c.so",
            install_base / "libmanus_ros2_msgs__python.so",
        ]
        for lib_path in preload_libs:
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

        # Help ament index discover overlay packages.
        _prepend_env_path("AMENT_PREFIX_PATH", str(pkg_root))
        _prepend_env_path("AMENT_PREFIX_PATH", str(install_root))
        _prepend_env_path("CMAKE_PREFIX_PATH", str(pkg_root))
        _prepend_env_path("CMAKE_PREFIX_PATH", str(install_root))

        # Ensure core ROS2 Python packages (geometry_msgs, etc.) are importable
        # even when the ROS2 setup.bash has not been sourced.
        ros_site = "/opt/ros/foxy/lib/python3.8/site-packages"
        if os.path.isdir(ros_site) and ros_site not in sys.path:
            sys.path.append(ros_site)
        ros_lib = "/opt/ros/foxy/lib"
        _prepend_env_path("LD_LIBRARY_PATH", ros_lib)

        return

_bootstrap_local_manus_msgs_path()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from manus_ros2_msgs.msg import ManusGlove, ManusVibrationCommand

# =========================================================
# 固定設定（あなたのコマンドライン指定をコード内に反映）
# =========================================================
IN_TOPIC = "/manus_glove_0"
OUT_TOPIC = "/allegroHand_0/joint_cmd"
HAPTIC_TOPIC = "/manus_glove_0/vibration_cmd"
SIDE_FILTER = "Left"   # "Right" / "Left" / ""(無視)
SIDE_FILTER_NORM = SIDE_FILTER.strip().lower()
# =========================================================

# ---- Haptic feedback parameters ----
# Average of MCP+PIP stretch angles (degrees) that maps to intensity 1.0
HAPTIC_MAX_FLEX_DEG  = 55.0   # angle at which vibration reaches full intensity
HAPTIC_DEAD_ZONE_DEG = 12.0   # below this no vibration (ignore noise)

# ---- Index/middle/ring self-touch parameters ----
# Self-touch fades in only while all three fingers are curled.  The outer
# fingers abduct toward the middle finger; contact itself is then provided by
# the Allegro hand/controller's normal physical self-collision.
SELF_TOUCH_START_DEG = 22.0
SELF_TOUCH_FULL_DEG = 48.0
SELF_TOUCH_INDEX_SPREAD_RAD = 0.36  # positive moves index toward middle
SELF_TOUCH_RING_SPREAD_RAD = -0.24  # negative moves ring toward middle
SELF_TOUCH_SYNC_FLEX = 0.65    # match fingertip heights without discarding glove motion

PI = math.pi
MOTOR_COUNT = 20
ALLEGRO_JOINT_COUNT = 16

# Allegro home pose from provided initial_position params (degrees).
# Order is j00..j03, j10..j13, j20..j23, j30..j33.
HOME_POS_DEG_16 = [
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
]
HOME_POS_RAD_16 = [d * (PI / 180.0) for d in HOME_POS_DEG_16]

# Manus ergonomics 20個 -> q[0..19] (degree) へ詰める順番
ERG_ORDER = [
    "ThumbMCPSpread",   # q[0]
    "ThumbMCPStretch",  # q[1]
    "ThumbPIPStretch",  # q[2]
    "ThumbDIPStretch",  # q[3]
    "IndexSpread",      # q[4]
    "IndexMCPStretch",  # q[5]
    "IndexPIPStretch",  # q[6]
    "IndexDIPStretch",  # q[7]
    "MiddleSpread",     # q[8]
    "MiddleMCPStretch", # q[9]
    "MiddlePIPStretch", # q[10]
    "MiddleDIPStretch", # q[11]
    "RingSpread",       # q[12]
    "RingMCPStretch",   # q[13]
    "RingPIPStretch",   # q[14]
    "RingDIPStretch",   # q[15]
    "PinkySpread",      # q[16]
    "PinkyMCPStretch",  # q[17]
    "PinkyPIPStretch",  # q[18]
    "PinkyDIPStretch",  # q[19]
]

# Allegro 16点として publish する name（index 0..15）
JOINT_NAMES_16 = [
    "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7",
    "joint_8", "joint_9", "joint_10", "joint_11",
    "joint_12", "joint_13", "joint_14", "joint_15",
]

# ---- Smoothing parameters ----
EMA_ALPHA = 0.15          # EMA filter on raw glove input (0..1, lower = smoother)
INTERP_RATE_HZ = 40.0    # Fixed-rate publish frequency
INTERP_SPEED = 10.0        # Interpolation speed toward target (per second, ~0..20)


class ManusToAllegroPublisher(Node):
    SELF_TOUCH_HAPTIC_FINGERS = (1, 2, 3)  # index, middle, ring

    def __init__(self, node_name="manus_to_allegro_joint_cmd_publisher"):
        super().__init__(node_name)
        self.recv_count = 0
        self.pub_count = 0
        self.drop_count = 0
        self.last_side = ""
        self.last_erg_count = 0

        # Smoothing state
        self._target_pos = list(HOME_POS_RAD_16)   # latest converted target
        self._current_pos = list(HOME_POS_RAD_16)  # interpolated position being published
        self._ema_deg = None                        # EMA-filtered raw glove degrees
        self._have_target = False

        # /manus_glove_0 は BEST_EFFORT publisher が混在するので subscriber は BEST_EFFORT が安全
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(ManusGlove, IN_TOPIC, self.cb, qos)
        self.pub = self.create_publisher(JointState, OUT_TOPIC, 10)
        self.haptic_pub = self.create_publisher(ManusVibrationCommand, HAPTIC_TOPIC, 10)
        self.diag_timer = self.create_timer(2.0, self._diag_cb)

        # Fixed-rate publisher for smooth, dense output
        dt = 1.0 / INTERP_RATE_HZ
        self._interp_timer = self.create_timer(dt, self._interp_cb)

        self.get_logger().info(
            f"Subscribing: {IN_TOPIC}  -> Publishing: {OUT_TOPIC}\n"
            f"Side filter: {SIDE_FILTER} (empty means no filter)\n"
            f"Output: 16 values for Allegro joint_cmd\n"
            f"Smoothing: EMA alpha={EMA_ALPHA}, interp {INTERP_RATE_HZ}Hz, speed={INTERP_SPEED}"
        )

        self.publish_home(repeats=20, sleep_sec=0.02, reason="startup")

    # ----- fixed-rate interpolation publisher -----
    def _interp_cb(self):
        if not self._have_target:
            return
        dt = 1.0 / INTERP_RATE_HZ
        alpha = min(1.0, INTERP_SPEED * dt)  # per-tick blend factor
        for i in range(ALLEGRO_JOINT_COUNT):
            self._current_pos[i] += alpha * (self._target_pos[i] - self._current_pos[i])
        self._publish_joint_cmd(list(self._current_pos))
        self.pub_count += 1

    def _diag_cb(self):
        self.get_logger().info(
            f"diag recv={self.recv_count} pub={self.pub_count} drop={self.drop_count} "
            f"last_side='{self.last_side}' last_erg_count={self.last_erg_count}"
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

    def _publish_haptics(self, intensities_5):
        """Publish ManusVibrationCommand [Thumb, Index, Middle, Ring, Pinky]."""
        cmd = ManusVibrationCommand()
        cmd.intensities = [float(v) for v in intensities_5]
        self.haptic_pub.publish(cmd)

    def _compute_pinch_intensities(self, q_deg):
        """Return vibration intensities [Thumb, Index, Middle, Ring, Pinky] in [0,1].

        Each finger's intensity is the geometric mean of thumb flex and finger flex,
        normalised and dead-zoned so that intensity rises only when *both* the thumb
        and that finger are actively curling toward each other.

        q_deg indices follow ERG_ORDER:
          [1] ThumbMCPStretch  [2] ThumbPIPStretch
          [5] IndexMCPStretch  [6] IndexPIPStretch
          [9] MiddleMCPStretch [10] MiddlePIPStretch
          [13] RingMCPStretch  [14] RingPIPStretch
          [17] PinkyMCPStretch [18] PinkyPIPStretch
        """
        thumb_flex  = (q_deg[1]  + q_deg[2])  * 0.5
        index_flex  = (q_deg[5]  + q_deg[6])  * 0.5
        middle_flex = (q_deg[9]  + q_deg[10]) * 0.5
        ring_flex   = (q_deg[13] + q_deg[14]) * 0.5
        pinky_flex  = (q_deg[17] + q_deg[18]) * 0.5

        span = max(HAPTIC_MAX_FLEX_DEG - HAPTIC_DEAD_ZONE_DEG, 1.0)

        def _intensity(finger_flex):
            # Geometric mean so *both* thumb and finger must flex to get vibration
            score = math.sqrt(max(0.0, thumb_flex) * max(0.0, finger_flex))
            score = max(0.0, score - HAPTIC_DEAD_ZONE_DEG) / span
            return min(1.0, score)

        i_index  = _intensity(index_flex)
        i_middle = _intensity(middle_flex)
        i_ring   = _intensity(ring_flex)
        i_pinky  = _intensity(pinky_flex)
        i_thumb  = max(i_index, i_middle, i_ring, i_pinky)

        return [i_thumb, i_index, i_middle, i_ring, i_pinky]

    def _self_touch_amount(self, q_deg):
        """Return a smooth [0, 1] activation when index, middle and ring curl."""
        flex = (
            (q_deg[5] + q_deg[6]) * 0.5,
            (q_deg[9] + q_deg[10]) * 0.5,
            (q_deg[13] + q_deg[14]) * 0.5,
        )
        # Requiring the least-curled finger prevents accidental self-touch when
        # only one finger is being used to manipulate an object.
        x = (min(flex) - SELF_TOUCH_START_DEG) / max(
            SELF_TOUCH_FULL_DEG - SELF_TOUCH_START_DEG, 1.0
        )
        x = min(1.0, max(0.0, x))
        return x * x * (3.0 - 2.0 * x)  # smoothstep

    def _apply_three_finger_self_touch(self, mqd_20, q_deg):
        """Bias the three non-thumb fingertips into a controllable self-touch."""
        amount = self._self_touch_amount(q_deg)
        if amount <= 0.0:
            return amount

        # In Allegro coordinates positive index spread and negative ring spread
        # point toward the centre finger.  Keep the middle finger centred.
        spread_targets = {
            4: SELF_TOUCH_INDEX_SPREAD_RAD,
            8: 0.0,
            12: SELF_TOUCH_RING_SPREAD_RAD,
        }
        for joint, target in spread_targets.items():
            mqd_20[joint] += amount * (target - mqd_20[joint])

        # Align corresponding flex joints so adjacent fingertip pads arrive at
        # approximately the same height.  Partial blending preserves individual
        # glove control for in-hand manipulation.
        sync_amount = amount * SELF_TOUCH_SYNC_FLEX
        for offsets in ((5, 9, 13), (6, 10, 14), (7, 11, 15)):
            mean = sum(mqd_20[i] for i in offsets) / len(offsets)
            for i in offsets:
                mqd_20[i] += sync_amount * (mean - mqd_20[i])

        return amount

    # -----------------------------------------------------
    # SetTargetLeft 相当処理: q(deg, len=20) -> mQd(rad, len=20)
    # -----------------------------------------------------
    def set_target_left(self, q_deg):
        # C++の配列に合わせる
        dir_ = [ 1,  0.4,  1,  1,
                -1.0,  1,  1,  1,
                -1.0,  1,  1,  1,
                -1.0,  1,  1,  1,
                0.5, -0.5,  1,  1.2]

        calib = [1, 1.35, 1.6, 1.2,
                 1.4, 1, 1.2, 1.2,
                 1.47, 0.98, 1.18, 1.18,
                 1.7, 1, 1, 1,
                 0.5, 0.5, 1, 1]

        qd = [0.0] * MOTOR_COUNT
        mQd = [0.0] * MOTOR_COUNT

        # ---- Thumb special offsets (Quantum metagloves initial angles) ----
        qd[0] = (58.5 - q_deg[1]) * (PI / 180.0)
        qd[1] = (q_deg[0] + 20.0) * (PI / 180.0)

        # rest straightforward deg->rad
        for i in range(2, 16):
            qd[i] = q_deg[i] * (PI / 180.0)

        # ---- Thumb PIP offset: preload enough to lift while still allowing return motion ----
        qd[2] = qd[2] + 24.0 * (PI / 180.0)

        # ---- Thumb Spread offset: ensure joint 12 always has some command ----
        qd[0] = qd[0] + 18.0 * (PI / 180.0)

        # ---- Index Spread offset: forward bias ----
        qd[5] = qd[5] + 15.0 * (PI / 180.0)

        # ---- Index Spread offset: sideward bias ----
        qd[4] = qd[4] + 0.0 * (PI / 180.0)

        # ---- Middle Spread offset: sideward bias ----
        qd[8] = qd[8] + 20.0 * (PI / 180.0)

        # ---- Middle Spread bend  offset: sideward bias ----
        qd[9] = qd[9] + 20.0 * (PI / 180.0)

        # ---- Ring Spread offset: sideward bias ----
        qd[12] = qd[12] + 36.0 * (PI / 180.0)

        # ---- Ring Spread bend  offset: sideward bias ----
        qd[13] = qd[13] + 20.0 * (PI / 180.0)

        # ---- apply dir + calibration + clamp rules ----
        for i in range(MOTOR_COUNT):
            mQd[i] = qd[i] * calib[i] * dir_[i]

            if i in (2, 4, 8, 12, 16, 17):
                # pass-through (no clamp): spreads + thumb PIP
                pass
            elif i == 1:
                # prevent backward bending (thumb joint special)
                if mQd[i] <= 0.0:
                    mQd[i] = 0.0
            else:
                # prevent backward bending (others)
                if mQd[i] <= 0.0:
                    mQd[i] = 0.0

        return mQd

    # -----------------------------------------------------
    # callback
    # -----------------------------------------------------
    def cb(self, msg: ManusGlove):
        self.recv_count += 1
        self.last_side = msg.side
        self.last_erg_count = len(msg.ergonomics)

        if SIDE_FILTER_NORM and msg.side.strip().lower() != SIDE_FILTER_NORM:
            self.drop_count += 1
            return

        # ergonomics: string type -> float value (degree想定)
        erg_map = {e.type: float(e.value) for e in msg.ergonomics}

        # q[0..19] in degrees (missing types -> 0)
        q_deg_raw = [erg_map.get(k, 0.0) for k in ERG_ORDER]

        # EMA filter on raw glove degrees to suppress sensor noise
        if self._ema_deg is None:
            self._ema_deg = list(q_deg_raw)
        else:
            for i in range(len(q_deg_raw)):
                self._ema_deg[i] += EMA_ALPHA * (q_deg_raw[i] - self._ema_deg[i])

        # Work on a copy because self-touch processing must not alter EMA state.
        q_deg = list(self._ema_deg)

        # SetTargetLeft: filtered q(deg) -> mQd(rad)
        mQd_20 = self.set_target_left(q_deg)

        # Bring index/ring toward the middle finger as all three curl.  This is
        # a joint-space self-touch command; real contact/force limiting remains
        # the responsibility of the Allegro controller and hardware.
        self_touch = self._apply_three_finger_self_touch(mQd_20, q_deg)

        # Haptic feedback combines thumb pinching and commanded self-touch.
        pinch = self._compute_pinch_intensities(q_deg)
        if self_touch > 0.0:
            for i in self.SELF_TOUCH_HAPTIC_FINGERS:
                pinch[i] = max(pinch[i], self_touch)
        self._publish_haptics(pinch)

        # Source blocks are [thumb, index, middle, ring] each 4 joints.
        # Allegro command order is [index, middle, ring, thumb].
        self._target_pos = mQd_20[4:8] + mQd_20[8:12] + mQd_20[12:16] + mQd_20[0:4]

        # On first target, snap current position so there's no startup lag
        if not self._have_target:
            self._current_pos = list(self._target_pos)
        self._have_target = True


def main(args=None):
    rclpy.init(args=args)
    node = ManusToAllegroPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_home(repeats=20, sleep_sec=0.02, reason="shutdown")
        node._publish_haptics([0.0, 0.0, 0.0, 0.0, 0.0])  # stop all vibration
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
