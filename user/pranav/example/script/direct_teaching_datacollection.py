#!/usr/bin/env python3
"""Collect replayable Allegro thumb/middle direct-teaching demonstrations.

The Allegro controller remains running, but motor torque is switched off. No
joint-position or gravity-compensation commands are published while teaching.
Because physical direct teaching has no live controller target, each measured
pose is also stored as that timestep's desired position target for replay.
"""

import argparse
import glob
import os
import pickle
import shutil
import threading
import time

import numpy as np
import rospy
from pynput import keyboard
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from allegro_package import AllegroHand
from py_node_exec import NodeExec
from xela_py import TactileSubscriber


HAND_TOPIC_PREFIX = "allegroHand_0"
CTRL_FREQ = 15.0
MAX_TIMESTEP = 100
DATA_DIR = "/home/handlingteam2/HASA/user/pranav/example/data/tasks/smallbolt/M2"
TACTILE_TOPICS = ("thumb_tip", "middle_tip")
DEFAULT_TORQUE_BIAS = 0.08
DEFAULT_COMMAND_TIMEOUT = 0.5

# Allegro order: index[0:4], middle[4:8], ring[8:12], thumb[12:16].
ACTIVE_JOINT_INDICES = np.array((4, 5, 6, 7, 12, 13, 14, 15), dtype=int)
ACTIVE_JOINT_NAMES = (
    "middle_0", "middle_1", "middle_2", "middle_3",
    "thumb_0", "thumb_1", "thumb_2", "thumb_3",
)

# Positive torque closes the three flexion joints of the middle and thumb.
# Spread/abduction joints remain unbiased.  The values are desired training
# labels only; controller `off` mode keeps them from being applied while the
# demonstration is recorded.
TORQUE_BIAS_DIRECTION_16 = np.array(
    (
        0.0, 0.0, 0.0, 0.0,  # index
        0.0, 1.0, 1.0, 1.0,  # middle
        0.0, 0.0, 0.0, 0.0,  # ring
        0.0, 1.0, 1.0, 1.0,  # thumb
    ),
    dtype=float,
)


def flatten_dict(value, parent_key="", separator="_"):
    result = {}
    for key, item in value.items():
        flat_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(item, dict):
            result.update(flatten_dict(item, flat_key, separator))
        else:
            result[flat_key] = item
    return result


class PositionCommandUnavailable(RuntimeError):
    """Raised when external-command mode cannot provide a fresh target."""


class ExternalPositionCommandSubscriber:
    def __init__(self, topic):
        self.topic = topic
        self._lock = threading.Lock()
        self._position = None
        self._received_at = None
        self._subscriber = rospy.Subscriber(
            topic, JointState, self._callback, queue_size=1
        )

    def _callback(self, message):
        position = np.asarray(message.position, dtype=float)
        if position.shape != (16,):
            rospy.logwarn_throttle(
                1.0,
                "Ignoring %s command with %d positions; expected 16",
                self.topic,
                position.size,
            )
            return
        with self._lock:
            self._position = position.copy()
            self._received_at = time.monotonic()

    def latest(self, max_age):
        with self._lock:
            position = None if self._position is None else self._position.copy()
            received_at = self._received_at

        if position is None or received_at is None:
            raise PositionCommandUnavailable(
                f"no position command received on {self.topic}"
            )

        age = time.monotonic() - received_at
        if age > max_age:
            raise PositionCommandUnavailable(
                f"position command on {self.topic} is stale "
                f"({age:.3f}s old; limit {max_age:.3f}s)"
            )
        return position, age


class DirectTeachingRobot:
    def __init__(
        self,
        hand_topic_prefix,
        ctrl_freq,
        torque_bias,
        command_source="measured",
        command_topic=None,
        command_timeout=DEFAULT_COMMAND_TIMEOUT,
    ):
        self.hand = AllegroHand(
            hand_topic_prefix=hand_topic_prefix, ctrl_freq=ctrl_freq
        )
        self.tactile = {
            topic: TactileSubscriber(topic_prefix=topic)
            for topic in TACTILE_TOPICS
        }
        self.torque_bias_16 = TORQUE_BIAS_DIRECTION_16 * float(torque_bias)
        self.command_source = command_source
        self.command_timeout = float(command_timeout)
        self.command_topic = command_topic
        self.external_commands = (
            ExternalPositionCommandSubscriber(command_topic)
            if command_source == "external"
            else None
        )

    def disable_torque(self):
        # Keep the controller alive while disabling motor output.  Do not call
        # set_joint_cmd: a position command switches back to joint-PD mode.
        self.hand.pub_grasp_cmd.publish(String(data="off"))
        rospy.loginfo("Direct teaching enabled: Allegro motor torque OFF")

    def get_observation(self, tactile_offset):
        hand = self.hand.get_obs()
        tactile = {}
        for name, subscriber in self.tactile.items():
            reading = subscriber.get_obs()
            baseline = tactile_offset.get(name)
            tactile[name] = (
                np.asarray(reading) - baseline
                if reading is not None and baseline is not None
                else reading
            )

        positions = np.asarray(hand["jnt_pos"], dtype=float).copy()
        velocities = np.asarray(hand["jnt_vel"], dtype=float).copy()
        measured_torques = np.asarray(hand["jnt_trq"], dtype=float).copy()
        controller_last_position_command = hand.get("jnt_cmd_pos")
        if controller_last_position_command is not None:
            controller_last_position_command = np.asarray(
                controller_last_position_command, dtype=float
            ).copy()

        if self.command_source == "measured":
            # With motor torque disabled there is no live controller command.
            # The demonstrated pose is therefore the desired replay target.
            replay_target = positions.copy()
            command_source_label = "direct_teaching_measured_pose"
            external_command_age = None
        else:
            # Never fall back to the measured pose in external mode. Missing or
            # stale commands must stop the sample from being written so bad
            # command streams are immediately visible during collection.
            replay_target, external_command_age = self.external_commands.latest(
                self.command_timeout
            )
            command_source_label = f"external_topic:{self.command_topic}"

        # `hand_jnt_trq` is the canonical training target used by the motion
        # models.  Preserve hardware feedback under an unambiguous key and put
        # the requested thumb/middle closing bias in the target field.
        hand = dict(hand)
        hand["jnt_trq_measured"] = measured_torques.copy()
        hand["jnt_trq"] = self.torque_bias_16.copy()
        hand["jnt_cmd_pos_controller_last"] = controller_last_position_command
        hand["jnt_cmd_pos"] = replay_target
        hand["jnt_cmd_pos_source"] = command_source_label
        hand["jnt_cmd_pos_external_age_s"] = external_command_age
        return {
            "hand": hand,
            "tactile": tactile,
            "teaching": {
                "finger_pair": "thumb+middle",
                "jnt_names": ACTIVE_JOINT_NAMES,
                "jnt_indices": ACTIVE_JOINT_INDICES.copy(),
                "jnt_pos": positions[ACTIVE_JOINT_INDICES].copy(),
                "jnt_vel": velocities[ACTIVE_JOINT_INDICES].copy(),
                "jnt_trq": self.torque_bias_16[ACTIVE_JOINT_INDICES].copy(),
                "jnt_trq_measured": measured_torques[ACTIVE_JOINT_INDICES].copy(),
                "jnt_cmd_pos": replay_target[ACTIVE_JOINT_INDICES].copy(),
                "jnt_cmd_pos_source": command_source_label,
                "jnt_cmd_trq": self.torque_bias_16[ACTIVE_JOINT_INDICES].copy(),
                "jnt_cmd_trq_applied_during_teaching": np.zeros(
                    len(ACTIVE_JOINT_INDICES), dtype=float
                ),
                "control_mode": "off",
            },
            "timestamp": time.time(),
        }


class EpisodeController:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.lock = threading.Lock()
        self.episode_dir = None
        self.timestep = 0
        self.collecting = False
        self.exit_requested = False

    def start(self):
        with self.lock:
            if self.collecting:
                return
            os.makedirs(self.data_dir, exist_ok=True)
            indices = []
            for path in glob.glob(os.path.join(self.data_dir, "episode[0-9]*")):
                suffix = os.path.basename(path).replace("episode", "", 1)
                if suffix.isdigit():
                    indices.append(int(suffix))
            self.episode_dir = os.path.join(
                self.data_dir, f"episode{max(indices, default=-1) + 1}"
            )
            os.makedirs(self.episode_dir, exist_ok=False)
            self.timestep = 0
            self.collecting = True
            print(f"[Info] Recording {self.episode_dir}")

    def finish(self, keep):
        with self.lock:
            if not self.collecting:
                return
            episode_dir = self.episode_dir
            self.collecting = False
            self.episode_dir = None
            self.timestep = 0
        if keep:
            print(f"[Info] Episode saved: {episode_dir}")
        else:
            shutil.rmtree(episode_dir)
            print(f"[Info] Episode rejected and deleted: {episode_dir}")

    def next_path(self):
        with self.lock:
            if not self.collecting:
                return None, None
            timestep = self.timestep
            self.timestep += 1
            return os.path.join(self.episode_dir, f"timestep{timestep}.pkl"), timestep


class KeyboardHandler:
    def __init__(self, episodes):
        self.episodes = episodes
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        print("[Keys] s=start, y=save episode, n=reject episode, q=quit while idle")

    def _on_press(self, key):
        try:
            char = key.char.lower()
        except (AttributeError, TypeError):
            return
        try:
            if char == "s" and not self.episodes.collecting:
                self.episodes.start()
            elif char == "y":
                self.episodes.finish(keep=True)
            elif char == "n":
                self.episodes.finish(keep=False)
            elif char == "q" and not self.episodes.collecting:
                self.episodes.exit_requested = True
        except Exception as error:
            print(f"[Error] Keyboard action failed: {error}")

    def stop(self):
        self.listener.stop()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Direct-teaching data collection for Allegro thumb+middle."
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--frequency", type=float, default=CTRL_FREQ)
    parser.add_argument("--max-timestep", type=int, default=MAX_TIMESTEP)
    parser.add_argument("--hand-prefix", default=HAND_TOPIC_PREFIX)
    parser.add_argument(
        "--command-source",
        choices=("measured", "external"),
        default="measured",
        help=(
            "measured: torque-off teaching with measured poses as replay targets; "
            "external: record fresh JointState targets from --command-topic"
        ),
    )
    parser.add_argument(
        "--command-topic",
        default=None,
        help=(
            "JointState topic carrying the live desired positions in external "
            "mode (default: /<hand-prefix>/joint_cmd)"
        ),
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT,
        help="maximum command age in seconds in external mode (default: %(default)s)",
    )
    parser.add_argument(
        "--torque-bias",
        type=float,
        default=DEFAULT_TORQUE_BIAS,
        help=(
            "closing-torque label in Nm for thumb/middle flexion joints "
            "(default: %(default)s; recorded but not applied during teaching)"
        ),
    )
    return parser.parse_args(args)


def main(args=None):
    config = parse_args(args)
    if (
        config.frequency <= 0
        or config.max_timestep <= 0
        or config.command_timeout <= 0
    ):
        raise ValueError(
            "--frequency, --max-timestep, and --command-timeout must be positive"
        )
    if not 0.0 <= config.torque_bias <= 0.5:
        raise ValueError("--torque-bias must be between 0.0 and 0.5 Nm")

    node = NodeExec(node_name="thumb_middle_direct_teaching", freq=config.frequency)
    node.spin_thread_start()
    command_topic = config.command_topic or (
        f"/{config.hand_prefix.strip('/')}/joint_cmd"
    )
    robot = DirectTeachingRobot(
        config.hand_prefix,
        config.frequency,
        config.torque_bias,
        command_source=config.command_source,
        command_topic=command_topic,
        command_timeout=config.command_timeout,
    )
    episodes = EpisodeController(os.path.abspath(config.data_dir))
    keys = KeyboardHandler(episodes)

    print("[Info] Keep clear of pinch points, then physically move thumb and middle.")
    if config.command_source == "measured":
        print(
            "[Info] Replay targets: saving each demonstrated measured pose as "
            "hand_jnt_cmd_pos."
        )
    else:
        print(
            f"[Info] Replay targets: recording fresh JointState positions from "
            f"{command_topic}."
        )
        print(
            "[Safety] This recorder does not enable or configure the motors. "
            "The external controller must already be running in a verified "
            "compliant/low-stiffness mode."
        )
    print(
        "[Setup] Before each episode, return the bolt to the same marked "
        "start position and orientation."
    )
    print(
        f"[Info] Recording {config.torque_bias:.3f} Nm thumb/middle closing-torque "
        "targets (not applied during teaching)."
    )
    print("[Info] Waiting one second for tactile data before measuring baselines...")
    time.sleep(1.0)
    tactile_offset = {
        name: (np.asarray(value).copy() if value is not None else None)
        for name, value in (
            (name, subscriber.get_obs())
            for name, subscriber in robot.tactile.items()
        )
    }
    for name, baseline in tactile_offset.items():
        status = "OK" if baseline is not None else "NO DATA (saved as None)"
        print(f"[Tactile] {name}: {status}")

    if config.command_source == "measured":
        # Send twice because the first publisher message can precede connection.
        robot.disable_torque()
        time.sleep(0.25)
        robot.disable_torque()
    else:
        print(
            "[Info] External mode leaves motor state under the external "
            "controller's ownership."
        )

    try:
        while node.ok() and not episodes.exit_requested:
            if episodes.collecting:
                try:
                    observation = robot.get_observation(tactile_offset)
                except PositionCommandUnavailable as error:
                    rospy.logwarn_throttle(
                        1.0, "Not recording direct-teaching sample: %s", error
                    )
                    node.sleep()
                    continue

                file_path, timestep = episodes.next_path()
                if file_path is not None:
                    with open(file_path, "wb") as file:
                        pickle.dump(flatten_dict(observation), file)
                    if timestep + 1 >= config.max_timestep:
                        print("[Info] Maximum timestep reached; saving episode.")
                        episodes.finish(keep=True)
            node.sleep()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        print("\n[Info] Stopping direct teaching...")
    finally:
        if episodes.collecting:
            episodes.finish(keep=True)
        if config.command_source == "measured":
            robot.disable_torque()
            time.sleep(0.1)
        keys.stop()
        node.spin_thread_finish()


if __name__ == "__main__":
    main()
