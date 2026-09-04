#!/usr/bin/env python3
"""Collect Allegro thumb/middle demonstrations by physically moving the hand.

The Allegro controller remains running, but motor torque is switched off.  No
joint-position or gravity-compensation commands are published while teaching;
positions and torques are collected from the hand's joint-state feedback.
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
from std_msgs.msg import String

from allegro_package import AllegroHand
from py_node_exec import NodeExec
from xela_py import TactileSubscriber


HAND_TOPIC_PREFIX = "allegroHand_0"
CTRL_FREQ = 15.0
MAX_TIMESTEP = 300
DATA_DIR = "/home/handlingteam2/HASA/user/pranav/example/data/tasks/newsmallbolt"
TACTILE_TOPICS = ("thumb_tip", "middle_tip")

# Allegro order: index[0:4], middle[4:8], ring[8:12], thumb[12:16].
ACTIVE_JOINT_INDICES = np.array((4, 5, 6, 7, 12, 13, 14, 15), dtype=int)
ACTIVE_JOINT_NAMES = (
    "middle_0", "middle_1", "middle_2", "middle_3",
    "thumb_0", "thumb_1", "thumb_2", "thumb_3",
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


class DirectTeachingRobot:
    def __init__(self, hand_topic_prefix, ctrl_freq):
        self.hand = AllegroHand(
            hand_topic_prefix=hand_topic_prefix, ctrl_freq=ctrl_freq
        )
        self.tactile = {
            topic: TactileSubscriber(topic_prefix=topic)
            for topic in TACTILE_TOPICS
        }

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

        positions = np.asarray(hand["jnt_pos"], dtype=float)
        velocities = np.asarray(hand["jnt_vel"], dtype=float)
        torques = np.asarray(hand["jnt_trq"], dtype=float)
        last_position_command = hand.get("jnt_cmd_pos")
        if last_position_command is not None:
            last_position_command = np.asarray(last_position_command, dtype=float)
        return {
            "hand": hand,
            "tactile": tactile,
            "teaching": {
                "finger_pair": "thumb+middle",
                "jnt_names": ACTIVE_JOINT_NAMES,
                "jnt_indices": ACTIVE_JOINT_INDICES.copy(),
                "jnt_pos": positions[ACTIVE_JOINT_INDICES].copy(),
                "jnt_vel": velocities[ACTIVE_JOINT_INDICES].copy(),
                "jnt_trq": torques[ACTIVE_JOINT_INDICES].copy(),
                "jnt_cmd_pos": (
                    last_position_command[ACTIVE_JOINT_INDICES].copy()
                    if last_position_command is not None
                    else None
                ),
                # In controller `off` mode the effective motor torque command
                # is zero.  jnt_trq above is the measured torque feedback.
                "jnt_cmd_trq": np.zeros(len(ACTIVE_JOINT_INDICES), dtype=float),
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
    return parser.parse_args(args)


def main(args=None):
    config = parse_args(args)
    if config.frequency <= 0 or config.max_timestep <= 0:
        raise ValueError("--frequency and --max-timestep must be positive")

    node = NodeExec(node_name="thumb_middle_direct_teaching", freq=config.frequency)
    node.spin_thread_start()
    robot = DirectTeachingRobot(config.hand_prefix, config.frequency)
    episodes = EpisodeController(os.path.abspath(config.data_dir))
    keys = KeyboardHandler(episodes)

    print("[Info] Keep clear of pinch points, then physically move thumb and middle.")
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

    # Send twice because the first publisher message can precede connection.
    robot.disable_torque()
    time.sleep(0.25)
    robot.disable_torque()

    try:
        while node.ok() and not episodes.exit_requested:
            if episodes.collecting:
                file_path, timestep = episodes.next_path()
                if file_path is not None:
                    observation = robot.get_observation(tactile_offset)
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
        robot.disable_torque()
        time.sleep(0.1)
        keys.stop()
        node.spin_thread_finish()


if __name__ == "__main__":
    main()
