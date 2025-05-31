#!/usr/bin/env python3

import os
import shutil
import glob
import threading
import pickle
import numpy as np
from pynput import keyboard
from playsound import playsound  # Ensure playsound is installed and available

from py_node_exec import NodeExec
from allegro_package import AllegroHand
from xela_py import TactileSubscriber
from allegro_leader import AllegroPrecesionGrasp

DATA_DIR = "/home/handlingteam2/HASA/user/pranav/example/data/selftouch"
MAX_TIMESTEP = 400
CTRL_FREQ = 20.0
HAND_TOPIC_PREFIX = "allegroHand_0"
TACTILE_TOPIC_PREFIXES = ["index_tip", "thumb_tip"]
BACKGROUND_AUDIO_PATH = "/home/handlingteam2/HASA/user/pranav/example/script/util/jazz.mp3"


class XelAllegro:
    def __init__(
        self,
        ctrl_freq: float = 20.0,
        hand_topic_prefix: str = "allegroHand_0",
        tactile_topic_prefixes: list = ["index_tip", "thumb_tip"],
    ):
        self._hand = AllegroHand(
            hand_topic_prefix=hand_topic_prefix,
            ctrl_freq=ctrl_freq
        )
        self._tactile_sensors = {
            prefix: TactileSubscriber(topic_prefix=prefix)
            for prefix in tactile_topic_prefixes
        }

    def set_joint_command(self, cmd: np.ndarray):
        if not isinstance(cmd, np.ndarray):
            raise TypeError("cmd must be a numpy.ndarray.")
        if cmd.shape != (16,):
            raise ValueError("cmd must have shape (16,).")
        self._hand.set_joint_cmd(cmd)
        return self.get_observation()

    def get_observation(self) -> dict:
        tactile_obs = {prefix: sub.get_obs()
                       for prefix, sub in self._tactile_sensors.items()}
        hand_obs = self._hand.get_obs()
        return {"tactile": tactile_obs, "hand": hand_obs}


class DataCollectorController:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = "idle"
        self._episode_dir = None
        self._timestep = 0
        self._collecting = False
        self._exit_flag = False

    def reset_to_idle(self):
        with self._lock:
            self._state = "idle"
            self._collecting = False
            self._episode_dir = None
            self._timestep = 0

    def start_new_episode(self, base_dir: str) -> bool:
        with self._lock:
            os.makedirs(base_dir, exist_ok=True)
            dirs = glob.glob(os.path.join(base_dir, "episode[0-9]*"))
            max_idx = -1
            for d in dirs:
                name = os.path.basename(d)
                try:
                    idx = int(name.replace("episode", ""))
                    max_idx = max(max_idx, idx)
                except ValueError:
                    continue
            next_idx = max_idx + 1
            new_dir = os.path.join(base_dir, f"episode{next_idx}")
            try:
                os.makedirs(new_dir, exist_ok=False)
            except Exception as e:
                print(f"[Error] Failed to create '{new_dir}': {e}")
                return False
            self._episode_dir = new_dir
            self._timestep = 0
            self._state = "collecting"
            self._collecting = True
            print(f"[Info] Started data collection in '{new_dir}'")
            return True

    def end_episode_success(self, hand: AllegroHand):
        with self._lock:
            if not self._collecting:
                return
            print(f"[Info] Episode succeeded: data kept at '{self._episode_dir}'")
            self._collecting = False
            self._state = "idle"

    def end_episode_failure(self, hand: AllegroHand):
        with self._lock:
            if not self._collecting:
                return
            try:
                shutil.rmtree(self._episode_dir)
                print(f"[Info] Episode failed: deleted '{self._episode_dir}'")
            except Exception as e:
                print(f"[Error] Error deleting '{self._episode_dir}': {e}")
            self._collecting = False
            self._state = "idle"
            self._episode_dir = None
            self._timestep = 0

    def request_exit(self):
        with self._lock:
            if self._state == "idle":
                self._exit_flag = True
                print("[Info] Exit flag set; program will terminate")
            else:
                print("[Info] Cannot exit while collecting; press 'y' or 'n' first")

    def is_collecting(self) -> bool:
        with self._lock:
            return self._collecting

    def should_exit(self) -> bool:
        with self._lock:
            return self._exit_flag

    def get_state(self) -> str:
        with self._lock:
            return self._state

    def increment_timestep(self) -> int:
        with self._lock:
            self._timestep += 1
            return self._timestep

    def get_timestep(self) -> int:
        with self._lock:
            return self._timestep

    def get_episode_dir(self) -> str:
        with self._lock:
            return self._episode_dir


class AudioPlayer:
    def __init__(self, audio_path: str):
        self._audio_path = audio_path
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)

    def _play_loop(self):
        while not self._stop_event.is_set():
            try:
                playsound(self._audio_path)
            except Exception as e:
                print(f"[Warning] Failed to play audio '{self._audio_path}': {e}")
                break

    def start(self):
        if not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._play_loop, daemon=True)
            self._thread.start()
            print(f"[Info] Background audio started: '{self._audio_path}'")

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class KeyboardHandler:
    def __init__(self, controller: DataCollectorController, robot: XelAllegro,
                 teleop: AllegroPrecesionGrasp, audio_player: AudioPlayer):
        self._controller = controller
        self._robot = robot
        self._teleop = teleop
        self._hand = robot._hand
        self._audio_player = audio_player
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self.start()

    def start(self):
        self._listener.start()
        print("[Info] Keyboard listener started.")
        print("  - Idle: 's' → start collection, 'q' → quit, 'h' → go to teleop device pos")
        print("  - Collecting: 'y' → success (keep data), 'n' → failure (delete data), 'h' → go to teleop device pos")

    def stop(self):
        self._listener.stop()

    def _on_press(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return

        state = self._controller.get_state()
        if state == "idle":
            if char == "s":
                if self._controller.start_new_episode(DATA_DIR):
                    self._audio_player.start()
            elif char == "q":
                self._controller.request_exit()

        elif state == "collecting":
            if char == "y":
                self._controller.end_episode_success(self._hand)
                self._audio_player.stop()
            elif char == "n":
                self._controller.end_episode_failure(self._hand)
                self._audio_player.stop()

def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def main():
    node_exec = NodeExec(node_name="data_collection", freq=CTRL_FREQ)
    node_exec.spin_thread_start()

    robot = XelAllegro(
        ctrl_freq=CTRL_FREQ,
        hand_topic_prefix=HAND_TOPIC_PREFIX,
        tactile_topic_prefixes=TACTILE_TOPIC_PREFIXES
    )
    teleop = AllegroPrecesionGrasp(wall_kp=0.0)
    controller = DataCollectorController()

    audio_player = AudioPlayer(BACKGROUND_AUDIO_PATH)

    keyboard_handler = KeyboardHandler(controller, robot, teleop, audio_player)

    print(f"[Info] MAX_TIMESTEP = {MAX_TIMESTEP}")

    try:
        while not controller.should_exit():
            if not controller.is_collecting():
                try:
                    leader_cmd = teleop.update()
                    follower_cmd = np.zeros(16, dtype=float)
                    finger_indices = [0, 1, 2, 3, 12, 13, 14, 15]
                    for i, idx in enumerate(finger_indices):
                        follower_cmd[idx] = leader_cmd[i]
                    robot.set_joint_command(follower_cmd)
                    audio_player.stop()
                except Exception as e:
                    print(f"[Warning] Failed to follow teleop in idle: {e}")
                node_exec.sleep()
                continue

            while node_exec.ok() and controller.is_collecting():
                ts = controller.get_timestep()
                episode_dir = controller.get_episode_dir()
                if episode_dir is None:
                    print("[Error] No episode directory found while collecting; forcing failure.")
                    controller.end_episode_failure(robot._hand)
                    audio_player.stop()
                    break

                leader_cmd = teleop.update()
                follower_cmd = np.zeros(16, dtype=float)
                finger_indices = [0, 1, 2, 3, 12, 13, 14, 15]
                for i, idx in enumerate(finger_indices):
                    follower_cmd[idx] = leader_cmd[i]

                ts_next = controller.increment_timestep()
                if ts_next > MAX_TIMESTEP:
                    print(f"[Info] MAX_TIMESTEP ({MAX_TIMESTEP}) exceeded. Auto-success.")
                    controller.end_episode_success(robot._hand)
                    audio_player.stop()
                    break

                node_exec.sleep()

                obs = robot.set_joint_command(follower_cmd)
                file_path = os.path.join(episode_dir, f"timestep{ts}.pkl")
                try:
                    with open(file_path, "wb") as f:
                        pickle.dump(flatten_dict(obs), f)
                except Exception as e:
                    print(f"[Error] Failed to save timestep {ts}: {e}")
                    controller.end_episode_failure(robot._hand)
                    audio_player.stop()
                    break

            continue

    except KeyboardInterrupt:
        print("\n[Main] Caught KeyboardInterrupt. Cleaning up...")

    finally:
        audio_player.stop()
        keyboard_handler.stop()
        node_exec.spin_thread_finish()
        try:
            teleop.driver.set_torque_mode(False)
            teleop.driver.close()
        except Exception:
            pass
        print("[Info] Program terminated.")


if __name__ == "__main__":
    main()
