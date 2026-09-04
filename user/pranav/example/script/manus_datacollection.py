#!/usr/bin/env python3

import os
import glob as _glob
# Clean up stale FastDDS shared memory port files that cause SHM transport errors
for _f in _glob.glob('/dev/shm/fastrtps_port*'):
    try:
        os.remove(_f)
    except OSError:
        pass
del _glob

import shutil
import glob
import threading
import pickle
import numpy as np
import cv2
import time
import sys
import ctypes
import importlib
from pathlib import Path
from pynput import keyboard
from playsound import playsound


def _prepend_env_path(var_name, path_value):
    if not path_value:
        return
    current = os.environ.get(var_name, "")
    parts = [p for p in current.split(":") if p]
    if path_value in parts:
        return
    os.environ[var_name] = f"{path_value}:{current}" if current else path_value


def _bootstrap_local_manus_msgs_path():
    """Allow direct execution without separately exporting Manus ROS2 msg paths."""
    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        install_root = parent / "manus_ros2_ws" / "install"
        pkg_root = install_root / "manus_ros2_msgs"
        install_base = pkg_root / "lib"
        if not install_base.exists() or not install_root.exists() or not pkg_root.exists():
            continue

        for candidate in sorted(install_base.glob("python*/site-packages")):
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

        _prepend_env_path("LD_LIBRARY_PATH", str(install_base))
        _prepend_env_path("LD_LIBRARY_PATH", str(install_root / "lib"))

        for lib_path in (
            install_base / "libmanus_ros2_msgs__rosidl_generator_c.so",
            install_base / "libmanus_ros2_msgs__rosidl_typesupport_c.so",
            install_base / "libmanus_ros2_msgs__python.so",
        ):
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

        _prepend_env_path("AMENT_PREFIX_PATH", str(pkg_root))
        _prepend_env_path("AMENT_PREFIX_PATH", str(install_root))
        _prepend_env_path("CMAKE_PREFIX_PATH", str(pkg_root))
        _prepend_env_path("CMAKE_PREFIX_PATH", str(install_root))

        ros_site = "/opt/ros/foxy/lib/python3.8/site-packages"
        if os.path.isdir(ros_site) and ros_site not in sys.path:
            sys.path.append(ros_site)
        _prepend_env_path("LD_LIBRARY_PATH", "/opt/ros/foxy/lib")
        return


_bootstrap_local_manus_msgs_path()


def _ensure_geometry_msgs_type_support():
    """Ensure geometry_msgs message classes expose ROS2 type support."""
    ros_site = "/opt/ros/foxy/lib/python3.8/site-packages"
    original_sys_path = list(sys.path)
    try:
        if os.path.isdir(ros_site):
            if ros_site in sys.path:
                sys.path.remove(ros_site)
            sys.path.insert(0, ros_site)

        # Force a clean import in case a shadowed geometry_msgs was loaded earlier.
        sys.modules.pop("geometry_msgs.msg", None)
        sys.modules.pop("geometry_msgs", None)

        geometry_msg_module = importlib.import_module("geometry_msgs.msg")
        PoseMsg = getattr(geometry_msg_module, "Pose", None)
        QuaternionMsg = getattr(geometry_msg_module, "Quaternion", None)
        if PoseMsg is None or QuaternionMsg is None:
            raise ModuleNotFoundError(
                "geometry_msgs.msg is missing Pose/Quaternion. "
                f"Loaded from: {getattr(geometry_msg_module, '__file__', 'unknown')}"
            )

        pose_meta = PoseMsg.__class__
        quat_meta = QuaternionMsg.__class__
        if not hasattr(pose_meta, "_TYPE_SUPPORT") or not hasattr(pose_meta, "__import_type_support__"):
            raise RuntimeError(
                "geometry_msgs Pose class does not expose ROS2 type support. "
                f"Pose module: {PoseMsg.__module__}"
            )
        if not hasattr(quat_meta, "_TYPE_SUPPORT") or not hasattr(quat_meta, "__import_type_support__"):
            raise RuntimeError(
                "geometry_msgs Quaternion class does not expose ROS2 type support. "
                f"Quaternion module: {QuaternionMsg.__module__}"
            )

        if pose_meta._TYPE_SUPPORT is None:
            pose_meta.__import_type_support__()
        if quat_meta._TYPE_SUPPORT is None:
            quat_meta.__import_type_support__()
    finally:
        sys.path[:] = original_sys_path


_ensure_geometry_msgs_type_support()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from manus_ros2_msgs.msg import ManusGlove
from py_node_exec import NodeExec
from allegro_package import AllegroHand
from xela_py import TactileSubscriber

DATA_DIR = "/home/handlingteam2/HASA/user/pranav/example/data/tasks/newbolt"
MAX_TIMESTEP = 200
CTRL_FREQ = 10.0
HAND_TOPIC_PREFIX = "allegroHand_0"
TACTILE_TOPIC_PREFIXES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip"]
MANUS_TOPIC = "/manus_glove_0"
MANUS_SIDE_FILTER = "Left"
BACKGROUND_AUDIO_PATH = "/home/handlingteam2/HASA/user/pranav/example/script/util/beep.mp3"

MANUS_ERG_ORDER = [
    "ThumbMCPSpread",
    "ThumbMCPStretch",
    "ThumbPIPStretch",
    "ThumbDIPStretch",
    "IndexSpread",
    "IndexMCPStretch",
    "IndexPIPStretch",
    "IndexDIPStretch",
    "MiddleSpread",
    "MiddleMCPStretch",
    "MiddlePIPStretch",
    "MiddleDIPStretch",
    "RingSpread",
    "RingMCPStretch",
    "RingPIPStretch",
    "RingDIPStretch",
    "PinkySpread",
    "PinkyMCPStretch",
    "PinkyPIPStretch",
    "PinkyDIPStretch",
]


class ManusGloveSubscriber(Node):
    def __init__(self, topic_name=MANUS_TOPIC, side_filter=MANUS_SIDE_FILTER):
        super().__init__("manus_glove_data_collection_subscriber")
        self._lock = threading.Lock()
        self._latest_obs = None
        self._side_filter = side_filter.strip().lower()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._sub = self.create_subscription(ManusGlove, topic_name, self._callback, qos)
        self.get_logger().info(
            f"Subscribing Manus glove data: {topic_name}, side_filter='{side_filter}'"
        )

    def _callback(self, msg):
        if self._side_filter and msg.side.strip().lower() != self._side_filter:
            return

        erg_map = {e.type: float(e.value) for e in msg.ergonomics}
        obs = {
            "glove_id": int(msg.glove_id),
            "side": msg.side,
            "jnt_names": list(MANUS_ERG_ORDER),
            "jnt_pos": np.array([erg_map.get(name, 0.0) for name in MANUS_ERG_ORDER], dtype=float),
            "ergonomics": erg_map,
            "ergonomics_count": int(msg.ergonomics_count),
        }
        with self._lock:
            self._latest_obs = obs

    def get_obs(self):
        with self._lock:
            if self._latest_obs is None:
                return {
                    "received": False,
                    "glove_id": None,
                    "side": "",
                    "jnt_names": list(MANUS_ERG_ORDER),
                    "jnt_pos": None,
                    "ergonomics": {},
                    "ergonomics_count": 0,
                }
            return {
                "received": True,
                "glove_id": self._latest_obs["glove_id"],
                "side": self._latest_obs["side"],
                "jnt_names": list(self._latest_obs["jnt_names"]),
                "jnt_pos": self._latest_obs["jnt_pos"].copy(),
                "ergonomics": dict(self._latest_obs["ergonomics"]),
                "ergonomics_count": self._latest_obs["ergonomics_count"],
            }

class XelAllegro:
    def __init__(self, ctrl_freq: float = CTRL_FREQ, hand_topic_prefix: str = HAND_TOPIC_PREFIX,
                 tactile_topic_prefixes: list = TACTILE_TOPIC_PREFIXES,
                 manus_subscriber: ManusGloveSubscriber = None):
        self._hand = AllegroHand(hand_topic_prefix=hand_topic_prefix, ctrl_freq=ctrl_freq)
        # Initialize all tactile sensors in parallel so a slow sensor doesn't
        # block the others from subscribing.
        self._tactile_sensors = {}
        _lock = threading.Lock()

        def _init_sensor(prefix):
            sub = TactileSubscriber(topic_prefix=prefix)
            with _lock:
                self._tactile_sensors[prefix] = sub

        _threads = [
            threading.Thread(target=_init_sensor, args=(p,), daemon=True)
            for p in tactile_topic_prefixes
        ]
        for t in _threads:
            t.start()
        for t in _threads:
            t.join()
        self._manus_subscriber = manus_subscriber

    def get_observation(self) -> dict:
        tactile_obs = {prefix: sub.get_obs() for prefix, sub in self._tactile_sensors.items()}
        hand_obs = self._hand.get_obs()
        manus_obs = self._manus_subscriber.get_obs() if self._manus_subscriber else None
        return {"tactile": tactile_obs, "hand": hand_obs, "manus": manus_obs}


class VideoRecorder:
    def __init__(self, device_id=1, fps=20.0, resolution=(640, 480)):
        self.device_id = device_id
        self.fps = fps
        self.resolution = resolution
        self._recording = False
        self._writer = None
        self._cap = None
        self._thread = None

    def _record_loop(self, video_path):
        self._cap = cv2.VideoCapture(self.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.resolution)

        while self._recording and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                continue
            self._writer.write(frame)

        self._cap.release()
        self._writer.release()

    def start(self, output_path):
        self._recording = True
        self._thread = threading.Thread(target=self._record_loop, args=(output_path,), daemon=True)
        self._thread.start()
        print(f"[Info] Video recording started: {output_path}")

    def stop(self):
        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        print(f"[Info] Video recording stopped.")


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

    def end_episode_success(self):
        with self._lock:
            if not self._collecting:
                return
            print(f"[Info] Episode succeeded: data kept at '{self._episode_dir}'")
            self._collecting = False
            self._state = "idle"

    def end_episode_failure(self):
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
    def __init__(self, controller: DataCollectorController, audio_player: AudioPlayer,
                 video_recorder: VideoRecorder):
        self._controller = controller
        self._audio_player = audio_player
        self._video_recorder = video_recorder
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self.start()

    def start(self):
        self._listener.start()
        print("[Info] Keyboard listener started.")
        print("  - Idle: 's' -> start collection, 'q' -> quit")
        print("  - Collecting: 'y' -> success (keep data), 'n' -> failure (delete data)")

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
                    episode_dir = self._controller.get_episode_dir()
                    video_path = os.path.join(episode_dir, "episode_video.mp4")
                    self._video_recorder.start(video_path)
                    self._audio_player.start()
            elif char == "q":
                self._controller.request_exit()

        elif state == "collecting":
            if char == "y":
                self._controller.end_episode_success()
                self._video_recorder.stop()
                self._audio_player.stop()
            elif char == "n":
                self._controller.end_episode_failure()
                self._video_recorder.stop()
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
    node_exec = NodeExec(node_name="manus_data_collection", freq=CTRL_FREQ)
    node_exec.spin_thread_start()

    manus_subscriber = None
    manus_spin_thread = None
    if not rclpy.ok():
        rclpy.init(args=None)
    manus_subscriber = ManusGloveSubscriber(topic_name=MANUS_TOPIC, side_filter=MANUS_SIDE_FILTER)
    manus_spin_thread = threading.Thread(target=rclpy.spin, args=(manus_subscriber,), daemon=True)
    manus_spin_thread.start()

    robot = XelAllegro(ctrl_freq=CTRL_FREQ, hand_topic_prefix=HAND_TOPIC_PREFIX,
                       tactile_topic_prefixes=TACTILE_TOPIC_PREFIXES,
                       manus_subscriber=manus_subscriber)
    controller = DataCollectorController()
    audio_player = AudioPlayer(BACKGROUND_AUDIO_PATH)
    video_recorder = VideoRecorder(device_id=0)
    keyboard_handler = KeyboardHandler(controller, audio_player, video_recorder)

    print(f"[Info] MAX_TIMESTEP = {MAX_TIMESTEP}")
    print("[Info] Hand control is expected from manusallegro.py (Manus glove).")
    print(f"[Info] Tactile sensors: {TACTILE_TOPIC_PREFIXES}")
    print(f"[Info] Manus glove topic: {MANUS_TOPIC} (side filter: {MANUS_SIDE_FILTER})")

    # === Check tactile sensors are working before data collection ===
    print("\n[Info] Checking tactile sensors...")
    node_exec.sleep()
    time.sleep(1.0)  # allow sensors to settle

    obs = robot.get_observation()
    for sensor_name, sensor_data in obs["tactile"].items():
        if sensor_data is not None:
            print(f"  [{sensor_name}] OK - sensor working")
        else:
            print(f"  [{sensor_name}] WARNING: No data received!")

    if obs["manus"]["received"]:
        print("  [manus] OK - glove joint positions received")
    else:
        print("  [manus] WARNING: No Manus glove data received yet!")

    # Record tactile offset at current (resting) position
    tactile_offset = obs["tactile"]
    valid_offset = all(v is not None for v in tactile_offset.values())
    if not valid_offset:
        print("[Warning] Some tactile sensors returned None. Offset subtraction may be incomplete.")

    # Verify offset subtraction
    zeroed_test = robot.get_observation()["tactile"]
    print("[Debug] Verifying offset subtraction:")
    for k in zeroed_test:
        if zeroed_test[k] is not None and tactile_offset.get(k) is not None:
            diff = zeroed_test[k] - tactile_offset[k]
            print(f"  {k}: mean={np.mean(diff):.2f}, max={np.max(diff):.2f}, min={np.min(diff):.2f}")
            if np.max(np.abs(diff)) > 50:
                print(f"  [Warning] Offset subtraction for {k} not near zero!")
        else:
            print(f"  {k}: skipped (no data)")

    print("\n[Info] Press 's' to start recording when ready.")
    print("[Info] Make sure manusallegro.py is running for Manus glove control.\n")

    try:
        while not controller.should_exit():
            if not controller.is_collecting():
                node_exec.sleep()
                continue

            while node_exec.ok() and controller.is_collecting():
                ts = controller.get_timestep()
                episode_dir = controller.get_episode_dir()
                if episode_dir is None:
                    print("[Error] No episode directory found while collecting; forcing failure.")
                    controller.end_episode_failure()
                    video_recorder.stop()
                    audio_player.stop()
                    break

                ts_next = controller.increment_timestep()
                if ts_next > MAX_TIMESTEP:
                    print(f"[Info] MAX_TIMESTEP ({MAX_TIMESTEP}) exceeded. Auto-success.")
                    controller.end_episode_success()
                    video_recorder.stop()
                    audio_player.stop()
                    break

                node_exec.sleep()

                obs = robot.get_observation()
                # Subtract tactile offset (only for sensors with valid data)
                obs["tactile"] = {
                    k: (obs["tactile"][k] - tactile_offset[k]
                        if obs["tactile"][k] is not None and tactile_offset.get(k) is not None
                        else obs["tactile"][k])
                    for k in obs["tactile"]
                }

                file_path = os.path.join(episode_dir, f"timestep{ts}.pkl")
                try:
                    with open(file_path, "wb") as f:
                        pickle.dump(flatten_dict(obs), f)
                except Exception as e:
                    print(f"[Error] Failed to save timestep {ts}: {e}")
                    controller.end_episode_failure()
                    video_recorder.stop()
                    audio_player.stop()
                    break

    except KeyboardInterrupt:
        print("\n[Main] Caught KeyboardInterrupt. Cleaning up...")

    finally:
        audio_player.stop()
        video_recorder.stop()
        keyboard_handler.stop()
        if manus_subscriber is not None:
            manus_subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if manus_spin_thread is not None and manus_spin_thread.is_alive():
            manus_spin_thread.join(timeout=1.0)
        node_exec.spin_thread_finish()
        print("[Info] Program terminated.")


if __name__ == "__main__":
    main()
