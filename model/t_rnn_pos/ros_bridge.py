from model.ros_bridge_utils import ensure_ros_workspace_paths, ensure_rospy_node
ensure_ros_workspace_paths()

from allegro_package import AllegroHand
from xela_py import TactileSubscriber
from sensor_msgs.msg import JointState
import numpy as np
import time
import sys
import os
import torch
import rospy
from threading import Lock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_preproc as dp

# Joint indices used by the model (all Allegro fingers)
_JOINT_IDX = list(range(16))

# Ready pose: full 16-DOF start configuration for rugby rotation task
# Sourced from episode0/timestep0 of training data
_READY_POSE = np.array([
    -0.1495,  0.1958,  0.0162,  0.4974,   # index finger
    -0.1159,  0.3758,  0.0091,  0.1932,   # middle finger
    -0.4109,  0.4392,  0.0090,  0.1292,   # ring finger
     1.5035,  0.0969,  0.0438,  0.6246,   # thumb
], dtype=np.float32)

# Home pose: open/flat hand posture used at motion start.
_HOME_POSE = np.zeros(16, dtype=np.float32)


class XelAllegro:
    def __init__(self,
                 ctrl_freq: float = 20.0,
                 hand_topic_prefix: str = "allegroHand_0",
                 tactile_topic_prefix: list = ["index_tip", "middle_tip", "ring_tip", "thumb_tip"]):
        ensure_rospy_node()
        self._hand = AllegroHand(
            hand_topic_prefix=hand_topic_prefix,
            ctrl_freq=ctrl_freq
        )
        self._ctrl_freq = ctrl_freq
        self._tactile_sensors = {}
        self._direct_effort_mode = False
        self._direct_effort_cmd = None
        self._direct_effort_lock = Lock()
        self._direct_effort_timer = None
        for prefix in tactile_topic_prefix:
            self._tactile_sensors[prefix] = TactileSubscriber(topic_prefix=prefix)

    def _ensure_direct_effort_mode(self):
        if self._direct_effort_mode:
            return
        if hasattr(self._hand, "timer"):
            self._hand.timer.shutdown()
        low_level_freq = float(getattr(self._hand, "low_level_freq", 100.0))
        self._direct_effort_timer = rospy.Timer(
            rospy.Duration(1.0 / low_level_freq),
            self._publish_direct_effort_cmd,
        )
        self._direct_effort_mode = True
        print("[torque] direct JointState.effort command publisher enabled")

    def _publish_direct_effort_cmd(self, event=None):
        del event
        with self._direct_effort_lock:
            effort = None if self._direct_effort_cmd is None else self._direct_effort_cmd.copy()
        with self._hand.lock:
            if self._hand.interpolated_jnt_cmds is not None:
                position = self._hand.interpolated_jnt_cmds[0].copy()
                if self._hand.interpolated_jnt_cmds.shape[0] > 1:
                    self._hand.interpolated_jnt_cmds = self._hand.interpolated_jnt_cmds[1:].copy()
            else:
                position = self._hand._jnt_cmd_pos
                if position is None:
                    position = self._hand._jnt_pos
                if position is None:
                    return
                position = np.asarray(position, dtype=np.float32).copy()
        msg = JointState()
        msg.position = list(position)
        if effort is not None:
            msg.effort = list(effort)
        self._hand.pub_joint_cmd.publish(msg)

    def set_jnt_cmd(self, cmd: np.ndarray, effort: np.ndarray = None):
        if not isinstance(cmd, np.ndarray):
            raise TypeError("cmd must be a numpy.ndarray.")
        if cmd.shape != (16,):
            raise ValueError("cmd must have shape (16,).")
        if effort is None and not self._direct_effort_mode:
            self._hand.set_joint_cmd(cmd)
            return
        if effort is None:
            effort = np.zeros(16, dtype=np.float32)
        if not isinstance(effort, np.ndarray):
            raise TypeError("effort must be a numpy.ndarray.")
        if effort.shape != (16,):
            raise ValueError("effort must have shape (16,).")
        self._ensure_direct_effort_mode()
        with self._direct_effort_lock:
            self._direct_effort_cmd = effort.astype(np.float32, copy=True)
        self._hand.set_joint_cmd(cmd)
        self._publish_direct_effort_cmd()

    def get_cmd_connection_count(self) -> int:
        return int(self._hand.pub_joint_cmd.get_num_connections())

    def torque_on(self):
        self._hand.torque_on()

    def torque_off(self):
        self._hand.torque_off()

    def wait_for_cmd_connection(self, timeout_sec: float = 5.0, poll_sec: float = 0.1) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.get_cmd_connection_count() > 0:
                return True
            time.sleep(poll_sec)
        return self.get_cmd_connection_count() > 0

    def get_obs(self) -> dict:
        tactile_obs = {}
        for prefix, subscriber in self._tactile_sensors.items():
            tactile_obs[prefix] = subscriber.get_obs()
        hand_obs = self._hand.get_obs()
        return self.flatten_dict({
            "tactile": tactile_obs,
            "hand": hand_obs
        })

    def flatten_dict(self, d, parent_key='', sep='_'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    def move_to_initial(self):
        print("Moving to home pose (open/flat hand)...")
        for _ in range(10):
            self.set_jnt_cmd(_HOME_POSE)
            time.sleep(1)

    def move_to_ready_pose(self, duration: float = 3.0):
        """Smoothly interpolate from current position to the rugby ready pose."""
        current = np.array(self._hand.get_obs()["jnt_pos"], dtype=np.float32)
        n_steps = int(duration * self._ctrl_freq)
        dt = 1.0 / self._ctrl_freq
        print("Moving to ready pose...")
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            cmd = (1.0 - alpha) * current + alpha * _READY_POSE
            self.set_jnt_cmd(cmd)
            time.sleep(dt)
        print("Ready pose reached.")

    def wait_for_start(self, wait_sec: float = 5.0, require_enter: bool = False):
        """Hold ready pose so the object can be placed before motion starts."""
        print("\n--- T-RNN: Rugby rotation ready ---")
        print("Place the rugby ball in the hand.")
        deadline = time.time() + max(float(wait_sec), 0.0)
        dt = 1.0 / self._ctrl_freq
        next_print = 0
        while time.time() < deadline:
            remaining = deadline - time.time()
            whole = int(np.ceil(remaining))
            if whole != next_print:
                next_print = whole
                print(f"  starts in {whole}s")
            self.set_jnt_cmd(_READY_POSE)
            time.sleep(min(dt, max(remaining, 0.0)))
        if require_enter:
            print("Press Enter to start motion generation.")
            self.set_jnt_cmd(_READY_POSE)
            input()
        self._start_wait_done = True
        print("Starting motion generation.")

    def run_motion(
        self,
        model,
        scaling_param: dict,
        dataset_param: dict,
        n_steps: int = 400,
        device=None,
    ):
        """
        Run the T-RNN motion generation loop for n_steps timesteps.

        Args:
            model:          Loaded ExternalTouchRNN (t_rnn_pos), already in eval mode.
            scaling_param:  dict loaded from the sweep's scaling_param.pkl.
            dataset_param:  dict containing the 'modality' normalization config.
            n_steps:        Number of control steps (400 matches rugby episode length).
            device:         torch.device to run inference on.
        """
        if device is None:
            device = next(model.parameters()).device
        model.eval()

        modality = dataset_param["modality"]
        dt = 1.0 / self._ctrl_freq

        hidden = None

        # Hold ready pose while warming up
        for _ in range(3):
            self.set_jnt_cmd(_READY_POSE)
            time.sleep(dt)
        if not getattr(self, "_start_wait_done", False):
            self.wait_for_start(
                wait_sec=float(dataset_param.get("object_place_wait_sec", 5.0)),
                require_enter=bool(dataset_param.get("wait_for_enter_before_motion", False)),
            )

        print(f"Running {n_steps} steps at {self._ctrl_freq} Hz...")
        t_start = time.time()

        for step in range(n_steps):
            step_start = time.time()

            # ---- 1. Get sensor observations ----
            obs = self.get_obs()

            # Tactile: sensor returns (30, 3) per finger; flatten to (90,)
            tac_idx_raw  = np.asarray(obs["tactile_index_tip"], dtype=np.float32).reshape(-1)
            tac_thb_raw  = np.asarray(obs["tactile_thumb_tip"], dtype=np.float32).reshape(-1)
            pos_raw_full = np.asarray(obs["hand_jnt_pos"],      dtype=np.float32)   # (16,)

            pos_raw = pos_raw_full[_JOINT_IDX]   # (16,)

            # ---- 2. Normalise ----
            tac_idx_n = dp.scaling_data(tac_idx_raw, scaling_param["tactile_index_tip"], modality["tactile_index_tip"])
            tac_thb_n = dp.scaling_data(tac_thb_raw, scaling_param["tactile_thumb_tip"],  modality["tactile_thumb_tip"])
            pos_n     = dp.scaling_data(pos_raw,     scaling_param["hand_jnt_pos"],        modality["hand_jnt_pos"])

            # ---- 3. To tensor (batch=1) ----
            def _t(x): return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            tac_idx_t = _t(tac_idx_n)
            tac_thb_t = _t(tac_thb_n)
            pos_t     = _t(pos_n)

            # ---- 4. Model forward ----
            with torch.no_grad():
                (pos_pred_n, _vel_pred, idx_ext, thb_ext), hidden = model.forward(
                    tac_idx_t, tac_thb_t, pos_t, hidden
                )

            # ---- 5. Unnormalise predicted joint positions ----
            pos_pred_np = pos_pred_n.squeeze(0).cpu().numpy()      # (16,)
            pos_cmd = dp.unscale_data(pos_pred_np,
                                      scaling_param["hand_jnt_pos"],
                                      modality["hand_jnt_pos"])  # (16,) raw rad

            # ---- 6. Expand to 16-DOF command ----
            cmd_16 = _READY_POSE.copy()
            cmd_16[_JOINT_IDX] = pos_cmd

            # ---- 7. Send command ----
            self.set_jnt_cmd(cmd_16.astype(np.float32))

            # ---- 8. Precise sleep ----
            elapsed = time.time() - step_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        total = time.time() - t_start
        print(f"Motion complete. {n_steps} steps in {total:.2f}s "
              f"(avg {n_steps/total:.1f} Hz)")
