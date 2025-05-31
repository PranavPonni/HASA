from py_node_exec import NodeExec
from allegro_package import AllegroHand
from xela_py import TactileSubscriber
import numpy as np


class XelAllegro:
    def __init__(self,
                 ctrl_freq: float = 20.0,
                 hand_topic_prefix: str = "allegroHand_0",
                 tactile_topic_prefix: list = ["index_tip", "thumb_tip"]):
        self._hand = AllegroHand(
            hand_topic_prefix=hand_topic_prefix,
            ctrl_freq=ctrl_freq
        )
        self._tactile_sensors = {}
        for prefix in tactile_topic_prefix:
            self._tactile_sensors[prefix] = TactileSubscriber(topic_prefix=prefix)

    def set_jnt_cmd(self, cmd: np.ndarray):

        if not isinstance(cmd, np.ndarray):
            raise TypeError("cmd must be a numpy.ndarray.")
        if cmd.shape != (16,):
            raise ValueError("cmd must have shape (16,).")
        self._hand.set_joint_cmd(cmd)

    def get_obs(self) -> dict:
        tactile_obs = {}
        for prefix, subscriber in self._tactile_sensors.items():
            tactile_obs[prefix] = subscriber.get_obs()
        hand_obs = self._hand.get_obs()
        return {
            "tactile": tactile_obs,
            "hand": hand_obs
        }


def main():
    CTRL_FREQ = 20.0

    node_exec = NodeExec(node_name="data_collection", freq=CTRL_FREQ)
    node_exec.spin_thread_start()

    robot = XelAllegro(
        ctrl_freq=CTRL_FREQ,
        hand_topic_prefix="allegroHand_0",
        tactile_topic_prefix=["index_tip"]
    )

    try:
        while node_exec.ok():
            obs = robot.get_obs()
            tactile_data = obs["tactile"] 
            hand_data = obs["hand"]        
            current_jnt_pos = hand_data["jnt_pos"]
            robot.set_jnt_cmd(current_jnt_pos)
            node_exec.sleep()

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")

    finally:
        # Ensure the node is properly shut down
        node_exec.spin_thread_finish()


if __name__ == "__main__":
    main()
