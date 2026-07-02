from py_node_exec import NodeExec
from allegro_package import AllegroHand
from xela_py import TactileSubscriber
import numpy as np
import time

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
        return self.flatten_dict({
            "tactile": tactile_obs,
            "hand": hand_obs
        })
    
    def flatten_dict(self,d, parent_key='', sep='_'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
    
    def move_to_initial(self):
        INITIAL_POS=np.array([0.17333983,0.70563116,0.49700978,1.41279631,0.,0.,0.,0.,0.,0.,0.,0.,1.46188369,-0.07056312,0.12885439,1.70578664])

        for _ in range(5):
            self.set_jnt_cmd(INITIAL_POS)
            time.sleep(1)