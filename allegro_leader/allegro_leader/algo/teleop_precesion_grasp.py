import numpy as np
from threading import Lock
import copy
from allegro_leader.teleop_base import AlgoTeleop
from allegro_leader.dynamixel.driver import DynamixelDriver

class AllegroPrecesionGrasp(AlgoTeleop):
    def __init__(self, wall_kp = 0.0):
        super().__init__()
        ids = [0, 1, 2, 3, 12, 13, 14, 15]
        types = ["XL330_M077_T"] * len(ids)
        self.driver = DynamixelDriver(ids, types)
        self.driver.set_torque_mode(False)
        self.directions = np.array([1.,1.,-1.,-1.,-1.,-1.,1.,1.])
        self.offsets = np.array([0.,-np.pi,-np.pi,-np.pi,-np.pi,-np.pi/2,-np.pi,-np.pi,])
        self.joint_limits = np.tile(
            np.array([-np.pi/2, np.pi/2], dtype=float),
            (len(ids), 1)
        )
        self.wall_kp = wall_kp
        self.leader_state = None
        self._lock = Lock()
    
    def set_follower_state(self, follower_state):
        pass

    def set_leader_state(self, leader_state):
        leader_state = np.array(leader_state, dtype=float)
        self.leader_state = leader_state * self.directions + self.offsets

    def calc_follower(self):
        if self.leader_state is None:
            raise RuntimeError("Leader state not set")
        return copy.deepcopy(self.leader_state)
    
    def calc_leader(self):
        return self.calc_wall_torque()

    def calc_wall_torque(self):
        if self.leader_state is None:
            raise RuntimeError("Leader state not set")
        q = self.leader_state
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        tau = np.zeros_like(q)
        idx_lo = q < lower
        tau[idx_lo] = self.wall_kp * (lower[idx_lo] - q[idx_lo])
        idx_hi = q > upper
        tau[idx_hi] = self.wall_kp * (upper[idx_hi] - q[idx_hi])
        return tau * self.directions

    def update(self):
        with self._lock:
            positions, _, _, _ = self.driver.get_states()
            self.set_leader_state(positions)
            # tau = self.calc_leader()
            # self.driver.set_torque(tau)
            return self.calc_follower()
