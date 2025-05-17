import time
import numpy as np
import copy
from allegro_leader import AllegroPrecesionGrasp, AllegroVis


class TeleopRunner:
    def __init__(self, side="left", wall_kp=0., rate_hz=50):
        self.teleop = AllegroPrecesionGrasp(wall_kp=wall_kp)
        self.vis = AllegroVis(side)
        self.q = copy.deepcopy(self.vis.q0)
        self.dt = 1.0 / rate_hz
        self.finger_idx = [0, 1, 2, 3, 4, 5, 6, 7]


    def run(self):
        try:
            while True:
                q_cmd = self.teleop.update() 
                for i, idx in enumerate(self.finger_idx):
                    self.q[idx] = q_cmd[i]
                print(self.q.shape)
                self.vis.set_pos(self.q)
                self.vis.update()
                time.sleep(self.dt)
        except KeyboardInterrupt:
            print("\nShutting down teleop.")
            self.teleop.driver.set_torque_mode(False)
            self.teleop.driver.close()

if __name__ == "__main__":
    runner = TeleopRunner()
    runner.run()
