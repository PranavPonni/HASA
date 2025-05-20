import os
import argparse
import time
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import pink
from pink.visualization import start_meshcat_visualizer

class AllegroVis:
    def __init__(self, side: str):
        assert side in ("left", "right"), "side must be 'left' or 'right'"
        self.side = side
        urdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"urdf")

        self.urdf_path = os.path.join(
            urdf_dir, f"allegro_hand_description_{side}.urdf"
        )
        mesh_dir = os.path.join(urdf_dir, "meshes")
        self.robot = RobotWrapper.BuildFromURDF(
            self.urdf_path,
            [urdf_dir, mesh_dir],
        )
        self.q0 = self.robot.q0.copy()

        self.viz = start_meshcat_visualizer(self.robot)
        self.configuration = pink.Configuration(
            self.robot.model,
            self.robot.data,
            self.robot.q0
        )

    def set_pos(self, q: np.ndarray):
        if not isinstance(q, np.ndarray):
            q = np.array(q)
        self.configuration.q = q.copy()

    def update(self):
        self.viz.display(self.configuration.q)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Allegro hand motion.")
    parser.add_argument(
        "--side", choices=["left", "right"], default="left",
        help="Select which hand to visualize (default: left)."
    )
    args = parser.parse_args()

    vis = AllegroVis(args.side)
    q0 = vis.q0
    vis.set_pos(q0)

    rate_hz = 30.0  # updates per second
    dt = 1.0 / rate_hz
    t = 0.0

    while True:
        q = q0.copy()
        amplitude = 0.5
        q[7 + 1] = amplitude * (1.0 + np.sin(t))  # index finger flex
        q[7 + 4] = amplitude * (1.0 - np.sin(t))  # thumb flex

        vis.set_pos(q)
        vis.update()

        time.sleep(dt)
        t += dt
