import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline

class Interpolation:
    def __init__(self, method, target):
        self.method = method
        self.target = target

    def set_interpolation_config(self, policy_freq, control_freq, ramp_ratio):
        self.policy_freq = policy_freq
        self.control_freq = control_freq
        self.ramp_ratio = ramp_ratio

    def get_interpolate(self, start, goal):
        if self.method == "linear":
            interpolated_values = self.linear_interpolation(start, goal)
        elif self.method == "spline":
            interpolated_values = self.spline_interpolation(start, goal)
        else:
            raise ValueError("Unsupported interpolation method")

        return interpolated_values

    def linear_interpolation(self, start, goal):
        timestep_count = max([int(self.ramp_ratio * self.control_freq / self.policy_freq),1])

        if self.target == "normal":
            # Perform linear interpolation for normal target
            interpolated_values = []
            for i in range(timestep_count):
                alpha = (i+1) / timestep_count
                interpolated_value = start + alpha * (goal - start)
                interpolated_values.append(interpolated_value)
            return np.array(interpolated_values)
        elif self.target == "rotation":
            # Perform linear interpolation for rotation target
            # Initialize Slerp object
            slerp = Slerp([0,1], Rotation.from_quat([start, goal]))
            # Interpolate rotations using Slerp
            alphas = np.linspace(0, 1, timestep_count)
            interpolated_rotations = slerp(alphas).as_quat()
            return interpolated_rotations
        else:
            raise ValueError("Unsupported target")

    def spline_interpolation(self, start, goal):
        # Spline interpolation implementation
        timestep_count = max([int(self.ramp_ratio * self.control_freq / self.policy_freq),1])

        if self.target == "normal":
            # Perform spline interpolation for normal target
            cs = CubicSpline([0, 1], [start, goal])
            alphas = np.linspace(0, 1, timestep_count)
            interpolated_values = cs(alphas)
            return interpolated_values
        elif self.target == "rotation":
            # Perform spline interpolation for rotation target
            # Initialize Slerp object
            slerp = Slerp([0,1], Rotation.from_quat([start, goal]))
            # Interpolate rotations using Slerp
            alphas = np.linspace(0, 1, timestep_count)
            interpolated_rotations = slerp(alphas).as_quat()
            return interpolated_rotations
        else:
            raise ValueError("Unsupported target")

def main():
    print("Method for interpolating in several methods")

if __name__=="__main__":
    main()
