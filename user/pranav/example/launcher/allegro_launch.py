import os
import sys
import signal
import subprocess
import argparse
import yaml
import termios

def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch allegro_hand.launch instances based on a YAML config."
    )
    parser.add_argument(
        "-c", "--config", default="./allegro_config/config.yaml",
        help="YAML configuration file (default: ./allegro_config/config.yaml)"
    )
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def direction(value, field, hand_name):
    value = int(value)
    if value not in (-1, 1):
        raise ValueError(
            f"'{hand_name}.{field}' must be either -1 or 1, got {value}"
        )
    return value

def controller_name(value, hand_name):
    supported = {"grasp", "pd", "velsat", "torque"}
    value = str(value).lower()
    if value not in supported:
        raise ValueError(
            f"'{hand_name}.controller' must be one of "
            f"{sorted(supported)}, got {value!r}"
        )
    return value

def main():
    args = parse_args()
    cfg = load_config(args.config)

    stdin_fd = sys.stdin.fileno()
    orig_attrs = termios.tcgetattr(stdin_fd)

    def restore_terminal():
        termios.tcsetattr(stdin_fd, termios.TCSANOW, orig_attrs)

    procs = []
    def shutdown(sig, frame):
        print("\nShutting down all launches...")
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception as e:
                print(f"  error killing pid {p.pid}: {e}")
        restore_terminal()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    for name, params in cfg.items():
        hand = params.get("hand_type")
        rviz = params.get("rviz")
        slidebar = params.get("slidebar")
        controller = controller_name(params.get("controller", "pd"), name)
        joint_11_position_direction = direction(
            params.get("joint_11_position_direction", 1),
            "joint_11_position_direction",
            name,
        )
        joint_11_motor_direction = direction(
            params.get("joint_11_motor_direction", 1),
            "joint_11_motor_direction",
            name,
        )
        joint_9_position_offset = float(
            params.get("joint_9_position_offset", 0.0)
        )
        joint_10_position_offset = float(
            params.get("joint_10_position_offset", 0.0)
        )
        joint_11_position_offset = float(
            params.get("joint_11_position_offset", 0.0)
        )
        ring_flexion_zero_on_start = bool(
            params.get("ring_flexion_zero_on_start", False)
        )
        joint_9_zero_on_start = bool(
            params.get(
                "joint_9_zero_on_start", ring_flexion_zero_on_start
            )
        )
        joint_10_zero_on_start = bool(
            params.get(
                "joint_10_zero_on_start", ring_flexion_zero_on_start
            )
        )
        joint_11_zero_on_start = bool(
            params.get(
                "joint_11_zero_on_start", ring_flexion_zero_on_start
            )
        )
        if not hand:
            print(f"Skipping '{name}': no 'hand_type' specified.")
            continue

        cmd = [
            "roslaunch", "allegro_hand_controllers", "allegro_hand.launch",
            f"HAND:={hand}", f"VISUALIZE:={rviz}", f"JSP_GUI:={slidebar}",
            f"CONTROLLER:={controller}",
            f"JOINT_11_POSITION_DIRECTION:={joint_11_position_direction}",
            f"JOINT_11_MOTOR_DIRECTION:={joint_11_motor_direction}",
            f"JOINT_9_POSITION_OFFSET:={joint_9_position_offset}",
            f"JOINT_10_POSITION_OFFSET:={joint_10_position_offset}",
            f"JOINT_11_POSITION_OFFSET:={joint_11_position_offset}",
            "JOINT_9_ZERO_ON_START:="
            f"{str(joint_9_zero_on_start).lower()}",
            "JOINT_10_ZERO_ON_START:="
            f"{str(joint_10_zero_on_start).lower()}",
            "JOINT_11_ZERO_ON_START:="
            f"{str(joint_11_zero_on_start).lower()}",
        ]
        print(f"Launching '{name}' (HAND={hand}): {' '.join(cmd)}")
        p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        procs.append(p)

    try:
        for p in procs:
            p.wait()
    finally:
        restore_terminal()

if __name__ == "__main__":
    main()
