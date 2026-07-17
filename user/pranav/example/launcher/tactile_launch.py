#!/usr/bin/env python3
import os
import sys
import signal
import subprocess
import argparse
import yaml
import time
import shlex

NOETIC_SETUP = "/opt/ros/noetic/setup.bash"
WORKSPACE_SETUP = "/home/handlingteam2/HASA/ros_ws/devel/setup.bash"
ROS2_PATH_MARKERS = ("/opt/ros/foxy", "/home/handlingteam2/HASA/manus_ros2_ws/install")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch multiple tactile_publisher.launch instances based on a YAML config."
    )
    parser.add_argument(
        "-c", "--config", default="./tactile_config/config.yaml",
        help="YAML configuration file (default: tips.yaml)"
    )
    return parser.parse_args()

def load_tips(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def roslaunch_command(args):
    setup_commands = [f"source {shlex.quote(NOETIC_SETUP)}"]
    if os.path.exists(WORKSPACE_SETUP):
        setup_commands.append(f"source {shlex.quote(WORKSPACE_SETUP)}")

    launch_command = " ".join(shlex.quote(arg) for arg in args)
    return ["bash", "-lc", " && ".join(setup_commands + [f"exec {launch_command}"])]

def roslaunch_env():
    env = os.environ.copy()
    for key in (
        "ROS_DISTRO",
        "ROS_ETC_DIR",
        "ROS_PACKAGE_PATH",
        "ROS_PYTHON_VERSION",
        "ROS_ROOT",
        "ROS_VERSION",
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
    ):
        env.pop(key, None)

    for key, value in list(env.items()):
        if os.pathsep not in value:
            continue
        parts = [
            part for part in value.split(os.pathsep)
            if not any(marker in part for marker in ROS2_PATH_MARKERS)
        ]
        env[key] = os.pathsep.join(parts)
    return env

def main():
    args = parse_args()
    tips = load_tips(args.config)

    procs = []

    def shutdown(sig, frame):
        print("\nShutting down all launches...")
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception as e:
                print(f"  error killing pid {p.pid}: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    for ns, cfg in tips.items():
        if not cfg.get("enable", False):
            continue

        file_arg = cfg["config"]
        port_arg = cfg["port"]
        ip_arg   = cfg["ip"]

        cmd = [
            "roslaunch", "xela_server_ros", "tactile_publisher.launch",
            f"file:={file_arg}",
            f"port:={port_arg}",
            f"ip:={ip_arg}",
            f"ns:={ns}"
        ]
        print(f"Launching namespace `{ns}`: {' '.join(cmd)}")
        p = subprocess.Popen(roslaunch_command(cmd), preexec_fn=os.setsid, env=roslaunch_env())
        time.sleep(3)
        procs.append(p)

    for p in procs:
        p.wait()

if __name__ == "__main__":
    main()
