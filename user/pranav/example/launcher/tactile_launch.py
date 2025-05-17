#!/usr/bin/env python3
import os
import sys
import signal
import subprocess
import argparse
import yaml

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
        p = subprocess.Popen(cmd, preexec_fn=os.setsid)
        procs.append(p)

    for p in procs:
        p.wait()

if __name__ == "__main__":
    main()
