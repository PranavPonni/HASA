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
        if not hand:
            print(f"Skipping '{name}': no 'hand_type' specified.")
            continue

        cmd = [
            "roslaunch", "allegro_hand_controllers", "allegro_hand.launch",
            f"HAND:={hand}"
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
