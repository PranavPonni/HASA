#!/usr/bin/env python3
import sys, time, traceback

print("=== BRIDGE_DEBUG starting ===", flush=True)
print("python:", sys.executable, flush=True)

try:
    import rospy
    print("import rospy: OK", flush=True)
except Exception as e:
    print("import rospy: FAIL", e, flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    import zmq
    print("import zmq: OK", flush=True)
except Exception as e:
    print("import zmq: FAIL", e, flush=True)
    traceback.print_exc()
    sys.exit(1)

import numpy as np
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

# IMPORTANT: using LEFT topics (based on your rosnode list)
ALLEGRO_CMD_TOPIC  = "/allegroHand_left_0/joint_cmd"
ALLEGRO_STATE_TOPIC = "/allegroHand_0/joint_states"
MANUS_TOPIC = "/manus/keypoints"

MAX_STEP = 0.05

def main():
    rospy.init_node("hasa_geort_bridge_debug", anonymous=False, disable_signals=True)
    print("ROS_MASTER_URI:", rospy.get_param("/roslaunch/uris", "unknown"), flush=True)
    print("Node name:", rospy.get_name(), flush=True)

    # ZMQ client -> GeoRT server
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://127.0.0.1:5555")
    print("ZMQ connected to tcp://127.0.0.1:5555", flush=True)

    print(f"Waiting for Allegro joint_states: {ALLEGRO_STATE_TOPIC}", flush=True)
    try:
        js = rospy.wait_for_message(ALLEGRO_STATE_TOPIC, JointState, timeout=10.0)
    except Exception as e:
        print("FAILED waiting for joint_states (10s timeout).", flush=True)
        print("Error:", repr(e), flush=True)
        sys.exit(2)

    joint_names = list(js.name)
    print("Got joint_names len =", len(joint_names), flush=True)
    print("First few names:", joint_names[:5], flush=True)

    pub = rospy.Publisher(ALLEGRO_CMD_TOPIC, JointState, queue_size=1)
    last_cmd = None

    def cb(msg):
        nonlocal last_cmd
        flat = np.array(msg.data, dtype=np.float32)
        print(f"[CB] got keypoints floats={flat.size}", flush=True)
        if flat.size % 3 != 0:
            print("[CB] keypoints not multiple of 3 -> ignoring", flush=True)
            return

        # Request to GeoRT server
        try:
            sock.send_json({"keypoints": flat.tolist()})
            out = sock.recv_json()
        except Exception as e:
            print("[CB] ZMQ request failed:", repr(e), flush=True)
            return

        q = np.array(out.get("qpos", []), dtype=np.float32)
        print(f"[CB] GeoRT returned joints={q.size}", flush=True)

        if q.size != len(joint_names):
            print(f"[CB] size mismatch: got {q.size}, expected {len(joint_names)}", flush=True)
            return

        # Slew-rate limiting
        if last_cmd is None:
            q_cmd = q
        else:
            dq = np.clip(q - last_cmd, -MAX_STEP, MAX_STEP)
            q_cmd = last_cmd + dq
        last_cmd = q_cmd

        cmd = JointState()
        cmd.header.stamp = rospy.Time.now()
        cmd.name = joint_names
        cmd.position = q_cmd.tolist()
        pub.publish(cmd)
        print("[CB] published JointState cmd", flush=True)

    sub = rospy.Subscriber(MANUS_TOPIC, Float32MultiArray, cb, queue_size=1)
    print(f"Subscribed to {MANUS_TOPIC}. Waiting for keypoints...", flush=True)

    # Spin manually so Ctrl+C still works even with disable_signals=True
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt -> exiting", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(99)
PY

chmod +x /home/handlingteam2/HASA/user/pranav/example/script/bridge_debug.py
