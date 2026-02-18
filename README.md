
# 🖐️ HASA Usage Guide

This guide helps you set up and run the **HASA** environment for controlling the Allegro Hand and Tactile Sensors using ROS.

---

## 1. 🛠️ ROS Workspace Setup

Navigate to your `ros_ws` directory and build the ROS packages:

```bash
cd ~/ros_ws
catkin build
source_hasa
````

> ✅ **Always work inside** `user/{your_name}` directory.
> 🚫 **Do NOT** modify anything inside the `example/` directory.

---

## 2. 🤖 Launch Allegro Hand

**Terminal 1** – Start the Allegro Hand controller:

```bash
cd launcher
python3 allegro_launch.py --config ./allegro_config/config.yaml
```

**Terminal 2** – Run the motion script:

```bash
cd script
python3 example_motion.py
```

---

## 3. 🧤 Launch Tactile Sensor

**Terminal 3** – Start the tactile sensor:

```bash
cd launcher
python3 tactile_launch.py --config ./tactile_config/config.yaml
```

**Terminal 4** – Run the tactile script:

```bash
cd script
python3 example_tactile.py
```

---

## 4. 🧪 Sample Code

### ✋ Allegro Hand Control Example

```python
from py_node_exec import NodeExec
from allegro_package import AllegroHand
import numpy as np

if __name__ == "__main__":
    node = NodeExec(freq=5)
    node.spin_thread_start()
    
    robot = AllegroHand(hand_topic_prefix="allegroHand_0", ctrl_freq=5)

    while node.ok():
        obs = robot.get_obs()
        robot.set_joint_cmd(obs["jnt_pos"])
        node.sleep()
        
    node.spin_thread_finish()
```

This example continuously reads joint positions from the hand and sends them back as commands, effectively holding the current pose.

---

### 🧠 Tactile Sensor Subscriber Example

```python
from py_node_exec import NodeExec
from xela_py import TactileSubscriber
import time
import numpy as np

if __name__ == "__main__":
    node = NodeExec(freq=5)
    node.spin_thread_start()
    index = TactileSubscriber(topic_prefix="index_tip")

    while node.ok():
        print(index.get_obs())
        node.sleep()
        
    node.spin_thread_finish()
```

This script subscribes to the tactile data of the **index finger tip** and prints the sensor readings at 5 Hz.

---

## 5. 💻 Configuring CAN Bus
**Open random terminal** – UP all CAN buses:

```bash
cd HASA/example/launcher
ip link show
sudo sh can_up.sh
```
If it doesn't work manually up:
```bash
sudo ip link set up can0 type can bitrate 1000000
sudo ip link set up can1 type can bitrate 1000000
sudo ip link set up can2 type can bitrate 1000000
sudo ip link set up can3 type can bitrate 1000000
sudo ip link set up can0
sudo ip link set up can1
sudo ip link set up can2
sudo ip link set up can3
```
---
## 6. Connecting Manus Glove via GeoRT to Allegro Hand

**Open terminal 1** 

```bash

source_hasa
roscore

```

**Open terminal 2** 

```bash

source_hasa
cd user/pranav/example/launcher
python3 allegro_launch.py --config ./allegro_config/config.yaml

```

**Open terminal 3** 

```bash

source /opt/ros/foxy/setup.bash
source ~/HASA/manus_ros2_ws/install/setup.bash
ros2 run manus_ros2 manus_data_publisher

```

**Open terminal 4** 

```bash

source /opt/ros/noetic/setup.bash
source /opt/ros/foxy/setup.bash
export ROS_MASTER_URI=http://localhost:11311
ros2 run ros1_bridge dynamic_bridge

```
**Open terminal 5** 

```bash

source /opt/ros/foxy/setup.bash
source ~/HASA/manus_ros2_ws/install/setup.bash
cd user/pranav/example/GeoRT/
python3 manus_allegro_control.py -ckpt_tag geort_1 -hand allegro_left

```
