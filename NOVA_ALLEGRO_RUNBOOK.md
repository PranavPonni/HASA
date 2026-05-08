# Nova Glove -> Allegro Hand (ROS2 Foxy + ROS1)

This mirrors the Manus/Haptikos flow used in this repository.

## 1) One-time setup in HASA root

```bash
cd ~/HASA
bash setup_nova_foxy.sh
```

Default workspace path is:

`~/HASA/senseglove_ros2_ws`

## 2) Runtime terminals

### Terminal 1: ROS1 master

```bash
source_hasa
roscore
```

### Terminal 2: Allegro controller launch

```bash
source_hasa
cd ~/HASA/user/pranav/example/launcher
python3 allegro_launch.py --config ./allegro_config/config.yaml
```

### Terminal 3: CAN setup (if needed)

```bash
source_hasa
cd ~/HASA/user/pranav/example/launcher
sudo sh can_up.sh
```

### Terminal 4: SenseGlove bringup (ROS2 Foxy)

```bash
source /opt/ros/foxy/setup.bash
source ~/HASA/senseglove_ros2_ws/install/setup.bash
ros2 launch senseglove_bringup senseglove.launch.py run_rviz:=false run_sensecom:=true run_finger_distance:=false
```

### Terminal 5: ROS1 <-> ROS2 bridge

```bash
source /opt/ros/noetic/setup.bash
source /opt/ros/foxy/setup.bash
export ROS_MASTER_URI=http://localhost:11311
ros2 run ros1_bridge dynamic_bridge
```

### Terminal 6: Nova to Allegro command bridge

```bash
source /opt/ros/foxy/setup.bash
source ~/HASA/senseglove_ros2_ws/install/setup.bash
cd ~/HASA/user/pranav/example/script
python3 novaallegro.py --serial 01001 --side lh
```

## 3) Notes

- `--serial` and `--side` must match your SenseGlove launch config.
- Default Nova state topic is `/senseglove/glove{SERIAL}/{lh|rh}/senseglove_states`.
- If your topic differs, pass `--in-topic` explicitly.
- If the hand moves incorrectly, force units with `--unit-mode rad` or `--unit-mode deg`.
