import os
import sys


def ensure_ros_workspace_paths(workspace=None):
    workspace = workspace or os.environ.get("ROS_WORKSPACE", "/root/ros_ws")
    candidates = [
        os.path.join(workspace, "devel", "lib", "python3", "dist-packages"),
        os.path.join(workspace, "src", "allegro_package", "src"),
        os.path.join(workspace, "src", "xela_package", "xela_py", "src"),
        os.path.join(workspace, "src", "py_node_exec", "src"),
    ]
    for path in reversed(candidates):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def ensure_rospy_node(node_name="motionlearning_motion_bridge"):
    import rospy

    try:
        initialized = rospy.core.is_initialized()
    except AttributeError:
        initialized = rospy.get_node_uri() is not None
    if not initialized:
        rospy.init_node(node_name, anonymous=True, disable_signals=True)
