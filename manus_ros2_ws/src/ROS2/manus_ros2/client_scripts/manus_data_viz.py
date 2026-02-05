import sys
import threading
import numpy as np
import open3d as o3d
import rclpy
from loop_rate_limiters import RateLimiter
from manus_ros2_msgs.msg import ManusGlove
from rclpy.node import Node

# ---------------------------
# Behavior switches
# ---------------------------

# If True: only visualize the glove whose msg.side == "left"
# (Best if you actually have a left glove connected.)
SHOW_ONLY_LEFT_GLOVE = False

# If True: when a glove reports msg.side == "right", mirror it so it appears like a left hand
# (Pure visualization trick; useful if you only have a right glove but want it to look left.)
MIRROR_RIGHT_TO_LEFT = True

# Which axis to mirror for the "left/right" flip.
# Commonly X is left-right, but depending on your coordinate frame you may need Y instead.
# Try these in order if it looks wrong:
#   np.array([-1.0, 1.0, 1.0])  # flip X
#   np.array([ 1.0,-1.0, 1.0])  # flip Y
#   np.array([ 1.0, 1.0,-1.0])  # flip Z
MIRROR_AXIS = np.array([1.0, -1.0, 1.0], dtype=float)

# Topic indexes to subscribe to (e.g., /manus_glove_0, /manus_glove_1, ...)
GLOVE_INDEXES = [0, 1]


class GloveViz:
    """Open3D visualization for glove data"""

    def __init__(self, glove_id: int):
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(window_name=f"MANUS glove {glove_id}")

        self.glove_id = glove_id
        self.node_meshes = {}     # node_id -> sphere mesh
        self.node_positions = {}  # node_id -> np.array([x,y,z])

        self.frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.line_set = o3d.geometry.LineSet()

        self.viz.add_geometry(self.frame_mesh)
        self.viz.add_geometry(self.line_set)


class MinimalSubscriber(Node):
    def __init__(self, glove_indexes):
        super().__init__("manus_ros2_client_py")

        self.glove_viz_map = {}

        # Subscribe to glove data topics for each glove index
        self.sub_poses = []
        for glove_id in glove_indexes:
            topic_name = f"/manus_glove_{glove_id}"
            self.sub_poses.append(
                self.create_subscription(
                    ManusGlove,
                    topic_name,
                    self.glove_callback,
                    20,
                )
            )

        # Render loop timer (50 Hz)
        self.timer = self.create_timer(0.02, self.timer_callback)

    def glove_callback(self, msg: ManusGlove):
        """Callback for glove data"""

        # ---- Option A: Only show the real left glove ----
        if SHOW_ONLY_LEFT_GLOVE and hasattr(msg, "side"):
            if str(msg.side).lower() != "left":
                return

        if msg.glove_id not in self.glove_viz_map:
            self.glove_viz_map[msg.glove_id] = GloveViz(msg.glove_id)

        glove_viz = self.glove_viz_map[msg.glove_id]

        # 1) Collect raw positions + parent relations
        raw_positions = {}
        node_parent = {}

        for node in msg.raw_nodes:
            p = node.pose.position
            raw_positions[node.node_id] = np.array([p.x, p.y, p.z], dtype=float)
            node_parent[node.node_id] = node.parent_node_id

        if not raw_positions:
            # Nothing to draw yet
            return

        # 2) Choose a "root" to mirror around (typically wrist/palm):
        # pick a node whose parent is not present / invalid
        node_ids = set(raw_positions.keys())
        root_id = None
        for nid, pid in node_parent.items():
            if pid not in node_ids or pid < 0 or pid == nid:
                root_id = nid
                break
        if root_id is None:
            root_id = next(iter(node_ids))

        root = raw_positions[root_id]

        # 3) Option B: Mirror right glove to look like left glove
        is_right = hasattr(msg, "side") and str(msg.side).lower() == "right"
        if MIRROR_RIGHT_TO_LEFT and is_right:
            for nid in raw_positions:
                raw_positions[nid] = root + (raw_positions[nid] - root) * MIRROR_AXIS

        # Store positions for line drawing
        glove_viz.node_positions = raw_positions

        # 4) Update node spheres
        for node in msg.raw_nodes:
            node_id = node.node_id
            pos = glove_viz.node_positions[node_id]

            if node_id not in glove_viz.node_meshes:
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                mesh.compute_vertex_normals()
                glove_viz.node_meshes[node_id] = mesh
                glove_viz.viz.add_geometry(mesh)

            mesh = glove_viz.node_meshes[node_id]
            mesh.translate(-np.asarray(mesh.get_center()), relative=True)  # reset to origin
            mesh.translate(pos, relative=False)
            glove_viz.viz.update_geometry(mesh)

        # 5) Update line connections (uses glove_viz.node_positions, so mirroring applies)
        self.update_lines(glove_viz, msg)

    def update_lines(self, glove_viz: GloveViz, msg: ManusGlove):
        """Update the lines connecting child and parent nodes"""
        line_points = []
        line_indices = []

        for node in msg.raw_nodes:
            node_id = node.node_id
            parent_id = node.parent_node_id

            if parent_id in glove_viz.node_positions and node_id in glove_viz.node_positions:
                parent_pos = glove_viz.node_positions[parent_id]
                child_pos = glove_viz.node_positions[node_id]

                start_idx = len(line_points)
                line_points.append(parent_pos)
                line_points.append(child_pos)
                line_indices.append([start_idx, start_idx + 1])

        if line_points:
            glove_viz.line_set.points = o3d.utility.Vector3dVector(line_points)
            glove_viz.line_set.lines = o3d.utility.Vector2iVector(line_indices)
            glove_viz.line_set.paint_uniform_color([0, 0, 0])
            glove_viz.viz.update_geometry(glove_viz.line_set)

    def timer_callback(self):
        for glove_viz in self.glove_viz_map.values():
            glove_viz.viz.poll_events()
            glove_viz.viz.update_renderer()


def spin_node(glove_indexes):
    rclpy.init(args=sys.argv)
    node = MinimalSubscriber(glove_indexes)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main():
    spin_thread = threading.Thread(target=spin_node, args=(GLOVE_INDEXES,), daemon=True)
    spin_thread.start()

    rate = RateLimiter(frequency=120.0, warn=False)
    while True:
        rate.sleep()


if __name__ == "__main__":
    main()
