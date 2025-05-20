import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import tkinter as tk

from py_node_exec import NodeExec
from allegro_package import AllegroHand
from xela_py import TactileSubscriber

# ---- ROS Initialization ----
node = NodeExec(freq=10)
node.spin_thread_start()

robot = AllegroHand(hand_topic_prefix="allegroHand_0", ctrl_freq=5)
index = TactileSubscriber(topic_prefix="index_tip")

# ---- Taxel layout ----
x_coords = np.array([
     0,  1,  2,  3,  4,  5,
     0,  1,  2,  3,  4,  5,
     0,  1,  2,  3,  4,  5,
     0,  1,  2,  3,  4,  5,
     1,  2,  3,  4,
     1.95, 3.05
])
y_coords = np.array([
     0,  0,  0,  0,  0,  0,
     1,  1,  1,  1,  1,  1,
     2,  2,  2,  2,  2,  2,
     3,  3,  3,  3,  3,  3,
     4,  4,  4,  4,
     5,  5
])

assert len(x_coords) == 30 and len(y_coords) == 30

# ---- Tkinter GUI Setup ----
root = tk.Tk()
root.title("XELA Index Tip Tactile Heatmap with Vectors")

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

magnitudes = np.zeros(30)
vectors = np.zeros((30, 2))  # (x, y) vector components

# Color magnitude visualization
sc = ax.scatter(x_coords, y_coords, c=magnitudes, cmap='plasma', s=1000, edgecolors='lime', linewidths=0.5)

# Arrows (quiver plot)
quiver = ax.quiver(x_coords, y_coords, vectors[:, 0], vectors[:, 1],
                   angles='xy', scale_units='xy', scale=35000,
                   color='cyan', width=0.01)

# Colorbar setup
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Tactile Magnitude", color='white', fontsize=8)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Plot layout
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 7)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.set_title("XELA Index Tip Heatmap with Force Vectors", color='white', fontsize=10)
for spine in ax.spines.values():
    spine.set_visible(False)

# Optional grid
for y in range(6):
    ax.plot([-0.5, 5.5], [y, y], color='green', alpha=0.2, linewidth=0.5)
for x in range(6):
    ax.plot([x, x], [-0.5, 5.5], color='green', alpha=0.2, linewidth=0.5)

# Labels (optional)
text_labels = [
    ax.text(x, y, "", color='black', ha='center', va='center', fontsize=7)
    for x, y in zip(x_coords, y_coords)
]

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ---- Update function ----
def update(frame):
    if not node.ok():
        return sc, quiver, *text_labels

    obs = np.array(index.get_obs())
    if obs.shape != (30, 3):
        print("Invalid tactile shape:", obs.shape)
        return sc, quiver, *text_labels

    # Extract raw force vectors
    vectors[:, 0] = obs[:, 0]  # Fx
    vectors[:, 1] = obs[:, 1]  # Fy

    # Compute raw magnitude (norm of Fx, Fy, Fz)
    raw_magnitudes = np.linalg.norm(obs, axis=1)

    # Normalize magnitudes to [0, 1]
    max_val = 100000  
    norm_magnitudes = raw_magnitudes / max_val
    clipped_magnitudes = np.clip(norm_magnitudes, 0.65, 0.75)

    # Update scatter colors
    sc.set_array(clipped_magnitudes)
    sc.set_clim(0.65, 0.75)  
    quiver.set_UVC(vectors[:, 0], vectors[:, 1])

    # Update label text to show normalized magnitude 
    for i, val in enumerate(clipped_magnitudes):
        text_labels[i].set_text(f"{val:.2f}" if val > 0.5 else "")

    return sc, quiver, *text_labels


# ---- ROS + GUI Loop ----
def ros_loop():
    if node.ok():
        obs_joint = robot.get_obs()
        joint_positions = obs_joint.get("jnt_pos", None)
        if joint_positions is not None:
            print("\nJoint Positions:", np.round(joint_positions, 4))
        root.after(100, ros_loop)

ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
root.after(0, ros_loop)
root.protocol("WM_DELETE_WINDOW", lambda: (node.spin_thread_finish(), root.destroy()))
root.mainloop()