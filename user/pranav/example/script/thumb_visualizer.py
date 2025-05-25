import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import tkinter as tk

from py_node_exec import NodeExec
from xela_py import TactileSubscriber

class XelaVisualizer:
    def __init__(self, topic_prefix="thumb_tip", freq=10):
        self.node = NodeExec(freq=freq)
        self.thumb = TactileSubscriber(topic_prefix=topic_prefix)

        self.remapped_coords = [
            (5, 3), (5, 2), (5, 1), (5, 0),
            (4, 4), (4, 3), (4, 2), (4, 1), (4, 0),
            (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0),
            (2, 5), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0),
            (1, 4), (1, 3), (1, 2), (1, 1), (1, 0),
            (0, 3), (0, 2), (0, 1), (0, 0)
        ]

        self.x_coords, self.y_coords = zip(*self.remapped_coords)
        self.x_coords = np.array(self.x_coords)
        self.y_coords = np.array(self.y_coords)

        assert len(self.x_coords) == 30 and len(self.y_coords) == 30

        self.magnitudes = np.zeros(30)
        self.vectors = np.zeros((30, 2))  # (x, y) vectors

        self.setup_gui()
        self.node.spin_thread_start()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("XELA Thumb Tip Tactile Heatmap")

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')

        self.sc = self.ax.scatter(
            self.x_coords, self.y_coords,
            c=self.magnitudes, cmap='plasma', s=1000,
            edgecolors='lime', linewidths=0.5
        )

        self.quiver = self.ax.quiver(
            self.x_coords, self.y_coords,
            self.vectors[:, 0], self.vectors[:, 1],
            angles='xy', scale_units='xy', scale=50000,
            color='cyan', width=0.01
        )

        self.cbar = plt.colorbar(self.sc, ax=self.ax)
        self.cbar.set_label("Tactile Magnitude", color='white', fontsize=8)
        self.cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(self.cbar.ax.axes, 'yticklabels'), color='white')

        self.ax.set_xlim(-1, 6)
        self.ax.set_ylim(-1, 7)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')
        self.ax.set_title("XELA Thumb Tip Heatmap", color='white', fontsize=10)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        for y in range(6):
            self.ax.plot([-0.5, 5.5], [y, y], color='green', alpha=0.2, linewidth=0.5)
        for x in range(6):
            self.ax.plot([x, x], [-0.5, 5.5], color='green', alpha=0.2, linewidth=0.5)

        self.text_labels = [
            self.ax.text(x, y, "", color='black', ha='center', va='center', fontsize=7)
            for x, y in zip(self.x_coords, self.y_coords)
        ]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update(self, frame):
        if not self.node.ok():
            return self.sc, self.quiver, *self.text_labels

        obs = np.array(self.thumb.get_obs())
        if obs.shape != (30, 3):
            print("Invalid tactile shape:", obs.shape)
            return self.sc, self.quiver, *self.text_labels

        self.vectors[:, 0] = obs[:, 0]
        self.vectors[:, 1] = obs[:, 1]
        self.quiver.set_UVC(self.vectors[:, 0], self.vectors[:, 1])

        raw_magnitudes = np.linalg.norm(obs, axis=1)
        norm_magnitudes = raw_magnitudes / 100000
        clipped_magnitudes = np.clip(norm_magnitudes, 0.670, 0.675)

        self.sc.set_array(clipped_magnitudes)
        self.sc.set_clim(0.670, 0.675)

        for i, val in enumerate(clipped_magnitudes):
            self.text_labels[i].set_text(f"{val:.2f}")

        return self.sc, self.quiver, *self.text_labels

    def run(self):
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=100, blit=False, cache_frame_data=False)
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.root.mainloop()

    def shutdown(self):
        self.node.spin_thread_finish()
        self.root.destroy()


if __name__ == "__main__":
    visualizer = XelaVisualizer()
    visualizer.run()
