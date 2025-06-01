import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import tkinter as tk

class DualXelaVisualizer:
    def __init__(self):
        self.index_obs = np.zeros((30, 3))
        self.thumb_obs = np.zeros((30, 3))

        self.remapped_coords_index = [
            (5, 3), (5, 2), (5, 1), (5, 0),
            (4, 4), (4, 3), (4, 2), (4, 1), (4, 0),
            (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0),
            (2, 5), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0),
            (1, 4), (1, 3), (1, 2), (1, 1), (1, 0),
            (0, 3), (0, 2), (0, 1), (0, 0)
        ]

        self.remapped_coords_thumb = [
            (5, 3), (5, 2), (5, 1), (5, 0),
            (4, 4), (4, 3), (4, 2), (4, 1), (4, 0),
            (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0),
            (2, 5), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0),
            (1, 4), (1, 3), (1, 2), (1, 1), (1, 0),
            (0, 3), (0, 0), (0, 1), (0, 2)
        ]

        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("XELA Dual Tactile Heatmaps")
        self.fig, (self.ax_index, self.ax_thumb) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.patch.set_facecolor('black')
        self.ax_index.set_facecolor('black')
        self.ax_thumb.set_facecolor('black')

        self.setup_single_plot(self.ax_index, self.remapped_coords_index, "index_tip", 0.6730, 0.6735)
        self.setup_single_plot(self.ax_thumb, self.remapped_coords_thumb, "thumb_tip", 0.6820, 0.6825)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_single_plot(self, ax, coords, tag, clim_min, clim_max):
        x_coords, y_coords = zip(*coords)
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        magnitudes = np.zeros(len(coords))
        vectors = np.zeros((len(coords), 2))

        sc = ax.scatter(
            x_coords, y_coords, c=magnitudes, cmap='plasma', s=1000,
            edgecolors='lime', linewidths=0.5
        )
        quiver = ax.quiver(
            x_coords, y_coords, vectors[:, 0], vectors[:, 1],
            angles='xy', scale_units='xy', scale=50000,
            color='cyan', width=0.01
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Tactile Magnitude", color='white', fontsize=8)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(f"XELA {tag.replace('_tip', '').capitalize()} Tip Heatmap", color='white', fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for y in range(6):
            ax.plot([-0.5, 5.5], [y, y], color='green', alpha=0.2, linewidth=0.5)
        for x in range(6):
            ax.plot([x, x], [-0.5, 5.5], color='green', alpha=0.2, linewidth=0.5)

        text_labels = [
            ax.text(x, y, "", color='black', ha='center', va='center', fontsize=7)
            for x, y in zip(x_coords, y_coords)
        ]

        setattr(self, f"{tag}_x", x_coords)
        setattr(self, f"{tag}_y", y_coords)
        setattr(self, f"{tag}_sc", sc)
        setattr(self, f"{tag}_quiver", quiver)
        setattr(self, f"{tag}_text_labels", text_labels)
        setattr(self, f"{tag}_clim_min", clim_min)
        setattr(self, f"{tag}_clim_max", clim_max)

    def set_index_obs(self, obs):
        self.index_obs = obs

    def set_thumb_obs(self, obs):
        self.thumb_obs = obs

    def update(self, frame):
        self.update_plot(
            self.index_obs,
            self.index_tip_sc, self.index_tip_quiver, self.index_tip_text_labels,
            self.index_tip_clim_min, self.index_tip_clim_max
        )
        self.update_plot(
            self.thumb_obs,
            self.thumb_tip_sc, self.thumb_tip_quiver, self.thumb_tip_text_labels,
            self.thumb_tip_clim_min, self.thumb_tip_clim_max
        )
        return []

    def update_plot(self, obs, sc, quiver, text_labels, clim_min, clim_max):
        if obs.shape != (30, 3):
            return

        vectors = obs[:, :2]
        quiver.set_UVC(vectors[:, 0], vectors[:, 1])
        magnitudes = np.linalg.norm(obs, axis=1) / 100000
        clipped = np.clip(magnitudes, clim_min, clim_max)
        sc.set_array(clipped)
        sc.set_clim(clim_min, clim_max)

        for i, val in enumerate(clipped):
            text_labels[i].set_text(f"{val:.2f}")

    def export_video(self, filename="xela_dual_output.mp4"):
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=10, metadata=dict(artist='XELA Visualizer'))
        print(f"Saving MP4 video to {filename} ...")
        with writer.saving(self.fig, filename, dpi=100):
            for _ in self.frames_to_record:
                self.canvas.draw()
                writer.grab_frame()
        print("Video export complete.")


    def run(self):
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=100, blit=False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.mainloop()
