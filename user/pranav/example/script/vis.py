import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.animation import FFMpegWriter

class DualXelaVisualizer:
    def __init__(self):
        self.index_obs = np.zeros((30, 3))
        self.thumb_obs = np.zeros((30, 3))

        # Fully filled sample ranges for index and thumb (replace with real calibration)
        self.index_ranges = np.array([
            [0.65 + i*0.001, 0.68 + i*0.001] for i in range(30)
        ])
        self.thumb_ranges = np.array([
            [0.67 + i*0.001, 0.69 + i*0.001] for i in range(30)
        ])

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

        self.setup_figure()

    def setup_figure(self):
        self.fig, (self.ax_index, self.ax_thumb) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.patch.set_facecolor('black')
        self.ax_index.set_facecolor('black')
        self.ax_thumb.set_facecolor('black')

        self.setup_single_plot(self.ax_index, self.remapped_coords_index, "index_tip", self.index_ranges)
        self.setup_single_plot(self.ax_thumb, self.remapped_coords_thumb, "thumb_tip", self.thumb_ranges)

    def setup_single_plot(self, ax, coords, tag, taxel_ranges):
        x_coords, y_coords = zip(*coords)
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        magnitudes = np.zeros(len(coords))
        vectors = np.zeros((len(coords), 2))

        sc = ax.scatter(
            x_coords, y_coords, c=magnitudes, cmap='plasma', s=1000,
            edgecolors='lime', linewidths=0.5, vmin=0.0, vmax=1.0  # normalized 0–1
        )
        quiver = ax.quiver(
            x_coords, y_coords, vectors[:, 0], vectors[:, 1],
            angles='xy', scale_units='xy', scale=7500,
            color='cyan', width=0.01
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Normalized Tactile Magnitude", color='white', fontsize=8)
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
        setattr(self, f"{tag}_ranges", taxel_ranges)

    def set_index_obs(self, obs):
        self.index_obs = obs

    def set_thumb_obs(self, obs):
        self.thumb_obs = obs

    def update_plot(self, obs, sc, quiver, text_labels, taxel_ranges):
        if obs.shape != (30, 3):
            return
        vectors = obs[:, :2]
        quiver.set_UVC(vectors[:, 0], vectors[:, 1])

        magnitudes = np.linalg.norm(obs, axis=1) / 10000  # original scale

        # Normalize per taxel using individual min-max ranges
        min_vals = taxel_ranges[:, 0]
        max_vals = taxel_ranges[:, 1]
        normalized = (magnitudes - min_vals) / (max_vals - min_vals)
        normalized = np.clip(normalized, 0.0, 1.0)

        sc.set_array(normalized)
        for i, val in enumerate(normalized):
            text_labels[i].set_text(f"{val:.2f}")

    def live_update(self, frame):
        self.update_plot(
            self.index_obs, self.index_tip_sc, self.index_tip_quiver,
            self.index_tip_text_labels, self.index_tip_ranges
        )
        self.update_plot(
            self.thumb_obs, self.thumb_tip_sc, self.thumb_tip_quiver,
            self.thumb_tip_text_labels, self.thumb_tip_ranges
        )

    def start_animation(self, interval=100):
        ani = animation.FuncAnimation(self.fig, self.live_update, interval=interval, cache_frame_data=False)
        plt.show()

    def export_video_from_array(self, data_array, fps=10):
        os.makedirs("exported_videos", exist_ok=True)
        existing = [f for f in os.listdir("exported_videos") if f.startswith("episode_") and f.endswith(".mp4")]
        episode_nums = [int(f.split("_")[1].split(".")[0]) for f in existing if f.split("_")[1].split(".")[0].isdigit()]
        next_episode = max(episode_nums, default=0) + 1
        path = os.path.join("exported_videos", f"episode_{next_episode}.mp4")

        writer = FFMpegWriter(fps=fps, metadata=dict(artist='XELA Visualizer'))
        print(f"Saving MP4 video to {path} ...")
        with writer.saving(self.fig, path, dpi=100):
            for t in range(data_array.shape[0]):
                self.set_index_obs(data_array[t, 0])
                self.set_thumb_obs(data_array[t, 1])
                self.update_plot(
                    self.index_obs, self.index_tip_sc, self.index_tip_quiver,
                    self.index_tip_text_labels, self.index_tip_ranges
                )
                self.update_plot(
                    self.thumb_obs, self.thumb_tip_sc, self.thumb_tip_quiver,
                    self.thumb_tip_text_labels, self.thumb_tip_ranges
                )
                writer.grab_frame()
        print("Video export complete.")

# Example dummy usage
# if __name__ == "__main__":
#     visualizer = DualXelaVisualizer()
#     dummy_data = np.random.rand(60, 2, 30, 3) * 100000
#     visualizer.export_video_from_array(dummy_data, fps=10)
