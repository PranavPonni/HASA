import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, shutil
import matplotlib
from matplotlib.animation import FFMpegWriter
matplotlib.use('Agg')  # Use a non-interactive backend for saving videos
ffmpeg_bin = shutil.which("ffmpeg")
if ffmpeg_bin:
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_bin


XELA_COORDS_INDEX = [
    (5, 3), (5, 2), (5, 1), (5, 0),
    (4, 4), (4, 3), (4, 2), (4, 1), (4, 0),
    (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0),
    (2, 5), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0),
    (1, 4), (1, 3), (1, 2), (1, 1), (1, 0),
    (0, 3), (0, 2), (0, 1), (0, 0)
]

XELA_COORDS_THUMB = [
    (5, 3), (5, 2), (5, 1), (5, 0),
    (4, 4), (4, 3), (4, 2), (4, 1), (4, 0),
    (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0),
    (2, 5), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0),
    (1, 4), (1, 3), (1, 2), (1, 1), (1, 0),
    (0, 3), (0, 0), (0, 1), (0, 2)
]

XELA_COORDS_INDEX_LIKE = XELA_COORDS_INDEX

FINGER_COORDS = {
    "index": XELA_COORDS_INDEX,
    "thumb": XELA_COORDS_THUMB,
}

UNVERIFIED_FINGER_COORD_FALLBACKS = {
    "middle": XELA_COORDS_INDEX,
    "ring": XELA_COORDS_INDEX,
}


class DualXelaVisualizer:
    def __init__(self):
        self.index_obs = np.zeros((30, 3))
        self.thumb_obs = np.zeros((30, 3))

        self.remapped_coords_index = XELA_COORDS_INDEX
        self.remapped_coords_thumb = XELA_COORDS_THUMB

        self.setup_figure()

    def setup_figure(self):
        self.fig, (self.ax_index, self.ax_thumb) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.patch.set_facecolor('black')
        self.ax_index.set_facecolor('black')
        self.ax_thumb.set_facecolor('black')

        self.setup_single_plot(self.ax_index, self.remapped_coords_index, "index_tip", -1., 1.)
        self.setup_single_plot(self.ax_thumb, self.remapped_coords_thumb, "thumb_tip", -1., 1.)

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
            angles='xy', scale_units='xy', scale=3,
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

    def update_plot(self, obs, sc, quiver, text_labels, clim_min, clim_max):
        if obs.shape != (30, 3):
            return
        # scale vector
        vectors = obs[:, :2]*10
        quiver.set_UVC(vectors[:, 0], vectors[:, 1])
        magnitudes = np.linalg.norm(obs, axis=1)
        clipped = np.clip(magnitudes, clim_min, clim_max)
        sc.set_array(clipped)
        sc.set_clim(clim_min, clim_max)
        for i, val in enumerate(clipped):
            text_labels[i].set_text(f"{val:.2f}")

    def live_update(self, frame):
        # Called by matplotlib animation
        self.update_plot(
            self.index_obs, self.index_tip_sc, self.index_tip_quiver,
            self.index_tip_text_labels, self.index_tip_clim_min, self.index_tip_clim_max
        )
        self.update_plot(
            self.thumb_obs, self.thumb_tip_sc, self.thumb_tip_quiver,
            self.thumb_tip_text_labels, self.thumb_tip_clim_min, self.thumb_tip_clim_max
        )

    def start_animation(self, interval=100):
        ani = animation.FuncAnimation(self.fig, self.live_update, interval=interval, cache_frame_data=False)
        plt.show()

    def export_video_from_array(self, data_array, path="", fps=10):
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        ffmpeg_path = matplotlib.rcParams.get("animation.ffmpeg_path")
        can_mp4 = path.lower().endswith(".mp4") and ffmpeg_path and os.path.exists(ffmpeg_path)

        if can_mp4:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='XELA Visualizer'))
            print(f"Saving MP4 video to {path} ...")
        else:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            # fallback to GIF if mp4 impossible
            root, _ = os.path.splitext(path or "./log/vis.mp4")
            path = root + ".gif"
            print(f"[vis] ffmpeg not available; saving GIF to {path} ...")

        with writer.saving(self.fig, path, dpi=100):
            for t in range(data_array.shape[0]):
                self.set_index_obs(data_array[t, 0])
                self.set_thumb_obs(data_array[t, 1])
                self.update_plot(
                    self.index_obs, self.index_tip_sc, self.index_tip_quiver,
                    self.index_tip_text_labels, self.index_tip_clim_min, self.index_tip_clim_max
                )
                self.update_plot(
                    self.thumb_obs, self.thumb_tip_sc, self.thumb_tip_quiver,
                    self.thumb_tip_text_labels, self.thumb_tip_clim_min, self.thumb_tip_clim_max
                )
                self.fig.canvas.draw()
                writer.grab_frame()
        print("Video export complete.")


class FourFingerTouchStateVisualizer:
    """Render self-touch and object-touch state on all four XELA fingertips."""

    FINGER_ORDER = ("index", "thumb", "middle", "ring")
    STATE_ORDER = ("selftouch", "object")
    STATE_TITLES = {
        "selftouch": "Self-touch",
        "object": "Object touch",
    }
    STATE_CMAPS = {
        "selftouch": "magma",
        "object": "viridis",
    }
    STATE_ARROW_COLORS = {
        "selftouch": "#70f7ff",
        "object": "#f4f4f4",
    }

    def __init__(self, fingers=None, clim_max=None):
        self.fingers = tuple(fingers or self.FINGER_ORDER)
        self.clim_min = 0.0
        self.clim_max = clim_max
        self.fig, self.axes = plt.subplots(
            len(self.fingers),
            len(self.STATE_ORDER),
            figsize=(11.5, max(2.65 * len(self.fingers) + 1.2, 5.8)),
            squeeze=False,
            gridspec_kw={"hspace": 0.22, "wspace": 0.10},
        )
        self.fig.patch.set_facecolor("black")
        self.fig.patch.set_alpha(1.0)
        self.artists = {}
        self.state_mappables = {}
        self.colorbars = {}
        self._state_vmax = {state: 1.0 for state in self.STATE_ORDER}
        self._setup_figure()

    @staticmethod
    def _as_taxel_vectors(value):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3:
            raise ValueError(f"touch arrays must have shape (T, taxels, axes), got {arr.shape}")
        if arr.shape[-1] == 3:
            return arr
        if arr.shape[-1] == 1:
            out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
            out[..., 2] = arr[..., 0]
            return out
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
        width = min(3, arr.shape[-1])
        out[..., :width] = arr[..., :width]
        return out

    def _coords_for_finger(self, finger, taxel_count):
        coords = FINGER_COORDS.get(finger)
        if coords is None:
            coords = UNVERIFIED_FINGER_COORD_FALLBACKS.get(finger, XELA_COORDS_INDEX)
            print(
                f"[vis] {finger} taxel layout is not PlotJuggler-verified; "
                "using index fallback layout."
            )
        coords = list(coords)
        if len(coords) >= taxel_count:
            coords = coords[:taxel_count]
        else:
            cols = 6
            coords.extend(
                (idx // cols, idx % cols)
                for idx in range(len(coords), taxel_count)
            )
        coords = np.asarray(coords, dtype=np.float32)
        return coords[:, 0], coords[:, 1]

    def _setup_figure(self):
        for row, finger in enumerate(self.fingers):
            coords = self._coords_for_finger(finger, 30)
            for col, state in enumerate(self.STATE_ORDER):
                ax = self.axes[row, col]
                x_coords, y_coords = coords
                ax.set_facecolor("black")
                sc = ax.scatter(
                    x_coords,
                    y_coords,
                    c=np.zeros_like(x_coords),
                    cmap=self.STATE_CMAPS[state],
                    s=330,
                    edgecolors="#e8e8e8",
                    linewidths=0.55,
                    vmin=self.clim_min,
                    vmax=1.0,
                )
                self.state_mappables.setdefault(state, sc)
                quiver = ax.quiver(
                    x_coords,
                    y_coords,
                    np.zeros_like(x_coords),
                    np.zeros_like(y_coords),
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=self.STATE_ARROW_COLORS[state],
                    width=0.0034,
                    headwidth=3.2,
                    headlength=4.2,
                    headaxislength=3.7,
                    alpha=0.72,
                )
                peak = ax.scatter(
                    [x_coords[0]],
                    [y_coords[0]],
                    s=520,
                    facecolors="none",
                    edgecolors="yellow",
                    linewidths=2.35,
                    visible=False,
                )
                ax.set_xlim(-0.8, 5.8)
                ax.set_ylim(-0.8, 5.8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
                if row == 0:
                    ax.set_title(self.STATE_TITLES[state], color="white", fontsize=12, pad=8)
                if col == 0:
                    ax.text(
                        -0.13,
                        0.5,
                        finger.capitalize(),
                        color="white",
                        fontsize=10,
                        ha="right",
                        va="center",
                        rotation=90,
                        transform=ax.transAxes,
                    )
                value_text = ax.text(
                    0.02,
                    0.03,
                    "",
                    color="white",
                    fontsize=7.5,
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.58, pad=1.6),
                )
                for spine in ax.spines.values():
                    spine.set_visible(False)
                for y in range(6):
                    ax.plot([-0.5, 5.5], [y, y], color="white", alpha=0.08, linewidth=0.5)
                for x in range(6):
                    ax.plot([x, x], [-0.5, 5.5], color="white", alpha=0.08, linewidth=0.5)
                self.artists[(finger, state)] = {
                    "scatter": sc,
                    "quiver": quiver,
                    "peak": peak,
                    "x": x_coords,
                    "y": y_coords,
                    "value_text": value_text,
                }
        for col, state in enumerate(self.STATE_ORDER):
            mappable = self.state_mappables.get(state)
            if mappable is None:
                continue
            cbar = self.fig.colorbar(
                mappable,
                ax=self.axes[:, col].ravel().tolist(),
                fraction=0.032,
                pad=0.018,
            )
            cbar.set_label("magnitude", color="white", fontsize=8)
            cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
            cbar.outline.set_edgecolor("white")
            self.colorbars[state] = cbar

    def _set_clim(self, touch_state):
        for state in self.STATE_ORDER:
            if self.clim_max is not None:
                vmax = float(self.clim_max)
            else:
                values = []
                for finger_data in touch_state.values():
                    arr = finger_data.get(state)
                    if arr is None:
                        continue
                    vec = self._as_taxel_vectors(arr)
                    values.append(np.linalg.norm(vec, axis=-1).reshape(-1))
                if values:
                    finite = np.concatenate(values)
                    finite = finite[np.isfinite(finite)]
                    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
                else:
                    vmax = 1.0
            vmax = max(vmax, 1e-6)
            self._state_vmax[state] = vmax
            for finger in self.fingers:
                artist = self.artists.get((finger, state))
                if artist is not None:
                    artist["scatter"].set_clim(self.clim_min, vmax)
            if state in self.colorbars and state in self.state_mappables:
                self.colorbars[state].update_normal(self.state_mappables[state])

    def _normalise_touch_state(self, touch_state):
        frame_count = 1
        for finger_data in (touch_state or {}).values():
            if not isinstance(finger_data, dict):
                continue
            for value in finger_data.values():
                if value is None:
                    continue
                frame_count = max(frame_count, self._as_taxel_vectors(value).shape[0])
        normalised = {}
        for finger in self.fingers:
            source = touch_state.get(finger, {}) if isinstance(touch_state, dict) else {}
            normalised[finger] = {}
            for state in self.STATE_ORDER:
                value = source.get(state)
                if value is None:
                    value = np.zeros((frame_count, 30, 3), dtype=np.float32)
                normalised[finger][state] = self._as_taxel_vectors(value)
        return normalised

    def update_frame(self, frame, touch_state):
        for finger in self.fingers:
            for state in self.STATE_ORDER:
                arr = touch_state[finger][state]
                frame_idx = min(frame, arr.shape[0] - 1)
                obs = arr[frame_idx]
                artist = self.artists[(finger, state)]
                taxel_count = min(obs.shape[0], artist["x"].shape[0])
                obs = np.nan_to_num(obs[:taxel_count], nan=0.0, posinf=0.0, neginf=0.0)
                magnitudes = np.linalg.norm(obs, axis=-1)
                artist["scatter"].set_offsets(
                    np.column_stack([artist["x"][:taxel_count], artist["y"][:taxel_count]])
                )
                artist["scatter"].set_array(magnitudes)
                lateral = obs[:, :2]
                lateral_mag = np.linalg.norm(lateral, axis=-1)
                vmax = max(float(self._state_vmax.get(state, 1.0)), 1e-6)
                strength = np.clip(magnitudes / vmax, 0.0, 1.0)
                arrow_length = np.where(strength > 0.08, 0.10 + 0.34 * strength, 0.0)
                vectors = np.divide(
                    lateral,
                    lateral_mag[:, None],
                    out=np.zeros_like(lateral, dtype=np.float32),
                    where=lateral_mag[:, None] > 1e-6,
                )
                vectors = vectors * arrow_length[:, None]
                artist["quiver"].set_offsets(
                    np.column_stack([artist["x"][:taxel_count], artist["y"][:taxel_count]])
                )
                artist["quiver"].set_UVC(vectors[:, 0], vectors[:, 1])
                if magnitudes.size:
                    peak_idx = int(np.argmax(magnitudes))
                    artist["peak"].set_offsets([[artist["x"][peak_idx], artist["y"][peak_idx]]])
                    peak_visible = bool(magnitudes[peak_idx] > 0.02 * vmax)
                    artist["peak"].set_visible(peak_visible)
                    peak_text = f"peak {peak_idx}: {float(magnitudes[peak_idx]):.1f}" if peak_visible else ""
                else:
                    artist["peak"].set_visible(False)
                    peak_text = ""
                artist["value_text"].set_text(peak_text)
        self.fig.suptitle(f"Touch states  |  frame {frame}", color="white", fontsize=13)

    def export_touch_state_video(self, touch_state, path, fps=10, frame_stride=1):
        touch_state = self._normalise_touch_state(touch_state)
        self._set_clim(touch_state)
        frame_count = min(
            arr.shape[0]
            for finger_data in touch_state.values()
            for arr in finger_data.values()
        )
        frame_stride = max(int(frame_stride or 1), 1)
        frames = range(0, frame_count, frame_stride)

        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        ffmpeg_path = matplotlib.rcParams.get("animation.ffmpeg_path")
        can_mp4 = path.lower().endswith(".mp4") and ffmpeg_path and os.path.exists(ffmpeg_path)
        if can_mp4:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist="XELA Touch State Visualizer"))
            out_path = path
            print(f"Saving touch-state MP4 video to {out_path} ...")
        else:
            from matplotlib.animation import PillowWriter
            root, _ = os.path.splitext(path or "./log/touch_state.mp4")
            out_path = root + ".gif"
            writer = PillowWriter(fps=fps)
            print(f"[vis] ffmpeg not available; saving touch-state GIF to {out_path} ...")

        with writer.saving(self.fig, out_path, dpi=110):
            for frame in frames:
                self.update_frame(frame, touch_state)
                self.fig.canvas.draw()
                writer.grab_frame(facecolor=self.fig.get_facecolor())
        print("Touch-state video export complete.")
        return out_path

# if __name__ == "__main__":
#     visualizer = DualXelaVisualizer()
#     dummy_data = np.random.rand(60, 2, 30, 3)
#     visualizer.export_video_from_array(dummy_data, fps=10)
