import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import time

print("[INFO] Using pyrealsense2")

# ==========================
# MediaPipe Hands setup
# ==========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solu
tions.drawing_utils

# Yellow landmarks, black connections (BGR)
landmark_style = mp_drawing.DrawingSpec(
    color=(0, 255, 255),  # yellow
    thickness=3,
    circle_radius=3
)
connection_style = mp_drawing.DrawingSpec(
    color=(0, 0, 0),      # black
    thickness=3,
    circle_radius=2
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,          # 0, 1, 2 (2 = slower but more accurate)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ==========================
# RealSense setup
# ==========================

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    raise RuntimeError(
        "No RealSense device connected (SDK cannot see it).\n"
        "Check USB cable/port, maybe replug and try again."
    )

print("[INFO] RealSense devices found:")
for dev in devices:
    print("  -", dev.get_info(rs.camera_info.name))

pipeline = rs.pipeline()
config = rs.config()

# Enable depth + color
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("[INFO] Starting RealSense pipeline...")
profile = pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

depth_intrinsics = None  # will fill after first frame

prev_time = time.time()
print("[INFO] Press 'q' to quit.")

try:
    while True:
        # --------------------------
        # Get frames and align
        # --------------------------
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        if depth_intrinsics is None:
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # Convert color frame to numpy (BGR)
        color_image = np.asanyarray(color_frame.get_data())

        # ***** MIRROR THE IMAGE HORIZONTALLY *****
        color_image = cv2.flip(color_image, 1)

        h, w, _ = color_image.shape

        # --------------------------
        # MediaPipe hand detection
        # --------------------------
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        if result.multi_hand_landmarks:
            handedness_list = []
            if result.multi_handedness:
                handedness_list = [h.classification[0].label
                                   for h in result.multi_handedness]

            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Draw landmarks + connections on the mirrored image
                mp_drawing.draw_landmarks(
                    color_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=landmark_style,
                    connection_drawing_spec=connection_style,
                )

                # Label Left/Right (now matches your real left/right)
                label = handedness_list[hand_idx] if hand_idx < len(handedness_list) else f"Hand{hand_idx}"
                cv2.putText(
                    color_image,
                    label,
                    (10, 30 + 30 * hand_idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # Which landmarks to show 3D for (fingertips)
                landmarks_to_print = [4, 8, 12, 16, 20]
                # For all 21 points: landmarks_to_print = list(range(21))

                print(f"--- {label} ---")

                for i, lm in enumerate(hand_landmarks.landmark):
                    # Pixel coordinates in the *mirrored* image
                    u = int(lm.x * w)
                    v = int(lm.y * h)

                    if u < 0 or u >= w or v < 0 or v >= h:
                        continue

                    # Convert mirrored coordinate back to original for depth lookup
                    u_depth = w - 1 - u

                    depth = depth_frame.get_distance(u_depth, v)  # meters
                    if depth <= 0:
                        continue

                    # Back-project (u_depth, v, depth) to 3D camera coords
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsics, [u_depth, v], depth
                    )

                    if i in landmarks_to_print:
                        print(
                            f"Landmark {i:02d}: "
                            f"X={X:+.3f} m, Y={Y:+.3f} m, Z={Z:+.3f} m"
                        )
                        cv2.putText(
                            color_image,
                            f"{i}",
                            (u + 5, v - 5),
                            cv2.FONT_HERSHEY_PLAIN,
                            0.8,
                            (0, 0, 0),
                            1,
                        )

                print()

        # --------------------------
        # Show FPS
        # --------------------------
        cv2.putText(
            color_image,
            f"FPS: {fps:.1f}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # --------------------------
        # Display
        # --------------------------
        cv2.imshow("RealSense 3D Hand Pose (mirror view, 'q' to quit)", color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    hands.close()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")
