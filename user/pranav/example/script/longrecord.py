#!/usr/bin/env python3
import cv2
import os
import time
from datetime import datetime

SAVE_DIR = "/home/handlingteam2/HASA/user/pranav/example/data/longrecord"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Open default camera (0). Change to 1/2 if you have multiple webcams.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try explicitly with V4L2 on Linux
        cap.open(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    # (Optional) set a resolution:
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read a frame from the webcam.")
    h, w = frame.shape[:2]

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 30.0  # sensible default

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(SAVE_DIR, f"webcam_{ts}")

    # Try common codecs (auto-fallback)
    candidates = [("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]
    writer, out_path = None, None
    for fourcc_name, ext in candidates:
        path = base + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        wtr = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if wtr.isOpened():
            writer, out_path = wtr, path
            break
    if writer is None:
        cap.release()
        raise RuntimeError("Could not create a video writer (mp4v/XVID/MJPG). Install ffmpeg/gstreamer.")

    print(f"Recording... Press Ctrl+C to stop.\nSaving to: {out_path}")

    try:
        writer.write(frame)  # write the first frame we already grabbed
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed, stopping.")
                break
            writer.write(frame)
            time.sleep(0.001)  # small sleep to reduce CPU
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping...")
    finally:
        writer.release()
        cap.release()
        print(f"Saved video: {out_path}")

if __name__ == "__main__":
    main()
