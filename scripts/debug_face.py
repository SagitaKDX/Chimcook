#!/usr/bin/env python3
"""
scripts/debug_face.py
=====================
Standalone face detection debugger for Chimcook.

Usage (on HOST, outside Docker — needs a display):
    python3 scripts/debug_face.py [--camera 0] [--scale 0.5] [--no-gui]

Usage (inside Docker — text-only mode, no display needed):
    sudo docker exec -it chimcook-assistant python3 /app/scripts/debug_face.py --no-gui

What it shows
-------------
• Live camera preview with HOG bounding boxes drawn in green
• Frame number, FPS, detection count printed every second
• In --no-gui mode: prints detection results to terminal only

Controls (GUI mode)
-------------------
  q   — quit
  s   — save a snapshot to /tmp/debug_face_snap.jpg
  +/- — increase / decrease HOG upsampling (more = finds smaller faces, slower)

Interpreting results
--------------------
  HOG = 0 faces → camera image is too dark/blurry/far, or person is at wrong angle
  HOG = 1+ faces → detection working — check FACE_STABLE_FRAMES in constants.py
"""

import argparse
import sys
import time
from pathlib import Path

# ── Path setup (works whether run from project root or scripts/) ──────────────
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

import cv2
import face_recognition
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Chimcook face detection debugger")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default: 0)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Downscale factor for detection (0.5 = half res, faster)")
    p.add_argument("--upsample", type=int, default=1,
                   help="HOG upsample passes (1=normal, 2=finds small faces, slower)")
    p.add_argument("--no-gui", action="store_true",
                   help="Text-only mode — no OpenCV window (use inside Docker)")
    p.add_argument("--brightness", type=float, default=1.2,
                   help="Brightness multiplier applied before detection (default: 1.2)")
    p.add_argument("--beta", type=int, default=20,
                   help="Brightness offset applied before detection (default: 20)")
    return p.parse_args()


def preprocess(frame: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    """Apply brightness + contrast enhancement, same as FaceDetector in components.py."""
    bright = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # CLAHE contrast
    lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    bright = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(bright, -1, kernel)


def main():
    args = parse_args()
    gui = not args.no_gui

    print(f"\n{'='*60}")
    print(f"  Chimcook Face Detection Debugger")
    print(f"{'='*60}")
    print(f"  Camera index : {args.camera}")
    print(f"  Scale factor : {args.scale}  (detection resolution)")
    print(f"  HOG upsample : {args.upsample}")
    print(f"  Brightness   : alpha={args.brightness}, beta={args.beta}")
    print(f"  GUI mode     : {'YES (press q to quit)' if gui else 'NO (Ctrl+C to stop)'}")
    print(f"{'='*60}\n")

    # ── Open camera ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        print("        Available cameras — try indices 0, 1, 2, 4:")
        for i in range(5):
            t = cv2.VideoCapture(i)
            if t.isOpened():
                print(f"          [{i}] ✓")
                t.release()
            else:
                print(f"          [{i}] ✗")
        sys.exit(1)

    # Read one frame to confirm resolution
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera opened but cannot read frames")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"[Camera] Resolution: {w}x{h}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    upsample = args.upsample
    frame_count = 0
    detect_count = 0
    last_report = time.time()
    fps_frames = 0
    fps = 0.0

    print("[Running] Detecting faces... (Ctrl+C or 'q' in window to stop)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed — skipping")
                time.sleep(0.05)
                continue

            frame_count += 1
            fps_frames += 1

            # ── Preprocess (mirrors FaceDetector in components.py) ────────────
            processed = preprocess(frame, args.brightness, args.beta)

            # ── Downscale for faster detection ────────────────────────────────
            small = cv2.resize(processed, (0, 0), fx=args.scale, fy=args.scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # ── HOG face detection ────────────────────────────────────────────
            locations = face_recognition.face_locations(
                rgb_small,
                number_of_times_to_upsample=upsample,
                model="hog",
            )

            n_faces = len(locations)
            if n_faces > 0:
                detect_count += 1

            # ── FPS calc (every 1s) ───────────────────────────────────────────
            now = time.time()
            if now - last_report >= 1.0:
                fps = fps_frames / (now - last_report)
                hit_rate = detect_count / max(frame_count, 1) * 100
                print(
                    f"[Frame {frame_count:5d}]  "
                    f"FPS: {fps:.1f}  |  "
                    f"Faces this frame: {n_faces}  |  "
                    f"Detection rate: {hit_rate:.0f}%  |  "
                    f"Upsample: {upsample}"
                )
                fps_frames = 0
                detect_count = 0
                frame_count = 0
                last_report = now

            if gui:
                # ── Draw bounding boxes on the ORIGINAL frame ─────────────────
                inv_scale = 1.0 / args.scale
                for top, right, bottom, left in locations:
                    # Scale back to original resolution
                    t = int(top    * inv_scale)
                    r = int(right  * inv_scale)
                    b = int(bottom * inv_scale)
                    l = int(left   * inv_scale)
                    color = (0, 255, 0) if n_faces > 0 else (0, 0, 255)
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    cv2.putText(frame, "HOG FACE", (l, t - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # ── HUD overlay ───────────────────────────────────────────────
                status = f"Faces: {n_faces}  FPS: {fps:.1f}  Upsample: {upsample}"
                cv2.putText(frame, status, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if n_faces > 0 else (0, 80, 255), 2)

                cv2.imshow("Chimcook — Face Debug (q=quit, +/-=upsample, s=snap)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Quit] User pressed q")
                    break
                elif key == ord('+') or key == ord('='):
                    upsample = min(upsample + 1, 3)
                    print(f"[+] Upsample increased to {upsample}")
                elif key == ord('-'):
                    upsample = max(upsample - 1, 0)
                    print(f"[-] Upsample decreased to {upsample}")
                elif key == ord('s'):
                    snap_path = "/tmp/debug_face_snap.jpg"
                    cv2.imwrite(snap_path, frame)
                    print(f"[s] Snapshot saved to {snap_path}")

    except KeyboardInterrupt:
        print("\n[Stopped] Ctrl+C received")

    finally:
        cap.release()
        if gui:
            cv2.destroyAllWindows()
        print(f"\n[Done] Total detection rate for last window: check terminal above")


if __name__ == "__main__":
    main()
