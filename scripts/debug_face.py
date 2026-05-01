#!/usr/bin/env python3
"""
scripts/debug_face.py
=====================
Standalone face detection debugger — uses the same FaceDetector class
as the main assistant so camera opening is identical.

IMPORTANT: Stop the main container first, then run this:
    sudo docker compose down
    sudo docker run --rm -it --privileged \
        --device /dev/video0 --device /dev/video4 \
        -v $(pwd)/core:/app/core \
        -v $(pwd)/pipeline:/app/pipeline \
        -v $(pwd)/scripts:/app/scripts \
        -v $(pwd)/known_faces:/app/known_faces \
        chimcook-assistant \
        python3 /app/scripts/debug_face.py --no-gui

OR just run it directly from the stopped container:
    sudo docker compose down
    sudo docker compose run --rm assistant python3 /app/scripts/debug_face.py --no-gui

Text-only output (--no-gui): works inside Docker without a display.
"""

import argparse
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))


def parse_args():
    p = argparse.ArgumentParser(description="Chimcook face detection debugger")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index override (default: auto-detect)")
    p.add_argument("--camera-name", default="DV20",
                   help="Camera name substring to search in /sys (default: DV20)")
    p.add_argument("--upsample", type=int, default=1,
                   help="HOG upsample passes — 1=normal, 2=finds smaller faces (default: 1)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Downscale factor for detection (default: 0.5)")
    p.add_argument("--no-gui", action="store_true",
                   help="Text-only mode — no window (required inside Docker)")
    p.add_argument("--duration", type=int, default=30,
                   help="How many seconds to run (default: 30, 0=forever)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Chimcook Face Detection Debugger")
    print(f"{'='*60}")
    print(f"  Camera index   : {args.camera}")
    print(f"  Camera name    : '{args.camera_name}'")
    print(f"  HOG upsample   : {args.upsample}")
    print(f"  Scale factor   : {args.scale}")
    print(f"  GUI mode       : {'NO (text only)' if args.no_gui else 'YES'}")
    print(f"  Run duration   : {args.duration}s (0=forever)")
    print(f"{'='*60}\n")

    # ── Check /dev/video* devices first ──────────────────────────────────────
    import glob
    video_devs = sorted(glob.glob("/dev/video*"))
    if not video_devs:
        print("[ERROR] No /dev/video* devices found in container!")
        print("        Make sure docker-compose.yml has 'devices:' entries and")
        print("        the container is started with --privileged or the device mounted.")
        sys.exit(1)
    print(f"[Device check] Found: {', '.join(video_devs)}")

    # Check /sys for V4L2 camera names
    import os
    sysfs = "/sys/class/video4linux"
    if os.path.isdir(sysfs):
        print("[Device names]")
        for entry in sorted(os.listdir(sysfs)):
            name_file = os.path.join(sysfs, entry, "name")
            try:
                name = open(name_file).read().strip()
                print(f"  /dev/{entry} → '{name}'")
            except Exception:
                print(f"  /dev/{entry} → (unknown)")
    print()

    # ── Load FaceDetector (same class as the main assistant) ──────────────────
    try:
        from core.face_detector import FaceDetector, FaceDetectorConfig
    except ImportError as e:
        print(f"[ERROR] Cannot import FaceDetector: {e}")
        sys.exit(1)

    cfg = FaceDetectorConfig(
        camera_index=args.camera,
        camera_name=args.camera_name,
        detection_scale=args.scale,
        camera_width=640,
        camera_height=480,
        camera_fps=15,
        enhance_brightness=True,
        brightness_alpha=1.2,
        brightness_beta=20,
        enhance_contrast=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise=False,   # Off for debug speed (slow on CPU)
        sharpen=True,
        detection_interval_ms=0,  # No throttle in debug mode
    )

    detector = FaceDetector(cfg)

    print("[Starting camera...]")
    if not detector.start():
        print("\n[ERROR] Camera failed to open. Diagnose:")
        print("  1. Is the main assistant running? If so, stop it first:")
        print("     sudo docker compose down")
        print("  2. Check devices: ls -la /dev/video*")
        print("  3. Re-run with different --camera N (try 0, 1, 2, 4)")
        sys.exit(1)

    print("[Camera open] ✅ Running detection loop...\n")
    print(f"{'Frame':>8}  {'FPS':>5}  {'Faces':>5}  {'Hit%':>5}  {'Name'}")
    print("-" * 50)

    frame_n = 0
    hit_n = 0
    t_start = time.time()
    t_window = time.time()
    window_frames = 0
    window_hits = 0
    fps = 0.0

    gui = not args.no_gui
    if gui:
        try:
            import cv2 as _cv2
        except ImportError:
            print("[WARN] cv2 not available on host — falling back to --no-gui")
            gui = False

    try:
        while True:
            elapsed_total = time.time() - t_start
            if args.duration > 0 and elapsed_total >= args.duration:
                print(f"\n[Done] {args.duration}s elapsed.")
                break

            result = detector.process_frame()
            frame_n += 1
            window_frames += 1

            if result.face_count > 0:
                hit_n += 1
                window_hits += 1

            # ── FPS + report every second ─────────────────────────────────────
            now = time.time()
            if now - t_window >= 1.0:
                fps = window_frames / (now - t_window)
                hit_rate = window_hits / max(window_frames, 1) * 100
                name = result.recognized_name or ("FACE" if result.face_count > 0 else "none")
                symbol = "✅" if result.face_count > 0 else "❌"
                print(
                    f"{frame_n:>8}  {fps:>5.1f}  "
                    f"{result.face_count:>5}  {hit_rate:>4.0f}%  "
                    f"{symbol} {name}"
                )
                window_frames = 0
                window_hits = 0
                t_window = now

    except KeyboardInterrupt:
        print("\n[Stopped] Ctrl+C")
    finally:
        detector.stop()
        total_rate = hit_n / max(frame_n, 1) * 100
        print(f"\n{'='*50}")
        print(f"  Total frames   : {frame_n}")
        print(f"  Frames w/ face : {hit_n}  ({total_rate:.0f}%)")
        print(f"  Avg FPS        : {frame_n / max(time.time() - t_start, 1):.1f}")
        print(f"{'='*50}")

        # Diagnostic summary
        print()
        if total_rate >= 70:
            print("✅ Face detection is working well.")
            print("   If assistant still doesn't greet, check FACE_STABLE_FRAMES=2")
            print("   and GREET_COOLDOWN_SEC in pipeline/constants.py")
        elif total_rate >= 20:
            print("⚠️  Intermittent detection. Suggestions:")
            print("   • Improve lighting (face needs to be well-lit)")
            print("   • Sit closer to the camera (< 1.5m)")
            print("   • Try --upsample 2 for smaller/farther faces")
        else:
            print("❌ Very low detection rate. Suggestions:")
            print("   • Camera may be wrong index — try --camera 4")
            print("   • Try --scale 1.0 --upsample 2")
            print("   • Check lighting (dark room = HOG fails)")
            print("   • Verify camera isn't already in use")


if __name__ == "__main__":
    main()
