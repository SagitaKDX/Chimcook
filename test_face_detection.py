#!/usr/bin/env python3
"""
Test Face Detection with Visualization
Run this to verify face detection is working independently with visual feedback
"""

import sys
from pathlib import Path
import cv2
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.face_detector import FaceDetector, FaceDetectorConfig

print("=" * 60)
print("Face Detection Test with Visualization")
print("=" * 60)

# Create config with enhanced preprocessing
config = FaceDetectorConfig(
    known_faces_dir="known_faces",
    detection_interval_ms=100,  # Check every 100ms
    detection_scale=0.5,  # Start with 0.5x scale
    enable_talking_detection=True,
    camera_index=0,
    camera_width=640,
    camera_height=480,
    camera_fps=15,
    # Image preprocessing for better detection
    enhance_brightness=True,      # Enhance brightness
    brightness_alpha=1.2,         # Brightness multiplier (1.0 = no change)
    brightness_beta=20,           # Brightness offset
    enhance_contrast=True,        # Apply CLAHE contrast enhancement
    clahe_clip_limit=2.0,        # CLAHE clip limit
    clahe_tile_size=8,            # CLAHE tile size
    denoise=True,                 # Apply denoising
    sharpen=True,                 # Apply sharpening
)

print("\nInitializing face detector...")
detector = FaceDetector(config)

print("\nStarting camera...")
if not detector.start():
    print("ERROR: Could not start camera!")
    print("\nTroubleshooting:")
    print("1. Check if camera is connected: ls /dev/video*")
    print("2. Try different camera index: change camera_index in config")
    print("3. Check camera permissions")
    sys.exit(1)

print("Camera started successfully!")
print("\nImage preprocessing enabled:")
print("  ✓ Brightness enhancement")
print("  ✓ Contrast enhancement (CLAHE)")
print("  ✓ Denoising")
print("  ✓ Sharpening")
print("\nLooking for faces...")
print("Press 'q' to quit or Ctrl+C to stop\n")

frame_count = 0
face_detected_count = 0
fps_start_time = time.time()
fps_counter = 0
fps = 0.0

# Face window tracking (like in orchestrator)
face_window_until = 0.0
face_window_sec = 5.0

try:
    while True:
        # Get frame from camera
        frame = detector.get_frame()
        if frame is None:
            print("Warning: Failed to get frame from camera")
            time.sleep(0.1)
            continue
        
        # Process frame with debug visualization
        result = detector.process_frame(frame, debug=True)
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        # Update face window (like orchestrator logic)
        face_detected = result.face_count >= 1
        if face_detected:
            face_window_until = time.time() + face_window_sec
            face_detected_count += 1
        
        face_window_active = face_detected or (time.time() < face_window_until)
        
        # Get debug frame with annotations (preprocessed)
        processed_display = result.debug_frame if result.debug_frame is not None else frame.copy()
        
        # Create side-by-side comparison: Original | Preprocessed
        original_labeled = frame.copy()
        cv2.putText(original_labeled, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(processed_display, "Preprocessed (Enhanced)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine side-by-side
        display_frame = cv2.hconcat([original_labeled, processed_display])
        
        # Add semi-transparent overlay for status text (on preprocessed side)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (640, 0), (1280, 140), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
        
        # Status text overlay
        y_offset = 25
        line_height = 25
        
        # Face detection status
        if face_detected:
            face_text = f"Faces: {result.face_count} ✓ DETECTED"
            face_color = (0, 255, 0)  # Green
            if result.recognized_name:
                face_text += f" ({result.recognized_name})"
        else:
            face_text = f"Faces: {result.face_count} ✗ NOT DETECTED"
            face_color = (0, 0, 255)  # Red
        
        # Status text on preprocessed side (offset by 640 pixels)
        status_x = 650
        cv2.putText(display_frame, face_text, (status_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        y_offset += line_height
        
        # Face window status (like orchestrator)
        if face_window_active:
            remaining = max(0, int(face_window_until - time.time()))
            window_text = f"Face Window: ✓ ACTIVE ({remaining}s remaining)"
            window_color = (0, 255, 0)  # Green
        else:
            window_text = "Face Window: ✗ INACTIVE"
            window_color = (0, 0, 255)  # Red
        
        cv2.putText(display_frame, window_text, (status_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, window_color, 2)
        y_offset += line_height
        
        # Talking detection
        if result.is_talking:
            cv2.putText(display_frame, "TALKING ✓", (status_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
        
        # Face location info
        if result.face_location:
            top, right, bottom, left = result.face_location
            location_text = f"Location: ({left},{top}) to ({right},{bottom})"
            cv2.putText(display_frame, location_text, (status_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Statistics
        detection_rate = (face_detected_count / frame_count * 100) if frame_count > 0 else 0
        stats_text = f"Frames: {frame_count} | Detection Rate: {detection_rate:.1f}% | FPS: {fps:.1f}"
        cv2.putText(display_frame, stats_text, (status_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame (resize if needed for smaller screens)
        cv2.imshow("Face Detection Test - Original | Preprocessed (Press 'q' to quit)", display_frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n\nStopped by user")

finally:
    detector.stop()
    cv2.destroyAllWindows()
    print(f"\nSummary:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames with face detected: {face_detected_count}")
    if frame_count > 0:
        detection_rate = (face_detected_count / frame_count) * 100
        print(f"  Detection rate: {detection_rate:.1f}%")
    print("\nDone!")
