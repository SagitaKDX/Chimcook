#!/usr/bin/env python3
"""
Visualize Face Detection + Wake Word Detection
Shows camera feed with face detection overlays and wake word status
"""

import sys
from pathlib import Path
import numpy as np
import time
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.face_detector import FaceDetector, FaceDetectorConfig
from openwakeword.model import Model as WakeWordModel
import sounddevice as sd
import warnings

print("=" * 60)
print("Face Detection + Wake Word Visualization")
print("=" * 60)

# Initialize face detector
print("\n[1/2] Initializing Face Detector...")
face_config = FaceDetectorConfig(
    known_faces_dir="known_faces",
    detection_interval_ms=100,  # Check every 100ms
    detection_scale=0.5,
    enable_talking_detection=True,
    camera_index=0,
    camera_width=640,
    camera_height=480,
    camera_fps=15,
)

face_detector = FaceDetector(face_config)

if not face_detector.start():
    print("ERROR: Could not start camera!")
    print("Troubleshooting:")
    print("1. Check camera: ls /dev/video*")
    print("2. Check permissions")
    sys.exit(1)

print("✓ Camera started")

# Initialize wake word detector
print("\n[2/2] Initializing Wake Word Detector...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    wake_model = WakeWordModel(
        wakeword_models=["alexa"],
        enable_speex_noise_suppression=True,
    )

wake_word_name = "alexa"
wake_threshold = 0.3
wake_buffer = []
wake_buffer_frames_needed = 4  # 4 frames = 1280 samples

print(f"✓ Wake word model loaded: {wake_word_name}")
print(f"  Threshold: {wake_threshold}")

# Audio settings
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

print("\n" + "=" * 60)
print("Starting visualization...")
print("Press 'q' to quit")
print("=" * 60)

# Face window tracking
face_window_until = 0.0
face_window_sec = 5.0
face_greeted = False

# Audio callback for wake word
audio_queue = []

def audio_callback(indata, frames, time_info, status):
    """Capture audio for wake word detection."""
    if status:
        print(f"Audio status: {status}")
    audio_queue.append(indata.copy())

# Start audio stream
print("\nStarting audio capture...")
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    blocksize=FRAME_SAMPLES,
    callback=audio_callback,
)
stream.start()

frame_count = 0
fps_start_time = time.time()
fps_counter = 0

try:
    while True:
        # Get camera frame
        frame = face_detector.get_frame()
        if frame is None:
            print("Warning: Failed to get frame")
            time.sleep(0.1)
            continue
        
        # Process face detection
        face_result = face_detector.process_frame(frame, debug=True)
        
        # Update face window
        face_detected = face_result.face_count >= 1
        if face_detected:
            face_window_until = time.time() + face_window_sec
        
        face_window_active = face_detected or (time.time() < face_window_until)
        
        # Get debug frame with annotations
        display_frame = face_result.debug_frame if face_result.debug_frame is not None else frame.copy()
        
        # Add status overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 120), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
        
        # Status text
        y_offset = 25
        line_height = 25
        
        # Face status
        if face_detected:
            face_text = f"Faces: {face_result.face_count} ✓ DETECTED"
            color = (0, 255, 0)  # Green
            if face_result.recognized_name:
                face_text += f" ({face_result.recognized_name})"
        else:
            face_text = f"Faces: {face_result.face_count} ✗ NOT DETECTED"
            color = (0, 0, 255)  # Red
        
        cv2.putText(display_frame, face_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
        
        # Face window status
        if face_window_active:
            window_text = f"Face Window: ✓ ACTIVE ({max(0, int(face_window_until - time.time()))}s)"
            window_color = (0, 255, 0)
        else:
            window_text = "Face Window: ✗ INACTIVE"
            window_color = (0, 0, 255)
        
        cv2.putText(display_frame, window_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, window_color, 2)
        y_offset += line_height
        
        # Talking detection
        if face_result.is_talking:
            cv2.putText(display_frame, "TALKING ✓", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
        
        # Process wake word audio
        wake_score = 0.0
        if len(audio_queue) > 0:
            # Get audio frames
            while len(audio_queue) > 0:
                audio_frame = audio_queue.pop(0)
                wake_buffer.append(audio_frame.flatten())
            
            # When we have enough frames, check wake word
            if len(wake_buffer) >= wake_buffer_frames_needed:
                combined_audio = np.concatenate(wake_buffer[:wake_buffer_frames_needed])
                wake_buffer = wake_buffer[wake_buffer_frames_needed:]
                
                # Convert to int16
                audio_int16 = (combined_audio * 32767).astype(np.int16)
                
                # Predict
                prediction = wake_model.predict(audio_int16)
                wake_score = prediction.get(wake_word_name, 0.0)
        
        # Wake word status
        if wake_score > wake_threshold:
            wake_text = f"Wake Word: ✓ DETECTED! ({wake_score:.3f})"
            wake_color = (0, 255, 0)
        else:
            wake_text = f"Wake Word: {wake_word_name} ({wake_score:.3f})"
            wake_color = (255, 255, 255)
        
        cv2.putText(display_frame, wake_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, wake_color, 2)
        
        # FPS counter
        frame_count += 1
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (500, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Face Detection + Wake Word Visualization", display_frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    stream.stop()
    stream.close()
    face_detector.stop()
    cv2.destroyAllWindows()
    print("\nVisualization stopped. Goodbye!")
