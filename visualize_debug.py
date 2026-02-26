#!/usr/bin/env python3
"""
Visual Debug Tool
=================
Shows real-time visualization of:
- Face detection (camera view with boxes)
- Audio level (bar graph)
- Wake word scores (all models)

Press 'q' to quit
"""

import numpy as np
import sounddevice as sd
import cv2
import time
import threading
from collections import deque

# ============================================================================
# Configuration
# ============================================================================
AUDIO_DEVICE = None  # Use default device (or set to specific ID: 4=laptop mic, 5=USB)
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms

# ============================================================================
# Shared State
# ============================================================================
audio_rms = 0.0
audio_level_history = deque(maxlen=50)
wake_scores = {}
face_count = 0
face_window_active = False
face_window_until = 0.0
lock = threading.Lock()

# ============================================================================
# Audio Thread
# ============================================================================
def audio_thread():
    global audio_rms, wake_scores, face_window_active
    
    print("Loading OpenWakeWord...")
    from openwakeword.model import Model
    model = Model(enable_speex_noise_suppression=False)
    print(f"Models loaded: {list(model.models.keys())}")
    
    print(f"Starting audio stream on device {AUDIO_DEVICE}...")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', device=AUDIO_DEVICE) as stream:
        while True:
            audio_data, _ = stream.read(CHUNK_SIZE)
            audio_int16 = audio_data.flatten()
            
            # Calculate RMS
            audio_float = audio_int16.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float ** 2))
            
            # Get predictions
            prediction = model.predict(audio_int16)
            
            with lock:
                audio_rms = rms
                audio_level_history.append(rms)
                wake_scores = dict(prediction)

# ============================================================================
# Face Detection Thread  
# ============================================================================
def face_thread():
    global face_count, face_window_active, face_window_until
    
    print("Loading face detection...")
    import face_recognition
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("Camera opened")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize for faster processing
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        
        with lock:
            face_count = len(face_locations)
            
            # Update face window
            if face_count == 1:
                face_window_until = time.time() + 5.0  # 5 second window
            
            face_window_active = time.time() < face_window_until
        
        # Draw boxes on frame
        for (top, right, bottom, left) in face_locations:
            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            color = (0, 255, 0) if face_count == 1 else (0, 165, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Show face count and window status
        status = f"Faces: {face_count}"
        if face_window_active:
            remaining = max(0, face_window_until - time.time())
            status += f" | Window: {remaining:.1f}s"
        else:
            status += " | Window: CLOSED"
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ============================================================================
# Main Visualization
# ============================================================================
def main():
    print("=" * 60)
    print("VISUAL DEBUG TOOL")
    print("=" * 60)
    print("This will show:")
    print("  - Camera window with face detection boxes")
    print("  - Terminal with audio level and wake word scores")
    print()
    print("Press 'q' in camera window to quit")
    print("=" * 60)
    
    # Start audio thread
    audio_t = threading.Thread(target=audio_thread, daemon=True)
    audio_t.start()
    
    # Wait for audio to initialize
    time.sleep(2)
    
    # Start face thread (runs in main for OpenCV window)
    face_t = threading.Thread(target=face_thread, daemon=True)
    face_t.start()
    
    # Main loop - print audio/wake word info
    print("\nAudio & Wake Word Monitor:")
    print("-" * 60)
    
    try:
        while True:
            with lock:
                rms = audio_rms
                scores = dict(wake_scores)
                faces = face_count
                window = face_window_active
            
            # Audio level bar
            bar_len = int(rms * 200)
            bar = "█" * min(bar_len, 40)
            
            # Wake word scores
            if scores:
                # Find max score
                max_name = max(scores, key=scores.get)
                max_score = scores[max_name]
                
                # Color code based on threshold
                score_color = "\033[92m" if max_score > 0.5 else "\033[93m" if max_score > 0.1 else "\033[90m"
                reset = "\033[0m"
                
                # Status indicators
                face_indicator = "👤" if faces == 1 else "❌"
                window_indicator = "🟢" if window else "🔴"
                
                # Format scores (show top 3)
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                scores_str = " | ".join([f"{k}: {v:.3f}" for k, v in sorted_scores])
                
                print(f"\r{face_indicator} {window_indicator} Audio: {rms:.4f} [{bar:40s}] | {score_color}{scores_str}{reset}    ", end="", flush=True)
                
                # Alert on detection
                if max_score > 0.5:
                    if window:
                        print(f"\n\n🎉 DETECTED: {max_name} (score: {max_score:.3f}) - WINDOW OPEN - WOULD ACTIVATE!\n")
                    else:
                        print(f"\n\n⚠️  DETECTED: {max_name} (score: {max_score:.3f}) - BUT WINDOW CLOSED\n")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == "__main__":
    main()
