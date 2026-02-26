#!/usr/bin/env python3
"""
Test Hybrid VAD Live Capture: Uses Silero VAD for accurate speech detection
===========================================================================

This test uses the HYBRID VAD approach:
1. High energy threshold to filter obvious noise
2. Silero VAD for accurate speech detection (better than WebRTC)
3. Pre-buffer to capture speech start

Run from voice_assistant_v2:
    python tests/test_vad_live_capture.py

Options:
    --compare    Compare with/without pre-buffer
    --webrtc     Use WebRTC VAD instead of Hybrid (for comparison)
"""

import sys
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from collections import deque

# Config matching orchestrator
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# Higher thresholds for noise rejection
SILENCE_TIMEOUT_MS = 500  # End speech after this much silence
MIN_SPEECH_DURATION_MS = 400  # Minimum speech to process (filter short noise bursts)
MAX_SPEECH_DURATION_SEC = 15.0
ENERGY_THRESHOLD = 0.015  # Higher RMS threshold to filter background noise

# Derived
SILENCE_FRAMES = int(SILENCE_TIMEOUT_MS / FRAME_MS)  # 25 frames
MIN_SPEECH_FRAMES = int(MIN_SPEECH_DURATION_MS / FRAME_MS)  # 20 frames
MAX_FRAMES = int(MAX_SPEECH_DURATION_SEC * 1000 / FRAME_MS)

# Pre-buffer: keep last N frames to capture speech start
PRE_BUFFER_MS = 300  # 300ms of audio before VAD triggers
PRE_BUFFER_FRAMES = int(PRE_BUFFER_MS / FRAME_MS)  # 15 frames


def compute_rms(frame: np.ndarray) -> float:
    """Compute RMS energy of audio frame."""
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


def test_hybrid_vad_capture():
    """
    Test Hybrid VAD (Silero-based) with higher thresholds for noise rejection.
    """
    from core.audio_input import AudioInput, AudioInputConfig
    from core.audio_output import AudioOutput, AudioOutputConfig
    from core.vad_hybrid import HybridVAD, HybridVADConfig, HAS_SILERO
    
    print("=" * 60)
    print("  Hybrid VAD Capture Test (Silero + Pre-buffer)")
    print("=" * 60)
    print()
    print(f"Silero VAD available: {HAS_SILERO}")
    print()
    print(f"Settings:")
    print(f"  - Sample rate: {SAMPLE_RATE} Hz")
    print(f"  - Frame: {FRAME_MS} ms ({FRAME_SAMPLES} samples)")
    print(f"  - Energy threshold: {ENERGY_THRESHOLD} (high = less noise)")
    print(f"  - Silero threshold: 0.5 (speech probability)")
    print(f"  - Silence timeout: {SILENCE_TIMEOUT_MS} ms ({SILENCE_FRAMES} frames)")
    print(f"  - Min speech: {MIN_SPEECH_DURATION_MS} ms ({MIN_SPEECH_FRAMES} frames)")
    print(f"  - Pre-buffer: {PRE_BUFFER_MS} ms ({PRE_BUFFER_FRAMES} frames)")
    print()
    
    # Initialize components
    audio_input = AudioInput(AudioInputConfig(
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
    ))
    
    # Hybrid VAD with higher thresholds
    vad = HybridVAD(HybridVADConfig(
        sample_rate=SAMPLE_RATE,
        silero_threshold=0.5,      # Higher = less false positives
        energy_threshold=ENERGY_THRESHOLD,  # Higher = filter more noise
        hangover_frames=5,         # Keep speech active a bit after it ends
        smooth_window=3,           # Smooth decisions over 3 frames
        enable_rnnoise=False,      # Disable RNNoise for now (Silero is enough)
    ))
    
    # Start audio
    audio_input.start()
    
    # Calibration - measure noise floor
    print("Calibrating (2 seconds of silence)...")
    noise_rms_values = []
    cal_start = time.time()
    
    for frame in audio_input.frames():
        rms = compute_rms(frame)
        noise_rms_values.append(rms)
        if time.time() - cal_start > 2.0:
            break
    
    if noise_rms_values:
        noise_floor = np.mean(noise_rms_values)
        noise_p95 = np.percentile(noise_rms_values, 95)
        # Set energy threshold above noise floor, but cap it to avoid being too aggressive
        # Use p95 * 1.3 (not 2x) and cap at 0.05 max to ensure speech can still be detected
        adaptive_threshold = min(0.05, max(ENERGY_THRESHOLD, noise_p95 * 1.3))
        print(f"Noise floor: mean={noise_floor:.4f}, p95={noise_p95:.4f}")
        print(f"Adaptive energy threshold: {adaptive_threshold:.4f} (capped at 0.05)")
    else:
        adaptive_threshold = ENERGY_THRESHOLD
    
    print("Calibration complete!\n")
    
    # State variables
    audio_buffer = []
    in_speech = False
    silence_count = 0
    
    # Pre-buffer: ring buffer of recent frames
    pre_buffer = deque(maxlen=PRE_BUFFER_FRAMES)
    
    captured_segments = []
    stop_requested = [False]
    
    def wait_for_enter():
        try:
            input()
        except EOFError:
            pass
        stop_requested[0] = True
    
    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()
    
    print("🎤 Speak now! (Press Enter to stop)")
    print("   When you speak, I'll capture and play back what I heard.")
    print()
    
    try:
        for frame in audio_input.frames():
            if stop_requested[0]:
                break
            
            # Always add to pre-buffer (rolling window)
            pre_buffer.append(frame.copy())
            
            # Energy pre-filter (fast rejection of obvious silence/noise)
            rms = compute_rms(frame)
            
            if rms < adaptive_threshold:
                # Too quiet - definitely not speech
                is_speech = False
                confidence = 0.0
            else:
                # Run Silero VAD
                is_speech, confidence = vad.is_speech_with_confidence(frame)
            
            # Show status
            if is_speech:
                status = "🗣️  SPEECH"
            else:
                status = "   silence"
            
            bar_len = min(30, int(rms * 200))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            
            if in_speech:
                print(f"\r  {status}  [{bar}]  conf={confidence:.2f}  frames: {len(audio_buffer)}, silence: {silence_count}/{SILENCE_FRAMES}  ", end="", flush=True)
            else:
                print(f"\r  {status}  [{bar}]  conf={confidence:.2f}  rms={rms:.4f}  (waiting)         ", end="", flush=True)
            
            # === SPEECH COLLECTION with PRE-BUFFER ===
            if is_speech:
                if not in_speech:
                    # Speech just started!
                    in_speech = True
                    silence_count = 0
                    
                    # Include pre-buffer to capture the START of speech
                    audio_buffer = list(pre_buffer)  # Copy pre-buffer
                    print(f"\n\n✨ Speech started! (pre-buffer: {len(audio_buffer)} frames)")
                else:
                    audio_buffer.append(frame)
                
                silence_count = 0
                
                # Max duration check
                if len(audio_buffer) >= MAX_FRAMES:
                    print(f"\n⚠️ Max duration reached")
                    in_speech = False
                    captured_segments.append(audio_buffer.copy())
                    audio_buffer = []
            
            elif in_speech:
                # Silence during speech - still collect
                audio_buffer.append(frame)
                silence_count += 1
                
                # Speech ended?
                if silence_count >= SILENCE_FRAMES:
                    in_speech = False
                    
                    # Check minimum duration
                    if len(audio_buffer) >= MIN_SPEECH_FRAMES:
                        duration_ms = len(audio_buffer) * FRAME_MS
                        print(f"\n\n✅ Speech captured! ({duration_ms} ms, {len(audio_buffer)} frames)")
                        captured_segments.append(audio_buffer.copy())
                    else:
                        duration_ms = len(audio_buffer) * FRAME_MS
                        print(f"\n\n⚠️ Too short ({duration_ms} ms < {MIN_SPEECH_DURATION_MS} ms) - likely noise")
                    
                    audio_buffer = []
                    silence_count = 0
    
    except KeyboardInterrupt:
        print("\n\n(Stopped by user)")
    finally:
        audio_input.stop()
    
    # Show VAD stats
    stats = vad.get_stats()
    print("\n--- VAD Statistics ---")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Speech frames: {stats['speech_frames']}")
    print(f"  Energy filtered: {stats['energy_filtered']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.3f}")
    
    # Play back captured segments
    print("\n" + "=" * 60)
    print(f"  Captured {len(captured_segments)} speech segment(s)")
    print("=" * 60)
    
    if captured_segments:
        try:
            output = AudioOutput(AudioOutputConfig())
            
            for i, segment in enumerate(captured_segments):
                audio = np.concatenate(segment).astype(np.float32)
                duration = len(audio) / SAMPLE_RATE
                print(f"\n▶️  Playing segment {i+1}/{len(captured_segments)} ({duration:.2f}s)...")
                output.play(audio, SAMPLE_RATE, blocking=True)
                time.sleep(0.5)
            
            print("\n✅ Playback complete!")
        except Exception as e:
            print(f"\n❌ Playback error: {e}")
    else:
        print("\nNo speech captured.")
    
    print("\nDone.")


def test_webrtc_vad_capture():
    """
    Test WebRTC VAD for comparison (original approach).
    """
    from core.audio_input import AudioInput, AudioInputConfig
    from core.audio_output import AudioOutput, AudioOutputConfig
    from core.vad import VAD, VADConfig
    
    print("=" * 60)
    print("  WebRTC VAD Capture Test (with pre-buffer)")
    print("=" * 60)
    print()
    print(f"Settings:")
    print(f"  - Energy threshold: 0.10 (high)")
    print(f"  - Aggressiveness: 3 (most aggressive)")
    print()
    
    audio_input = AudioInput(AudioInputConfig(
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
    ))
    
    # WebRTC VAD with high thresholds
    vad = VAD(VADConfig(
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        aggressiveness=3,          # Most aggressive
        energy_threshold=0.10,     # Very high threshold
        hangover_frames=5,
        smooth_window=5,
    ))
    
    audio_input.start()
    
    # Calibration
    print("Calibrating (2 seconds of silence)...")
    calibration_frames = []
    cal_start = time.time()
    
    for frame in audio_input.frames():
        calibration_frames.append(frame)
        if time.time() - cal_start > 2.0:
            break
    
    if len(calibration_frames) > 20:
        vad.calibrate_noise_floor(calibration_frames)
    print("Calibration complete!\n")
    
    # State variables
    audio_buffer = []
    in_speech = False
    silence_count = 0
    pre_buffer = deque(maxlen=PRE_BUFFER_FRAMES)
    captured_segments = []
    stop_requested = [False]
    
    def wait_for_enter():
        try:
            input()
        except:
            pass
        stop_requested[0] = True
    
    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()
    
    print("🎤 Speak now! (Press Enter to stop)\n")
    
    try:
        for frame in audio_input.frames():
            if stop_requested[0]:
                break
            
            pre_buffer.append(frame.copy())
            is_speech = vad.is_speech(frame)
            rms = VAD.compute_rms(frame)
            
            status = "🗣️ " if is_speech else "   "
            bar = "█" * min(20, int(rms * 150))
            
            if in_speech:
                print(f"\r  {status} [{bar:<20}] frames: {len(audio_buffer)}", end="", flush=True)
            else:
                print(f"\r  {status} [{bar:<20}] rms={rms:.4f}", end="", flush=True)
            
            if is_speech:
                if not in_speech:
                    in_speech = True
                    silence_count = 0
                    audio_buffer = list(pre_buffer)
                    print(f"\n\n✨ Speech started!")
                else:
                    audio_buffer.append(frame)
                silence_count = 0
                
                if len(audio_buffer) >= MAX_FRAMES:
                    in_speech = False
                    captured_segments.append(audio_buffer.copy())
                    audio_buffer = []
            
            elif in_speech:
                audio_buffer.append(frame)
                silence_count += 1
                
                if silence_count >= SILENCE_FRAMES:
                    in_speech = False
                    if len(audio_buffer) >= MIN_SPEECH_FRAMES:
                        duration_ms = len(audio_buffer) * FRAME_MS
                        print(f"\n\n✅ Speech captured! ({duration_ms} ms)")
                        captured_segments.append(audio_buffer.copy())
                    else:
                        print(f"\n\n⚠️ Too short - likely noise")
                    audio_buffer = []
                    silence_count = 0
    
    except KeyboardInterrupt:
        print("\n\n(Stopped)")
    finally:
        audio_input.stop()
    
    # Play back
    print(f"\n\nCaptured {len(captured_segments)} segment(s)")
    
    if captured_segments:
        try:
            output = AudioOutput(AudioOutputConfig())
            for i, segment in enumerate(captured_segments):
                audio = np.concatenate(segment).astype(np.float32)
                duration = len(audio) / SAMPLE_RATE
                print(f"\n▶️  Playing {i+1} ({duration:.2f}s)...")
                output.play(audio, SAMPLE_RATE, blocking=True)
                time.sleep(0.5)
        except Exception as e:
            print(f"Playback error: {e}")
    
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test VAD live capture")
    parser.add_argument("--webrtc", action="store_true", 
                       help="Use WebRTC VAD instead of Hybrid")
    parser.add_argument("--compare", action="store_true",
                       help="Run both and compare")
    args = parser.parse_args()
    
    if args.webrtc:
        test_webrtc_vad_capture()
    elif args.compare:
        print("\n" + "=" * 60)
        print("  COMPARISON: Hybrid VAD vs WebRTC VAD")
        print("=" * 60)
        print("\nFirst, testing Hybrid VAD (Silero)...\n")
        test_hybrid_vad_capture()
        print("\n\n" + "-" * 60)
        print("\nNow testing WebRTC VAD...\n")
        test_webrtc_vad_capture()
    else:
        test_hybrid_vad_capture()
