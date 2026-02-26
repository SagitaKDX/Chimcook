#!/usr/bin/env python3
"""
Test VAD: Capture mic → show real-time speech/silence → play back captured audio
===============================================================================

Run from voice_assistant_v2:
    python test_vad_capture_play.py

- Records for 10 seconds (or press Enter to stop early).
- Shows live: SPEECH / silence, probability, level bar.
- At the end: plays back what was captured and prints a short summary.
"""

import sys
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# Config
SAMPLE_RATE = 16000
FRAME_MS = 20
RECORD_SEC = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# USB Composite = 0; M70 = 1. Override with AUDIO_DEVICE=1
import os
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE")
if AUDIO_DEVICE not in (None, ""):
    AUDIO_DEVICE = int(AUDIO_DEVICE)
else:
    AUDIO_DEVICE = 0  # USB Composite for VAD test


def main():
    from core.audio_input import AudioInput, AudioInputConfig
    from core.silero_vad import SileroVAD, HAS_SILERO
    from core.audio_output import AudioOutput, AudioOutputConfig, HAS_SOUNDDEVICE

    print("=" * 60)
    print("  VAD Test: Capture → Show Result → Play Back")
    print("=" * 60)
    print(f"\nSilero VAD: {'loaded' if HAS_SILERO else 'fallback (energy only)'}")
    print(f"Mic device: index {AUDIO_DEVICE} (USB Composite; override with AUDIO_DEVICE=1 for M70)")
    print(f"Record up to {RECORD_SEC} seconds. Press Enter to stop early.\n")

    # Audio input – use same device as main app so you hear your voice
    audio_config = AudioInputConfig(
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        device=AUDIO_DEVICE,
    )
    audio_input = AudioInput(audio_config)

    # Silero VAD
    vad = SileroVAD(
        sample_rate=SAMPLE_RATE,
        threshold=0.4,
        silence_limit_sec=0.9,
        speech_start_threshold=0.4,
    )

    captured_frames = []
    speech_frames = 0
    silence_frames = 0
    start_time = time.time()
    stop_requested = [False]  # list so inner thread can set it

    def wait_for_enter():
        try:
            input()
        except EOFError:
            pass
        stop_requested[0] = True

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    print("Recording... (press Enter to stop)")
    audio_input.start()

    try:
        while (time.time() - start_time) < RECORD_SEC and not stop_requested[0]:
            frame = audio_input.get_frame(timeout=0.05)
            if frame is None:
                continue

            captured_frames.append(frame.copy())
            prob, is_speech = vad.process(frame)
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

            if is_speech:
                speech_frames += 1
                label = "SPEECH"
            else:
                silence_frames += 1
                label = "silence"

            # Level bar (0..1 -> 0..20 chars)
            bar_len = min(30, int(rms * 200))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            prob_pct = int(prob * 100)
            line = f"\r  {label:8}  prob={prob_pct:3}%  [{bar}]  rms={rms:.4f}  "
            print(line, end="", flush=True)

    except KeyboardInterrupt:
        print("\n(Stopped by user)")
    finally:
        audio_input.stop()

    elapsed = time.time() - start_time
    print("\n")

    if not captured_frames:
        print("No audio captured.")
        return

    # Concatenate and convert for playback
    audio_float = np.concatenate(captured_frames).astype(np.float32)
    total_samples = len(audio_float)
    duration_sec = total_samples / SAMPLE_RATE

    # Summary
    total_frames = len(captured_frames)
    max_rms = float(np.max(np.sqrt(np.mean(np.array(captured_frames).astype(np.float64) ** 2, axis=1)))) if captured_frames else 0
    print("--- Result ---")
    print(f"  Captured: {duration_sec:.2f} s  ({total_samples} samples)")
    print(f"  Frames:   {total_frames}  (speech={speech_frames}, silence={silence_frames})")
    if total_frames > 0:
        pct_speech = 100 * speech_frames / total_frames
        print(f"  VAD:      {pct_speech:.1f}% speech, {100 - pct_speech:.1f}% silence")
    if max_rms < 0.005 and total_frames > 50:
        print("\n  ⚠️  Level very low (rms < 0.005). Check:")
        print("     - alsamixer: F4 for Capture, select correct device (M70), unmute and raise level")
        print("     - AUDIO_DEVICE=0 or AUDIO_DEVICE=1 to match your mic (python -m sounddevice to list)")
    print()

    # Play back
    if HAS_SOUNDDEVICE:
        print("Playing back captured audio...")
        try:
            output = AudioOutput(AudioOutputConfig())
            output.play(audio_float, SAMPLE_RATE, blocking=True)
            print("Playback done.")
        except Exception as e:
            print(f"Playback error: {e}")
    else:
        print("(sounddevice not available; skip playback)")

    print("\nDone.")


if __name__ == "__main__":
    main()
