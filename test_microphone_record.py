#!/usr/bin/env python3
"""
Microphone record test (input-only)
===================================

Records a short clip from a selected input device and saves it to a WAV file.

Usage:
  cd /home/minipc/voice_assistant_v2
  source venv/bin/activate
  python test_microphone_record.py

Optional:
  AUDIO_DEVICE=0 python test_microphone_record.py   # choose device index
  RECORD_SEC=5  python test_microphone_record.py    # seconds to record
"""

import os
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
RECORD_SEC = float(os.environ.get("RECORD_SEC", "3"))

_dev = os.environ.get("AUDIO_DEVICE", "").strip()
DEVICE = int(_dev) if _dev not in ("", "None") else None


def main() -> None:
    import sounddevice as sd
    import wave

    out_dir = Path(__file__).resolve().parent / "recordings" / "mic_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_path = out_dir / f"{ts}.wav"

    dev_info = None
    if DEVICE is not None:
        print(f"Input device: {DEVICE}")
        try:
            dev_info = sd.query_devices(DEVICE)
            print(f"Device info: {dev_info}")
        except Exception as e:
            print(f"Warning: could not query device {DEVICE}: {e}")
    else:
        print("Input device: default")

    # Some USB mics (e.g. M70) reject 16000 Hz directly and only allow 48000 Hz.
    # Try requested SAMPLE_RATE first, then fall back to device default samplerate.
    sr_to_try = SAMPLE_RATE
    n_samples = int(RECORD_SEC * sr_to_try)
    print(f"Recording {RECORD_SEC:.1f}s @ {sr_to_try} Hz ...")

    try:
        audio = sd.rec(
            frames=n_samples,
            samplerate=sr_to_try,
            channels=1,
            dtype="float32",
            device=DEVICE,
        )
        sd.wait()
    except Exception as e:
        fallback_sr = None
        if dev_info and isinstance(dev_info, dict):
            try:
                fallback_sr = int(dev_info.get("default_samplerate", 0) or 0)
            except Exception:
                fallback_sr = None
        if not fallback_sr:
            fallback_sr = 48000
        if fallback_sr == sr_to_try:
            raise
        print(f"\nRetrying with samplerate={fallback_sr} Hz (device default)...")
        sr_to_try = fallback_sr
        n_samples = int(RECORD_SEC * sr_to_try)
        audio = sd.rec(
            frames=n_samples,
            samplerate=sr_to_try,
            channels=1,
            dtype="float32",
            device=DEVICE,
        )
        sd.wait()

    audio = audio.reshape(-1).astype(np.float32)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f"Captured samples: {audio.size}")
    print(f"RMS:  {rms:.6f}")
    print(f"Peak: {peak:.6f}")

    # Save WAV (int16)
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr_to_try)
        f.writeframes(audio_i16.tobytes())

    print(f"Saved: {out_path} (sr={sr_to_try})")


if __name__ == "__main__":
    main()

