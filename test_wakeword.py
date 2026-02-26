#!/usr/bin/env python3
"""
Test Wake Word Detection
Run this to debug OpenWakeWord issues
"""

import numpy as np
import sounddevice as sd
import scipy.signal
from openwakeword.model import Model

# Audio device: None = default, or specify ID (laptop: 4, mini PC USB: 5)
AUDIO_DEVICE = 5

print("=" * 50)
print("Wake Word Debug Test")
print("=" * 50)
print(f"Using audio device: {AUDIO_DEVICE or 'default'}")
if AUDIO_DEVICE is not None:
    print(f"Device info: {sd.query_devices(AUDIO_DEVICE)}")
else:
    print(f"Default input: {sd.query_devices(kind='input')['name']}")

# Try loading model
print("\n1. Loading OpenWakeWord model...")
try:
    # Try with wakeword_models (new API)
    model = Model(
        wakeword_models=["alexa"],
        enable_speex_noise_suppression=True,
    )
    print(f"   SUCCESS with wakeword_models=['alexa']")
except Exception as e:
    print(f"   FAILED with wakeword_models: {e}")
    try:
        # Try without specifying model (loads all)
        model = Model(enable_speex_noise_suppression=True)
        print(f"   SUCCESS with default (all models)")
    except Exception as e2:
        print(f"   FAILED with default: {e2}")
        exit(1)

print(f"\n2. Available models: {list(model.models.keys())}")

# Test audio capture
print("\n3. Testing audio capture...")
PROCESS_SR = 16000          # what OpenWakeWord expects
INPUT_SR = PROCESS_SR       # what sounddevice will try to use
CHUNK_MS = 80

try:
    if AUDIO_DEVICE is not None:
        dev = sd.query_devices(AUDIO_DEVICE)
        dev_default_sr = int(dev.get("default_samplerate", 0) or 0) if isinstance(dev, dict) else 0
        # If device has a different default (e.g. 48000 for M70), use that for capture to avoid paInvalidSampleRate
        if dev_default_sr and dev_default_sr != PROCESS_SR:
            INPUT_SR = dev_default_sr
except Exception:
    pass

CHUNK_IN = int(INPUT_SR * CHUNK_MS / 1000)      # samples at input rate
CHUNK_PROC = int(PROCESS_SR * CHUNK_MS / 1000)  # samples at 16k (1280)

try:
    with sd.InputStream(samplerate=INPUT_SR, channels=1, dtype='int16', device=AUDIO_DEVICE) as stream:
        print(f"   Audio stream OK (device {AUDIO_DEVICE}, sr={INPUT_SR})")
except Exception as e:
    print(f"   Audio stream FAILED: {e}")
    exit(1)

# Test predictions
print("\n4. Testing predictions (speak 'Alexa' now)...")
print("   Press Ctrl+C to stop\n")

frame_count = 0
try:
    with sd.InputStream(samplerate=INPUT_SR, channels=1, dtype='int16', device=AUDIO_DEVICE) as stream:
        while True:
            audio_data, overflowed = stream.read(CHUNK_IN)
            audio_int16_in = audio_data.flatten()
            
            # Convert to float [-1, 1] at INPUT_SR
            audio_float_in = audio_int16_in.astype(np.float32) / 32768.0
            # Resample to 16k for OpenWakeWord
            if INPUT_SR != PROCESS_SR:
                audio_float = scipy.signal.resample_poly(
                    audio_float_in, up=PROCESS_SR, down=INPUT_SR
                ).astype(np.float32)
            else:
                audio_float = audio_float_in
            # Ensure exact 80ms chunk at 16k
            if audio_float.size != CHUNK_PROC:
                if audio_float.size > CHUNK_PROC:
                    audio_float = audio_float[:CHUNK_PROC]
                else:
                    audio_float = np.pad(audio_float, (0, CHUNK_PROC - audio_float.size))
            # Level at 16k
            rms = float(np.sqrt(np.mean(audio_float ** 2)))
            
            # Run prediction (int16 at 16k)
            audio_int16_proc = (np.clip(audio_float, -1.0, 1.0) * 32767.0).astype(np.int16)
            prediction = model.predict(audio_int16_proc)
            
            frame_count += 1
            
            # Show all predictions
            if frame_count % 5 == 0:  # Every ~400ms
                pred_str = " | ".join([f"{k}: {v:.3f}" for k, v in prediction.items()])
                print(f"\rRMS: {rms:.4f} | {pred_str}      ", end="", flush=True)
            
            # Check for detection
            for name, score in prediction.items():
                if score > 0.5:
                    print(f"\n\n*** DETECTED: {name} (score: {score:.3f}) ***\n")

except KeyboardInterrupt:
    print("\n\nStopped.")

print("\nDone!")
