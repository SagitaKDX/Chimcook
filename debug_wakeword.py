#!/usr/bin/env python3
"""
Deep debug script for wake word detection
"""

import numpy as np
import sounddevice as sd
import time

# USB Composite Device
AUDIO_DEVICE = 5
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms

print("=" * 60)
print("DEEP WAKE WORD DEBUG")
print("=" * 60)

# 1. Check device
print(f"\n1. Device {AUDIO_DEVICE} info:")
info = sd.query_devices(AUDIO_DEVICE)
print(f"   Name: {info['name']}")
print(f"   Max input channels: {info['max_input_channels']}")
print(f"   Default sample rate: {info['default_samplerate']}")

# 2. Test raw audio capture
print(f"\n2. Recording 2 seconds of audio...")
try:
    recording = sd.rec(
        int(2 * SAMPLE_RATE), 
        samplerate=SAMPLE_RATE, 
        channels=1, 
        dtype='int16',
        device=AUDIO_DEVICE
    )
    sd.wait()
    print(f"   Shape: {recording.shape}")
    print(f"   Dtype: {recording.dtype}")
    print(f"   Min: {recording.min()}, Max: {recording.max()}")
    print(f"   Mean: {recording.mean():.2f}, Std: {recording.std():.2f}")
    
    # Check if audio is silent (all near zero)
    if recording.std() < 100:
        print("\n   ⚠️  WARNING: Audio appears silent or very quiet!")
        print("   The microphone may not be working properly.")
    else:
        print("\n   ✓ Audio levels look OK")
except Exception as e:
    print(f"   ERROR: {e}")
    exit(1)

# 3. Load OpenWakeWord
print("\n3. Loading OpenWakeWord...")
from openwakeword.model import Model

# Try loading with different options
model = Model(enable_speex_noise_suppression=False)  # Disable noise suppression for debug
print(f"   Models: {list(model.models.keys())}")

# 4. Test with recorded audio
print("\n4. Testing with recorded audio...")
audio_flat = recording.flatten()

# Feed in chunks
print("   Feeding recorded audio to model...")
for i in range(0, len(audio_flat) - CHUNK_SIZE, CHUNK_SIZE):
    chunk = audio_flat[i:i+CHUNK_SIZE]
    prediction = model.predict(chunk)
    
    # Check if any score is non-zero
    max_score = max(prediction.values()) if prediction else 0
    if max_score > 0.01:
        print(f"   Frame {i//CHUNK_SIZE}: max={max_score:.4f} {prediction}")

print("   Done feeding recorded audio")

# 5. Test with synthetic audio (sine wave)
print("\n5. Testing with synthetic 440Hz sine wave...")
t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
sine_wave = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

# Reset model
model.reset()

for i in range(0, len(sine_wave) - CHUNK_SIZE, CHUNK_SIZE):
    chunk = sine_wave[i:i+CHUNK_SIZE]
    prediction = model.predict(chunk)
    max_score = max(prediction.values()) if prediction else 0
    if max_score > 0.01:
        print(f"   Frame {i//CHUNK_SIZE}: max={max_score:.4f}")

print("   Sine wave test complete (scores should be near 0 for non-speech)")

# 6. Check what format the model expects
print("\n6. Inspecting model internals...")
for name, m in model.models.items():
    print(f"   {name}:")
    if hasattr(m, 'model'):
        inputs = m.model.get_inputs()
        for inp in inputs:
            print(f"      Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")

# 7. Real-time test with visualization
print("\n7. Real-time test (5 seconds) - SPEAK NOW!")
print("   Showing audio level bars and predictions...")
print("-" * 60)

model.reset()
start_time = time.time()

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', device=AUDIO_DEVICE) as stream:
    while time.time() - start_time < 5:
        audio_data, _ = stream.read(CHUNK_SIZE)
        audio_int16 = audio_data.flatten()
        
        # Calculate level
        level = np.abs(audio_int16).mean()
        level_bar = "█" * int(level / 500)
        
        # Predict
        prediction = model.predict(audio_int16)
        
        # Get max score
        max_name = max(prediction, key=prediction.get)
        max_score = prediction[max_name]
        
        # Print status
        print(f"\rLevel: {level:6.0f} {level_bar:20s} | {max_name}: {max_score:.4f}", end="", flush=True)

print("\n" + "-" * 60)

# 8. Check OpenWakeWord version
print("\n8. Version info:")
try:
    import openwakeword
    print(f"   OpenWakeWord version: {openwakeword.__version__}")
except:
    print("   Could not get version")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
