#!/usr/bin/env python3
"""
Test to compare audio capture methods
"""

import numpy as np
import sounddevice as sd
import time

AUDIO_DEVICE = 5
SAMPLE_RATE = 16000
DURATION = 2  # seconds

print("=" * 60)
print("AUDIO FORMAT COMPARISON TEST")
print("=" * 60)

# Method 1: Direct int16 (like visualize_debug)
print("\n1. Recording as INT16 (like visualize_debug)...")
recording_int16 = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='int16',
    device=AUDIO_DEVICE
)
sd.wait()
print(f"   Shape: {recording_int16.shape}")
print(f"   Dtype: {recording_int16.dtype}")
print(f"   Min: {recording_int16.min()}, Max: {recording_int16.max()}")
print(f"   Mean: {recording_int16.mean():.2f}, Std: {recording_int16.std():.2f}")

# Method 2: Float32 then convert (like orchestrator)
print("\n2. Recording as FLOAT32 (like AudioInput)...")
recording_float = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    device=AUDIO_DEVICE
)
sd.wait()
print(f"   Shape: {recording_float.shape}")
print(f"   Dtype: {recording_float.dtype}")
print(f"   Min: {recording_float.min():.6f}, Max: {recording_float.max():.6f}")
print(f"   Mean: {recording_float.mean():.6f}, Std: {recording_float.std():.6f}")

# Convert float32 to int16 (like orchestrator does)
print("\n3. Converting float32 -> int16 (like orchestrator)...")
converted_int16 = (recording_float * 32767).astype(np.int16)
print(f"   Shape: {converted_int16.shape}")
print(f"   Dtype: {converted_int16.dtype}")
print(f"   Min: {converted_int16.min()}, Max: {converted_int16.max()}")
print(f"   Mean: {converted_int16.mean():.2f}, Std: {converted_int16.std():.2f}")

# Compare values
print("\n4. Comparison:")
print(f"   Direct int16 range: [{recording_int16.min()}, {recording_int16.max()}]")
print(f"   Converted int16 range: [{converted_int16.min()}, {converted_int16.max()}]")

# Test with OpenWakeWord
print("\n5. Testing both with OpenWakeWord...")
from openwakeword.model import Model
model = Model(enable_speex_noise_suppression=True)
print(f"   Models: {list(model.models.keys())}")

# Feed direct int16
print("\n   Feeding direct int16 recording...")
model.reset()
chunk_size = 1280
for i in range(0, len(recording_int16) - chunk_size, chunk_size):
    chunk = recording_int16[i:i+chunk_size].flatten()
    pred = model.predict(chunk)
    max_score = max(pred.values())
    if max_score > 0.1:
        print(f"   Frame {i//chunk_size}: {max(pred, key=pred.get)} = {max_score:.3f}")

# Feed converted int16
print("\n   Feeding converted int16 (from float32)...")
model.reset()
for i in range(0, len(converted_int16) - chunk_size, chunk_size):
    chunk = converted_int16[i:i+chunk_size].flatten()
    pred = model.predict(chunk)
    max_score = max(pred.values())
    if max_score > 0.1:
        print(f"   Frame {i//chunk_size}: {max(pred, key=pred.get)} = {max_score:.3f}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
