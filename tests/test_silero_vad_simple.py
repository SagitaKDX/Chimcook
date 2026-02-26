#!/usr/bin/env python3
"""
Test Silero VAD + noisereduce Pipeline
======================================

Flow:
1. Silero VAD detects speech start → start capturing
2. Silero VAD detects silence → stop capturing  
3. noisereduce cleans the audio
4. Playback the cleaned audio

Run:
    cd voice_assistant_v2
    source venv/bin/activate
    python tests/test_silero_vad_simple.py
"""

import sys
import time
import threading
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
FRAME_MS = 32  # Silero works best with 32ms frames (512 samples at 16kHz)
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 512 samples

# VAD settings
SILERO_THRESHOLD = 0.5  # Speech probability threshold
SILENCE_TIMEOUT_MS = 600  # End speech after this much silence
MIN_SPEECH_MS = 300  # Minimum speech duration to keep

# Derived
SILENCE_FRAMES = SILENCE_TIMEOUT_MS // FRAME_MS  # ~19 frames
MIN_SPEECH_FRAMES = MIN_SPEECH_MS // FRAME_MS  # ~9 frames


def load_silero_vad():
    """Load Silero VAD model."""
    print("Loading Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    print("✓ Silero VAD loaded")
    return model


def silero_vad_predict(model, audio_chunk: np.ndarray) -> float:
    """
    Get speech probability from Silero VAD.
    
    Args:
        model: Silero VAD model
        audio_chunk: float32 audio, must be 512 samples for 16kHz
        
    Returns:
        Speech probability (0.0 to 1.0)
    """
    # Silero expects 512 samples at 16kHz (32ms)
    if len(audio_chunk) != 512:
        # Pad or truncate
        if len(audio_chunk) < 512:
            audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[:512]
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_chunk).float()
    
    # Get prediction
    with torch.no_grad():
        speech_prob = model(audio_tensor.unsqueeze(0), SAMPLE_RATE).item()
    
    return speech_prob


def apply_noise_reduction(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply noise reduction to recorded audio.
    
    Args:
        audio: float32 audio array
        sample_rate: sample rate
        
    Returns:
        Cleaned audio
    """
    try:
        import noisereduce as nr
        
        cleaned = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=False,  # Better for non-stationary noise
            prop_decrease=0.8,  # Don't be too aggressive
        )
        
        return cleaned.astype(np.float32)
        
    except ImportError:
        print("  ⚠ noisereduce not installed, skipping")
        return audio


def test_vad_pipeline():
    """Test the full VAD pipeline with live audio."""
    from core.audio_input import AudioInput, AudioInputConfig
    
    print("=" * 60)
    print("  Silero VAD + Noisereduce Pipeline Test")
    print("=" * 60)
    print()
    print(f"Settings:")
    print(f"  - Sample rate: {SAMPLE_RATE} Hz")
    print(f"  - Frame: {FRAME_MS} ms ({FRAME_SAMPLES} samples)")
    print(f"  - Silero threshold: {SILERO_THRESHOLD}")
    print(f"  - Silence timeout: {SILENCE_TIMEOUT_MS} ms ({SILENCE_FRAMES} frames)")
    print(f"  - Min speech: {MIN_SPEECH_MS} ms ({MIN_SPEECH_FRAMES} frames)")
    print()
    
    # Load model
    model = load_silero_vad()
    
    # Initialize audio input with 32ms frames (512 samples)
    audio_input = AudioInput(AudioInputConfig(
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
    ))
    audio_input.start()
    
    # State variables
    audio_buffer = []
    in_speech = False
    silence_count = 0
    captured_segments = []
    
    # Stop signal
    stop_requested = [False]
    
    def wait_for_enter():
        try:
            input()
        except EOFError:
            pass
        stop_requested[0] = True
    
    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()
    
    print()
    print("🎤 Speak now! Press Enter to stop.")
    print("   I'll detect when you start/stop speaking.")
    print()
    
    try:
        for frame in audio_input.frames():
            if stop_requested[0]:
                break
            
            # Get speech probability from Silero
            speech_prob = silero_vad_predict(model, frame)
            is_speech = speech_prob >= SILERO_THRESHOLD
            
            # Visual feedback
            bar_len = int(speech_prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            
            if in_speech:
                status = f"🔴 Recording ({len(audio_buffer)} frames, silence: {silence_count}/{SILENCE_FRAMES})"
            elif is_speech:
                status = "🗣️  Speech detected!"
            else:
                status = "   Waiting..."
            
            print(f"\r  [{bar}] {speech_prob:.2f}  {status}          ", end="", flush=True)
            
            # === SPEECH STATE MACHINE ===
            if is_speech:
                if not in_speech:
                    # Speech just started
                    in_speech = True
                    audio_buffer = [frame.copy()]
                    silence_count = 0
                    print(f"\n\n✨ Speech started!")
                else:
                    # Continue recording
                    audio_buffer.append(frame.copy())
                    silence_count = 0
            
            elif in_speech:
                # Still in speech state but current frame is silence
                audio_buffer.append(frame.copy())
                silence_count += 1
                
                # Check if silence timeout reached
                if silence_count >= SILENCE_FRAMES:
                    # Speech ended!
                    in_speech = False
                    
                    # Check minimum duration
                    speech_frames = len(audio_buffer) - silence_count
                    if speech_frames >= MIN_SPEECH_FRAMES:
                        print(f"\n\n✅ Speech ended! Captured {len(audio_buffer)} frames ({len(audio_buffer) * FRAME_MS}ms)")
                        
                        # Concatenate and store
                        audio_data = np.concatenate(audio_buffer)
                        captured_segments.append(audio_data)
                        
                        print(f"   Total segments captured: {len(captured_segments)}")
                    else:
                        print(f"\n\n⚠️ Too short ({speech_frames} frames), discarded")
                    
                    audio_buffer = []
                    silence_count = 0
                    
                    # Reset Silero state for next utterance
                    model.reset_states()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        audio_input.stop()
    
    print()
    print("=" * 60)
    
    # Process captured segments
    if captured_segments:
        print(f"\n📊 Processing {len(captured_segments)} captured segment(s)...")
        
        try:
            from core.audio_output import AudioOutput, AudioOutputConfig
            audio_output = AudioOutput(AudioOutputConfig())
        except:
            audio_output = None
        
        for i, segment in enumerate(captured_segments):
            print(f"\nSegment {i+1}: {len(segment) / SAMPLE_RATE:.2f}s")
            
            # Play ORIGINAL first
            if audio_output:
                print("  🔊 Playing ORIGINAL audio...")
                audio_int16 = (segment * 32767).astype(np.int16)
                audio_output.play(audio_int16, SAMPLE_RATE)
                time.sleep(len(segment) / SAMPLE_RATE + 0.5)  # Wait for playback
            
            # Apply noise reduction
            print("  🔧 Applying noise reduction...")
            cleaned = apply_noise_reduction(segment, SAMPLE_RATE)
            
            # Play CLEANED
            if audio_output:
                print("  🔊 Playing CLEANED audio...")
                audio_int16 = (cleaned * 32767).astype(np.int16)
                audio_output.play(audio_int16, SAMPLE_RATE)
                time.sleep(len(cleaned) / SAMPLE_RATE + 0.5)
    else:
        print("\n❌ No speech captured")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_vad_pipeline()
