#!/usr/bin/env python3
"""
Test DeepFilterNet + Silero VAD Pipeline
=========================================

Step-by-step test of the noise cancellation + VAD pipeline:
1. DeepFilterNet v3 - State-of-the-art noise cancellation (removes noise, reverb, echo)
2. Silero VAD - Industry standard voice activity detection

Flow:
  Raw Audio → DeepFilterNet (denoise) → Silero VAD (detect speech) → Capture

Requirements:
    pip install deepfilternet torch torchaudio sounddevice

Run:
    python tests/test_deepfilter_silero.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import time
import threading
from collections import deque

# =============================================================================
# STEP 1: Test DeepFilterNet Installation
# =============================================================================

def test_deepfilternet():
    """Test DeepFilterNet installation and basic functionality."""
    print("=" * 60)
    print("  Step 1: Testing DeepFilterNet v3")
    print("=" * 60)
    print()
    
    try:
        # Fix for newer torchaudio versions (2.0+) that removed 'backend' module
        import sys
        import types
        
        # Check if torchaudio.backend needs to be mocked
        try:
            import torchaudio.backend
        except ImportError:
            # Create a dummy backend module for DeepFilterNet compatibility
            import torchaudio
            backend_module = types.ModuleType('torchaudio.backend')
            backend_module.utils = types.ModuleType('torchaudio.backend.utils')
            sys.modules['torchaudio.backend'] = backend_module
            sys.modules['torchaudio.backend.utils'] = backend_module.utils
        
        import torch
        from df.enhance import enhance, init_df
        print("✓ DeepFilterNet imported successfully")
        
        # Initialize model
        print("  Loading DeepFilterNet model (this may download ~100MB on first run)...")
        model, df_state, _ = init_df()
        print(f"✓ Model loaded!")
        print(f"  Sample rate: {df_state.sr()} Hz")
        print(f"  Frame size: {df_state.frame_size()} samples")
        print(f"  Hop size: {df_state.hop_size()} samples")
        
        # Test with dummy audio
        print("\n  Testing with dummy audio...")
        # Create noise at the correct sample rate (48kHz for DeepFilterNet)
        sr = df_state.sr()
        dummy_audio = np.random.randn(sr).astype(np.float32) * 0.1  # 1 second of noise
        enhanced = enhance(model, df_state, dummy_audio)
        print(f"✓ Enhancement works! Input shape: {dummy_audio.shape}, Output shape: {enhanced.shape}")
        
        return model, df_state
        
    except ImportError as e:
        print(f"✗ DeepFilterNet import error: {e}")
        print("\n  Install with: pip install deepfilternet")
        return None, None
    except Exception as e:
        print(f"✗ DeepFilterNet error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =============================================================================
# STEP 2: Test Silero VAD Installation
# =============================================================================

def test_silero_vad():
    """Test Silero VAD installation and basic functionality."""
    print("\n" + "=" * 60)
    print("  Step 2: Testing Silero VAD")
    print("=" * 60)
    print()
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
        
        # Load Silero VAD
        print("  Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        print("✓ Silero VAD loaded!")
        
        # Get utility functions
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils
        
        # Test with dummy audio (speech-like)
        # IMPORTANT: Silero VAD requires exactly 512 samples at 16kHz!
        print("\n  Testing with dummy audio (512 samples)...")
        t = np.linspace(0, 512/16000, 512, dtype=np.float32)
        dummy_speech = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone, 512 samples
        
        audio_tensor = torch.from_numpy(dummy_speech)
        
        # Get speech probability - must be exactly 512 samples!
        with torch.no_grad():
            speech_prob = model(audio_tensor.unsqueeze(0), 16000).item()
        print(f"✓ VAD works! Speech probability for sine wave: {speech_prob:.3f}")
        
        # Reset model state between sessions
        model.reset_states()
        print("✓ Model state reset works")
        
        return model, utils
        
    except ImportError as e:
        print(f"✗ Silero VAD not installed: {e}")
        print("\n  Install with: pip install torch torchaudio")
        return None, None
    except Exception as e:
        print(f"✗ Silero VAD error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =============================================================================
# STEP 3: Test Audio Input
# =============================================================================

def test_audio_input():
    """Test audio input capture."""
    print("\n" + "=" * 60)
    print("  Step 3: Testing Audio Input")
    print("=" * 60)
    print()
    
    try:
        import sounddevice as sd
        print("✓ sounddevice imported")
        
        # List devices
        print("\n  Available input devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"    [{i}] {dev['name']} (inputs: {dev['max_input_channels']})")
        
        default_input = sd.query_devices(kind='input')
        print(f"\n  Default input: {default_input['name']}")
        
        # Test capture
        print("\n  Recording 1 second of audio...")
        audio = sd.rec(16000, samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        
        rms = np.sqrt(np.mean(audio ** 2))
        print(f"✓ Audio captured! Shape: {audio.shape}, RMS: {rms:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio input error: {e}")
        return False


# =============================================================================
# STEP 4: Combined Pipeline Test (DeepFilterNet + Silero VAD)
# =============================================================================

def test_combined_pipeline(df_model, df_state, vad_model):
    """Test the combined DeepFilterNet + Silero VAD pipeline."""
    print("\n" + "=" * 60)
    print("  Step 4: Testing Combined Pipeline")
    print("=" * 60)
    print()
    
    if df_model is None or vad_model is None:
        print("✗ Missing models, skipping combined test")
        return False
    
    try:
        import torch
        import sounddevice as sd
        from df.enhance import enhance
        
        # Settings
        SAMPLE_RATE = 16000
        FRAME_MS = 20
        FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320 samples
        
        # DeepFilterNet needs larger chunks for processing
        DF_HOP_SIZE = df_state.hop_size()  # Usually 480 samples (30ms at 16kHz)
        DF_FRAME_SIZE = df_state.frame_size()  # Usually 960 samples (60ms at 16kHz)
        
        print(f"  DeepFilterNet hop size: {DF_HOP_SIZE} samples ({DF_HOP_SIZE/SAMPLE_RATE*1000:.1f}ms)")
        print(f"  DeepFilterNet frame size: {DF_FRAME_SIZE} samples ({DF_FRAME_SIZE/SAMPLE_RATE*1000:.1f}ms)")
        
        # We'll accumulate frames for DeepFilterNet processing
        df_buffer = np.zeros(0, dtype=np.float32)
        enhanced_buffer = np.zeros(0, dtype=np.float32)
        
        # Silero VAD settings
        SILERO_THRESHOLD = 0.5  # Speech probability threshold
        
        print("\n🎤 Recording for 5 seconds - speak to test!")
        print("   Watch the speech detection status...\n")
        
        # Record audio
        duration = 5
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        
        # Process in real-time simulation
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, int(duration * SAMPLE_RATE), FRAME_SAMPLES):
            # Wait for this frame to be recorded
            time.sleep(FRAME_MS / 1000)
            
            # Get frame
            end_idx = min(i + FRAME_SAMPLES, len(audio))
            frame = audio[i:end_idx].flatten()
            
            if len(frame) < FRAME_SAMPLES:
                break
            
            # Accumulate for DeepFilterNet
            df_buffer = np.concatenate([df_buffer, frame])
            
            # Process when we have enough samples
            if len(df_buffer) >= DF_FRAME_SIZE:
                # Enhance with DeepFilterNet
                enhanced = enhance(df_model, df_state, df_buffer)
                enhanced_buffer = np.concatenate([enhanced_buffer, enhanced])
                df_buffer = np.zeros(0, dtype=np.float32)
            
            # Get enhanced frame for VAD (use most recent enhanced audio)
            if len(enhanced_buffer) >= 512:  # Silero needs 512 samples
                vad_frame = enhanced_buffer[-512:]
                
                # Run Silero VAD
                audio_tensor = torch.from_numpy(vad_frame).float()
                with torch.no_grad():
                    speech_prob = vad_model(audio_tensor.unsqueeze(0), SAMPLE_RATE).item()
                
                is_speech = speech_prob >= SILERO_THRESHOLD
                total_frames += 1
                
                if is_speech:
                    speech_frames += 1
                    status = "🗣️  SPEECH"
                else:
                    status = "   silence"
                
                # Visual feedback
                rms = np.sqrt(np.mean(frame ** 2))
                bar_len = min(30, int(rms * 100))
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(f"\r  {status}  [{bar}]  prob={speech_prob:.2f}  rms={rms:.4f}  ", end="", flush=True)
        
        print(f"\n\n✓ Pipeline test complete!")
        print(f"  Total frames: {total_frames}")
        print(f"  Speech frames: {speech_frames} ({100*speech_frames/max(1,total_frames):.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Combined pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# STEP 5: Full Live Test with Speech Capture
# =============================================================================

def test_live_capture(df_model, df_state, vad_model, vad_utils):
    """Full live test: Listen → Detect Speech → Wait for Silence → Capture."""
    print("\n" + "=" * 60)
    print("  Step 5: Live Speech Capture Test")
    print("=" * 60)
    print()
    
    if df_model is None or vad_model is None:
        print("✗ Missing models, skipping live test")
        return
    
    try:
        import torch
        import sounddevice as sd
        from df.enhance import enhance
        
        # Settings
        SAMPLE_RATE = 16000
        FRAME_MS = 30  # 30ms frames (match DeepFilterNet hop size)
        FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples
        
        SILERO_THRESHOLD = 0.5
        SILENCE_TIMEOUT_MS = 500  # End speech after 500ms silence
        SILENCE_FRAMES = int(SILENCE_TIMEOUT_MS / FRAME_MS)
        MIN_SPEECH_MS = 300  # Minimum speech to process
        MIN_SPEECH_FRAMES = int(MIN_SPEECH_MS / FRAME_MS)
        
        PRE_BUFFER_MS = 300  # Capture 300ms before VAD triggers
        PRE_BUFFER_FRAMES = int(PRE_BUFFER_MS / FRAME_MS)
        
        print(f"  Frame size: {FRAME_SAMPLES} samples ({FRAME_MS}ms)")
        print(f"  Silence timeout: {SILENCE_TIMEOUT_MS}ms ({SILENCE_FRAMES} frames)")
        print(f"  Min speech: {MIN_SPEECH_MS}ms ({MIN_SPEECH_FRAMES} frames)")
        print(f"  Pre-buffer: {PRE_BUFFER_MS}ms ({PRE_BUFFER_FRAMES} frames)")
        
        # State
        pre_buffer = deque(maxlen=PRE_BUFFER_FRAMES)
        audio_buffer = []
        in_speech = False
        silence_count = 0
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
        
        print("\n🎤 Speak now! (Press Enter to stop)")
        print("   I'll capture your speech segments.\n")
        
        # Audio callback
        audio_queue = deque(maxlen=100)
        
        def audio_callback(indata, frames, time_info, status):
            audio_queue.append(indata.copy().flatten())
        
        # Start audio stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=FRAME_SAMPLES,
            callback=audio_callback
        )
        
        # DeepFilterNet buffer (accumulate for processing)
        df_buffer = np.zeros(0, dtype=np.float32)
        enhanced_cache = np.zeros(0, dtype=np.float32)
        
        with stream:
            while not stop_requested[0]:
                # Get frame from queue
                if not audio_queue:
                    time.sleep(0.001)
                    continue
                
                frame = audio_queue.popleft()
                
                # Add raw frame to pre-buffer
                pre_buffer.append(frame.copy())
                
                # Accumulate for DeepFilterNet
                df_buffer = np.concatenate([df_buffer, frame])
                
                # Process through DeepFilterNet when we have enough
                if len(df_buffer) >= df_state.frame_size():
                    enhanced = enhance(df_model, df_state, df_buffer)
                    enhanced_cache = np.concatenate([enhanced_cache, enhanced])
                    df_buffer = df_buffer[df_state.hop_size():]  # Keep overlap
                
                # Run Silero VAD on enhanced audio
                if len(enhanced_cache) >= 512:
                    vad_frame = enhanced_cache[-512:]
                    enhanced_cache = enhanced_cache[-512:]  # Keep only what we need
                    
                    audio_tensor = torch.from_numpy(vad_frame).float()
                    with torch.no_grad():
                        speech_prob = vad_model(audio_tensor.unsqueeze(0), SAMPLE_RATE).item()
                    
                    is_speech = speech_prob >= SILERO_THRESHOLD
                else:
                    is_speech = False
                    speech_prob = 0.0
                
                # Display
                rms = np.sqrt(np.mean(frame ** 2))
                if is_speech:
                    status = "🗣️  SPEECH"
                else:
                    status = "   silence"
                
                bar_len = min(30, int(rms * 100))
                bar = "█" * bar_len + "░" * (30 - bar_len)
                
                if in_speech:
                    print(f"\r  {status}  [{bar}]  prob={speech_prob:.2f}  frames={len(audio_buffer)}  silence={silence_count}/{SILENCE_FRAMES}  ", end="", flush=True)
                else:
                    print(f"\r  {status}  [{bar}]  prob={speech_prob:.2f}  (waiting)           ", end="", flush=True)
                
                # Speech collection with pre-buffer
                if is_speech:
                    if not in_speech:
                        # Speech started - include pre-buffer
                        in_speech = True
                        silence_count = 0
                        audio_buffer = list(pre_buffer)
                        print(f"\n\n✨ Speech started! (pre-buffer: {len(audio_buffer)} frames)")
                    else:
                        audio_buffer.append(frame)
                    silence_count = 0
                    
                elif in_speech:
                    # Silence during speech
                    audio_buffer.append(frame)
                    silence_count += 1
                    
                    if silence_count >= SILENCE_FRAMES:
                        # Speech ended
                        in_speech = False
                        
                        if len(audio_buffer) >= MIN_SPEECH_FRAMES:
                            duration_ms = len(audio_buffer) * FRAME_MS
                            print(f"\n\n✅ Speech captured! Duration: {duration_ms}ms ({len(audio_buffer)} frames)")
                            
                            # Save segment
                            segment = np.concatenate(audio_buffer)
                            captured_segments.append(segment)
                            
                            # Play back (optional)
                            print("   Playing back...")
                            sd.play(segment, SAMPLE_RATE)
                            sd.wait()
                            print("   Playback done!\n")
                        else:
                            print(f"\n\n⚠️ Speech too short ({len(audio_buffer)} frames), discarded\n")
                        
                        audio_buffer = []
                        silence_count = 0
        
        print(f"\n\n{'='*60}")
        print(f"  Captured {len(captured_segments)} speech segments")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"✗ Live capture error: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("🎙️  DeepFilterNet + Silero VAD Pipeline Test")
    print("=" * 60)
    print()
    
    # Step 1: Test DeepFilterNet
    df_model, df_state = test_deepfilternet()
    
    # Step 2: Test Silero VAD
    vad_model, vad_utils = test_silero_vad()
    
    # Step 3: Test Audio Input
    audio_ok = test_audio_input()
    
    if not audio_ok:
        print("\n✗ Audio input failed, cannot continue")
        return
    
    # Step 4: Test Combined Pipeline
    if df_model is not None and vad_model is not None:
        test_combined_pipeline(df_model, df_state, vad_model)
    
    # Step 5: Live Capture Test
    if df_model is not None and vad_model is not None:
        print("\n" + "-" * 60)
        response = input("Run live capture test? (y/n): ").strip().lower()
        if response == 'y':
            test_live_capture(df_model, df_state, vad_model, vad_utils)
    
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    main()
