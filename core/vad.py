"""
Voice Assistant v2 - Voice Activity Detection Module
====================================================

Step 4: Detect when someone is speaking vs silence.

Uses WebRTC VAD with enhancements:
1. Energy pre-filter (noise gate) - skip VAD for quiet frames
2. Smoothing (anti-flutter) - majority vote over window
3. Hangover - keep speech active slightly after it ends

WebRTC VAD Aggressiveness:
- 0: Least aggressive (catches quiet speech, more false positives)
- 1: Moderate
- 2: Aggressive (good balance)
- 3: Most aggressive (fewer false positives, may miss quiet speech)

For crowded places: Use 2-3
"""

from dataclasses import dataclass
from typing import Optional, Dict
from collections import deque
import numpy as np

# Import WebRTC VAD
try:
    import webrtcvad
    HAS_WEBRTC_VAD = True
except ImportError:
    HAS_WEBRTC_VAD = False
    print("[VAD] Warning: webrtcvad not installed, using energy-only detection")


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    sample_rate: int = 16000        # Must be 8000, 16000, 32000, or 48000
    frame_ms: int = 20              # Must be 10, 20, or 30 ms
    aggressiveness: int = 2         # 0-3, higher = fewer false positives
    energy_threshold: float = 0.01  # RMS threshold for pre-filter
    smooth_window: int = 5          # Frames to smooth decisions
    hangover_frames: int = 8        # Keep speech active after it ends


class VAD:
    """
    Voice Activity Detection with noise robustness.
    
    Pipeline:
    1. Energy check (skip WebRTC if clearly silence)
    2. WebRTC VAD (actual speech detection)  
    3. Smoothing (majority vote over window)
    4. Hangover (extend speech detection slightly)
    
    Usage:
        vad = VAD(VADConfig())
        
        for frame in audio_frames:
            if vad.is_speech(frame):
                # Frame contains speech
                process_speech(frame)
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        
        # Validate config
        if self.config.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000, got {self.config.sample_rate}")
        if self.config.frame_ms not in (10, 20, 30):
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms, got {self.config.frame_ms}")
        if not 0 <= self.config.aggressiveness <= 3:
            raise ValueError(f"Aggressiveness must be 0-3, got {self.config.aggressiveness}")
        
        # Calculate frame size
        self.frame_samples = int(self.config.sample_rate * self.config.frame_ms / 1000)
        
        # Initialize WebRTC VAD
        if HAS_WEBRTC_VAD:
            self._vad = webrtcvad.Vad(self.config.aggressiveness)
        else:
            self._vad = None
        
        # Smoothing state
        self._history: deque = deque(maxlen=self.config.smooth_window)
        
        # Hangover state
        self._hangover_counter = 0
        self._last_raw_speech = False
        
        # Statistics
        self._stats = {
            "frames_processed": 0,
            "speech_frames": 0,
            "energy_filtered": 0,
        }
    
    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Determine if frame contains speech (with smoothing + hangover).
        
        Args:
            frame: np.ndarray float32, shape (frame_samples,)
            
        Returns:
            True if speech detected, False otherwise
        """
        self._stats["frames_processed"] += 1
        
        # Get raw detection
        raw = self.is_speech_raw(frame)
        self._last_raw_speech = raw
        
        # Hangover logic: extend speech detection
        if raw:
            self._hangover_counter = self.config.hangover_frames
        elif self._hangover_counter > 0:
            self._hangover_counter -= 1
            raw = True  # Keep as speech during hangover
        
        # Add to history for smoothing
        self._history.append(raw)
        
        # Smoothing: majority vote
        if len(self._history) < 2:
            result = raw
        else:
            votes = sum(self._history)
            result = votes > len(self._history) / 2
        
        if result:
            self._stats["speech_frames"] += 1
        
        return result
    
    def is_speech_raw(self, frame: np.ndarray) -> bool:
        """
        Raw VAD decision (without smoothing or hangover).
        
        Useful for debugging or when you need instant decisions.
        
        Args:
            frame: np.ndarray float32, shape (frame_samples,)
            
        Returns:
            True if speech detected
        """
        # Validate frame size (allow slight variation)
        if abs(len(frame) - self.frame_samples) > 10:
            # Resize if needed
            if len(frame) > self.frame_samples:
                frame = frame[:self.frame_samples]
            else:
                frame = np.pad(frame, (0, self.frame_samples - len(frame)))
        
        # Step 1: Energy pre-filter
        rms = self.compute_rms(frame)
        if rms < self.config.energy_threshold:
            self._stats["energy_filtered"] += 1
            return False  # Too quiet, skip VAD
        
        # Step 2: WebRTC VAD (if available)
        if self._vad is not None:
            try:
                pcm_bytes = self.float_to_pcm(frame)
                return self._vad.is_speech(pcm_bytes, self.config.sample_rate)
            except Exception as e:
                # Fallback to energy-only on error
                pass
        
        # Fallback: energy-based detection (if no WebRTC)
        # Higher threshold for speech
        return rms > self.config.energy_threshold * 2
    
    def reset(self) -> None:
        """Reset smoothing history and hangover state."""
        self._history.clear()
        self._hangover_counter = 0
        self._last_raw_speech = False
    
    def get_stats(self) -> Dict:
        """
        Get VAD statistics.
        
        Returns:
            Dict with processing statistics
        """
        total = self._stats["frames_processed"]
        speech = self._stats["speech_frames"]
        
        return {
            **self._stats,
            "speech_ratio": speech / total if total > 0 else 0,
            "energy_filter_ratio": self._stats["energy_filtered"] / total if total > 0 else 0,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "frames_processed": 0,
            "speech_frames": 0,
            "energy_filtered": 0,
        }
    
    def calibrate_noise_floor(self, frames: list) -> float:
        """
        Calibrate energy threshold based on background noise level.
        
        Call this with a few seconds of background audio (no speech)
        to automatically set the energy threshold above your noise floor.
        
        Args:
            frames: List of audio frames (background noise, no speech)
            
        Returns:
            New energy threshold value
        """
        if not frames:
            return self.config.energy_threshold
        
        # Compute RMS for all frames
        energies = [self.compute_rms(f) for f in frames]
        
        # Calculate noise statistics using percentiles (ignore outlier spikes)
        mean_energy = np.mean(energies)
        p90_energy = np.percentile(energies, 90)  # 90th percentile (ignores loud spikes)
        p95_energy = np.percentile(energies, 95)  # 95th percentile
        
        # Set threshold based on percentiles (more robust than max)
        # Use 95th percentile + margin, or 2x mean (reduced from 3x for better sensitivity)
        new_threshold = max(p95_energy * 1.2, mean_energy * 2.0)
        
        # Apply 20% reduction to make VAD more sensitive to speech
        # (user reported needing to boost mic 20% after calibration)
        new_threshold = new_threshold * 0.8
        
        # Ensure threshold is reasonable: min 0.03, max 0.12 (lowered for better sensitivity)
        new_threshold = max(0.03, min(0.12, new_threshold))
        
        print(f"      [VAD] Noise floor: mean={mean_energy:.4f}, p90={p90_energy:.4f}, p95={p95_energy:.4f}")
        print(f"      [VAD] Energy threshold: {self.config.energy_threshold:.4f} → {new_threshold:.4f}")
        
        self.config.energy_threshold = new_threshold
        return new_threshold
    
    @staticmethod
    def compute_rms(frame: np.ndarray) -> float:
        """
        Compute RMS (Root Mean Square) energy of audio frame.
        
        Args:
            frame: Audio samples
            
        Returns:
            RMS value (0.0 to ~1.0 for normalized audio)
        """
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    
    @staticmethod
    def float_to_pcm(frame: np.ndarray) -> bytes:
        """
        Convert float32 [-1,1] to int16 PCM bytes for WebRTC VAD.
        
        Args:
            frame: Float32 audio samples in range [-1, 1]
            
        Returns:
            Bytes in int16 PCM format
        """
        clipped = np.clip(frame, -1.0, 1.0)
        int16 = (clipped * 32767).astype(np.int16)
        return int16.tobytes()
    
    @staticmethod
    def pcm_to_float(pcm_bytes: bytes) -> np.ndarray:
        """
        Convert int16 PCM bytes to float32 array.
        
        Args:
            pcm_bytes: Bytes in int16 PCM format
            
        Returns:
            Float32 numpy array in range [-1, 1]
        """
        int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16.astype(np.float32) / 32767.0


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test VAD module."""
    import time
    
    print("=" * 60)
    print("VOICE ACTIVITY DETECTION TEST")
    print("=" * 60)
    
    print(f"\nWebRTC VAD available: {HAS_WEBRTC_VAD}")
    
    # Test 1: Basic functionality with synthetic data
    print("\n[Test 1] Basic VAD with synthetic data...")
    
    config = VADConfig(
        sample_rate=16000,
        frame_ms=20,
        aggressiveness=2,
        energy_threshold=0.01,
        smooth_window=5,
    )
    vad = VAD(config)
    
    frame_size = vad.frame_samples
    print(f"  Frame size: {frame_size} samples ({config.frame_ms}ms)")
    
    # Test frames
    silence = np.random.randn(frame_size).astype(np.float32) * 0.001
    quiet_noise = np.random.randn(frame_size).astype(np.float32) * 0.008
    loud_noise = np.random.randn(frame_size).astype(np.float32) * 0.05
    
    print(f"\n  Testing frames:")
    print(f"    Silence (RMS=0.001):     is_speech={vad.is_speech(silence)}")
    print(f"    Quiet noise (RMS=0.008): is_speech={vad.is_speech(quiet_noise)}")
    print(f"    Loud noise (RMS=0.05):   is_speech={vad.is_speech(loud_noise)}")
    
    # Test 2: Smoothing behavior
    print("\n[Test 2] Smoothing behavior...")
    
    vad2 = VAD(VADConfig(smooth_window=5, hangover_frames=3))
    
    # Simulate: silence -> speech -> silence
    pattern = [0.001, 0.001, 0.05, 0.05, 0.05, 0.001, 0.001, 0.001, 0.001, 0.001]
    results = []
    
    for rms in pattern:
        frame = np.random.randn(frame_size).astype(np.float32) * rms
        results.append(vad2.is_speech(frame))
    
    print(f"  Input RMS:  {['S' if r < 0.01 else 'L' for r in pattern]}")
    print(f"  VAD output: {['✓' if r else '·' for r in results]}")
    print(f"  (Hangover keeps speech active after it ends)")
    
    # Test 3: Live audio test
    print("\n[Test 3] Live audio test...")
    
    try:
        from audio_input import AudioInput, AudioInputConfig
        
        print("  Speak and watch VAD detect your voice!")
        print("  (10 seconds)\n")
        
        audio_config = AudioInputConfig(sample_rate=16000, frame_ms=20)
        audio_input = AudioInput(audio_config)
        
        vad_live = VAD(VADConfig(
            aggressiveness=2,
            energy_threshold=0.015,
            smooth_window=5,
            hangover_frames=10,
        ))
        
        audio_input.start()
        time.sleep(0.2)
        
        speech_started = False
        speech_frames = 0
        
        start_time = time.time()
        while time.time() - start_time < 10.0:
            frame = audio_input.get_frame(timeout=0.05)
            if frame is not None:
                is_speech = vad_live.is_speech(frame)
                rms = vad_live.compute_rms(frame)
                
                # Track speech segments
                if is_speech and not speech_started:
                    speech_started = True
                    speech_frames = 0
                elif not is_speech and speech_started:
                    speech_started = False
                
                if is_speech:
                    speech_frames += 1
                
                # Visual indicator
                bar = int(rms * 200)
                status = "🗣️  SPEECH" if is_speech else "   silence"
                
                print(f"\r  {status} {'█' * min(bar, 35):<35} RMS={rms:.3f}", end="", flush=True)
        
        audio_input.stop()
        
        stats = vad_live.get_stats()
        print(f"\n\n  Results:")
        print(f"    Speech ratio: {stats['speech_ratio']:.1%}")
        print(f"    Energy filtered: {stats['energy_filter_ratio']:.1%}")
        
    except ImportError as e:
        print(f"  Skipped - {e}")
        print("  Run from project root to test with live audio")
    
    print("\n" + "=" * 60)
    print("VAD TEST COMPLETE")
    print("=" * 60)
