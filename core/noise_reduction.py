"""
Voice Assistant v2 - Noise Reduction Module
============================================

Step 2: Filter out background noise BEFORE VAD.

This is CRITICAL for crowded places - reduces false positives from:
- Air conditioning
- Traffic noise
- Distant conversations
- Computer fans

Features:
1. Noise gate (silence below threshold)
2. Adaptive threshold (learns noise floor)
3. Smooth transitions (avoid clicks)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
import numpy as np


@dataclass
class NoiseReducerConfig:
    """Configuration for noise reduction."""
    gate_threshold: float = 0.01    # Initial RMS threshold (will be calibrated)
    adaptive: bool = True           # Learn noise floor over time
    adaptation_rate: float = 0.05   # How fast to adapt (0.01=slow, 0.1=fast)
    attack_ms: float = 5.0          # Fade-in time (gate opening)
    release_ms: float = 20.0        # Fade-out time (gate closing)
    headroom_db: float = 6.0        # dB above noise floor for threshold
    min_threshold: float = 0.001    # Minimum threshold (avoid silence detection issues)
    max_threshold: float = 0.1      # Maximum threshold (avoid cutting speech)


class NoiseReducer:
    """
    Reduces background noise for cleaner VAD input.
    
    Methods:
    1. Noise gate: Silence audio below RMS threshold
    2. Adaptive threshold: Learn and track noise floor
    3. Smooth envelope: Prevent clicks/pops when gating
    
    Usage:
        reducer = NoiseReducer(config)
        
        # Optional: Calibrate with 2-3 seconds of silence
        reducer.calibrate(silence_frames)
        
        for frame in audio_frames:
            clean_frame = reducer.process(frame)
    """
    
    def __init__(self, config: Optional[NoiseReducerConfig] = None, sample_rate: int = 16000):
        self.config = config or NoiseReducerConfig()
        self.sample_rate = sample_rate
        
        # Calculate samples for attack/release
        self._attack_samples = int(self.config.attack_ms * sample_rate / 1000)
        self._release_samples = int(self.config.release_ms * sample_rate / 1000)
        
        # Gate state
        self._noise_floor = self.config.gate_threshold
        self._threshold = self.config.gate_threshold
        self._is_open = False  # Gate state: True = passing audio
        self._envelope = 0.0   # Current gain envelope (0 to 1)
        
        # History for adaptive threshold
        self._rms_history: deque = deque(maxlen=50)  # ~1 second at 20ms frames
        self._silence_frames = 0  # Count consecutive silence frames
        
        # Statistics for calibration
        self._calibrated = False
        self._stats = {
            "frames_processed": 0,
            "frames_gated": 0,
            "current_noise_floor": self._noise_floor,
        }
    
    def calibrate(self, silence_audio: np.ndarray, frame_size: int = 320) -> float:
        """
        Calibrate noise floor from silence recording.
        
        Call this at startup with 2-5 seconds of ambient noise.
        
        Args:
            silence_audio: np.ndarray of silence (float32, mono)
            frame_size: Frame size for analysis (default: 320 = 20ms at 16kHz)
            
        Returns:
            Detected noise floor (RMS)
        """
        if len(silence_audio) < frame_size:
            print("[NoiseReducer] Warning: Not enough audio for calibration")
            return self._noise_floor
        
        # Split into frames and calculate RMS
        rms_values = []
        for i in range(0, len(silence_audio) - frame_size, frame_size):
            frame = silence_audio[i:i + frame_size]
            rms = self.compute_rms(frame)
            rms_values.append(rms)
        
        if not rms_values:
            return self._noise_floor
        
        rms_array = np.array(rms_values)
        
        # Use robust statistics (median + MAD) to handle outliers
        median_rms = np.median(rms_array)
        mad = np.median(np.abs(rms_array - median_rms))  # Median absolute deviation
        
        # Set noise floor as median + 2*MAD (robust equivalent of mean + 2*std)
        self._noise_floor = median_rms + 2 * mad
        
        # Apply headroom (threshold is noise_floor * headroom)
        headroom_linear = 10 ** (self.config.headroom_db / 20)
        self._threshold = np.clip(
            self._noise_floor * headroom_linear,
            self.config.min_threshold,
            self.config.max_threshold
        )
        
        self._calibrated = True
        self._stats["current_noise_floor"] = self._noise_floor
        
        print(f"[NoiseReducer] Calibrated: noise_floor={self._noise_floor:.4f}, "
              f"threshold={self._threshold:.4f}")
        
        return self._noise_floor
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to audio frame.
        
        Args:
            frame: np.ndarray float32, shape (samples,)
            
        Returns:
            Cleaned frame (same shape, may be silenced or attenuated)
        """
        self._stats["frames_processed"] += 1
        
        # Compute frame energy
        rms = self.compute_rms(frame)
        
        # Update history for adaptive threshold
        self._rms_history.append(rms)
        
        # Determine gate state
        should_open = rms > self._threshold
        
        # Apply envelope smoothing
        output = self._apply_envelope(frame, should_open)
        
        # Track statistics
        if not should_open:
            self._stats["frames_gated"] += 1
            self._silence_frames += 1
        else:
            self._silence_frames = 0
        
        # Adaptive threshold update during extended silence
        if self.config.adaptive and self._silence_frames > 10:
            self._adapt_threshold(rms)
        
        return output
    
    def _apply_envelope(self, frame: np.ndarray, should_open: bool) -> np.ndarray:
        """
        Apply smooth envelope to avoid clicks.
        
        Args:
            frame: Input audio frame
            should_open: Whether gate should be open (passing audio)
            
        Returns:
            Envelope-applied frame
        """
        frame_len = len(frame)
        output = np.zeros_like(frame)
        
        # Calculate envelope change per sample
        if should_open:
            # Gate opening (attack)
            if self._attack_samples > 0:
                attack_rate = 1.0 / self._attack_samples
            else:
                attack_rate = 1.0
        else:
            # Gate closing (release)
            if self._release_samples > 0:
                attack_rate = -1.0 / self._release_samples
            else:
                attack_rate = -1.0
        
        # Apply sample-by-sample envelope
        for i in range(frame_len):
            # Update envelope
            self._envelope += attack_rate
            self._envelope = np.clip(self._envelope, 0.0, 1.0)
            
            # Apply gain
            output[i] = frame[i] * self._envelope
        
        self._is_open = should_open
        return output
    
    def _adapt_threshold(self, current_rms: float) -> None:
        """
        Slowly adapt noise floor during confirmed silence.
        
        Args:
            current_rms: Current frame RMS
        """
        # Only adapt if we have enough history
        if len(self._rms_history) < 10:
            return
        
        # Use recent silence RMS to update noise floor
        rate = self.config.adaptation_rate
        self._noise_floor = (1 - rate) * self._noise_floor + rate * current_rms
        
        # Update threshold with headroom
        headroom_linear = 10 ** (self.config.headroom_db / 20)
        self._threshold = np.clip(
            self._noise_floor * headroom_linear,
            self.config.min_threshold,
            self.config.max_threshold
        )
        
        self._stats["current_noise_floor"] = self._noise_floor
    
    def update_from_vad(self, is_speech: bool, rms: float) -> None:
        """
        Update threshold based on VAD feedback.
        
        Call this method with VAD results to improve adaptation.
        
        Args:
            is_speech: VAD decision
            rms: Frame RMS value
        """
        if not self.config.adaptive:
            return
        
        if not is_speech:
            # Confirmed silence - good for adaptation
            self._adapt_threshold(rms)
    
    def get_stats(self) -> dict:
        """
        Get noise reduction statistics.
        
        Returns:
            Dict with processing statistics
        """
        total = self._stats["frames_processed"]
        gated = self._stats["frames_gated"]
        
        return {
            "frames_processed": total,
            "frames_gated": gated,
            "gate_ratio": gated / total if total > 0 else 0,
            "noise_floor": self._noise_floor,
            "threshold": self._threshold,
            "calibrated": self._calibrated,
        }
    
    def reset(self) -> None:
        """Reset state for new recording session."""
        self._is_open = False
        self._envelope = 0.0
        self._rms_history.clear()
        self._silence_frames = 0
        self._stats = {
            "frames_processed": 0,
            "frames_gated": 0,
            "current_noise_floor": self._noise_floor,
        }
    
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
    def compute_db(rms: float, ref: float = 1.0) -> float:
        """
        Convert RMS to decibels.
        
        Args:
            rms: RMS value
            ref: Reference value (default 1.0)
            
        Returns:
            dB value (negative for values < ref)
        """
        if rms <= 0:
            return -100.0
        return 20 * np.log10(rms / ref)


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test noise reduction module."""
    import sys
    import time
    
    # Try to import audio input for live test
    try:
        from audio_input import AudioInput, AudioInputConfig
        HAS_AUDIO_INPUT = True
    except ImportError:
        HAS_AUDIO_INPUT = False
    
    print("=" * 60)
    print("NOISE REDUCTION TEST")
    print("=" * 60)
    
    # Test 1: Basic functionality with synthetic data
    print("\n[Test 1] Basic noise gate with synthetic data...")
    
    config = NoiseReducerConfig(
        gate_threshold=0.02,
        adaptive=True,
        adaptation_rate=0.05,
    )
    reducer = NoiseReducer(config, sample_rate=16000)
    
    # Create test frames
    frame_size = 320  # 20ms at 16kHz
    
    # Silence frame (should be gated)
    silence = np.random.randn(frame_size).astype(np.float32) * 0.001
    
    # Loud frame (should pass through)
    loud = np.random.randn(frame_size).astype(np.float32) * 0.1
    
    # Process silence
    processed_silence = reducer.process(silence)
    silence_rms_before = reducer.compute_rms(silence)
    silence_rms_after = reducer.compute_rms(processed_silence)
    
    # Process loud
    processed_loud = reducer.process(loud)
    loud_rms_before = reducer.compute_rms(loud)
    loud_rms_after = reducer.compute_rms(processed_loud)
    
    print(f"  Silence: RMS {silence_rms_before:.4f} -> {silence_rms_after:.4f} (should decrease)")
    print(f"  Loud:    RMS {loud_rms_before:.4f} -> {loud_rms_after:.4f} (should be similar)")
    
    # Test 2: Calibration
    print("\n[Test 2] Calibration with synthetic noise...")
    
    reducer2 = NoiseReducer(NoiseReducerConfig(), sample_rate=16000)
    
    # Create 3 seconds of synthetic noise
    noise_duration = 3.0
    noise_samples = int(16000 * noise_duration)
    synthetic_noise = np.random.randn(noise_samples).astype(np.float32) * 0.015
    
    noise_floor = reducer2.calibrate(synthetic_noise)
    print(f"  Calibrated noise floor: {noise_floor:.4f}")
    print(f"  Detection threshold:    {reducer2._threshold:.4f}")
    
    # Test 3: Smooth envelope transitions
    print("\n[Test 3] Envelope transitions...")
    
    reducer3 = NoiseReducer(NoiseReducerConfig(
        gate_threshold=0.02,
        attack_ms=10.0,
        release_ms=20.0,
    ), sample_rate=16000)
    
    # Process alternating frames
    transitions = []
    for i in range(10):
        if i % 2 == 0:
            frame = np.random.randn(frame_size).astype(np.float32) * 0.001  # Quiet
        else:
            frame = np.random.randn(frame_size).astype(np.float32) * 0.1   # Loud
        
        processed = reducer3.process(frame)
        rms_in = reducer3.compute_rms(frame)
        rms_out = reducer3.compute_rms(processed)
        transitions.append((rms_in, rms_out))
    
    print("  Frame transitions (RMS in -> RMS out):")
    for i, (rms_in, rms_out) in enumerate(transitions):
        label = "QUIET" if i % 2 == 0 else "LOUD "
        print(f"    {i}: {label} {rms_in:.4f} -> {rms_out:.4f}")
    
    # Test 4: Statistics
    print("\n[Test 4] Statistics...")
    stats = reducer3.get_stats()
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Frames gated:     {stats['frames_gated']}")
    print(f"  Gate ratio:       {stats['gate_ratio']:.1%}")
    print(f"  Noise floor:      {stats['noise_floor']:.4f}")
    
    # Test 5: Live audio test (if audio input available)
    if HAS_AUDIO_INPUT:
        print("\n[Test 5] Live audio test (5 seconds)...")
        print("  Speak and pause to see noise reduction in action!")
        print("  First, calibrating with 2 seconds of silence...")
        print("  >> Please be QUIET for calibration <<")
        
        # Setup audio input
        audio_config = AudioInputConfig(
            sample_rate=16000,
            frame_ms=20,
        )
        audio_input = AudioInput(audio_config)
        reducer_live = NoiseReducer(NoiseReducerConfig(
            adaptive=True,
            adaptation_rate=0.02,
        ), sample_rate=16000)
        
        # Calibration phase
        audio_input.start()
        time.sleep(0.1)  # Let stream stabilize
        
        calibration_frames = []
        start_time = time.time()
        while time.time() - start_time < 2.0:
            frame = audio_input.get_frame(timeout=0.1)
            if frame is not None:
                calibration_frames.append(frame)
        
        if calibration_frames:
            calibration_audio = np.concatenate(calibration_frames)
            reducer_live.calibrate(calibration_audio)
        
        # Now test with live audio
        print("\n  >> Now SPEAK and PAUSE to test noise reduction <<")
        
        test_start = time.time()
        while time.time() - test_start < 5.0:
            frame = audio_input.get_frame(timeout=0.1)
            if frame is not None:
                rms_before = reducer_live.compute_rms(frame)
                processed = reducer_live.process(frame)
                rms_after = reducer_live.compute_rms(processed)
                
                # Visual indicator
                bars_before = int(rms_before * 200)
                bars_after = int(rms_after * 200)
                
                gate_status = "OPEN  " if rms_after > 0.001 else "GATED "
                print(f"\r  {gate_status} IN: {'█' * min(bars_before, 40):<40} "
                      f"OUT: {'█' * min(bars_after, 40):<40}", end="")
        
        audio_input.stop()
        
        print("\n\n  Final stats:")
        live_stats = reducer_live.get_stats()
        print(f"    Gate ratio: {live_stats['gate_ratio']:.1%}")
        print(f"    Adapted noise floor: {live_stats['noise_floor']:.4f}")
    else:
        print("\n[Test 5] Skipped - audio_input module not available")
        print("  Run from project root or ensure audio_input.py is importable")
    
    print("\n" + "=" * 60)
    print("NOISE REDUCTION TEST COMPLETE")
    print("=" * 60)
