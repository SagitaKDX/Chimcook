"""
Simple Silero VAD for Speech Detection
=======================================

Uses Silero VAD model directly with a simple counter-based approach:
1. Buffer audio in chunks (~512 samples = 32ms at 16kHz)
2. Feed chunk to Silero VAD → get probability (0.0 to 1.0)
3. If probability > threshold → speech, reset silence counter
4. If probability < threshold → increment silence counter
5. If silence counter > limit (e.g., 1.0s) → end of speech

This is simpler and more reliable than complex hybrid approaches.
"""

import numpy as np
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Try to load Silero VAD
HAS_SILERO = False
_silero_model = None

try:
    import torch
    torch.set_num_threads(1)
    
    # Load Silero VAD v5
    _silero_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    HAS_SILERO = True
    print("      [SileroVAD] Model loaded successfully")
except Exception as e:
    print(f"      [SileroVAD] Warning: Could not load Silero VAD - {e}")


class SileroVAD:
    """
    Simple Silero VAD with silence-duration-based end detection.
    
    Usage:
        vad = SileroVAD(threshold=0.5, silence_limit_sec=1.0)
        
        for frame in audio_frames:
            prob, is_speech = vad.process(frame)
            
            if vad.should_stop_recording():
                # User stopped talking
                process_audio()
                vad.reset()
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_limit_sec: float = 1.0,
        speech_start_threshold: float = 0.6,  # Higher threshold to start
    ):
        """
        Args:
            sample_rate: Audio sample rate (must be 16000 for Silero)
            threshold: Probability threshold for speech detection
            silence_limit_sec: Seconds of silence before stopping
            speech_start_threshold: Higher threshold to start recording (reduces false starts)
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_limit_sec = silence_limit_sec
        self.speech_start_threshold = speech_start_threshold
        
        # State
        self._is_recording = False
        self._silence_duration = 0.0
        self._speech_duration = 0.0
        self._last_prob = 0.0
        
        # Silero requires 512 samples per chunk for best accuracy
        self.chunk_samples = 512
        self.chunk_duration = self.chunk_samples / sample_rate  # ~32ms
        
        # Muted mic: only override to silence when RMS below this AND Silero prob < 0.4
        self.muted_rms_threshold = 0.004
        
        # Calibrated noise floor (set after ambient calibration)
        self._noise_floor_mean = 0.0
        self._noise_floor_p95 = 0.02   # default until calibrated
        self._calibrated = False
        
        # Speech hangover: require N consecutive silence chunks before counting silence
        self._speech_hangover_chunks = 3
        self._consecutive_silence_chunks = 0
        
        # Time-since-last-speech: when muted, some mics repeat last buffer (high RMS)
        # so we never "see" silence; stop after this many seconds with no speech
        self._max_no_speech_sec = 2.0
        self._last_speech_time = 0.0
        
        # Buffer for accumulating samples
        self._buffer = np.array([], dtype=np.float32)
    
    def set_noise_floor(self, mean_rms: float, p95_rms: float) -> None:
        """Set noise floor from ambient calibration (no speech). Call after 2–3 sec of quiet."""
        self._noise_floor_mean = max(0.001, mean_rms)
        self._noise_floor_p95 = max(self._noise_floor_mean * 1.2, p95_rms, 0.008)
        self._calibrated = True
    
    def process(self, frame: np.ndarray) -> tuple:
        """
        Process audio frame and update state.
        
        Args:
            frame: Audio frame (float32, any length)
            
        Returns:
            (probability, is_speech) tuple
        """
        # Ensure float32
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        
        # Add to buffer
        self._buffer = np.concatenate([self._buffer, frame])
        
        # Process complete chunks
        prob = 0.0
        processed_any = False
        
        while len(self._buffer) >= self.chunk_samples:
            chunk = self._buffer[:self.chunk_samples]
            self._buffer = self._buffer[self.chunk_samples:]
            
            # Get Silero probability
            prob = self._get_probability(chunk)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            
            # Muted mic: very low energy + Silero not sure = silence
            if rms < self.muted_rms_threshold and prob < 0.4:
                prob = 0.0
            
            # Calibrated noise floor: when energy is near ambient, treat as silence
            # (so we detect "stopped talking" even with background noise)
            if self._calibrated and rms <= self._noise_floor_p95 * 2.2:
                prob = min(prob, 0.35)  # Push toward silence; don't fully override if Silero says strong speech
            if self._calibrated and rms <= self._noise_floor_p95 * 1.5:
                prob = 0.0  # Clearly at/below ambient = silence
            
            self._last_prob = prob
            processed_any = True
            
            # Update silence/speech counters
            if prob >= self.threshold:
                # Speech detected
                self._last_speech_time = time.time()
                self._silence_duration = 0.0
                self._consecutive_silence_chunks = 0
                self._speech_duration += self.chunk_duration
                
                if prob >= self.speech_start_threshold:
                    self._is_recording = True
                    if self._last_speech_time == 0.0:
                        self._last_speech_time = time.time()
            else:
                # Silence (or below threshold)
                if self._is_recording:
                    self._consecutive_silence_chunks += 1
                    # Hard mute: very low RMS → count silence immediately (no hangover)
                    if rms < 0.005:
                        self._consecutive_silence_chunks = max(
                            self._consecutive_silence_chunks,
                            self._speech_hangover_chunks,
                        )
                    if self._consecutive_silence_chunks >= self._speech_hangover_chunks:
                        self._silence_duration += self.chunk_duration
        
        is_speech = prob >= self.threshold
        return prob, is_speech
    
    def _get_probability(self, chunk: np.ndarray) -> float:
        """Get speech probability from Silero model."""
        if not HAS_SILERO or _silero_model is None:
            # Fallback to energy-based
            rms = np.sqrt(np.mean(chunk ** 2))
            return min(1.0, rms / 0.05)  # Rough mapping
        
        try:
            import torch
            audio_tensor = torch.from_numpy(chunk).float()
            
            with torch.no_grad():
                prob = _silero_model(
                    audio_tensor.unsqueeze(0),
                    self.sample_rate
                ).item()
            
            return float(prob)
        except Exception:
            # Fallback
            rms = np.sqrt(np.mean(chunk ** 2))
            return min(1.0, rms / 0.05)
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording speech."""
        return self._is_recording
    
    @property
    def silence_duration(self) -> float:
        """Get current silence duration in seconds."""
        return self._silence_duration
    
    @property
    def speech_duration(self) -> float:
        """Get current speech duration in seconds."""
        return self._speech_duration
    
    @property
    def last_probability(self) -> float:
        """Get last speech probability."""
        return self._last_prob
    
    def should_stop_recording(self) -> bool:
        """
        Check if recording should stop.
        
        Returns True if:
        - Currently recording AND
        - (Silence >= limit AND at least 300ms speech) OR
        - (Silence >= 1.5s) OR
        - (No speech for 2s = muted mic / stuck buffer)
        """
        if not self._is_recording:
            return False
        # Normal case: silence after some speech
        if self._silence_duration >= self.silence_limit_sec and self._speech_duration > 0.3:
            return True
        # Long silence (muted or long pause)
        if self._silence_duration >= 1.5:
            return True
        # Muted mic fallback: some mics repeat last buffer so we never "see" silence.
        # Stop after 2s with no new speech.
        if self._last_speech_time > 0 and (time.time() - self._last_speech_time) >= self._max_no_speech_sec:
            return True
        return False
    
    def reset(self) -> None:
        """Reset state for new recording."""
        self._is_recording = False
        self._silence_duration = 0.0
        self._speech_duration = 0.0
        self._consecutive_silence_chunks = 0
        self._last_speech_time = 0.0
        self._buffer = np.array([], dtype=np.float32)
        
        # Reset Silero model state
        if HAS_SILERO and _silero_model is not None:
            try:
                _silero_model.reset_states()
            except Exception:
                pass
    
    def force_start_recording(self) -> None:
        """Force start recording (e.g., after wake word)."""
        self._is_recording = True
        self._silence_duration = 0.0
        self._speech_duration = 0.0


# =============================================================================
# COMPATIBILITY WRAPPER
# =============================================================================
class SileroVADCompat:
    """
    Wrapper to make SileroVAD compatible with existing VAD interface.
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: Ignored (for compatibility)
        """
        self._vad = SileroVAD(
            threshold=0.5,
            silence_limit_sec=1.0,
            speech_start_threshold=0.6,
        )
    
    def is_speech(self, frame: np.ndarray) -> bool:
        """Check if frame contains speech."""
        prob, is_speech = self._vad.process(frame)
        return is_speech
    
    def reset(self) -> None:
        """Reset VAD state."""
        self._vad.reset()
    
    @staticmethod
    def compute_rms(frame: np.ndarray) -> float:
        """Compute RMS energy."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


# Test code
if __name__ == "__main__":
    print("=" * 50)
    print("Silero VAD Test")
    print("=" * 50)
    
    print(f"\nSilero available: {HAS_SILERO}")
    
    if HAS_SILERO:
        vad = SileroVAD(threshold=0.5, silence_limit_sec=1.0)
        
        # Test with synthetic data
        silence = np.random.randn(512).astype(np.float32) * 0.001
        noise = np.random.randn(512).astype(np.float32) * 0.05
        
        prob_s, _ = vad.process(silence)
        vad.reset()
        prob_n, _ = vad.process(noise)
        
        print(f"\nSilence probability: {prob_s:.3f}")
        print(f"Noise probability: {prob_n:.3f}")
        print("\nVAD initialized successfully!")
