"""
Voice Assistant v2 - Hybrid VAD Module (RNNoise + Silero)
=========================================================

UPGRADE: Replaces WebRTC VAD with:
1. RNNoise - ML-based noise suppression (much better than simple gate)
2. Silero VAD - State-of-the-art voice activity detection

Benefits over WebRTC VAD:
- Much better accuracy for non-native speakers
- Superior noise handling in crowded environments
- Lower false positive rate
- Better speech/music discrimination

Requirements:
    pip install silero-vad torch torchaudio
    pip install rnnoise-python  # or: pip install rnnoise

RAM Usage: ~100MB total (Silero ~60MB, RNNoise ~40MB)
CPU Usage: Slightly higher than WebRTC but still very efficient
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from collections import deque
from pathlib import Path
import numpy as np
import time
import warnings

# Suppress torch warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# SILERO VAD IMPORTS
# =============================================================================
HAS_SILERO = False
HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
    torch.set_num_threads(1)  # Limit threads for efficiency
except ImportError:
    print("[HybridVAD] Warning: torch not installed")

if HAS_TORCH:
    try:
        # Silero VAD v5 (latest)
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        HAS_SILERO = True
    except Exception as e:
        print(f"[HybridVAD] Warning: Could not load Silero VAD - {e}")

# =============================================================================
# RNNOISE NATIVE LIBRARY (via ctypes)
# =============================================================================
HAS_RNNOISE_NATIVE = False
_rnnoise_lib = None

def _load_rnnoise_native():
    """Load RNNoise native library via ctypes."""
    global _rnnoise_lib, HAS_RNNOISE_NATIVE
    
    if _rnnoise_lib is not None:
        return _rnnoise_lib
    
    import ctypes
    import ctypes.util
    
    # Try to find the library
    lib_paths = [
        "librnnoise.so",
        "librnnoise.so.0",
        "/usr/local/lib/librnnoise.so",
        "/usr/lib/librnnoise.so",
        ctypes.util.find_library("rnnoise"),
    ]
    
    for path in lib_paths:
        if path is None:
            continue
        try:
            _rnnoise_lib = ctypes.CDLL(path)
            HAS_RNNOISE_NATIVE = True
            
            # Define function signatures
            _rnnoise_lib.rnnoise_get_size.restype = ctypes.c_int
            _rnnoise_lib.rnnoise_get_size.argtypes = []
            
            _rnnoise_lib.rnnoise_init.restype = ctypes.c_void_p
            _rnnoise_lib.rnnoise_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            
            _rnnoise_lib.rnnoise_destroy.restype = None
            _rnnoise_lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
            
            _rnnoise_lib.rnnoise_process_frame.restype = ctypes.c_float
            _rnnoise_lib.rnnoise_process_frame.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float)
            ]
            
            _rnnoise_lib.rnnoise_create.restype = ctypes.c_void_p
            _rnnoise_lib.rnnoise_create.argtypes = [ctypes.c_void_p]
            
            return _rnnoise_lib
        except (OSError, AttributeError):
            continue
    
    return None


class RNNoiseNative:
    """Python wrapper for native RNNoise library."""
    
    FRAME_SIZE = 480  # RNNoise processes 480 samples (10ms at 48kHz)
    
    def __init__(self):
        self._lib = _load_rnnoise_native()
        if self._lib is None:
            raise RuntimeError("Could not load RNNoise native library")
        
        import ctypes
        self._ctypes = ctypes
        
        # Create RNNoise state
        self._state = self._lib.rnnoise_create(None)
        if not self._state:
            raise RuntimeError("Failed to create RNNoise state")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through RNNoise.
        
        Args:
            frame: Input audio frame (float32, 480 samples)
            
        Returns:
            Denoised audio frame (float32, 480 samples)
        """
        if len(frame) != self.FRAME_SIZE:
            raise ValueError(f"Frame must be {self.FRAME_SIZE} samples, got {len(frame)}")
        
        # Ensure float32
        frame = frame.astype(np.float32)
        
        # RNNoise expects values in range [-32768, 32767]
        frame_scaled = frame * 32767.0
        
        # Create output buffer
        output = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        
        # Get ctypes pointers
        input_ptr = frame_scaled.ctypes.data_as(self._ctypes.POINTER(self._ctypes.c_float))
        output_ptr = output.ctypes.data_as(self._ctypes.POINTER(self._ctypes.c_float))
        
        # Process
        vad_prob = self._lib.rnnoise_process_frame(self._state, output_ptr, input_ptr)
        
        # Scale back to [-1, 1]
        output = output / 32767.0
        
        return output
    
    def __del__(self):
        if hasattr(self, '_state') and self._state and hasattr(self, '_lib') and self._lib:
            try:
                self._lib.rnnoise_destroy(self._state)
            except:
                pass


# Try to load native RNNoise
try:
    _load_rnnoise_native()
    if HAS_RNNOISE_NATIVE:
        print("[HybridVAD] Native RNNoise library found")
except:
    pass


@dataclass
class HybridVADConfig:
    """Configuration for Hybrid VAD (RNNoise + Silero)."""
    
    # Audio settings
    sample_rate: int = 16000        # Silero requires 16kHz
    
    # Silero VAD settings
    silero_threshold: float = 0.5   # Speech probability threshold (0.3-0.7)
    min_speech_duration_ms: int = 250   # Minimum speech duration
    min_silence_duration_ms: int = 100  # Minimum silence duration
    speech_pad_ms: int = 30         # Padding around speech
    
    # RNNoise settings
    enable_rnnoise: bool = True     # Use RNNoise for denoising
    rnnoise_attenuation: float = 1.0  # Noise attenuation (0.5-1.0)
    
    # Smoothing settings
    smooth_window: int = 3          # Frames to smooth decisions
    hangover_frames: int = 5        # Keep speech active after it ends
    
    # Energy pre-filter (skip VAD for very quiet frames)
    energy_threshold: float = 0.005  # RMS threshold for pre-filter
    
    # Advanced: window size for Silero (512 samples = 32ms at 16kHz)
    window_size_samples: int = 512


class HybridVAD:
    """
    Hybrid Voice Activity Detection using RNNoise + Silero VAD.
    
    Pipeline:
    1. RNNoise: Remove background noise (ML-based)
    2. Energy pre-filter: Skip very quiet frames
    3. Silero VAD: Accurate speech detection
    4. Smoothing + Hangover: Stable decisions
    
    Usage:
        vad = HybridVAD(HybridVADConfig())
        
        for frame in audio_frames:
            is_speech, confidence = vad.is_speech(frame)
            if is_speech:
                process_speech(frame)
        
        # Or use the pipeline that returns denoised audio:
        is_speech, denoised_frame, confidence = vad.process(frame)
    """
    
    def __init__(self, config: Optional[HybridVADConfig] = None):
        self.config = config or HybridVADConfig()
        
        if self.config.sample_rate != 16000:
            raise ValueError("Silero VAD requires 16kHz sample rate")
        
        # Initialize Silero VAD
        self._silero_model = None
        self._silero_utils = None
        
        if HAS_SILERO:
            self._silero_model = model
            self._silero_utils = utils
            print("[HybridVAD] ✓ Silero VAD loaded (SOTA accuracy)")
        else:
            print("[HybridVAD] ✗ Silero VAD not available, falling back to energy-based")
        
        # Initialize noise reduction - try native RNNoise first
        self._rnnoise = None
        self._noise_reducer_type = None
        
        if self.config.enable_rnnoise:
            # Priority 1: Native RNNoise (built from source)
            if HAS_RNNOISE_NATIVE:
                try:
                    self._rnnoise = RNNoiseNative()
                    self._noise_reducer_type = "rnnoise_native"
                    print("[HybridVAD] ✓ RNNoise (native) loaded - best quality noise suppression")
                except Exception as e:
                    print(f"[HybridVAD] ✗ Native RNNoise init failed: {e}")
            
            if self._noise_reducer_type is None:
                print("[HybridVAD] ✗ No noise reducer available")
                print("            Build RNNoise: cd ~/rnnoise && ./autogen.sh && ./configure && make && sudo make install")
        
        # State for Silero (requires tensor state between calls)
        self._silero_state = None
        self._reset_silero_state()
        
        # Smoothing state
        self._history: deque = deque(maxlen=self.config.smooth_window)
        self._confidence_history: deque = deque(maxlen=self.config.smooth_window)
        
        # Hangover state
        self._hangover_counter = 0
        self._last_raw_speech = False
        
        # Statistics
        self._stats = {
            "frames_processed": 0,
            "speech_frames": 0,
            "energy_filtered": 0,
            "rnnoise_applied": 0,
            "avg_confidence": 0.0,
        }
        self._total_confidence = 0.0
        
        # CRITICAL: Reset Silero model state to avoid interference from other instances
        self._reset_silero_state()
    
    def _reset_silero_state(self):
        """Reset Silero's internal state."""
        if HAS_SILERO and HAS_TORCH and self._silero_model is not None:
            # Silero VAD maintains internal LSTM state - must reset it!
            try:
                self._silero_model.reset_states()
            except Exception as e:
                print(f"[HybridVAD] Warning: Could not reset Silero state: {e}")
    
    def process(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, float]:
        """
        Full processing pipeline: VAD on ORIGINAL audio → Denoise for output.
        
        IMPORTANT: Speech detection uses the ORIGINAL audio because RNNoise 
        can be too aggressive and remove speech. Denoising is only applied 
        to the output audio for STT.
        
        Args:
            frame: np.ndarray float32, shape (samples,) at 16kHz
            
        Returns:
            Tuple of:
            - is_speech: bool - True if speech detected
            - denoised_frame: np.ndarray - Noise-reduced audio (for STT)
            - confidence: float - Speech probability (0.0-1.0)
        """
        self._stats["frames_processed"] += 1
        
        # Ensure correct format
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        
        if frame.ndim > 1:
            frame = frame.flatten()
        
        # Step 1: Get VAD decision on ORIGINAL audio (more reliable)
        is_speech, confidence = self._detect_speech(frame)
        
        # Step 2: Apply noise reduction ONLY if speech detected (for STT)
        # This avoids RNNoise destroying silence/low-level audio
        if is_speech and self.config.enable_rnnoise:
            denoised = self._apply_noise_reduction(frame)
        else:
            denoised = frame
        
        return is_speech, denoised, confidence
    
    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Check if frame contains speech.
        
        This matches the interface of the original WebRTC VAD.
        
        Args:
            frame: np.ndarray float32, shape (samples,) at 16kHz
            
        Returns:
            True if speech detected
        """
        is_speech, _, confidence = self.process(frame)
        return is_speech
    
    def is_speech_with_confidence(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if frame contains speech, with confidence score.
        
        Args:
            frame: np.ndarray float32, shape (samples,) at 16kHz
            
        Returns:
            Tuple of (is_speech, confidence)
        """
        is_speech, _, confidence = self.process(frame)
        return is_speech, confidence
    
    def is_speech_simple(self, frame: np.ndarray) -> bool:
        """
        Simple interface matching original VAD.is_speech().
        
        Args:
            frame: np.ndarray float32, shape (samples,) at 16kHz
            
        Returns:
            True if speech detected
        """
        return self.is_speech(frame)
    
    def _apply_noise_reduction(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction using native RNNoise."""
        if self._noise_reducer_type == "rnnoise_native" and self._rnnoise is not None:
            return self._apply_rnnoise_native(frame)
        return frame
    
    def _apply_rnnoise_native(self, frame: np.ndarray) -> np.ndarray:
        """Apply native RNNoise for noise suppression."""
        try:
            # RNNoise processes 480 samples at a time (10ms at 48kHz, 30ms at 16kHz)
            # We need to resample or process in chunks
            chunk_size = 480
            output_chunks = []
            
            for i in range(0, len(frame), chunk_size):
                chunk = frame[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                # Process chunk through RNNoise
                denoised_chunk = self._rnnoise.process_frame(chunk)
                output_chunks.append(denoised_chunk[:min(chunk_size, len(frame) - i)])
            
            # Concatenate
            if output_chunks:
                denoised = np.concatenate(output_chunks)
                self._stats["rnnoise_applied"] += 1
                return denoised[:len(frame)]  # Ensure same length
            
            return frame
            
        except Exception as e:
            # Fallback to original on error
            return frame
    
    def _detect_speech(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect speech using Silero VAD with smoothing.
        
        Args:
            frame: Denoised audio frame
            
        Returns:
            Tuple of (is_speech, confidence)
        """
        # Energy pre-filter
        rms = self._compute_rms(frame)
        if rms < self.config.energy_threshold:
            self._stats["energy_filtered"] += 1
            confidence = 0.0
            raw_speech = False
        else:
            # Get raw Silero prediction
            confidence = self._silero_predict(frame)
            raw_speech = confidence >= self.config.silero_threshold
        
        # Update statistics
        self._total_confidence += confidence
        self._stats["avg_confidence"] = self._total_confidence / self._stats["frames_processed"]
        
        self._last_raw_speech = raw_speech
        
        # Hangover logic
        if raw_speech:
            self._hangover_counter = self.config.hangover_frames
        elif self._hangover_counter > 0:
            self._hangover_counter -= 1
            raw_speech = True  # Keep as speech during hangover
        
        # Smoothing
        self._history.append(raw_speech)
        self._confidence_history.append(confidence)
        
        if len(self._history) < 2:
            result = raw_speech
        else:
            votes = sum(self._history)
            result = votes > len(self._history) / 2
        
        if result:
            self._stats["speech_frames"] += 1
        
        # Return smoothed confidence
        avg_confidence = np.mean(list(self._confidence_history)) if self._confidence_history else confidence
        
        return result, avg_confidence
    
    def _silero_predict(self, frame: np.ndarray) -> float:
        """
        Get speech probability from Silero VAD.
        
        Args:
            frame: Audio frame
            
        Returns:
            Speech probability (0.0 to 1.0)
        """
        if self._silero_model is None:
            # Fallback to energy-based
            rms = self._compute_rms(frame)
            return min(1.0, rms / 0.1)  # Rough energy-to-probability mapping
        
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(frame).float()
            
            # Ensure correct length for Silero (process in 512-sample windows)
            window_size = self.config.window_size_samples
            
            if len(audio_tensor) < window_size:
                # Pad short frames
                audio_tensor = torch.nn.functional.pad(
                    audio_tensor, (0, window_size - len(audio_tensor))
                )
            
            # Process windows and average
            confidences = []
            for i in range(0, len(audio_tensor) - window_size + 1, window_size):
                window = audio_tensor[i:i + window_size]
                
                # Silero VAD forward pass
                with torch.no_grad():
                    speech_prob = self._silero_model(
                        window.unsqueeze(0),
                        self.config.sample_rate
                    ).item()
                confidences.append(speech_prob)
            
            # If we have multiple windows, average them
            if confidences:
                return float(np.mean(confidences))
            else:
                # Process single window
                window = audio_tensor[:window_size]
                with torch.no_grad():
                    speech_prob = self._silero_model(
                        window.unsqueeze(0),
                        self.config.sample_rate
                    ).item()
                return float(speech_prob)
                
        except Exception as e:
            # Fallback on error
            rms = self._compute_rms(frame)
            return min(1.0, rms / 0.1)
    
    @staticmethod
    def _compute_rms(frame: np.ndarray) -> float:
        """Compute RMS energy of frame."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    
    def reset(self) -> None:
        """Reset all state for new recording session."""
        self._history.clear()
        self._confidence_history.clear()
        self._hangover_counter = 0
        self._last_raw_speech = False
        self._reset_silero_state()
    
    def get_stats(self) -> Dict:
        """Get VAD statistics."""
        total = self._stats["frames_processed"]
        speech = self._stats["speech_frames"]
        
        return {
            **self._stats,
            "speech_ratio": speech / total if total > 0 else 0,
            "energy_filter_ratio": self._stats["energy_filtered"] / total if total > 0 else 0,
            "silero_available": self._silero_model is not None,
            "rnnoise_available": self._rnnoise is not None,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "frames_processed": 0,
            "speech_frames": 0,
            "energy_filtered": 0,
            "rnnoise_applied": 0,
            "avg_confidence": 0.0,
        }
        self._total_confidence = 0.0
    
    def calibrate_threshold(self, silence_frames: List[np.ndarray]) -> float:
        """
        Calibrate speech threshold from silence samples.
        
        Args:
            silence_frames: List of audio frames containing only background noise
            
        Returns:
            Recommended threshold value
        """
        if not silence_frames:
            return self.config.silero_threshold
        
        # Get confidence values for silence frames
        silence_confidences = []
        for frame in silence_frames:
            if self._silero_model is not None:
                conf = self._silero_predict(frame)
            else:
                conf = self._compute_rms(frame) / 0.1
            silence_confidences.append(conf)
        
        # Set threshold just above 95th percentile of silence confidences
        # Allow very low thresholds for sensitive detection
        p95 = np.percentile(silence_confidences, 95)
        new_threshold = min(0.5, max(0.03, p95 + 0.02))  # Allow threshold as low as 0.03
        
        print(f"[HybridVAD] Calibrated threshold: {self.config.silero_threshold:.2f} → {new_threshold:.2f}")
        print(f"            Silence confidence: mean={np.mean(silence_confidences):.3f}, p95={p95:.3f}")
        
        self.config.silero_threshold = new_threshold
        return new_threshold


# =============================================================================
# COMPATIBILITY WRAPPER
# =============================================================================
class VADCompatWrapper:
    """
    Wrapper to make HybridVAD compatible with original VAD interface.
    
    Drop-in replacement for the original VAD class.
    """
    
    def __init__(self, config=None):
        """
        Initialize with optional config.
        
        Accepts either HybridVADConfig or original VADConfig.
        """
        if config is None:
            hybrid_config = HybridVADConfig()
        elif hasattr(config, 'silero_threshold'):
            # Already a HybridVADConfig
            hybrid_config = config
        else:
            # Convert from original VADConfig
            hybrid_config = HybridVADConfig(
                sample_rate=getattr(config, 'sample_rate', 16000),
                energy_threshold=getattr(config, 'energy_threshold', 0.005),
                smooth_window=getattr(config, 'smooth_window', 3),
                hangover_frames=getattr(config, 'hangover_frames', 5),
            )
        
        self._vad = HybridVAD(hybrid_config)
        self.config = hybrid_config
        
        # Print config for debugging
        print(f"      VAD Config: silero_threshold={hybrid_config.silero_threshold}, "
              f"energy_threshold={hybrid_config.energy_threshold}, "
              f"hangover={hybrid_config.hangover_frames}")
    
    def is_speech(self, frame: np.ndarray) -> bool:
        """Compatible with original VAD.is_speech()"""
        return self._vad.is_speech_simple(frame)
    
    def is_speech_raw(self, frame: np.ndarray) -> bool:
        """Compatible with original VAD.is_speech_raw()"""
        # Get raw prediction without smoothing
        is_speech, confidence = self._vad._detect_speech(frame)
        return self._vad._last_raw_speech
    
    def reset(self) -> None:
        """Compatible with original VAD.reset()"""
        self._vad.reset()
    
    def get_stats(self) -> Dict:
        """Compatible with original VAD.get_stats()"""
        return self._vad.get_stats()
    
    def reset_stats(self) -> None:
        """Compatible with original VAD.reset_stats()"""
        self._vad.reset_stats()
    
    def calibrate_noise_floor(self, frames: list) -> float:
        """Compatible with original VAD.calibrate_noise_floor()"""
        return self._vad.calibrate_threshold(frames)
    
    @staticmethod
    def compute_rms(frame: np.ndarray) -> float:
        """Compatible with original VAD.compute_rms()"""
        return HybridVAD._compute_rms(frame)
    
    @staticmethod
    def float_to_pcm(frame: np.ndarray) -> bytes:
        """Compatible with original VAD.float_to_pcm()"""
        clipped = np.clip(frame, -1.0, 1.0)
        int16 = (clipped * 32767).astype(np.int16)
        return int16.tobytes()
    
    # Expose the underlying HybridVAD for advanced usage
    @property
    def hybrid_vad(self) -> HybridVAD:
        return self._vad


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test Hybrid VAD module."""
    import time
    
    print("=" * 60)
    print("HYBRID VAD TEST (RNNoise + Silero)")
    print("=" * 60)
    
    # Check dependencies
    print(f"\n✓ Torch available: {HAS_TORCH}")
    print(f"✓ Silero VAD available: {HAS_SILERO}")
    print(f"✓ RNNoise (native) available: {HAS_RNNOISE_NATIVE}")
    
    if not HAS_SILERO:
        print("\n⚠ Install Silero: pip install silero-vad torch torchaudio")
    if not HAS_RNNOISE_NATIVE:
        print("⚠ Build RNNoise: cd ~/rnnoise && ./autogen.sh && ./configure && make && sudo make install")
    
    if HAS_SILERO:
        print("\n--- Testing Hybrid VAD ---")
        
        # Create VAD
        config = HybridVADConfig(
            silero_threshold=0.5,
            enable_rnnoise=HAS_RNNOISE_NATIVE,
        )
        vad = HybridVAD(config)
        
        # Test with synthetic audio
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        
        # Silence
        silence = np.zeros(samples, dtype=np.float32)
        is_speech, conf = vad.is_speech(silence)
        print(f"Silence: is_speech={is_speech}, confidence={conf:.3f}")
        
        # Noise
        noise = np.random.randn(samples).astype(np.float32) * 0.01
        is_speech, conf = vad.is_speech(noise)
        print(f"Low noise: is_speech={is_speech}, confidence={conf:.3f}")
        
        # Loud noise
        loud_noise = np.random.randn(samples).astype(np.float32) * 0.1
        is_speech, conf = vad.is_speech(loud_noise)
        print(f"Loud noise: is_speech={is_speech}, confidence={conf:.3f}")
        
        # Synthetic "speech" (sine wave at speech frequency)
        t = np.linspace(0, duration, samples)
        speech_like = (np.sin(2 * np.pi * 300 * t) * 0.3).astype(np.float32)
        is_speech, conf = vad.is_speech(speech_like)
        print(f"Speech-like (300Hz): is_speech={is_speech}, confidence={conf:.3f}")
        
        print(f"\nStats: {vad.get_stats()}")
        
        # Test compatibility wrapper
        print("\n--- Testing Compatibility Wrapper ---")
        wrapper = VADCompatWrapper()
        is_speech = wrapper.is_speech(speech_like)
        print(f"Wrapper.is_speech(): {is_speech}")
        
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
