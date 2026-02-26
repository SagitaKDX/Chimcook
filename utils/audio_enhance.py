"""
Voice Assistant v2 - Audio Enhancement Module
==============================================

Enhances audio quality for better STT accuracy, especially for:
- Non-native speakers
- Noisy environments
- Soft/quiet speech
- Varied accents

Techniques:
1. Volume normalization - consistent input level
2. Noise reduction - spectral subtraction
3. Speech enhancement - frequency filtering
4. Dynamic range compression - balance loud/soft parts
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# Try to import scipy for advanced filtering
try:
    from scipy import signal
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class AudioEnhanceConfig:
    """Configuration for audio enhancement."""
    # Normalization
    normalize: bool = True
    target_db: float = -3.0         # Target peak level in dB
    
    # Noise reduction
    noise_reduce: bool = True
    noise_floor_db: float = -50.0   # Noise floor in dB
    
    # High-pass filter (remove low rumble)
    highpass: bool = True
    highpass_freq: float = 80.0     # Hz - removes room rumble
    
    # Low-pass filter (remove high hiss)
    lowpass: bool = True
    lowpass_freq: float = 7500.0    # Hz - keeps speech, removes hiss
    
    # Dynamic range compression
    compress: bool = True
    compress_threshold: float = 0.3  # Start compressing above this
    compress_ratio: float = 3.0      # Compression ratio
    
    # De-esser (reduce harsh 's' sounds)
    deess: bool = False
    deess_freq: float = 5000.0      # Center frequency for de-essing
    
    # Speech clarity
    clarity_boost: bool = True
    clarity_freq_low: float = 1000.0   # Boost speech frequencies
    clarity_freq_high: float = 4000.0
    clarity_gain_db: float = 3.0


class AudioEnhancer:
    """
    Enhances audio quality for better STT recognition.
    
    Particularly helpful for:
    - Non-native English speakers
    - Speakers with accents
    - Quiet or mumbled speech
    - Noisy recordings
    
    Usage:
        enhancer = AudioEnhancer(AudioEnhanceConfig())
        
        # Enhance audio before STT
        enhanced = enhancer.enhance(audio)
        text = stt.transcribe(enhanced)
        
        # Or use individual methods
        audio = enhancer.normalize(audio)
        audio = enhancer.reduce_noise(audio)
    """
    
    def __init__(self, config: Optional[AudioEnhanceConfig] = None, sample_rate: int = 16000):
        self.config = config or AudioEnhanceConfig()
        self.sample_rate = sample_rate
        
        # Pre-compute filter coefficients if scipy available
        self._filters_ready = False
        if HAS_SCIPY:
            self._setup_filters()
    
    def _setup_filters(self) -> None:
        """Pre-compute filter coefficients for efficiency."""
        nyquist = self.sample_rate / 2
        
        # High-pass filter
        if self.config.highpass:
            hp_freq = min(self.config.highpass_freq / nyquist, 0.99)
            self._hp_b, self._hp_a = signal.butter(4, hp_freq, btype='high')
        
        # Low-pass filter
        if self.config.lowpass:
            lp_freq = min(self.config.lowpass_freq / nyquist, 0.99)
            self._lp_b, self._lp_a = signal.butter(4, lp_freq, btype='low')
        
        # Bandpass for clarity boost
        if self.config.clarity_boost:
            low = min(self.config.clarity_freq_low / nyquist, 0.99)
            high = min(self.config.clarity_freq_high / nyquist, 0.99)
            if low < high:
                self._clarity_b, self._clarity_a = signal.butter(2, [low, high], btype='band')
        
        # De-esser bandstop
        if self.config.deess:
            center = self.config.deess_freq / nyquist
            bandwidth = 1000 / nyquist  # 1kHz bandwidth
            low = max(0.01, center - bandwidth/2)
            high = min(0.99, center + bandwidth/2)
            self._deess_b, self._deess_a = signal.butter(2, [low, high], btype='bandstop')
        
        self._filters_ready = True
    
    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply full enhancement pipeline.
        
        Args:
            audio: np.ndarray float32, shape (samples,)
            
        Returns:
            Enhanced audio array
        """
        if audio is None or len(audio) == 0:
            return audio
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Make a copy to avoid modifying original
        enhanced = audio.copy()
        
        # Step 1: High-pass filter (remove rumble)
        if self.config.highpass and HAS_SCIPY and self._filters_ready:
            enhanced = self._apply_highpass(enhanced)
        
        # Step 2: Noise reduction
        if self.config.noise_reduce:
            enhanced = self.reduce_noise(enhanced)
        
        # Step 3: Dynamic range compression
        if self.config.compress:
            enhanced = self.compress(enhanced)
        
        # Step 4: Clarity boost (enhance speech frequencies)
        if self.config.clarity_boost and HAS_SCIPY and self._filters_ready:
            enhanced = self._boost_clarity(enhanced)
        
        # Step 5: Low-pass filter (remove hiss)
        if self.config.lowpass and HAS_SCIPY and self._filters_ready:
            enhanced = self._apply_lowpass(enhanced)
        
        # Step 6: De-esser
        if self.config.deess and HAS_SCIPY and self._filters_ready:
            enhanced = self._apply_deesser(enhanced)
        
        # Step 7: Normalize (do last to set final level)
        if self.config.normalize:
            enhanced = self.normalize(enhanced)
        
        return enhanced
    
    def normalize(self, audio: np.ndarray, target_db: Optional[float] = None) -> np.ndarray:
        """
        Normalize audio to target peak level.
        
        Args:
            audio: Input audio array
            target_db: Target peak level in dB (default from config)
            
        Returns:
            Normalized audio
        """
        if len(audio) == 0:
            return audio
        
        target_db = target_db if target_db is not None else self.config.target_db
        
        # Find current peak
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            return audio  # Silence, don't amplify noise
        
        # Calculate target amplitude
        target_amp = 10 ** (target_db / 20)
        
        # Apply gain
        gain = target_amp / peak
        normalized = audio * gain
        
        # Clip to prevent overflow
        return np.clip(normalized, -1.0, 1.0)
    
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce background noise using spectral gating.
        
        Simple but effective noise reduction for STT.
        """
        if len(audio) == 0:
            return audio
        
        # Calculate noise floor in linear scale
        noise_floor = 10 ** (self.config.noise_floor_db / 20)
        
        # Simple noise gate with smooth transition
        abs_audio = np.abs(audio)
        
        # Compute envelope (smoothed amplitude)
        if HAS_SCIPY:
            envelope = uniform_filter1d(abs_audio, size=int(self.sample_rate * 0.01))
        else:
            # Simple moving average fallback
            window = int(self.sample_rate * 0.01)
            envelope = np.convolve(abs_audio, np.ones(window)/window, mode='same')
        
        # Create gain curve (soft knee)
        gain = np.clip((envelope - noise_floor) / (noise_floor * 2 + 1e-10), 0, 1)
        gain = gain ** 0.5  # Square root for gentler gating
        
        # Smooth the gain to avoid artifacts
        if HAS_SCIPY:
            gain = uniform_filter1d(gain, size=int(self.sample_rate * 0.005))
        
        return audio * gain
    
    def compress(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Makes quiet parts louder and loud parts quieter,
        helping with varied speaking volumes.
        """
        if len(audio) == 0:
            return audio
        
        threshold = self.config.compress_threshold
        ratio = self.config.compress_ratio
        
        # Compute envelope for compression
        abs_audio = np.abs(audio)
        if HAS_SCIPY:
            envelope = uniform_filter1d(abs_audio, size=int(self.sample_rate * 0.02))
        else:
            window = int(self.sample_rate * 0.02)
            envelope = np.convolve(abs_audio, np.ones(window)/window, mode='same')
        
        # Calculate compression gain
        gain = np.ones_like(envelope)
        above_threshold = envelope > threshold
        
        # Soft knee compression
        if np.any(above_threshold):
            excess = envelope[above_threshold] - threshold
            compressed_excess = excess / ratio
            gain[above_threshold] = (threshold + compressed_excess) / (envelope[above_threshold] + 1e-10)
        
        # Apply gain with smoothing
        if HAS_SCIPY:
            gain = uniform_filter1d(gain, size=int(self.sample_rate * 0.01))
        
        return audio * gain
    
    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove low frequency rumble."""
        if hasattr(self, '_hp_b') and hasattr(self, '_hp_a'):
            return signal.filtfilt(self._hp_b, self._hp_a, audio).astype(np.float32)
        return audio
    
    def _apply_lowpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to remove high frequency hiss."""
        if hasattr(self, '_lp_b') and hasattr(self, '_lp_a'):
            return signal.filtfilt(self._lp_b, self._lp_a, audio).astype(np.float32)
        return audio
    
    def _boost_clarity(self, audio: np.ndarray) -> np.ndarray:
        """Boost speech frequencies for clarity."""
        if hasattr(self, '_clarity_b') and hasattr(self, '_clarity_a'):
            # Extract speech frequencies
            speech_band = signal.filtfilt(self._clarity_b, self._clarity_a, audio)
            
            # Calculate gain
            gain_linear = 10 ** (self.config.clarity_gain_db / 20)
            
            # Add boosted band back (mix)
            boosted = audio + speech_band * (gain_linear - 1)
            
            # Prevent clipping
            return np.clip(boosted, -1.0, 1.0).astype(np.float32)
        return audio
    
    def _apply_deesser(self, audio: np.ndarray) -> np.ndarray:
        """Reduce harsh 's' sounds that can confuse STT."""
        if hasattr(self, '_deess_b') and hasattr(self, '_deess_a'):
            # Detect sibilance
            sibilance = signal.filtfilt(self._deess_b, self._deess_a, audio)
            
            # Only reduce where sibilance is strong
            sib_envelope = np.abs(sibilance)
            if HAS_SCIPY:
                sib_envelope = uniform_filter1d(sib_envelope, size=int(self.sample_rate * 0.005))
            
            # Create reduction mask
            threshold = np.percentile(sib_envelope, 90)
            reduction = np.where(sib_envelope > threshold, 0.5, 1.0)
            
            return (audio * reduction).astype(np.float32)
        return audio


class TextCorrector:
    """
    Post-processing text corrections for non-native speaker transcriptions.
    
    Fixes common STT errors:
    - Homophones (their/there/they're)
    - Common mishearings
    - Filler words
    - Repeated words
    """
    
    def __init__(self):
        # Common STT errors for non-native speakers
        self._corrections = {
            # Filler words to remove
            " um ": " ",
            " uh ": " ",
            " ah ": " ",
            " er ": " ",
            " like like ": " like ",
            " you know ": " ",
            
            # Common mishearings
            " gonna ": " going to ",
            " wanna ": " want to ",
            " gotta ": " got to ",
            " kinda ": " kind of ",
            " sorta ": " sort of ",
            " lemme ": " let me ",
            " gimme ": " give me ",
            " dunno ": " don't know ",
            " coulda ": " could have ",
            " woulda ": " would have ",
            " shoulda ": " should have ",
            
            # Double words (stuttering)
            " the the ": " the ",
            " a a ": " a ",
            " to to ": " to ",
            " and and ": " and ",
            " is is ": " is ",
            " it it ": " it ",
            " i i ": " I ",
            " that that ": " that ",
        }
        
        # Words that are often misheard (context-dependent)
        self._confused_words = {
            # These need context to fix properly
            "there": ["their", "they're"],
            "your": ["you're"],
            "its": ["it's"],
            "to": ["too", "two"],
            "then": ["than"],
            "affect": ["effect"],
            "accept": ["except"],
        }
    
    def correct(self, text: str) -> str:
        """
        Apply corrections to transcribed text.
        
        Args:
            text: Raw transcription
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Add spaces for matching
        corrected = " " + text.lower() + " "
        
        # Apply basic corrections
        for wrong, right in self._corrections.items():
            corrected = corrected.replace(wrong, right)
        
        # Remove extra spaces
        corrected = " ".join(corrected.split())
        
        # Capitalize first letter
        if corrected:
            corrected = corrected[0].upper() + corrected[1:]
        
        # Capitalize 'I'
        corrected = corrected.replace(" i ", " I ")
        corrected = corrected.replace(" i'", " I'")
        
        return corrected.strip()
    
    def remove_fillers(self, text: str) -> str:
        """Remove filler words only."""
        fillers = ["um", "uh", "ah", "er", "like", "you know", "basically", "actually"]
        
        words = text.split()
        filtered = []
        
        i = 0
        while i < len(words):
            word = words[i].lower().strip(".,!?")
            
            # Check for two-word fillers
            if i < len(words) - 1:
                two_words = f"{word} {words[i+1].lower().strip('.,!?')}"
                if two_words in ["you know", "i mean"]:
                    i += 2
                    continue
            
            # Check single word fillers
            if word not in fillers:
                filtered.append(words[i])
            
            i += 1
        
        return " ".join(filtered)


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test audio enhancement."""
    import time
    
    print("=" * 60)
    print("AUDIO ENHANCEMENT TEST")
    print("=" * 60)
    
    print(f"\nscipy available: {HAS_SCIPY}")
    
    # Create enhancer
    config = AudioEnhanceConfig(
        normalize=True,
        noise_reduce=True,
        compress=True,
        clarity_boost=True,
        highpass=True,
        lowpass=True,
    )
    enhancer = AudioEnhancer(config, sample_rate=16000)
    
    # Test 1: Synthetic audio
    print("\n[Test 1] Enhance synthetic audio...")
    
    # Create test signal: speech-like with noise
    duration = 2.0
    t = np.linspace(0, duration, int(16000 * duration))
    
    # Simulate speech (mix of frequencies)
    speech = (
        0.3 * np.sin(2 * np.pi * 200 * t) +   # Fundamental
        0.2 * np.sin(2 * np.pi * 400 * t) +   # Harmonics
        0.1 * np.sin(2 * np.pi * 800 * t) +
        0.05 * np.sin(2 * np.pi * 1600 * t)
    )
    
    # Add noise
    noise = np.random.randn(len(t)) * 0.05
    noisy = (speech + noise).astype(np.float32)
    
    # Enhance
    start = time.time()
    enhanced = enhancer.enhance(noisy)
    elapsed = time.time() - start
    
    print(f"  Input RMS:    {np.sqrt(np.mean(noisy**2)):.4f}")
    print(f"  Output RMS:   {np.sqrt(np.mean(enhanced**2)):.4f}")
    print(f"  Input peak:   {np.max(np.abs(noisy)):.4f}")
    print(f"  Output peak:  {np.max(np.abs(enhanced)):.4f}")
    print(f"  Processing:   {elapsed*1000:.1f}ms for {duration}s audio")
    
    # Test 2: Text correction
    print("\n[Test 2] Text correction...")
    
    corrector = TextCorrector()
    
    test_texts = [
        "um i i wanna go to the the store",
        "she gonna like like finish the work",
        "uh you know i dunno what to do",
        "he coulda been there you know",
    ]
    
    for text in test_texts:
        corrected = corrector.correct(text)
        print(f"  '{text}'")
        print(f"  → '{corrected}'")
        print()
    
    # Test 3: Live audio enhancement
    print("[Test 3] Live audio enhancement test...")
    
    try:
        import sys
        sys.path.insert(0, str(__file__).replace('/utils/audio_enhance.py', '/core'))
        from audio_input import AudioInput, AudioInputConfig
        from stt import STT, STTConfig
        
        print("  Recording 5 seconds - speak naturally!")
        print()
        
        audio_input = AudioInput(AudioInputConfig(sample_rate=16000, frame_ms=20))
        stt = STT(STTConfig(model_size='tiny', language='en'))
        
        audio_input.start()
        time.sleep(0.3)
        
        frames = []
        start = time.time()
        while time.time() - start < 5.0:
            frame = audio_input.get_frame(timeout=0.05)
            if frame is not None:
                frames.append(frame)
                rms = np.sqrt(np.mean(frame**2))
                bar = '█' * int(rms * 150)
                print(f"\r  🎤 {bar:<40}", end="", flush=True)
        
        audio_input.stop()
        print()
        
        if frames:
            audio = np.concatenate(frames)
            
            # Transcribe original
            print("\n  Original transcription:")
            text_orig = stt.transcribe(audio)
            print(f"    '{text_orig}'")
            
            # Enhance and transcribe
            enhanced = enhancer.enhance(audio)
            
            print("\n  Enhanced transcription:")
            text_enh = stt.transcribe(enhanced)
            print(f"    '{text_enh}'")
            
            # Apply text correction
            corrected = corrector.correct(text_enh)
            print("\n  After text correction:")
            print(f"    '{corrected}'")
        
    except ImportError as e:
        print(f"  Skipped live test - {e}")
    
    print("\n" + "=" * 60)
    print("AUDIO ENHANCEMENT TEST COMPLETE")
    print("=" * 60)
