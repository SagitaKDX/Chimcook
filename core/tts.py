"""
Voice Assistant v2 - Text-to-Speech Module
==========================================

Step 7: Convert text to speech using Piper TTS.

Features:
- Fast, lightweight synthesis (~0.1s for short sentences)
- Fully offline
- Natural sounding voices
- Adjustable speed and variation

Recommended voices (~60MB each):
- en_US-amy-medium (female, clear) - DEFAULT
- en_US-lessac-medium (male, natural)
- en_GB-alba-medium (British female)

Voice Download Instructions:
============================

# Download amy voice (default)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('rhasspy/piper-voices', 'en/en_US/amy/medium/en_US-amy-medium.onnx', local_dir='models/tts')
hf_hub_download('rhasspy/piper-voices', 'en/en_US/amy/medium/en_US-amy-medium.onnx.json', local_dir='models/tts')
"

# Download lessac voice (male)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('rhasspy/piper-voices', 'en/en_US/lessac/medium/en_US-lessac-medium.onnx', local_dir='models/tts')
hf_hub_download('rhasspy/piper-voices', 'en/en_US/lessac/medium/en_US-lessac-medium.onnx.json', local_dir='models/tts')
"

Browse all voices: https://huggingface.co/rhasspy/piper-voices/tree/main
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import numpy as np
import re
import time
import wave
import io

# Import Piper TTS
try:
    from piper import PiperVoice
    HAS_PIPER = True
except ImportError:
    HAS_PIPER = False
    print("[TTS] Warning: piper-tts not installed")


@dataclass
class TTSConfig:
    """Configuration for TTS synthesis."""
    model_path: str = ""            # Path to .onnx voice model
    length_scale: float = 1.0       # Speed: <1 faster, >1 slower
    noise_scale: float = 0.667      # Variation in pronunciation
    noise_w_scale: float = 0.8      # Variation in duration
    speaker_id: int = 0             # For multi-speaker models (usually 0)


@dataclass
class TTSConfigCute(TTSConfig):
    """Pre-configured TTS with cute voice (libritts speaker 100)."""
    # Uses libritts high quality with speaker 100 (cute female voice)
    speaker_id: int = 100
    length_scale: float = 0.95      # Slightly faster, more energetic


@dataclass
class TTSConfigFast(TTSConfig):
    """Pre-configured TTS for fast synthesis (slightly less natural)."""
    length_scale: float = 0.9       # Slightly faster
    noise_scale: float = 0.5        # Less variation
    noise_w_scale: float = 0.6


@dataclass  
class TTSConfigNatural(TTSConfig):
    """Pre-configured TTS for more natural sounding speech."""
    length_scale: float = 1.05      # Slightly slower, more natural
    noise_scale: float = 0.8        # More variation
    noise_w_scale: float = 0.9


class TextNormalizer:
    """
    Normalize text for better TTS output.
    
    Handles abbreviations, numbers, symbols, etc.
    """
    
    # Common abbreviations
    ABBREVIATIONS = {
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Ms.": "Miss",
        "Prof.": "Professor",
        "Jr.": "Junior",
        "Sr.": "Senior",
        "St.": "Saint",
        "vs.": "versus",
        "etc.": "et cetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "approx.": "approximately",
        "govt.": "government",
        "dept.": "department",
    }
    
    # Number words
    ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
             "sixteen", "seventeen", "eighteen", "nineteen"]
    TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Normalize text for TTS.
        
        Args:
            text: Raw text input
            
        Returns:
            Normalized text suitable for TTS
        """
        # Replace abbreviations
        for abbr, full in cls.ABBREVIATIONS.items():
            text = text.replace(abbr, full)
        
        # Replace symbols
        text = text.replace("&", " and ")
        text = text.replace("%", " percent")
        text = text.replace("@", " at ")
        text = text.replace("#", " number ")
        text = text.replace("$", " dollars ")
        text = text.replace("€", " euros ")
        text = text.replace("£", " pounds ")
        text = text.replace("+", " plus ")
        text = text.replace("=", " equals ")
        
        # Convert simple numbers (up to 999)
        text = re.sub(r'\b(\d{1,3})\b', lambda m: cls._number_to_words(int(m.group(1))), text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove or replace problematic characters
        text = text.replace('"', '')
        text = text.replace("'", "'")  # Normalize apostrophes
        
        return text.strip()
    
    @classmethod
    def _number_to_words(cls, n: int) -> str:
        """Convert number (0-999) to words."""
        if n == 0:
            return "zero"
        if n < 10:
            return cls.ONES[n]
        if n < 20:
            return cls.TEENS[n - 10]
        if n < 100:
            tens, ones = divmod(n, 10)
            return cls.TENS[tens] + (" " + cls.ONES[ones] if ones else "")
        if n < 1000:
            hundreds, remainder = divmod(n, 100)
            result = cls.ONES[hundreds] + " hundred"
            if remainder:
                result += " " + cls._number_to_words(remainder)
            return result
        return str(n)  # Return as-is for larger numbers


class TTS:
    """
    Text-to-Speech using Piper TTS.
    
    Features:
    - Fast CPU synthesis (~0.1s for short text)
    - Natural sounding voices
    - Adjustable speed and variation
    - Text normalization for better pronunciation
    - Sentence splitting for long texts
    
    Usage:
        # Basic usage
        config = TTSConfig(model_path="models/tts/en/en_US/amy/medium/en_US-amy-medium.onnx")
        tts = TTS(config)
        audio, sample_rate = tts.synthesize("Hello! How are you?")
        
        # Save to file
        tts.synthesize_to_file("Hello world!", "output.wav")
        
        # Get raw bytes (for streaming)
        audio_bytes = tts.synthesize_to_bytes("Hello!")
    """
    
    def __init__(self, config: Optional[TTSConfig] = None):
        if not HAS_PIPER:
            raise ImportError(
                "piper-tts not installed. "
                "Run: pip install piper-tts"
            )
        
        self.config = config or TTSConfig()
        
        # Find model if not specified
        if not self.config.model_path:
            self.config.model_path = self._find_model()
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Voice model not found: {model_path}\n"
                "Download a voice model first. See docstring for instructions."
            )
        
        print(f"[TTS] Loading voice: {model_path.stem}")
        
        load_start = time.time()
        
        # Load Piper voice
        self._voice = PiperVoice.load(str(model_path))
        
        load_time = time.time() - load_start
        print(f"[TTS] Voice loaded in {load_time:.2f}s")
        print(f"      Sample rate: {self._voice.config.sample_rate} Hz")
        
        # Text normalizer
        self._normalizer = TextNormalizer()
        
        # Statistics
        self._stats = {
            "syntheses": 0,
            "total_chars": 0,
            "total_time": 0.0,
        }
    
    def _find_model(self) -> str:
        """Find a voice model in the models directory."""
        models_dir = Path(__file__).parent.parent / "models" / "tts"
        
        if models_dir.exists():
            # Search for .onnx files recursively
            onnx_files = list(models_dir.rglob("*.onnx"))
            if onnx_files:
                return str(onnx_files[0])
        
        raise FileNotFoundError(
            "No voice model found in models/tts/\n"
            "Download a voice with:\n"
            "  python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('rhasspy/piper-voices', 'en/en_US/amy/medium/en_US-amy-medium.onnx', local_dir='models/tts')\""
        )
    
    def synthesize(
        self, 
        text: str,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to synthesize
            normalize: Whether to normalize text (default: True)
            
        Returns:
            Tuple of (audio_array, sample_rate)
            audio_array: np.ndarray int16, shape (samples,)
        """
        if not text.strip():
            return np.array([], dtype=np.int16), self.sample_rate
        
        # Normalize text
        if normalize:
            text = self._normalizer.normalize(text)
        
        start_time = time.time()
        
        # Import synthesis config
        from piper.config import SynthesisConfig
        
        syn_config = SynthesisConfig(
            speaker_id=self.config.speaker_id,
            length_scale=self.config.length_scale,
            noise_scale=self.config.noise_scale,
            noise_w_scale=self.config.noise_w_scale,
        )
        
        # Synthesize
        audio_chunks = []
        
        for chunk in self._voice.synthesize(text, syn_config=syn_config):
            audio_chunks.append(chunk.audio_int16_array)
        
        # Combine chunks
        if not audio_chunks:
            return np.array([], dtype=np.int16), self.sample_rate
            
        audio_array = np.concatenate(audio_chunks)
        
        # Update stats
        self._stats["syntheses"] += 1
        self._stats["total_chars"] += len(text)
        self._stats["total_time"] += time.time() - start_time
        
        return audio_array, self.sample_rate
    
    def synthesize_to_bytes(
        self,
        text: str,
        normalize: bool = True,
    ) -> bytes:
        """
        Synthesize text and return raw audio bytes (int16 PCM).
        
        Useful for streaming audio output.
        
        Args:
            text: Text to synthesize
            normalize: Whether to normalize text
            
        Returns:
            Raw audio bytes (int16 PCM, mono)
        """
        audio, _ = self.synthesize(text, normalize)
        return audio.tobytes()
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        normalize: bool = True,
    ) -> float:
        """
        Synthesize text and save to WAV file.
        
        Args:
            text: Text to synthesize
            output_path: Path to output WAV file
            normalize: Whether to normalize text
            
        Returns:
            Duration of audio in seconds
        """
        audio, sample_rate = self.synthesize(text, normalize)
        
        # Save as WAV
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
        
        duration = len(audio) / sample_rate
        return duration
    
    def synthesize_sentences(
        self,
        text: str,
        normalize: bool = True,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Synthesize text sentence by sentence.
        
        Useful for streaming output or showing progress.
        
        Args:
            text: Text to synthesize
            normalize: Whether to normalize text
            
        Returns:
            List of (sentence, audio_array) tuples
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        results = []
        for sentence in sentences:
            if sentence.strip():
                audio, _ = self.synthesize(sentence, normalize)
                results.append((sentence, audio))
        
        return results
    
    @property
    def sample_rate(self) -> int:
        """Get output sample rate."""
        return self._voice.config.sample_rate
    
    def get_stats(self) -> dict:
        """Get synthesis statistics."""
        total_time = self._stats["total_time"]
        total_chars = self._stats["total_chars"]
        
        return {
            **self._stats,
            "chars_per_second": total_chars / total_time if total_time > 0 else 0,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "syntheses": 0,
            "total_chars": 0,
            "total_time": 0.0,
        }


def download_voice(
    voice_name: str = "amy",
    quality: str = "medium",
    output_dir: str = "models/tts",
) -> str:
    """
    Download a Piper voice model.
    
    Args:
        voice_name: Voice name (amy, lessac, alba, etc.)
        quality: Quality level (low, medium, high)
        output_dir: Directory to save model
        
    Returns:
        Path to downloaded model file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Install huggingface-hub: pip install huggingface-hub")
    
    # Voice mappings (name -> path prefix)
    voices = {
        "amy": "en/en_US/amy",           # Female, American
        "lessac": "en/en_US/lessac",     # Male, American
        "alba": "en/en_GB/alba",         # Female, British
        "danny": "en/en_GB/danny",       # Male, British
        "ryan": "en/en_US/ryan",         # Male, American (low)
    }
    
    if voice_name not in voices:
        available = ", ".join(voices.keys())
        raise ValueError(f"Unknown voice: {voice_name}. Available: {available}")
    
    voice_path = voices[voice_name]
    model_name = f"{voice_path.split('/')[-1]}"  # e.g., "amy"
    locale = voice_path.split('/')[1]  # e.g., "en_US"
    
    # Full filename
    filename = f"{locale}-{model_name}-{quality}"
    
    print(f"Downloading voice: {filename}")
    
    # Download model and config
    onnx_path = f"{voice_path}/{quality}/{filename}.onnx"
    json_path = f"{voice_path}/{quality}/{filename}.onnx.json"
    
    model_file = hf_hub_download(
        repo_id="rhasspy/piper-voices",
        filename=onnx_path,
        local_dir=output_dir,
    )
    
    hf_hub_download(
        repo_id="rhasspy/piper-voices",
        filename=json_path,
        local_dir=output_dir,
    )
    
    print(f"Downloaded to: {model_file}")
    return model_file


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test TTS module."""
    import sys
    
    print("=" * 60)
    print("TTS MODULE TEST")
    print("=" * 60)
    
    print(f"\nPiper TTS available: {HAS_PIPER}")
    
    if not HAS_PIPER:
        print("Install with: pip install piper-tts")
        sys.exit(1)
    
    # Check for voice model
    model_dir = Path(__file__).parent.parent / "models" / "tts"
    model_files = list(model_dir.rglob("*.onnx")) if model_dir.exists() else []
    
    if not model_files:
        print("\n" + "=" * 60)
        print("NO VOICE MODEL FOUND")
        print("=" * 60)
        print("\nDownload a voice first:")
        print()
        print("Option 1: Use the download function:")
        print("  python -c \"from core.tts import download_voice; download_voice('amy')\"")
        print()
        print("Option 2: Manual download:")
        print("  python -c \"from huggingface_hub import hf_hub_download;")
        print("  hf_hub_download('rhasspy/piper-voices', 'en/en_US/amy/medium/en_US-amy-medium.onnx', local_dir='models/tts');")
        print("  hf_hub_download('rhasspy/piper-voices', 'en/en_US/amy/medium/en_US-amy-medium.onnx.json', local_dir='models/tts')\"")
        print()
        print("Available voices: amy (female), lessac (male), alba (British)")
        sys.exit(0)
    
    print(f"\nFound voice model: {model_files[0].name}")
    
    # Test 1: Load TTS
    print("\n[Test 1] Loading TTS...")
    
    config = TTSConfig(
        model_path=str(model_files[0]),
        length_scale=1.0,
    )
    
    tts = TTS(config)
    
    # Test 2: Simple synthesis
    print("\n[Test 2] Simple synthesis...")
    
    test_text = "Hello! I am your voice assistant."
    start_time = time.time()
    audio, sample_rate = tts.synthesize(test_text)
    synth_time = time.time() - start_time
    
    duration = len(audio) / sample_rate
    print(f"  Text: \"{test_text}\"")
    print(f"  Audio: {duration:.2f}s at {sample_rate} Hz")
    print(f"  Synthesis time: {synth_time:.3f}s")
    print(f"  Real-time factor: {synth_time / duration:.2f}x")
    
    # Test 3: Text normalization
    print("\n[Test 3] Text normalization...")
    
    test_texts = [
        "Dr. Smith costs $50.",
        "The temperature is 23 degrees.",
        "Call me at 5 o'clock.",
        "I'm 100% sure!",
    ]
    
    for text in test_texts:
        normalized = TextNormalizer.normalize(text)
        print(f"  \"{text}\" -> \"{normalized}\"")
    
    # Test 4: Save to file
    print("\n[Test 4] Save to WAV file...")
    
    output_file = Path(__file__).parent.parent / "test_output.wav"
    duration = tts.synthesize_to_file(
        "This is a test of the text to speech system.",
        str(output_file)
    )
    print(f"  Saved: {output_file}")
    print(f"  Duration: {duration:.2f}s")
    
    # Test 5: Sentence-by-sentence
    print("\n[Test 5] Sentence-by-sentence synthesis...")
    
    long_text = "First sentence. Second sentence! Third sentence?"
    sentences = tts.synthesize_sentences(long_text)
    
    for sentence, audio in sentences:
        duration = len(audio) / sample_rate
        print(f"  \"{sentence}\" -> {duration:.2f}s")
    
    # Stats
    stats = tts.get_stats()
    print(f"\n  Stats: {stats['syntheses']} syntheses, {stats['chars_per_second']:.1f} chars/s")
    
    print("\n" + "=" * 60)
    print("TTS TEST COMPLETE")
    print("=" * 60)
    print(f"\nOutput file saved to: {output_file}")
    print("Play it with: aplay test_output.wav (or any audio player)")
