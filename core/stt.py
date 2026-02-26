"""
Voice Assistant v2 - Speech-to-Text Module
==========================================

Step 5: Convert audio to text using faster-whisper.

Optimized for low RAM (6GB total):
- Use "tiny" or "tiny.en" model (~400MB)
- int8 quantization
- CPU optimized

Features:
- Fast CPU inference with CTranslate2 backend
- Low RAM usage (~400MB for tiny model)
- Optional language detection
- Partial transcription for streaming

Model sizes:
- tiny: ~400MB RAM, fastest, good quality
- tiny.en: ~400MB RAM, faster (English only), better English
- base: ~500MB RAM, medium speed, better quality
- small: ~1GB RAM, slower, best quality
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import numpy as np
import time
import os

# Import faster-whisper
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    print("[STT] Warning: faster-whisper not installed")


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text."""
    model_size: str = "tiny"        # tiny, tiny.en, base, small
    device: str = "cpu"             # cpu or cuda
    compute_type: str = "int8"      # int8, float16, float32
    beam_size: int = 1              # 1 for speed, 3-5 for accuracy
    language: Optional[str] = "en"  # None for auto-detect
    cpu_threads: int = 4            # CPU threads to use
    vad_filter: bool = False        # We do our own VAD
    download_root: Optional[str] = None  # Model download directory
    
    # Enhancement options for non-native speakers
    enhance_audio: bool = False     # Apply audio enhancement
    correct_text: bool = False      # Apply text post-processing
    best_of: int = 1                # Candidates per temperature (more = better for accents)
    temperature: float = 0.0        # 0 = greedy, >0 = sampling (can help with accents)


@dataclass 
class STTConfigForAccents(STTConfig):
    """
    Pre-configured STT settings optimized for non-native speakers.
    
    Uses distil-large-v3 - a multilingual model that handles accents much better.
    This is a distilled (faster) version of large-v3 with similar accuracy.
    
    Key improvements for non-native speakers:
    - Multilingual training data includes many accents
    - Larger model = better understanding of pronunciation variations
    - beam_size=3 considers more transcription options
    - temperature=0.2 allows slight variations (helps with accents)
    - best_of=3 picks best from multiple attempts
    
    RAM usage: ~2GB for model
    """
    model_size: str = "distil-large-v3"  # Best for accents, multilingual
    beam_size: int = 3              # More search candidates (balance speed/accuracy)
    language: Optional[str] = "en"  # Still English, but model understands accents
    enhance_audio: bool = True      # Pre-process audio
    correct_text: bool = True       # Post-process text
    best_of: int = 3                # Multiple candidates
    temperature: float = 0.2        # Slight sampling helps with accent variations
    

class STT:
    """
    Speech-to-Text using faster-whisper.
    
    Features:
    - Fast CPU inference with int8 quantization
    - Low RAM usage (~400MB for tiny model)
    - Optional language detection
    - Partial transcription for responsive UI
    
    Usage:
        stt = STT(STTConfig())
        
        # Simple transcription
        text = stt.transcribe(audio_array)
        
        # With metadata
        text, info = stt.transcribe_with_info(audio_array)
        
        # Partial/streaming (faster, less accurate)
        partial_text = stt.transcribe(audio_array, partial=True)
    """
    
    def __init__(self, config: Optional[STTConfig] = None):
        if not HAS_FASTER_WHISPER:
            raise ImportError(
                "faster-whisper not installed. "
                "Run: pip install faster-whisper"
            )
        
        self.config = config or STTConfig()
        
        # Determine model download path
        if self.config.download_root:
            download_root = self.config.download_root
        else:
            # Default to models/stt in project directory
            project_root = Path(__file__).parent.parent
            download_root = str(project_root / "models" / "stt")
        
        # Create directory if needed
        os.makedirs(download_root, exist_ok=True)
        
        print(f"[STT] Loading model '{self.config.model_size}'...")
        print(f"      Device: {self.config.device}, Compute: {self.config.compute_type}")
        
        load_start = time.time()
        
        self._model = WhisperModel(
            model_size_or_path=self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads,
            download_root=download_root,
        )
        
        load_time = time.time() - load_start
        print(f"[STT] Model loaded in {load_time:.1f}s")
        
        # Initialize audio enhancer if enabled
        self._enhancer = None
        self._corrector = None
        
        if self.config.enhance_audio:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
                from audio_enhance import AudioEnhancer, AudioEnhanceConfig
                
                self._enhancer = AudioEnhancer(
                    AudioEnhanceConfig(
                        normalize=True,
                        noise_reduce=True,
                        compress=True,
                        clarity_boost=True,
                        highpass=True,
                        lowpass=True,
                    ),
                    sample_rate=16000
                )
                print("[STT] Audio enhancement enabled")
            except ImportError as e:
                print(f"[STT] Warning: Audio enhancement unavailable - {e}")
        
        if self.config.correct_text:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
                from audio_enhance import TextCorrector
                
                self._corrector = TextCorrector()
                print("[STT] Text correction enabled")
            except ImportError as e:
                print(f"[STT] Warning: Text correction unavailable - {e}")
        
        # Statistics
        self._stats = {
            "transcriptions": 0,
            "total_audio_seconds": 0.0,
            "total_processing_seconds": 0.0,
        }
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio enhancement if enabled."""
        if self._enhancer is not None:
            return self._enhancer.enhance(audio)
        return audio
    
    def _postprocess_text(self, text: str) -> str:
        """Apply text correction if enabled."""
        if self._corrector is not None:
            return self._corrector.correct(text)
        return text
    
    def transcribe(self, audio: np.ndarray, partial: bool = False) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: np.ndarray float32, shape (samples,) at 16kHz
            partial: If True, optimize for speed over accuracy
                     (use for streaming/live transcription)
            
        Returns:
            Transcribed text string
        """
        # Handle empty audio
        if audio is None or audio.size == 0:
            return ""
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Apply audio enhancement (if enabled and not partial)
        if not partial:
            audio = self._preprocess_audio(audio)
        
        # Track timing
        start_time = time.time()
        audio_duration = len(audio) / 16000  # Assume 16kHz
        
        # Configure for partial vs full transcription
        beam_size = 1 if partial else self.config.beam_size
        best_of = 1 if partial else self.config.best_of
        temperature = 0.0 if partial else self.config.temperature
        
        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                language=self.config.language,
                vad_filter=self.config.vad_filter,
                word_timestamps=False,  # Faster without timestamps
                condition_on_previous_text=not partial,  # Disable for streaming
            )
            
            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            text = " ".join(text_parts).strip()
            
            # Apply text correction (if enabled and not partial)
            if not partial:
                text = self._postprocess_text(text)
            
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            text = ""
        
        # Update statistics
        process_time = time.time() - start_time
        self._stats["transcriptions"] += 1
        self._stats["total_audio_seconds"] += audio_duration
        self._stats["total_processing_seconds"] += process_time
        
        return text
    
    def transcribe_with_info(
        self, 
        audio: np.ndarray
    ) -> Tuple[str, Dict]:
        """
        Transcribe with additional info (language, confidence, timing).
        
        Args:
            audio: np.ndarray float32, shape (samples,) at 16kHz
            
        Returns:
            Tuple of (text, info_dict) where info_dict contains:
            - language: detected/specified language
            - language_probability: confidence in language detection
            - duration: audio duration in seconds
            - processing_time: transcription time in seconds
            - real_time_factor: processing_time / duration
        """
        # Handle empty audio
        if audio is None or audio.size == 0:
            return "", {"error": "Empty audio"}
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Apply audio enhancement
        original_audio = audio.copy()
        audio = self._preprocess_audio(audio)
        enhanced = self._enhancer is not None
        
        start_time = time.time()
        audio_duration = len(audio) / 16000
        
        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=self.config.temperature,
                language=self.config.language if self.config.language else None,
                vad_filter=self.config.vad_filter,
                word_timestamps=False,
            )
            
            # Collect segments
            text_parts = []
            segment_list = []
            for segment in segments:
                text_parts.append(segment.text)
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                })
            
            raw_text = " ".join(text_parts).strip()
            
            # Apply text correction
            text = self._postprocess_text(raw_text)
            corrected = text != raw_text
            
            process_time = time.time() - start_time
            
            result_info = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "processing_time": process_time,
                "real_time_factor": process_time / audio_duration if audio_duration > 0 else 0,
                "segments": segment_list,
                "enhanced": enhanced,
                "corrected": corrected,
                "raw_text": raw_text if corrected else None,
            }
            
        except Exception as e:
            text = ""
            result_info = {"error": str(e)}
        
        # Update statistics
        self._stats["transcriptions"] += 1
        self._stats["total_audio_seconds"] += audio_duration
        self._stats["total_processing_seconds"] += time.time() - start_time
        
        return text, result_info
    
    def transcribe_segments(
        self, 
        audio: np.ndarray
    ) -> List[Dict]:
        """
        Transcribe and return individual segments with timestamps.
        
        Useful for subtitles or detailed analysis.
        
        Args:
            audio: np.ndarray float32, shape (samples,) at 16kHz
            
        Returns:
            List of segment dictionaries with start, end, text
        """
        if audio is None or audio.size == 0:
            return []
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        try:
            segments, _ = self._model.transcribe(
                audio,
                beam_size=self.config.beam_size,
                language=self.config.language,
                word_timestamps=False,
            )
            
            return [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
                for seg in segments
            ]
            
        except Exception as e:
            print(f"[STT] Segment transcription error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get transcription statistics.
        
        Returns:
            Dict with transcription stats including real-time factor
        """
        total_audio = self._stats["total_audio_seconds"]
        total_process = self._stats["total_processing_seconds"]
        
        return {
            **self._stats,
            "average_real_time_factor": total_process / total_audio if total_audio > 0 else 0,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "transcriptions": 0,
            "total_audio_seconds": 0.0,
            "total_processing_seconds": 0.0,
        }


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test STT module."""
    import time
    
    print("=" * 60)
    print("SPEECH-TO-TEXT TEST")
    print("=" * 60)
    
    print(f"\nfaster-whisper available: {HAS_FASTER_WHISPER}")
    
    if not HAS_FASTER_WHISPER:
        print("Install with: pip install faster-whisper")
        exit(1)
    
    # Test 1: Load model
    print("\n[Test 1] Loading STT model...")
    
    config = STTConfig(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        beam_size=1,
        language="en",
        cpu_threads=4,
    )
    
    stt = STT(config)
    
    # Test 2: Transcribe synthetic audio (silence)
    print("\n[Test 2] Transcribing silence...")
    
    silence = np.zeros(16000, dtype=np.float32)  # 1 second
    text = stt.transcribe(silence)
    print(f"  Result: '{text}' (expected empty or noise)")
    
    # Test 3: Live recording test
    print("\n[Test 3] Live recording test...")
    
    try:
        from audio_input import AudioInput, AudioInputConfig
        from vad import VAD, VADConfig
        
        print("  Recording for 5 seconds...")
        print("  Speak clearly into the microphone!")
        print()
        
        audio_input = AudioInput(AudioInputConfig(sample_rate=16000, frame_ms=20))
        vad = VAD(VADConfig(
            aggressiveness=3,
            energy_threshold=0.035,
            smooth_window=8,
            hangover_frames=10,
        ))
        
        audio_input.start()
        time.sleep(0.2)
        
        # Collect audio
        frames = []
        speech_frames = []
        in_speech = False
        
        start_time = time.time()
        while time.time() - start_time < 5.0:
            frame = audio_input.get_frame(timeout=0.05)
            if frame is not None:
                frames.append(frame)
                is_speech = vad.is_speech(frame)
                
                if is_speech:
                    speech_frames.append(frame)
                    if not in_speech:
                        in_speech = True
                else:
                    in_speech = False
                
                # Visual feedback
                rms = vad.compute_rms(frame)
                bar = "█" * int(rms * 150)
                status = "🗣️" if is_speech else "  "
                print(f"\r  {status} {bar:<30}", end="", flush=True)
        
        audio_input.stop()
        print()
        
        if speech_frames:
            # Transcribe speech only
            audio = np.concatenate(speech_frames)
            print(f"\n  Transcribing {len(audio)/16000:.1f}s of speech...")
            
            text, info = stt.transcribe_with_info(audio)
            
            print(f"\n  ╔{'═'*56}╗")
            print(f"  ║ Transcription: {text[:50]:<40} ║")
            print(f"  ╠{'═'*56}╣")
            print(f"  ║ Language: {info.get('language', 'N/A'):<20} Confidence: {info.get('language_probability', 0):.1%}      ║")
            print(f"  ║ Duration: {info.get('duration', 0):.2f}s                Processing: {info.get('processing_time', 0):.2f}s       ║")
            print(f"  ║ Real-time factor: {info.get('real_time_factor', 0):.2f}x                          ║")
            print(f"  ╚{'═'*56}╝")
        else:
            print("\n  No speech detected!")
        
        # Stats
        stats = stt.get_stats()
        print(f"\n  Stats:")
        print(f"    Transcriptions: {stats['transcriptions']}")
        print(f"    Avg RTF: {stats['average_real_time_factor']:.2f}x")
        
    except ImportError as e:
        print(f"  Skipped - {e}")
        print("  Run from project root to test with live audio")
    
    print("\n" + "=" * 60)
    print("STT TEST COMPLETE")
    print("=" * 60)
