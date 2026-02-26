"""
Speech Processor
================

Handles the speech processing pipeline: STT → LLM → TTS
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import time
import wave
import sys

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.config import VoiceAssistantConfig


# Goodbye phrases that end conversation
GOODBYE_PHRASES = [
    "goodbye", "bye bye", "bye", "see you", 
    "that's all", "stop", "go to sleep"
]


class SpeechProcessor:
    """
    Processes speech through the STT → LLM → TTS pipeline.
    
    Also manages conversation history.
    """
    
    def __init__(self, config: VoiceAssistantConfig, stt, llm, tts, audio_output):
        self.config = config
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._audio_output = audio_output
        
        self._conversation_history: List[Dict] = []
    
    @property
    def history(self) -> List[Dict]:
        """Get conversation history copy."""
        return self._conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
    
    def process(self, audio_buffer: List[np.ndarray], record_ts: Optional[int] = None) -> Tuple[bool, float]:
        """
        Process collected speech.
        
        Args:
            audio_buffer: List of audio frames
            record_ts: Optional timestamp for stage recording (saves STT-prepared to recordings/stt/<record_ts>_prepared.wav)
        
        Returns:
            Tuple of (should_end_conversation, mute_until_timestamp)
        """
        # Check minimum speech duration
        if len(audio_buffer) < self.config.min_speech_frames:
            duration_ms = len(audio_buffer) * self.config.frame_ms
            print(f"\r(too short: {duration_ms}ms, need {self.config.min_speech_duration_ms}ms)")
            return False, 0.0
        
        # Combine audio (full capture, no dropping)
        audio = np.concatenate([f.astype(np.float32).flatten() for f in audio_buffer])
        
        # Normalize so STT receives full voice level (consistent level for transcription)
        audio = self._prepare_audio_for_stt(audio)
        
        # Stage recording: save STT-prepared audio for review
        if self.config.save_audio:
            self._save_debug_audio(audio, record_ts=record_ts)
        
        # Step 1: STT
        text = self._transcribe(audio)
        if not text:
            return False, 0.0
        
        print(f"👤 You: {text}")
        
        # Check for goodbye
        if self._is_goodbye(text):
            return self._handle_goodbye()
        
        # Add to history
        self._add_to_history("user", text)
        
        # Step 2: LLM
        response = self._generate_response(text)
        print(f"🤖 Assistant: {response}")
        
        self._add_to_history("assistant", response)
        
        # Step 3: TTS + Play
        mute_until = self._speak(response)
        
        return False, mute_until
    
    def _prepare_audio_for_stt(self, audio: np.ndarray) -> np.ndarray:
        """
        Prepare full captured audio for STT so it receives full voice clearly.
        - Ensure float32, 1D
        - Peak-normalize to ~0.95 so full dynamic range is used (not too quiet)
        - Pad with short silence at start/end so Whisper doesn't cut first/last phoneme
        - Pad to minimum 0.5s if shorter (some models need minimum length)
        """
        if audio is None or audio.size == 0:
            return audio
        audio = np.asarray(audio, dtype=np.float32).flatten()
        sr = self.config.sample_rate
        # Peak normalize so full voice is at good level (avoid too-quiet input)
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            audio = audio / peak * 0.95
        # Pad 80ms silence at start and end (helps Whisper with boundaries)
        pad_samples = int(0.08 * sr)
        audio = np.concatenate([np.zeros(pad_samples, dtype=np.float32), audio, np.zeros(pad_samples, dtype=np.float32)])
        # Pad to minimum 0.5s if shorter (avoids odd behavior on very short clips)
        min_samples = int(0.5 * sr)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode="constant", constant_values=0)
        return audio

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        print(f"\rStatus: 🧠 Transcribing...            ", end="", flush=True)
        
        start_time = time.time()
        text = self._stt.transcribe(audio)
        stt_time = time.time() - start_time
        
        if not text.strip():
            print("\r(empty transcription)            ")
            return ""
        
        print(f"\r" + " " * 50 + "\r", end="")  # Clear line
        
        if self.config.debug:
            print(f"   [STT: {stt_time:.2f}s]")
        
        return text.strip()
    
    def _is_goodbye(self, text: str) -> bool:
        """Check if text contains goodbye phrase."""
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in GOODBYE_PHRASES)
    
    def _handle_goodbye(self) -> Tuple[bool, float]:
        """Handle goodbye response."""
        print("🤖 Assistant: Goodbye! Say the wake word when you need me!")
        
        goodbye_audio, sr = self._tts.synthesize("Goodbye! Say the wake word when you need me!")
        self._audio_output.play(goodbye_audio, sr)
        
        # Mute for full audio duration + buffer
        audio_duration = len(goodbye_audio) / sr
        mute_until = time.time() + audio_duration + 1.0
        
        return True, mute_until
    
    def _generate_response(self, text: str) -> str:
        """Generate LLM response."""
        print(f"🤖 Thinking...", end="", flush=True)
        
        start_time = time.time()
        response = self._llm.generate(
            text,
            history=self._conversation_history[:-1],  # Exclude current message
            system_prompt=self.config.system_prompt,
        )
        llm_time = time.time() - start_time
        
        print(f"\r", end="")  # Clear "Thinking..."
        
        if self.config.debug:
            print(f"   [LLM: {llm_time:.2f}s]")
        
        return response
    
    def _speak(self, text: str) -> float:
        """Synthesize and play text. Returns mute_until timestamp."""
        start_time = time.time()
        tts_audio, sr = self._tts.synthesize(text)
        tts_time = time.time() - start_time
        
        if self.config.debug:
            print(f"   [TTS: {tts_time:.2f}s]")
        
        self._audio_output.play(tts_audio, sr)
        
        # Return mute timestamp
        return time.time() + (self.config.mute_during_speech_ms / 1000.0)
    
    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self._conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Trim to max turns
        max_messages = self.config.max_history_turns * 2
        if len(self._conversation_history) > max_messages:
            self._conversation_history = self._conversation_history[-max_messages:]
    
    def _save_debug_audio(self, audio: np.ndarray, record_ts: Optional[int] = None) -> None:
        """Save audio for debugging. If record_ts given, save to recordings/stt/<record_ts>_prepared.wav."""
        if record_ts is not None:
            recordings_stt = _project_root / "recordings" / "stt"
            recordings_stt.mkdir(parents=True, exist_ok=True)
            path = recordings_stt / f"{record_ts}_prepared.wav"
        else:
            path = Path(f"debug_audio_{int(time.time())}.wav")
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.config.sample_rate)
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            f.writeframes(audio_int16.tobytes())
    
    def say(self, text: str) -> float:
        """
        Speak text immediately (for greetings, acknowledgments).
        
        Returns mute_until timestamp.
        """
        audio, sr = self._tts.synthesize(text)
        self._audio_output.play(audio, sr)
        return time.time() + (self.config.mute_during_speech_ms / 1000.0)
    
    def play_acknowledgment(self) -> float:
        """
        Play wake word acknowledgment sound.
        
        Returns mute_until timestamp.
        """
        import scipy.io.wavfile as wav
        
        ack_path = Path(__file__).parent.parent / "assets" / "wake_ack.wav"
        
        if ack_path.exists():
            ack_sr, ack_audio = wav.read(str(ack_path))
            if ack_audio.dtype == np.int16:
                ack_audio = ack_audio.astype(np.float32) / 32768.0
            self._audio_output.play(ack_audio, ack_sr)
            return time.time() + 0.3  # Short mute after ack
        
        # No ack file - just return with minimal mute
        return time.time() + 0.1
