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
        
        # Step 1: STT with language detection
        text, info = self._transcribe(audio)
        if not text:
            return False, 0.0
        
        detected_language = info.get("language", "en")
        print(f"👤 You ({detected_language}): {text}")
        
        # Check for goodbye
        if self._is_goodbye(text):
            return self._handle_goodbye()
        
        # Add to history
        self._add_to_history("user", text)
        
        # Step 2: LLM
        response = self._generate_response(text, detected_language)
        print(f"🤖 Assistant: {response}")
        
        self._add_to_history("assistant", response)
        
        # Step 3: TTS + Play using detected language
        mute_until = self._speak(response, language=detected_language)
        
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

    def _transcribe(self, audio: np.ndarray) -> Tuple[str, Dict]:
        """Transcribe audio to text and detect language."""
        print(f"\rStatus: 🧠 Transcribing...            ", end="", flush=True)
        
        start_time = time.time()
        text, info = self._stt.transcribe_with_info(audio)
        stt_time = time.time() - start_time
        
        if not text.strip():
            print("\r(empty transcription)            ")
            return "", {}
        
        print(f"\r" + " " * 50 + "\r", end="")  # Clear line
        
        if self.config.debug:
            print(f"   [STT: {stt_time:.2f}s]")
        
        return text.strip(), info
    
    def _is_goodbye(self, text: str) -> bool:
        """Check if text contains goodbye phrase."""
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in GOODBYE_PHRASES)
    
    def _handle_goodbye(self) -> Tuple[bool, float]:
        """Handle goodbye response. Synthesizes TTS, calculates mute duration, then plays."""
        goodbye_text = "Goodbye! I'll be here when you need me."
        print(f"🤖 Assistant: {goodbye_text}")
        
        # Synthesize FIRST so we know the exact audio duration
        goodbye_audio, sr = self._tts.synthesize(goodbye_text)
        audio_duration = len(goodbye_audio) / sr
        
        # Set mute to cover the full playback + normal buffer
        mute_until = time.time() + audio_duration + (self.config.mute_during_speech_ms / 1000.0)
        
        # NOW play (caller should have set _muted_until before this returns)
        self._audio_output.play(goodbye_audio, sr)
        
        return True, mute_until
    
    def _generate_response(self, text: str, detected_language: str = "en") -> str:
        """Generate LLM response."""
        print(f"🤖 Thinking...", end="", flush=True)
        
        # Give the LLM a strong hint about the spoken language
        lang_hint = "Vietnamese" if detected_language == "vi" else "English"
        system_prompt = self.config.system_prompt + f"\n\n[DETECTED LANGUAGE: {lang_hint}. YOU MUST REPLY IN {lang_hint}.]"
        
        start_time = time.time()
        response = self._llm.generate(
            text,
            history=self._conversation_history[:-1],  # Exclude current message
            system_prompt=system_prompt,
        )
        llm_time = time.time() - start_time
        
        print(f"\r", end="")  # Clear "Thinking..."
        
        if self.config.debug:
            print(f"   [LLM: {llm_time:.2f}s]")
        
        return response
    
    def _speak(self, text: str, language: str = "en") -> float:
        """Synthesize and play text using the correct language voice. Returns mute_until timestamp."""
        start_time = time.time()
        tts_audio, sr = self._tts.synthesize(text, language=language)
        tts_time = time.time() - start_time
        
        if self.config.debug:
            print(f"   [TTS: {tts_time:.2f}s]")
        
        # Calculate mute BEFORE playing so it covers the actual playback window
        audio_duration = len(tts_audio) / sr
        mute_until = time.time() + audio_duration + (self.config.mute_during_speech_ms / 1000.0)
        
        self._audio_output.play(tts_audio, sr)
        
        return mute_until
    
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
        # Mute for full audio playback duration + buffer
        audio_duration = len(audio) / sr
        return time.time() + audio_duration + (self.config.mute_during_speech_ms / 1000.0)
    
    def play_acknowledgment(self) -> float:
        """
        Speak a TTS acknowledgment when the wake word is triggered.
        
        Returns mute_until timestamp.
        """
        phrase = "I am listening."
        if self.config.debug:
            print(f"   [WakeWord: \"{phrase}\"]")
            
        return self.say(phrase)
    
    def play_thinking_chime(self, skip_speech: bool = False) -> float:
        """
        Speak a short acknowledgment and play a background loop to let the user know the AI is processing.
        
        Returns mute_until timestamp.
        """
        import scipy.io.wavfile as wav
        import time
        from pathlib import Path
        import numpy as np

        end_time = time.time()
        if not skip_speech:
            phrase = "I'm thinking wait for me a moment."
                
            # 1. Speak the acknowledgment (blocking natively)
            end_time = self.say(phrase)
            
            # Wait briefly so it doesn't overlap exactly with the end of speech
            time.sleep(max(0, end_time - time.time() - 0.2))
            
        # 2. Start the background looping sound
        processing_path = _project_root / "assets" / "processing.wav"
        if processing_path.exists():
            try:
                proc_sr, proc_audio = wav.read(str(processing_path))
                if proc_audio.dtype == np.int16:
                    proc_audio = proc_audio.astype(np.float32) / 32768.0
                
                # Loop the processing audio 20 times (approx 1.5 minutes)
                if proc_audio.ndim == 1:
                    looped_audio = np.tile(proc_audio, 20)
                else:
                    looped_audio = np.tile(proc_audio, (20, 1))
                
                # Drop volume slightly for background ambience
                original_volume = self._audio_output.config.volume
                self._audio_output.set_volume(0.3)
                
                # Play non-blocking. When TTS is ready, it will trigger play() which interrupts this.
                self._audio_output.play(looped_audio, proc_sr, blocking=False)
                
                # Restore volume for the TTS
                self._audio_output.set_volume(original_volume)
            except Exception as e:
                print(f"[Audio] Failed to play processing sound: {e}")
                
        return end_time
