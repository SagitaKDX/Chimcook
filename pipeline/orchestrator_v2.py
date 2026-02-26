"""
Voice Assistant v2 - Main Orchestrator
======================================

Simplified orchestrator using DIRECT Silero VAD for reliable speech detection.

Pipeline:
1. Wake word detection (if enabled)
2. Silero VAD detects speech start → start recording
3. Silero VAD detects silence → stop recording
4. noisereduce cleans the captured audio
5. Send cleaned audio to STT → LLM → TTS

This file contains only the main loop logic.
"""

from typing import List, Optional
import numpy as np
import time
import sys
import signal
import threading
from collections import deque
from pathlib import Path
import json

# Add project root to path for direct script execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Add pipeline folder to path for local imports
_pipeline_dir = Path(__file__).parent
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

import torch

# Import from pipeline folder (use explicit path to avoid conflict with config/ folder)
from pipeline.config import (
    VoiceAssistantConfig, 
    AssistantState, 
    STATE_DISPLAY,
    get_wake_phrase,
)
from pipeline.components import ComponentManager
from pipeline.wake_word_handler import WakeWordHandler
from pipeline.speech_processor import SpeechProcessor

# Debug logging (NDJSON) for VAD visualization
DEBUG_LOG_PATH = _project_root / ".cursor" / "debug-b7d16a.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "vad-meter") -> None:
    """Append one NDJSON log line for this debug session."""
    try:
        # Ensure debug directory exists
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "sessionId": "b7d16a",
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with DEBUG_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Logging must never break the app
        pass

# =============================================================================
# SILERO VAD CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
FRAME_MS = 32  # Silero works best with 32ms frames (512 samples at 16kHz)
SILERO_THRESHOLD = 0.45  # Speech probability threshold (lower = more sensitive)
SILENCE_TIMEOUT_MS = 1000  # End speech after this much silence (was 600)
MIN_SPEECH_MS = 200  # Minimum speech duration to process (was 300)



class VoiceAssistant:
    """
    Main voice assistant orchestrator.
    
    Usage:
        config = VoiceAssistantConfig()
        assistant = VoiceAssistant(config)
        assistant.run()  # Blocking main loop
    """
    
    def __init__(self, config: VoiceAssistantConfig):
        self.config = config
        self._state = AssistantState.IDLE
        self._running = False
        self._muted_until = 0.0
        
        # Initialize all components
        self._components = ComponentManager(config)
        self._components.initialize_all()
        
        # Create wake word handler
        if config.enable_wake_word and self._components.wake_word_model:
            self._wake_word = WakeWordHandler(
                config,
                self._components.wake_word_model,
                self._components.wake_word_name,
            )
        else:
            self._wake_word = None
        
        # Create speech processor
        self._speech = SpeechProcessor(
            config,
            self._components.stt,
            self._components.llm,
            self._components.tts,
            self._components.audio_output,
        )
        
        # Face tracking state
        self._face_greeted = False
        self._last_face_count = 0
        self._face_window_until = 0.0  # Accept wake word until this time if face was seen
        self._session_until = 0.0  # Don't greet again until this time expires
        self._face_gone_since = 0.0  # Track when face disappeared
        self._session_duration = 30.0  # Seconds before allowing another greeting
        # Presence state for smoother face gating (used only when enabled)
        self._is_user_present = False
        self._last_face_seen = 0.0
        self._face_timeout_sec = getattr(config, "face_window_sec", 5.0)
        self._face_thread_stop = threading.Event()
        self._face_thread: Optional[threading.Thread] = None
        
        # Load Silero VAD directly (proven to work in test)
        print("[+] Loading Silero VAD...")
        self._silero_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        print("    ✓ Silero VAD loaded")
        
        # Timing parameters
        self._frame_ms = FRAME_MS
        self._frame_samples = SAMPLE_RATE * FRAME_MS // 1000  # 512 samples
        self._silence_frames = SILENCE_TIMEOUT_MS // FRAME_MS  # ~19 frames
        self._min_speech_frames = MIN_SPEECH_MS // FRAME_MS  # ~9 frames
        self._max_frames = int(config.max_speech_duration_sec * 1000 / FRAME_MS)
        
        # Print instructions
        if config.enable_wake_word:
            phrase = get_wake_phrase(self._components.wake_word_name)
            print(f"\nSay: \"{phrase}\" to activate")
        print("Press Ctrl+C to stop.\n")

        # Start face detection thread (decoupled) if enabled
        if config.enable_face_detection and self._components.face_detector:
            self._start_face_thread()
    
    def run(self) -> None:
        """Main loop: continuously process audio and respond."""
        self._running = True
        
        # Use 80ms frames (1280 samples) - works for wake word
        # We'll process in smaller chunks for Silero VAD
        import sounddevice as sd
        import scipy.signal
        
        CHUNK_MS = 80  # 80ms = 1280 samples - good for wake word
        PROCESS_SR = SAMPLE_RATE  # all downstream expects 16k
        
        # Some USB mics (e.g. M70) reject 16k capture; prefer device default (often 48k) and resample.
        input_device = self.config.audio_device
        input_sr = PROCESS_SR
        try:
            if input_device is not None:
                dev = sd.query_devices(input_device)
                dev_default_sr = int(dev.get("default_samplerate", 0) or 0) if isinstance(dev, dict) else 0
                # If device default is set and differs from 16k, use device default to avoid paInvalidSampleRate spam
                if dev_default_sr and dev_default_sr != PROCESS_SR:
                    input_sr = dev_default_sr
        except Exception:
            pass
        
        input_chunk_samples = int(input_sr * CHUNK_MS / 1000)
        process_chunk_samples = int(PROCESS_SR * CHUNK_MS / 1000)  # 1280 @16k
        
        def resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
            """High quality resample to 16k using polyphase."""
            if sr_in == PROCESS_SR:
                y = x
            else:
                y = scipy.signal.resample_poly(x, up=PROCESS_SR, down=sr_in).astype(np.float32)
            # Ensure exact chunk length for wake word path
            if y.size != process_chunk_samples:
                if y.size > process_chunk_samples:
                    y = y[:process_chunk_samples]
                else:
                    y = np.pad(y, (0, process_chunk_samples - y.size))
            return y.astype(np.float32)
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: self._signal_handler())
        
        print("Silero VAD ready - no calibration needed!\n")
        print(f"Using single audio stream: {CHUNK_MS}ms chunks (input_sr={input_sr}, process_sr={PROCESS_SR})")
        
        # Main loop state variables
        collected_frames: List[np.ndarray] = []
        is_recording = False
        silence_frame_count = 0
        raw_chunks_for_utterance: List[np.ndarray] = []  # raw mic during utterance (when save_audio)
        
        # Buffer for Silero (needs 512 samples = 32ms)
        silero_buffer = np.zeros(0, dtype=np.float32)
        
        # Recordings directory for stage review (raw, VAD input, STT input)
        recordings_dir = Path(_project_root) / "recordings"
        if self.config.save_audio:
            for sub in ("raw", "vad", "stt"):
                (recordings_dir / sub).mkdir(parents=True, exist_ok=True)
            print(f"Recordings: {recordings_dir}/ (raw, vad, stt)")
        
        # Initial state
        if self.config.enable_wake_word:
            self._state = AssistantState.WAKE_WORD_LISTENING
        else:
            self._state = AssistantState.IDLE
            if self._wake_word:
                self._wake_word.activate()
        
        self._print_status()
        
        try:
            # Single audio stream with blocking reads
            def open_stream(sr: int):
                return sd.InputStream(
                    samplerate=sr,
                    device=input_device,
                    channels=1,
                    dtype="float32",
                    blocksize=int(sr * CHUNK_MS / 1000),
                    latency="low",
                )
            
            try:
                stream_cm = open_stream(input_sr)
            except Exception:
                # Fallback to device default samplerate (often 48000)
                if input_device is not None:
                    dev = sd.query_devices(input_device)
                    fallback_sr = int(dev.get("default_samplerate", 48000) or 48000) if isinstance(dev, dict) else 48000
                else:
                    fallback_sr = 48000
                input_sr = fallback_sr
                input_chunk_samples = int(input_sr * CHUNK_MS / 1000)
                stream_cm = open_stream(input_sr)
            
            with stream_cm as stream:
                channels=1,
                print("Audio stream started (single stream for all)")
                
                while self._running:
                    # Blocking read - get one 80ms chunk
                    audio_data, overflowed = stream.read(input_chunk_samples)
                    if overflowed:
                        continue
                    
                    # Convert to 1D float32
                    chunk_in = audio_data.flatten().astype(np.float32)
                    chunk = resample_to_16k(chunk_in, input_sr)
                    # Apply mic gain boost at 16k stage (pre wakeword/VAD)
                    if getattr(self.config, "mic_gain", 1.0) != 1.0:
                        chunk = (chunk * float(self.config.mic_gain)).clip(-1.0, 1.0).astype(np.float32)
                    
                    # Skip if muted
                    current_time = time.time()
                    if current_time < self._muted_until:
                        continue
                    
                    # Face detection is now decoupled from the main audio pipeline.
                    # Any camera/face issues will NOT stop wake word or VAD/STT flow.
                    # Optional gating: only block NEW wake word if face gating is enabled
                    gate_allows_wake = True
                    if self.config.require_face_for_wake_word and self._wake_word and not self._wake_word.is_active:
                        gate_allows_wake = self._is_user_present
                    if not gate_allows_wake:
                        # Ignore this chunk for wake word; still keep VAD buffer alive
                        silero_buffer = np.concatenate([silero_buffer, chunk])
                        continue
                    
                    # === WAKE WORD (uses full 80ms chunk) ===
                    # Pass face_detected=True so wake word is fully audio-driven now.
                    if not self._handle_wake_word(chunk, face_detected=True):
                        collected_frames = []
                        raw_chunks_for_utterance = []
                        is_recording = False
                        silence_frame_count = 0
                        silero_buffer = np.zeros(0, dtype=np.float32)
                        continue
                    
                    # Check wake word timeout
                    if self._check_wake_word_timeout():
                        if is_recording and len(collected_frames) > 0:
                            print(f"\n⏰ Processing speech before timeout...")
                            self._process_collected_speech(collected_frames, raw_chunks_for_utterance, recordings_dir)
                        collected_frames = []
                        raw_chunks_for_utterance = []
                        is_recording = False
                        silence_frame_count = 0
                        silero_buffer = np.zeros(0, dtype=np.float32)
                        continue
                    
                    # Accumulate raw mic for stage recording (when save_audio)
                    if is_recording and self.config.save_audio:
                        raw_chunks_for_utterance.append(chunk.copy())
                    
                    # === SILERO VAD (needs 512 sample chunks) ===
                    # Add chunk to silero buffer and process in 512-sample pieces
                    silero_buffer = np.concatenate([silero_buffer, chunk])
                    
                    while len(silero_buffer) >= 512:
                        frame_512 = silero_buffer[:512].copy()
                        silero_buffer = silero_buffer[512:]
                        
                        speech_prob = self._silero_vad_predict(frame_512)
                        is_speech = speech_prob >= SILERO_THRESHOLD

                        # VAD volume visualization & logging when debug enabled
                        if self.config.debug:
                            rms = float(np.sqrt(np.mean(frame_512.astype(np.float64) ** 2)))
                            bar_len = min(30, int(rms * 200))
                            bar = "█" * bar_len + "░" * (30 - bar_len)
                            label = "SPEECH" if is_speech else "silence"
                            prob_pct = int(speech_prob * 100)
                            # Simple console meter (overwrites same line)
                            print(f"\r  {label:8}  prob={prob_pct:3}%  [{bar}]  rms={rms:.4f}  ", end="", flush=True)
                            # #region agent log
                            _agent_log(
                                hypothesis_id="VAD_LEVEL",
                                location="pipeline/orchestrator_v2.py:VAD_LOOP",
                                message="vad_level",
                                data={"rms": rms, "speech_prob": float(speech_prob), "is_speech": bool(is_speech)},
                            )
                            # #endregion
                        
                        # === SPEECH COLLECTION ===
                        if is_speech:
                            if self._wake_word:
                                self._wake_word.extend_timeout()
                            
                            if not is_recording:
                                is_recording = True
                                silence_frame_count = 0
                                collected_frames = [frame_512]
                                if self.config.save_audio:
                                    raw_chunks_for_utterance.append(chunk.copy())
                                print(f"\r🎤 Listening...", end="", flush=True)
                            else:
                                collected_frames.append(frame_512)
                            silence_frame_count = 0
                            
                            if len(collected_frames) >= self._max_frames:
                                print(f"\n⚠️ Max duration")
                                self._process_collected_speech(collected_frames, raw_chunks_for_utterance, recordings_dir)
                                collected_frames = []
                                raw_chunks_for_utterance = []
                                is_recording = False
                                silence_frame_count = 0
                                self._silero_model.reset_states()
                        
                        elif is_recording:
                            collected_frames.append(frame_512)
                            silence_frame_count += 1
                            
                            if silence_frame_count >= self._silence_frames:
                                is_recording = False
                                
                                speech_frames = len(collected_frames) - silence_frame_count
                                if speech_frames >= self._min_speech_frames:
                                    print()
                                    
                                    should_end = self._process_collected_speech(collected_frames, raw_chunks_for_utterance, recordings_dir)
                                    
                                    raw_chunks_for_utterance = []
                                    if should_end:
                                        if self._wake_word:
                                            self._wake_word.deactivate(with_cooldown=False)
                                        self._state = AssistantState.WAKE_WORD_LISTENING
                                        self._print_status()
                                    elif self._wake_word:
                                        self._wake_word.extend_timeout()
                                        self._state = AssistantState.IDLE
                                        self._print_status("listening for follow-up, say 'goodbye' to end")
                                    else:
                                        self._state = AssistantState.IDLE
                                
                                collected_frames = []
                                raw_chunks_for_utterance = []
                                silence_frame_count = 0
                                self._silero_model.reset_states()
        
        finally:
            if self._wake_word:
                self._wake_word.stop()
            self._components.stop()
    
    def _signal_handler(self) -> None:
        """Handle interrupt signal."""
        print("\n\nReceived interrupt signal...")
        self._running = False
    
    def _start_face_thread(self) -> None:
        """Start background face detection thread for presence tracking."""
        if self._face_thread is not None:
            return
        def loop():
            detector = self._components.face_detector
            if not detector:
                return
            interval = max(0.1, self.config.face_detection_interval_ms / 1000.0)
            while not self._face_thread_stop.is_set():
                try:
                    result = detector.process_frame()
                    now = time.time()
                    if result.face_count >= 1:
                        self._last_face_seen = now
                        if not self._is_user_present and self.config.debug:
                            print("[FaceThread] User present")
                        self._is_user_present = True
                    else:
                        # Only mark user gone after timeout (face-lock timeout)
                        if self._is_user_present and (now - self._last_face_seen) > self._face_timeout_sec:
                            if self.config.debug:
                                print("[FaceThread] User gone (timeout)")
                            self._is_user_present = False
                    time.sleep(interval)
                except Exception as e:
                    if self.config.debug:
                        print(f"[FaceThread] Error: {e}")
                    time.sleep(1.0)
        self._face_thread = threading.Thread(target=loop, daemon=True)
        self._face_thread.start()
    
    def _silero_vad_predict(self, frame: np.ndarray) -> float:
        """
        Get speech probability from Silero VAD.
        
        Args:
            frame: float32 audio, should be 512 samples for 16kHz
            
        Returns:
            Speech probability (0.0 to 1.0)
        """
        # Silero expects 512 samples at 16kHz (32ms)
        if len(frame) != 512:
            if len(frame) < 512:
                frame = np.pad(frame, (0, 512 - len(frame)))
            else:
                frame = frame[:512]
        
        # Ensure float32
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(frame).float()
        
        # Get prediction
        with torch.no_grad():
            speech_prob = self._silero_model(audio_tensor.unsqueeze(0), SAMPLE_RATE).item()
        
        return speech_prob

    def _handle_face_detection(self) -> bool:
        """Handle face detection and greeting. Returns True if face detected. Updates face window for wake word."""
        if not self._components.face_detector:
            return True  # No face detection = always "detected"
        
        # Only check face every 500ms to avoid slowing down audio loop
        now = time.time()
        if not hasattr(self, '_last_face_check'):
            self._last_face_check = 0
            self._cached_face_detected = False
        
        if now - self._last_face_check < 0.5:
            # Use cached result
            return self._cached_face_detected
        
        self._last_face_check = now
        
        face_result = self._components.face_detector.process_frame()
        # One or more faces = face detected (opens face window)
        face_detected = face_result.face_count >= 1
        self._cached_face_detected = face_detected
        
        if self.config.debug:
            if not hasattr(self, '_last_face_debug') or now - self._last_face_debug >= 2.0:
                self._last_face_debug = now
                print(f"[DEBUG] face_detection: face_count={face_result.face_count} face_detected={face_detected} window_until={self._face_window_until:.1f}")
        
        # When face detected, extend the window during which wake word is accepted
        if face_detected:
            self._face_window_until = now + self.config.face_window_sec
            self._face_gone_since = 0.0  # Reset "face gone" timer
        else:
            # Track when face disappeared
            if self._face_gone_since == 0.0 and self._last_face_count > 0:
                self._face_gone_since = now
        
        # Check if we should greet
        # Conditions to greet:
        # 1. Face is detected
        # 2. Haven't greeted yet (or session expired and face was gone for a while)
        # 3. greet_on_face is enabled
        # 4. NOT during active wake word session (don't interrupt conversation)
        # 5. NOT during LISTENING/PROCESSING/SPEAKING states
        can_greet = (
            face_detected
            and self.config.greet_on_face
            and not self._face_greeted
            and (self._wake_word is None or not self._wake_word.is_active)
            and self._state in (AssistantState.IDLE, AssistantState.WAKE_WORD_LISTENING)
        )
        
        if can_greet:
            self._face_greeted = True
            self._session_until = now + self._session_duration  # Start 30s session
            greeting = "Hello there, I am Alexa. Say my name to listen to the command."
            print(f"\n👋 {greeting}")
            
            self._state = AssistantState.SPEAKING
            self._muted_until = self._speech.say(greeting)
            self._state = AssistantState.WAKE_WORD_LISTENING if self.config.enable_wake_word else AssistantState.IDLE
            self._print_status("say wake word to begin")
        
        # Reset greeting only when:
        # 1. Session has expired (30s passed since last greeting)
        # 2. Face has been gone for at least 3 seconds
        session_expired = now > self._session_until
        face_gone_long_enough = self._face_gone_since > 0 and (now - self._face_gone_since) > 3.0
        
        if session_expired and face_gone_long_enough:
            if self._face_greeted:
                self._face_greeted = False
                if self.config.debug:
                    print("\n(Session expired and face gone, ready to greet again)")
        
        self._last_face_count = face_result.face_count
        return face_detected
    
    def _handle_wake_word(self, frame: np.ndarray, face_detected: bool) -> bool:
        """
        Handle wake word detection.
        
        Returns True if wake word is active (should collect speech).
        """
        if not self._wake_word:
            return True  # No wake word = always active
        
        if self._wake_word.is_active:
            return True  # Already active
        
        # Check for wake word
        score = self._wake_word.process_frame(frame, face_detected)
        
        if score is not None:
            print(f"\n✨ Wake word detected! (confidence: {score:.2f})")
            self._wake_word.activate()
            
            # Extend session to prevent greeting during conversation
            self._session_until = time.time() + self._session_duration
            
            # Play acknowledgment
            self._state = AssistantState.SPEAKING
            self._muted_until = self._speech.play_acknowledgment()
            
            self._state = AssistantState.IDLE
            self._print_status("listening for your question...")
            return True
        
        return False  # Wake word not detected
    
    def _check_wake_word_timeout(self) -> bool:
        """Check and handle wake word timeout. Returns True if timed out."""
        if not self._wake_word or not self._wake_word.is_active:
            return False
        
        if self._wake_word.check_timeout():
            print("\n⏰ Wake word timeout. Going back to sleep...")
            self._wake_word.deactivate(with_cooldown=True)  # Enable cooldown
            self._state = AssistantState.WAKE_WORD_LISTENING
            self._print_status()
            
            # Reset Silero VAD state
            self._silero_model.reset_states()
            
            return True
        
        return False
    
    def _save_stage_recordings(
        self,
        record_ts: int,
        raw_chunks: List[np.ndarray],
        collected_frames: List[np.ndarray],
        recordings_dir: Path,
    ) -> None:
        """Save per-stage recordings: raw mic, VAD segment, for review."""
        import wave
        sr = SAMPLE_RATE
        def write_wav(path: Path, audio: np.ndarray) -> None:
            audio = np.asarray(audio, dtype=np.float32).flatten()
            if audio.size == 0:
                return
            p = path.parent
            p.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sr)
                f.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
        # 1. Raw mic (chunks that corresponded to this utterance)
        if raw_chunks:
            raw_audio = np.concatenate([c.flatten().astype(np.float32) for c in raw_chunks])
            write_wav(recordings_dir / "raw" / f"{record_ts}.wav", raw_audio)
        # 2. VAD segment (512-sample frames that were sent to STT pipeline)
        if collected_frames:
            vad_audio = np.concatenate([f.flatten().astype(np.float32) for f in collected_frames])
            write_wav(recordings_dir / "vad" / f"{record_ts}.wav", vad_audio)
    
    def _process_collected_speech(
        self,
        collected_frames: List[np.ndarray],
        raw_chunks_for_utterance: Optional[List[np.ndarray]] = None,
        recordings_dir: Optional[Path] = None,
    ) -> bool:
        """
        Process collected speech through STT → LLM → TTS.
        Optionally save per-stage recordings (raw, VAD, STT) when save_audio and recordings_dir provided.
        
        Returns True if conversation should end.
        """
        self._state = AssistantState.PROCESSING
        print(f"\rStatus: {STATE_DISPLAY[self._state]}...")
        
        record_ts = int(time.time() * 1000)
        if self.config.save_audio and recordings_dir is not None:
            self._save_stage_recordings(
                record_ts,
                raw_chunks_for_utterance or [],
                collected_frames,
                recordings_dir,
            )
        
        # Pass frames to speech processor (record_ts so it can save STT-prepared to recordings/stt)
        should_end, mute_until = self._speech.process(collected_frames, record_ts=record_ts if self.config.save_audio else None)
        
        if mute_until > 0:
            self._muted_until = mute_until
        
        # Extend session to prevent greeting during conversation
        self._session_until = time.time() + self._session_duration
        
        self._state = AssistantState.IDLE
        
        return should_end
    
    def _print_status(self, extra: str = "") -> None:
        """Print current status."""
        status = f"Status: {STATE_DISPLAY[self._state]}"
        if extra:
            status += f" ({extra})"
        print(status)
    
    def stop(self) -> None:
        """Stop the assistant."""
        self._running = False
        if self._wake_word:
            self._wake_word.stop()
        self._components.stop()
        # Stop face thread
        if hasattr(self, "_face_thread_stop"):
            self._face_thread_stop.set()
        if getattr(self, "_face_thread", None) is not None:
            self._face_thread.join(timeout=1.0)
    
    @property
    def state(self) -> AssistantState:
        """Current state."""
        return self._state
    
    @property
    def history(self) -> list:
        """Conversation history."""
        return self._speech.history
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._speech.clear_history()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the voice assistant."""
    import os
    print()
    print("🎙️  Voice Assistant v2")
    print("=" * 40)
    print()
    
    # Audio input device: 1 = M70/MB50 USB mic (default). Override via terminal: AUDIO_DEVICE=0 python -m pipeline.orchestrator_v2
    _env_device = os.environ.get("AUDIO_DEVICE")
    audio_device = int(_env_device) if _env_device not in (None, "") else 1
    
    # Debug: set DEBUG=1 to trace face vs wake word when going back to sleep
    debug = os.environ.get("DEBUG", "").strip().lower() in ("1", "true", "yes")
    
    # Stage recordings: SAVE_AUDIO=1 saves raw, VAD, STT per utterance to recordings/{raw,vad,stt}/
    save_audio = os.environ.get("SAVE_AUDIO", "").strip().lower() in ("1", "true", "yes")
    
    # Note: we lower the wake word threshold a bit here so
    # "Alexa" is easier to trigger while debugging.
    config = VoiceAssistantConfig(
        audio_device=audio_device,
        enable_wake_word=True,
        enable_noise_reduction=True,
        enable_speaker_isolation=False,
        # Face detection
        enable_face_detection=True,
        require_face_for_wake_word=True,
        greet_on_face=True,
        track_talking=True,
        # Wake word tuning for smoother UX
        wake_word_threshold=0.25,     # easier to trigger than default 0.5
        wake_word_timeout_sec=30.0,   # 30s to speak after wake word
        wake_word_cooldown_sec=1.0,   # re-arm quickly if you miss it
        # Speech timing
        silence_timeout_ms=400,       # 400ms silence = end of speech (was 500)
        debug=debug,                  # DEBUG=1 in env to trace face/wake word
        save_audio=save_audio,        # SAVE_AUDIO=1 to record raw, VAD, STT per utterance
    )
    
    try:
        assistant = VoiceAssistant(config)
        assistant.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have downloaded the required models:")
        print("  - LLM: models/llm/*.gguf")
        print("  - TTS: models/tts/**/*.onnx")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
