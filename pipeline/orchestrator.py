"""
Voice Assistant v2 - Main Orchestrator
======================================

Step 9: Coordinate all components into a working assistant.

This is the main entry point that connects:
Audio Input → Noise Reduction → Speaker Isolation → VAD → STT → LLM → TTS → Audio Output

State Machine:
- IDLE: Waiting for speech
- LISTENING: Capturing speech
- PROCESSING: Running STT + LLM
- SPEAKING: Playing TTS output

Features:
- Full pipeline integration
- Wake word support (optional)
- Conversation history
- Graceful shutdown
- Debug/timing information
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List, Dict
from pathlib import Path
import numpy as np
import time
import sys
import signal


class AssistantState(Enum):
    """Voice assistant states."""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    WAKE_WORD_LISTENING = auto()


@dataclass
class VoiceAssistantConfig:
    """Configuration for the voice assistant."""
    # Audio
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20
    audio_device: Optional[int] = None  # None = default, or device ID (e.g., 5 for USB mic)
    
    # Feature flags
    enable_noise_reduction: bool = True
    enable_speaker_isolation: bool = False  # Disable by default, use wake word
    enable_wake_word: bool = True
    
    # Model paths
    llm_model_path: str = ""
    tts_model_path: str = ""
    tts_speaker_id: int = 100  # Cute voice speaker ID
    
    # Conversation
    max_history_turns: int = 6
    system_prompt: str = """You are a cute and helpful voice assistant. 
Keep your responses brief and conversational (1-3 sentences).
Be friendly and cheerful in your tone."""
    
    # Timing
    silence_timeout_ms: int = 500    # End speech after this much silence (shorter = faster response)
    max_speech_duration_sec: float = 15.0  # Maximum utterance length (shorter for noisy env)
    min_speech_duration_ms: int = 300  # Minimum speech duration to process (ignore short bursts)
    
    # Wake word (using OpenWakeWord - more accurate than STT-based detection)
    # Use model name for built-in (alexa, hey_mycroft, hey_jarvis) or path to .onnx file
    wake_word_model: str = "alexa"        # Built-in model: "Alexa"
    wake_word_threshold: float = 0.5      # Detection threshold (lower for Alexa)
    wake_word_timeout_sec: float = 30.0  # Listen this long after wake word (longer for conversation)
    
    # Face detection (requires 1 face to activate)
    enable_face_detection: bool = True
    known_faces_dir: str = "known_faces"  # Drop photos here for recognition
    face_detection_interval_ms: int = 500  # Detection frequency (ms between captures)
    require_face_for_wake_word: bool = True  # Must see face + hear wake word
    face_window_sec: float = 5.0          # Window to say wake word after face detected
    greet_on_face: bool = True            # Say "Hello" when face first detected
    track_talking: bool = True            # Track if person is moving lips
    
    # Self-voice filtering
    mute_during_speech_ms: int = 300  # Extra mute time after TTS finishes
    
    # Debug
    debug: bool = False
    save_audio: bool = False


class VoiceAssistant:
    """
    Main voice assistant orchestrator.
    
    Coordinates all components:
    - Audio input capture
    - Noise reduction (crowded place handling)
    - Speaker isolation (single speaker focus)
    - Voice activity detection
    - Speech-to-text
    - Language model
    - Text-to-speech
    - Audio output
    
    Usage:
        config = VoiceAssistantConfig(
            llm_model_path="models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            tts_model_path="models/tts/en/en_US/libritts/high/en_US-libritts-high.onnx",
        )
        assistant = VoiceAssistant(config)
        assistant.run()  # Blocking main loop
    """
    
    def __init__(self, config: VoiceAssistantConfig):
        self.config = config
        self._state = AssistantState.IDLE
        self._running = False
        self._conversation_history: List[Dict] = []
        self._should_end_conversation = False  # Flag for ending conversation mode
        self._muted_until = 0.0  # Timestamp until which audio should be ignored (self-voice filtering)
        
        print("=" * 60)
        print(" Initializing Voice Assistant v2")
        print("=" * 60)
        
        # Import all components
        from core.audio_input import AudioInput, AudioInputConfig
        from core.noise_reduction import NoiseReducer, NoiseReducerConfig
        from core.speaker_isolation import SpeakerIsolator, SpeakerIsolatorConfig
        from core.face_detector import FaceDetector, FaceDetectorConfig
        
        # Use Hybrid VAD (RNNoise + Silero) for better accuracy with accents/noise
        # Fallback to WebRTC VAD if dependencies not installed
        try:
            from core.vad_hybrid import VADCompatWrapper as VAD, HybridVADConfig as VADConfig
            USE_HYBRID_VAD = True
        except ImportError:
            from core.vad import VAD, VADConfig
            USE_HYBRID_VAD = False
        from core.stt import STT, STTConfigForAccents
        from core.llm import LLM, LLMConfig
        from core.tts import TTS, TTSConfigCute
        from core.audio_output import AudioOutput, AudioOutputConfig
        
        # Import OpenWakeWord for robust wake word detection
        from openwakeword.model import Model as WakeWordModel
        
        # Initialize Audio Input
        print("\n[1/8] Audio Input...")
        device_info = f" (device={config.audio_device})" if config.audio_device is not None else " (default)"
        print(f"      {device_info}")
        self._audio_input = AudioInput(AudioInputConfig(
            sample_rate=config.sample_rate,
            channels=config.channels,
            frame_ms=config.frame_ms,
            device=config.audio_device,
        ))
        
        # Initialize Noise Reducer
        print("[2/8] Noise Reduction...")
        if config.enable_noise_reduction:
            self._noise_reducer = NoiseReducer(NoiseReducerConfig(
                adaptive=True,
            ))
        else:
            self._noise_reducer = None
            print("      (disabled)")
        
        # Initialize Speaker Isolator
        print("[3/8] Speaker Isolation...")
        if config.enable_speaker_isolation:
            self._speaker_isolator = SpeakerIsolator(SpeakerIsolatorConfig(
                volume_threshold=0.10,      # Higher threshold - only detect loud/close speech
                volume_margin=0.25,         # Stricter - reject voices not matching baseline
                min_frames_to_lock=2,       # Quick lock onto primary speaker
                release_silence_frames=30,  # Release lock after ~600ms silence
            ))
        else:
            self._speaker_isolator = None
            print("      (disabled - using wake word instead)")
        
        # Initialize VAD
        print("[4/8] Voice Activity Detection...")
        if USE_HYBRID_VAD:
            # Hybrid VAD - MAXIMUM sensitivity
            self._vad = VAD(VADConfig(
                sample_rate=config.sample_rate,
                silero_threshold=0.02,      # Extremely low - catch everything
                enable_rnnoise=False,       # Disable if no RNNoise installed
                energy_threshold=0.0001,    # Almost disabled
                hangover_frames=30,         # Keep speech active 600ms after silence
                smooth_window=1,            # No smoothing - instant response
            ))
            print("      (Hybrid: Silero VAD - MAX sensitivity)")
        else:
            # Fallback: WebRTC VAD
            self._vad = VAD(VADConfig(
                sample_rate=config.sample_rate,
                frame_ms=config.frame_ms,
                aggressiveness=1,           # Least aggressive = most sensitive
                energy_threshold=0.01,
                hangover_frames=10,
            ))
            print("      (WebRTC VAD)")
        
        # Initialize STT
        print("[5/8] Speech-to-Text (distil-large-v3)...")
        self._stt = STT(STTConfigForAccents())
        
        # Initialize LLM
        print("[6/8] Language Model (Qwen 2.5 3B)...")
        if not config.llm_model_path:
            # Try to find model automatically - prefer Qwen 2.5 for speed
            models_dir = Path(__file__).parent.parent / "models" / "llm"
            # Prefer Qwen 2.5 3B (faster on CPU)
            qwen_files = list(models_dir.glob("*qwen*.gguf")) if models_dir.exists() else []
            if qwen_files:
                config.llm_model_path = str(qwen_files[0])
            else:
                gguf_files = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
                if gguf_files:
                    config.llm_model_path = str(gguf_files[0])
                else:
                    raise FileNotFoundError("No LLM model found. Download one first.")
        
        self._llm = LLM(LLMConfig(
            model_path=config.llm_model_path,
            n_ctx=2048,
            n_threads=4,
            max_tokens=150,
        ))
        
        # Initialize TTS
        print("[7/8] Text-to-Speech (cute voice)...")
        if not config.tts_model_path:
            # Try to find model automatically
            models_dir = Path(__file__).parent.parent / "models" / "tts"
            # Prefer libritts for cute voice
            libritts = list(models_dir.rglob("*libritts*.onnx")) if models_dir.exists() else []
            if libritts:
                config.tts_model_path = str(libritts[0])
            else:
                onnx_files = list(models_dir.rglob("*.onnx")) if models_dir.exists() else []
                if onnx_files:
                    config.tts_model_path = str(onnx_files[0])
                else:
                    raise FileNotFoundError("No TTS model found. Download one first.")
        
        self._tts = TTS(TTSConfigCute(
            model_path=config.tts_model_path,
            speaker_id=config.tts_speaker_id,
        ))
        
        # Initialize Audio Output
        print("[8/8] Audio Output...")
        self._audio_output = AudioOutput(AudioOutputConfig(
            volume=0.9,
        ))
        
        # Initialize Face Detector (optional - for security/greeting)
        print("[+] Face Detection...")
        if config.enable_face_detection:
            self._face_detector = FaceDetector(FaceDetectorConfig(
                known_faces_dir=str(Path(__file__).parent.parent / config.known_faces_dir),
                detection_interval_ms=config.face_detection_interval_ms,
                enable_talking_detection=config.track_talking,
                # Enable preprocessing for better detection
                enhance_brightness=True,
                brightness_alpha=1.2,
                brightness_beta=20,
                enhance_contrast=True,
                clahe_clip_limit=2.0,
                clahe_tile_size=8,
                denoise=True,
                sharpen=True,
            ))
            # Start camera
            if not self._face_detector.start():
                print("      Warning: Camera not available, face detection disabled")
                self._face_detector = None
        else:
            self._face_detector = None
            print("      (disabled)")
        
        # Face state tracking
        self._face_greeted = False  # Have we said hello?
        self._last_face_count = 0
        self._face_window_until = 0.0  # Accept wake word until this time if face was seen
        
        # Initialize OpenWakeWord (runs separately from STT pipeline)
        print("[+] Wake Word Detection (OpenWakeWord)...")
        import warnings
        import os
        
        # Determine if using custom model path or built-in model name
        model_path = config.wake_word_model
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if model_path.endswith('.onnx'):
                # Custom model - resolve path relative to project root
                project_root = Path(__file__).parent.parent
                full_path = project_root / model_path
                if not full_path.exists():
                    raise FileNotFoundError(f"Wake word model not found: {full_path}")
                self._wake_word_model = WakeWordModel(
                    wakeword_model_paths=[str(full_path)],
                    enable_speex_noise_suppression=True,
                )
                model_name = Path(model_path).stem
            else:
                # Built-in model: load only requested model (wakeword_models via kwargs)
                try:
                    self._wake_word_model = WakeWordModel(
                        wakeword_models=[model_path],
                        enable_speex_noise_suppression=True,
                    )
                except TypeError:
                    self._wake_word_model = WakeWordModel(
                        enable_speex_noise_suppression=True,
                    )
                model_name = self._resolve_wake_word_key(model_path)
        
        self._wake_word_name = model_name
        print(f"      Model requested: {config.wake_word_model} -> {model_name}")
        print(f"      Models loaded: {list(self._wake_word_model.models.keys())}")
        print(f"      Threshold: {config.wake_word_threshold}")
        print(f"      Noise suppression: enabled")
        
        # Calculate timing parameters
        self._frame_samples = int(config.sample_rate * config.frame_ms / 1000)
        self._silence_frames = int(config.silence_timeout_ms / config.frame_ms)
        self._max_frames = int(config.max_speech_duration_sec * 1000 / config.frame_ms)
        self._min_speech_frames = int(config.min_speech_duration_ms / config.frame_ms)
        
        print("\n" + "=" * 60)
        print(" Voice Assistant Ready!")
        print("=" * 60)
        
        if config.enable_wake_word:
            # Map model names to wake phrases
            wake_phrase_map = {
                "hey_jarvis": "Hey Jarvis",
                "hey_jarvis_v2": "Hey Jarvis",  # Community model v2
                "jarvis": "Jarvis",
                "jarvis_v1": "Jarvis",  # Community model v1
                "jarvis_v2": "Jarvis",  # Community model v2
                "alexa": "Alexa", 
                "hey_mycroft": "Hey Mycroft",
            }
            phrase = wake_phrase_map.get(self._wake_word_name, self._wake_word_name.replace('_', ' ').title())
            print(f"\nSay: \"{phrase}\" to activate")
        print("Press Ctrl+C to stop.\n")
    
    def _resolve_wake_word_key(self, requested: str) -> str:
        """Resolve actual model key from loaded models (e.g. 'alexa' -> 'alexa_v0.1.0')."""
        keys = list(self._wake_word_model.models.keys())
        if not keys:
            return requested
        if requested in keys:
            return requested
        for k in keys:
            if k.startswith(requested + "_") or k.startswith(requested + "."):
                return k
        return keys[0]
    
    def run(self) -> None:
        """
        Main loop: continuously process audio and respond.
        
        Press Ctrl+C to stop.
        """
        self._running = True
        self._audio_input.start()
        
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\nReceived interrupt signal...")
            self._running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Skip calibration - use fixed sensitive thresholds
        # Calibration can be unreliable in noisy environments
        print("Using fixed VAD thresholds (max sensitivity)...")
        print("Calibration complete!\n")
        
        # Main loop variables
        audio_buffer = []
        in_speech = False
        silence_count = 0
        wake_word_active = not self.config.enable_wake_word  # If no wake word, always active
        wake_word_timeout = 0
        wake_word_buffer = []  # Buffer for accumulating frames for OpenWakeWord
        wake_word_cooldown = 0  # Cooldown to prevent immediate re-triggering
        
        state_str = {
            AssistantState.IDLE: "🔇 Idle",
            AssistantState.LISTENING: "🎤 Listening",
            AssistantState.PROCESSING: "🧠 Processing",
            AssistantState.SPEAKING: "🔊 Speaking",
            AssistantState.WAKE_WORD_LISTENING: "👂 Waiting for wake word",
        }
        
        if self.config.enable_wake_word:
            self._state = AssistantState.WAKE_WORD_LISTENING
            print(f"Status: {state_str[self._state]}")
        else:
            self._state = AssistantState.IDLE
            print(f"Status: {state_str[self._state]} (listening for speech...)")
        
        try:
            for frame in self._audio_input.frames():
                if not self._running:
                    break
                
                # Self-voice filtering: Skip processing if we're still muted (after TTS playback)
                current_time = time.time()
                is_muted = current_time < self._muted_until
                if is_muted:
                    # Clear wake word buffer during mute to avoid stale audio
                    wake_word_buffer = []
                    continue  # Ignore audio while muted (own voice filtering)
                
                # Step 1: Noise reduction
                processed_frame = frame
                if self._noise_reducer:
                    processed_frame = self._noise_reducer.process(frame)
                
                # Step 2: VAD - get speech detection
                is_speech = self._vad.is_speech(processed_frame)
                
                # Step 3: Speaker isolation (if enabled)
                is_primary = True
                if self._speaker_isolator:
                    is_primary = self._speaker_isolator.update(processed_frame, is_speech)
                
                # Step 3.5: Face detection (if enabled)
                face_detected = False
                face_result = None
                
                if self._face_detector:
                    face_result = self._face_detector.process_frame()
                    # One or more faces = face detected (opens face window)
                    face_detected = face_result.face_count >= 1
                    
                    # When face detected, start/extend the wake word window
                    if face_detected:
                        self._face_window_until = time.time() + self.config.face_window_sec
                
                    # Greet on first face detection
                    if face_detected and not self._face_greeted and self.config.greet_on_face:
                        self._face_greeted = True
                        name = face_result.recognized_name or "there"
                        greeting = f"Hello, {name}!"
                        print(f"\n👋 {greeting}")
                        
                        # Say hello
                        self._state = AssistantState.SPEAKING
                        greet_audio, sr = self._tts.synthesize(greeting)
                        self._audio_output.play(greet_audio, sr)
                        self._muted_until = time.time() + (self.config.mute_during_speech_ms / 1000.0)
                        self._state = AssistantState.WAKE_WORD_LISTENING if self.config.enable_wake_word else AssistantState.IDLE
                        print(f"Status: {state_str[self._state]} (say wake word to begin)")
                    
                    # Reset greeting when face disappears
                    elif face_result.face_count == 0 and self._last_face_count > 0:
                        self._face_greeted = False
                        if self.config.debug:
                            print("\n(Face lost, resetting greeting)")
                    
                    # Track talking status for potential use
                    if face_detected and face_result.is_talking and self.config.track_talking:
                        # Person is moving lips - could be used to confirm speech
                        pass
                    
                    self._last_face_count = face_result.face_count
                
                # Check if face window is active (face detected now OR within window from previous detection)
                face_window_active = (face_detected or 
                                     (hasattr(self, '_face_window_until') and time.time() < self._face_window_until))
                
                # Step 4: Wake word handling (using OpenWakeWord)
                if self.config.enable_wake_word and not wake_word_active:
                    # Check cooldown to prevent immediate re-triggering
                    if time.time() < wake_word_cooldown:
                        continue
                    
                    # Accumulate frames for wake word detection (needs 1280 samples = 80ms for stable detection)
                    wake_word_buffer.append(frame)
                    if len(wake_word_buffer) >= 4:  # 4 x 320 = 1280 samples (80ms)
                        # Combine frames and convert to int16
                        combined_audio = np.concatenate(wake_word_buffer)
                        audio_int16 = (combined_audio * 32767).astype(np.int16)
                        wake_word_buffer = []  # Reset buffer
                        
                        # Check audio level (debug)
                        rms = np.sqrt(np.mean(combined_audio ** 2))
                        
                        prediction = self._wake_word_model.predict(audio_int16)
                        
                        # Check if wake word detected (use resolved model name, not path)
                        score = prediction.get(self._wake_word_name, 0)
                        
                        # Debug: show ALL scores periodically
                        if score > 0.1 or rms > 0.01:
                            scores_str = " | ".join([f"{k}: {v:.3f}" for k, v in prediction.items() if v > 0.01])
                            if not scores_str:
                                scores_str = "(all zeros)"
                            window_status = "✓" if face_window_active else "✗"
                            face_info = ""
                            if face_result:
                                face_info = f" | Faces: {face_result.face_count}"
                            print(f"\r[Wake] RMS: {rms:.4f} | {scores_str} | FaceWindow: {window_status}{face_info}   ", end="", flush=True)
                        
                        if score > self.config.wake_word_threshold:
                            # Check face requirement: must have seen face within the window
                            if self.config.require_face_for_wake_word and self._face_detector:
                                if not face_window_active:
                                    if self.config.debug:
                                        print(f"\n(Wake word heard but no face in window - ignoring)")
                                    continue
                            
                            print(f"\n✨ Wake word detected! (confidence: {score:.2f})")
                            wake_word_active = True
                            wake_word_timeout = time.time() + self.config.wake_word_timeout_sec
                            audio_buffer = []
                            
                            # Reset wake word model state
                            self._wake_word_model.reset()
                            
                            # Acknowledge with custom sound
                            self._state = AssistantState.SPEAKING
                            import scipy.io.wavfile as wav
                            ack_path = Path(__file__).parent.parent / "assets" / "wake_ack.wav"
                            if ack_path.exists():
                                ack_sr, ack_audio = wav.read(str(ack_path))
                                # Convert to float32 if needed
                                if ack_audio.dtype == np.int16:
                                    ack_audio = ack_audio.astype(np.float32) / 32768.0
                                self._audio_output.play(ack_audio, ack_sr)
                            else:
                                # Fallback to TTS
                                ack_audio, sr = self._tts.synthesize("Yes?")
                                self._audio_output.play(ack_audio, sr)
                            
                            # Small delay to let audio system settle (prevent ALSA conflicts)
                            time.sleep(0.1)
                            
                            # Self-voice filtering: Mute after acknowledgment
                            self._muted_until = time.time() + (self.config.mute_during_speech_ms / 1000.0)
                            
                            self._state = AssistantState.IDLE
                            print(f"Status: {state_str[self._state]} (listening for your question...)")
                    
                    # Skip to next frame if wake word not active
                    if not wake_word_active:
                        continue
                
                # Check wake word timeout
                if self.config.enable_wake_word and wake_word_active:
                    if time.time() > wake_word_timeout:
                        print("\n⏰ Wake word timeout. Going back to sleep...")
                        wake_word_active = False
                        audio_buffer = []
                        wake_word_buffer = []  # Clear wake word buffer to prevent false triggers
                        in_speech = False
                        silence_count = 0
                        wake_word_cooldown = time.time() + 3.0  # 3 second cooldown before re-listening
                        self._state = AssistantState.WAKE_WORD_LISTENING
                        # Reset wake word model state to clear any partial detections
                        self._wake_word_model.reset()
                        print(f"Status: {state_str[self._state]}")
                        # Flush any pending audio from input
                        try:
                            for _ in range(10):  # Discard ~200ms of audio
                                next(self._audio_input.frames())
                        except:
                            pass
                        continue
                
                # Step 5: Collect speech
                if is_primary and is_speech:
                    # Reset wake word timeout on speech
                    if self.config.enable_wake_word:
                        wake_word_timeout = time.time() + self.config.wake_word_timeout_sec
                    
                    if not in_speech:
                        in_speech = True
                        audio_buffer = []
                        silence_count = 0
                        self._on_speech_start()
                        self._state = AssistantState.LISTENING
                        print(f"\rStatus: {state_str[self._state]}...", end="", flush=True)
                    
                    audio_buffer.append(frame)
                    silence_count = 0
                    
                    # Show we're capturing audio
                    if len(audio_buffer) % 25 == 0:  # Every 0.5 seconds
                        print(".", end="", flush=True)
                    
                    # Check max duration
                    if len(audio_buffer) >= self._max_frames:
                        print(f"\n⚠️ Max speech duration reached")
                        in_speech = False
                        self._process_speech(audio_buffer)
                        audio_buffer = []
                
                elif in_speech:
                    # Still collect during short silences (for pauses in speech)
                    audio_buffer.append(frame)
                    silence_count += 1
                    
                    # Speech ended after silence timeout
                    if silence_count >= self._silence_frames:
                        in_speech = False
                        self._process_speech(audio_buffer)
                        audio_buffer = []
                        silence_count = 0
                        
                        # Check if user said goodbye
                        if self._should_end_conversation:
                            self._should_end_conversation = False
                            wake_word_active = False
                            self._state = AssistantState.WAKE_WORD_LISTENING
                            print(f"\nStatus: {state_str[self._state]}")
                        # Stay active for continued conversation (extend timeout)
                        elif self.config.enable_wake_word:
                            wake_word_timeout = time.time() + self.config.wake_word_timeout_sec
                            self._state = AssistantState.IDLE
                            print(f"Status: {state_str[self._state]} (listening for follow-up, say 'goodbye' to end)")
                        else:
                            self._state = AssistantState.IDLE
        
        finally:
            self._audio_input.stop()
    
    def _process_speech(self, audio_buffer: List[np.ndarray]) -> None:
        """Process collected speech: STT → LLM → TTS → Play."""
        # Check minimum speech duration (ignore short bursts like background noise)
        if len(audio_buffer) < self._min_speech_frames:
            duration_ms = len(audio_buffer) * self.config.frame_ms
            print(f"\r(too short: {duration_ms}ms, need {self.config.min_speech_duration_ms}ms)")
            return
        
        # Combine audio
        audio = np.concatenate(audio_buffer)
        
        # Debug: save audio
        if self.config.save_audio:
            import wave
            timestamp = int(time.time())
            with wave.open(f"debug_audio_{timestamp}.wav", 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.config.sample_rate)
                audio_int16 = (audio * 32767).astype(np.int16)
                f.writeframes(audio_int16.tobytes())
        
        # STT
        self._state = AssistantState.PROCESSING
        print(f"\rStatus: 🧠 Transcribing...            ", end="", flush=True)
        
        start_time = time.time()
        text = self._stt.transcribe(audio)
        stt_time = time.time() - start_time
        
        if not text.strip():
            print("\r(empty transcription)            ")
            return
        
        print(f"\r" + " " * 50 + "\r", end="")  # Clear line
        print(f"👤 You: {text}")
        
        if self.config.debug:
            print(f"   [STT: {stt_time:.2f}s]")
        
        # Check for goodbye/end conversation
        goodbye_phrases = ["goodbye", "bye bye", "bye", "see you", "that's all", "stop", "go to sleep"]
        text_lower = text.lower().strip()
        if any(phrase in text_lower for phrase in goodbye_phrases):
            print("🤖 Assistant: Goodbye! Say 'Hey Jarvis' when you need me!")
            self._state = AssistantState.SPEAKING
            goodbye_audio, sr = self._tts.synthesize("Goodbye! Say Hey Jarvis when you need me!")
            self._audio_output.play(goodbye_audio, sr)
            # Self-voice filtering after goodbye - mute for full audio duration + extra buffer
            audio_duration = len(goodbye_audio) / sr
            self._muted_until = time.time() + audio_duration + 1.0  # Extra 1 second buffer
            self._should_end_conversation = True  # Signal to end conversation mode
            return
        
        self._add_to_history("user", text)
        
        # LLM
        print(f"🤖 Thinking...", end="", flush=True)
        
        start_time = time.time()
        response = self._llm.generate(
            text,
            history=self._conversation_history[:-1],  # Exclude current message
            system_prompt=self.config.system_prompt,
        )
        llm_time = time.time() - start_time
        
        print(f"\r🤖 Assistant: {response}")
        
        if self.config.debug:
            print(f"   [LLM: {llm_time:.2f}s]")
        
        self._add_to_history("assistant", response)
        
        # TTS + Play
        self._state = AssistantState.SPEAKING
        
        start_time = time.time()
        tts_audio, sr = self._tts.synthesize(response)
        tts_time = time.time() - start_time
        
        if self.config.debug:
            print(f"   [TTS: {tts_time:.2f}s]")
        
        self._audio_output.play(tts_audio, sr)
        
        # Self-voice filtering: Mute microphone for a bit after speaking
        # This prevents the assistant from hearing its own voice
        audio_duration = len(tts_audio) / sr
        self._muted_until = time.time() + (self.config.mute_during_speech_ms / 1000.0)
        
        self._state = AssistantState.IDLE
        print()  # New line after response
    
    def _on_speech_start(self) -> None:
        """Called when speech begins."""
        if self.config.debug:
            print("\n[Speech started]")
    
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
    
    def stop(self) -> None:
        """Stop the assistant gracefully."""
        self._running = False
        self._audio_output.stop()
        self._audio_input.stop()
        if self._face_detector:
            self._face_detector.stop()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
    
    @property
    def state(self) -> AssistantState:
        """Current assistant state."""
        return self._state
    
    @property
    def history(self) -> List[Dict]:
        """Current conversation history."""
        return self._conversation_history.copy()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the voice assistant."""
    print()
    print("🎙️  Voice Assistant v2")
    print("=" * 40)
    print()
    
    # Create config
    config = VoiceAssistantConfig(
        # Audio - Use default device (None) or specify device ID
        # Laptop: 4=ALC256 (built-in), 10=pulse, 11=default
        # Mini PC: 5=USB Composite
        audio_device=None,  # Use default system device
        
        enable_wake_word=True,
        wake_word_threshold=0.3,  # Detection threshold
        enable_noise_reduction=False,  # Disable to test raw audio
        enable_speaker_isolation=False,
        # Face detection: require 1 face + wake word
        enable_face_detection=True,
        require_face_for_wake_word=True,
        greet_on_face=True,
        track_talking=True,
        debug=True,  # Enable debug output
    )
    
    # Create and run assistant
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
