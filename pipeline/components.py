"""
Component Manager
=================

Handles initialization of all voice assistant components.
Separates component setup from main orchestration logic.
"""

from pathlib import Path
from typing import Optional, Any
import warnings
import sys

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.config import VoiceAssistantConfig


class ComponentManager:
    """
    Manages all voice assistant components.
    
    Components:
    - audio_input: Microphone capture
    - audio_output: Speaker playback
    - noise_reducer: Background noise removal
    - speaker_isolator: Focus on single speaker
    - vad: Voice activity detection
    - stt: Speech-to-text
    - llm: Language model
    - tts: Text-to-speech
    - face_detector: Face detection/recognition
    - wake_word: Wake word detection
    """
    
    def __init__(self, config: VoiceAssistantConfig):
        self.config = config
        self._components_initialized = False
        
        # Component references (initialized lazily)
        self.audio_input = None
        self.audio_output = None
        self.noise_reducer = None
        self.speaker_isolator = None
        self.vad = None
        self.stt = None
        self.llm = None
        self.tts = None
        self.face_detector = None
        self.wake_word_model = None
        self.wake_word_name = ""
        
        # Track which VAD is used
        self.use_hybrid_vad = False
    
    def initialize_all(self) -> None:
        """Initialize all components."""
        if self._components_initialized:
            return
        
        print("=" * 60)
        print(" Initializing Voice Assistant v2")
        print("=" * 60)
        
        self._init_audio_input()
        self._init_noise_reducer()
        self._init_speaker_isolator()
        self._init_vad()
        self._init_stt()
        self._init_llm()
        self._init_tts()
        self._init_audio_output()
        self._init_face_detector()
        self._init_wake_word()
        
        self._components_initialized = True
        
        print("\n" + "=" * 60)
        print(" Voice Assistant Ready!")
        print("=" * 60)
    
    def _init_audio_input(self) -> None:
        """Initialize audio input."""
        print("\n[1/8] Audio Input...")
        from core.audio_input import AudioInput, AudioInputConfig
        
        self.audio_input = AudioInput(AudioInputConfig(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            frame_ms=self.config.frame_ms,
            device=self.config.audio_device,
            mic_gain=getattr(self.config, 'mic_gain', 1.2),
        ))
    
    def _init_noise_reducer(self) -> None:
        """Initialize noise reducer."""
        print("[2/8] Noise Reduction...")
        
        if self.config.enable_noise_reduction:
            from core.noise_reduction import NoiseReducer, NoiseReducerConfig
            self.noise_reducer = NoiseReducer(NoiseReducerConfig(adaptive=True))
        else:
            print("      (disabled)")
    
    def _init_speaker_isolator(self) -> None:
        """Initialize speaker isolator."""
        print("[3/8] Speaker Isolation...")
        
        if self.config.enable_speaker_isolation:
            from core.speaker_isolation import SpeakerIsolator, SpeakerIsolatorConfig
            self.speaker_isolator = SpeakerIsolator(SpeakerIsolatorConfig(
                volume_threshold=0.10,
                volume_margin=0.25,
                min_frames_to_lock=2,
                release_silence_frames=30,
            ))
        else:
            print("      (disabled - using wake word instead)")
    
    def _init_vad(self) -> None:
        """Initialize voice activity detection."""
        print("[4/8] Voice Activity Detection...")
        
        # Try Hybrid VAD first (better accuracy)
        try:
            from core.vad_hybrid import VADCompatWrapper as VAD, HybridVADConfig as VADConfig
            self.use_hybrid_vad = True
            
            self.vad = VAD(VADConfig(
                sample_rate=self.config.sample_rate,
                silero_threshold=0.03,      # Very low - let energy do the filtering
                enable_rnnoise=False,       # Disable if not installed
                energy_threshold=0.02,      # Reasonable energy floor
                hangover_frames=12,         # ~240ms hangover
                smooth_window=2,            # Minimal smoothing
            ))
            print("      (Hybrid: Silero VAD - energy-assisted)")
            
        except ImportError:
            # Fallback to WebRTC VAD
            from core.vad import VAD, VADConfig
            self.use_hybrid_vad = False
            
            self.vad = VAD(VADConfig(
                sample_rate=self.config.sample_rate,
                frame_ms=self.config.frame_ms,
                aggressiveness=1,
                energy_threshold=0.01,
                hangover_frames=10,
            ))
            print("      (WebRTC VAD)")
    
    def _init_stt(self) -> None:
        """Initialize speech-to-text."""
        print("[5/8] Speech-to-Text (distil-large-v3)...")
        from core.stt import STT, STTConfigForAccents
        self.stt = STT(STTConfigForAccents())
    
    def _init_llm(self) -> None:
        """Initialize language model."""
        print("[6/8] Language Model (Qwen 2.5 3B)...")
        from core.llm import LLM, LLMConfig
        
        model_path = self.config.llm_model_path
        if not model_path:
            model_path = self._find_llm_model()
        
        self.llm = LLM(LLMConfig(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            max_tokens=150,
        ))
    
    def _find_llm_model(self) -> str:
        """Auto-detect LLM model path."""
        models_dir = Path(__file__).parent.parent / "models" / "llm"
        
        # Prefer Qwen 2.5 3B (faster on CPU)
        qwen_files = list(models_dir.glob("*qwen*.gguf")) if models_dir.exists() else []
        if qwen_files:
            return str(qwen_files[0])
        
        # Fallback to any GGUF
        gguf_files = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
        if gguf_files:
            return str(gguf_files[0])
        
        raise FileNotFoundError("No LLM model found. Download one first.")
    
    def _init_tts(self) -> None:
        """Initialize text-to-speech."""
        print("[7/8] Text-to-Speech (cute voice)...")
        from core.tts import TTS, TTSConfigCute
        
        model_path = self.config.tts_model_path
        if not model_path:
            model_path = self._find_tts_model()
        
        self.tts = TTS(TTSConfigCute(
            model_path=model_path,
            speaker_id=self.config.tts_speaker_id,
        ))
    
    def _find_tts_model(self) -> str:
        """Auto-detect TTS model path."""
        models_dir = Path(__file__).parent.parent / "models" / "tts"
        
        # Prefer libritts for cute voice
        libritts = list(models_dir.rglob("*libritts*.onnx")) if models_dir.exists() else []
        if libritts:
            return str(libritts[0])
        
        # Fallback to any ONNX
        onnx_files = list(models_dir.rglob("*.onnx")) if models_dir.exists() else []
        if onnx_files:
            return str(onnx_files[0])
        
        raise FileNotFoundError("No TTS model found. Download one first.")
    
    def _init_audio_output(self) -> None:
        """Initialize audio output."""
        print("[8/8] Audio Output...")
        from core.audio_output import AudioOutput, AudioOutputConfig
        
        self.audio_output = AudioOutput(AudioOutputConfig(volume=0.9))
    
    def _init_face_detector(self) -> None:
        """Initialize face detector."""
        print("[+] Face Detection...")
        
        if not self.config.enable_face_detection:
            print("      (disabled)")
            return
        
        try:
            from core.face_detector import FaceDetector, FaceDetectorConfig
            
            known_faces_path = Path(__file__).parent.parent / self.config.known_faces_dir
            
            self.face_detector = FaceDetector(FaceDetectorConfig(
                known_faces_dir=str(known_faces_path),
                detection_interval_ms=self.config.face_detection_interval_ms,
                enable_talking_detection=self.config.track_talking,
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
            
            if not self.face_detector.start():
                print("      Warning: Camera not available, face detection disabled")
                self.face_detector = None
                
        except ImportError as e:
            print(f"      Warning: Face detection not available ({e})")
            self.face_detector = None
    
    def _init_wake_word(self) -> None:
        """Initialize wake word detection."""
        print("[+] Wake Word Detection (OpenWakeWord)...")
        
        if not self.config.enable_wake_word:
            print("      (disabled)")
            return
        
        from openwakeword.model import Model as WakeWordModel
        
        model_path = self.config.wake_word_model
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Handle custom model path vs built-in name
            if model_path.endswith('.onnx'):
                project_root = Path(__file__).parent.parent
                full_path = project_root / model_path
                if not full_path.exists():
                    raise FileNotFoundError(f"Wake word model not found: {full_path}")
                self.wake_word_model = WakeWordModel(
                    wakeword_model_paths=[str(full_path)],
                    enable_speex_noise_suppression=True,
                )
                self.wake_word_name = Path(model_path).stem
            else:
                # Built-in model: load only the requested model (wakeword_models via kwargs)
                try:
                    self.wake_word_model = WakeWordModel(
                        wakeword_models=[model_path],
                        enable_speex_noise_suppression=True,
                    )
                except TypeError:
                    # Older API: no wakeword_models, load all then match by name
                    self.wake_word_model = WakeWordModel(
                        enable_speex_noise_suppression=True,
                    )
                self.wake_word_name = self._resolve_wake_word_key(model_path)
        
        print(f"      Available models: {list(self.wake_word_model.models.keys())}")
        print(f"      Using: {self.wake_word_name}")
        print(f"      Threshold: {self.config.wake_word_threshold}")
        print(f"      Noise suppression: enabled")
    
    def _resolve_wake_word_key(self, requested: str) -> str:
        """Resolve actual model key from loaded models (e.g. 'alexa' -> 'alexa_v0.1.0')."""
        keys = list(self.wake_word_model.models.keys())
        if not keys:
            return requested
        if requested in keys:
            return requested
        # Match by prefix (e.g. alexa -> alexa_v0.1.0)
        for k in keys:
            if k == requested or k.startswith(requested + "_") or k.startswith(requested + "."):
                return k
        return keys[0]
    
    def stop(self) -> None:
        """Stop all components."""
        if self.audio_input:
            self.audio_input.stop()
        if self.audio_output:
            self.audio_output.stop()
        if self.face_detector:
            self.face_detector.stop()
