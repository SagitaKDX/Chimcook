"""
Voice Assistant Configuration
=============================

Centralized configuration for all voice assistant components.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class AssistantState(Enum):
    """Voice assistant states."""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    WAKE_WORD_LISTENING = auto()


# State display strings
STATE_DISPLAY = {
    AssistantState.IDLE: "🔇 Idle",
    AssistantState.LISTENING: "🎤 Listening",
    AssistantState.PROCESSING: "🧠 Processing",
    AssistantState.SPEAKING: "🔊 Speaking",
    AssistantState.WAKE_WORD_LISTENING: "👂 Waiting for wake word",
}


@dataclass
class VoiceAssistantConfig:
    """Configuration for the voice assistant."""
    
    # =========================================================================
    # Audio Settings
    # =========================================================================
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20
    audio_device: Optional[int] = 1  # 1 = M70/MB50 USB mic; 0 = USB Composite; None = system default
    mic_gain: float = 1.2  # Microphone boost (1.0 = none; >1.2 dễ rè nếu mic đã to)
    
    # =========================================================================
    # Feature Flags
    # =========================================================================
    enable_noise_reduction: bool = False  # Disabled - causes delay
    enable_speaker_isolation: bool = False  # Disable by default, use wake word
    enable_wake_word: bool = True
    enable_face_detection: bool = True
    
    # =========================================================================
    # Model Paths (auto-detected if empty)
    # =========================================================================
    llm_model_path: str = ""
    tts_model_path: str = ""
    tts_speaker_id: int = 100  # Cute voice speaker ID
    
    # =========================================================================
    # Conversation Settings
    # =========================================================================
    max_history_turns: int = 6
    system_prompt: str = """You are a cute and helpful voice assistant. 
Keep your responses brief and conversational (1-3 sentences).
Be friendly and cheerful in your tone."""
    
    # =========================================================================
    # Timing Settings
    # =========================================================================
    silence_timeout_ms: int = 500    # End speech after this much silence
    max_speech_duration_sec: float = 15.0  # Maximum utterance length
    min_speech_duration_ms: int = 300  # Minimum speech to process
    
    # =========================================================================
    # Wake Word Settings
    # =========================================================================
    wake_word_model: str = "alexa"        # Built-in: alexa, hey_mycroft, hey_jarvis
    wake_word_threshold: float = 0.5      # Detection threshold
    wake_word_timeout_sec: float = 30.0   # Listen this long after wake word
    wake_word_cooldown_sec: float = 5.0   # Cooldown after timeout (prevent immediate re-trigger)
    
    # =========================================================================
    # Face Detection Settings
    # =========================================================================
    known_faces_dir: str = "known_faces"  # Drop photos here for recognition
    face_detection_interval_ms: int = 500  # Detection frequency (ms between captures)
    require_face_for_wake_word: bool = False  # Do NOT block wake word when face is briefly lost
    face_window_sec: float = 10.0         # Longer grace window if you re-enable face gating
    greet_on_face: bool = True            # Say "Hello" when face first detected
    track_talking: bool = True            # Track if person is moving lips
    
    # =========================================================================
    # Self-Voice Filtering
    # =========================================================================
    mute_during_speech_ms: int = 300  # Extra mute time after TTS
    
    # =========================================================================
    # Debug Settings
    # =========================================================================
    debug: bool = False
    save_audio: bool = False
    
    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def frame_samples(self) -> int:
        """Samples per frame."""
        return int(self.sample_rate * self.frame_ms / 1000)
    
    @property
    def silence_frames(self) -> int:
        """Frames of silence before speech ends."""
        return int(self.silence_timeout_ms / self.frame_ms)
    
    @property
    def max_frames(self) -> int:
        """Maximum frames for speech."""
        return int(self.max_speech_duration_sec * 1000 / self.frame_ms)
    
    @property
    def min_speech_frames(self) -> int:
        """Minimum frames to consider as speech."""
        return int(self.min_speech_duration_ms / self.frame_ms)


# Wake word display names
WAKE_PHRASE_MAP = {
    "hey_jarvis": "Hey Jarvis",
    "hey_jarvis_v2": "Hey Jarvis",
    "jarvis": "Jarvis",
    "jarvis_v1": "Jarvis",
    "jarvis_v2": "Jarvis",
    "alexa": "Alexa",
    "hey_mycroft": "Hey Mycroft",
}


def get_wake_phrase(model_name: str) -> str:
    """Get display phrase for wake word model."""
    return WAKE_PHRASE_MAP.get(model_name, model_name.replace('_', ' ').title())
