# Optionally specify the input device for sounddevice (index or name)
# Example: audio_device = 1  # or 'USB Audio'
audio_device = 5  # Use default input device
"""
Voice Assistant v2 - Configuration Module

Centralized configuration loading from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


def _load_env(path: str = ".env") -> None:
    """Simple .env file loader."""
    env_path = Path(path)
    if not env_path.exists():
        return
    
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key, str(default)).lower().strip()
    return value in ("true", "1", "yes", "on")


def _env_list(key: str, default: List[str] = None) -> List[str]:
    value = os.getenv(key, "")
    if not value:
        return default or []
    return [item.strip() for item in value.split(",")]


@dataclass
class AudioConfig:
    """Audio input/output configuration."""
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    
    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)


@dataclass
class NoiseReductionConfig:
    """Noise reduction configuration."""
    enabled: bool = True
    gate_threshold: float = 0.01
    adaptive: bool = True
    adaptation_rate: float = 0.05


@dataclass
class SpeakerIsolationConfig:
    """Speaker isolation configuration for crowded places."""
    enabled: bool = True
    volume_threshold: float = 0.1
    consistency_window: int = 10
    min_speech_ratio: float = 0.6


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    aggressiveness: int = 3            # 0-3, higher = fewer false positives
    energy_threshold: float = 0.035    # RMS threshold (tune for your environment)
    smooth_window: int = 8             # Frames for majority-vote smoothing
    hangover_frames: int = 5           # Keep speech active after it ends


@dataclass
class SegmenterConfig:
    """Speech segmentation configuration."""
    start_trigger_frames: int = 3
    end_silence_ms: int = 1000
    min_utterance_ms: int = 300
    max_utterance_ms: int = 15000


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""
    model: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    language: str = "en"


@dataclass
class LLMConfig:
    """Language Model configuration."""
    model_path: str = ""
    context_tokens: int = 2048
    max_tokens: int = 150
    temperature: float = 0.7
    threads: int = 4
    gpu_layers: int = 0


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""
    model_path: str = ""
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8
    volume: float = 1.0


@dataclass
class WakeWordConfig:
    """Wake word detection configuration."""
    enabled: bool = False
    words: List[str] = field(default_factory=lambda: ["hey assistant"])
    timeout_sec: int = 10


@dataclass
class ConversationConfig:
    """Conversation management configuration."""
    max_turns: int = 6
    max_chars_per_msg: int = 500
    system_prompt: str = "You are a helpful voice assistant. Keep responses brief."


@dataclass
class DebugConfig:
    """Debug and logging configuration."""
    save_audio: bool = False
    show_vad: bool = False
    show_timing: bool = False
    log_dir: str = "logs/debug"


@dataclass
class Config:
    """Main configuration container."""
    audio: AudioConfig
    noise_reduction: NoiseReductionConfig
    speaker_isolation: SpeakerIsolationConfig
    vad: VADConfig
    segmenter: SegmenterConfig
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    wake_word: WakeWordConfig
    conversation: ConversationConfig
    debug: DebugConfig


def load_config(env_path: str = ".env") -> Config:
    """Load configuration from environment variables."""
    _load_env(env_path)
    
    # Parse input device
    input_dev = _env_str("AUDIO_INPUT_DEVICE", "")
    input_device = int(input_dev) if input_dev.isdigit() else None
    
    output_dev = _env_str("TTS_OUTPUT_DEVICE", "")
    output_device = int(output_dev) if output_dev.isdigit() else None
    
    return Config(
        audio=AudioConfig(
            sample_rate=_env_int("AUDIO_SAMPLE_RATE", 16000),
            channels=_env_int("AUDIO_CHANNELS", 1),
            frame_ms=_env_int("AUDIO_FRAME_MS", 20),
            input_device=input_device,
            output_device=output_device,
        ),
        noise_reduction=NoiseReductionConfig(
            enabled=_env_bool("NOISE_GATE_ENABLED", True),
            gate_threshold=_env_float("NOISE_GATE_THRESHOLD", 0.01),
            adaptive=_env_bool("NOISE_ADAPTIVE", True),
            adaptation_rate=_env_float("NOISE_ADAPTATION_RATE", 0.05),
        ),
        speaker_isolation=SpeakerIsolationConfig(
            enabled=_env_bool("SPEAKER_ISOLATION_ENABLED", True),
            volume_threshold=_env_float("SPEAKER_VOLUME_THRESHOLD", 0.1),
            consistency_window=_env_int("SPEAKER_CONSISTENCY_WINDOW", 10),
            min_speech_ratio=_env_float("SPEAKER_MIN_SPEECH_RATIO", 0.6),
        ),
        vad=VADConfig(
            aggressiveness=_env_int("VAD_AGGRESSIVENESS", 2),
            energy_threshold=_env_float("VAD_ENERGY_THRESHOLD", 0.01),
            smooth_window=_env_int("VAD_SMOOTH_WINDOW", 5),
        ),
        segmenter=SegmenterConfig(
            start_trigger_frames=_env_int("START_TRIGGER_FRAMES", 3),
            end_silence_ms=_env_int("END_SILENCE_MS", 1000),
            min_utterance_ms=_env_int("MIN_UTTERANCE_MS", 300),
            max_utterance_ms=_env_int("MAX_UTTERANCE_MS", 15000),
        ),
        stt=STTConfig(
            model=_env_str("STT_MODEL", "tiny"),
            device=_env_str("STT_DEVICE", "cpu"),
            compute_type=_env_str("STT_COMPUTE_TYPE", "int8"),
            beam_size=_env_int("STT_BEAM_SIZE", 1),
            language=_env_str("STT_LANGUAGE", "en"),
        ),
        llm=LLMConfig(
            model_path=_env_str("LLM_MODEL_PATH", ""),
            context_tokens=_env_int("LLM_CONTEXT_TOKENS", 2048),
            max_tokens=_env_int("LLM_MAX_TOKENS", 150),
            temperature=_env_float("LLM_TEMPERATURE", 0.7),
            threads=_env_int("LLM_THREADS", 4),
            gpu_layers=_env_int("LLM_GPU_LAYERS", 0),
        ),
        tts=TTSConfig(
            model_path=_env_str("TTS_MODEL_PATH", ""),
            length_scale=_env_float("TTS_LENGTH_SCALE", 1.0),
            noise_scale=_env_float("TTS_NOISE_SCALE", 0.667),
            noise_w_scale=_env_float("TTS_NOISE_W_SCALE", 0.8),
            volume=_env_float("TTS_VOLUME", 1.0),
        ),
        wake_word=WakeWordConfig(
            enabled=_env_bool("WAKE_WORD_ENABLED", False),
            words=_env_list("WAKE_WORDS", ["hey assistant"]),
            timeout_sec=_env_int("WAKE_WORD_TIMEOUT", 10),
        ),
        conversation=ConversationConfig(
            max_turns=_env_int("MAX_TURNS", 6),
            max_chars_per_msg=_env_int("MAX_CHARS_PER_MSG", 500),
            system_prompt=_env_str("SYSTEM_PROMPT", "You are a helpful voice assistant."),
        ),
        debug=DebugConfig(
            save_audio=_env_bool("DEBUG_AUDIO", False),
            show_vad=_env_bool("DEBUG_VAD", False),
            show_timing=_env_bool("DEBUG_TIMING", False),
            log_dir=_env_str("DEBUG_LOG_DIR", "logs/debug"),
        ),
    )


# Example usage:
if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded:")
    print(f"  Audio: {config.audio.sample_rate}Hz, {config.audio.frame_ms}ms frames")
    print(f"  STT Model: {config.stt.model}")
    print(f"  LLM Model: {config.llm.model_path}")
    print(f"  Speaker Isolation: {config.speaker_isolation.enabled}")
