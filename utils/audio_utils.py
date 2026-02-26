"""
Voice Assistant v2 - Audio Utilities
====================================

Common audio processing functions used across modules.
"""

import numpy as np
from typing import Tuple


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute RMS (Root Mean Square) energy of audio.
    
    Args:
        audio: Audio samples (any dtype)
        
    Returns:
        RMS value (float)
    """
    if audio.size == 0:
        return 0.0
    
    audio = audio.astype(np.float32)
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_db(audio: np.ndarray, ref: float = 1.0) -> float:
    """
    Compute audio level in decibels.
    
    Args:
        audio: Audio samples
        ref: Reference value (default 1.0 for full scale)
        
    Returns:
        Level in dB (float)
    """
    rms = compute_rms(audio)
    if rms <= 0:
        return -100.0
    return 20 * np.log10(rms / ref)


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """
    Convert float32 [-1, 1] audio to int16 [-32768, 32767].
    
    Args:
        audio: Float32 audio array
        
    Returns:
        Int16 audio array
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


def int16_to_float(audio: np.ndarray) -> np.ndarray:
    """
    Convert int16 [-32768, 32767] audio to float32 [-1, 1].
    
    Args:
        audio: Int16 audio array
        
    Returns:
        Float32 audio array
    """
    return audio.astype(np.float32) / 32768.0


def float_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """
    Convert float32 audio to PCM bytes (little-endian int16).
    
    Args:
        audio: Float32 audio array
        
    Returns:
        PCM bytes
    """
    int16_audio = float_to_int16(audio)
    return int16_audio.tobytes()


def pcm_bytes_to_float(pcm: bytes) -> np.ndarray:
    """
    Convert PCM bytes to float32 audio.
    
    Args:
        pcm: PCM bytes (little-endian int16)
        
    Returns:
        Float32 audio array
    """
    int16_audio = np.frombuffer(pcm, dtype=np.int16)
    return int16_to_float(int16_audio)


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """
    Resample audio to different sample rate.
    
    Uses simple linear interpolation (for better quality, use soxr or scipy).
    
    Args:
        audio: Audio array
        from_sr: Source sample rate
        to_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if from_sr == to_sr:
        return audio
    
    ratio = to_sr / from_sr
    new_length = int(len(audio) * ratio)
    
    # Linear interpolation
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, new_length)
    
    return np.interp(x_new, x_old, audio).astype(np.float32)


def normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio array
        target_db: Target level in dB (default -3.0)
        
    Returns:
        Normalized audio
    """
    current_db = compute_db(audio)
    if current_db <= -100:
        return audio
    
    gain = 10 ** ((target_db - current_db) / 20)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def apply_fade(
    audio: np.ndarray,
    fade_in_samples: int = 0,
    fade_out_samples: int = 0
) -> np.ndarray:
    """
    Apply fade-in and/or fade-out to audio.
    
    Args:
        audio: Audio array
        fade_in_samples: Number of samples for fade-in
        fade_out_samples: Number of samples for fade-out
        
    Returns:
        Audio with fades applied
    """
    audio = audio.copy()
    
    if fade_in_samples > 0:
        fade_in = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in
    
    if fade_out_samples > 0:
        fade_out = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out
    
    return audio


def split_frames(
    audio: np.ndarray, 
    frame_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split audio into fixed-size frames.
    
    Args:
        audio: Audio array
        frame_size: Samples per frame
        
    Returns:
        Tuple of (frames, remainder)
        frames: 2D array of shape (num_frames, frame_size)
        remainder: 1D array of leftover samples
    """
    num_frames = len(audio) // frame_size
    if num_frames == 0:
        return np.zeros((0, frame_size), dtype=audio.dtype), audio
    
    complete = audio[:num_frames * frame_size]
    frames = complete.reshape(num_frames, frame_size)
    remainder = audio[num_frames * frame_size:]
    
    return frames, remainder
