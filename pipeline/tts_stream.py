"""
pipeline/tts_stream.py
======================
Helpers for First-Byte-Out (streaming) TTS and audio utilities.

Debugging tip
-------------
If the assistant clips the first word of its reply, check _split_first_sentence —
the sentence boundary regex may be too greedy for certain punctuation styles.

If the earcon sounds distorted, increase the envelope fade length in _generate_earcon.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np

from pipeline.constants import SAMPLE_RATE

# Path to the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent


# ─── Sentence splitting ───────────────────────────────────────────────────────

def split_first_sentence(buf: str) -> Tuple[str, str]:
    """
    Extract the first complete sentence from *buf*.

    Splits on sentence-ending punctuation followed by whitespace (.  !  ?).

    Returns
    -------
    sentence : str
        The first complete sentence (stripped).
    remainder : str
        Everything that follows the split point.

    Examples
    --------
    >>> split_first_sentence("Hello there! How are you?")
    ('Hello there!', 'How are you?')
    >>> split_first_sentence("No boundary yet")
    ('No boundary yet', '')
    """
    m = re.search(r"(?<=[.!?])\s+", buf)
    if m:
        return buf[: m.start() + 1].strip(), buf[m.end():]
    return buf, ""


def has_sentence_boundary(buf: str) -> bool:
    """Return True when *buf* contains at least one complete sentence."""
    return bool(re.search(r"[.!?]\s", buf))


# ─── Earcon (Phase-4 ping) ────────────────────────────────────────────────────

def generate_earcon() -> np.ndarray:
    """
    Generate a short 80 ms 880 Hz sine-wave ping as a float32 array @ 16 kHz.

    This is the Phase-4 MIC_ACTIVE acknowledgment sound when no WAV asset
    is found at  assets/wake_ack.wav.

    Tune the frequency or duration here if you want a different feel.
    """
    duration_s = 0.08            # 80 ms
    frequency_hz = 880           # A5 note
    amplitude = 0.30             # 30 % of full scale — not too loud

    n_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, dtype=np.float32)
    tone = np.sin(2.0 * np.pi * frequency_hz * t).astype(np.float32) * amplitude

    # Apply a short fade-in / fade-out envelope to avoid clicks
    fade_len = n_samples // 4
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    tone[:fade_len] *= fade_in
    tone[-fade_len:] *= fade_in[::-1]

    return tone


def load_earcon_from_assets() -> np.ndarray:
    """
    Try to load  assets/wake_ack.wav as the Phase-4 earcon.

    Falls back to the synthesised sine tone if the file is missing or
    cannot be read.

    Returns float32 array at SAMPLE_RATE.
    """
    import scipy.io.wavfile as _wav

    ack_path = _PROJECT_ROOT / "assets" / "wake_ack.wav"
    if ack_path.exists():
        try:
            sr, data = _wav.read(str(ack_path))
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32_768.0
            return data.astype(np.float32), sr
        except Exception:
            pass

    return generate_earcon(), SAMPLE_RATE
