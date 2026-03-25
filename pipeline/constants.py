"""
pipeline/constants.py
=====================
All tunable constants for the voice-assistant pipeline.

Centralised here so you can tweak timing / thresholds in ONE place
without hunting through the orchestrator.
"""

# ─── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000        # Hz – all components expect 16 kHz
CHUNK_MS: int = 80               # ms per audio-loop chunk (80 ms = 1 280 samples)

# ─── Silero VAD ───────────────────────────────────────────────────────────────
FRAME_MS: int = 32               # ms per Silero frame (512 samples @ 16 kHz)
FRAME_SAMPLES: int = SAMPLE_RATE * FRAME_MS // 1000   # = 512

SILERO_THRESHOLD: float = 0.6   # speech-probability gate  (lower → more sensitive)
SILENCE_TIMEOUT_MS: int = 600    # ms of silence → end of utterance
MIN_SPEECH_MS: int = 200         # ms – ignore utterances shorter than this

# Derived silence / min-speech frame counts (handy for loop comparisons)
SILENCE_FRAMES: int = SILENCE_TIMEOUT_MS // FRAME_MS   # ≈ 31 frames
MIN_SPEECH_FRAMES: int = MIN_SPEECH_MS // FRAME_MS     # ≈ 6 frames

# ─── Vision (Phase 1) ─────────────────────────────────────────────────────────
FACE_STABLE_FRAMES: int = 1      # consecutive HOG-positive frames before FACE_STABLE

# ─── Timing ───────────────────────────────────────────────────────────────────
# (seconds unless noted)
SOFT_GONE_SEC: float = 5.0             # face absent → soft-lock wake word
HARD_GONE_TIMEOUT_SEC: float = 35.0    # face absent → full session reset
GREET_COOLDOWN_SEC: float = 1_800.0    # 30 min between automated greetings
SESSION_DURATION_SEC: float = 30.0     # stay "in session" for this long after any utterance
WAKE_WORD_DECAY_COOLDOWN: float = 3.0  # extra cooldown to let OWW internal state decay

# ─── Sentinel – signals inference thread to exit ──────────────────────────────
STOP_SENTINEL = object()
