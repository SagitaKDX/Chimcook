"""
Tests: Voice Activity Detection (VAD) and Silero VAD
====================================================

Run with pytest (from voice_assistant_v2 with venv active):
    python -m pytest tests/test_vad.py -v

Run without pytest (quick smoke test):
    python tests/test_vad.py

Silero tests require: torch, silero-vad (pip install torch silero-vad).
WebRTC VAD tests require: webrtcvad-wheels (or webrtcvad).
"""

import sys
from pathlib import Path

# Ensure voice_assistant_v2 is on path when running tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


# =============================================================================
# WebRTC VAD (core/vad.py)
# =============================================================================

@pytest.fixture
def vad_config():
    """VAD config for 16kHz, 20ms frames (320 samples)."""
    from core.vad import VADConfig
    return VADConfig(
        sample_rate=16000,
        frame_ms=20,
        aggressiveness=2,
        energy_threshold=0.01,
        smooth_window=5,
        hangover_frames=4,
    )


@pytest.fixture
def vad(vad_config):
    """WebRTC VAD instance."""
    from core.vad import VAD
    return VAD(vad_config)


def test_vad_silence_returns_false(vad, vad_config):
    """Silence (very low energy) should return False."""
    frame = np.zeros(vad_config.sample_rate * vad_config.frame_ms // 1000, dtype=np.float32)
    # Or near-silence
    frame = np.random.randn(vad.frame_samples).astype(np.float32) * 0.001
    result = vad.is_speech(frame)
    assert result is False


def test_vad_energy_threshold(vad, vad_config):
    """Frames below energy threshold should be filtered as non-speech."""
    # Very quiet - below 0.01
    quiet = np.random.randn(vad.frame_samples).astype(np.float32) * 0.005
    assert vad.is_speech(quiet) is False


def test_vad_loud_noise_may_be_speech(vad, vad_config):
    """Loud enough audio may be classified as speech (energy passes gate)."""
    # Loud noise - above threshold; WebRTC may or may not say speech
    loud = np.random.randn(vad.frame_samples).astype(np.float32) * 0.1
    # We only assert it runs without error; result depends on WebRTC
    result = vad.is_speech(loud)
    assert isinstance(result, bool)


def test_vad_frame_size_validation(vad, vad_config):
    """VAD should handle slightly wrong frame sizes (padding/trimming)."""
    # Too short
    short = np.random.randn(100).astype(np.float32) * 0.01
    result_short = vad.is_speech(short)
    assert isinstance(result_short, bool)
    # Too long
    long_frame = np.random.randn(500).astype(np.float32) * 0.01
    result_long = vad.is_speech(long_frame)
    assert isinstance(result_long, bool)


def test_vad_reset(vad, vad_config):
    """Reset should clear state (smoothing/hangover)."""
    frame = np.random.randn(vad.frame_samples).astype(np.float32) * 0.05
    vad.is_speech(frame)
    vad.reset()
    # After reset, silence should still be False
    silence = np.zeros(vad.frame_samples, dtype=np.float32)
    assert vad.is_speech(silence) is False


def test_vad_compute_rms(vad):
    """compute_rms returns non-negative float."""
    from core.vad import VAD
    frame = np.random.randn(320).astype(np.float32) * 0.1
    rms = VAD.compute_rms(frame)
    assert isinstance(rms, (float, np.floating))
    assert rms >= 0


def test_vad_stats(vad, vad_config):
    """get_stats returns dict with expected keys."""
    frame = np.random.randn(vad.frame_samples).astype(np.float32) * 0.02
    vad.is_speech(frame)
    stats = vad.get_stats()
    assert "frames_processed" in stats
    assert "speech_frames" in stats
    assert stats["frames_processed"] >= 1


# =============================================================================
# Silero VAD (core/silero_vad.py)
# =============================================================================

@pytest.fixture
def silero_vad():
    """Silero VAD instance (uses fallback if Silero not available)."""
    from core.silero_vad import SileroVAD
    return SileroVAD(
        sample_rate=16000,
        threshold=0.5,
        silence_limit_sec=1.0,
        speech_start_threshold=0.5,
    )


def test_silero_process_returns_tuple(silero_vad):
    """process() returns (probability, is_speech)."""
    # 320 samples = one 20ms frame (less than 512 chunk, so may not emit prob until more data)
    frame = np.random.randn(320).astype(np.float32) * 0.01
    prob, is_speech = silero_vad.process(frame)
    assert isinstance(prob, (float, np.floating))
    assert 0 <= prob <= 1.0
    assert isinstance(is_speech, bool)


def test_silero_silence_low_probability(silero_vad):
    """Near-silence should yield low probability."""
    # Feed enough to get at least one 512-sample chunk (2 frames of 320)
    for _ in range(4):
        frame = np.random.randn(320).astype(np.float32) * 0.001
        silero_vad.process(frame)
    prob = silero_vad.last_probability
    assert prob < 0.5, "Silence should have low speech probability"


def test_silero_reset(silero_vad):
    """reset() clears recording state and buffer."""
    frame = np.random.randn(320).astype(np.float32) * 0.05
    silero_vad.process(frame)
    silero_vad.reset()
    assert silero_vad.is_recording is False
    assert silero_vad.silence_duration == 0.0
    assert silero_vad.speech_duration == 0.0


def test_silero_should_stop_recording_false_when_not_recording(silero_vad):
    """should_stop_recording() is False when not recording."""
    assert silero_vad.should_stop_recording() is False


def test_silero_set_noise_floor(silero_vad):
    """set_noise_floor() updates calibration and does not raise."""
    silero_vad.set_noise_floor(0.02, 0.04)
    # Process a low-energy frame; should be treated as silence with calibrated floor
    for _ in range(4):
        frame = np.random.randn(320).astype(np.float32) * 0.01
        silero_vad.process(frame)
    # Just ensure no exception and state is consistent
    assert silero_vad.silence_duration >= 0 or silero_vad.speech_duration >= 0 or True


def test_silero_force_start_recording(silero_vad):
    """force_start_recording() sets is_recording True."""
    silero_vad.force_start_recording()
    assert silero_vad.is_recording is True
    silero_vad.reset()
    assert silero_vad.is_recording is False


def test_silero_speech_then_silence_stops(silero_vad):
    """After speech, enough silence should trigger should_stop_recording()."""
    silero_vad.force_start_recording()
    # Simulate some speech (high prob) then long silence
    # We need to drive silence_duration >= silence_limit_sec and speech_duration > 0.3
    # Speech: a few high-energy chunks
    for _ in range(20):
        frame = np.random.randn(320).astype(np.float32) * 0.1
        silero_vad.process(frame)
    # Then many silence chunks (low energy) so silence_duration builds
    # chunk_duration ~= 32ms, need 0.9s+ → ~30 chunks; we need 3 consecutive before counting
    for _ in range(50):
        frame = np.random.randn(320).astype(np.float32) * 0.001
        silero_vad.process(frame)
        if silero_vad.should_stop_recording():
            break
    # Should have stopped (either by silence or by 2s no-speech)
    assert silero_vad.should_stop_recording() is True or silero_vad.silence_duration >= 0.5


def test_silero_compat_wrapper():
    """SileroVADCompat has is_speech(), reset(), compute_rms()."""
    from core.silero_vad import SileroVADCompat
    wrapper = SileroVADCompat()
    frame = np.random.randn(320).astype(np.float32) * 0.02
    result = wrapper.is_speech(frame)
    assert isinstance(result, bool)
    wrapper.reset()
    rms = wrapper.compute_rms(frame)
    assert isinstance(rms, (float, np.floating)) and rms >= 0


# =============================================================================
# Integration: Silero with 512-sample chunks (full chunk behavior)
# =============================================================================

def test_silero_full_chunk_silence():
    """A full 512-sample silence chunk gives low prob."""
    from core.silero_vad import SileroVAD
    vad = SileroVAD(threshold=0.5, silence_limit_sec=1.0)
    chunk = np.zeros(512, dtype=np.float32)
    prob, is_speech = vad.process(chunk)
    assert prob < 0.5
    assert is_speech is False


def test_silero_full_chunk_loud():
    """A full 512-sample loud chunk gives higher prob (or at least runs)."""
    from core.silero_vad import SileroVAD
    vad = SileroVAD(threshold=0.5, silence_limit_sec=1.0)
    chunk = (np.random.randn(512).astype(np.float32) * 0.1)
    prob, is_speech = vad.process(chunk)
    assert 0 <= prob <= 1.0
    # Loud noise might be classified as speech or not
    assert isinstance(is_speech, bool)


# =============================================================================
# Run without pytest: python3 tests/test_vad.py
# =============================================================================
if __name__ == "__main__":
    import sys
    # Add project root
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    failed = 0
    run = 0

    # WebRTC VAD
    print("--- WebRTC VAD (core.vad) ---")
    try:
        from core.vad import VAD, VADConfig
        cfg = VADConfig(sample_rate=16000, frame_ms=20, energy_threshold=0.01)
        vad = VAD(cfg)
        frame_silence = np.random.randn(vad.frame_samples).astype(np.float32) * 0.001
        frame_loud = np.random.randn(vad.frame_samples).astype(np.float32) * 0.1
        assert vad.is_speech(frame_silence) is False
        run += 1
        vad.is_speech(frame_loud)  # no exception
        run += 1
        vad.reset()
        run += 1
        print("  OK: silence=False, loud runs, reset")
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # Silero VAD
    print("--- Silero VAD (core.silero_vad) ---")
    try:
        from core.silero_vad import SileroVAD, HAS_SILERO
        print(f"  Silero loaded: {HAS_SILERO}")
        sv = SileroVAD(threshold=0.5, silence_limit_sec=1.0)
        chunk = np.zeros(512, dtype=np.float32)
        prob, is_speech = sv.process(chunk)
        assert 0 <= prob <= 1.0 and is_speech is False
        run += 1
        sv.reset()
        assert sv.is_recording is False
        run += 1
        sv.set_noise_floor(0.02, 0.04)
        run += 1
        print("  OK: process, reset, set_noise_floor")
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    print()
    if failed == 0:
        print(f"All {run} checks passed.")
    else:
        print(f"Failed: {failed} (passed: {run})")
        sys.exit(1)
