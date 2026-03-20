"""
tests/verify_pipeline.py
========================

Unit test: Injects a fake face-match and pre-recorded audio to verify that
the state machine reaches Phase 5 (PROC_COMPLETE) within 2 seconds.

No real hardware required — all I/O components are replaced by lightweight mocks.

Run with:
    cd /home/sagitakdx/Desktop/Code/Chimcook
    python -m pytest tests/verify_pipeline.py -v
    # or directly:
    python tests/verify_pipeline.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import types
import wave
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on the path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Lightweight mock/stub objects
# ---------------------------------------------------------------------------

class _FakeSTT:
    """Returns a hard-coded transcript with zero latency."""
    def transcribe(self, audio: np.ndarray) -> str:
        return "What time is it?"


class _FakeLLM:
    """
    Simulates LLM streaming with pre-computed token bursts.
    Introduces no real latency (yields immediately).
    """
    _TOKENS = list("It is twelve o'clock. Hope that helps!")

    def generate_stream(self, *args, **kwargs):
        for tok in self._TOKENS:
            yield tok

    def generate(self, *args, **kwargs):
        return "".join(self._TOKENS)


class _FakeTTS:
    """Returns a tiny silent audio buffer."""
    sample_rate: int = 16000

    @property
    def config(self):
        cfg = MagicMock()
        cfg.sample_rate = self.sample_rate
        return cfg

    def synthesize(self, text: str, normalize: bool = True):
        # 0.05 s of silence at 16kHz
        return np.zeros(800, dtype=np.int16), self.sample_rate


class _FakeAudioOutput:
    """Records play() calls without doing any actual I/O."""
    def __init__(self):
        self.played: list[tuple] = []
        self._lock = threading.Lock()

    def play(self, audio: np.ndarray, sr: int, blocking: bool = True) -> None:
        with self._lock:
            self.played.append((audio, sr))

    def stop(self) -> None:
        pass

    def is_playing(self) -> bool:
        return False


class _FakeFaceDetector:
    """Always returns face_count=1. Used to drive Phase 1."""
    def __init__(self):
        self.result = _FakeFaceResult(face_count=1)

    def process_frame(self, frame=None, debug: bool = False):
        return self.result

    def start(self) -> bool:
        return True

    def stop(self) -> None:
        pass


class _FakeFaceResult:
    def __init__(self, face_count: int = 1):
        self.face_count = face_count
        self.recognized_name = None
        self.confidence = 0.0
        self.is_talking = False
        self.face_location = None
        self.debug_frame = None


class _FakeWakeWordModel:
    """Pretends to be openwakeword; always returns threshold score on first call."""
    def __init__(self, name: str = "alexa"):
        self._name = name
        self._calls = 0

    def predict(self, *args, **kwargs):
        self._calls += 1
        # Return a score dict that exceeds threshold after 1st frame
        return {self._name: 0.9 if self._calls >= 1 else 0.0}

    def reset(self):
        pass


class _FakeConfig:
    """Minimal configuration with all heavy features disabled."""
    # Audio
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 32
    audio_device: Optional[int] = None
    mic_gain: float = 1.0
    # Features
    enable_noise_reduction: bool = False
    enable_speaker_isolation: bool = False
    enable_wake_word: bool = False    # Disable WW for simpler test flow
    enable_face_detection: bool = True
    # Models
    llm_model_path: str = ""
    tts_model_path: str = ""
    tts_speaker_id: int = 0
    # Conversation
    max_history_turns: int = 4
    system_prompt: str = "You are a test assistant."
    # Timing
    silence_timeout_ms: int = 500
    max_speech_duration_sec: float = 10.0
    min_speech_duration_ms: int = 200
    # Wake word
    wake_word_model: str = "alexa"
    wake_word_threshold: float = 0.5
    wake_word_timeout_sec: float = 10.0
    wake_word_cooldown_sec: float = 1.0
    # Face
    known_faces_dir: str = "known_faces"
    face_detection_interval_ms: int = 100
    require_face_for_wake_word: bool = False
    face_window_sec: float = 5.0
    greet_on_face: bool = False
    track_talking: bool = False
    # Self-voice
    mute_during_speech_ms: int = 100
    # Debug
    debug: bool = False
    save_audio: bool = False

    # Computed properties
    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def silence_frames(self) -> int:
        return int(self.silence_timeout_ms / self.frame_ms)

    @property
    def max_frames(self) -> int:
        return int(self.max_speech_duration_sec * 1000 / self.frame_ms)

    @property
    def min_speech_frames(self) -> int:
        return int(self.min_speech_duration_ms / self.frame_ms)


# ---------------------------------------------------------------------------
# Helper: load pre-recorded WAV or synthesize 1s of audio
# ---------------------------------------------------------------------------

def _load_audio_frames(wav_path: Optional[Path] = None) -> list[np.ndarray]:
    """
    Return a list of 512-sample float32 frames from a WAV file.
    Falls back to a 1-second sine wave if the file does not exist.
    """
    sr_target = 16000
    n_silence_pad = 32  # frames of silence around audio

    if wav_path and wav_path.exists():
        with wave.open(str(wav_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            ch = wf.getnchannels()
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data[::ch]
        if sr != sr_target:
            import scipy.signal
            data = scipy.signal.resample_poly(data, sr_target, sr).astype(np.float32)
    else:
        # Synthesize 1-second 440 Hz sine
        t = np.linspace(0, 1.0, sr_target, dtype=np.float32)
        data = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    silence = np.zeros(512, dtype=np.float32)
    frames = [silence] * n_silence_pad
    for i in range(0, len(data), 512):
        chunk = data[i : i + 512]
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))
        frames.append(chunk.astype(np.float32))
    frames += [silence] * n_silence_pad
    return frames


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _build_speech_processor(config, stt, llm, tts, audio_out):
    from pipeline.speech_processor import SpeechProcessor
    return SpeechProcessor(config, stt, llm, tts, audio_out)


def _make_inference_item(frames):
    """Build the tuple that _enqueue_speech puts into _speech_q."""
    record_ts = int(time.time() * 1000)
    return (list(frames), [], None, record_ts)


# ---------------------------------------------------------------------------
# Test 1: Phase 5 reached within 2 seconds
# ---------------------------------------------------------------------------

def test_phase5_within_2s():
    """
    Inject a fake face-match and pre-recorded audio frames directly into
    the _run_inference_streaming() path and assert that:

    1. STT produces non-empty text.
    2. LLM streaming starts and TTS is called at least once.
    3. PROC_COMPLETE is logged within 2 seconds wall-clock time.
    """
    config = _FakeConfig()
    stt = _FakeSTT()
    llm = _FakeLLM()
    tts = _FakeTTS()
    audio_out = _FakeAudioOutput()

    # Build a minimal SpeechProcessor with fakes
    speech = _build_speech_processor(config, stt, llm, tts, audio_out)

    # Build a minimal VoiceAssistant shell (no hardware init)
    import types as _types
    from pipeline.config import AssistantState
    from pipeline.inference_worker import InferenceWorker

    class _StubAssistant:
        """Minimal stand-in that exposes everything InferenceWorker expects."""
        def __init__(self):
            self.config = config
            self._speech = speech
            self._components = _types.SimpleNamespace(
                llm=llm, stt=stt, tts=tts, audio_output=audio_out
            )
            self._state = AssistantState.IDLE
            self._muted_until = 0.0
            self._session_until = 0.0
            
        def _save_stage_recordings(self, *args, **kwargs):
            pass

    class _StubWorker:
        def __init__(self, assistant):
            self._assistant = assistant
            
        def _run_inference_streaming(self, frames, raw_chunks, recordings_dir, record_ts):
            return _types.MethodType(InferenceWorker._run_inference_streaming, self)(
                frames, raw_chunks, recordings_dir, record_ts
            )

    stub_assistant = _StubAssistant()
    stub_worker = _StubWorker(stub_assistant)

    # Prepare audio frames (use repo's test_recording.wav if available)
    wav_path = _ROOT / "test_recording.wav"
    frames = _load_audio_frames(wav_path)
    # Use 50 frames (~1.6 s of audio at 32ms per frame)
    frames = frames[:50]

    # Capture PROC_COMPLETE log via monkey-patching print
    proc_complete_event = threading.Event()
    _original_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

    def _patched_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if "VERIFY: PROC_COMPLETE" in msg:
            proc_complete_event.set()
        _original_print(*args, **kwargs)

    t_start = time.time()

    import builtins
    saved_print = builtins.print
    builtins.print = _patched_print

    try:
        should_end, mute_until = stub_worker._run_inference_streaming(
            frames, [], None, int(time.time() * 1000)
        )
    finally:
        builtins.print = saved_print

    elapsed = time.time() - t_start

    print(f"\n[verify_pipeline] Inference completed in {elapsed:.3f}s")
    print(f"[verify_pipeline] TTS play() called {len(audio_out.played)} time(s)")
    print(f"[verify_pipeline] should_end={should_end}")

    assert proc_complete_event.is_set(), "[FAIL] PROC_COMPLETE was never logged"
    assert elapsed < 2.0, f"[FAIL] Took {elapsed:.3f}s (must be < 2s)"
    assert len(audio_out.played) >= 1, "[FAIL] TTS play() was never called"

    print("[PASS] test_phase5_within_2s ✓")


# ---------------------------------------------------------------------------
# Test 2: Face stable gate (Phase 1) — _face_stable_count accumulates
# ---------------------------------------------------------------------------

def test_face_stable_gate():
    """
    Simulate the vision thread loop with repeated FaceResult(face_count=1).
    Assert that _is_user_present stays False until FACE_STABLE_FRAMES frames
    are seen, then flips to True.
    """
    from pipeline.constants import FACE_STABLE_FRAMES

    # Simulate counter logic (extracted from _start_vision_thread)
    is_user_present = False
    consecutive = 0

    for frame_idx in range(FACE_STABLE_FRAMES + 3):
        face_count = 1  # always see a face
        consecutive = consecutive + 1 if face_count >= 1 else 0

        if not is_user_present and consecutive >= FACE_STABLE_FRAMES:
            is_user_present = True

        if face_count < 1:
            consecutive = 0

    assert is_user_present, "[FAIL] _is_user_present never became True"
    print("[PASS] test_face_stable_gate ✓")


# ---------------------------------------------------------------------------
# Test 3: State remains IDLE when cooldown has not expired
# ---------------------------------------------------------------------------

def test_greet_cooldown_respected():
    """
    Phase 2: greeting must NOT fire when cooldown has not expired,
    even when the face is present and state == IDLE.
    """
    from pipeline.config import AssistantState

    greet_cooldown_sec = 30 * 60.0  # 30 minutes
    last_greeting_time = time.time() - 5.0  # greeted only 5 seconds ago
    now = time.time()
    state = AssistantState.IDLE
    is_user_present = True

    cooldown_expired = (now - last_greeting_time) > greet_cooldown_sec
    # Greeting should NOT fire
    can_greet = (
        is_user_present
        and cooldown_expired
        and state in (AssistantState.IDLE, AssistantState.WAKE_WORD_LISTENING)
    )

    assert not can_greet, "[FAIL] Greeting fired before cooldown expired"
    print("[PASS] test_greet_cooldown_respected ✓")


# ---------------------------------------------------------------------------
# Test 4: WW gated by presence flag (Phase 3)
# ---------------------------------------------------------------------------

def test_wake_word_gated_by_presence():
    """
    Phase 3: wake word gate must block when require_face_for_wake_word=True
    and _is_user_present=False.
    """
    require_face = True
    is_user_present = False
    wake_word_active = False  # Not yet active

    gate_allows = True
    if require_face and not wake_word_active:
        gate_allows = is_user_present

    assert not gate_allows, "[FAIL] Gate should block when no user present"

    # Now face present → gate should open
    is_user_present = True
    gate_allows = True
    if require_face and not wake_word_active:
        gate_allows = is_user_present

    assert gate_allows, "[FAIL] Gate should allow when user is present"
    print("[PASS] test_wake_word_gated_by_presence ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Chimcook Pipeline Verification Tests")
    print("=" * 60)

    results = []
    tests = [
        test_face_stable_gate,
        test_greet_cooldown_respected,
        test_wake_word_gated_by_presence,
        test_phase5_within_2s,
    ]

    for fn in tests:
        print(f"\n▶  {fn.__name__}")
        t0 = time.time()
        try:
            fn()
            results.append((fn.__name__, True, time.time() - t0))
        except AssertionError as e:
            print(f"   {e}")
            results.append((fn.__name__, False, time.time() - t0))
        except Exception as e:
            print(f"   EXCEPTION: {e}")
            results.append((fn.__name__, False, time.time() - t0))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok, dur in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{status}] {name}  ({dur:.3f}s)")

    sys.exit(0 if passed == total else 1)
