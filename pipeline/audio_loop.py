"""
pipeline/audio_loop.py
======================
AudioLoop — the 80ms audio capture + VAD + wake-word loop.

Extracted from orchestrator_v2.py for easier debugging.

Responsibilities
----------------
Phase 3  [VERIFY: WW_TRIGGERED]   Wake word gating and activation
Phase 4  [VERIFY: MIC_ACTIVE]     VAD speech collection → speech_q

Debugging tips
--------------
• Wake word fires on silence?
      Check RMS in [Wake] log line. If face_detected=False but triggers,
      set require_face_for_wake_word=True in config.
      Raise wake_word_threshold in config.py (currently 0.80).

• VAD captures too much silence?
      Raise SILERO_THRESHOLD in constants.py (currently 0.6).
      Raise SILENCE_FRAMES (currently ~18 frames = 576ms hangover).

• Speech cuts off too early?
      Lower SILERO_THRESHOLD or raise SILENCE_TIMEOUT_MS in constants.py.

• Recording debug?
      Set SAVE_AUDIO=1 env var. Files go to recordings/raw/ and recordings/vad/.

• Run VAD smoke-test offline:
      python pipeline/audio_loop.py
"""

from __future__ import annotations

import queue
import sys
import time
import wave as _wave
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

# ── Path setup (needed when run as: python3 pipeline/audio_loop.py) ──────────
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch

from pipeline.config import AssistantState
from pipeline.constants import (
    CHUNK_MS,
    MIN_SPEECH_FRAMES,
    SAMPLE_RATE,
    SESSION_DURATION_SEC,
    SILENCE_FRAMES,
    SILERO_THRESHOLD,
)
from pipeline.debug_log import agent_log

if TYPE_CHECKING:
    from pipeline.orchestrator_v2 import VoiceAssistant


class AudioLoop:
    """
    Encapsulates all audio-loop helpers that used to live inline in
    VoiceAssistant.run() and as private methods on VoiceAssistant.

    Call start(stream) from VoiceAssistant.run() after all workers are up.
    All state mutations go through the assistant reference.
    """

    def __init__(self, assistant: "VoiceAssistant") -> None:
        self._a = assistant

    # ── Wake word helpers ─────────────────────────────────────────────────────

    def handle_wake_word(self, frame: np.ndarray, face_detected: bool) -> bool:
        """
        Phase 3: process one 80ms chunk through OWW.

        Returns True  → wake word is active (or just became active).
        Returns False → wake word not triggered; caller should skip VAD.

        [VERIFY: WW_TRIGGERED] is printed on first detection.
        """
        a = self._a
        if not a._wake_word:
            return True
        if a._wake_word.is_active:
            return True

        score = a._wake_word.process_frame(frame, face_detected)
        if score is not None:
            print(f"\n[VERIFY: WW_TRIGGERED] ✨ Wake word detected! (confidence: {score:.2f})")
            a._wake_word.activate()
            a._session_until = time.time() + SESSION_DURATION_SEC
            a._state = AssistantState.SPEAKING
            a._muted_until = a._speech.play_acknowledgment()
            a._state = AssistantState.IDLE
            a._print_status("listening for your question...")
            return True
        return False

    def check_wake_word_timeout(self) -> bool:
        """
        Return True and deactivate OWW if the wake-word listen window expired.
        Resets Silero VAD state so old audio doesn't bleed into next session.
        """
        a = self._a
        if not a._wake_word or not a._wake_word.is_active:
            return False
        if a._wake_word.check_timeout():
            print("\n⏰ Wake word timeout. Going back to sleep...")
            a._wake_word.deactivate(with_cooldown=True)
            a._wake_word._state.cooldown = max(
                a._wake_word._state.cooldown,
                time.time() + 3.0,
            )
            a._wake_word_soft_locked = False
            a._state = AssistantState.WAKE_WORD_LISTENING
            a._print_status()
            a._silero_model.reset_states()
            return True
        return False

    # ── VAD helper ────────────────────────────────────────────────────────────

    def silero_vad_predict(self, frame: np.ndarray) -> float:
        """
        Run Silero VAD on a single 512-sample frame.
        Returns speech probability in [0.0, 1.0].

        Debugging: enable config.debug to see per-frame probability bars.
        """
        a = self._a
        if len(frame) != 512:
            frame = (
                np.pad(frame, (0, max(0, 512 - len(frame))))
                if len(frame) < 512
                else frame[:512]
            )
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        tensor = torch.from_numpy(frame).float()
        with torch.no_grad():
            return float(a._silero_model(tensor.unsqueeze(0), SAMPLE_RATE).item())

    def log_vad_debug(self, frame: np.ndarray, speech_prob: float, is_speech: bool) -> None:
        """Print a live VAD level bar and emit an agent_log event (debug mode only)."""
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        bar_len = min(30, int(rms * 200))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        label = "SPEECH" if is_speech else "silence"
        print(
            f"\r  {label:8}  prob={int(speech_prob * 100):3}%  [{bar}]",
            end="",
            flush=True,
        )
        agent_log(
            "VAD_LEVEL",
            "pipeline/audio_loop.py:VAD_LOOP",
            "vad_level",
            {
                "rms": rms,
                "speech_prob": float(speech_prob),
                "is_speech": bool(is_speech),
            },
        )

    # ── Speech queue helpers ──────────────────────────────────────────────────

    def enqueue_speech(
        self,
        frames: List[np.ndarray],
        raw_chunks: List[np.ndarray],
        recordings_dir: Optional[Path],
    ) -> None:
        """
        Put a completed utterance onto the inference queue.

        Raises queue.Full if the inference worker is saturated — caller should
        print a warning and drop the utterance rather than blocking.
        """
        record_ts = int(time.time() * 1000)
        item = (list(frames), list(raw_chunks), recordings_dir, record_ts)
        self._a._speech_q.put_nowait(item)

    def drain_face_events(self) -> None:
        """Non-blocking drain of the face-event queue (called each audio loop tick)."""
        try:
            while True:
                self._a._face_event_q.get_nowait()
        except queue.Empty:
            pass

    def reset_vad(self) -> None:
        """Reset Silero VAD internal state between utterances."""
        self._a._silero_model.reset_states()


    # ── Recording helpers ─────────────────────────────────────────────────────

    def save_stage_recordings(
        self,
        record_ts: int,
        raw_chunks: List[np.ndarray],
        collected_frames: List[np.ndarray],
        recordings_dir: Path,
    ) -> None:
        """
        Write raw mic and VAD-gated audio to disk for offline debugging.

        Files:
            recordings/raw/<timestamp>.wav   — raw mic before VAD
            recordings/vad/<timestamp>.wav   — VAD-gated speech only
        """

        def _write(path: Path, audio: np.ndarray) -> None:
            audio = np.asarray(audio, dtype=np.float32).flatten()
            if not audio.size:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with _wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(
                    (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                )

        if raw_chunks:
            _write(
                recordings_dir / "raw" / f"{record_ts}.wav",
                np.concatenate([c.flatten().astype(np.float32) for c in raw_chunks]),
            )
        if collected_frames:
            _write(
                recordings_dir / "vad" / f"{record_ts}.wav",
                np.concatenate([f.flatten().astype(np.float32) for f in collected_frames]),
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# Run: python pipeline/audio_loop.py
# Prints live Silero VAD probabilities for 5 seconds using the default mic.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import sounddevice as sd
    import torch

    print("Loading Silero VAD...")
    silero_model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
    )
    print("Listening for 5 seconds. Speak to test VAD...\n")

    buffer = np.zeros(0, dtype=np.float32)
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)

    def _predict(frame: np.ndarray) -> float:
        t = torch.from_numpy(frame.astype(np.float32)).float()
        with torch.no_grad():
            return float(silero_model(t.unsqueeze(0), SAMPLE_RATE).item())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=chunk_samples) as stream:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            data, _ = stream.read(chunk_samples)
            buffer = np.concatenate([buffer, data.flatten()])
            while len(buffer) >= 512:
                frame = buffer[:512]
                buffer = buffer[512:]
                prob = _predict(frame)
                is_speech = prob >= SILERO_THRESHOLD
                bar = "█" * int(prob * 30)
                label = "SPEECH " if is_speech else "silence"
                print(f"\r{label}  {prob:.2f}  [{bar:<30}]", end="", flush=True)

    print("\nDone.")
