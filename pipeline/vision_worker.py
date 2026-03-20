"""
pipeline/vision_worker.py
=========================
VisionWorker — HOG face-detection background thread.

Responsibilities
----------------
Phase 1  [VERIFY: FACE_STABLE]   Track consecutive frames; set _is_user_present
Phase 2  [VERIFY: GREET_SENT]    Greet user when IDLE + cooldown expired

Debugging tips
--------------
• Too many false positives?
      Lower  FaceDetectorConfig.recognition_tolerance  (e.g. 0.45 → 0.40)
      or increase FACE_STABLE_FRAMES in pipeline/constants.py

• Greeting fires too often?
      Raise  GREET_COOLDOWN_SEC  in pipeline/constants.py

• Soft-lock triggers too soon?
      Raise  SOFT_GONE_SEC  in pipeline/constants.py

• Hard reset wipes history too fast?
      Raise  HARD_GONE_TIMEOUT_SEC  in pipeline/constants.py
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Optional

from pipeline.constants import (
    FACE_STABLE_FRAMES,
    GREET_COOLDOWN_SEC,
    HARD_GONE_TIMEOUT_SEC,
    SOFT_GONE_SEC,
)
from pipeline.config import AssistantState

if TYPE_CHECKING:
    from pipeline.orchestrator_v2 import VoiceAssistant


class VisionWorker:
    """
    Runs the HOG face detector on a background daemon thread.

    Shared state (protected by GIL / atomic flag reads):
        assistant._is_user_present      bool
        assistant._last_face_seen       float (epoch timestamp)
        assistant._last_greeting_time   float
        assistant._wake_word_soft_locked bool
        assistant._state                AssistantState
        assistant._muted_until          float

    Call  start()  once.  Call  stop()  to request thread exit.
    """

    def __init__(self, assistant: "VoiceAssistant") -> None:
        self._assistant = assistant
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the detection loop in a daemon thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="VisionWorker"
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the thread to stop and wait up to *timeout* seconds."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── Internal loop ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        a = self._assistant
        detector = a._components.face_detector
        if not detector:
            return

        cfg = a.config
        interval = max(0.1, cfg.face_detection_interval_ms / 1000.0)
        consecutive = 0   # Phase-1 frame counter

        while not self._stop_event.is_set():
            try:
                result = detector.process_frame()
                now = time.time()

                if result.face_count >= 1:
                    consecutive += 1

                    # ── Phase 1: FACE_STABLE ───────────────────────────────
                    if not a._is_user_present and consecutive >= FACE_STABLE_FRAMES:
                        a._is_user_present = True
                        print(
                            f"[VERIFY: FACE_STABLE] ✅ Face confirmed after "
                            f"{consecutive} frames"
                        )

                    was_soft_locked = a._wake_word_soft_locked
                    a._wake_word_soft_locked = False
                    a._last_face_seen = now

                    # Auto-resume after soft-lock
                    if was_soft_locked and a._wake_word and not a._wake_word.is_active:
                        print("\n🔄 Welcome back! Resuming conversation...")
                        a._wake_word.activate()
                        a._wake_word.extend_timeout()
                        a._state = AssistantState.IDLE
                        a._print_status("listening for your question...")

                    # ── Phase 2: GREET_SENT ────────────────────────────────
                    elif (
                        a._is_user_present
                        and cfg.greet_on_face
                        and (now - a._last_greeting_time) > GREET_COOLDOWN_SEC
                        and (a._wake_word is None or not a._wake_word.is_active)
                        and a._state in (
                            AssistantState.IDLE,
                            AssistantState.WAKE_WORD_LISTENING,
                        )
                    ):
                        self._send_greeting(now)

                else:
                    # ── Face lost ──────────────────────────────────────────
                    consecutive = 0
                    self._handle_face_lost(now)

                # Push result into face-event queue (audio thread drains it)
                try:
                    a._face_event_q.put_nowait(result)
                except queue.Full:
                    pass

                self._stop_event.wait(timeout=interval)

            except Exception as exc:
                if self._assistant.config.debug:
                    print(f"[VisionWorker] Error: {exc}")
                self._stop_event.wait(timeout=1.0)

    # ── Helper methods ─────────────────────────────────────────────────────────

    def _send_greeting(self, now: float) -> None:
        """Synthesise and play the Phase-2 greeting."""
        a = self._assistant
        greeting = "Hello there! Just say the wake word when you're ready to talk."
        print(f"[VERIFY: GREET_SENT] 👋 {greeting}")
        a._last_greeting_time = now
        a._state = AssistantState.SPEAKING

        audio, sr = a._speech._tts.synthesize(greeting)
        duration = len(audio) / sr
        a._muted_until = now + duration + a.config.mute_during_speech_ms / 1000.0

        if a._wake_word:
            a._wake_word._state.cooldown = a._muted_until
            a._wake_word._model.reset()

        a._speech._audio_output.play(audio, sr)
        a._state = (
            AssistantState.WAKE_WORD_LISTENING
            if a.config.enable_wake_word
            else AssistantState.IDLE
        )
        a._print_status("say wake word to begin")

    def _handle_face_lost(self, now: float) -> None:
        """Apply soft-gone / hard-gone logic when no face is detected."""
        a = self._assistant
        if not a._is_user_present or a._last_face_seen <= 0:
            return

        elapsed = now - a._last_face_seen

        # Soft gone → lock wake word only during an active session
        if elapsed > SOFT_GONE_SEC and not a._wake_word_soft_locked:
            if a._wake_word and a._wake_word.is_active:
                a._wake_word_soft_locked = True
                if a.config.debug:
                    print("[VisionWorker] Soft gone – locking wake word (active session)")
            elif a.config.debug:
                print("[VisionWorker] Soft gone – no active session, staying in listen mode")

        # Hard gone → full session reset
        if elapsed > HARD_GONE_TIMEOUT_SEC:
            if a.config.debug:
                print("[VisionWorker] Hard gone – resetting session")
            a._is_user_present = False
            a._wake_word_soft_locked = False
            a._last_face_seen = 0.0
            a._last_greeting_time = 0.0  # Reset greeting cooldown so it welcomes them back
            if a._wake_word and a._wake_word.is_active:
                a._wake_word.deactivate(with_cooldown=False)
            a._speech.clear_history()
            a._state = (
                AssistantState.WAKE_WORD_LISTENING
                if a.config.enable_wake_word
                else AssistantState.IDLE
            )
            a._session_until = 0.0
