"""
pipeline/inference_worker.py
============================
InferenceWorker — consumes speech frames and runs STT → LLM → TTS.

Responsibilities
----------------
Phase 5  [VERIFY: PROC_COMPLETE]  Stream LLM text and TTS audio

Debugging tips
--------------
• Assistant sounds cut off?
      Check `tts_stream.split_first_sentence` regex, it might be splitting
      too early on abbreviations.
• Transcriptions are slow?
      Ensure Whisper/STT is running on GPU if available, or use a smaller model.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from pipeline.constants import SESSION_DURATION_SEC, STOP_SENTINEL
from pipeline.config import AssistantState, STATE_DISPLAY
from pipeline.tts_stream import split_first_sentence

if TYPE_CHECKING:
    from pipeline.orchestrator_v2 import VoiceAssistant


class InferenceWorker:
    """
    Consumes utterances from the audio thread, transcribes them, and
    streams the LLM reply out via TTS.
    """

    def __init__(self, assistant: "VoiceAssistant") -> None:
        self._assistant = assistant
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the inference consumer daemon thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="InferenceWorker"
        )
        self._thread.start()

    def stop(self, timeout: float = 4.0) -> None:
        """Signal exit and wait."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        a = self._assistant

        while not self._stop_event.is_set():
            try:
                item = a._speech_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is STOP_SENTINEL:
                break

            frames, raw_chunks, recordings_dir, record_ts = item

            try:
                should_end, mute_until = self._run_inference_streaming(
                    frames, raw_chunks, recordings_dir, record_ts
                )
                if mute_until > 0:
                    a._muted_until = mute_until

                # Extend session
                a._session_until = time.time() + SESSION_DURATION_SEC

                if should_end:
                    a._speech.clear_history()
                    a._wake_word_soft_locked = False
                    if a._wake_word:
                        a._wake_word.deactivate(with_cooldown=True)
                        a._wake_word._state.cooldown = time.time() + 8.0
                        a._wake_word._model.reset()
                    a._state = AssistantState.WAKE_WORD_LISTENING
                    a._print_status()
                elif a._wake_word:
                    a._wake_word.extend_timeout()
                    a._state = AssistantState.IDLE
                    a._print_status("listening for follow-up, say 'goodbye' to end")
                else:
                    a._state = AssistantState.IDLE

            except Exception as e:
                print(f"\n[InferenceWorker] Error: {e}")
                a._state = AssistantState.IDLE

            finally:
                a._speech_q.task_done()

    def _run_inference_streaming(
        self,
        collected_frames: List[np.ndarray],
        raw_chunks: List[np.ndarray],
        recordings_dir: Optional[Path],
        record_ts: int,
    ) -> Tuple[bool, float]:
        """
        STT → LLM-stream → TTS sentence-by-sentence (First-Byte-Out).
        Phase 5 verification log emitted when the last chunk finishes.
        """
        a = self._assistant
        a._state = AssistantState.PROCESSING
        print(f"\rStatus: {STATE_DISPLAY[a._state]}...")

        if a.config.save_audio and recordings_dir is not None:
            a._save_stage_recordings(record_ts, raw_chunks, collected_frames, recordings_dir)

        # ── STT ──────────────────────────────────────────────────────────────
        audio = np.concatenate([f.astype(np.float32).flatten() for f in collected_frames])
        audio = a._speech._prepare_audio_for_stt(audio)

        if a.config.save_audio:
            a._speech._save_debug_audio(audio, record_ts=record_ts)

        # 1. Start the visual/audio thinking cues immediately!
        a._speech.play_thinking_chime()

        # 2. Start a Watchdog Thread to warn user if processing takes > 30s
        import threading
        first_sentence_played_event = threading.Event()
        watchdog_lock = threading.Lock()
        
        def processing_watchdog():
            if not first_sentence_played_event.wait(30.0):
                with watchdog_lock:
                    # If the event was NOT set within 30 seconds...
                    if not first_sentence_played_event.is_set():
                        print("\n[Watchdog] Processing exceeded 30s, notifying user...")
                        a._speech._audio_output.stop()
                        a._speech.say("I am currently working, wait a moment.")
                        # Restart just the background processing loop smoothly
                        a._speech.play_thinking_chime(skip_speech=True)
                    
        wd_thread = threading.Thread(target=processing_watchdog, daemon=True)
        wd_thread.start()

        print("\rStatus: 🧠 Transcribing...", end="", flush=True)
        t0 = time.time()
        text, info = a._speech._stt.transcribe_with_info(audio)
        stt_ms = int((time.time() - t0) * 1000)
        
        detected_language = info.get("language", "en")

        if not text.strip():
            print("\r(empty transcription)")
            first_sentence_played_event.set() # Stop watchdog
            a._speech._audio_output.stop()    # Stop background music
            return False, 0.0

        print(f"\r" + " " * 60 + "\r", end="")

        if a.config.debug:
            print(f"   [STT: {stt_ms}ms]")

        print(f"👤 You: {text}")

        # Query Expansion: Automatically map generic university mentions to Greenwich for RAG accuracy
        import re
        text = re.sub(r'\b(?:my )?university\b', 'Greenwich Vietnam', text, flags=re.IGNORECASE)

        # Goodbye?
        from pipeline.speech_processor import GOODBYE_PHRASES
        if any(p in text.lower() for p in GOODBYE_PHRASES):
            first_sentence_played_event.set() # Stop watchdog
            return a._speech._handle_goodbye()

        a._speech._add_to_history("user", text)

        # ── LLM stream + TTS First-Byte-Out ─────────────────────────────────
        print("🤖 Thinking...", end="", flush=True)
        t1 = time.time()

        sentence_buf = ""
        full_response: List[str] = []
        first_sentence_played = False
        mute_until = 0.0
        audio_duration_total = 0.0

        # Start RAG in the background immediately to overlap with prompt prep overhead
        rag_future = None
        rag_executor = None
        if a._components.rag is not None:
            import concurrent.futures
            # Keep pool alive just for this request
            rag_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            rag_future = rag_executor.submit(a._components.rag.get_context, text, 3)

        history_for_llm = a._speech._conversation_history[:-1]

        import datetime
        tz = datetime.timezone(datetime.timedelta(hours=7))
        current_time = datetime.datetime.now(tz).strftime("%A, %Y-%m-%d %I:%M %p")
        
        # Dynamic prompt injection
        dynamic_prompt = f"{a.config.system_prompt}\nThe current date and time is {current_time}."

        # Build final prompt with RAG context
        if rag_future is not None and rag_executor is not None:
            context = rag_future.result()
            rag_executor.shutdown(wait=False)
            if context:
                dynamic_prompt += f"\nFacts:\n{context}\nRules: 1. Use facts only. 2. English ONLY. 3. REPEAT PHONETICS EXACTLY (output 'G P A', not 'GPA'; '20 percent', not '20%')."

        try:
            for token in a._components.llm.generate_stream(
                text,
                history=history_for_llm,
                system_prompt=dynamic_prompt,
            ):
                sentence_buf += token
                full_response.append(token)

                if re.search(r"[.!?]\s", sentence_buf):
                    sentence, sentence_buf = split_first_sentence(sentence_buf)
                    if sentence.strip():
                        with watchdog_lock:
                            if not first_sentence_played:
                                first_sentence_played = True
                                first_sentence_played_event.set() # Stop 30s watchdog!
                                a._speech._audio_output.stop() # Stop thinking chime/watchdog Audio
                                
                                llm_first_ms = int((time.time() - t1) * 1000)
                                print(f"\r" + " " * 40 + "\r", end="")
                                if a.config.debug:
                                    print(f"   [LLM first sentence: {llm_first_ms}ms]")
                                print(f"🤖 Assistant: {sentence}", end=" ", flush=True)
                            else:
                                print(sentence, end=" ", flush=True)

                            tts_audio, sr = a._speech._tts.synthesize(sentence)
                            seg_dur = len(tts_audio) / sr
                            audio_duration_total += seg_dur
                            # Keep mic muted while this chunk plays
                            a._muted_until = time.time() + seg_dur + (a.config.mute_during_speech_ms / 1000.0)
                            a._speech._audio_output.play(tts_audio, sr)

        except Exception as e:
            print(f"\n[LLM stream error: {e}]")

        if sentence_buf.strip():
            with watchdog_lock:
                if not first_sentence_played:
                    first_sentence_played = True
                    first_sentence_played_event.set()
                    a._speech._audio_output.stop()
                    llm_first_ms = int((time.time() - t1) * 1000)
                    print(f"\r" + " " * 40 + "\r", end="")
                    if a.config.debug:
                        print(f"   [LLM first sentence: {llm_first_ms}ms]")
                    print(f"🤖 Assistant: {sentence_buf}", end=" ", flush=True)
                else:
                    print(sentence_buf, end=" ", flush=True)
                
                try:
                    tts_audio, sr = a._speech._tts.synthesize(sentence_buf)
                    seg_dur = len(tts_audio) / sr
                    audio_duration_total += seg_dur
                    # Keep mic muted while this chunk plays
                    a._muted_until = time.time() + seg_dur + (a.config.mute_during_speech_ms / 1000.0)
                    a._speech._audio_output.play(tts_audio, sr)
                except Exception:
                    pass

        print()

        # Stop watchdog in case it finished before speaking the first sentence
        first_sentence_played_event.set()

        full_text = "".join(full_response).strip()
        if full_text:
            a._speech._add_to_history("assistant", full_text)

        # Final safety mute
        mute_until = time.time() + (a.config.mute_during_speech_ms / 1000.0)

        llm_total_ms = int((time.time() - t1) * 1000)
        print(f"[VERIFY: PROC_COMPLETE] (LLM+TTS: {llm_total_ms}ms, audio: {audio_duration_total:.2f}s)")

        return False, mute_until
