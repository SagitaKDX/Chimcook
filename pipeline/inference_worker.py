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

import datetime
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
from pipeline.rag_search import RAGSearch
from pipeline.speech_processor import GOODBYE_PHRASES

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
                        # Short cooldown — just long enough for goodbye TTS to finish
                        a._wake_word._state.cooldown = time.time() + 3.0
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
            a._audio_loop.save_stage_recordings(record_ts, raw_chunks, collected_frames, recordings_dir)

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
                    # Double-check: LLM may have finished just before the 30s mark
                    if not first_sentence_played_event.is_set():
                        print("\n[Watchdog] Processing exceeded 30s, notifying user...")
                        a._speech._audio_output.stop()   # inside the guarded block
                        a._speech.say("I am currently working, wait a moment.")
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
            print(f"   [STT: {stt_ms}ms, lang={detected_language}]")
        else:
            print(f"   [STT: {stt_ms}ms]")

        # Language guard: skip non-English transcriptions.
        # small.en always returns 'en'; multilingual models may detect Vietnamese etc.
        lang_confidence = info.get("language_probability", 1.0)
        if detected_language not in ("en", "english") and lang_confidence > 0.6:
            print(f"[STT] ⚠️  Non-English detected: '{detected_language}' ({lang_confidence:.0%}) — skipping")
            first_sentence_played_event.set()
            a._speech._audio_output.stop()
            return False, 0.0

        print(f"👤 You: {text}")

        # Query Expansion: map generic university mentions to Greenwich for RAG accuracy
        text = re.sub(r'\b(?:my )?university\b', 'Greenwich Vietnam', text, flags=re.IGNORECASE)

        # Goodbye?
        if any(p in text.lower() for p in GOODBYE_PHRASES):
            first_sentence_played_event.set()   # Stop watchdog
            a._speech._audio_output.stop()      # Stop thinking chime
            return a._speech._handle_goodbye()

        # ── Routing ────────────────────────────────────────────────────────────
        # Design: LLM is a *reformatter*, not a *knowledge source*.
        # RAG always retrieves the facts; the LLM only rephrases them into
        # smooth spoken English. This prevents hallucination while keeping
        # the output natural.
        #
        # Thresholds:
        #   score < 0.30  → instant direct (no LLM wait, raw chunk → TTS)
        #   0.30-0.75     → LLM reformats RAG chunks into spoken prose
        #   >= 0.75       → no match → LLM says "I don't have that info"
        _rag = RAGSearch(a)
        a._speech._add_to_history("user", text)
        direct_answer, rag_score = _rag.direct_answer(text)

        # ── Instant path: ultra-high confidence, skip LLM latency ─────────────
        if direct_answer and rag_score < 0.30:
            first_sentence_played_event.set()
            a._speech._audio_output.stop()
            print(f"[RAG] ⚡ Instant — score={rag_score:.3f}")
            print(f"\n🤖 Assistant: {direct_answer}")
            a._speech._add_to_history("assistant", direct_answer)
            tts_audio, sr = a._speech._tts.synthesize(direct_answer)
            seg_dur = len(tts_audio) / sr
            mute_until = time.time() + seg_dur + (a.config.mute_during_speech_ms / 1000.0)
            a._muted_until = mute_until
            a._speech._audio_output.play(tts_audio, sr)
            if a._wake_word:
                a._wake_word.reset_full()
            print(f"[VERIFY: PROC_COMPLETE] (instant RAG: score={rag_score:.3f})")
            return False, mute_until

        # ── LLM reformatter path ───────────────────────────────────────────────
        print("🤖 Thinking...", end="", flush=True)
        t1 = time.time()

        # Always fetch RAG context — LLM reformats it, never invents
        context = _rag.get_context(text, top_k=3)

        if context and rag_score < 0.75:
            # Reformatter prompt: one simple instruction the 1B model can follow
            dynamic_prompt = (
                f"{a.config.system_prompt}\n"
                "You are a voice assistant. Below are facts from a university knowledge base.\n"
                "Rephrase ONLY these facts into 2 to 3 smooth spoken sentences.\n"
                "Do NOT add any information not in the facts.\n"
                "No bullet points, no markdown, no lists.\n\n"
                f"FACTS:\n{context}"
            )
            print(f"[RAG] 🔄 Reformat — score={rag_score:.3f} ({len(context)} chars)")
        else:
            # No relevant context — tell LLM to say "I don't know"
            dynamic_prompt = (
                f"{a.config.system_prompt}\n"
                "You have no relevant information in your knowledge base for this query. "
                "Say exactly: 'I'm sorry, I don't have information about that in my knowledge base.' "
                "Do not add anything else."
            )
            print(f"[RAG] ⚠️  No match — score={rag_score:.3f} (LLM will decline)")

        history_for_llm = a._speech._conversation_history[:-1]

        sentence_buf = ""
        full_response: List[str] = []
        first_sentence_played = False
        mute_until = 0.0
        audio_duration_total = 0.0
        sentences_played = 0          # Hard cap: stop after 3 sentences
        MAX_SENTENCES = 3

        try:
            for token in a._components.llm.generate_stream(
                text,
                history=history_for_llm,
                system_prompt=dynamic_prompt,
            ):
                if sentences_played >= MAX_SENTENCES:
                    break              # Hard stop — prevents 43-second runaway

                sentence_buf += token
                full_response.append(token)

                if re.search(r"[.!?]\s", sentence_buf):
                    sentence, sentence_buf = split_first_sentence(sentence_buf)
                    if sentence.strip():
                        with watchdog_lock:
                            if not first_sentence_played:
                                first_sentence_played = True
                                first_sentence_played_event.set()
                                a._speech._audio_output.stop()
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
                            a._muted_until = time.time() + seg_dur + (a.config.mute_during_speech_ms / 1000.0)
                            a._speech._audio_output.play(tts_audio, sr)
                            sentences_played += 1

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
