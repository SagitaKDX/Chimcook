"""
Voice Assistant v2 - Main Orchestrator (Refactored)
===================================================

Three-thread architecture for maximum responsiveness:

  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │  Vision Worker   │     │  Audio Thread    │     │  Inference Worker    │
  │  (Phase 1 & 2)   │────▶│  (Phase 3 & 4)  │────▶│  (Phase 5)            │
  └──────────────────┘     └──────────────────┘     └──────────────────────┘
         face_event_q              speech_q

State-Machine 5-Phase Verification:
  Phase 1: FACE_STABLE  - face stable ≥ 3 consecutive frames
  Phase 2: GREET_SENT   - greeting when IDLE + cooldown expired
  Phase 3: WW_TRIGGERED - wake word gated by _is_user_present
  Phase 4: MIC_ACTIVE   - earcon ping at VAD speech start
  Phase 5: PROC_COMPLETE- LLM streaming + TTS first-byte-out complete
"""

from __future__ import annotations

import queue
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_pipeline_dir = Path(__file__).parent
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from pipeline.config import (
    AssistantState,
    STATE_DISPLAY,
    VoiceAssistantConfig,
    get_wake_phrase,
)
from pipeline.constants import (
    CHUNK_MS,
    FRAME_MS,
    MIN_SPEECH_FRAMES,
    SAMPLE_RATE,
    SESSION_DURATION_SEC,
    SILENCE_FRAMES,
    SILERO_THRESHOLD,
    STOP_SENTINEL,
)
from pipeline.components import ComponentManager
from pipeline.debug_log import agent_log
from pipeline.speech_processor import SpeechProcessor
from pipeline.tts_stream import load_earcon_from_assets
from pipeline.wake_word_handler import WakeWordHandler

# Modular workers
from pipeline.vision_worker import VisionWorker
from pipeline.inference_worker import InferenceWorker


class VoiceAssistant:
    """
    Refactored main orchestrator.
    Manages the audio loop (Phase 3 & 4) and delegates to VisionWorker
    and InferenceWorker via thread-safe queues.
    """

    def __init__(self, config: VoiceAssistantConfig) -> None:
        self.config = config
        self._state = AssistantState.IDLE
        self._running = False
        self._muted_until = 0.0

        # ── Inter-thread queues ──────────────────────────────────────────────
        self._face_event_q: queue.Queue = queue.Queue(maxsize=4)
        self._speech_q: queue.Queue = queue.Queue(maxsize=0)  # Infinite queue

        # ── Component init ───────────────────────────────────────────────────
        self._components = ComponentManager(config)
        self._components.initialize_all()

        if config.enable_wake_word and self._components.wake_word_model:
            self._wake_word = WakeWordHandler(
                config,
                self._components.wake_word_model,
                self._components.wake_word_name,
            )
        else:
            self._wake_word = None

        self._speech = SpeechProcessor(
            config,
            self._components.stt,
            self._components.llm,
            self._components.tts,
            self._components.audio_output,
        )

        # ── Presence / face state (shared with VisionWorker) ─────────────────
        self._is_user_present = False
        self._last_face_seen = 0.0
        self._last_greeting_time = 0.0
        self._wake_word_soft_locked = False
        self._session_until = 0.0

        # ── Workers ──────────────────────────────────────────────────────────
        self._vision_worker = VisionWorker(self)
        self._inference_worker = InferenceWorker(self)

        # ── Silero VAD (loaded once, shared with audio thread) ───────────────
        print("[+] Loading Silero VAD...")
        self._silero_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        print("    ✓ Silero VAD loaded")

        self._max_frames = int(config.max_speech_duration_sec * 1000 / FRAME_MS)
        
        # Load Phase-4 earcon pre-emptively
        self._earcon_audio, self._earcon_sr = load_earcon_from_assets()

        # Print startup info
        if config.enable_wake_word:
            phrase = get_wake_phrase(self._components.wake_word_name)
            print(f'\nSay: "{phrase}" to activate')
        print("Press Ctrl+C to stop.\n")

    # =========================================================================
    # Public API
    # =========================================================================

    def run(self) -> None:
        """Main loop: audio capture → wake word → Silero VAD → speech_q."""
        self._running = True

        import scipy.signal
        import sounddevice as sd

        PROCESS_SR = SAMPLE_RATE

        # --- Determine input device sample rate ---
        input_device = self.config.audio_device
        input_sr = PROCESS_SR
        input_channels = 1
        try:
            if input_device is not None:
                dev = sd.query_devices(input_device)
                if isinstance(dev, dict):
                    dev_sr = int(dev.get("default_samplerate", 0) or 0)
                    dev_ch = int(dev.get("max_input_channels", 0) or 0)
                    if dev_sr and dev_sr != PROCESS_SR:
                        input_sr = dev_sr
                    if dev_ch == 0:
                        print(f"⚠️  Device {input_device} has 0 input channels → using default")
                        input_device = None
                    elif dev_ch > 0:
                        input_channels = dev_ch
        except Exception:
            pass

        input_chunk_samples = int(input_sr * CHUNK_MS / 1000)
        process_chunk_samples = int(PROCESS_SR * CHUNK_MS / 1000)

        def resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
            if sr_in == PROCESS_SR:
                y = x
            else:
                y = scipy.signal.resample_poly(x, PROCESS_SR, sr_in).astype(np.float32)
            if y.size != process_chunk_samples:
                if y.size > process_chunk_samples:
                    y = y[:process_chunk_samples]
                else:
                    y = np.pad(y, (0, process_chunk_samples - y.size))
            return y.astype(np.float32)

        signal.signal(signal.SIGINT, lambda s, f: self._signal_handler())

        # Start delegates
        if self.config.enable_face_detection:
            self._vision_worker.start()
        self._inference_worker.start()

        # --- Initial state ---
        if self.config.enable_wake_word:
            self._state = AssistantState.WAKE_WORD_LISTENING
        else:
            self._state = AssistantState.IDLE
            if self._wake_word:
                self._wake_word.activate()
        self._print_status()

        print(f"Silero VAD ready | Single stream {CHUNK_MS}ms chunks (in_sr={input_sr}, proc_sr={PROCESS_SR})\n")

        # --- Recordings dir ---
        recordings_dir = Path(_project_root) / "recordings"
        if self.config.save_audio:
            for sub in ("raw", "vad", "stt"):
                (recordings_dir / sub).mkdir(parents=True, exist_ok=True)
            print(f"Recordings: {recordings_dir}/")

        # --- Main audio loop state ---
        collected_frames: List[np.ndarray] = []
        is_recording = False
        silence_frame_count = 0
        raw_chunks: List[np.ndarray] = []
        silero_buffer = np.zeros(0, dtype=np.float32)

        try:
            # --- Open audio stream (try multiple rates) ---
            def open_stream(sr: int, ch: int = input_channels):
                return sd.InputStream(
                    samplerate=sr,
                    device=input_device,
                    channels=ch,
                    dtype="float32",
                    blocksize=int(sr * CHUNK_MS / 1000),
                    latency="low",
                )

            trial_rates: List[int] = []
            if input_device is not None:
                dev = sd.query_devices(input_device)
                if isinstance(dev, dict) and "default_samplerate" in dev:
                    trial_rates.append(int(dev["default_samplerate"]))
            for r in [16000, 48000, 44100]:
                if r not in trial_rates:
                    trial_rates.append(r)

            stream_cm = None
            last_err = None
            for rate in trial_rates:
                try:
                    stream_cm = open_stream(rate)
                    input_sr = rate
                    input_chunk_samples = int(input_sr * CHUNK_MS / 1000)
                    break
                except Exception as e:
                    last_err = e

            if stream_cm is None:
                raise RuntimeError(
                    f"Could not open audio stream with any of {trial_rates}. Last error: {last_err}"
                )

            with stream_cm as stream:
                print(f"Audio stream started ({input_channels}ch @ {input_sr}Hz)\n")

                while self._running:
                    # 1. Read chunk from mic
                    try:
                        audio_data, overflowed = stream.read(input_chunk_samples)
                    except Exception as e:
                        print(f"Audio read error: {e}")
                        continue
                    if overflowed:
                        continue

                    # 2. Convert to mono float32 + resample to 16kHz
                    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                        chunk_in = audio_data[:, 0].astype(np.float32)
                    else:
                        chunk_in = audio_data.flatten().astype(np.float32)
                    chunk = resample_to_16k(chunk_in, input_sr)

                    # 3. Mic gain
                    gain = float(getattr(self.config, "mic_gain", 1.0))
                    if gain != 1.0:
                        chunk = np.clip(chunk * gain, -1.0, 1.0).astype(np.float32)

                    # 4. Skip if TTS is playing or Assistant is busy
                    now = time.time()
                    if now < self._muted_until or self._state in (AssistantState.PROCESSING, AssistantState.SPEAKING):
                        self._was_muted = True
                        continue

                    if getattr(self, "_was_muted", False):
                        self._was_muted = False
                        if self._wake_word:
                            self._wake_word._model.reset()

                    # 5. Drain face events from vision thread (non-blocking)
                    self._drain_face_events()

                    # 6. Wake word gate (Phase 3 logic)
                    gate_allows_wake = True
                    if (
                        self.config.require_face_for_wake_word
                        and self._wake_word
                        and not self._wake_word.is_active
                    ):
                        gate_allows_wake = self._is_user_present

                    if not gate_allows_wake:
                        self._was_gate_closed = True
                        silero_buffer = np.concatenate([silero_buffer, chunk])
                        continue

                    if getattr(self, "_was_gate_closed", False):
                        self._was_gate_closed = False
                        if self._wake_word:
                            self._wake_word._model.reset()

                    # 7. Wake word processing
                    if not self._handle_wake_word(chunk, face_detected=True):
                        collected_frames = []
                        raw_chunks = []
                        is_recording = False
                        silence_frame_count = 0
                        silero_buffer = np.zeros(0, dtype=np.float32)
                        continue

                    # 8. Wake word timeout
                    if self._check_wake_word_timeout():
                        if is_recording and collected_frames:
                            print("\n⏰ Processing speech before timeout...")
                            self._enqueue_speech(collected_frames, raw_chunks, recordings_dir)
                        collected_frames = []
                        raw_chunks = []
                        is_recording = False
                        silence_frame_count = 0
                        silero_buffer = np.zeros(0, dtype=np.float32)
                        continue

                    # 9. Accumulate raw mic
                    if is_recording and self.config.save_audio:
                        raw_chunks.append(chunk.copy())

                    # 10. Silero VAD (512-sample frames)
                    silero_buffer = np.concatenate([silero_buffer, chunk])

                    while len(silero_buffer) >= 512:
                        frame_512 = silero_buffer[:512].copy()
                        silero_buffer = silero_buffer[512:]

                        speech_prob = self._silero_vad_predict(frame_512)
                        is_speech = speech_prob >= SILERO_THRESHOLD

                        # Debug VAD logging
                        if self.config.debug:
                            rms = float(np.sqrt(np.mean(frame_512.astype(np.float64) ** 2)))
                            bar_len = min(30, int(rms * 200))
                            bar = "█" * bar_len + "░" * (30 - bar_len)
                            label = "SPEECH" if is_speech else "silence"
                            print(
                                f"\r  {label:8}  prob={int(speech_prob*100):3}%  [{bar}]",
                                end="",
                                flush=True,
                            )
                            agent_log(
                                "VAD_LEVEL",
                                "pipeline/orchestrator_v2.py:VAD_LOOP",
                                "vad_level",
                                {
                                    "rms": rms,
                                    "speech_prob": float(speech_prob),
                                    "is_speech": bool(is_speech),
                                },
                            )

                        # ── Speech collection ──────────────────────────────
                        if is_speech:
                            if self._wake_word:
                                self._wake_word.extend_timeout()

                            if not is_recording:
                                # Phase 4: MIC_ACTIVE
                                is_recording = True
                                silence_frame_count = 0
                                collected_frames = [frame_512]
                                if self.config.save_audio:
                                    raw_chunks.append(chunk.copy())
                                
                                print(f"\r[VERIFY: MIC_ACTIVE] 🎤 Listening...", end="", flush=True)
                            else:
                                collected_frames.append(frame_512)
                            silence_frame_count = 0

                            if len(collected_frames) >= self._max_frames:
                                print("\n⚠️ Max duration reached")
                                self._enqueue_speech(collected_frames, raw_chunks, recordings_dir)
                                collected_frames = []
                                raw_chunks = []
                                is_recording = False
                                silence_frame_count = 0
                                self._silero_model.reset_states()

                        elif is_recording:
                            collected_frames.append(frame_512)
                            silence_frame_count += 1

                            if silence_frame_count >= SILENCE_FRAMES:
                                is_recording = False
                                speech_frames = len(collected_frames) - silence_frame_count

                                if speech_frames >= MIN_SPEECH_FRAMES:
                                    print()
                                    try:
                                        self._enqueue_speech(
                                            collected_frames, raw_chunks, recordings_dir
                                        )
                                    except queue.Full:
                                        print("⚠️  Inference queue full – dropping utterance")

                                collected_frames = []
                                raw_chunks = []
                                silence_frame_count = 0
                                self._silero_model.reset_states()

        finally:
            self.stop()
            self._vision_worker.stop()
            self._inference_worker.stop()
            if self._wake_word:
                self._wake_word.stop()
            self._components.stop()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _drain_face_events(self) -> None:
        """Non-blocking drain of face event queue (called from audio thread)."""
        try:
            while True:
                self._face_event_q.get_nowait()
        except queue.Empty:
            pass

    def _enqueue_speech(
        self,
        frames: List[np.ndarray],
        raw_chunks: List[np.ndarray],
        recordings_dir: Optional[Path],
    ) -> None:
        """Put collected speech onto inference queue."""
        record_ts = int(time.time() * 1000)
        item = (list(frames), list(raw_chunks), recordings_dir, record_ts)
        self._speech_q.put_nowait(item)
        
        # Play thinking phrase so user knows we heard them
        chime_until = self._speech.play_thinking_chime()
        if chime_until > self._muted_until:
            self._muted_until = chime_until

    def _silero_vad_predict(self, frame: np.ndarray) -> float:
        """Return speech probability [0, 1] for a 512-sample frame."""
        if len(frame) != 512:
            frame = np.pad(frame, (0, max(0, 512 - len(frame)))) if len(frame) < 512 else frame[:512]
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        tensor = torch.from_numpy(frame).float()
        with torch.no_grad():
            return float(self._silero_model(tensor.unsqueeze(0), SAMPLE_RATE).item())

    def _handle_wake_word(self, frame: np.ndarray, face_detected: bool) -> bool:
        """
        Phase 3: WW_TRIGGERED log emitted on detection.
        """
        if not self._wake_word:
            return True
        if self._wake_word.is_active:
            return True

        score = self._wake_word.process_frame(frame, face_detected)
        if score is not None:
            print(f"\n[VERIFY: WW_TRIGGERED] ✨ Wake word detected! (confidence: {score:.2f})")
            self._wake_word.activate()
            self._session_until = time.time() + SESSION_DURATION_SEC
            self._state = AssistantState.SPEAKING
            self._muted_until = self._speech.play_acknowledgment()
            self._state = AssistantState.IDLE
            self._print_status("listening for your question...")
            return True
        return False

    def _check_wake_word_timeout(self) -> bool:
        if not self._wake_word or not self._wake_word.is_active:
            return False
        if self._wake_word.check_timeout():
            print("\n⏰ Wake word timeout. Going back to sleep...")
            self._wake_word.deactivate(with_cooldown=True)
            self._wake_word._state.cooldown = max(
                self._wake_word._state.cooldown,
                time.time() + 3.0,
            )
            self._wake_word_soft_locked = False
            self._state = AssistantState.WAKE_WORD_LISTENING
            self._print_status()
            self._silero_model.reset_states()
            return True
        return False

    def _save_stage_recordings(
        self,
        record_ts: int,
        raw_chunks: List[np.ndarray],
        collected_frames: List[np.ndarray],
        recordings_dir: Path,
    ) -> None:
        import wave as _wave
        sr = SAMPLE_RATE

        def _write(path: Path, audio: np.ndarray) -> None:
            audio = np.asarray(audio, dtype=np.float32).flatten()
            if not audio.size:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with _wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

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

    # =========================================================================
    # Utility / status
    # =========================================================================

    def _print_status(self, extra: str = "") -> None:
        status = f"Status: {STATE_DISPLAY[self._state]}"
        if extra:
            status += f" ({extra})"
        print(status)

    def _signal_handler(self) -> None:
        print("\n\nReceived interrupt signal...")
        self._running = False

    def stop(self) -> None:
        self._running = False
        self._speech_q.put(STOP_SENTINEL)

    @property
    def state(self) -> AssistantState:
        return self._state

    @property
    def history(self) -> list:
        return self._speech.history

    def clear_history(self) -> None:
        self._speech.clear_history()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def _find_preferred_input_device() -> "int | None":
    import os
    import sounddevice as sd

    # Check both common env var names for consistency
    env = os.environ.get("AUDIO_DEVICE") or os.environ.get("AUDIO_INPUT_DEVICE")
    
    preferred_names = ["M70", "MB50"]
    
    if env:
        try:
            idx = int(env)
            print(f"🔧 Using audio device override: index {idx}")
            return idx
        except ValueError:
            # If not an integer, treat it as a preferred name substring
            print(f"🔧 Using audio device search term: '{env}'")
            preferred_names.insert(0, env)

    devices = sd.query_devices()

    for pref in preferred_names:
        for idx, dev in enumerate(devices):
            name = dev.get("name", "") if isinstance(dev, dict) else ""
            max_in = int(dev.get("max_input_channels", 0) or 0) if isinstance(dev, dict) else 0
            if pref.lower() in name.lower() and max_in > 0:
                print(f"🎤 Auto-selected input device {idx}: {name} ({max_in}ch)")
                return idx

    print("🎤 No preferred device found. Available input devices:")
    found_any = False
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0) or 0) if isinstance(dev, dict) else 0
        if max_in > 0:
            print(f"   [{idx}] {dev.get('name', 'Unknown')}")
            found_any = True
    
    if not found_any:
        print("   (No input devices detected by sounddevice)")
        
    print("🎤 Using system default input")
    return None


def main() -> None:
    import os

    print()
    print("🎙️  Voice Assistant v2  (Modular Refactor)")
    print("=" * 50)
    print()

    audio_device = _find_preferred_input_device()
    debug = os.environ.get("DEBUG", "").strip().lower() in ("1", "true", "yes")
    save_audio = os.environ.get("SAVE_AUDIO", "").strip().lower() in ("1", "true", "yes")

    config = VoiceAssistantConfig(
        audio_device=audio_device,
        enable_wake_word=True,
        enable_noise_reduction=True,
        enable_speaker_isolation=False,
        enable_face_detection=True,
        require_face_for_wake_word=True,
        greet_on_face=True,
        track_talking=True,
        wake_word_threshold=0.25,
        wake_word_timeout_sec=30.0,
        wake_word_cooldown_sec=1.0,
        silence_timeout_ms=1000,
        debug=debug,
        save_audio=save_audio,
    )

    try:
        assistant = VoiceAssistant(config)
        assistant.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have downloaded the required models:")
        sys.exit(1)
    except KeyboardInterrupt:
        pass

    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
