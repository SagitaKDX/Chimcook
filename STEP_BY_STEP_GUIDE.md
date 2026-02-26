# Step-by-Step Implementation Guide

This guide walks you through implementing each component of the voice assistant.
**Code each step yourself** - this ensures you understand every part.

---

## Table of Contents

1. [Step 1: Audio Input Module](#step-1-audio-input-module)
2. [Step 2: Noise Reduction](#step-2-noise-reduction)
3. [Step 3: Speaker Isolation](#step-3-speaker-isolation)
4. [Step 4: Voice Activity Detection](#step-4-voice-activity-detection)
5. [Step 5: Speech-to-Text](#step-5-speech-to-text)
6. [Step 6: Language Model](#step-6-language-model)
7. [Step 7: Text-to-Speech](#step-7-text-to-speech)
8. [Step 8: Audio Output](#step-8-audio-output)
9. [Step 9: Pipeline Orchestrator](#step-9-pipeline-orchestrator)

---

## Step 1: Audio Input Module

**File:** `core/audio_input.py`

### Purpose
Capture audio from microphone with consistent frame sizes.

### Key Concepts
- Use `sounddevice` (more reliable) or `pyaudio` as fallback
- Sample rate: 16kHz (optimal for speech recognition)
- Frame size: 20ms (320 samples) - required by WebRTC VAD
- Output format: float32 normalized [-1, 1]

### What to Implement

```python
# Pseudo-code structure - implement yourself

class AudioInput:
    """
    Captures audio from microphone in fixed-size frames.
    
    Features:
    - Auto-selects best input device
    - Handles buffer underruns gracefully
    - Thread-safe frame queue
    """
    
    def __init__(self, sample_rate=16000, frame_ms=20, device=None):
        # TODO: Initialize audio stream
        # TODO: Create thread-safe queue for frames
        pass
    
    def start(self):
        # TODO: Start audio capture in background thread
        pass
    
    def stop(self):
        # TODO: Stop capture and cleanup
        pass
    
    def get_frame(self, timeout=0.1):
        # TODO: Return next frame or None if timeout
        pass
    
    def frames(self):
        # TODO: Generator that yields frames continuously
        pass
```

### Key Implementation Details

1. **Frame alignment**: Mic may deliver varying chunk sizes. Use a ring buffer to output exactly `frame_size` samples each time.

2. **Thread safety**: Audio callback runs in separate thread. Use `queue.Queue` for safe frame passing.

3. **Device selection**: 
   ```python
   # List devices and pick one with max input channels
   devices = sd.query_devices()
   # Filter for input devices with sample_rate support
   ```

4. **Error recovery**: If device disconnects, attempt reconnection after delay.

### Test It

```python
# tests/test_audio_input.py
# Record 5 seconds, save to WAV, verify audio quality
```

---

## Step 2: Noise Reduction

**File:** `core/noise_reduction.py`

### Purpose
Filter out background noise BEFORE VAD to improve detection in crowded places.

### Key Concepts
- **Noise Gate**: Silence audio below threshold (removes constant hum)
- **Spectral Subtraction**: Estimate noise profile, subtract from signal
- **RMS-based gating**: More reliable than simple amplitude threshold

### What to Implement

```python
class NoiseReducer:
    """
    Reduces background noise for cleaner VAD input.
    
    Methods:
    1. Noise gate (fast, good for constant noise)
    2. Adaptive threshold (learns noise floor)
    """
    
    def __init__(self, 
                 noise_gate_threshold=0.01,  # RMS below this = silence
                 adaptive=True,               # Learn noise floor
                 adaptation_rate=0.05):       # How fast to adapt
        # TODO: Initialize
        pass
    
    def calibrate(self, silence_frames, duration_sec=2):
        # TODO: Analyze silence to set noise floor
        # Called once at startup
        pass
    
    def process(self, frame):
        # TODO: Apply noise reduction
        # Returns: cleaned frame (same shape)
        pass
```

### Key Implementation Details

1. **Noise Gate Algorithm**:
   ```python
   rms = sqrt(mean(frame ** 2))
   if rms < threshold:
       return zeros_like(frame)  # Complete silence
   return frame
   ```

2. **Adaptive Threshold**:
   ```python
   # During silence (detected by VAD feedback):
   noise_floor = noise_floor * (1 - rate) + current_rms * rate
   threshold = noise_floor * 1.5  # 50% above noise floor
   ```

3. **Smooth transitions** to avoid clicks:
   ```python
   # Apply short fade when gating
   if gating:
       frame *= linspace(1, 0, len(frame))  # Fade out
   ```

### Calibration Procedure

```
1. Print "Please be silent for 3 seconds..."
2. Record 3 seconds of ambient noise
3. Calculate RMS statistics (mean, max, std)
4. Set threshold = mean_rms * 2 (or + 2*std)
5. Save to config for next run
```

---

## Step 3: Speaker Isolation (Crowded Place Solution)

**File:** `core/speaker_isolation.py`

### Purpose
In crowded places, multiple people talk. We need to focus on ONE speaker (the user).

### Approach: **Proximity-Based Isolation**

The key insight: **The nearest speaker is loudest and clearest.**

### What to Implement

```python
class SpeakerIsolator:
    """
    Isolates the primary speaker based on:
    1. Volume (nearest = loudest)
    2. Consistency (same speaker across frames)
    3. Direction (if using stereo/array mic)
    """
    
    def __init__(self,
                 volume_threshold=0.1,      # Min RMS to consider
                 consistency_window=10,      # Frames to track
                 min_speech_ratio=0.6):      # 60% frames must be speech
        # TODO: Initialize
        pass
    
    def update(self, frame, is_speech):
        # TODO: Track speech patterns
        # Returns: is_primary_speaker (bool)
        pass
    
    def reset(self):
        # TODO: Reset tracking (after utterance ends)
        pass
```

### Key Implementation Details

1. **Volume-Based Primary Speaker Detection**:
   ```python
   # Track RMS over sliding window
   recent_rms = deque(maxlen=window_size)
   
   def update(frame, is_speech):
       rms = compute_rms(frame)
       
       if is_speech:
           recent_rms.append(rms)
           
           # Primary speaker = consistent high volume
           if len(recent_rms) >= min_frames:
               avg_rms = mean(recent_rms)
               # If current speech is significantly quieter, it's background
               if rms < avg_rms * 0.5:
                   return False  # Background speaker
               return True
       return False
   ```

2. **Lock onto first speaker**:
   ```python
   # Once speech starts, lock onto that volume level
   # Ignore other speakers until utterance ends
   
   if not locked and is_speech:
       locked = True
       primary_rms_baseline = rms
   
   if locked and is_speech:
       # Accept if within 50% of baseline
       if rms >= primary_rms_baseline * 0.5:
           return True
       return False  # Different (quieter) speaker
   ```

3. **Release lock after silence**:
   ```python
   silence_count += 1
   if silence_count > release_frames:
       locked = False
       primary_rms_baseline = 0
   ```

### Advanced: Wake Word for Isolation

The BEST solution for crowded places:
- User says "Hey Assistant" → System activates
- Listen for response within timeout
- Ignore all other speakers

```python
# Integrate with wake word detector (Step in utils/wake_word.py)
```

---

## Step 4: Voice Activity Detection

**File:** `core/vad.py`

### Purpose
Detect when someone is speaking vs silence.

### Improvements Over v1

1. **Energy pre-filter**: Skip WebRTC VAD if frame is clearly silence
2. **Smoothing**: Prevent flutter (rapid on/off)
3. **Hangover**: Keep detecting speech slightly after it ends

### What to Implement

```python
class VAD:
    """
    Voice Activity Detection with noise robustness.
    
    Pipeline:
    1. Energy check (noise gate)
    2. WebRTC VAD (actual speech detection)
    3. Smoothing (anti-flutter)
    """
    
    def __init__(self,
                 sample_rate=16000,
                 frame_ms=20,
                 aggressiveness=2,       # 0-3, higher = more aggressive
                 energy_threshold=0.01,  # RMS threshold
                 smooth_window=5):       # Frames to smooth
        # TODO: Initialize WebRTC VAD
        pass
    
    def is_speech(self, frame):
        # TODO: Full detection pipeline
        # Returns: bool
        pass
```

### Key Implementation Details

1. **Aggressiveness setting**:
   - 0: Least aggressive (more false positives, catches quiet speech)
   - 3: Most aggressive (fewer false positives, may miss quiet speech)
   - **For crowded places: Use 2-3**

2. **Smoothing algorithm**:
   ```python
   history = deque(maxlen=window)
   
   def smooth(is_speech):
       history.append(is_speech)
       # Vote: majority wins
       return sum(history) > len(history) / 2
   ```

3. **Convert float32 to int16 for WebRTC**:
   ```python
   def float_to_pcm(frame):
       clipped = clip(frame, -1, 1)
       return (clipped * 32767).astype(int16).tobytes()
   ```

---

## Step 5: Speech-to-Text

**File:** `core/stt.py`

### Purpose
Convert audio to text using Whisper (faster-whisper for CPU efficiency).

### RAM Optimization

| Model | RAM Usage | Speed | Quality |
|-------|-----------|-------|---------|
| tiny | ~400MB | Fast | Good |
| tiny.en | ~400MB | Faster | Better (English) |
| base | ~500MB | Medium | Better |
| small | ~1GB | Slow | Best |

**For 6GB RAM: Use `tiny` or `tiny.en` with int8 quantization**

### What to Implement

```python
class STT:
    """
    Speech-to-Text using faster-whisper.
    
    Features:
    - Streaming partial results
    - Final transcription
    - Language detection (optional)
    """
    
    def __init__(self,
                 model_size="tiny",      # tiny, base, small
                 device="cpu",           # cpu or cuda
                 compute_type="int8",    # int8, float16, float32
                 language="en"):         # None for auto-detect
        # TODO: Load model
        pass
    
    def transcribe(self, audio, partial=False):
        # TODO: Transcribe audio array
        # Returns: text string
        pass
```

### Key Implementation Details

1. **Model loading (do once)**:
   ```python
   from faster_whisper import WhisperModel
   
   model = WhisperModel(
       model_size_or_path="tiny",
       device="cpu",
       compute_type="int8",  # Important for low RAM!
       cpu_threads=4,        # Adjust based on CPU
   )
   ```

2. **Transcription**:
   ```python
   segments, info = model.transcribe(
       audio,                # float32 numpy array
       beam_size=1,          # 1 for speed, 5 for accuracy
       language="en",        # Set to avoid detection overhead
       vad_filter=False,     # We do our own VAD
       word_timestamps=False,
   )
   
   text = " ".join([seg.text for seg in segments])
   ```

3. **Streaming partial results**:
   ```python
   # Transcribe rolling window every 400ms while speaking
   # Emit partial results for responsive UI
   ```

---

## Step 6: Language Model

**File:** `core/llm.py`

### Purpose
Generate responses using local LLM.

### RAM Optimization

| Model | Params | Quantization | RAM |
|-------|--------|--------------|-----|
| Qwen2.5-0.5B | 0.5B | Q4_K_M | ~0.5GB |
| Llama-3.2-1B | 1B | Q4_K_M | ~0.8GB |
| Llama-3.2-3B | 3B | Q4_K_M | ~2GB |
| Phi-3-mini | 3.8B | Q4_K_M | ~2.5GB |

**For 6GB RAM total: Use 1B-3B model with Q4_K_M**

### What to Implement

```python
class LLM:
    """
    Local LLM inference using llama-cpp-python.
    
    Features:
    - Conversation memory
    - Token streaming
    - System prompt support
    """
    
    def __init__(self,
                 model_path,
                 n_ctx=2048,           # Context window
                 n_threads=4,          # CPU threads
                 n_gpu_layers=0):      # 0 for CPU-only
        # TODO: Load model
        pass
    
    def generate(self, user_message, history=None, system_prompt=None):
        # TODO: Generate response
        # Returns: string
        pass
```

### Key Implementation Details

1. **Model loading**:
   ```python
   from llama_cpp import Llama
   
   llm = Llama(
       model_path="models/llm/llama-3.2-1b-q4_k_m.gguf",
       n_ctx=2048,
       n_threads=4,
       n_gpu_layers=0,  # CPU only
       verbose=False,
   )
   ```

2. **Chat completion**:
   ```python
   messages = [
       {"role": "system", "content": "You are a helpful voice assistant. Keep responses brief."},
       *history,
       {"role": "user", "content": user_message}
   ]
   
   response = llm.create_chat_completion(
       messages=messages,
       max_tokens=150,
       temperature=0.7,
   )
   
   return response["choices"][0]["message"]["content"]
   ```

3. **Memory management**:
   ```python
   # Keep only last N turns to fit context window
   history = history[-6:]  # 3 user + 3 assistant
   ```

---

## Step 7: Text-to-Speech

**File:** `core/tts.py`

### Purpose
Convert text to speech using Piper TTS (offline, fast, small).

### Voice Models

Download from: https://github.com/rhasspy/piper/releases

| Voice | Size | Quality |
|-------|------|---------|
| en_US-amy-low | 15MB | Good |
| en_US-amy-medium | 60MB | Better |
| en_US-lessac-medium | 60MB | Natural |

### What to Implement

```python
class TTS:
    """
    Text-to-Speech using Piper.
    
    Features:
    - Fast synthesis
    - Adjustable speed
    - Multiple voices
    """
    
    def __init__(self,
                 model_path,
                 length_scale=1.0,   # Speed: <1 faster, >1 slower
                 volume=1.0):
        # TODO: Load Piper model
        pass
    
    def synthesize(self, text):
        # TODO: Convert text to audio
        # Returns: (audio_array, sample_rate)
        pass
```

### Key Implementation Details

1. **Model loading**:
   ```python
   from piper import PiperVoice
   
   voice = PiperVoice.load(model_path)
   sample_rate = voice.config.sample_rate  # Usually 22050
   ```

2. **Synthesis**:
   ```python
   # Piper returns audio chunks
   audio_chunks = []
   for chunk in voice.synthesize(text):
       # Extract audio data (implementation varies by version)
       audio_chunks.append(chunk_audio)
   
   audio = concatenate(audio_chunks)
   return audio, sample_rate
   ```

3. **Text normalization** (important!):
   ```python
   # Expand abbreviations, numbers, etc.
   text = text.replace("Dr.", "Doctor")
   text = text.replace("&", "and")
   # Handle numbers: "123" → "one hundred twenty three"
   ```

---

## Step 8: Audio Output

**File:** `core/audio_output.py`

### Purpose
Play synthesized audio through speakers.

### What to Implement

```python
class AudioOutput:
    """
    Plays audio through system speakers.
    
    Features:
    - Non-blocking playback
    - Interrupt current playback
    - Volume control
    """
    
    def __init__(self, device=None, volume=1.0):
        # TODO: Initialize
        pass
    
    def play(self, audio, sample_rate, blocking=True):
        # TODO: Play audio
        pass
    
    def stop(self):
        # TODO: Stop current playback
        pass
    
    def is_playing(self):
        # TODO: Check if audio is playing
        pass
```

### Key Implementation Details

1. **Using sounddevice**:
   ```python
   import sounddevice as sd
   
   def play(audio, sample_rate, blocking=True):
       # Apply volume
       audio = audio * volume
       
       # Ensure proper format
       if audio.ndim == 1:
           audio = audio.reshape(-1, 1)
       
       sd.play(audio, samplerate=sample_rate)
       if blocking:
           sd.wait()
   ```

2. **Non-blocking with interrupt**:
   ```python
   def play_async(audio, sample_rate):
       sd.play(audio, samplerate=sample_rate)
       # Don't wait
   
   def stop():
       sd.stop()
   ```

---

## Step 9: Pipeline Orchestrator

**File:** `pipeline/orchestrator.py`

### Purpose
Coordinate all components into a working voice assistant.

### State Machine

```
     ┌──────────┐
     │  IDLE    │◄─────────────────────────────────┐
     └────┬─────┘                                  │
          │ Speech detected                        │
          ▼                                        │
     ┌──────────┐                                  │
     │ LISTENING│                                  │
     └────┬─────┘                                  │
          │ Speech ended                           │
          ▼                                        │
     ┌──────────┐                                  │
     │PROCESSING│ (STT + LLM)                      │
     └────┬─────┘                                  │
          │ Response ready                         │
          ▼                                        │
     ┌──────────┐                                  │
     │ SPEAKING │ (TTS playing)                    │
     └────┬─────┘                                  │
          │ Playback finished                      │
          └────────────────────────────────────────┘
```

### What to Implement

```python
class VoiceAssistant:
    """
    Main orchestrator for voice assistant.
    
    Coordinates:
    - Audio input → Noise reduction → Speaker isolation → VAD
    - STT → LLM → TTS
    - Audio output
    """
    
    def __init__(self, config):
        # TODO: Initialize all components
        pass
    
    def run(self):
        # TODO: Main loop
        pass
    
    def _on_speech_start(self):
        # TODO: Handle speech start
        pass
    
    def _on_speech_end(self, audio):
        # TODO: Process complete utterance
        pass
    
    def stop(self):
        # TODO: Cleanup
        pass
```

### Key Implementation Details

1. **Main loop**:
   ```python
   def run(self):
       self.audio_input.start()
       
       audio_buffer = []
       in_speech = False
       
       for frame in self.audio_input.frames():
           # 1. Noise reduction
           frame = self.noise_reducer.process(frame)
           
           # 2. VAD
           is_speech = self.vad.is_speech(frame)
           
           # 3. Speaker isolation (crowded place handling)
           is_primary = self.speaker_isolator.update(frame, is_speech)
           
           if is_primary:
               if not in_speech:
                   in_speech = True
                   audio_buffer = []
                   self._on_speech_start()
               audio_buffer.append(frame)
           
           elif in_speech:
               # Speech ended
               in_speech = False
               audio = concatenate(audio_buffer)
               self._on_speech_end(audio)
               self.speaker_isolator.reset()
   ```

2. **Process utterance**:
   ```python
   def _on_speech_end(self, audio):
       # STT
       text = self.stt.transcribe(audio)
       if not text.strip():
           return
       
       print(f"You: {text}")
       
       # LLM
       response = self.llm.generate(text)
       print(f"Assistant: {response}")
       
       # TTS
       audio, sr = self.tts.synthesize(response)
       self.audio_output.play(audio, sr)
   ```

---

## Bonus: Wake Word Detection

**File:** `utils/wake_word.py`

For crowded places, wake word is the BEST solution:
- User says "Hey Assistant" → Activates
- Listen for question
- Respond
- Go back to sleep

### Options

1. **Porcupine** (Picovoice) - Free for personal use, very accurate
2. **OpenWakeWord** - Fully open source
3. **Simple keyword spotting** - Use STT to detect keywords

### Simple Implementation

```python
class WakeWordDetector:
    """
    Detects wake word using STT-based keyword spotting.
    (Simple but uses more CPU than dedicated wake word engine)
    """
    
    def __init__(self, wake_words=["hey assistant", "okay assistant"]):
        self.wake_words = [w.lower() for w in wake_words]
        self.stt = STT(model_size="tiny")
    
    def check(self, audio):
        text = self.stt.transcribe(audio).lower()
        return any(w in text for w in self.wake_words)
```

---

## Testing Each Component

### Test Checklist

- [ ] Audio Input: Record 5 seconds, verify clear audio
- [ ] Noise Reduction: Compare before/after in noisy room
- [ ] Speaker Isolation: Test with music/TV in background
- [ ] VAD: Verify start/end detection accuracy
- [ ] STT: Transcribe test phrases, check accuracy
- [ ] LLM: Test conversation, verify responses
- [ ] TTS: Synthesize test text, verify audio quality
- [ ] Full Pipeline: Complete conversation test

### Performance Targets

| Component | Target Latency | Target RAM |
|-----------|---------------|------------|
| Audio frame | <50ms | <10MB |
| VAD | <5ms | <50MB |
| STT | <2s | <500MB |
| LLM | <5s | <2GB |
| TTS | <1s | <200MB |

---

## Troubleshooting

### Common Issues

1. **Audio crackling**: Increase buffer size or reduce sample rate
2. **VAD too sensitive**: Increase aggressiveness (2→3)
3. **VAD misses speech**: Decrease aggressiveness (2→1), lower energy threshold
4. **Background voices detected**: Enable speaker isolation, use wake word
5. **Out of memory**: Use smaller models, reduce context window

### Debug Logging

```python
# Enable debug mode in config
DEBUG_AUDIO = True      # Save audio to files
DEBUG_VAD = True        # Print VAD decisions
DEBUG_TIMING = True     # Print component latencies
```

---

## Next Steps

1. **Implement components in order** (Step 1 → Step 9)
2. **Test each component individually** before integration
3. **Profile memory usage** with `memory_profiler`
4. **Optimize bottlenecks** (usually STT and LLM)
5. **Add wake word** for best crowded-place experience

Good luck! 🚀
