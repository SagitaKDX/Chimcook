# Voice Assistant v2 - Installation Guide

## 🎯 Overview

A fully **offline voice assistant** optimized for:
- 🎤 Noisy/crowded environments
- 🌍 Non-native English speakers
- 💾 Low RAM systems (16GB recommended, 8GB minimum)
- 🔒 Complete privacy (no internet required)

---

## 📋 System Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| CPU | 4 cores | 8 cores |
| Storage | 5GB | 10GB |
| Microphone | Any USB/built-in | Noise-canceling mic |

### Software
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 11+
- **Python**: 3.10 or higher
- **Audio**: ALSA (Linux) / PortAudio

---

## 🚀 Quick Installation

### Step 1: Clone/Navigate to Project
```bash
cd /path/to/voice_assistant_v2
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate   # Windows
```

### Step 3: Install System Dependencies (Linux)
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    portaudio19-dev \
    python3-dev \
    build-essential \
    libsndfile1

# For Speex noise suppression (optional but recommended)
sudo apt install -y libspeexdsp-dev
```

### Step 4: Install Python Dependencies
```bash
pip install --upgrade pip

# Core dependencies
pip install sounddevice numpy webrtcvad-wheels
pip install faster-whisper piper-tts
pip install llama-cpp-python python-dotenv

# Wake word detection (OpenWakeWord)
pip install openwakeword

# Noise suppression (recommended for noisy environments)
pip install speexdsp-ns
```

### Step 5: Download Models

#### 5a. LLM Model (Required)
Download a GGUF model file. Recommended: **Llama 3.1 8B Q4_K_M** (~4.7GB)

```bash
mkdir -p models/llm
cd models/llm

# Option 1: Using wget
wget https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Option 2: Using curl
curl -L -o Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

cd ../..
```

#### 5b. STT Model (Auto-downloaded)
The **distil-large-v3** model (~756MB) downloads automatically on first run.

#### 5c. TTS Model (Auto-downloaded)
The **piper-tts** voice model downloads automatically on first run.

#### 5d. Wake Word Model (Auto-downloaded)
**OpenWakeWord** models download automatically.

---

## ⚙️ Configuration

### Environment File
Copy the example and edit:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# LLM Model Path (required)
LLM_MODEL_PATH=models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Optional: Custom paths
STT_MODEL_PATH=models/stt
TTS_MODEL_PATH=models/tts
```

---

## 🎤 Wake Word Configuration

### Current Wake Word: **"Jarvis"** (Community Model)

The assistant uses **OpenWakeWord** with the community-trained **Jarvis** model.

#### Available Pre-trained Models (Built-in)

| Wake Word | Model Name | Accuracy |
|-----------|------------|----------|
| "Alexa" | `alexa` | ⭐⭐⭐⭐⭐ Best |
| "Hey Mycroft" | `hey_mycroft` | ⭐⭐⭐⭐ Good |
| "Hey Jarvis" | `hey_jarvis` | ⭐⭐⭐ OK |

#### Community Models (60+ options!)

Community models from [home-assistant-wakewords-collection](https://github.com/fwartner/home-assistant-wakewords-collection):

| Wake Word | Download URL |
|-----------|--------------|
| "Jarvis" | `en/jarvis/jarvis_v1.onnx` |
| "Hey Friday" | `en/hey_friday/` |
| "Computer" | `en/computer/` |
| "Ok Jarvis" | `en/ok_jarvis/` |

### Changing Wake Word

Edit [pipeline/orchestrator.py](pipeline/orchestrator.py), find `VoiceAssistantConfig`:

```python
# For built-in models:
wake_word_model: str = "alexa"  # alexa, hey_mycroft, hey_jarvis

# For community models (path to .onnx file):
wake_word_model: str = "models/wake_word/jarvis_v1.onnx"

wake_word_threshold: float = 0.5      # Lower = more sensitive (0.3-0.7)
wake_word_timeout_sec: float = 30.0   # How long to listen after wake word
```

### Installing Community Wake Words

```bash
# Create models directory
mkdir -p models/wake_word
cd models/wake_word

# Download Jarvis model
curl -L -o jarvis_v1.onnx "https://raw.githubusercontent.com/fwartner/home-assistant-wakewords-collection/main/en/jarvis/jarvis_v1.onnx"

# Download Hey Friday model  
curl -L -o hey_friday.onnx "https://raw.githubusercontent.com/fwartner/home-assistant-wakewords-collection/main/en/hey_friday/hey_friday_v1.onnx"
```

---

## 🗣️ Speech-to-Text (STT) Configuration

### Current Model: **distil-large-v3** (Optimized for non-native speakers)

| Model | Size | RAM | Best For |
|-------|------|-----|----------|
| `tiny.en` | 39MB | ~400MB | Fast, native English only |
| `small.en` | 244MB | ~1GB | Balance (English only) |
| `distil-large-v3` | 756MB | ~2GB | **Non-native speakers** ✓ |
| `large-v3` | 1.5GB | ~4GB | Best accuracy |

### Settings for Non-Native Speakers

In [core/stt.py](core/stt.py), the `STTConfigForAccents` uses:
```python
model_size: str = "distil-large-v3"  # Multilingual, handles accents
beam_size: int = 3                    # More search options
temperature: float = 0.2             # Helps with pronunciation variations
best_of: int = 3                     # Picks best from multiple attempts
```

---

## 🔊 Text-to-Speech (TTS) Configuration

### Current Voice: **libritts-high** (Speaker 100 - Cute voice)

Edit [pipeline/orchestrator.py](pipeline/orchestrator.py):
```python
tts_speaker_id: int = 100  # Change speaker ID for different voices
```

Available speakers vary by model. Test different IDs (0-200+) to find your preferred voice.

---

## 🏃 Running the Assistant

### Basic Usage
```bash
cd /path/to/voice_assistant_v2
source venv/bin/activate
python -m pipeline.orchestrator
```

### What Happens on First Run
1. ⏳ Downloads STT model (distil-large-v3, ~756MB) - **one time only**
2. ⏳ Downloads TTS model (~60MB) - **one time only**
3. ⏳ Downloads wake word model (~5MB) - **one time only**
4. 🎤 Calibrates noise floor (2 seconds)
5. 👂 Starts listening for "Alexa"

### Usage Flow
```
1. Say "Alexa" → Assistant activates
2. Ask your question → Speech recognized
3. Wait for response → LLM generates answer
4. Listen to response → TTS plays audio
5. Continue conversation or wait 30s to deactivate
```

---

## 🐛 Troubleshooting

### No Audio Input
```bash
# Check available audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Test microphone
python -c "
import sounddevice as sd
import numpy as np
print('Recording 3 seconds...')
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
sd.wait()
print(f'Max volume: {np.max(np.abs(audio)):.4f}')
print('Audio captured!' if np.max(np.abs(audio)) > 0.01 else 'No audio detected')
"
```

### Wake Word Not Detecting
- Move closer to microphone
- Speak clearly: "Alexa" (ah-LEK-sah)
- Lower threshold in config: `wake_word_threshold: float = 0.15`
- Check for background noise

### STT Hallucinating (e.g., "Thanks for watching!")
This happens when there's background noise. Solutions:
1. Increase VAD energy threshold in [core/vad.py](core/vad.py)
2. Speak louder/closer to mic
3. Reduce background noise

### High CPU Usage
- Use smaller STT model: `small.en` instead of `distil-large-v3`
- Reduce LLM context: `n_ctx: 2048` instead of `4096`
- Use fewer CPU threads

### Memory Issues
```bash
# Check memory usage during run
watch -n 1 free -h
```

If RAM is insufficient:
- Use `small.en` STT model (~1GB)
- Use smaller LLM (3B instead of 8B)
- Reduce `n_ctx` in LLM config

---

## 📁 File Structure

```
voice_assistant_v2/
├── INSTALLATION.md          # This file
├── README.md                # Overview
├── requirements.txt         # Python dependencies
├── .env                     # Your configuration
│
├── core/                    # Core modules
│   ├── audio_input.py       # Microphone capture
│   ├── noise_reduction.py   # Noise filtering
│   ├── vad.py               # Voice Activity Detection
│   ├── stt.py               # Speech-to-Text (distil-large-v3)
│   ├── llm.py               # Language Model (Llama 3.1)
│   ├── tts.py               # Text-to-Speech (Piper)
│   └── audio_output.py      # Speaker output
│
├── pipeline/
│   └── orchestrator.py      # Main controller
│
├── utils/
│   ├── audio_enhance.py     # Audio enhancement
│   └── wake_word.py         # Wake word config
│
└── models/                  # Downloaded models (auto-created)
    ├── stt/                 # Whisper models
    ├── tts/                 # Piper voices
    └── llm/                 # GGUF LLM (manual download)
```

---

## 🔄 Updating

### Update Python Packages
```bash
source venv/bin/activate
pip install --upgrade faster-whisper piper-tts llama-cpp-python openwakeword
```

### Update Models
Delete old models and restart (auto-downloads new versions):
```bash
rm -rf models/stt models/tts
python -m pipeline.orchestrator
```

---

## 📞 Support

### Logs
The assistant prints debug info. Look for:
- `[STT]` - Speech recognition issues
- `[VAD]` - Voice detection issues
- `[LLM]` - Language model issues
- `[TTS]` - Text-to-speech issues

### Common Issues
| Issue | Solution |
|-------|----------|
| "ALSA underrun" | Normal on Linux, can ignore |
| Model download stuck | Check internet, retry |
| "No speech detected" | Speak louder, check mic |
| Slow responses | Use smaller models |

---

## ✅ Quick Test Commands

```bash
# Test audio input
python -c "from core.audio_input import AudioInput; a = AudioInput(); a.start(); import time; time.sleep(2); print('Audio works!')"

# Test STT
python -c "from core.stt import STT, STTConfigForAccents; s = STT(STTConfigForAccents()); print('STT loaded!')"

# Test TTS
python -c "from core.tts import TTS; t = TTS(); t.speak('Hello, I am your assistant'); print('TTS works!')"

# Full test
python -m pipeline.orchestrator
```

---

**Happy voice assisting! 🎤🤖**
