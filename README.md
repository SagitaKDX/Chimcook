# Voice Assistant v2 - Lightweight Offline Voice Assistant

## Overview
A redesigned voice assistant optimized for:
- ✅ **Single speaker isolation** in crowded environments
- ✅ **100% offline operation** - no internet required
- ✅ **Low RAM usage** (~4-5GB) for mini PCs with 6GB RAM
- ✅ **Reliable audio pipeline** with noise reduction

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VOICE ASSISTANT v2                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────────┐    ┌─────────────────────────┐  │
│  │   Audio     │───▶│  Speaker Isolation  │───▶│   Speech Recognition    │  │
│  │   Input     │    │  (Noise Reduction)  │    │        (STT)            │  │
│  │             │    │                     │    │                         │  │
│  │ • Mic Input │    │ • Beamforming       │    │ • faster-whisper tiny   │  │
│  │ • 16kHz     │    │ • Noise Gate        │    │ • int8 quantized        │  │
│  │ • Mono      │    │ • VAD Filtering     │    │ • CPU optimized         │  │
│  └─────────────┘    └─────────────────────┘    └───────────┬─────────────┘  │
│                                                            │                │
│                                                            ▼                │
│  ┌─────────────┐    ┌─────────────────────┐    ┌─────────────────────────┐  │
│  │   Audio     │◀───│     Text-to-Speech  │◀───│   Language Model        │  │
│  │   Output    │    │        (TTS)        │    │       (LLM)             │  │
│  │             │    │                     │    │                         │  │
│  │ • Speaker   │    │ • Piper TTS         │    │ • llama.cpp             │  │
│  │ • 22kHz     │    │ • ONNX optimized    │    │ • Q4_K_M quantized      │  │
│  │             │    │ • ~50MB model       │    │ • 1-3B params           │  │
│  └─────────────┘    └─────────────────────┘    └─────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
voice_assistant_v2/
├── README.md                 # This file
├── STEP_BY_STEP_GUIDE.md    # Detailed implementation guide
├── requirements.txt          # Minimal dependencies
├── .env.example             # Configuration template
│
├── config/
│   └── settings.py          # Centralized configuration
│
├── core/
│   ├── __init__.py
│   ├── audio_input.py       # Step 1: Microphone capture
│   ├── noise_reduction.py   # Step 2: Noise gate + filtering
│   ├── speaker_isolation.py # Step 3: Target speaker detection
│   ├── vad.py               # Step 4: Voice Activity Detection
│   ├── stt.py               # Step 5: Speech-to-Text
│   ├── llm.py               # Step 6: Language Model
│   ├── tts.py               # Step 7: Text-to-Speech
│   └── audio_output.py      # Step 8: Speaker output
│
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py      # Main pipeline coordinator
│
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py       # Audio conversion helpers
│   ├── ring_buffer.py       # Circular buffer for streaming
│   └── wake_word.py         # Optional: Wake word detection
│
├── models/                   # Model files (gitignored)
│   ├── stt/                 # Whisper models
│   ├── tts/                 # Piper voice models
│   └── llm/                 # GGUF LLM models
│
└── tests/
    ├── test_audio_input.py
    ├── test_noise_reduction.py
    ├── test_speaker_isolation.py
    ├── test_vad.py
    ├── test_stt.py
    ├── test_tts.py
    └── test_full_pipeline.py
```

---

## Key Improvements Over v1

| Feature | v1 (Current) | v2 (New) |
|---------|--------------|----------|
| Noise handling | Basic VAD only | Noise gate + spectral filtering |
| Crowded places | Picks up all voices | Speaker isolation via proximity/volume |
| RAM usage | ~5-6GB | ~4GB optimized |
| Wake word | None | Optional "Hey Assistant" |
| Reliability | Basic error handling | Robust with fallbacks |
| Offline | Yes | Yes (fully offline) |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB free | 6GB total |
| CPU | 4 cores | 6+ cores |
| Storage | 2GB | 5GB |
| Microphone | Any USB | Directional mic |

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (run once)
python scripts/download_models.py

# 4. Configure
cp .env.example .env
# Edit .env with your settings

# 5. Run
python -m pipeline.orchestrator
```

---

## Next Steps

👉 **Read [STEP_BY_STEP_GUIDE.md](STEP_BY_STEP_GUIDE.md) for detailed implementation instructions**
