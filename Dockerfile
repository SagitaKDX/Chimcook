# ─────────────────────────────────────────────────────────────────────────────
# Chimcook Voice Assistant - Dockerfile
# Python 3.11 | CPU-only | Fully offline after build
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm AS builder

# System dependencies for building dlib, llama-cpp-python, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    python3-dev \
    portaudio19-dev \
    libspeexdsp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download OpenWakeWord models INTO site-packages (so COPY carries them)
RUN python -c "\
import openwakeword; \
openwakeword.utils.download_models(); \
print('OpenWakeWord models downloaded')"

# Verify alexa model exists
RUN ls -la /usr/local/lib/python3.11/site-packages/openwakeword/resources/models/alexa*

# Pre-cache Silero VAD into torch hub cache
RUN python -c "\
import torch; \
torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True); \
print('Silero VAD cached')"

# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libportaudio2 \
    libopenblas0 \
    liblapack3 \
    libspeexdsp1 \
    libsndfile1 \
    libasound2-plugins \
    libgl1 \
    libglib2.0-0 \
    wget \
    pulseaudio-utils \
    alsa-utils \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Configure ALSA to pipe through PulseAudio
RUN echo 'pcm.!default { \n\
    type pulse \n\
}\n\
ctl.!default { \n\
    type pulse \n\
}' > /etc/asound.conf

# Copy installed Python packages + openwakeword models from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy Silero VAD cache from builder (needed for offline operation)
COPY --from=builder /root/.cache/torch /root/.cache/torch

WORKDIR /app

# First copy scripts needed for downloading
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p models/llm models/tts models/stt models/wake_word \
    known_faces recordings logs

# Download models (LLM, TTS, STT) early to cache this heavy layer
RUN chmod +x scripts/download_models.sh && \
    bash scripts/download_models.sh models

# Now copy application code (changes frequently)
COPY core/ ./core/
COPY pipeline/ ./pipeline/
COPY utils/ ./utils/
COPY config/ ./config/
COPY assets/ ./assets/
COPY templates/ ./templates/
COPY static/ ./static/
COPY web_ui.py ./
COPY .env.example ./.env.example

# Copy .env if it exists, otherwise use example
COPY .env* ./

# Verify openwakeword models survived the COPY
RUN python -c "\
import os; \
p = '/usr/local/lib/python3.11/site-packages/openwakeword/resources/models/alexa_v0.1.onnx'; \
assert os.path.exists(p), f'MISSING: {p}'; \
print(f'✅ Verified: {p}')"

# Ensure .env exists
RUN if [ ! -f .env ]; then cp .env.example .env; fi

# Update .env with correct model paths
RUN sed -i 's|^LLM_MODEL_PATH=.*|LLM_MODEL_PATH=models/llm/qwen2.5-3b-instruct-q4_k_m.gguf|' .env && \
    sed -i 's|^TTS_MODEL_PATH=.*|TTS_MODEL_PATH=models/tts/en_US-hfc_female-medium.onnx|' .env

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-m", "pipeline.orchestrator_v2"]
