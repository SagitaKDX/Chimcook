#!/bin/bash
# Download all required models for the Voice Assistant
# Usage: ./scripts/download_models.sh [models_dir]

set -e

MODELS_DIR="${1:-models}"
echo "📦 Downloading models to ${MODELS_DIR}/"

# ─────────────────────────────────────────────────
# 1. LLM - Qwen 2.5 3B Instruct (Q4_K_M, ~2GB)
# ─────────────────────────────────────────────────
mkdir -p "${MODELS_DIR}/llm"
LLM_FILE="${MODELS_DIR}/llm/qwen2.5-3b-instruct-q4_k_m.gguf"
if [ ! -f "$LLM_FILE" ]; then
    echo "⬇️  Downloading LLM model..."
    wget -q --show-progress -O "$LLM_FILE" \
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
else
    echo "✅ LLM model already exists"
fi

# ─────────────────────────────────────────────────
# 2. TTS - Piper en_US-hfc_female-medium
# ─────────────────────────────────────────────────
mkdir -p "${MODELS_DIR}/tts"
TTS_FILE="${MODELS_DIR}/tts/en_US-hfc_female-medium.onnx"
if [ ! -f "$TTS_FILE" ]; then
    echo "⬇️  Downloading TTS model..."
    wget -q --show-progress -O "$TTS_FILE" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx"
    wget -q --show-progress -O "${TTS_FILE}.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json"
else
    echo "✅ TTS model already exists"
fi

# ─────────────────────────────────────────────────
# 3. STT - faster-whisper base.en (~150MB)
# ─────────────────────────────────────────────────
mkdir -p "${MODELS_DIR}/stt"
STT_MARKER="${MODELS_DIR}/stt/model.bin"
if [ ! -f "$STT_MARKER" ]; then
    echo "⬇️  Downloading STT model (base.en)..."
    python3 -c "
from faster_whisper import WhisperModel
WhisperModel('base.en', device='cpu', compute_type='int8', download_root='${MODELS_DIR}/stt')
print('STT model downloaded successfully')
"
else
    echo "✅ STT model already exists"
fi

# ─────────────────────────────────────────────────
# 4. Wake Word - OpenWakeWord built-in models
# ─────────────────────────────────────────────────
echo "⬇️  Downloading wake word models..."
python3 -c "
import openwakeword
openwakeword.utils.download_models()
print('Wake word models downloaded successfully')
"

# ─────────────────────────────────────────────────
# 5. RAG Embeddings (Native ONNX)
# ─────────────────────────────────────────────────
mkdir -p "${MODELS_DIR}/embed_onnx"
ONNX_MODEL_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
ONNX_TOKENIZER_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
ONNX_CONFIG_URL="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json"

if [ ! -f "${MODELS_DIR}/embed_onnx/model_quantized.onnx" ]; then
    echo "⬇️  Downloading Native ONNX embedding model..."
    wget -q --show-progress -O "${MODELS_DIR}/embed_onnx/model_quantized.onnx" "$ONNX_MODEL_URL"
    wget -q --show-progress -O "${MODELS_DIR}/embed_onnx/tokenizer.json" "$ONNX_TOKENIZER_URL"
    wget -q --show-progress -O "${MODELS_DIR}/embed_onnx/tokenizer_config.json" "$ONNX_CONFIG_URL"
else
    echo "✅ Native ONNX embedding model already exists"
fi

# ─────────────────────────────────────────────────
echo ""
echo "✅ All models downloaded!"
echo "   LLM:       ${MODELS_DIR}/llm/qwen2.5-3b-instruct-q4_k_m.gguf"
echo "   TTS:       ${MODELS_DIR}/tts/en_US-hfc_female-medium.onnx"
echo "   STT:       ${MODELS_DIR}/stt/ (base.en)"
echo "   Wake Word: openwakeword built-in models"
echo "   RAG Embed: ${MODELS_DIR}/embed_onnx/ (Native ONNX)"
