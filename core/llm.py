"""
Voice Assistant v2 - Language Model Module



















































































































































































































































































































































































```python -m core.llm# Test LLMpython -m core.stt# Test STThuggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models/llmpython -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8')"# Download recommended modelspip install faster-whisper llama-cpp-python huggingface-hub scipy numpy# Install all dependenciessource venv/bin/activate# Activate environmentcd /home/sagitakdx/Desktop/Code/LLM/voice_assistant_v2# Navigate to project```bash## Quick Reference Commands---3. Verify model file isn't corrupted (re-download)2. Check `chat_format` matches model (llama-3, chatml, etc.)1. Make sure you're using an **Instruct** model (not base)### LLM Generates Garbage3. Subsequent loads are faster2. First load is slower (memory mapping)1. Store models on SSD, not HDD### Model Loads Slowly4. Close other applications3. Use `tiny` STT model instead of `small`2. Reduce `n_ctx` (512 instead of 2048)1. Use smaller model (1B instead of 3B)### Out of Memory```    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"wget -P models/llm/ \# Or use wget    Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models/llmhuggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \huggingface-cli login# Try with explicit token (create account at huggingface.co)```bash### Model Download Fails## Troubleshooting---```)    max_tokens=150,    n_ctx=2048,    model_path="models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",LLMConfig(# core/llm.py config)    enhance_audio=True,    best_of=5,    beam_size=5,    compute_type="int8",    model_size="small.en",STTConfig(# core/stt.py config```python### Configuration 3: Quality Priority (8GB+ RAM)```)    max_tokens=100,    n_ctx=1024,    model_path="models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf",LLMConfig(# core/llm.py config)    correct_text=True,    enhance_audio=True,    beam_size=5,    compute_type="int8",    model_size="small.en",STTConfig(# core/stt.py config```python### Configuration 2: Balanced (6GB RAM) ⭐ RECOMMENDED```)    max_tokens=50,    n_ctx=512,    model_path="models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf",LLMConfig(# core/llm.py config)    beam_size=3,    compute_type="int8",    model_size="tiny.en",STTConfig(# core/stt.py config```python### Configuration 1: Speed Priority (6GB RAM)## Recommended Configurations---- Total: ~8.5 GB- LLM: Llama-3.1-8B-Q4 (~6.0 GB)- STT: small.en (~800 MB)**Maximum Quality (8GB+ RAM)**:- Total: ~6.0 GB- LLM: Llama-3.2-3B-Q4 (~3.0 GB)- STT: small.en (~800 MB)**Quality (6GB Limit)**:- Total: ~4.5 GB- LLM: Llama-3.2-1B-Q4 (~1.5 GB)- STT: small.en (~800 MB)**Balanced (Recommended)**:- Total: ~3.5 GB- LLM: Llama-3.2-1B-Q4 (~1.5 GB)- STT: tiny.en (~200 MB)**Minimal (Very Fast)**:### Alternative Configurations| **TOTAL** | - | **~6.0 GB** || Buffer | - | ~2.2 GB || LLM | Llama-3.2-1B-Q4 | ~1.5 GB || STT | small.en | ~0.8 GB || System/OS | - | ~1.5 GB ||-----------|-------|-----------|| Component | Model | RAM Usage |### For 6GB RAM System## RAM Usage Summary---```download_model("qwen-3b", output_dir="models/llm")# Download Qwen 3B alternativedownload_model("llama-3.1-8b", output_dir="models/llm")# Download Llama 3.1 8B (best quality, needs 8GB+ RAM)download_model("llama-3.2-3b", output_dir="models/llm")# Download Llama 3.2 3B (better quality)download_model("llama-3.2-1b", output_dir="models/llm")# Download Llama 3.2 1B (fast, good for 6GB RAM)from core.llm import download_model# In project directory```python### Python Helper to Download```print(response)response = llm.generate("Hello! What can you do?")llm = LLM(config))    max_tokens=100,    n_threads=4,    n_ctx=1024,    model_path="models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf",config = LLMConfig(from core.llm import LLM, LLMConfig```python### Use Downloaded LLM Model```https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf```**Llama 3.1 8B Q4_K_M**:```https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf```**Llama 3.2 3B Q4_K_M**:```https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf```**Llama 3.2 1B Q4_K_M** (Recommended):If `huggingface-cli` doesn't work, download directly:### Direct Download Links```    Mistral-7B-Instruct-v0.3-Q4_K_M.gguf --local-dir models/llmhuggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF \# ============================================================# Size: ~4.0 GB | RAM: ~5.0 GB | Good reasoning# OPTION 5: Mistral 7B v0.3 (ALTERNATIVE 7B)# ============================================================    Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models/llmhuggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \# ============================================================# Size: ~4.5 GB | RAM: ~6.0 GB | Speed: Moderate# OPTION 4: Llama 3.1 8B (BEST QUALITY - NEEDS 8GB+ RAM)# ============================================================    qwen2.5-3b-instruct-q4_k_m.gguf --local-dir models/llmhuggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \# ============================================================# Size: ~2.0 GB | RAM: ~3.0 GB | Good for conversation# OPTION 3: Qwen2.5 3B (ALTERNATIVE)# ============================================================    Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/llmhuggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \# ============================================================# Size: ~2.0 GB | RAM: ~3.0 GB | Speed: Fast# OPTION 2: Llama 3.2 3B (BALANCED)# ============================================================    Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models/llmhuggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \# ============================================================# Size: ~0.8 GB | RAM: ~1.5 GB | Speed: Very Fast# OPTION 1: Llama 3.2 1B (RECOMMENDED FOR 6GB RAM)# ============================================================mkdir -p models/llm# Create models directorycd /home/sagitakdx/Desktop/Code/LLM/voice_assistant_v2# Navigate to projectpip install huggingface-hub```bash### Download LLM Models| Llama-3.1-8B | 4.5 GB | ~6.0 GB | ⚡⚡ | ★★★★★ | **Best quality** || Mistral-7B | 4.0 GB | ~5.0 GB | ⚡⚡⚡ | ★★★★★ | 8GB RAM || Qwen2.5-3B | 2.0 GB | ~3.0 GB | ⚡⚡⚡⚡ | ★★★★☆ | Conversation, coding || Llama-3.2-3B | 2.0 GB | ~3.0 GB | ⚡⚡⚡⚡ | ★★★★☆ | **6GB RAM, balanced** || Llama-3.2-1B | 0.8 GB | ~1.5 GB | ⚡⚡⚡⚡⚡ | ★★★☆☆ | **6GB RAM, fast** ||-------|------|-----------|-------|---------|----------|| Model | Size | RAM Usage | Speed | Quality | Best For |### Available Models (Q4_K_M Quantization)We use **llama-cpp-python** with **GGUF** format models.## LLM Models (Language Model)---```stt = STT(config))    compute_type="int8"    device="cpu",    model_size="/path/to/models/stt/small.en",  # Local pathconfig = STTConfig(# Use downloaded model from local pathfrom core.stt import STT, STTConfig```python### Use Custom STT Model Path```huggingface-cli download Systran/faster-whisper-large-v3 --local-dir models/stt/large-v3# Large model (3 GB) - NOT recommended for 6GB RAMhuggingface-cli download Systran/faster-whisper-medium.en --local-dir models/stt/medium.enhuggingface-cli download Systran/faster-whisper-medium --local-dir models/stt/medium# Medium models (1.5 GB each)huggingface-cli download Systran/faster-whisper-small.en --local-dir models/stt/small.enhuggingface-cli download Systran/faster-whisper-small --local-dir models/stt/small# Small models (466 MB each) - RECOMMENDEDhuggingface-cli download Systran/faster-whisper-base.en --local-dir models/stt/base.enhuggingface-cli download Systran/faster-whisper-base --local-dir models/stt/base# Base models (142 MB each)huggingface-cli download Systran/faster-whisper-tiny.en --local-dir models/stt/tiny.enhuggingface-cli download Systran/faster-whisper-tiny --local-dir models/stt/tiny# Tiny models (75 MB each)pip install huggingface-hub# Method 2: Hugging Face CLIpython -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8')"python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"# Method 1: Python (creates cache automatically)```bashModels auto-download on first use. To pre-download:### Download STT Models| `large-v3` | 3 GB | ~5 GB | ⚡ | ★★★★★ | Maximum accuracy || `medium.en` | 1.5 GB | ~2.5 GB | ⚡⚡ | ★★★★★ | Best English || `medium` | 1.5 GB | ~2.5 GB | ⚡⚡ | ★★★★★ | High accuracy || `small.en` | 466 MB | ~800 MB | ⚡⚡⚡ | ★★★★★ | **English + accents** || `small` | 466 MB | ~800 MB | ⚡⚡⚡ | ★★★★☆ | Multi-language || `base.en` | 142 MB | ~300 MB | ⚡⚡⚡⚡ | ★★★★☆ | English only || `base` | 142 MB | ~300 MB | ⚡⚡⚡⚡ | ★★★☆☆ | Quick transcription || `tiny.en` | 75 MB | ~200 MB | ⚡⚡⚡⚡⚡ | ★★★☆☆ | English only, fast || `tiny` | 75 MB | ~200 MB | ⚡⚡⚡⚡⚡ | ★★☆☆☆ | Testing, real-time ||-------|------|-----------|-------|----------|----------|| Model | Size | RAM Usage | Speed | Accuracy | Best For |### Available ModelsWe use **faster-whisper** for STT. Models download automatically to `~/.cache/huggingface/` on first use.## STT Models (Speech-to-Text)---4. [Recommended Configurations](#recommended-configurations)3. [RAM Usage Summary](#ram-usage-summary)2. [LLM Models (Language Model)](#llm-models-language-model)1. [STT Models (Speech-to-Text)](#stt-models-speech-to-text)## Table of Contents---```    Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models/llmhuggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \# Download LLM model (Llama 3.2 1B - ~800MB, fast)python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8')"# This downloads automatically when first used, or manually:# Download STT model (small.en - ~500MB, best for non-native English speakers)pip install huggingface-hub# Install huggingface-hub for downloadingcd /home/sagitakdx/Desktop/Code/LLM/voice_assistant_v2```bash## Quick Start (Recommended for 6GB RAM)Complete guide for downloading all models required for the Voice Assistant.==========================================

Step 6: Generate responses using local LLM (llama.cpp).

Supported Models (GGUF format):
================================

For 6GB RAM (recommended):
- Llama-3.2-1B-Instruct-Q4_K_M     (~0.8GB)  - Fast, basic quality
- Llama-3.2-3B-Instruct-Q4_K_M     (~2.0GB)  - Good balance
- Qwen2.5-3B-Instruct-Q4_K_M       (~2.0GB)  - Good for conversation

For 8GB+ RAM (better quality):
- Llama-3.1-8B-Instruct-Q4_K_M     (~4.5GB)  - Excellent quality
- Mistral-7B-Instruct-Q4_K_M       (~4.0GB)  - Very good

Model Download Instructions:
============================

Option 1: Download from Hugging Face (recommended)
--------------------------------------------------
# Install huggingface-hub if needed
pip install huggingface-hub

# Download Llama 3.2 1B (fast, 0.8GB) - FOR 6GB RAM
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --local-dir models/llm

# Download Llama 3.2 3B (better, 2GB)
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
    Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --local-dir models/llm

# Download Llama 3.1 8B (best quality, 4.5GB) - RECOMMENDED FOR 16GB RAM
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --local-dir models/llm

Option 2: Direct download links
-------------------------------
Llama 3.2 1B Q4_K_M:
https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

Llama 3.2 3B Q4_K_M:
https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

Llama 3.1 8B Q4_K_M:
https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

Place downloaded files in: models/llm/
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Generator
from pathlib import Path
import time
import os

# Import llama-cpp-python
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("[LLM] Warning: llama-cpp-python not installed")


# Default system prompt for voice assistant
DEFAULT_SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses:
- Concise (1-3 sentences for simple questions)
- Natural and conversational
- Friendly but professional

When asked about yourself, say you're a local AI assistant running on the user's computer.
If you don't know something, say so briefly."""


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""
    model_path: str = ""            # Path to GGUF model file
    n_ctx: int = 2048               # Context window size
    n_threads: int = 4              # CPU threads for inference
    n_gpu_layers: int = 0           # GPU layers (0 = CPU only)
    max_tokens: int = 150           # Max tokens in response
    temperature: float = 0.7        # Randomness (0 = deterministic)
    top_p: float = 0.9              # Nucleus sampling
    top_k: int = 40                 # Top-k sampling
    repeat_penalty: float = 1.1     # Penalize repetition
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    verbose: bool = False           # Show llama.cpp logs


@dataclass
class LLMConfigForVoice(LLMConfig):
    """
    Pre-configured LLM settings optimized for voice assistant.
    
    Shorter responses, faster inference.
    """
    max_tokens: int = 100           # Shorter responses for voice
    temperature: float = 0.7
    n_ctx: int = 1024               # Smaller context for speed


class LLM:
    """
    Local LLM inference using llama-cpp-python.
    
    Features:
    - Chat completion with history
    - System prompt support
    - Token streaming for responsive output
    - Conversation memory management
    
    Usage:
        # Initialize
        config = LLMConfig(model_path="models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        llm = LLM(config)
        
        # Generate response
        response = llm.generate("What's the weather like?")
        
        # With conversation history
        history = [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help?"}
        ]
        response = llm.generate("What can you do?", history=history)
        
        # Streaming generation
        for token in llm.generate_stream("Tell me a joke"):
            print(token, end="", flush=True)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        if not HAS_LLAMA_CPP:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Run: pip install llama-cpp-python"
            )
        
        self.config = config or LLMConfig()
        
        # Validate model path
        if not self.config.model_path:
            raise ValueError(
                "model_path is required. Download a model first:\n"
                "  huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \\\n"
                "      Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models/llm"
            )
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download a GGUF model and place it in the models/llm/ directory."
            )
        
        print(f"[LLM] Loading model: {model_path.name}")
        print(f"      Context: {self.config.n_ctx}, Threads: {self.config.n_threads}")
        
        load_start = time.time()
        
        # Auto-detect chat format based on model name
        model_name_lower = model_path.name.lower()
        if "qwen" in model_name_lower:
            chat_format = "chatml"  # Qwen uses ChatML format
        elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
            chat_format = "llama-3"
        elif "mistral" in model_name_lower:
            chat_format = "mistral-instruct"
        else:
            chat_format = "chatml"  # Default fallback
        
        print(f"      Chat format: {chat_format}")
        
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=self.config.verbose,
            chat_format=chat_format,
        )
        
        # Store detected format for stop tokens
        self._chat_format = chat_format
        
        load_time = time.time() - load_start
        print(f"[LLM] Model loaded in {load_time:.1f}s")
        
        # Statistics
        self._stats = {
            "generations": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }
    
    def _get_stop_tokens(self) -> List[str]:
        """Get appropriate stop tokens based on chat format."""
        if self._chat_format == "chatml":
            # Qwen, Yi, and other ChatML models
            return ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        elif self._chat_format == "llama-3":
            # Llama 3.x models
            return ["<|eot_id|>", "<|end_of_text|>"]
        elif self._chat_format == "mistral-instruct":
            # Mistral models
            return ["</s>", "[/INST]"]
        else:
            # Generic fallback
            return ["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>"]
    
    def _clean_response(self, text: str) -> str:
        """Remove any leaked special tokens from response."""
        import re
        # Remove common special tokens that might leak through
        patterns = [
            r'<\|im_end\|>',
            r'<\|im_start\|>',
            r'<\|endoftext\|>',
            r'<\|eot_id\|>',
            r'<\|end_of_text\|>',
            r'<\|start_header_id\|>',
            r'<\|end_header_id\|>',
            r'<\|begin_of_text\|>',
            r'</s>',
            r'\[/INST\]',
            r'<end_of_turn>',
            r'<eos>',
            # Role markers that might leak
            r'(system|user|assistant)\|end_header_id\|>',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up any leftover whitespace/newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        return text.strip()
    
    def generate(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate response to user message.
        
        Args:
            user_message: User's input text
            history: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system instructions (uses default if None)
            
        Returns:
            Assistant's response text
        """
        messages = self._build_messages(user_message, history, system_prompt)
        
        start_time = time.time()
        
        try:
            response = self._model.create_chat_completion(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=self._get_stop_tokens(),
            )
            
            content = response["choices"][0]["message"]["content"]
            
            # Clean any remaining special tokens from output
            content = self._clean_response(content)
            
            # Update stats
            self._stats["generations"] += 1
            self._stats["total_tokens"] += response.get("usage", {}).get("completion_tokens", 0)
            self._stats["total_time"] += time.time() - start_time
            
            return content.strip()
            
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return "I'm sorry, I encountered an error generating a response."
    
    def generate_stream(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate response with token streaming.
        
        Yields tokens as they're generated for responsive output.
        
        Args:
            user_message: User's input text
            history: Conversation history
            system_prompt: Optional system instructions
            
        Yields:
            Partial response strings as they're generated
        """
        messages = self._build_messages(user_message, history, system_prompt)
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            for chunk in self._model.create_chat_completion(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stop=self._get_stop_tokens(),
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                # Skip any special tokens that leak through
                if content and not content.startswith("<|") and not content.startswith("<end"):
                    total_tokens += 1
                    yield content
            
            # Update stats
            self._stats["generations"] += 1
            self._stats["total_tokens"] += total_tokens
            self._stats["total_time"] += time.time() - start_time
            
        except Exception as e:
            print(f"[LLM] Stream error: {e}")
            yield "I'm sorry, I encountered an error."
    
    def _build_messages(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """Build messages list for chat completion."""
        messages = []
        
        # Add system prompt
        prompt = system_prompt or self.config.system_prompt
        if prompt:
            messages.append({
                "role": "system",
                "content": prompt
            })
        
        # Add trimmed history
        if history:
            trimmed = self._trim_history(history)
            messages.extend(trimmed)
        
        # Add current message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _trim_history(self, history: List[Dict], max_turns: int = 5) -> List[Dict]:
        """
        Trim conversation history to stay within context limits.
        
        Args:
            history: Full conversation history
            max_turns: Maximum conversation turns to keep
            
        Returns:
            Trimmed history (most recent turns)
        """
        # Each turn = 1 user + 1 assistant message
        max_messages = max_turns * 2
        
        if len(history) <= max_messages:
            return history
        
        return history[-max_messages:]
    
    def get_stats(self) -> Dict:
        """Get generation statistics."""
        total_time = self._stats["total_time"]
        total_tokens = self._stats["total_tokens"]
        
        return {
            **self._stats,
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "generations": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }


def download_model(model_name: str = "llama-3.2-1b", output_dir: str = "models/llm") -> str:
    """
    Download a GGUF model from Hugging Face.
    
    Args:
        model_name: One of "llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"
        output_dir: Directory to save the model
        
    Returns:
        Path to downloaded model file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Install huggingface-hub: pip install huggingface-hub")
    
    models = {
        "llama-3.2-1b": {
            "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        },
        "llama-3.2-3b": {
            "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        },
        "llama-3.1-8b": {
            "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        },
        "qwen-3b": {
            "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
        },
    }
    
    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_info = models[model_name]
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    print(f"  Repo: {model_info['repo']}")
    print(f"  File: {model_info['file']}")
    
    path = hf_hub_download(
        repo_id=model_info["repo"],
        filename=model_info["file"],
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    
    print(f"  Saved to: {path}")
    return path


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test LLM module."""
    import sys
    
    print("=" * 60)
    print("LLM MODULE TEST")
    print("=" * 60)
    
    print(f"\nllama-cpp-python available: {HAS_LLAMA_CPP}")
    
    if not HAS_LLAMA_CPP:
        print("Install with: pip install llama-cpp-python")
        sys.exit(1)
    
    # Check for model
    model_dir = Path(__file__).parent.parent / "models" / "llm"
    model_files = list(model_dir.glob("*.gguf")) if model_dir.exists() else []
    
    if not model_files:
        print("\n" + "=" * 60)
        print("NO MODEL FOUND")
        print("=" * 60)
        print("\nDownload a model first:")
        print()
        print("Option 1: Use the download function:")
        print("  python -c \"from core.llm import download_model; download_model('llama-3.2-1b')\"")
        print()
        print("Option 2: Manual download:")
        print("  huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \\")
        print("      Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models/llm")
        print()
        print("Available models:")
        print("  - llama-3.2-1b  (~0.8GB) - Fast, basic quality")
        print("  - llama-3.2-3b  (~2.0GB) - Good balance")
        print("  - llama-3.1-8b  (~4.5GB) - Best quality (needs 8GB+ RAM)")
        sys.exit(0)
    
    # Use first found model
    model_path = str(model_files[0])
    print(f"\nFound model: {model_files[0].name}")
    
    # Test 1: Load model
    print("\n[Test 1] Loading LLM...")
    
    config = LLMConfig(
        model_path=model_path,
        n_ctx=1024,
        n_threads=4,
        max_tokens=100,
        temperature=0.7,
    )
    
    llm = LLM(config)
    
    # Test 2: Simple generation
    print("\n[Test 2] Simple generation...")
    
    response = llm.generate("What is 2 + 2?")
    print(f"  Q: What is 2 + 2?")
    print(f"  A: {response}")
    
    # Test 3: With conversation history
    print("\n[Test 3] With conversation history...")
    
    history = [
        {"role": "user", "content": "My name is Alex."},
        {"role": "assistant", "content": "Nice to meet you, Alex! How can I help you today?"}
    ]
    
    response = llm.generate("What's my name?", history=history)
    print(f"  Q: What's my name?")
    print(f"  A: {response}")
    
    # Test 4: Streaming
    print("\n[Test 4] Streaming generation...")
    print("  Q: Tell me a very short joke.")
    print("  A: ", end="")
    
    for token in llm.generate_stream("Tell me a very short joke."):
        print(token, end="", flush=True)
    print()
    
    # Stats
    stats = llm.get_stats()
    print(f"\n  Stats: {stats['generations']} generations, {stats['tokens_per_second']:.1f} tok/s")
    
    print("\n" + "=" * 60)
    print("LLM TEST COMPLETE")
    print("=" * 60)
