# Voice Assistant v2 - Pipeline Package

# Use the new modular orchestrator
from .orchestrator_v2 import VoiceAssistant, main
from .config import VoiceAssistantConfig, AssistantState

# Legacy orchestrator (kept for reference)
# from .orchestrator import VoiceAssistant

__all__ = [
    "VoiceAssistant",
    "VoiceAssistantConfig", 
    "AssistantState",
    "main",
]
