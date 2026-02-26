"""
Wake Word Handler
=================

Handles wake word detection and activation logic.

Uses frames passed from the main audio loop (80ms chunks).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import time


@dataclass
class WakeWordState:
    """Current wake word detection state."""
    is_active: bool = False
    timeout: float = 0.0
    cooldown: float = 0.0


class WakeWordHandler:
    """
    Handles wake word detection and activation.
    
    Receives 80ms (1280 sample) chunks from main loop.
    """
    
    def __init__(self, config, wake_word_model, wake_word_name: str):
        self.config = config
        self._model = wake_word_model
        self._name = wake_word_name
        self._state = WakeWordState()
        
        print("      Wake word handler initialized (shared audio stream)")
    
    def stop(self) -> None:
        """Stop the handler (cleanup)."""
        pass
    
    @property
    def is_active(self) -> bool:
        """Check if wake word is currently active."""
        return self._state.is_active
    
    @property
    def is_in_cooldown(self) -> bool:
        """Check if in cooldown period."""
        return time.time() < self._state.cooldown
    
    def activate(self) -> None:
        """Activate wake word mode."""
        self._state.is_active = True
        self._state.timeout = time.time() + self.config.wake_word_timeout_sec
    
    def deactivate(self, with_cooldown: bool = True) -> None:
        """Deactivate wake word mode."""
        self._state.is_active = False
        if with_cooldown:
            self._state.cooldown = time.time() + self.config.wake_word_cooldown_sec
        self._model.reset()
    
    def extend_timeout(self) -> None:
        """Extend the wake word timeout."""
        self._state.timeout = time.time() + self.config.wake_word_timeout_sec
    
    def check_timeout(self) -> bool:
        """Check if wake word has timed out."""
        if self._state.is_active and time.time() > self._state.timeout:
            return True
        return False
    
    def process_frame(self, chunk: np.ndarray, face_detected: bool = True) -> Optional[float]:
        """
        Process an audio chunk for wake word detection.
        
        Args:
            chunk: float32 audio, 80ms (1280 samples) from main loop
            face_detected: Whether a face is currently detected
        
        Returns:
            Detection score if wake word detected, None otherwise
        """
        if self._state.is_active:
            return None  # Already active
        
        if self.is_in_cooldown:
            return None  # In cooldown
        
        # Convert to int16 for wake word model
        chunk_int16 = (chunk * 32767).astype(np.int16)
        
        # Run prediction
        prediction = self._model.predict(chunk_int16)
        score = prediction.get(self._name, 0.0)
        
        # Only print when score is notable
        if score > 0.1:
            print(f"\r[Wake] {self._name}: {score:.2f} | Face: {face_detected}   ", end="", flush=True)
        
        # Check threshold
        if score > self.config.wake_word_threshold:
            if self.config.require_face_for_wake_word and not face_detected:
                print(f"\n⚠️  Wake word heard ({self._name}: {score:.2f}) but no face detected - ignoring")
                return None
            
            return score  # Wake word detected!
        
        return None
    
    def reset(self) -> None:
        """Reset all state."""
        self._state = WakeWordState()
        if self._model:
            self._model.reset()
