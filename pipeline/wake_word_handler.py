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
        # Only reset the prediction score history, NOT the audio feature buffers.
        # Preserving mel/embedding buffers means OWW can detect again immediately
        # after cooldown ends without needing a warm-up period.
        if self._model:
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
        
        # Convert to int16 for wake word model
        chunk_int16 = (chunk * 32767).astype(np.int16)
        
        # ALWAYS run predict() to keep the preprocessor's internal audio
        # feature buffers (melspectrogram, embeddings) moving in real-time.
        # If we skip predict() during cooldown, those buffers freeze and
        # the first call after cooldown reads stale TTS features → false trigger.
        prediction = self._model.predict(chunk_int16)
        score = prediction.get(self._name, 0.0)
        
        # During cooldown, we fed the audio but ignore the result
        if self.is_in_cooldown:
            return None
        
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
        self.reset_full()
    
    def reset_full(self) -> None:
        """Full reset: clear Model prediction buffer AND preprocessor audio buffers.
        
        Model.reset() only clears prediction_buffer (score history).
        The preprocessor still holds mel/embedding features from old audio.
        We must clear those too, otherwise the first predict() after reset
        reads stale features and triggers falsely.
        """
        if self._model:
            self._model.reset()
            # Clear preprocessor audio feature buffers
            pp = self._model.preprocessor
            if hasattr(pp, 'raw_data_buffer'):
                pp.raw_data_buffer = []
            if hasattr(pp, 'melspectrogram_buffer'):
                import numpy as _np
                pp.melspectrogram_buffer = _np.zeros(pp.melspectrogram_buffer.shape, dtype=pp.melspectrogram_buffer.dtype)
            if hasattr(pp, 'feature_buffer'):
                import numpy as _np
                pp.feature_buffer = _np.zeros(pp.feature_buffer.shape, dtype=pp.feature_buffer.dtype)
            if hasattr(pp, 'accumulated_samples'):
                pp.accumulated_samples = 0
